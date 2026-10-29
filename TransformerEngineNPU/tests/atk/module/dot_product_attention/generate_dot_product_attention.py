# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Deterministic ATK P0 case generator for
``transformer_engine.pytorch.DotProductAttention``.
"""

from typing import Dict, Iterator, List

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig


_RANGE = [-0.5, 0.5]


def _tensor(dtype: str, shape: List[int]) -> InputCaseConfig:
    return InputCaseConfig(
        name=None,
        type="tensor",
        dtype=dtype,
        shape=shape,
        range_values=_RANGE,
        backward=False,
    )


def _attr(dtype: str, value) -> InputCaseConfig:
    return InputCaseConfig(
        name=None,
        type="attr",
        dtype=dtype,
        shape=None,
        range_values=value,
        backward=False,
    )


# Small tier: minimal branch coverage — MHA/GQA x causal/no_mask plus a
# kv_channels != heads shape special, all in bf16 (the only dtype used in
# model training). Shapes stay inside the GPU fused-attention (cuDNN) support
# matrix: batch >= 2, seq a multiple of 16, head_dim a multiple of 8. MQA
# (gqa=1), odd head counts, non-aligned head dims and batch == 1 are excluded
# here — they run on TENPU/NPU but cuDNN on a flash-attn-less GPU host rejects
# them (batch == 1 fails its backward with "s_q = s_kv = 1 is not supported").
_P0_SMALL: List[Dict] = [
    dict(b=2, s=16, heads=4, kv=16, gqa=None, mask="causal"),  # MHA baseline
    dict(b=2, s=16, heads=4, kv=16, gqa=None, mask="no_mask"),  # mask branch
    dict(b=2, s=16, heads=4, kv=16, gqa=2, mask="causal"),  # GQA branch
    dict(b=2, s=16, heads=4, kv=32, gqa=None, mask="causal"),  # kv != heads
]

# Real tier: LLM-scale attention (hidden = heads * kv) plus shape specials —
# long sequence and deep heads, with no_mask guarded at scale. Dimensions are
# multiples of 128 so the same cases stay valid when FP8 support lands. batch
# stays >= 2 (cuDNN fused-attention backward rejects 1).
_P0_REAL: List[Dict] = [
    dict(b=2, s=2048, heads=32, kv=128, gqa=None, mask="causal"),  # LLM baseline
    dict(b=2, s=2048, heads=32, kv=128, gqa=8, mask="causal"),  # GQA at scale
    dict(b=2, s=4096, heads=16, kv=128, gqa=None, mask="causal"),  # long seq
    dict(b=2, s=2048, heads=64, kv=64, gqa=None, mask="no_mask"),  # deep heads + no_mask
]


@GENERATOR_REGISTRY.register("generate_dot_product_attention_p0")
class GenerateDotProductAttentionP0(CaseGenerator):
    """Generate a stable, cross-device case matrix instead of random shapes."""

    def __init__(self, config):
        self.config = config
        self._cases = self._build_cases()
        self._index = 0

    def __iter__(self) -> Iterator[CaseConfig]:
        return self

    def __len__(self) -> int:
        return len(self._cases)

    def __next__(self) -> CaseConfig:
        if self._index >= len(self._cases):
            raise StopIteration
        case = self._cases[self._index]
        self._index += 1
        return case

    def _build_cases(self) -> List[CaseConfig]:
        dtypes = list(self.config.inputs[0].dtypes.values)
        cases: List[CaseConfig] = []
        for tier, matrix in (("small", _P0_SMALL), ("real", _P0_REAL)):
            for dtype in dtypes:
                for spec in matrix:
                    b, s = spec["b"], spec["s"]
                    heads, kv = spec["heads"], spec["kv"]
                    gqa = spec["gqa"]
                    kv_heads = gqa if gqa else heads
                    mask = spec["mask"]
                    q_shape = [b, s, heads, kv]
                    kv_shape = [b, s, kv_heads, kv]
                    dy_shape = [b, s, heads * kv]
                    case = CaseConfig(
                        name=self.config.name,
                        api=self.config.api,
                        api_type=self.config.api_type,
                        version=self.config.version,
                        backward=False,
                        standard=self.config.standard,
                        save_name=tier,
                        inputs=[
                            _tensor(dtype, q_shape),
                            _tensor(dtype, kv_shape),
                            _tensor(dtype, kv_shape),
                            _tensor(dtype, dy_shape),
                            _attr("int", heads),
                            _attr("int", kv),
                            _attr("int", gqa if gqa is not None else 0),
                            _attr("string", mask),
                            # Slot 8: qkv_format. This curated matrix is
                            # the 4-D path, i.e. "sbhd" (the module default
                            # the adapter now passes explicitly). thd
                            # (3-D packed) cases are the hook route's job —
                            # see generate_dpa_hook.py.
                            _attr("string", "sbhd"),
                        ],
                    )
                    cases.append(case)
        return cases
