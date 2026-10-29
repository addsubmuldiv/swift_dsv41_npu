# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Deterministic ATK P0 case generator for the Megatron-facing fusible
activation ops in ``transformer_engine.pytorch.ops`` (see the Megatron
``TE_API_Analysis`` table, "Fusible ops" category): GELU, SiLU, ReLU (unary),
GEGLU, ReGLU, SwiGLU (gated, split and halve the last dimension) and the
per-row post-scaled Megatron fused-MoE variants ScaledSwiGLU,
ScaledClampedQGeGLU (gated) and ScaledSReLU (unary). The scaled variants take
an extra per-row scale tensor of shape ``x.shape[:-1]``; every case emits a
uniform 4-input layout (x, dy, scale, op) so the case matrix has a fixed
arity, and the non-scaled cases carry a placeholder scale the adapter ignores.
"""

from typing import Dict, Iterator, List

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig


_RANGE = [-0.5, 0.5]

# Gated ops split the input along the last dim into gate/linear halves, so the
# last dim must be even and the output last dim is half the input last dim.
_GATED = {"geglu", "reglu", "swiglu", "scaled_swiglu", "scaled_clamped_qgeglu"}

# Ops that take a per-row scale extra input (shape x.shape[:-1]).
_SCALED = {"scaled_swiglu", "scaled_srelu", "scaled_clamped_qgeglu"}


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


# Small tier: minimal branch coverage — one case per op (each is a distinct
# kernel path) plus two shape specials (3D and a gated non-multiple-of-8 last
# dim). All bf16 (the only dtype used in model training).
_P0_SMALL: List[Dict] = [
    dict(op="gelu", shape=[8, 16]),
    dict(op="silu", shape=[8, 16]),
    dict(op="relu", shape=[8, 16]),
    dict(op="geglu", shape=[8, 32]),
    dict(op="reglu", shape=[8, 32]),
    dict(op="swiglu", shape=[8, 32]),
    dict(op="scaled_swiglu", shape=[8, 32]),
    dict(op="scaled_clamped_qgeglu", shape=[8, 32]),
    dict(op="scaled_srelu", shape=[8, 16]),
    dict(op="gelu", shape=[4, 2, 16]),  # 3D special
    dict(op="swiglu", shape=[8, 30]),  # gated, even but non-aligned last dim
]

# Real tier: LLM-scale token counts and hidden sizes, dimensions multiples of
# 128 so the same cases stay valid when FP8 support lands.
_P0_REAL: List[Dict] = [
    dict(op="gelu", shape=[2048, 1024]),
    dict(op="silu", shape=[2048, 1024]),
    dict(op="relu", shape=[2048, 1024]),
    dict(op="geglu", shape=[2048, 2048]),
    dict(op="reglu", shape=[2048, 2048]),
    dict(op="swiglu", shape=[2048, 2048]),
    dict(op="scaled_swiglu", shape=[2048, 2048]),
    dict(op="scaled_clamped_qgeglu", shape=[2048, 2048]),
    dict(op="scaled_srelu", shape=[2048, 1024]),
]


@GENERATOR_REGISTRY.register("generate_fused_activation_p0")
class GenerateFusedActivationP0(CaseGenerator):
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
                    shape = list(spec["shape"])
                    op = spec["op"]
                    if op in _GATED:
                        out_shape = shape[:-1] + [shape[-1] // 2]
                    else:
                        out_shape = shape
                    # Per-row scale extra input for scaled variants; a
                    # placeholder for non-scaled ops (adapter ignores it). The
                    # uniform 4-input arity keeps the ATK slot count stable.
                    scale_shape = shape[:-1]
                    case = CaseConfig(
                        name=self.config.name,
                        api=self.config.api,
                        api_type=self.config.api_type,
                        version=self.config.version,
                        backward=False,
                        standard=self.config.standard,
                        save_name=tier,
                        inputs=[
                            _tensor(dtype, shape),
                            _tensor(dtype, out_shape),
                            _tensor(dtype, scale_shape),
                            _attr("string", op),
                        ],
                    )
                    cases.append(case)
        return cases
