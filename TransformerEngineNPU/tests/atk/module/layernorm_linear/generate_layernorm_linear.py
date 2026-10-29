# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Deterministic ATK P0 case generator for
``transformer_engine.pytorch.LayerNormLinear``.
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


# Small tier: minimal branch coverage — normalization (LayerNorm/RMSNorm),
# bias on/off, zero_centered on/off, eps special, 2D/3D/4D and non-aligned
# dimensions. All bf16 (the only dtype used in model training).
_P0_SMALL: List[Dict] = [
    dict(
        shape=[8, 16], out_features=32, normalization="LayerNorm", bias=True, zero_centered=False
    ),  # baseline
    dict(
        shape=[4, 2, 16], out_features=24, normalization="RMSNorm", bias=True, zero_centered=False
    ),  # RMSNorm
    dict(
        shape=[3, 2, 17], out_features=15, normalization="LayerNorm", bias=True, zero_centered=True
    ),  # zc + non-aligned
    dict(
        shape=[2, 3, 4, 16],
        out_features=12,
        normalization="RMSNorm",
        bias=False,
        zero_centered=False,
    ),  # 4D + bias off
    dict(
        shape=[5, 17], out_features=15, normalization="LayerNorm", bias=True, zero_centered=False
    ),  # non-aligned
    dict(
        shape=[8, 16],
        out_features=32,
        normalization="LayerNorm",
        bias=True,
        zero_centered=False,
        eps=1e-4,
    ),  # eps
]

# Real tier: LLM-training-scale shapes, dimensions multiples of 128 so the same
# cases stay valid when FP8 support lands.
_P0_REAL: List[Dict] = [
    dict(
        shape=[2048, 1024],
        out_features=1024,
        normalization="LayerNorm",
        bias=True,
        zero_centered=False,
    ),  # LLM baseline
    dict(
        shape=[2048, 2, 1024],
        out_features=1024,
        normalization="LayerNorm",
        bias=True,
        zero_centered=False,
    ),  # 3D layout
    dict(
        shape=[1024, 1024],
        out_features=2048,
        normalization="RMSNorm",
        bias=True,
        zero_centered=False,
    ),  # RMSNorm MLP
    dict(
        shape=[1024, 1024],
        out_features=2048,
        normalization="LayerNorm",
        bias=False,
        zero_centered=False,
    ),  # big up, bias off
    dict(
        shape=[1024, 2048],
        out_features=1024,
        normalization="RMSNorm",
        bias=True,
        zero_centered=False,
    ),  # MLP down
    dict(
        shape=[4096, 512],
        out_features=512,
        normalization="LayerNorm",
        bias=True,
        zero_centered=False,
    ),  # long seq
]


@GENERATOR_REGISTRY.register("generate_layernorm_linear_p0")
class GenerateLayerNormLinearP0(CaseGenerator):
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
                    input_shape = list(spec["shape"])
                    in_features = input_shape[-1]
                    out_features = spec["out_features"]
                    output_shape = input_shape[:-1] + [out_features]
                    normalization = spec["normalization"]
                    bias = spec["bias"]
                    zero_centered = spec["zero_centered"]
                    eps = spec.get("eps", 1e-5)
                    case = CaseConfig(
                        name=self.config.name,
                        api=self.config.api,
                        api_type=self.config.api_type,
                        version=self.config.version,
                        backward=False,
                        standard=self.config.standard,
                        save_name=tier,
                        inputs=[
                            _tensor(dtype, input_shape),
                            _tensor(dtype, output_shape),
                            _tensor(dtype, [in_features]),
                            _tensor(dtype, [in_features]),
                            _tensor(dtype, [out_features, in_features]),
                            _tensor(dtype, [out_features]),
                            _attr("float", eps),
                            _attr("string", normalization),
                            _attr("attr_bool", bias),
                            _attr("attr_bool", zero_centered),
                        ],
                    )
                    cases.append(case)
        return cases
