# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Deterministic ATK P0 case generator for the TE multi-tensor guard."""

from typing import Dict, Iterator, List

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

_RANGE = [-0.5, 0.5]
_CHUNK_SIZE = 128  # ignored on NPU; NVTE kernel chunking param (does not affect numerics)


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


# Small tier: cover all three ops + per_tensor variant.
_P0_SMALL: List[Dict] = [
    dict(shape=[8, 16], op="scale", scale=2.0, per_tensor=False),  # scale baseline
    dict(shape=[8, 16], op="scale_tensor", scale=0.5, per_tensor=False),  # tensor-scale variant
    dict(shape=[16, 32], op="l2norm", scale=0.0, per_tensor=False),  # l2norm
    dict(shape=[32, 16], op="l2norm", scale=0.0, per_tensor=True),  # l2norm per-tensor
    dict(shape=[8, 64], op="scale", scale=1e-4, per_tensor=False),  # tiny scale
]

# Real tier: LLM-scale shapes.
_P0_REAL: List[Dict] = [
    dict(shape=[2048, 1024], op="scale", scale=2.0, per_tensor=False),
    dict(shape=[1024, 2048], op="l2norm", scale=0.0, per_tensor=False),
    dict(shape=[2048, 2, 1024], op="scale_tensor", scale=0.1, per_tensor=False),  # 3D
]


@GENERATOR_REGISTRY.register("generate_multi_tensor_p0")
class GenerateMultiTensorP0(CaseGenerator):
    """Generate a stable, cross-device multi-tensor case matrix (3 tensors per case)."""

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
                            _tensor(dtype, shape),
                            _tensor(dtype, shape),
                            _attr("string", spec["op"]),
                            _attr("float", spec["scale"]),
                            _attr("attr_bool", spec["per_tensor"]),
                            _attr("int", _CHUNK_SIZE),
                        ],
                    )
                    cases.append(case)
        return cases
