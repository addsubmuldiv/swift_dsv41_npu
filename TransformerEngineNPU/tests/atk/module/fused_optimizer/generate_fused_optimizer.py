# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Deterministic ATK P0 case generator for the TE fused-optimizer guard."""

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


# Small tier: step count 7 matches NVTE's own test_fused_optimizer.py default;
# cover Adam (betas/eps/wd variants) and SGD (momentum/nesterov).
_P0_SMALL: List[Dict] = [
    dict(
        shape=[8, 16],
        optim_type="adam",
        lr=1e-3,
        weight_decay=0.0,
        steps=7,
        b0=0.9,
        b1=0.999,
        eps=1e-8,
        momentum=0.0,
        nesterov=False,
    ),  # adam baseline
    dict(
        shape=[8, 16],
        optim_type="sgd",
        lr=1e-3,
        weight_decay=0.0,
        steps=7,
        b0=0.0,
        b1=0.0,
        eps=0.0,
        momentum=0.9,
        nesterov=False,
    ),  # sgd momentum
    dict(
        shape=[16, 32],
        optim_type="adam",
        lr=3e-3,
        weight_decay=1e-2,
        steps=7,
        b0=0.9,
        b1=0.99,
        eps=1e-6,
        momentum=0.0,
        nesterov=False,
    ),  # adam wd + eps variant
    dict(
        shape=[32, 16],
        optim_type="sgd",
        lr=1e-2,
        weight_decay=0.0,
        steps=7,
        b0=0.0,
        b1=0.0,
        eps=0.0,
        momentum=0.9,
        nesterov=True,
    ),  # sgd nesterov (momentum>0 required)
    dict(
        shape=[8, 64],
        optim_type="adam",
        lr=1e-3,
        weight_decay=0.0,
        steps=7,
        b0=0.95,
        b1=0.999,
        eps=1e-8,
        momentum=0.0,
        nesterov=False,
    ),  # adam betas variant
]

# Real tier: LLM-scale shapes, 3 steps (large tensors, keep the run cheap).
_P0_REAL: List[Dict] = [
    dict(
        shape=[2048, 1024],
        optim_type="adam",
        lr=1e-3,
        weight_decay=1e-2,
        steps=3,
        b0=0.9,
        b1=0.999,
        eps=1e-8,
        momentum=0.0,
        nesterov=False,
    ),
    dict(
        shape=[1024, 2048],
        optim_type="sgd",
        lr=1e-3,
        weight_decay=0.0,
        steps=3,
        b0=0.0,
        b1=0.0,
        eps=0.0,
        momentum=0.9,
        nesterov=False,
    ),
    dict(
        shape=[2048, 2, 1024],
        optim_type="adam",
        lr=1e-3,
        weight_decay=0.0,
        steps=3,
        b0=0.9,
        b1=0.999,
        eps=1e-6,
        momentum=0.0,
        nesterov=False,
    ),  # 3D
]


@GENERATOR_REGISTRY.register("generate_fused_optimizer_p0")
class GenerateFusedOptimizerP0(CaseGenerator):
    """Generate a stable, cross-device fused-optimizer case matrix."""

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
                    case = CaseConfig(
                        name=self.config.name,
                        api=self.config.api,
                        api_type=self.config.api_type,
                        version=self.config.version,
                        backward=False,
                        standard=self.config.standard,
                        save_name=tier,
                        inputs=[
                            _tensor(dtype, list(spec["shape"])),
                            _tensor(dtype, list(spec["shape"])),
                            _attr("string", spec["optim_type"]),
                            _attr("float", spec["lr"]),
                            _attr("float", spec["weight_decay"]),
                            _attr("int", spec["steps"]),
                            _attr("float", spec["b0"]),
                            _attr("float", spec["b1"]),
                            _attr("float", spec["eps"]),
                            _attr("float", spec["momentum"]),
                            _attr("attr_bool", spec["nesterov"]),
                        ],
                    )
                    cases.append(case)
        return cases
