# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Deterministic ATK P0 case generator for ``transformer_engine.pytorch.Linear``."""

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


# Each entry is intentionally small enough for presubmit while covering the
# public behavior that tends to regress during an upstream rebase: bias on/off,
# return_bias, consume_returned_bias, tuple2/dict2 parameter splits, 2D/3D/4D
# and the empty-input edge case. All bf16 (the only dtype used in model
# training).
_P0_MATRIX: List[Dict] = [
    dict(input_shape=[8, 16], out_features=32, bias=True, return_bias=False),
    dict(input_shape=[4, 2, 16], out_features=24, bias=False, return_bias=False),
    dict(input_shape=[3, 2, 17], out_features=15, bias=True, return_bias=True),
    # Catches implementations that only restore three-dimensional output.
    dict(input_shape=[2, 3, 4, 16], out_features=12, bias=True, return_bias=False),
    dict(input_shape=[0, 16], out_features=8, bias=True, return_bias=False),
    dict(
        input_shape=[8, 16],
        out_features=32,
        bias=True,
        return_bias=False,
        parameters_split="tuple2",
    ),
    dict(
        input_shape=[5, 17],
        out_features=15,
        bias=True,
        return_bias=False,
        parameters_split="dict2",
    ),
    dict(
        input_shape=[4, 2, 16],
        out_features=24,
        bias=True,
        return_bias=True,
        consume_returned_bias=True,
    ),
]

# Real-scale cases mirror LLM training shapes: attention QKV/projection and
# MLP up/gate/down projections. Dimensions are multiples of 128 so the same
# cases stay valid (and aligned) when FP8 support lands; a deliberately
# non-aligned case is kept as a tagged outlier. Run together with _P0_MATRIX
# for the full set, or as the "real" tier for pre-merge/golden runs.
_REAL_MATRIX: List[Dict] = [
    # Attention QKV/proj projection: [tokens, hidden] -> hidden.
    dict(input_shape=[2048, 1024], out_features=1024, bias=True, return_bias=False),
    # TE transformer layers feed [seq, batch, hidden] directly.
    dict(input_shape=[2048, 2, 1024], out_features=1024, bias=False, return_bias=False),
    # MLP up/gate projection, 2x inter hidden, dict2 split.
    dict(
        input_shape=[1024, 1024],
        out_features=2048,
        bias=True,
        return_bias=False,
        parameters_split="dict2",
    ),
    # MLP down projection: inter_hidden -> hidden.
    dict(input_shape=[1024, 2048], out_features=1024, bias=True, return_bias=False),
    # Long-sequence token count.
    dict(input_shape=[4096, 512], out_features=512, bias=True, return_bias=False),
    # Deliberately non-aligned (in 1028, out 1000), tagged separately.
    dict(input_shape=[1024, 1028], out_features=1000, bias=True, return_bias=False),
]


@GENERATOR_REGISTRY.register("generate_te_linear_p0")
class GenerateTELinearP0(CaseGenerator):
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
        # Two tiers: "small" for fast presubmit branch coverage, "real" for
        # pre-merge/golden runs at LLM-training scale. The tier is recorded in
        # save_name (main.py overwrites it at task time, so it only labels the
        # generated case set) and is otherwise implied by the shapes.
        for tier, matrix in (("small", _P0_MATRIX), ("real", _REAL_MATRIX)):
            for dtype in dtypes:
                for spec in matrix:
                    input_shape = list(spec["input_shape"])
                    in_features = input_shape[-1]
                    out_features = spec["out_features"]
                    bias = spec.get("bias", True)
                    return_bias = spec.get("return_bias", False)
                    consume_returned_bias = spec.get("consume_returned_bias", False)
                    split_kind = spec.get("parameters_split", "none")
                    output_shape = input_shape[:-1] + [out_features]

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
                            _tensor(dtype, [out_features, in_features]),
                            _tensor(dtype, [out_features]),
                            _tensor(dtype, output_shape),
                            _tensor(dtype, [out_features]),
                            _attr("attr_bool", bias),
                            _attr("attr_bool", return_bias),
                            _attr("attr_bool", consume_returned_bias),
                            _attr("string", split_kind),
                        ],
                    )
                    cases.append(case)
        return cases
