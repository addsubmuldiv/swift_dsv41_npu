# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK RMSNorm generator using native sampling and input coupling.

Constrain x before aligning dy and parameter shapes/dtypes. Independent phase
APIs wrap the sampled configuration as a shared-input bundle; tensor generation
and CPU statistics run later in the producer data plugin. Boundary flags remain
controlled by YAML. See README.md for limits and execution commands.
"""

from pathlib import Path
import sys
from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from rmsnorm_inputs import create_input_bundle_case  # noqa: E402

_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from _stress_constraints import constrain_input_shape  # noqa: E402

# Fail-closed layout check: the hook addresses slots by index, so a yaml
# whose inputs don't match (x, dy, gamma, eps, zero_centered_gamma) must
# raise here instead of silently patching the wrong slots.
_EXPECTED_TYPES = ("tensor", "tensor", "tensor", "attr", "attr")


@GENERATOR_REGISTRY.register("generate_rmsnorm_hook")
class GenerateRMSNormHook(CaseGenerator):
    """Native sampling + coupling patch for the TE RMSNorm case matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_rmsnorm_hook expects the input layout "
                "(x, dy, gamma, eps, zero_centered_gamma) with types "
                "tensor x3 + attr x2, got {}".format(got)
            )
        super().__init__(config)
        self._x_shape = None
        self._x_dtype = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        if index == 0:
            if not input_case.shape or not all(d > 0 for d in input_case.shape):
                raise ValueError(
                    "generate_rmsnorm_hook: degenerate x shape {} — scalar/"
                    "empty inputs are illegal for RMSNorm (2-4-D contract; "
                    "backward rejects an empty dy). Keep has_empty and "
                    "has_scalar false for x; NaN/Inf inputs are supported".format(input_case.shape)
                )
            # x — refresh the coupling anchors every case (generate() calls
            # the hook in input order, so slot 0 always fires first).
            constrain_input_shape(input_case, normalization=True)
            self._x_shape = list(input_case.shape)
            self._x_dtype = input_case.dtype
        elif index == 1:
            # dy := x.shape / x.dtype
            input_case.shape = list(self._x_shape)
            input_case.dtype = self._x_dtype
        elif index == 2:
            # gamma := [x.shape[-1]] / x.dtype (weight; RMSNorm has no bias)
            input_case.shape = [self._x_shape[-1]]
            input_case.dtype = self._x_dtype
        # index 3/4 (eps / zero_centered_gamma): free attrs, untouched.
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """Wrap native five-input cases for the shared-input phase test."""
        if self.config.api_type not in ("function_rmsnorm_forward", "function_rmsnorm_backward"):
            raise ValueError("Expected function_rmsnorm_forward or function_rmsnorm_backward")
        source = case_config.model_dump()
        # Native generate() increments index before this callback. ATK assigns
        # report IDs later; this local ID pairs forward/backward input seeds.
        source["id"] = self.index - 1
        if self.is_gen_extra:
            source["id"] += self.config.dtype_numbers * max(
                len(i.dtypes.values) for i in self.config.inputs
            )
        phase = self.config.api_type.rsplit("_", 1)[-1]
        return CaseConfig(**create_input_bundle_case(source, phase))
