# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import os
from dataclasses import dataclass
from typing import Optional

from transformer_engine.common.recipe import Format

from .base import MMParams, QParams, Recipe, _default_backward_override


@dataclass(repr=False)
class Float8CurrentScaling(Recipe):
    """
    Use the per-tensor current scaling factor strategy.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
                Controls the FP8 data format used during forward and backward
                pass.
    """

    use_power_2_scales: bool = os.getenv("NVTE_FP8_CURRENT_SCALING_POWER_2_SCALES", "0") == "1"
    fp8_format: Format = Format.HYBRID
    fp8_quant_fwd_inp = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_quant_fwd_weight = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_quant_bwd_grad = QParams(power_2_scale=use_power_2_scales, amax_epsilon=0.0)
    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=False)
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    @classmethod
    def float8_current_scaling(cls):
        return True

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"fp8_quant_fwd_inp={self.fp8_quant_fwd_inp}, "
            f"fp8_quant_fwd_weight={self.fp8_quant_fwd_weight}, "
            f"fp8_quant_bwd_grad={self.fp8_quant_bwd_grad}, "
            f"fp8_gemm_fprop={self.fp8_gemm_fprop}, "
            f"fp8_gemm_dgrad={self.fp8_gemm_dgrad}, "
            f"fp8_gemm_wgrad={self.fp8_gemm_wgrad}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )
