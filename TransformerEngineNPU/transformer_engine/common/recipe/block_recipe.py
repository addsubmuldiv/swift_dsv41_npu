# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Optional

from transformer_engine.common.recipe import Format
from transformer_engine.common.recipe.base import (
    MMParams,
    QParams,
    Recipe,
    _default_backward_override,
)


@dataclass(repr=False)
class Float8BlockScaling(Recipe):
    """Block scaling FP8 recipe.

    This recipe uses block-level scaling factors similar to MXFP8
    but with a different scaling approach.

    Parameters
    ----------
    block_size : int, default = 128
        Size of quantization blocks
    fp8_format : Format, default = Format.E4M3
        FP8 format to use
    """

    block_size: int = 128
    fp8_format: Format = Format.E4M3
    x_block_scaling_dim: int = (1, 2)
    w_block_scaling_dim: int = 2
    grad_block_scaling_dim: int = 1

    # ---------- 暂不使用 ----------
    fp8_quant_fwd_inp = QParams(amax_epsilon=0.0)
    fp8_quant_fwd_weight = QParams(amax_epsilon=0.0)
    fp8_quant_bwd_grad = QParams(amax_epsilon=0.0)

    fp8_gemm_fprop: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    @classmethod
    def float8_block_scaling(cls):
        return True

    def update(self) -> None:
        """No update needed for block scaling."""
        pass

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"fp8_quant_fwd_inp={self.fp8_quant_fwd_inp}, "
            f"fp8_quant_fwd_weight={self.fp8_quant_fwd_weight}, "
            f"fp8_quant_bwd_grad={self.fp8_quant_bwd_grad}, "
            f"x_block_scaling_dim={self.x_block_scaling_dim}, "
            f"w_block_scaling_dim={self.w_block_scaling_dim}, "
            f"grad_block_scaling_dim={self.grad_block_scaling_dim}, "
            f"fp8_gemm_fprop={self.fp8_gemm_fprop}, "
            f"fp8_gemm_dgrad={self.fp8_gemm_dgrad}, "
            f"fp8_gemm_wgrad={self.fp8_gemm_wgrad}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )
