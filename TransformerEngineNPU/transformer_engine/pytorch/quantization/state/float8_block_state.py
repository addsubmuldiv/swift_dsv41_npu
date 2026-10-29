# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import itertools
from typing import List, Optional

import torch

from transformer_engine.common.recipe import Float8BlockScaling

from ..quantizer_role import QuantizerRole
from ..utils import get_fp8_torch_dtype
from .base import RecipeState


class Float8BlockScalingRecipeState(RecipeState):
    """Configuration for Float8BlockScaling quantization.

    Float8BlockScaling quantization does not require state,
    but different quantizers use different modes.
    """

    recipe: Float8BlockScaling
    mode: str
    qx_dtype: torch.dtype
    qw_dtype: torch.dtype
    qgrad_dtype: torch.dtype

    def __init__(
        self,
        recipe: Float8BlockScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self.recipe = recipe
        self.mode = mode
        self.roles = roles
        self.num_quantizers = num_quantizers
        self.qx_dtype = get_fp8_torch_dtype(recipe, True)
        self.qw_dtype = get_fp8_torch_dtype(recipe, True)
        self.qgrad_dtype = get_fp8_torch_dtype(recipe, False)

        # Allocate buffers
        if device is None:
            device = torch.device("npu")
        self.device = device

    def make_quantizers(self) -> list:
        from ...tensor import Float8BlockQuantizer

        if self.mode == "forward":
            # The index convention (coming from base.py set_meta_tensor)
            # is somewhat awkward, and doesn't play nicely with QuantizeOp,
            # which is not associated with a GEMM.
            assert self.num_quantizers % 3 == 0  # x, w, output per gemm
            return list(
                itertools.chain.from_iterable(
                    [
                        [
                            Float8BlockQuantizer(
                                fp8_dtype=self.qx_dtype,
                                rowwise=True,
                                columnwise=True,
                                amax_epsilon=self.recipe.fp8_quant_fwd_inp.amax_epsilon,
                                force_pow_2_scales=self.recipe.fp8_quant_fwd_inp.power_2_scale,
                                block_scaling_dim=self.recipe.x_block_scaling_dim,
                            ),
                            Float8BlockQuantizer(
                                fp8_dtype=self.qw_dtype,
                                rowwise=True,
                                columnwise=True,
                                amax_epsilon=self.recipe.fp8_quant_fwd_weight.amax_epsilon,
                                force_pow_2_scales=self.recipe.fp8_quant_fwd_weight.power_2_scale,
                                block_scaling_dim=self.recipe.w_block_scaling_dim,
                            ),
                            Float8BlockQuantizer(
                                fp8_dtype=self.qx_dtype,
                                rowwise=True,
                                columnwise=True,
                                amax_epsilon=self.recipe.fp8_quant_fwd_inp.amax_epsilon,
                                force_pow_2_scales=self.recipe.fp8_quant_fwd_inp.power_2_scale,
                                block_scaling_dim=self.recipe.x_block_scaling_dim,
                            ),
                        ]
                        for _ in range(self.num_quantizers // 3)
                    ]
                )
            )

        assert self.mode == "backward", f"Unexpected mode {self.mode}"
        assert self.num_quantizers % 2 == 0  # grad_output and grad_input per gemm
        return list(
            itertools.chain.from_iterable(
                [
                    [
                        Float8BlockQuantizer(
                            fp8_dtype=self.qgrad_dtype,
                            rowwise=True,
                            columnwise=True,
                            amax_epsilon=self.recipe.fp8_quant_bwd_grad.amax_epsilon,
                            force_pow_2_scales=self.recipe.fp8_quant_bwd_grad.power_2_scale,
                            block_scaling_dim=self.recipe.grad_block_scaling_dim,
                            columnwise_group_quant_lhs_trans=True,
                        ),
                        Float8BlockQuantizer(
                            fp8_dtype=self.qgrad_dtype,
                            rowwise=True,
                            columnwise=True,
                            amax_epsilon=self.recipe.fp8_quant_bwd_grad.amax_epsilon,
                            force_pow_2_scales=self.recipe.fp8_quant_bwd_grad.power_2_scale,
                            block_scaling_dim=self.recipe.grad_block_scaling_dim,
                        ),
                    ]
                    for _ in range(self.num_quantizers // 2)
                ]
            )
        )
