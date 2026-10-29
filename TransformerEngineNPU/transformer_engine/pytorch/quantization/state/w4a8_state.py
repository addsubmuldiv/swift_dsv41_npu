# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Optional

import torch

from transformer_engine.common.recipe import W4A8BlockScaling

from ..quantizer_role import QuantizerRole
from ..utils import get_fp8_torch_dtype
from .base import RecipeState


class W4A8BlockScalingRecipeState(RecipeState):
    """Configuration for W4A8 mixed block scaling.

    Forward quantizer layout follows the existing per-GEMM convention:
    input, weight, output. The weight quantizer builds hybrid storage
    with MXFP4 data for fprop and MXFP8 data for dgrad.
    """

    recipe: W4A8BlockScaling
    mode: str
    qx_dtype: torch.dtype
    qw_bwd_dtype: torch.dtype
    qgrad_dtype: torch.dtype

    def __init__(
        self,
        recipe: W4A8BlockScaling,
        *,
        mode: str = "forward",
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.roles = roles
        self.num_quantizers = num_quantizers
        self.qx_dtype = get_fp8_torch_dtype(recipe, True)
        self.qw_bwd_dtype = get_fp8_torch_dtype(recipe, False)
        self.qgrad_dtype = get_fp8_torch_dtype(recipe, False)
        self.device = torch.device("npu") if device is None else device

    def make_quantizers(self) -> list:
        from ...tensor.dummy_tensor import DummyQuantizer
        from ...tensor.mxfp8_tensor import MXFP8Quantizer
        from ...tensor.w4a8_tensor import W4A8WeightQuantizer

        if self.mode == "forward":
            assert self.num_quantizers % 3 == 0  # input, weight, output per GEMM
            quantizers = []
            for index, enabled in enumerate(self._quantization_scope_mask(self.num_quantizers)):
                if not enabled:
                    quantizer = DummyQuantizer()
                elif index % 3 == 1:
                    quantizer = W4A8WeightQuantizer(
                        mxfp8_dtype=self.qw_bwd_dtype,
                        block_size=self.recipe.block_size,
                    )
                else:
                    quantizer = MXFP8Quantizer(self.qx_dtype)
                quantizers.append(quantizer)
            return quantizers

        assert self.mode == "backward", f"Unexpected mode {self.mode}"
        return [
            MXFP8Quantizer(self.qgrad_dtype) if enabled else DummyQuantizer()
            for enabled in self._quantization_scope_mask(self.num_quantizers)
        ]
