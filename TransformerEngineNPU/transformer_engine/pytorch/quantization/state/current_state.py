# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Optional

import torch

from transformer_engine.common.recipe import Float8CurrentScaling

from ..quantizer_role import QuantizerRole
from ..utils import get_fp8_max, get_fp8_torch_dtype
from .base import RecipeState


class Float8CurrentScalingRecipeState(RecipeState):
    """Configuration for per-tensor current scaling quantization."""

    recipe: Float8CurrentScaling
    mode: str
    qx_dtype: torch.dtype
    qw_dtype: torch.dtype
    qgrad_dtype: torch.dtype
    scale: torch.Tensor
    amax: torch.Tensor

    def __init__(
        self,
        recipe: Float8CurrentScaling,
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
        self.dtype = get_fp8_torch_dtype(recipe, mode == "forward")
        self.fp8_max = get_fp8_max(recipe, mode == "forward")
        # Allocate buffers
        if device is None:
            device = torch.device("npu")
        self.device = device

    def make_quantizers(self) -> list:
        from ...tensor.float8_tensor import Float8CurrentScalingQuantizer

        return [
            Float8CurrentScalingQuantizer(
                self.dtype,
                device=self.device,
                force_pow_2_scales=self.recipe.use_power_2_scales,
                fp8_max=self.fp8_max,
            )
            for _ in range(self.num_quantizers)
        ]
