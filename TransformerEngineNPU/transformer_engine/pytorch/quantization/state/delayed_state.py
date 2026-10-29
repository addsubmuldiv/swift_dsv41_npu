# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Optional

import torch

from transformer_engine.common.recipe import DelayedScaling


from ..quantizer_role import QuantizerRole
from ..utils import get_fp8_torch_dtype
from .base import RecipeState


class DelayedScalingRecipeState(RecipeState):
    """Configuration for delayed per-tensor FP8 scaling."""

    recipe: DelayedScaling
    mode: str
    dtype: torch.dtype
    scale: torch.Tensor
    scale_inv: torch.Tensor
    amax: torch.Tensor
    amax_history: torch.Tensor

    # Persistent training state inherited across role-driven rebuilds.
    # See ``RecipeState.inherit_state_from``.
    _persistent_state_buffers = ("scale", "amax_history")

    def __init__(
        self,
        recipe: DelayedScaling,
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

        # Allocate buffers
        if device is None:
            device = torch.device("npu")
        self.scale = torch.ones(num_quantizers, dtype=torch.float32, device=device)
        self.amax_history = torch.zeros(
            recipe.amax_history_len,
            num_quantizers,
            dtype=torch.float32,
            device=device,
        )

    def make_quantizers(self) -> list:
        from ...tensor.float8_tensor import Float8Quantizer

        return [
            Float8Quantizer(
                self.scale[i],
                self.amax_history[0][i].reshape((1,)),
                self.dtype,
            )
            for i in range(self.num_quantizers)
        ]
