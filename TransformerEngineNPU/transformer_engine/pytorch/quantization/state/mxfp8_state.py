# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Optional

import torch

from transformer_engine.common.recipe import MXFP8BlockScaling


from ..quantizer_role import QuantizerRole
from ..utils import get_fp8_torch_dtype
from .base import RecipeState


class MXFP8BlockScalingRecipeState(RecipeState):
    """Configuration for MXFP8 quantization.

    MXFP8 quantization does not require state.

    """

    recipe: MXFP8BlockScaling
    mode: str
    dtype: torch.dtype

    def __init__(
        self,
        recipe: MXFP8BlockScaling,
        *,
        mode: str = "forward",
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
        self.device = device

    def make_quantizers(self) -> list:
        from ...tensor.mxfp8_tensor import MXFP8Quantizer

        return [MXFP8Quantizer(self.dtype) for _ in range(self.num_quantizers)]
