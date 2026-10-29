# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional

import torch

from transformer_engine.common.recipe import MXFP832x32BlockScaling

from ..quantizer_role import QuantizerRole
from ..utils import get_fp8_torch_dtype
from .base import RecipeState


class MXFP832x32BlockScalingRecipeState(RecipeState):
    """Stateless recipe state for 32x32 MXFP8 weight scaling."""

    def __init__(
        self,
        recipe: MXFP832x32BlockScaling,
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
        self.dtype = get_fp8_torch_dtype(recipe, mode == "forward")
        self.device = torch.device("npu") if device is None else device

    def make_quantizers(self) -> list:
        from ...tensor.mxfp8_tensor import MXFP832x32Quantizer, MXFP8Quantizer

        quantizers = []
        for index in range(self.num_quantizers):
            role = self.roles[index] if self.roles is not None else None
            if role is not None and role.tensor_type == "weight":
                quantizers.append(MXFP832x32Quantizer(self.dtype))
            else:
                quantizers.append(MXFP8Quantizer(self.dtype))
        return quantizers
