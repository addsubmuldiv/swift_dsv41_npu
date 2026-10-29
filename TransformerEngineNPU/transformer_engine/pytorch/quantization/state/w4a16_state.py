# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from typing import List, Optional

import torch

from transformer_engine.common.recipe import W4A16BlockScaling

from ..quantizer_role import QuantizerRole
from .base import RecipeState


class W4A16BlockScalingRecipeState(RecipeState):
    """Configuration for W4A16 QAT fake quantization."""

    recipe: W4A16BlockScaling
    mode: str

    def __init__(
        self,
        recipe: W4A16BlockScaling,
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
        self.device = torch.device("npu") if device is None else device

    def make_quantizers(self) -> list:
        from ...tensor.dummy_tensor import DummyQuantizer
        from ...tensor.w4a16_quantizer import FakeMXFP4Quantizer

        if self.mode == "forward":
            assert self.num_quantizers % 3 == 0  # input, weight, output per GEMM
            return [
                FakeMXFP4Quantizer(
                    block_size=self.recipe.block_size,
                    qparams=self.recipe.fp4_quant_weight,
                )
                if enabled and index % 3 == 1
                else DummyQuantizer()
                for index, enabled in enumerate(self._quantization_scope_mask(self.num_quantizers))
            ]

        assert self.mode == "backward", f"Unexpected mode {self.mode}"
        return [DummyQuantizer() for _ in range(self.num_quantizers)]
