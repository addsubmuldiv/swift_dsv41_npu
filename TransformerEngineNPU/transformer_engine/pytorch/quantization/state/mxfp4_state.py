# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import itertools
from typing import List, Optional

import torch

from transformer_engine.common.recipe import MXFP4BlockScaling

from ..quantizer_role import QuantizerRole
from .base import RecipeState


class MXFP4BlockScalingRecipeState(RecipeState):
    """Configuration for W4A4 MXFP4 block scaling."""

    recipe: MXFP4BlockScaling
    mode: str

    def __init__(
        self,
        recipe: MXFP4BlockScaling,
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
        self.device = torch.device("npu") if device is None else device

    def make_quantizers(self) -> list:
        from ...tensor.mxfp4_tensor import MXFP4Quantizer

        def make_quantizer(qparams):
            return MXFP4Quantizer(
                block_size=self.recipe.block_size,
                qparams=qparams,
                with_rht=qparams.random_hadamard_transform,
            )

        if self.mode == "forward" and self.num_quantizers % 3 == 0:
            return list(
                itertools.chain.from_iterable(
                    [
                        [
                            make_quantizer(self.recipe.fp4_quant_fwd_inp),
                            make_quantizer(self.recipe.fp4_quant_fwd_weight),
                            make_quantizer(self.recipe.fp4_quant_fwd_inp),
                        ]
                        for _ in range(self.num_quantizers // 3)
                    ]
                )
            )

        if self.mode == "backward":
            return [
                make_quantizer(self.recipe.fp4_quant_bwd_grad) for _ in range(self.num_quantizers)
            ]

        return [make_quantizer(self.recipe.fp4_quant_fwd_inp) for _ in range(self.num_quantizers)]
