# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Constant scaling operation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch

from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


class ConstantScale(BasicOperation):
    """Scale a tensor by a constant value."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        return input_ * self.scale

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        return grad_output * self.scale, ()
