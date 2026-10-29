# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Reshape operation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch

from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


class Reshape(BasicOperation):
    """Reshape tensor while preserving the inverse shape for backward."""

    def __init__(self, shape: Iterable[int]) -> None:
        super().__init__()
        self._shape = tuple(shape)

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        if ctx.requires_grad:
            ctx.input_shape = input_.size()
        return input_.reshape(*self._shape)

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        return grad_output.reshape(*ctx.input_shape), ()
