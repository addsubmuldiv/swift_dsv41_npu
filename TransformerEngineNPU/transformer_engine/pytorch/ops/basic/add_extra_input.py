# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Add an extra input to the main tensor."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import torch

from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


class AddExtraInput(BasicOperation):
    """Add the operation's extra input to its primary input."""

    num_extra_inputs: int = 1

    def __init__(self, *, in_place: bool = False) -> None:
        super().__init__()
        self._in_place = in_place

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        raise RuntimeError("AddExtraInput expects an extra input in the fuser path")

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        raise RuntimeError("AddExtraInput expects an extra input in the fuser path")

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, ...]]]:
        x = basic_op_extra_inputs[0][0]
        if self._in_place:
            x = x.detach()
            x += input_
            return x, [()]
        return x + input_, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[Optional[torch.Tensor], ...]],
    ]:
        return grad_output, [()], [(grad_output,)]
