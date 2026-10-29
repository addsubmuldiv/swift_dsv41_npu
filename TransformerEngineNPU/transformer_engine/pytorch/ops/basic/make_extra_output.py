# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Expose the primary input as an extra output."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import torch

from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


class MakeExtraOutput(BasicOperation):
    """Return the input as both primary output and one extra output."""

    num_extra_outputs: int = 1

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
        raise RuntimeError("MakeExtraOutput expects the fuser path")

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        raise RuntimeError("MakeExtraOutput expects the fuser path")

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
        return input_, [(input_,)]

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
        grad_extra_output = basic_op_grad_extra_outputs[0][0]
        if self._in_place:
            grad_extra_output += grad_output
            grad_input = grad_extra_output
        else:
            grad_input = grad_extra_output + grad_output
        return grad_input, [()], [()]
