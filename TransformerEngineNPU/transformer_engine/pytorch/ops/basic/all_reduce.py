# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Communication op for all-reduce."""

from __future__ import annotations

from typing import Optional

import torch

from ...distributed import allreduce
from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


class AllReduce(BasicOperation):
    """All-reduce tensor over a process group."""

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        reduce_in_backward: bool = True,
    ) -> None:
        super().__init__()
        self.process_group = process_group
        self._reduce_in_backward = reduce_in_backward

    def op_forward(
        self,
        ctx: OperationContext,  # pylint: disable=unused-argument
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
        next_op_input_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        out, handle = allreduce(input_.contiguous(), group=self.process_group)
        if handle is not None:
            handle.wait()
        return out

    def op_backward(
        self,
        ctx: OperationContext,  # pylint: disable=unused-argument
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output, ()
