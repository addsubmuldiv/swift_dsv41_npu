# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Communication op for all-gather along the first dimension."""

from __future__ import annotations

from typing import Optional

import torch

from ...distributed import gather_along_first_dim, reduce_scatter_along_dim
from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


class AllGather(BasicOperation):
    """All-gather tensor along the first dimension."""

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.process_group = process_group

    def op_forward(
        self,
        ctx: OperationContext,  # pylint: disable=unused-argument
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
        next_op_input_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        out, handle = gather_along_first_dim(input_, self.process_group)
        if handle is not None:
            handle.wait()
        return out

    def op_backward(
        self,
        ctx: OperationContext,  # pylint: disable=unused-argument
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        grad_input, handle = reduce_scatter_along_dim(
            grad_output,
            self.process_group,
        )
        if handle is not None:
            handle.wait()
        return grad_input, ()
