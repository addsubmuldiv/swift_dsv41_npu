# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Bias operation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

import torch

from ...tensor import Quantizer
from ...utils import canonicalize_device, canonicalize_dtype
from ..op import BasicOperation, OperationContext


class Bias(BasicOperation):
    """Add a learnable 1D bias to the last tensor dimension."""

    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel: bool = False,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        device = canonicalize_device(device)
        dtype = canonicalize_dtype(dtype)
        self.size = size
        self.tensor_parallel = tensor_parallel
        self.tensor_parallel_group = tensor_parallel_group
        self.sequence_parallel = sequence_parallel
        self.device = device

        local_size = size
        if tensor_parallel:
            if tensor_parallel_group is None:
                raise RuntimeError("Tensor-parallel bias requires a process group")
            world_size = torch.distributed.get_world_size(tensor_parallel_group)
            local_size = math.ceil(size / world_size)

        self.bias = torch.nn.Parameter(torch.empty(local_size, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset bias to zero."""

        with torch.no_grad():
            self.bias.zero_()

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        bias_dims = (1,) * (input_.dim() - 1) + (-1,)
        return input_ + self.bias.view(bias_dims)

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        if grad_output.dim() > 1:
            dbias = grad_output.sum(tuple(range(grad_output.dim() - 1)))
        else:
            dbias = grad_output
        return grad_output, (dbias,)
