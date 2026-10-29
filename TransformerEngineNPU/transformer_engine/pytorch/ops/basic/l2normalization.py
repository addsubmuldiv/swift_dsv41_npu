# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for L2 normalization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch

from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Quantizer
from ..op import BasicOperation, OperationContext


class L2Normalization(BasicOperation):
    r"""Apply L2 normalization over the last tensor dimension."""

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        x = input_
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        rsqrt_norm = torch.rsqrt(norm_sq + self.eps)
        output = x * rsqrt_norm
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x, rsqrt_norm)
            ctx.save_for_backward(x, rsqrt_norm)
        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        x, rsqrt_norm = ctx.saved_tensors
        dot = (grad_output * x).sum(dim=-1, keepdim=True)
        grad_input = grad_output * rsqrt_norm - x * dot * rsqrt_norm.pow(3)
        return grad_input, ()
