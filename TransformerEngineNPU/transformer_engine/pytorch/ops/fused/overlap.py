# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import torch
import torch_npu

from ...constants import TensorUsage
from ...distributed import get_hccl_comm_name, reduce_scatter_along_dim
from ...quantized_tensor import QuantizedTensor, Quantizer
from ...tensor import MXFP8Tensor


class CommOverlapOps:
    @staticmethod
    def allgather_matmul(
        A,
        B,
        bias,
        world_size,
        group,
        usage_a,
        usage_b,
        out_dtype,
        return_gathered: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(A, QuantizedTensor):
            return A.allgather_matmul(
                B, bias, world_size, group, usage_a, usage_b, out_dtype, return_gathered
            )
        if len(A.shape) == 3:
            a = A.view(A.shape[0] * A.shape[1], A.shape[2])
        else:
            a = A.view(-1, A.shape[-1])
        out, allgather_grad_out = torch_npu.npu_all_gather_base_mm(
            a.t() if usage_a == TensorUsage.LHS_TRANS else a,
            B.t() if usage_b == TensorUsage.RHS_TRANS else B,
            get_hccl_comm_name(group),
            world_size,
            bias=bias,
            gather_index=0,
        )
        if len(A.shape) == 3:
            out = out.view(int(out.shape[0] / A.shape[1]), A.shape[1], out.shape[1])
        return out, allgather_grad_out

    @staticmethod
    def matmul_reduce_scatter(
        A,
        B,
        bias,
        world_size,
        group,
        usage_a: TensorUsage,
        usage_b: TensorUsage,
        out_dtype,
    ) -> torch.Tensor:
        if isinstance(A, QuantizedTensor):
            return A.matmul_reduce_scatter(B, bias, world_size, group, usage_a, usage_b, out_dtype)
        if len(A.shape) == 3:
            a = A.view(A.shape[0] * A.shape[1], A.shape[2])
        else:
            a = A.view(-1, A.shape[-1])
        out = torch_npu.npu_mm_reduce_scatter_base(
            a.t() if usage_a == TensorUsage.LHS_TRANS else a,
            B.t() if usage_b == TensorUsage.RHS_TRANS else B,
            get_hccl_comm_name(group),
            world_size,
            reduce_op="sum",
            bias=bias,
        )
        if len(A.shape) == 3:
            out = out.view(int(out.shape[0] / A.shape[1]), A.shape[1], out.shape[1])
        return out

    @staticmethod
    def reduce_scatter(tensor, quantizer: Quantizer, world_size, group, use_quant):
        if use_quant:
            # 暂时只支持MXFP8
            out = MXFP8Tensor.quant_reduce_scatter(tensor, quantizer, world_size, group)
            return out, None
        return reduce_scatter_along_dim(tensor, group)
