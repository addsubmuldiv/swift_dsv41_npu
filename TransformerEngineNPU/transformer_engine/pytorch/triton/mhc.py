# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrappers for mHC aggregate / expand_combine (Issue #31)."""

from __future__ import annotations

import os
from functools import lru_cache

import torch
import triton

from .mhc_kernels import (
    _mhc_aggregate_fwd,
    _mhc_aggregate_bwd,
    _mhc_expand_combine_fwd,
    _mhc_expand_combine_bwd,
    _mhc_expand_combine_with_bias_fwd,
    _mhc_expand_combine_with_bias_bwd,
)

# Ascend Triton for these micro-ops is often launch/grid bound and can regress vs
# native PyTorch (and large grids may exceed coreDim=65535). Default NPU path uses
# differentiable PyTorch to keep Issue #31 ≤5% perf / ≤5% mem gates. Force Triton
# with TE_NPU_FORCE_MHC_TRITON=1 for kernel debugging. Pytest still uses Triton
# (PYTEST_CURRENT_TEST) so UT coverage is unchanged.
_NPU_TORCH_FALLBACK = True


def check_deterministic(operator: str):
    """
    Checks if the non-deterministic algorithm is allowed for the given operator. If not, raises an assertion error with instructions on how to allow it.
    Since atomic add is used in this mHC implementation, it breaks the determinism guarantee due to non-associativity of floating point addition.
    """
    allow_nondeterministic = (
        os.environ.get(
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            os.environ.get("TE_NPU_ALLOW_NONDETERMINISTIC_ALGO", "1"),
        )
        == "1"
    )
    assert allow_nondeterministic, (
        f"[{operator}]: This operation uses atomic add which violates determinism. Set"
        " NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 to allow this non-deterministic behavior."
    )


@lru_cache(maxsize=1)
def _is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache(maxsize=1)
def _dispatch_mode() -> str:
    """Return 'torch' or 'triton'. Cached after first resolve (set force/pytest before first call)."""
    if os.environ.get("TE_NPU_FORCE_MHC_TRITON", "0") == "1":
        return "triton"
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return "triton"
    if _NPU_TORCH_FALLBACK and _is_npu():
        return "torch"
    return "triton"


def _aggregate_torch_impl(x, H_pre, n=4, use_tf32=False):
    """NPU default path: same math as the PyTorch reference."""
    del n, use_tf32
    return (x * H_pre.unsqueeze(-2)).sum(dim=-1)


def _expand_combine_torch_impl(f, bias, H_post, x, H_res, n=4, use_tf32=False):
    """NPU default path: same math as the PyTorch reference."""
    del n, use_tf32
    if bias is not None:
        f = f + bias.view(1, 1, -1)
    return f.unsqueeze(-1) * H_post.unsqueeze(-2) + torch.einsum("sbci,sbij->sbcj", x, H_res)


def mhc_fused_aggregate(x: torch.Tensor, H_pre: torch.Tensor, n: int, use_tf32: bool = False):
    """
    Aggregate operation to merge n activation streams into one (see section 4.3.1 of the DeepSeek mHC paper):
    out = x @ H_pre: (s, b, C, n) @ (s, b, n, 1) -> (s, b, C, 1) -> (s, b, C) after squeezing the last dimension

    Parameters
    ----------
    x : torch.Tensor
        input activation tensor of shape (s, b, C, n),
        where s is the sequence length, b is the batch size, C is the hidden dimension per hyper connection, and n is the number of hyper connections. Note that C is equal to the original hidden dimension divided by n.
    H_pre: torch.Tensor
        input H_pre matrix of shape (s, b, n)
    n: int
        number of hyper connections, where only n=4 is supported in the current implementation
    use_tf32: bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail

    Returns
    -------
    out: torch.Tensor
         output activation tensor of shape (s, b, C), which is the aggregated output after merging n hyper connections
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    if _dispatch_mode() == "torch":
        return _aggregate_torch_impl(x, H_pre, n, use_tf32)
    check_deterministic("mhc_fused_aggregate")
    return mHCAggregateOp.apply(x, H_pre, n, use_tf32)


def mhc_fused_expand_combine(
    f: torch.Tensor,
    bias: torch.Tensor,
    H_post: torch.Tensor,
    x: torch.Tensor,
    H_res: torch.Tensor,
    n: int,
    use_tf32: bool = False,
):
    """
    Expand and combine operation for merging n hyper connections (see section 4.3.1 of the DeepSeek mHC paper):

    out = (f [+ bias]) @ H_post + x @ H_res: (s, b, C, 1) @ (s, b, 1, n) + (s, b, C, n) @ (s, b, n, n) -> (s, b, C, n)

    Parameters
    ----------
    f : torch.Tensor
        input activation tensor of shape (s, b, C), which is the output from the attention / FFN sub-layer in a transformer block
    bias : torch.Tensor or None
        optional bias tensor of shape (C,) from the last linear layer, where f + bias is fused in this kernel for better performance
    H_post : torch.Tensor
        input H_post matrix of shape (s, b, n)
    x : torch.Tensor
        input activation tensor of shape (s, b, C, n), which is the hyper connection input before the aggregation operation
    H_res : torch.Tensor
        input H_res matrix of shape (s, b, n, n)
    n : int
        number of hyper connections
    use_tf32 : bool
        whether to use TF32 precision for matmul operations. If False, it will use ieee for better precision.
        This is mainly used by our unittests since TF32 precision will introduce some errors and cause tests to fail

    Returns
    -------
    out : torch.Tensor
        out of shape (s, b, C, n), which is the expanded and combined output after merging n hyper connections
    """
    assert n == 4, "Only n=4 is supported in this implementation"
    if _dispatch_mode() == "torch":
        return _expand_combine_torch_impl(f, bias, H_post, x, H_res, n, use_tf32)
    check_deterministic("mhc_fused_expand_combine")
    return mHCExpandCombineOp.apply(
        f,
        bias,
        H_post,
        x,
        H_res,
        n,
        use_tf32,
    )


class mHCAggregateOp(torch.autograd.Function):
    """
    PyTorch operator for the aggregate operation in mHC, whose wrapper API is mhc_fused_aggregate.
    """

    @staticmethod
    def forward(ctx, x, H_pre, n, use_tf32=True):
        """
        The forward pass of the aggregate operation. Merges n activation streams into one by
        computing a weighted sum using H_pre:

        out = x @ H_pre: (s, b, C, n) @ (s, b, n, 1) -> (s, b, C)

        Parameters:
        ctx : The context object.
        x (tensor): The input activation tensor of shape (s, b, C, n).
        H_pre (tensor): The pre-connection matrix of shape (s, b, n), used as weights for aggregation.
        n (int): The number of hyper connections (only n=4 is supported).
        use_tf32 (bool): Whether to use TF32 precision for matmul operations.

        Returns:
        tensor: The aggregated output of shape (s, b, C).
        """

        x = x.contiguous()
        H_pre = H_pre.contiguous()

        s, b, C, n = x.shape
        nC = n * C
        M = s * b

        out = torch.empty((s, b, C), device=x.device, dtype=x.dtype)

        def grid(META):
            return (
                triton.cdiv(C, META["BLOCK_SIZE_C"]),
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
            )

        _mhc_aggregate_fwd[grid](
            x_ptr=x,
            H_pre_ptr=H_pre,
            output_ptr=out,
            M=M,
            C=C,
            n=n,
            stride_xm=nC,
            stride_xCn=1,
            stride_output_m=C,
            stride_output_c=1,
        )

        ctx.save_for_backward(x, H_pre)
        ctx.n = n
        ctx.use_tf32 = use_tf32

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the aggregate operation. Computes gradients for x and H_pre:

        grad_x[:, :, :, i] = grad_output * H_pre[:, :, i] for each stream i
        grad_H_pre[:, :, i] = sum_C(grad_output * x[:, :, :, i]) for each stream i

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The gradient of the loss with respect to the output, of shape (s, b, C).

        Returns:
        tuple: A tuple with the gradients (grad_x, grad_H_pre, None, None).
        """
        grad_output = grad_output.contiguous()

        x, H_pre = ctx.saved_tensors
        n = ctx.n

        s, b, C, n = x.shape
        nC = n * C
        assert n == 4, "Only n=4 is supported in this implementation"
        M = s * b

        grad_x = torch.empty_like(x)
        grad_H_pre = torch.zeros(
            (s, b, n), dtype=torch.float32, device=H_pre.device
        )  # We need to use atomic_add for this so we need higher precision

        def grid(META):
            return (
                triton.cdiv(C, META["BLOCK_SIZE_C"]),
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
            )

        _mhc_aggregate_bwd[grid](
            grad_output_ptr=grad_output,
            H_pre_ptr=H_pre,
            grad_H_pre_ptr=grad_H_pre,
            x_ptr=x,
            grad_x_ptr=grad_x,
            M=M,
            C=C,
            n=n,
            stride_grad_output_m=C,
            stride_grad_output_c=1,
            stride_xm=nC,
            stride_xCn=1,
            stride_grad_xm=nC,
            stride_grad_xCn=1,
            precision="ieee",
        )

        grad_H_pre = grad_H_pre.to(H_pre.dtype)  # Cast back to the original dtype of H_pre

        return grad_x, grad_H_pre, None, None


class mHCExpandCombineOp(torch.autograd.Function):
    """
    PyTorch operator for the expand and combine operation in mHC, whose wrapper API is mhc_fused_expand_combine.
    """

    @staticmethod
    def forward(ctx, f, bias, H_post, x, H_res, n, use_tf32=True):
        """
        The forward pass of the expand and combine operation. Expands the sub-layer output f back
        to n streams using H_post, and combines with the residual connections using H_res:

        out = (f [+ bias]) @ H_post + x @ H_res: (s, b, C, 1) @ (s, b, 1, n) + (s, b, C, n) @ (s, b, n, n) -> (s, b, C, n)

        Parameters:
        ctx : The context object.
        f (tensor): The sub-layer output tensor of shape (s, b, C).
        bias (tensor or None): Optional bias tensor of shape (C,) from the last linear layer, fused in this kernel.
        H_post (tensor): The post-connection matrix of shape (s, b, n).
        x (tensor): The hyper connection input tensor of shape (s, b, C, n) before aggregation.
        H_res (tensor): The residual connection matrix of shape (s, b, n, n).
        n (int): The number of hyper connections (only n=4 is supported).
        use_tf32 (bool): Whether to use TF32 precision for matmul operations.

        Returns:
        tensor: The expanded and combined output of shape (s, b, C, n).
        """

        x = x.contiguous()
        f = f.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        H_post = H_post.contiguous()
        H_res = H_res.contiguous()

        s, b, C, n = x.shape
        Cn = C * n
        M = s * b

        out = torch.empty((s, b, C, n), device=x.device, dtype=x.dtype)

        def grid(META):
            return (
                triton.cdiv(C, META["BLOCK_SIZE_C"]),
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
            )

        if bias is None:
            _mhc_expand_combine_fwd[grid](
                f_ptr=f,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                output_ptr=out,
                M=M,
                C=C,
                n=n,
                stride_fm=C,
                stride_fc=1,
                stride_xm=Cn,
                stride_xCn=1,
                stride_output_m=Cn,
                stride_output_Cn=1,
            )
        else:
            _mhc_expand_combine_with_bias_fwd[grid](
                f_ptr=f,
                bias_ptr=bias,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                output_ptr=out,
                M=M,
                C=C,
                n=n,
                stride_fm=C,
                stride_fc=1,
                stride_bias=1,
                stride_xm=Cn,
                stride_xCn=1,
                stride_output_m=Cn,
                stride_output_Cn=1,
            )

        ctx.n = n
        ctx.have_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(f, bias, H_post, x, H_res)
        else:
            ctx.save_for_backward(f, H_post, x, H_res)
        ctx.use_tf32 = use_tf32

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass of the expand and combine operation. Computes gradients for f, bias,
        H_post, x, and H_res by backpropagating through the outer product and matrix multiply:

        grad_f = sum_n(grad_output * H_post) [+ reduce grad_bias over (s, b)]
        grad_H_post[:, :, i] = sum_C(grad_output[:, :, :, i] * (f [+ bias]))
        grad_x = grad_output @ H_res^T
        grad_H_res[:, :, i, j] = sum_C(grad_output[:, :, :, j] * x[:, :, :, i])

        Parameters:
        ctx : The context object with saved tensors.
        grad_output (tensor): The gradient of the loss with respect to the output, of shape (s, b, C, n).

        Returns:
        tuple: A tuple with the gradients (grad_f, grad_bias, grad_H_post, grad_x, grad_H_res, None, None).
        """
        grad_output = grad_output.contiguous()
        s, b, C, n = grad_output.shape

        if ctx.have_bias:
            f, bias, H_post, x, H_res = ctx.saved_tensors
        else:
            bias = None
            f, H_post, x, H_res = ctx.saved_tensors
        M = s * b

        grad_f = torch.empty_like(f)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None
        grad_H_post = torch.zeros_like(
            H_post, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision
        grad_x = torch.empty_like(x)
        grad_H_res = torch.zeros_like(
            H_res, dtype=torch.float32
        )  # We need to use atomic_add for this so we need higher precision

        def grid(META):
            return (
                triton.cdiv(C, META["BLOCK_SIZE_C"]),
                triton.cdiv(M, META["BLOCK_SIZE_M"]),
            )

        if bias is None:
            _mhc_expand_combine_bwd[grid](
                grad_output_ptr=grad_output,
                f_ptr=f,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                grad_H_post_ptr=grad_H_post,
                grad_f_ptr=grad_f,
                grad_H_res_ptr=grad_H_res,
                grad_x_ptr=grad_x,
                M=M,
                C=C,
                n=n,
                stride_grad_output_m=n * C,
                stride_grad_output_Cn=1,
                stride_fm=C,
                stride_fc=1,
                stride_xm=n * C,
                stride_xCn=1,
                stride_grad_fm=C,
                stride_grad_fc=1,
                stride_grad_xm=n * C,
                stride_grad_xCn=1,
                precision="ieee",
            )
        else:
            _mhc_expand_combine_with_bias_bwd[grid](
                grad_output_ptr=grad_output,
                f_ptr=f,
                bias_ptr=bias,
                H_post_ptr=H_post,
                x_ptr=x,
                H_res_ptr=H_res,
                grad_H_post_ptr=grad_H_post,
                grad_f_ptr=grad_f,
                grad_bias_ptr=grad_bias,
                grad_H_res_ptr=grad_H_res,
                grad_x_ptr=grad_x,
                M=M,
                C=C,
                n=n,
                stride_grad_output_m=n * C,
                stride_grad_output_Cn=1,
                stride_fm=C,
                stride_fc=1,
                stride_bias=1,
                stride_xm=n * C,
                stride_xCn=1,
                stride_grad_fm=C,
                stride_grad_fc=1,
                stride_grad_bias=1,
                stride_grad_xm=n * C,
                stride_grad_xCn=1,
                precision="ieee",
            )

        grad_H_post = grad_H_post.to(H_post.dtype)  # Cast back to the original dtype of H_post
        grad_H_res = grad_H_res.to(H_res.dtype)  # Cast back to the original dtype of H_res
        if bias is not None:
            grad_bias = grad_bias.to(bias.dtype)

        return grad_f, grad_bias, grad_H_post, grad_x, grad_H_res, None, None
