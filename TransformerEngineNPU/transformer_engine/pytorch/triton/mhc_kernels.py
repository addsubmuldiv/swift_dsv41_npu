# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernels for mHC aggregate / expand_combine (Issue #31). NPU-adapted from NVTE."""

from __future__ import annotations

import itertools
import os

import triton
import triton.language as tl


def aggregate_config():
    block_m = [1, 2, 4]
    block_c = [64, 128, 256]
    warps = [1, 2, 4]
    stages = [1, 2, 3, 4]

    configs = []
    for m, c, w, s in itertools.product(block_m, block_c, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_C": c}, num_warps=w, num_stages=s)
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "1") == "1":  # NPU: default single config
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=aggregate_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_aggregate_fwd(
    x_ptr,  # # (M, C, n)
    H_pre_ptr,  # (M, n)
    output_ptr,  # (M, C)
    M,
    C,
    n: tl.constexpr,
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_c,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    output = x @ H_pre: (M, C, n) @ (M, n, 1) = (M, C, 1)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_c == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    offs_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + offs_H_pre, mask=offs_H_pre < M * n, other=0.0
    )  # (BLOCK_SIZE_M * n)
    H_pre = H_pre.reshape(BLOCK_SIZE_M, 2, 2)
    H_pre01, H_pre23 = tl.split(H_pre)
    H_pre0, H_pre1 = tl.split(H_pre01)
    H_pre2, H_pre3 = tl.split(H_pre23)  # (BLOCK_SIZE_M, 1)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)

    x = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x)
    x0, x1 = tl.split(x01)
    x2, x3 = tl.split(x23)  # (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # x @ H_pre: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1)
    # triton doesn't support dot prod with inner dimension < 16, so we need to manually unroll the computation for n=4:
    # x @ H_pre = x[:, :, 0] * H_pre[:, 0]
    #           + x[:, :, 1] * H_pre[:, 1]
    #           + x[:, :, 2] * H_pre[:, 2]
    #           + x[:, :, 3] * H_pre[:, 3]
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    out_acc = tl.fma(x0, H_pre0[:, None], out_acc)
    out_acc = tl.fma(x1, H_pre1[:, None], out_acc)
    out_acc = tl.fma(x2, H_pre2[:, None], out_acc)
    out_acc = tl.fma(x3, H_pre3[:, None], out_acc)

    out = out_acc.to(x.dtype)

    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_c[None, :] * stride_output_c
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_c[None, :])


@triton.autotune(configs=aggregate_config(), key=["M", "C"], reset_to_zero=["grad_H_pre_ptr"])
@triton.jit
def _mhc_aggregate_bwd(
    grad_output_ptr,  # (M, C)
    H_pre_ptr,  # (M, n)
    grad_H_pre_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    grad_x_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_c,
    stride_xm,
    stride_xCn,
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
):
    """
    Forward:
        out = x @ H_pre: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
    Backward:
        grad_H_pre = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) = (BLOCK_SIZE_M, n, 1)
        grad_H_pre.T = grad_output.T @ x: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
            which is easier to compute since transposing grad_H_pre and grad_output is just view change
        grad_x = grad_output @ H_pre.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_c == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    grad_output_ptrs = (
        grad_output_ptr
        + offs_m[:, None] * stride_grad_output_m
        + offs_c[None, :] * stride_grad_output_c
    )
    grad_output = tl.load(
        grad_output_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)

    # Unroll H_pre grad (n=4); Ascend tl.dot with K=4 is unreliable
    x_r = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x_r)
    x0, x1 = tl.split(x01)
    x2, x3 = tl.split(x23)
    g0 = tl.sum((grad_output * x0).to(tl.float32), axis=1)
    g1 = tl.sum((grad_output * x1).to(tl.float32), axis=1)
    g2 = tl.sum((grad_output * x2).to(tl.float32), axis=1)
    g3 = tl.sum((grad_output * x3).to(tl.float32), axis=1)
    # Pack (M,n) via join so last dim is 2 (tl.split/join constraint)
    g01 = tl.join(g0, g1)
    g23 = tl.join(g2, g3)
    grad_H_pre = tl.reshape(tl.join(g01, g23), (BLOCK_SIZE_M * n,))
    offs_grad_H_pre = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    tl.atomic_add(
        grad_H_pre_ptr + offs_grad_H_pre, grad_H_pre, mask=offs_grad_H_pre < M * n, sem="relaxed"
    )

    H_pre_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_pre = tl.load(
        H_pre_ptr + H_pre_offs, mask=H_pre_offs < M * n, other=0.0
    )  # (BLOCK_SIZE_M * n)
    H_pre = tl.reshape(H_pre, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    # grad_x = grad_output @ H_pre.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    grad_x = grad_output[:, :, None] * H_pre[:, None, :]  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))

    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
    tl.store(
        grad_x_ptrs,
        grad_x,
        mask=mask_m[:, None] & mask_cn[None, :],
    )


def expand_combine_config():
    block_m = [1, 2, 4]
    block_c = [128, 256]
    warps = [1, 2]
    stages = [1, 2, 3, 4]

    configs = []
    for m, c, w, s in itertools.product(block_m, block_c, warps, stages):
        configs.append(
            triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_C": c}, num_warps=w, num_stages=s)
        )
    if os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "1") == "1":  # NPU: default single config
        configs = configs[:1]
    return configs


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_expand_combine_fwd(
    f_ptr,  # (M, C)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    output_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_fm,
    stride_fc,
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_Cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    output = f @ H_post: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
           + x @ H_res: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_Cn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

    offs_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + offs_H_post, mask=offs_H_post < M * n, other=0.0)
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    # Residual connection path: res_out = f @ H_post:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # Due to broadcasting, it's equivalent to a multiplicaiton
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    out_acc = tl.fma(f[:, :, None], H_post[:, None, :], out_acc)

    # x @ H_res — load each H_res row with fully-inlined offsets
    # (Ascend: ptr + offs_row + k*n does NOT add k*n correctly)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0)
    x_reshape = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x_reshape)
    x0, x1 = tl.split(x01)
    x2, x3 = tl.split(x23)

    # H_res[m, i, j] at m*n*n + i*n + j ; load rows as (BLOCK_SIZE_M, n)
    H_res0 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 0 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    H_res1 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 1 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    H_res2 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 2 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    H_res3 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 3 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    # x0..x3 from (...,2,2) split are streams 0,2,1,3 — pair with H rows 0,2,1,3
    out_acc = tl.fma(x0[:, :, None], H_res0[:, None, :], out_acc)
    out_acc = tl.fma(x1[:, :, None], H_res2[:, None, :], out_acc)
    out_acc = tl.fma(x2[:, :, None], H_res1[:, None, :], out_acc)
    out_acc = tl.fma(x3[:, :, None], H_res3[:, None, :], out_acc)

    out = out_acc.to(x.dtype)
    out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    output_ptrs = (
        output_ptr + offs_m[:, None] * stride_output_m + offs_cn[None, :] * stride_output_Cn
    )
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_cn[None, :])


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
    reset_to_zero=["grad_H_post_ptr", "grad_H_res_ptr"],
)
@triton.jit
def _mhc_expand_combine_bwd(
    grad_output_ptr,  # (M, C, n)
    f_ptr,  # (M, C)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    grad_H_post_ptr,  # (M, n)
    grad_f_ptr,  # (M, C)
    grad_H_res_ptr,  # (M, n, n)
    grad_x_ptr,  # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_Cn,
    stride_fm,
    stride_fc,
    stride_xm,
    stride_xCn,
    stride_grad_fm,
    stride_grad_fc,
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
):
    """
    Each block
    It reads
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of f, which is the output of the attention / FFN module
    - (BLOCK_SIZE_M, n) of H_post, which is applied for the transformation of the attention / FFN output
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of x, which is the skip connection's input
    - (BLOCK_SIZE_M, n*n) of H_res, which is applied for the transformation of the skip connection
    and writes
    - (BLOCK_SIZE_M, n) of grad_H_post
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of grad_f
    - (BLOCK_SIZE_M, n, n) of grad_H_res
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of grad_x

    Forward:
        out = f @ H_post + x @ H_res
    Backward:
        GEMM:
        grad_H_post = f.T @ grad_output: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
        grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        Not GEMM:
        grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
        grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """

    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_Cn == 1)
    tl.assume(stride_grad_fm > 0 and stride_grad_fc == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

    H_post_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + H_post_offs, mask=H_post_offs < M * n, other=0.0)
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0
    )  # (BLOCK_SIZE_M, n, n)
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)

    grad_out_ptrs = (
        grad_output_ptr
        + offs_m[:, None] * stride_grad_output_m
        + offs_cn[None, :] * stride_grad_output_Cn
    )
    grad_out = tl.load(
        grad_out_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)
    grad_out = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_post =  f.T @ grad_output # (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
    # grad_H_post[:, i] = sum_C(f * grad_out[:,:,i]) — unrolled
    go_r = tl.reshape(grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    go01, go23 = tl.split(go_r)
    go0, go1 = tl.split(go01)
    go2, go3 = tl.split(go23)
    hp0 = tl.sum((f * go0).to(tl.float32), axis=1)
    hp1 = tl.sum((f * go1).to(tl.float32), axis=1)
    hp2 = tl.sum((f * go2).to(tl.float32), axis=1)
    hp3 = tl.sum((f * go3).to(tl.float32), axis=1)
    grad_H_post = tl.reshape(tl.join(tl.join(hp0, hp1), tl.join(hp2, hp3)), (BLOCK_SIZE_M * n,))
    offs_grad_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    tl.atomic_add(
        grad_H_post_ptr + offs_grad_H_post,
        grad_H_post,
        mask=offs_grad_H_post < M * n,
        sem="relaxed",
    )

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)
    x = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_res[i,j] = sum_C(x[:,:,i] * grad_out[:,:,j]) — unrolled (no tl.trans)
    x_r = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x_r)
    xx0, xx1 = tl.split(x01)
    xx2, xx3 = tl.split(x23)

    # Plain xx/go products (same basis as H_post hp pack). Store rows with atomics
    # — joining (M,n) rows then flattening transposes into (M,n*n) layout.
    row0 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx0 * go0).to(tl.float32), axis=1),
                tl.sum((xx0 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx0 * go2).to(tl.float32), axis=1),
                tl.sum((xx0 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    row1 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx1 * go0).to(tl.float32), axis=1),
                tl.sum((xx1 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx1 * go2).to(tl.float32), axis=1),
                tl.sum((xx1 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    row2 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx2 * go0).to(tl.float32), axis=1),
                tl.sum((xx2 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx2 * go2).to(tl.float32), axis=1),
                tl.sum((xx2 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    row3 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx3 * go0).to(tl.float32), axis=1),
                tl.sum((xx3 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx3 * go2).to(tl.float32), axis=1),
                tl.sum((xx3 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    # xx0,xx1,xx2,xx3 = streams 0,2,1,3 — write to natural rows 0,2,1,3
    offs_m_h = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_mh = offs_m_h < M
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 0 * n + tl.arange(0, n)[None, :],
        row0,
        mask=mask_mh[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 2 * n + tl.arange(0, n)[None, :],
        row1,
        mask=mask_mh[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 1 * n + tl.arange(0, n)[None, :],
        row2,
        mask=mask_mh[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 3 * n + tl.arange(0, n)[None, :],
        row3,
        mask=mask_mh[:, None],
        sem="relaxed",
    )

    grad_out_reshape = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    grad_out01, grad_out23 = tl.split(
        grad_out_reshape
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    grad_out0, grad_out1 = tl.split(
        grad_out01
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_out2, grad_out3 = tl.split(
        grad_out23
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)
    # Triton doesn't support dot prod with inner dimension < 16, so we need to hack this:
    # grad_f = grad_out[:, :, 0] @ H_post.T[:, 0, :] (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, 1)
    #        + grad_out[:, :, 1] @ H_post.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_post.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_post.T[:, 3, :]
    # where H_post.T[:, i, :] = H_post[:, :, i]
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, 2, 2))
    H_post01, H_post23 = tl.split(H_post)  # (BLOCK_SIZE_M, 2), (BLOCK_SIZE_M, 2)
    H_post0, H_post1 = tl.split(H_post01)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)
    H_post2, H_post3 = tl.split(H_post23)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)

    grad_f_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    # (BLOCK_SIZE_M, BLOCK_SIZE_C) * (BLOCK_SIZE_M, 1) -> (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_f_acc = tl.fma(grad_out0, H_post0[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out1, H_post1[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out2, H_post2[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out3, H_post3[:, None], grad_f_acc)
    grad_f = grad_f_acc.to(f.dtype)

    grad_f_ptrs = grad_f_ptr + offs_m[:, None] * stride_grad_fm + offs_c[None, :] * stride_grad_fc
    tl.store(grad_f_ptrs, grad_f, mask=mask_m[:, None] & mask_c[None, :])

    # grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # The inner dim is n=4 which is too small for triton, so we will manually unroll the matmul
    # grad_x = grad_out[:, :, 0] @ H_res.T[:, 0, :]
    #        + grad_out[:, :, 1] @ H_res.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_res.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_res.T[:, 3, :]
    # where H_res.T[:, i, :] = H_res[:, :, i]
    # Due to broadcasting, it's equivalent to multiplying each H_res[:, i, :].T with grad_out[:, i, :]

    H_res_reshape = tl.reshape(H_res, (BLOCK_SIZE_M, n, 2, 2))  # (BLOCK_SIZE_M, n, 2, 2)
    H_res01, H_res23 = tl.split(H_res_reshape)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    grad_x_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    grad_x_acc = tl.fma(grad_out0[:, :, None], H_res0[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out1[:, :, None], H_res1[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out2[:, :, None], H_res2[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out3[:, :, None], H_res3[:, None, :], grad_x_acc)

    grad_x = grad_x_acc.to(x.dtype)
    grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_cn[None, :])


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
)
@triton.jit
def _mhc_expand_combine_with_bias_fwd(
    f_ptr,  # (M, C)
    bias_ptr,  # (C,)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    output_ptr,  # # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_fm,
    stride_fc,
    stride_bias,
    stride_xm,
    stride_xCn,
    stride_output_m,
    stride_output_Cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    output = (f + bias[None, :, None]) @ H_post: (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
           + x @ H_res: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_bias == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_output_m > 0 and stride_output_Cn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)
    bias = tl.load(bias_ptr + offs_c * stride_bias, mask=mask_c, other=0.0)  # (BLOCK_SIZE_C,)

    offs_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + offs_H_post, mask=offs_H_post < M * n, other=0.0)
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    # Residual connection path: res_out = f @ H_post + bias @ H_post:
    # (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, n)  = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # Due to broadcasting, it's equivalent to a multiplicaiton
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    out_acc = tl.fma(bias[None, :, None], H_post[:, None, :], out_acc)
    out_acc = tl.fma(f[:, :, None], H_post[:, None, :], out_acc)

    # x @ H_res — load each H_res row with fully-inlined offsets
    # (Ascend: ptr + offs_row + k*n does NOT add k*n correctly)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0)
    x_reshape = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x_reshape)
    x0, x1 = tl.split(x01)
    x2, x3 = tl.split(x23)

    # H_res[m, i, j] at m*n*n + i*n + j ; load rows as (BLOCK_SIZE_M, n)
    H_res0 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 0 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    H_res1 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 1 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    H_res2 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 2 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )
    H_res3 = tl.load(
        H_res_ptr + offs_m[:, None] * (n * n) + 3 * n + tl.arange(0, n)[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    # x0..x3 from (...,2,2) split are streams 0,2,1,3 — pair with H rows 0,2,1,3
    out_acc = tl.fma(x0[:, :, None], H_res0[:, None, :], out_acc)
    out_acc = tl.fma(x1[:, :, None], H_res2[:, None, :], out_acc)
    out_acc = tl.fma(x2[:, :, None], H_res1[:, None, :], out_acc)
    out_acc = tl.fma(x3[:, :, None], H_res3[:, None, :], out_acc)

    out = out_acc.to(x.dtype)
    out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    output_ptrs = (
        output_ptr + offs_m[:, None] * stride_output_m + offs_cn[None, :] * stride_output_Cn
    )
    tl.store(output_ptrs, out, mask=mask_m[:, None] & mask_cn[None, :])


@triton.autotune(
    configs=expand_combine_config(),
    key=["M", "C"],
    reset_to_zero=["grad_H_post_ptr", "grad_H_res_ptr", "grad_bias_ptr"],
)
@triton.jit
def _mhc_expand_combine_with_bias_bwd(
    grad_output_ptr,  # (M, C, n)
    f_ptr,  # (M, C)
    bias_ptr,  # (C,)
    H_post_ptr,  # (M, n)
    x_ptr,  # (M, C, n)
    H_res_ptr,  # (M, n, n)
    grad_H_post_ptr,  # (M, n)
    grad_f_ptr,  # (M, C)
    grad_bias_ptr,  # (C,)
    grad_H_res_ptr,  # (M, n, n)
    grad_x_ptr,  # (M, C, n)
    M,
    C,
    n: tl.constexpr,
    stride_grad_output_m,
    stride_grad_output_Cn,
    stride_fm,
    stride_fc,
    stride_bias,
    stride_xm,
    stride_xCn,
    stride_grad_fm,
    stride_grad_fc,
    stride_grad_bias,
    stride_grad_xm,
    stride_grad_xCn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    precision: tl.constexpr,
):
    """
    Each block
    It reads
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of f, which is the output of the attention / FFN module
    - (BLOCK_SIZE_M, n) of H_post, which is applied for the transformation of the attention / FFN output
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of x, which is the skip connection's input
    - (BLOCK_SIZE_M, n*n) of H_res, which is applied for the transformation of the skip connection
    and writes
    - (BLOCK_SIZE_M, n) of grad_H_post
    - (BLOCK_SIZE_M, BLOCK_SIZE_C) of grad_f
    - (BLOCK_SIZE_M, n, n) of grad_H_res
    - (BLOCK_SIZE_M, BLOCK_SIZE_C, n) of grad_x

    Forward:
        out = f @ H_post + x @ H_res
    Backward:
        GEMM:
        grad_H_post = f.T @ grad_output: (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
        grad_H_res = x.T @ grad_output: (BLOCK_SIZE_M, n, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, n, n)
        Not GEMM:
        grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, 1) = (BLOCK_SIZE_M, BLOCK_SIZE_C, 1)
        grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    """

    pid_m = tl.program_id(1)
    pid_c = tl.program_id(0)

    tl.static_assert(n == 4)
    tl.assume(M > 0)
    tl.assume(C > 0)
    tl.assume(n == 4)
    tl.assume(stride_fm > 0 and stride_fc == 1)
    tl.assume(stride_bias == 1)
    tl.assume(stride_xm > 0 and stride_xCn == 1)
    tl.assume(stride_grad_output_m > 0 and stride_grad_output_Cn == 1)
    tl.assume(stride_grad_fm > 0 and stride_grad_fc == 1)
    tl.assume(stride_grad_bias == 1)
    tl.assume(stride_grad_xm > 0 and stride_grad_xCn == 1)

    tl.assume(BLOCK_SIZE_C % 32 == 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_cn = pid_c * BLOCK_SIZE_C * n + tl.arange(0, BLOCK_SIZE_C * n)
    mask_m = offs_m < M
    mask_c = offs_c < C
    mask_cn = offs_cn < C * n

    f_ptrs = f_ptr + offs_m[:, None] * stride_fm + offs_c[None, :] * stride_fc
    f = tl.load(f_ptrs, mask=mask_m[:, None] & mask_c[None, :], other=0.0)

    bias = tl.load(bias_ptr + offs_c * stride_bias, mask=mask_c, other=0.0)  # (BLOCK_SIZE_C,)

    H_post_offs = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    H_post = tl.load(H_post_ptr + H_post_offs, mask=H_post_offs < M * n, other=0.0)
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, n))  # (BLOCK_SIZE_M, n)

    H_res_offs = pid_m * BLOCK_SIZE_M * n * n + tl.arange(0, BLOCK_SIZE_M * n * n)
    H_res = tl.load(
        H_res_ptr + H_res_offs, mask=H_res_offs < M * n * n, other=0.0
    )  # (BLOCK_SIZE_M, n, n)
    H_res = tl.reshape(H_res, (BLOCK_SIZE_M, n, n))  # (BLOCK_SIZE_M, n, n)

    grad_out_ptrs = (
        grad_output_ptr
        + offs_m[:, None] * stride_grad_output_m
        + offs_cn[None, :] * stride_grad_output_Cn
    )
    grad_out = tl.load(
        grad_out_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C * n)
    grad_out = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, n)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_post =  f.T @ grad_output # (BLOCK_SIZE_M, 1, BLOCK_SIZE_C) @ (BLOCK_SIZE_M, BLOCK_SIZE_C, n) = (BLOCK_SIZE_M, 1, n)
    # grad_H_post from (f+bias); unrolled
    f_bias = f + bias[None, :]
    go_r = tl.reshape(grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    go01, go23 = tl.split(go_r)
    go0, go1 = tl.split(go01)
    go2, go3 = tl.split(go23)
    hp0 = tl.sum((f_bias * go0).to(tl.float32), axis=1)
    hp1 = tl.sum((f_bias * go1).to(tl.float32), axis=1)
    hp2 = tl.sum((f_bias * go2).to(tl.float32), axis=1)
    hp3 = tl.sum((f_bias * go3).to(tl.float32), axis=1)
    grad_H_post = tl.reshape(tl.join(tl.join(hp0, hp1), tl.join(hp2, hp3)), (BLOCK_SIZE_M * n,))
    offs_grad_H_post = pid_m * BLOCK_SIZE_M * n + tl.arange(0, BLOCK_SIZE_M * n)
    tl.atomic_add(
        grad_H_post_ptr + offs_grad_H_post,
        grad_H_post,
        mask=offs_grad_H_post < M * n,
        sem="relaxed",
    )

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_cn[None, :] * stride_xCn
    x = tl.load(
        x_ptrs, mask=mask_m[:, None] & mask_cn[None, :], other=0.0
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)
    x = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C, n)

    # grad_H_res[i,j] = sum_C(x[:,:,i] * grad_out[:,:,j]) — unrolled (no tl.trans)
    x_r = tl.reshape(x, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2))
    x01, x23 = tl.split(x_r)
    xx0, xx1 = tl.split(x01)
    xx2, xx3 = tl.split(x23)

    # Plain xx/go products (same basis as H_post hp pack). Store rows with atomics
    # — joining (M,n) rows then flattening transposes into (M,n*n) layout.
    row0 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx0 * go0).to(tl.float32), axis=1),
                tl.sum((xx0 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx0 * go2).to(tl.float32), axis=1),
                tl.sum((xx0 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    row1 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx1 * go0).to(tl.float32), axis=1),
                tl.sum((xx1 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx1 * go2).to(tl.float32), axis=1),
                tl.sum((xx1 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    row2 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx2 * go0).to(tl.float32), axis=1),
                tl.sum((xx2 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx2 * go2).to(tl.float32), axis=1),
                tl.sum((xx2 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    row3 = tl.reshape(
        tl.join(
            tl.join(
                tl.sum((xx3 * go0).to(tl.float32), axis=1),
                tl.sum((xx3 * go1).to(tl.float32), axis=1),
            ),
            tl.join(
                tl.sum((xx3 * go2).to(tl.float32), axis=1),
                tl.sum((xx3 * go3).to(tl.float32), axis=1),
            ),
        ),
        (BLOCK_SIZE_M, n),
    )
    # xx0,xx1,xx2,xx3 = streams 0,2,1,3 — write to natural rows 0,2,1,3
    offs_m_h = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_mh = offs_m_h < M
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 0 * n + tl.arange(0, n)[None, :],
        row0,
        mask=mask_mh[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 2 * n + tl.arange(0, n)[None, :],
        row1,
        mask=mask_mh[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 1 * n + tl.arange(0, n)[None, :],
        row2,
        mask=mask_mh[:, None],
        sem="relaxed",
    )
    tl.atomic_add(
        grad_H_res_ptr + offs_m_h[:, None] * (n * n) + 3 * n + tl.arange(0, n)[None, :],
        row3,
        mask=mask_mh[:, None],
        sem="relaxed",
    )

    grad_out_reshape = tl.reshape(
        grad_out, (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2, 2)
    grad_out01, grad_out23 = tl.split(
        grad_out_reshape
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C, 2), (BLOCK_SIZE_M, BLOCK_SIZE_C, 2)
    grad_out0, grad_out1 = tl.split(
        grad_out01
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_out2, grad_out3 = tl.split(
        grad_out23
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_C), (BLOCK_SIZE_M, BLOCK_SIZE_C)

    # grad_f = grad_output @ H_post.T: (BLOCK_SIZE_M, 1, n) @ (BLOCK_SIZE_M, n, BLOCK_SIZE_C) = (BLOCK_SIZE_M, 1, BLOCK_SIZE_C)
    # Triton doesn't support dot prod with inner dimension < 16, so we need to hack this:
    #        = grad_out[:, :, 0] @ H_post.T[:, 0, :] (BLOCK_SIZE_M, BLOCK_SIZE_C, 1) @ (BLOCK_SIZE_M, 1, 1)
    #        + grad_out[:, :, 1] @ H_post.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_post.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_post.T[:, 3, :]
    # where H_post.T[:, i, :] = H_post[:, :, i]
    H_post = tl.reshape(H_post, (BLOCK_SIZE_M, 2, 2))
    H_post01, H_post23 = tl.split(H_post)  # (BLOCK_SIZE_M, 2), (BLOCK_SIZE_M, 2)
    H_post0, H_post1 = tl.split(H_post01)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)
    H_post2, H_post3 = tl.split(H_post23)  # (BLOCK_SIZE_M,), (BLOCK_SIZE_M,)

    grad_f_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C), dtype=tl.float32)
    # (BLOCK_SIZE_M, BLOCK_SIZE_C) * (BLOCK_SIZE_M, 1) -> (BLOCK_SIZE_M, BLOCK_SIZE_C)
    grad_f_acc = tl.fma(grad_out0, H_post0[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out1, H_post1[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out2, H_post2[:, None], grad_f_acc)
    grad_f_acc = tl.fma(grad_out3, H_post3[:, None], grad_f_acc)
    grad_f = grad_f_acc.to(f.dtype)

    grad_f_ptrs = grad_f_ptr + offs_m[:, None] * stride_grad_fm + offs_c[None, :] * stride_grad_fc
    tl.store(grad_f_ptrs, grad_f, mask=mask_m[:, None] & mask_c[None, :])

    grad_bias = tl.sum(grad_f_acc, axis=0)  # (BLOCK_SIZE_C,)
    grad_bias_ptrs = grad_bias_ptr + offs_c * stride_grad_bias
    tl.atomic_add(grad_bias_ptrs, grad_bias, mask=mask_c, sem="relaxed")

    # grad_x = grad_output @ H_res.T: (BLOCK_SIZE_M, BLOCK_SIZE_C, n) @ (BLOCK_SIZE_M, n, n) = (BLOCK_SIZE_M, n, BLOCK_SIZE_C)
    # The inner dim is n=4 which is too small for triton, so we will manually unroll the matmul
    # grad_x = grad_out[:, :, 0] @ H_res.T[:, 0, :]
    #        + grad_out[:, :, 1] @ H_res.T[:, 1, :]
    #        + grad_out[:, :, 2] @ H_res.T[:, 2, :]
    #        + grad_out[:, :, 3] @ H_res.T[:, 3, :]
    # where H_res.T[:, i, :] = H_res[:, :, i]
    # Due to broadcasting, it's equivalent to multiplying each H_res[:, i, :].T with grad_out[:, i, :]

    H_res_reshape = tl.reshape(H_res, (BLOCK_SIZE_M, n, 2, 2))  # (BLOCK_SIZE_M, n, 2, 2)
    H_res01, H_res23 = tl.split(H_res_reshape)  # (BLOCK_SIZE_M, n, 2), (BLOCK_SIZE_M, n, 2)
    H_res0, H_res1 = tl.split(H_res01)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)
    H_res2, H_res3 = tl.split(H_res23)  # (BLOCK_SIZE_M, n), (BLOCK_SIZE_M, n)

    grad_x_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_C, n), dtype=tl.float32)
    grad_x_acc = tl.fma(grad_out0[:, :, None], H_res0[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out1[:, :, None], H_res1[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out2[:, :, None], H_res2[:, None, :], grad_x_acc)
    grad_x_acc = tl.fma(grad_out3[:, :, None], H_res3[:, None, :], grad_x_acc)

    grad_x = grad_x_acc.to(x.dtype)
    grad_x = tl.reshape(grad_x, (BLOCK_SIZE_M, BLOCK_SIZE_C * n))  # (BLOCK_SIZE_M, BLOCK_SIZE_C*n)

    grad_x_ptrs = grad_x_ptr + offs_m[:, None] * stride_grad_xm + offs_cn[None, :] * stride_grad_xCn
    tl.store(grad_x_ptrs, grad_x, mask=mask_m[:, None] & mask_cn[None, :])
