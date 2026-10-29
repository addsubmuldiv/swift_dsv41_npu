# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient Parallel Cross Entropy kernels written with Triton (Ascend NPU)."""

import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    m_d_X_y_ptr,
    m_d_X_y_stride,
    rank,
    n_cols,
    ignore_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute per-row m/d/X_y for online softmax on this TP rank."""
    program_id = tl.program_id(0).to(tl.int64)

    X_ptr += program_id * X_stride
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    vocab_start_idx = rank * n_cols
    vocab_end_idx = (rank + 1) * n_cols
    # Ascend Triton rejects chained `and`; nest predicates, default X_y=-inf.
    X_y = float("-inf")
    if y != ignore_idx:
        if y >= vocab_start_idx:
            if y < vocab_end_idx:
                X_y = tl.load(X_ptr + y - vocab_start_idx).to(tl.float32)

    m_d_X_y_ptr += program_id * m_d_X_y_stride * 3

    m = float("-inf")
    d = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(
            tl.float32
        )
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    tl.store(m_d_X_y_ptr, m)
    tl.store(m_d_X_y_ptr + m_d_X_y_stride, d)
    tl.store(m_d_X_y_ptr + (2 * m_d_X_y_stride), X_y)


@triton.jit
def cross_entropy_kernel(
    X_ptr,
    X_stride,
    Grad_ptr,
    Grad_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    scaled_ptr,
    scaled_stride,
    mld_ptr,
    mld_stride,
    m_d_X_y_ptr,
    m_d_X_y_stride,
    rank,
    world_size,
    # all_gather: == world_size; all_reduce: == 1 (stats already merged on host)
    num_stat_shards,
    ignore_idx,
    n_cols,
    n_rows,
    n_non_ignore,
    reduce_loss: tl.constexpr,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute CE loss; write softmax-minus-one gradients into Grad (not X)."""
    program_id = tl.program_id(0).to(tl.int64)
    n_non_ignore = tl.load(n_non_ignore)

    X_ptr += program_id * X_stride
    Grad_ptr += program_id * Grad_stride
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(Grad_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    scaled_ptr += program_id * scaled_stride
    mld_ptr += program_id * mld_stride
    m_d_X_y_ptr += program_id * 3 * m_d_X_y_stride

    m = tl.load(m_d_X_y_ptr)
    d = tl.load(m_d_X_y_ptr + m_d_X_y_stride)
    ori_X_y = tl.load(m_d_X_y_ptr + (2 * m_d_X_y_stride))

    for i in range(1, num_stat_shards):
        offset = i * 3 * n_rows * m_d_X_y_stride
        access_ptr = m_d_X_y_ptr + offset
        m_new = tl.load(access_ptr)
        d_new = tl.load(access_ptr + m_d_X_y_stride)
        X_y_new = tl.load(access_ptr + (2 * m_d_X_y_stride))

        d = d * tl.exp(m - tl.maximum(m, m_new)) + d_new * tl.exp(m_new - tl.maximum(m, m_new))
        m = tl.maximum(m, m_new)
        ori_X_y = tl.maximum(ori_X_y, X_y_new)

    scaled_x_sum = 0.0
    eps = label_smoothing / (n_cols * world_size)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"))
        grad_dtype = X_block.dtype
        X_block = X_block.to(tl.float32)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        if reduce_loss:
            G_block = (tl.exp(X_block - m) / d - eps) / n_non_ignore
        else:
            G_block = tl.exp(X_block - m) / d - eps
        tl.store(Grad_ptr + X_offsets, G_block.to(grad_dtype), mask=X_offsets < n_cols)

    tl.debug_barrier()

    # Hard CE only; host applies label smoothing after all_reduce(scaled).
    loss = -(ori_X_y - m - tl.log(d))
    tl.store(loss_ptr, loss)
    tl.store(scaled_ptr, scaled_x_sum)
    tl.store(mld_ptr, m + tl.log(d))

    vocab_start_idx = rank * n_cols
    vocab_end_idx = (rank + 1) * n_cols
    if y >= vocab_start_idx:
        if y < vocab_end_idx:
            G_y = tl.load(Grad_ptr + y - vocab_start_idx)
            if reduce_loss:
                G_y += -(1 - label_smoothing) / n_non_ignore
            else:
                G_y += -(1 - label_smoothing)
            tl.store(Grad_ptr + y - vocab_start_idx, G_y)


@triton.jit
def cross_entropy_fwd_single_kernel(
    X_ptr,
    X_stride,
    Grad_ptr,
    Grad_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    scaled_ptr,
    scaled_stride,
    mld_ptr,
    mld_stride,
    ignore_idx,
    n_cols,
    n_non_ignore,
    reduce_loss: tl.constexpr,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused single-device CE: online softmax + loss/grad; grads go to Grad."""
    program_id = tl.program_id(0).to(tl.int64)
    n_non_ignore = tl.load(n_non_ignore)

    X_ptr += program_id * X_stride
    Grad_ptr += program_id * Grad_stride
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(Grad_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    # Pass 1: m / d / X_y (read-only on logits).
    X_y = tl.load(X_ptr + y).to(tl.float32)
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")).to(
            tl.float32
        )
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # Pass 2: grads into Grad_ptr + local scaled_x_sum.
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"))
        grad_dtype = X_block.dtype
        X_block = X_block.to(tl.float32)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        if reduce_loss:
            G_block = (tl.exp(X_block - m) / d - eps) / n_non_ignore
        else:
            G_block = tl.exp(X_block - m) / d - eps
        tl.store(Grad_ptr + X_offsets, G_block.to(grad_dtype), mask=X_offsets < n_cols)

    tl.debug_barrier()

    loss = -(X_y - m - tl.log(d))
    loss_ptr += program_id * loss_stride
    scaled_ptr += program_id * scaled_stride
    mld_ptr += program_id * mld_stride
    tl.store(loss_ptr, loss)
    tl.store(scaled_ptr, scaled_x_sum)
    tl.store(mld_ptr, m + tl.log(d))

    G_yt = tl.load(Grad_ptr + y)
    if reduce_loss:
        G_yt += -(1 - label_smoothing) / n_non_ignore
    else:
        G_yt += -(1 - label_smoothing)
    tl.store(Grad_ptr + y, G_yt)


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    grad_output_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place multiply X rows by grad_output (scalar or per-row)."""
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    grad_output_ptr += program_id * grad_output_stride
    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)
