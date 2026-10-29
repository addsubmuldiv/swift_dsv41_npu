# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Triton kernel for MXFP8 dequantization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage


@triton.jit
def _decode_e8m0_scale(scale_uint8):
    """Decode E8M0 format scale to float32 in-kernel."""
    exponent = scale_uint8.to(tl.float32) - 127.0
    return tl.math.exp2(exponent)


@triton.jit
def _dequantize_k_blocks_kernel(
    q_data_ptr,
    scale_inv_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    K_blocks: tl.constexpr,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
    stride_om,
    stride_ok,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K_BLOCKS: tl.constexpr,
    BLOCK_LEN: tl.constexpr,
    IS_UINT8: tl.constexpr,
):
    """K-axis block dequantization with optional in-kernel E8M0 decoding."""
    pid_m = tl.program_id(0)
    pid_k_block = tl.program_id(1)

    m_start = pid_m * BLOCK_SIZE_M
    m_indices = m_start + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_indices < M

    k_block_start = pid_k_block * BLOCK_SIZE_K_BLOCKS

    for k_block_offset in range(BLOCK_SIZE_K_BLOCKS):
        k_block_id = k_block_start + k_block_offset
        block_valid = k_block_id < K_blocks

        k_start = k_block_id * BLOCK_LEN
        k_end = tl.minimum(k_start + BLOCK_LEN, K)

        scale_ptrs = scale_inv_ptr + m_indices * stride_sm + k_block_id * stride_sk

        if IS_UINT8:
            # Load uint8 scales, use 127 as default (represents scale=1.0 in E8M0)
            scale_raw = tl.load(scale_ptrs, mask=m_mask & block_valid, other=127.0)
            scale_decoded = _decode_e8m0_scale(scale_raw)
        else:
            # Load float32 scales directly
            scale_decoded = tl.load(scale_ptrs, mask=m_mask & block_valid, other=1.0)

        k_indices = k_start + tl.arange(0, BLOCK_LEN)
        k_mask = (k_indices < k_end) & block_valid

        q_ptrs = q_data_ptr + m_indices[:, None] * stride_qm + k_indices[None, :] * stride_qk
        fp8_vals = tl.load(
            q_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        dequant_vals = fp8_vals.to(tl.float32) * scale_decoded[:, None]

        out_ptrs = out_ptr + m_indices[:, None] * stride_om + k_indices[None, :] * stride_ok
        tl.store(
            out_ptrs,
            dequant_vals,
            mask=m_mask[:, None] & k_mask[None, :],
        )


@triton.jit
def _dequantize_m_blocks_kernel(
    q_data_ptr,
    scale_inv_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    M_blocks: tl.constexpr,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
    stride_om,
    stride_ok,
    BLOCK_SIZE_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_LEN: tl.constexpr,
    IS_UINT8: tl.constexpr,
):
    """M-axis block dequantization with optional in-kernel E8M0 decoding."""
    pid_m_block = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_start = pid_k * BLOCK_SIZE_K
    k_indices = k_start + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_indices < K

    m_block_start = pid_m_block * BLOCK_SIZE_M_BLOCKS

    for m_block_offset in range(BLOCK_SIZE_M_BLOCKS):
        m_block_id = m_block_start + m_block_offset
        block_valid = m_block_id < M_blocks

        m_start = m_block_id * BLOCK_LEN
        m_end = tl.minimum(m_start + BLOCK_LEN, M)

        scale_ptrs = scale_inv_ptr + m_block_id * stride_sm + k_indices * stride_sk

        if IS_UINT8:
            scale_raw = tl.load(scale_ptrs, mask=k_mask & block_valid, other=127.0)
            scale_decoded = _decode_e8m0_scale(scale_raw)
        else:
            scale_decoded = tl.load(scale_ptrs, mask=k_mask & block_valid, other=1.0)

        m_indices = m_start + tl.arange(0, BLOCK_LEN)
        m_mask = (m_indices < m_end) & block_valid

        q_ptrs = q_data_ptr + m_indices[:, None] * stride_qm + k_indices[None, :] * stride_qk
        fp8_vals = tl.load(
            q_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        dequant_vals = fp8_vals.to(tl.float32) * scale_decoded[None, :]

        out_ptrs = out_ptr + m_indices[:, None] * stride_om + k_indices[None, :] * stride_ok
        tl.store(
            out_ptrs,
            dequant_vals,
            mask=m_mask[:, None] & k_mask[None, :],
        )


def _decode_scale_inv(scale_inv: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scale tensor to float32 values (matches storage implementation)."""
    if scale_inv.dtype != torch.uint8:
        try:
            return scale_inv.to(torch.float32)
        except (TypeError, RuntimeError):
            pass

    exponent = scale_inv.to(torch.float32)
    return torch.pow(2.0, exponent - 127.0)


def _reshape_scale_inv(scale_inv: torch.Tensor, *, keep_first_dim: bool = False) -> torch.Tensor:
    """Reshape decoded scale tensors into a 2D logical view."""
    if scale_inv.ndim == 0:
        return scale_inv.reshape(1, 1)
    if scale_inv.ndim == 1:
        return scale_inv.reshape(-1, 1)
    if scale_inv.ndim == 2:
        return scale_inv
    if not keep_first_dim:
        return scale_inv.reshape(-1, scale_inv.shape[-1])
    return scale_inv.reshape(scale_inv.shape[0], -1)


def _reshape_logical_matrix(
    scale_matrix: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Flatten and rebuild a logical 2D scale matrix when enough data exists."""
    required_numel = rows * cols
    if scale_matrix.numel() < required_numel:
        return scale_matrix
    return scale_matrix.reshape(-1)[:required_numel].reshape(rows, cols)


def _coerce_scale_matrix_shape(
    scale_matrix: torch.Tensor,
    *,
    logical_dim0: int,
    logical_dim1: int,
) -> torch.Tensor:
    """Coerce decoded scale tensors into the expected logical 2D layout."""
    if scale_matrix.shape[0] >= logical_dim0 and scale_matrix.shape[1] >= logical_dim1:
        return scale_matrix

    return _reshape_logical_matrix(scale_matrix, logical_dim0, logical_dim1)


def _coerce_rowwise_scale_shape(
    scale_matrix: torch.Tensor,
    *,
    m_blocks: int,
    k_dim: int,
) -> torch.Tensor:
    """Coerce rowwise scale tensors into [M_blocks, K]."""
    if scale_matrix.shape[0] == m_blocks and scale_matrix.shape[1] >= k_dim:
        return scale_matrix[:, :k_dim]

    if scale_matrix.shape[0] == k_dim and scale_matrix.shape[1] >= m_blocks:
        return scale_matrix[:k_dim, :m_blocks].transpose(-2, -1).contiguous()

    if scale_matrix.ndim == 2 and scale_matrix.shape[1] > 0:
        packed_dim = scale_matrix.shape[1]
        if scale_matrix.shape[0] == m_blocks * math.ceil(k_dim / packed_dim):
            return scale_matrix.reshape(m_blocks, -1)[:, :k_dim].contiguous()
        if scale_matrix.shape[0] == k_dim * math.ceil(m_blocks / packed_dim):
            return scale_matrix.reshape(k_dim, -1)[:, :m_blocks].transpose(-2, -1).contiguous()

    return _reshape_logical_matrix(scale_matrix, m_blocks, k_dim)


def _prepare_scale_tensor_k_blocks(
    scale_inv: torch.Tensor,
    m_dim: int,
    k_blocks: int,
) -> tuple[torch.Tensor, bool]:
    """Prepare scale tensor for K-axis block dequantization."""
    is_uint8 = scale_inv.dtype == torch.uint8

    if is_uint8:
        # Keep uint8 format for in-kernel decoding
        decoded_scale = scale_inv
    else:
        # Decode to float32
        decoded_scale = _decode_scale_inv(scale_inv)

    if decoded_scale.ndim >= 3:
        scale_matrix = decoded_scale.reshape(-1, decoded_scale.shape[-2] * decoded_scale.shape[-1])
    else:
        scale_matrix = _reshape_scale_inv(decoded_scale, keep_first_dim=True)

    scale_matrix = _coerce_scale_matrix_shape(
        scale_matrix,
        logical_dim0=m_dim,
        logical_dim1=k_blocks,
    )

    if scale_matrix.shape[0] < m_dim or scale_matrix.shape[1] < k_blocks:
        raise RuntimeError(
            f"MXFP8 columnwise scale tensor has insufficient logical shape for dequantize: "
            f"got {tuple(scale_matrix.shape)}, expected at least ({m_dim}, {k_blocks})"
        )

    valid_scale = scale_matrix[:m_dim, :k_blocks].contiguous()
    return valid_scale, is_uint8


def _prepare_scale_tensor_m_blocks(
    scale_inv: torch.Tensor,
    m_blocks: int,
    k_dim: int,
) -> tuple[torch.Tensor, bool]:
    """Prepare scale tensor for M-axis block dequantization.

    Returns:
        tuple of (prepared_scale_tensor, is_uint8)
    """
    is_uint8 = scale_inv.dtype == torch.uint8

    if is_uint8:
        # Keep uint8 format for in-kernel decoding
        decoded_scale = scale_inv
    else:
        # Decode to float32
        decoded_scale = _decode_scale_inv(scale_inv)

    if (
        decoded_scale.ndim >= 3
        and decoded_scale.shape[0] * decoded_scale.shape[-1] >= m_blocks
        and decoded_scale.shape[1] >= k_dim
    ):
        scale_matrix = decoded_scale.transpose(-2, -1).reshape(-1, decoded_scale.shape[1])
    else:
        scale_matrix = _reshape_scale_inv(
            decoded_scale,
            keep_first_dim=True,
        )

    scale_matrix = _coerce_rowwise_scale_shape(
        scale_matrix,
        m_blocks=m_blocks,
        k_dim=k_dim,
    )

    if scale_matrix.shape[0] < m_blocks or scale_matrix.shape[1] < k_dim:
        raise RuntimeError(
            f"MXFP8 rowwise scale tensor has insufficient logical shape for dequantize: "
            f"got {tuple(scale_matrix.shape)}, expected at least ({m_blocks}, {k_dim})"
        )

    valid_scale = scale_matrix[:m_blocks, :k_dim].contiguous()
    return valid_scale, is_uint8


def _scale_matches_axis_k_blocks(
    scale_inv: torch.Tensor,
    m_dim: int,
    k_dim: int,
) -> bool:
    """Return whether scale layout looks like one scale per 1x32 K block."""
    BLOCK_LEN = 32
    if scale_inv.ndim < 2:
        return False
    k_blocks = math.ceil(k_dim / BLOCK_LEN)
    if scale_inv.shape[0] == m_dim and math.prod(scale_inv.shape[1:]) >= k_blocks:
        return True
    # NPU kernels may preserve the leading tensor dimensions instead of
    # flattening them into one M axis: [*, ceil(k / (2 * block)), 2].
    if (
        scale_inv.ndim >= 3
        and math.prod(scale_inv.shape[:-1]) == m_dim
        and math.prod(scale_inv.shape[-2:]) >= k_blocks
    ):
        return True
    return False


def _scale_matches_axis_m_blocks(
    scale_inv: torch.Tensor,
    m_dim: int,
    k_dim: int,
) -> bool:
    """Return whether scale layout looks like one scale per 32x1 M block."""
    BLOCK_LEN = 32
    if scale_inv.ndim < 2:
        return False
    m_blocks = math.ceil(m_dim / BLOCK_LEN)
    if (
        scale_inv.ndim >= 3
        and scale_inv.shape[0] * scale_inv.shape[-1] >= m_blocks
        and scale_inv.shape[1] >= k_dim
    ):
        return True
    return (scale_inv.shape[0] == m_blocks and scale_inv.shape[1] >= k_dim) or (
        scale_inv.ndim == 2 and scale_inv.shape[0] == k_dim and scale_inv.shape[1] >= m_blocks
    )


def mxfp8_dequantize(tensor: "MXFP8TensorStorage", dtype: torch.dtype) -> torch.Tensor:
    """Dequantize MXFP8 tensor using Triton kernel."""
    # Determine which data/scale to use
    if tensor._rowwise_data is not None:
        q_data = tensor._rowwise_data
        scale_inv = tensor._rowwise_scale_inv
        use_rowwise = True
        orig_shape = tuple(tensor._rowwise_data.shape)
    else:
        if tensor._columnwise_data is None:
            raise RuntimeError("No data available for dequantization")
        q_data = tensor._columnwise_data
        scale_inv = tensor._columnwise_scale_inv
        use_rowwise = False
        orig_shape = tuple(tensor._columnwise_data.shape)

    # Decode quantized data if needed
    if q_data.dtype != tensor._fp8_dtype:
        q_data = q_data.contiguous().view(dtype=tensor._fp8_dtype)

    # Flatten to 2D
    q_data_2d = q_data.reshape(-1, q_data.shape[-1]) if q_data.ndim > 1 else q_data.reshape(1, -1)
    if not q_data_2d.is_contiguous():
        q_data_2d = q_data_2d.contiguous()

    M, K = q_data_2d.shape
    BLOCK_LEN = 32
    K_blocks = math.ceil(K / BLOCK_LEN)
    M_blocks = math.ceil(M / BLOCK_LEN)

    # Determine block axis based on scale layout
    if use_rowwise:
        is_k_blocks = _scale_matches_axis_k_blocks(scale_inv, m_dim=M, k_dim=K)
    else:
        is_k_blocks = not _scale_matches_axis_m_blocks(scale_inv, m_dim=M, k_dim=K)

    output = torch.empty((M, K), dtype=torch.float32, device=q_data.device)

    if is_k_blocks:
        # K-axis blocking (1 x 32 blocks)
        scale_2d, is_uint8 = _prepare_scale_tensor_k_blocks(scale_inv, M, K_blocks)

        BLOCK_SIZE_M = 256
        BLOCK_SIZE_K_BLOCKS = max(1, triton.cdiv(M * K_blocks, 64 * BLOCK_SIZE_M))

        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(K_blocks, BLOCK_SIZE_K_BLOCKS),
        )

        _dequantize_k_blocks_kernel[grid](
            q_data_2d,
            scale_2d,
            output,
            M=M,
            K=K,
            K_blocks=K_blocks,
            stride_qm=q_data_2d.stride(0),
            stride_qk=q_data_2d.stride(1),
            stride_sm=scale_2d.stride(0),
            stride_sk=scale_2d.stride(1),
            stride_om=output.stride(0),
            stride_ok=output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_K_BLOCKS=BLOCK_SIZE_K_BLOCKS,
            BLOCK_LEN=BLOCK_LEN,
            IS_UINT8=is_uint8,
        )
    else:
        # M-axis blocking (32 x 1 blocks)
        scale_2d, is_uint8 = _prepare_scale_tensor_m_blocks(scale_inv, M_blocks, K)

        BLOCK_SIZE_K = 256
        BLOCK_SIZE_M_BLOCKS = max(1, triton.cdiv(K * M_blocks, 64 * BLOCK_SIZE_K))

        grid = (
            triton.cdiv(M_blocks, BLOCK_SIZE_M_BLOCKS),
            triton.cdiv(K, BLOCK_SIZE_K),
        )

        _dequantize_m_blocks_kernel[grid](
            q_data_2d,
            scale_2d,
            output,
            M=M,
            K=K,
            M_blocks=M_blocks,
            stride_qm=q_data_2d.stride(0),
            stride_qk=q_data_2d.stride(1),
            stride_sm=scale_2d.stride(0),
            stride_sk=scale_2d.stride(1),
            stride_om=output.stride(0),
            stride_ok=output.stride(1),
            BLOCK_SIZE_M_BLOCKS=BLOCK_SIZE_M_BLOCKS,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_LEN=BLOCK_LEN,
            IS_UINT8=is_uint8,
        )

    # Convert to requested dtype
    output = output.to(dtype)

    # Reshape to original shape
    if len(orig_shape) == 0:
        return output.reshape([])

    return output.reshape(*orig_shape).contiguous()
