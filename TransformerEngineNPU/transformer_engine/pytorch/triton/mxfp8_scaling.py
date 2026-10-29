# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Triton partial cast for MXFP8 quantization on NPU."""

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _mxfp8_scaling_partial_cast_kernel(
    input_ptr,
    output_rowwise_ptr,
    output_colwise_ptr,
    scale_inv_rowwise_ptr,
    scale_inv_colwise_ptr,
    rows,
    cols,
    input_numel,
    start_offset,
    scale_inv_rowwise_stride_0,
    scale_inv_rowwise_stride_1,
    scale_inv_colwise_stride_0,
    scale_inv_colwise_stride_1,
    tile_row_start,
    tile_cols,
    total_tasks,
    BLOCK_LEN: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
    COLS_PER_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    local_rows = tl.arange(0, BLOCK_ROWS)[:, None]
    local_cols = tl.arange(0, COLS_PER_TILE)[None, :]
    local_scale_cols = tl.arange(0, COLS_PER_TILE // BLOCK_LEN)[None, :]
    end_offset = start_offset + input_numel
    for task_id in range(pid, total_tasks, num_programs):
        tile_row = tile_row_start + task_id // tile_cols
        tile_col = task_id % tile_cols
        columns = tile_col * COLS_PER_TILE + local_cols
        scale_inv_colwise = tl.load(
            scale_inv_colwise_ptr
            + tile_row * scale_inv_colwise_stride_0
            + columns * scale_inv_colwise_stride_1,
            mask=columns < cols,
            other=127,
        )
        encode_scale_colwise = ((254 - scale_inv_colwise.to(tl.int32)) << 23).to(
            tl.float32, bitcast=True
        )

        rowwise_scale_cols = tile_col * (COLS_PER_TILE // BLOCK_LEN) + local_scale_cols
        for row_start in range(0, ROWS_PER_TILE, BLOCK_ROWS):
            row_offsets = tile_row * ROWS_PER_TILE + row_start + local_rows
            scale_inv_rowwise = tl.load(
                scale_inv_rowwise_ptr
                + row_offsets * scale_inv_rowwise_stride_0
                + rowwise_scale_cols * scale_inv_rowwise_stride_1,
                mask=(row_offsets < rows) & (rowwise_scale_cols < cols // BLOCK_LEN),
                other=127,
            )
            encode_scale_rowwise = ((254 - scale_inv_rowwise.to(tl.int32)) << 23).to(
                tl.float32, bitcast=True
            )
            encode_scale_rowwise = tl.reshape(
                tl.broadcast_to(
                    encode_scale_rowwise[:, :, None],
                    (BLOCK_ROWS, COLS_PER_TILE // BLOCK_LEN, BLOCK_LEN),
                ),
                (BLOCK_ROWS, COLS_PER_TILE),
            )

            global_offsets = row_offsets * cols + columns
            mask = (
                (row_offsets < rows)
                & (columns < cols)
                & (global_offsets >= start_offset)
                & (global_offsets < end_offset)
            )
            local_offsets = global_offsets - start_offset
            values = tl.load(
                input_ptr + local_offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            scaled_rowwise = tl.minimum(
                tl.maximum(values * encode_scale_rowwise, -448.0),
                448.0,
            )
            scaled_colwise = tl.minimum(
                tl.maximum(values * encode_scale_colwise, -448.0),
                448.0,
            )
            output_rowwise = scaled_rowwise.to(tl.float8e4nv)
            output_colwise = scaled_colwise.to(tl.float8e4nv)
            tl.store(output_rowwise_ptr + local_offsets, output_rowwise, mask=mask)
            tl.store(output_colwise_ptr + local_offsets, output_colwise, mask=mask)


def _validate_common(
    inp: torch.Tensor,
    rows: int,
    cols: int,
    start_offset: int,
) -> None:
    if not inp.is_contiguous() or inp.ndim != 1:
        raise ValueError("input must be a contiguous 1D tensor")
    if inp.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input must be float32 or bfloat16, got {inp.dtype}")
    if rows <= 0 or cols <= 0 or rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(f"rows and cols must be positive multiples of 32, got {rows} and {cols}")
    if start_offset < 0 or start_offset + inp.numel() > rows * cols:
        raise ValueError("input shard must be within the flattened model weight")


def mxfp8_scaling_partial_cast(
    inp: torch.Tensor,
    output_rowwise: torch.Tensor,
    output_colwise: torch.Tensor,
    scale_inv_rowwise: torch.Tensor,
    scale_inv_colwise: torch.Tensor,
    rows: int,
    cols: int,
    start_offset: int,
) -> None:
    """Cast a flattened input shard into rowwise and columnwise MXFP8 storage."""
    _validate_common(inp, rows, cols, start_offset)
    if not output_rowwise.is_contiguous() or not output_colwise.is_contiguous():
        raise ValueError("output tensors must be contiguous")
    if output_rowwise.numel() != inp.numel() or output_colwise.numel() != inp.numel():
        raise ValueError("input and output sizes must match")
    if output_rowwise.element_size() != 1 or output_colwise.element_size() != 1:
        raise ValueError("output tensors must use one byte per element")
    if not scale_inv_rowwise.is_contiguous() or not scale_inv_colwise.is_contiguous():
        raise ValueError("scale_inv tensors must be contiguous")
    if scale_inv_rowwise.dtype != torch.uint8 or scale_inv_colwise.dtype != torch.uint8:
        raise ValueError("scale_inv tensors must be uint8")
    expected_rowwise_shape = (rows, cols // 32)
    expected_colwise_shape = (rows // 32, cols)
    if tuple(scale_inv_rowwise.shape) != expected_rowwise_shape:
        raise ValueError(
            f"Unexpected rowwise scale shape: expected {expected_rowwise_shape}, "
            f"got {tuple(scale_inv_rowwise.shape)}"
        )
    if tuple(scale_inv_colwise.shape) != expected_colwise_shape:
        raise ValueError(
            f"Unexpected columnwise scale shape: expected {expected_colwise_shape}, "
            f"got {tuple(scale_inv_colwise.shape)}"
        )

    if inp.numel() == 0:
        return

    rows_per_tile = 32
    cols_per_tile = 128
    tile_cols = triton.cdiv(cols, cols_per_tile)
    tile_row_size = rows_per_tile * cols
    tile_row_start = start_offset // tile_row_size
    tile_row_end = triton.cdiv(start_offset + inp.numel(), tile_row_size)
    total_tasks = (tile_row_end - tile_row_start) * tile_cols
    num_vector_cores = driver.active.utils.get_device_properties(inp.device.index)["num_vectorcore"]
    grid = (min(total_tasks, num_vector_cores),)
    output_rowwise_fp8 = triton.reinterpret(output_rowwise, tl.float8e4nv)
    output_colwise_fp8 = triton.reinterpret(output_colwise, tl.float8e4nv)
    _mxfp8_scaling_partial_cast_kernel[grid](
        inp,
        output_rowwise_fp8,
        output_colwise_fp8,
        scale_inv_rowwise,
        scale_inv_colwise,
        rows,
        cols,
        inp.numel(),
        start_offset,
        scale_inv_rowwise.stride(0),
        scale_inv_rowwise.stride(1),
        scale_inv_colwise.stride(0),
        scale_inv_colwise.stride(1),
        tile_row_start,
        tile_cols,
        total_tasks,
        BLOCK_LEN=32,
        BLOCK_ROWS=8,
        ROWS_PER_TILE=rows_per_tile,
        COLS_PER_TILE=cols_per_tile,
    )
