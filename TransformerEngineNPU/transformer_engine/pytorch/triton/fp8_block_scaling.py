# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Triton partial cast for 2D block-scaled FP8 on NPU."""

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": block_rows, "multibuffer": multibuffer})
        for block_rows in (8, 16, 32)
        for multibuffer in (False, True)
    ],
    key=["input_numel", "width"],
)
@triton.jit
def _fp8_block_scaling_partial_cast_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    height,
    width,
    input_numel,
    start_offset,
    scale_stride_0,
    scale_stride_1,
    tile_row_start,
    block_cols,
    total_tasks,
    BLOCK_LEN: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    ALL_BLOCKS_NO_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    local_rows = tl.arange(0, BLOCK_ROWS)
    local_cols = tl.arange(0, BLOCK_LEN)
    end_offset = start_offset + input_numel

    for task_id in range(pid, total_tasks, num_programs):
        tile_row = tile_row_start + task_id // block_cols
        tile_col = task_id % block_cols
        scale = tl.load(scale_ptr + tile_row * scale_stride_0 + tile_col * scale_stride_1)
        block_row_start = tile_row * BLOCK_LEN
        block_col_start = tile_col * BLOCK_LEN
        cols = block_col_start + local_cols
        if ALL_BLOCKS_NO_MASK:
            for row_start in range(0, BLOCK_LEN, BLOCK_ROWS):
                rows = block_row_start + row_start + local_rows
                global_offsets = rows[:, None] * width + cols[None, :]
                local_offsets = global_offsets - start_offset
                values = tl.load(input_ptr + local_offsets)
                output = (values * scale).to(FP8_DTYPE)
                tl.store(output_ptr + local_offsets, output)
        else:
            block_row_end = block_row_start + BLOCK_LEN
            block_col_end = block_col_start + BLOCK_LEN
            is_interior_block = (block_row_end <= height) and (block_col_end <= width)
            if is_interior_block:
                for row_start in range(0, BLOCK_LEN, BLOCK_ROWS):
                    rows = block_row_start + row_start + local_rows
                    global_offsets = rows[:, None] * width + cols[None, :]
                    local_offsets = global_offsets - start_offset
                    mask = (global_offsets >= start_offset) & (global_offsets < end_offset)
                    values = tl.load(input_ptr + local_offsets, mask=mask, other=0.0)
                    output = (values * scale).to(FP8_DTYPE)
                    tl.store(output_ptr + local_offsets, output, mask=mask)
            else:
                col_mask = cols < width
                for row_start in range(0, BLOCK_LEN, BLOCK_ROWS):
                    rows = block_row_start + row_start + local_rows
                    global_offsets = rows[:, None] * width + cols[None, :]
                    local_offsets = global_offsets - start_offset
                    mask = (
                        (rows[:, None] < height)
                        & col_mask[None, :]
                        & (global_offsets >= start_offset)
                        & (global_offsets < end_offset)
                    )
                    values = tl.load(input_ptr + local_offsets, mask=mask, other=0.0)
                    output = (values * scale).to(FP8_DTYPE)
                    tl.store(output_ptr + local_offsets, output, mask=mask)


def fp8_block_scaling_partial_cast(
    inp: torch.Tensor,
    out: torch.Tensor,
    scale: torch.Tensor,
    h: int,
    w: int,
    start_offset: int,
    block_len: int,
    fp8_dtype: torch.dtype,
) -> None:
    """Scale and cast a contiguous flattened input shard into FP8 storage."""
    if block_len != 128:
        raise ValueError(f"Only block_len=128 is supported, got {block_len}")
    if not inp.is_contiguous() or not out.is_contiguous():
        raise ValueError("input and output must be contiguous")
    if inp.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input must be float32 or bfloat16, got {inp.dtype}")
    if out.element_size() != 1:
        raise ValueError(f"output must use one byte per element, got {out.dtype}")
    if scale.ndim != 2 or scale.dtype != torch.float32:
        raise ValueError("scale must be a 2D float32 tensor")
    if out.numel() != inp.numel():
        raise ValueError(f"input and output sizes must match, got {inp.numel()} and {out.numel()}")
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be positive, got {h} and {w}")
    if start_offset < 0 or start_offset + inp.numel() > h * w:
        raise ValueError("input shard must be within the flattened model weight")

    if fp8_dtype == torch.float8_e4m3fn:
        triton_fp8_dtype = tl.float8e4nv
    elif fp8_dtype == torch.float8_e5m2:
        triton_fp8_dtype = tl.float8e5
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")

    if inp.numel() == 0:
        return

    height_aligned = h % block_len == 0
    width_aligned = w % block_len == 0
    start_offset_aligned = start_offset == 0
    full_tensor = inp.numel() == h * w

    all_blocks_no_mask = height_aligned and width_aligned and start_offset_aligned and full_tensor

    block_cols = triton.cdiv(w, block_len)
    tile_row_size = block_len * w
    tile_row_start = start_offset // tile_row_size
    tile_row_end = triton.cdiv(start_offset + inp.numel(), tile_row_size)
    total_tasks = (tile_row_end - tile_row_start) * block_cols
    num_vector_cores = driver.active.utils.get_device_properties(inp.device.index)["num_vectorcore"]
    grid = (min(total_tasks, num_vector_cores),)
    fp8_out = triton.reinterpret(out, triton_fp8_dtype)
    _fp8_block_scaling_partial_cast_kernel[grid](
        inp,
        fp8_out,
        scale,
        h,
        w,
        inp.numel(),
        start_offset,
        scale.stride(0),
        scale.stride(1),
        tile_row_start,
        block_cols,
        total_tasks,
        BLOCK_LEN=block_len,
        FP8_DTYPE=triton_fp8_dtype,
        ALL_BLOCKS_NO_MASK=all_blocks_no_mask,
    )
