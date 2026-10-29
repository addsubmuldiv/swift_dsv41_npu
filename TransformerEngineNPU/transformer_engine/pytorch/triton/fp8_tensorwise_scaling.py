# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Triton explicit-scale tensorwise FP8 cast on NPU."""

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _fp8_tensorwise_partial_cast_kernel(
    input_ptr,
    output_address,
    scale_ptr,
    input_numel,
    total_tasks,
    FP8_DTYPE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_ptr = output_address.to(tl.pointer_type(FP8_DTYPE))
    scale = tl.load(scale_ptr).to(tl.float32)
    for task_id in range(pid, total_tasks, num_programs):
        indices = task_id * BLOCK_SIZE + offsets
        mask = indices < input_numel
        values = tl.load(input_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        scaled = tl.minimum(tl.maximum(values * scale, -FP8_MAX), FP8_MAX)
        tl.store(output_ptr + indices, scaled.to(FP8_DTYPE), mask=mask)


def fp8_tensorwise_partial_cast(
    inp: torch.Tensor,
    out: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> None:
    """Scale and cast a contiguous input shard directly into FP8 storage."""
    if not inp.is_contiguous() or not out.is_contiguous():
        raise ValueError("input and output must be contiguous")
    if inp.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input must be float32 or bfloat16, got {inp.dtype}")
    if inp.device != out.device or inp.device != scale.device:
        raise ValueError("input, output, and scale must be on the same device")
    if out.element_size() != 1:
        raise ValueError(f"output must use one byte per element, got {out.dtype}")
    if out.numel() != inp.numel():
        raise ValueError(f"input and output sizes must match, got {inp.numel()} and {out.numel()}")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("scale must be a single float32 value")
    if not scale.is_contiguous():
        raise ValueError("scale must be contiguous")

    if fp8_dtype == torch.float8_e4m3fn:
        triton_fp8_dtype = tl.float8e4nv
    elif fp8_dtype == torch.float8_e5m2:
        triton_fp8_dtype = tl.float8e5
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")

    if inp.numel() == 0:
        return

    block_size = 16384
    total_tasks = triton.cdiv(inp.numel(), block_size)
    num_vector_cores = driver.active.utils.get_device_properties(inp.device.index)["num_vectorcore"]
    grid = (min(total_tasks, num_vector_cores),)
    _fp8_tensorwise_partial_cast_kernel[grid](
        inp,
        # Keep the output as a scalar address so Level1 profiling does not inspect a
        # triton.reinterpret TensorWrapper that lacks the torch.Tensor shape API.
        out.data_ptr(),
        scale,
        inp.numel(),
        total_tasks,
        FP8_DTYPE=triton_fp8_dtype,
        FP8_MAX=torch.finfo(fp8_dtype).max,
        BLOCK_SIZE=block_size,
    )
