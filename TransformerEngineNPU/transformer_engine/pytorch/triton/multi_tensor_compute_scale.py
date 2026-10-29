# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fused multi-tensor FP8 scale computation on NPU."""

import triton
import triton.language as tl


@triton.jit
def _multi_tensor_compute_scale_and_scale_inv_kernel(
    metadata,
    max_fp8,
    epsilon,
    NUM_TENSORS: tl.constexpr,
    FORCE_POW_2_SCALES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    tensor_idx = tl.load(metadata + 4 * NUM_TENSORS + 2 * block)
    block_idx = tl.load(metadata + 4 * NUM_TENSORS + 2 * block + 1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    amax_ptr = tl.load(metadata + tensor_idx).to(tl.pointer_type(tl.float32))
    scale_ptr = tl.load(metadata + NUM_TENSORS + tensor_idx).to(tl.pointer_type(tl.float32))
    scale_inv_ptr = tl.load(metadata + 2 * NUM_TENSORS + tensor_idx).to(tl.pointer_type(tl.float32))
    numel = tl.load(metadata + 3 * NUM_TENSORS + tensor_idx)
    mask = offsets < numel

    amax = tl.load(amax_ptr + offsets, mask=mask, other=0.0)
    clamped_amax = tl.where(amax < epsilon, epsilon, amax)
    scale = max_fp8 / clamped_amax
    fp32_max = tl.full((), 3.40282347e38, dtype=tl.float32)
    scale = tl.where((scale > fp32_max) | (scale < -fp32_max), fp32_max, scale)

    if FORCE_POW_2_SCALES:
        scale_bits = scale.to(tl.int32, bitcast=True)
        scale = (scale_bits & 0x7F800000).to(tl.float32, bitcast=True)
        scale = tl.where(amax > fp32_max, 0.0, scale)
        scale = tl.where((amax <= fp32_max) | (amax > fp32_max), scale, 0.5)

    scale = tl.where(amax == 0.0, 1.0, scale)
    tl.store(scale_ptr + offsets, scale, mask=mask)
    tl.store(scale_inv_ptr + offsets, 1.0 / scale, mask=mask)
