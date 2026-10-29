# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused Triton kernel for FP8 amax and scale update.

Replaces the Python loop over ``(amax_history, scale)`` pairs with one Triton
kernel launch over the original history and scale buffers.

The kernel uses per-quantizer parallelism: each program instance processes one
quantizer across a single pair, replacing the 2-D tile approach for better
utilisation on NPU.
"""

from typing import List

import torch
import triton
import triton.language as tl


_AMAX_ALGO_MAP = {"max": 0, "mean": 1, "most_recent": 2}


@triton.jit
def _amax_most_recent(base, stride_h, H: tl.constexpr, BLOCK_H: tl.constexpr):
    first_val = tl.load(base)
    if H == 1:
        tl.store(base, 0.0)
    else:
        arange = tl.arange(0, BLOCK_H)
        shifted_vals = tl.load(
            base + (arange + 1) * stride_h,
            mask=(arange + 1) < H,
            other=0.0,
        )
        write_vals = tl.where(arange == 0, 0.0, tl.where(arange == H - 1, first_val, shifted_vals))
        tl.store(base + arange * stride_h, write_vals, mask=arange < H)
    return first_val


@triton.jit
def _amax_max(base, stride_h, H: tl.constexpr, BLOCK_H: tl.constexpr):
    first_val = tl.load(base)
    if H == 1:
        tl.store(base, 0.0)
        return first_val

    arange = tl.arange(0, BLOCK_H)
    vals = tl.load(base + arange * stride_h, mask=arange < H, other=0.0)
    amax_val = tl.max(vals, axis=0)

    shifted = tl.load(base + (arange + 1) * stride_h, mask=(arange + 1) < H, other=0.0)
    write_vals = tl.where(arange == 0, 0.0, tl.where(arange == H - 1, first_val, shifted))
    tl.store(base + arange * stride_h, write_vals, mask=arange < H)

    return amax_val


@triton.jit
def _amax_mean(base, stride_h, H: tl.constexpr, BLOCK_H: tl.constexpr):
    first_val = tl.load(base)
    if H == 1:
        tl.store(base, 0.0)
        return first_val

    arange = tl.arange(0, BLOCK_H)
    vals = tl.load(base + arange * stride_h, mask=arange < H, other=0.0)
    total = tl.sum(vals, axis=0)

    shifted = tl.load(base + (arange + 1) * stride_h, mask=(arange + 1) < H, other=0.0)
    write_vals = tl.where(arange == 0, 0.0, tl.where(arange == H - 1, first_val, shifted))
    tl.store(base + arange * stride_h, write_vals, mask=arange < H)

    return total / H


@triton.jit
def _update_one_amax_history(
    base,
    scale_slot_ptr,
    fp8_max,
    stride_h,
    AMAX_ALGO: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MARGIN_FACTOR: tl.constexpr,
):
    if AMAX_ALGO == 0:
        amax_val = _amax_max(base, stride_h, H, BLOCK_H)
    elif AMAX_ALGO == 1:
        amax_val = _amax_mean(base, stride_h, H, BLOCK_H)
    else:
        amax_val = _amax_most_recent(base, stride_h, H, BLOCK_H)

    sf = fp8_max / amax_val / MARGIN_FACTOR
    current_scale = tl.load(scale_slot_ptr)
    fp32_max = tl.full((), 3.40282347e38, dtype=tl.float32)
    sf = tl.where(
        (amax_val > 0.0) & (amax_val <= fp32_max),
        tl.minimum(sf, fp32_max),
        current_scale,
    )
    tl.store(scale_slot_ptr, sf)


@triton.jit
def _single_update_kernel(
    history_ptr,
    scale_ptr,
    fp8_max,
    stride_h,
    stride_n,
    AMAX_ALGO: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MARGIN_FACTOR: tl.constexpr,
):
    quantizer = tl.program_id(0)
    _update_one_amax_history(
        history_ptr + quantizer * stride_n,
        scale_ptr + quantizer,
        fp8_max,
        stride_h,
        AMAX_ALGO,
        H,
        BLOCK_H,
        MARGIN_FACTOR,
    )


@triton.jit
def _single_update_vectorized_kernel(
    history_ptr,
    scale_ptr,
    fp8_max,
    stride_h,
    stride_n,
    num_quantizers,
    AMAX_ALGO: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MARGIN_FACTOR: tl.constexpr,
    QUANTIZERS_PER_BLOCK: tl.constexpr,
):
    base_quantizer = tl.program_id(0) * QUANTIZERS_PER_BLOCK

    # Process multiple quantizers in a loop within the same thread block
    for offset in range(QUANTIZERS_PER_BLOCK):
        quantizer = base_quantizer + offset
        if quantizer < num_quantizers:
            _update_one_amax_history(
                history_ptr + quantizer * stride_n,
                scale_ptr + quantizer,
                fp8_max,
                stride_h,
                AMAX_ALGO,
                H,
                BLOCK_H,
                MARGIN_FACTOR,
            )


@triton.jit
def _multi_update_kernel(
    metadata,
    fp8_max,
    AMAX_ALGO: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MARGIN_FACTOR: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
):
    quantizer = tl.program_id(0)
    pair = tl.program_id(1)
    history_ptr = tl.load(metadata + pair).to(tl.pointer_type(tl.float32))
    scale_ptr = tl.load(metadata + NUM_PAIRS + pair).to(tl.pointer_type(tl.float32))
    num_quantizers = tl.load(metadata + 2 * NUM_PAIRS + pair)
    if quantizer >= num_quantizers:
        return

    _update_one_amax_history(
        history_ptr + quantizer,
        scale_ptr + quantizer,
        fp8_max,
        num_quantizers,
        AMAX_ALGO,
        H,
        BLOCK_H,
        MARGIN_FACTOR,
    )


@triton.jit
def _multi_update_vectorized_kernel(
    metadata,
    fp8_max,
    max_quantizers,
    AMAX_ALGO: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MARGIN_FACTOR: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
    QUANTIZERS_PER_BLOCK: tl.constexpr,
):
    base_quantizer = tl.program_id(0) * QUANTIZERS_PER_BLOCK
    pair = tl.program_id(1)

    history_ptr = tl.load(metadata + pair).to(tl.pointer_type(tl.float32))
    scale_ptr = tl.load(metadata + NUM_PAIRS + pair).to(tl.pointer_type(tl.float32))
    num_quantizers = tl.load(metadata + 2 * NUM_PAIRS + pair)

    # Process QUANTIZERS_PER_BLOCK quantizers in this thread block
    for offset in range(QUANTIZERS_PER_BLOCK):
        quantizer = base_quantizer + offset
        if quantizer < num_quantizers:
            _update_one_amax_history(
                history_ptr + quantizer,
                scale_ptr + quantizer,
                fp8_max,
                num_quantizers,
                AMAX_ALGO,
                H,
                BLOCK_H,
                MARGIN_FACTOR,
            )


def fused_amax_and_scale_update(
    amax_histories: List[torch.Tensor],
    scales: List[torch.Tensor],
    fp8_max: float,
    recipe,
) -> None:
    """Update all FP8 amax histories and scales with one Triton launch."""
    if len(amax_histories) != len(scales):
        raise ValueError("amax_histories and scales must have the same length")
    if not amax_histories:
        return
    if recipe.scaling_factor_compute_algo is not None:
        raise ValueError("fused amax/scale update only supports the default scaling algorithm")

    try:
        algo = _AMAX_ALGO_MAP[recipe.amax_compute_algo]
    except KeyError as exc:
        raise ValueError(f"unsupported amax_compute_algo: {recipe.amax_compute_algo}") from exc

    first_history = amax_histories[0]
    first_scale = scales[0]
    if first_history.ndim != 2 or first_scale.ndim != 1:
        raise ValueError("amax_history tensors must be 2D and scale tensors must be 1D")

    history_len = first_history.shape[0]
    if history_len == 0:
        raise ValueError("amax_history_len must be greater than 0")
    if first_history.shape[1] != first_scale.shape[0]:
        raise ValueError("amax_history and scale tensor shapes are inconsistent")

    margin_factor = 2.0**recipe.margin
    block_h = triton.next_power_of_2(history_len)

    if len(amax_histories) == 1:
        history = amax_histories[0]
        scale = scales[0]
        if scale.shape[0] == 0:
            return

        num_quantizers = scale.shape[0]
        if num_quantizers >= 128:
            # Use vectorized kernel for large numbers of quantizers
            QUANTIZERS_PER_BLOCK = 8
            grid_size = triton.cdiv(num_quantizers, QUANTIZERS_PER_BLOCK)
            _single_update_vectorized_kernel[(grid_size,)](
                history,
                scale,
                fp8_max,
                history.stride(0),
                history.stride(1),
                num_quantizers,
                AMAX_ALGO=algo,
                H=history_len,
                BLOCK_H=block_h,
                MARGIN_FACTOR=margin_factor,
                QUANTIZERS_PER_BLOCK=QUANTIZERS_PER_BLOCK,
            )
        else:
            _single_update_kernel[(num_quantizers,)](
                history,
                scale,
                fp8_max,
                history.stride(0),
                history.stride(1),
                AMAX_ALGO=algo,
                H=history_len,
                BLOCK_H=block_h,
                MARGIN_FACTOR=margin_factor,
            )
        return

    num_pairs = len(amax_histories)
    scale_offset = num_pairs
    width_offset = 2 * num_pairs
    metadata_parts: List[int] = [0] * (3 * num_pairs)
    total_quantizers = 0
    max_quantizers = 0
    device = amax_histories[0].device
    for i, (history, scale) in enumerate(zip(amax_histories, scales)):
        if i != 0:
            if history.ndim != 2 or scale.ndim != 1:
                raise ValueError("amax_history tensors must be 2D and scale tensors must be 1D")
            if history.shape[0] != history_len:
                raise ValueError("all amax_history tensors must have the same history length")
            if history.shape[1] != scale.shape[0]:
                raise ValueError("amax_history and scale tensor shapes are inconsistent")
        quantizers = scale.shape[0]
        total_quantizers += quantizers
        max_quantizers = max(max_quantizers, quantizers)
        metadata_parts[i] = history.data_ptr()
        metadata_parts[scale_offset + i] = scale.data_ptr()
        metadata_parts[width_offset + i] = quantizers

    if total_quantizers == 0:
        return

    metadata = torch.tensor(metadata_parts, dtype=torch.int64, device=device)

    if max_quantizers >= 128 and num_pairs < 4:
        # Use vectorized kernel for large numbers of quantizers
        QUANTIZERS_PER_BLOCK = 8
        grid_size = triton.cdiv(max_quantizers, QUANTIZERS_PER_BLOCK)
        _multi_update_vectorized_kernel[(grid_size, num_pairs)](
            metadata,
            fp8_max,
            max_quantizers,
            AMAX_ALGO=algo,
            H=history_len,
            BLOCK_H=block_h,
            MARGIN_FACTOR=margin_factor,
            NUM_PAIRS=num_pairs,
            QUANTIZERS_PER_BLOCK=QUANTIZERS_PER_BLOCK,
        )
    else:
        _multi_update_kernel[(max_quantizers, num_pairs)](
            metadata,
            fp8_max,
            AMAX_ALGO=algo,
            H=history_len,
            BLOCK_H=block_h,
            MARGIN_FACTOR=margin_factor,
            NUM_PAIRS=num_pairs,
        )
