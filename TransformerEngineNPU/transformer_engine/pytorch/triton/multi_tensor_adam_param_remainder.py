# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor Adam/AdamW update for BF16 parameters with INT16 remainders."""

import math
import struct

import torch
import triton
import triton.language as tl
from triton.runtime import driver


def _to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@triton.jit
def _multi_tensor_adam_param_remainder_kernel(
    metadata,
    num_tensors,
    total_tasks,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step_size,
    bias_correction2_inv_sqrt,
    GRAD_DTYPE: tl.constexpr,
    MOMENT_DTYPE: tl.constexpr,
    ADAM_W_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_offsets = tl.arange(0, BLOCK_SIZE)

    for task_idx in range(pid, total_tasks, num_programs):
        task_offset = 6 * num_tensors + 3 * task_idx
        tensor_idx = tl.load(metadata + task_offset)
        first_block = tl.load(metadata + task_offset + 1)
        num_blocks = tl.load(metadata + task_offset + 2)

        grad_ptr = tl.load(metadata + tensor_idx).to(tl.pointer_type(GRAD_DTYPE))
        param_ptr = tl.load(metadata + num_tensors + tensor_idx).to(tl.pointer_type(tl.bfloat16))
        exp_avg_ptr = tl.load(metadata + 2 * num_tensors + tensor_idx).to(
            tl.pointer_type(MOMENT_DTYPE)
        )
        exp_avg_sq_ptr = tl.load(metadata + 3 * num_tensors + tensor_idx).to(
            tl.pointer_type(MOMENT_DTYPE)
        )
        remainder_ptr = tl.load(metadata + 4 * num_tensors + tensor_idx).to(
            tl.pointer_type(tl.int16)
        )
        numel = tl.load(metadata + 5 * num_tensors + tensor_idx)

        for block_idx in range(first_block, first_block + num_blocks):
            offsets = block_idx * BLOCK_SIZE + block_offsets
            mask = offsets < numel
            grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            param_bf16 = tl.load(param_ptr + offsets, mask=mask, other=0.0)
            remainder = tl.load(remainder_ptr + offsets, mask=mask, other=0)
            exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            high = param_bf16.to(tl.int16, bitcast=True).to(tl.int32)
            high -= remainder < 0
            param_bits = ((high & 0xFFFF) << 16) | (remainder.to(tl.int32) & 0xFFFF)
            param = param_bits.to(tl.float32, bitcast=True)

            if ADAM_W_MODE:
                param *= 1.0 - lr * weight_decay
            else:
                grad += weight_decay * param

            exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
            exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
            denom = tl.sqrt(exp_avg_sq) * bias_correction2_inv_sqrt + eps
            param -= (exp_avg / denom) * step_size

            param_bits = param.to(tl.int32, bitcast=True)
            low = param_bits & 0xFFFF
            high = (param_bits >> 16) & 0xFFFF
            high += low >= 0x8000
            param_bf16 = high.to(tl.int16).to(tl.bfloat16, bitcast=True)

            tl.store(param_ptr + offsets, param_bf16, mask=mask)
            tl.store(remainder_ptr + offsets, low.to(tl.int16), mask=mask)
            tl.store(exp_avg_ptr + offsets, exp_avg.to(MOMENT_DTYPE), mask=mask)
            tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq.to(MOMENT_DTYPE), mask=mask)


def multi_tensor_adam_param_remainder(
    tensor_lists: list[list[torch.Tensor]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    adam_w_mode: bool,
    weight_decay: float,
    metadata_cache: dict | None = None,
    cache_key=None,
    block_size: int = 1024,
    tasks_per_core: int = 4,
) -> None:
    """Update ``[grads, BF16 params, m, v, INT16 remainders]`` in one launch."""

    if len(tensor_lists) != 5:
        raise ValueError(f"tensor_lists must contain five lists, got {len(tensor_lists)}")
    grads, params, exp_avgs, exp_avg_sqs, remainders = tensor_lists
    num_tensors = len(params)
    if not (len(grads) == len(exp_avgs) == len(exp_avg_sqs) == len(remainders) == num_tensors):
        raise ValueError("all parameter remainder tensor lists must have equal length")
    if num_tensors == 0:
        return
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a positive power of two, got {block_size}")
    if tasks_per_core <= 0:
        raise ValueError(f"tasks_per_core must be positive, got {tasks_per_core}")

    device = params[0].device
    grad_dtype = grads[0].dtype
    moment_dtype = exp_avgs[0].dtype
    if device.type != "npu":
        raise ValueError(f"Triton parameter remainder Adam requires NPU tensors, got {device}")
    if grad_dtype == torch.float32:
        triton_grad_dtype = tl.float32
    elif grad_dtype == torch.float16:
        triton_grad_dtype = tl.float16
    elif grad_dtype == torch.bfloat16:
        triton_grad_dtype = tl.bfloat16
    else:
        raise ValueError(f"unsupported gradient dtype: {grad_dtype}")
    if moment_dtype == torch.float32:
        triton_moment_dtype = tl.float32
    elif moment_dtype == torch.bfloat16:
        triton_moment_dtype = tl.bfloat16
    else:
        raise ValueError(f"unsupported optimizer state dtype: {moment_dtype}")

    signature = []
    total_blocks = 0
    num_blocks_per_tensor = []
    for grad, param, exp_avg, exp_avg_sq, remainder in zip(
        grads, params, exp_avgs, exp_avg_sqs, remainders
    ):
        tensors = (grad, param, exp_avg, exp_avg_sq, remainder)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all tensors must be on the same NPU device")
        if grad.dtype != grad_dtype or param.dtype != torch.bfloat16:
            raise ValueError("gradients must share a dtype and parameters must be BF16")
        if exp_avg.dtype != moment_dtype or exp_avg_sq.dtype != moment_dtype:
            raise ValueError("exp_avg and exp_avg_sq must share one FP32 or BF16 dtype")
        if remainder.dtype != torch.int16:
            raise ValueError("parameter remainders must be int16")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Triton parameter remainder Adam requires contiguous tensors")
        if any(tensor.numel() != param.numel() for tensor in tensors):
            raise ValueError("parameter, gradient, and optimizer state sizes must match")

        num_blocks = triton.cdiv(param.numel(), block_size)
        num_blocks_per_tensor.append(num_blocks)
        total_blocks += num_blocks
        signature.append(tuple(tensor.data_ptr() for tensor in tensors) + (param.numel(),))

    if total_blocks == 0:
        return

    num_vector_cores = driver.active.utils.get_device_properties(device.index)["num_vectorcore"]
    cache_slot = (cache_key, block_size, tasks_per_core, num_vector_cores, moment_dtype)
    signature = tuple(signature)
    cached = metadata_cache.get(cache_slot) if metadata_cache is not None else None
    if cached is not None and cached[0] == signature and cached[1].device == device:
        metadata, total_tasks = cached[1:]
    else:
        blocks_per_task = max(1, triton.cdiv(total_blocks, num_vector_cores * tasks_per_core))
        metadata_parts = [0] * (6 * num_tensors)
        for tensor_idx, tensors in enumerate(zip(grads, params, exp_avgs, exp_avg_sqs, remainders)):
            for list_idx, tensor in enumerate(tensors):
                metadata_parts[list_idx * num_tensors + tensor_idx] = tensor.data_ptr()
            metadata_parts[5 * num_tensors + tensor_idx] = params[tensor_idx].numel()
            for first_block in range(0, num_blocks_per_tensor[tensor_idx], blocks_per_task):
                metadata_parts.extend(
                    (
                        tensor_idx,
                        first_block,
                        min(
                            blocks_per_task,
                            num_blocks_per_tensor[tensor_idx] - first_block,
                        ),
                    )
                )
        total_tasks = (len(metadata_parts) - 6 * num_tensors) // 3
        metadata = torch.tensor(metadata_parts, dtype=torch.int64, device=device)
        if metadata_cache is not None:
            metadata_cache[cache_slot] = (signature, metadata, total_tasks)

    lr = _to_float32(lr)
    beta1 = _to_float32(beta1)
    beta2 = _to_float32(beta2)
    eps = _to_float32(eps)
    weight_decay = _to_float32(weight_decay)
    beta1_power = _to_float32(beta1**step)
    beta2_power = _to_float32(beta2**step)
    step_size = _to_float32(lr / _to_float32(1.0 - beta1_power))
    bias_correction2_inv_sqrt = _to_float32(1.0 / math.sqrt(_to_float32(1.0 - beta2_power)))

    _multi_tensor_adam_param_remainder_kernel[(min(total_tasks, num_vector_cores),)](
        metadata,
        num_tensors,
        total_tasks,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step_size,
        bias_correction2_inv_sqrt,
        GRAD_DTYPE=triton_grad_dtype,
        MOMENT_DTYPE=triton_moment_dtype,
        ADAM_W_MODE=adam_w_mode,
        BLOCK_SIZE=block_size,
    )
