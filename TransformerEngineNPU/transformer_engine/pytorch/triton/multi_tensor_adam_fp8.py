# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fused multi-tensor Adam/AdamW update for delayed-scaling FP8 parameters."""

import math
import struct

import torch
import triton
import triton.language as tl
from triton.runtime import driver


_NUM_TENSOR_METADATA_FIELDS = tl.constexpr(9)
_NUM_TASK_METADATA_FIELDS = tl.constexpr(3)


def _to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@triton.jit
def _multi_tensor_adam_fp8_kernel(
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
    FP8_DTYPE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ADAM_W_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_offsets = tl.arange(0, BLOCK_SIZE)

    for task_idx in range(pid, total_tasks, num_programs):
        # Per-tensor metadata arrays precede the fixed-width task records.
        # Operand order matters: num_tensors can be specialized to constexpr(1)
        # by Triton when a single tensor is passed, and raw-int globals do not
        # implement __mul__(constexpr).
        task_offset = (
            num_tensors * _NUM_TENSOR_METADATA_FIELDS + task_idx * _NUM_TASK_METADATA_FIELDS
        )
        tensor_idx = tl.load(metadata + task_offset)
        first_block = tl.load(metadata + task_offset + 1)
        num_blocks = tl.load(metadata + task_offset + 2)

        grad_ptr = tl.load(metadata + tensor_idx).to(tl.pointer_type(GRAD_DTYPE))
        param_ptr = tl.load(metadata + num_tensors + tensor_idx).to(tl.pointer_type(FP8_DTYPE))
        exp_avg_ptr = tl.load(metadata + 2 * num_tensors + tensor_idx).to(
            tl.pointer_type(MOMENT_DTYPE)
        )
        exp_avg_sq_ptr = tl.load(metadata + 3 * num_tensors + tensor_idx).to(
            tl.pointer_type(MOMENT_DTYPE)
        )
        master_ptr = tl.load(metadata + 4 * num_tensors + tensor_idx).to(
            tl.pointer_type(tl.float32)
        )
        scale_ptr = tl.load(metadata + 5 * num_tensors + tensor_idx).to(tl.pointer_type(tl.float32))
        amax_ptr = tl.load(metadata + 6 * num_tensors + tensor_idx).to(tl.pointer_type(tl.float32))
        scale_inv_ptr = tl.load(metadata + 7 * num_tensors + tensor_idx).to(
            tl.pointer_type(tl.float32)
        )
        numel = tl.load(metadata + 8 * num_tensors + tensor_idx)
        scale = tl.load(scale_ptr).to(tl.float32)
        task_amax = 0.0

        for block_idx in range(first_block, first_block + num_blocks):
            offsets = block_idx * BLOCK_SIZE + block_offsets
            mask = offsets < numel
            grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            master = tl.load(master_ptr + offsets, mask=mask, other=0.0)
            exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            if ADAM_W_MODE:
                master *= 1.0 - lr * weight_decay
            else:
                grad += weight_decay * master

            exp_avg = exp_avg * beta1 + grad * (1.0 - beta1)
            exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1.0 - beta2)
            denom = tl.sqrt(exp_avg_sq) * bias_correction2_inv_sqrt + eps
            master -= (exp_avg / denom) * step_size

            scaled = tl.minimum(tl.maximum(master * scale, -FP8_MAX), FP8_MAX)
            tl.store(param_ptr + offsets, scaled.to(FP8_DTYPE), mask=mask)
            tl.store(master_ptr + offsets, master, mask=mask)
            tl.store(exp_avg_ptr + offsets, exp_avg.to(MOMENT_DTYPE), mask=mask)
            tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq.to(MOMENT_DTYPE), mask=mask)
            block_amax = tl.max(tl.where(mask, tl.abs(master), 0.0), axis=0)
            task_amax = tl.maximum(task_amax, block_amax)

        tl.atomic_max(amax_ptr, task_amax)
        tl.store(scale_inv_ptr, 1.0 / scale)


def multi_tensor_adam_fp8(
    tensor_lists: list[list[torch.Tensor]],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    adam_w_mode: bool,
    bias_correction: bool,
    weight_decay: float,
    fp8_dtype: torch.dtype,
    metadata_cache: dict | None = None,
    cache_key=None,
    block_size: int = 1024,
    tasks_per_core: int = 4,
) -> None:
    """Update ``[g, p_fp8, m, v, p_master, scale, amax, scale_inv]`` in one launch."""

    if len(tensor_lists) != 8:
        raise ValueError(f"tensor_lists must contain eight lists, got {len(tensor_lists)}")
    grads, params, exp_avgs, exp_avg_sqs, masters, scales, amaxes, scale_invs = tensor_lists
    num_tensors = len(params)
    if any(len(tensors) != num_tensors for tensors in tensor_lists):
        raise ValueError("all delayed FP8 Adam tensor lists must have equal length")
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
        raise ValueError(f"Triton delayed FP8 Adam requires NPU tensors, got {device}")
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
    if fp8_dtype == torch.float8_e4m3fn:
        triton_fp8_dtype = tl.float8e4nv
    elif fp8_dtype == torch.float8_e5m2:
        triton_fp8_dtype = tl.float8e5
    else:
        raise ValueError(f"unsupported FP8 dtype: {fp8_dtype}")

    signature = []
    total_blocks = 0
    num_blocks_per_tensor = []
    for tensors in zip(
        grads,
        params,
        exp_avgs,
        exp_avg_sqs,
        masters,
        scales,
        amaxes,
        scale_invs,
    ):
        grad, param, exp_avg, exp_avg_sq, master, *_ = tensors
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("all delayed FP8 Adam tensors must be on the same NPU device")
        if grad.dtype != grad_dtype:
            raise ValueError("gradients in one delayed FP8 group must share a dtype")
        if param.element_size() != 1:
            raise ValueError("delayed FP8 parameter storage must use one byte per element")
        if master.dtype != torch.float32:
            raise ValueError("master parameters must be float32")
        if exp_avg.dtype != moment_dtype or exp_avg_sq.dtype != moment_dtype:
            raise ValueError("exp_avg and exp_avg_sq must share one FP32 or BF16 dtype")
        if any(tensor.dtype != torch.float32 or tensor.numel() != 1 for tensor in tensors[5:]):
            raise ValueError("scale, amax, and scale_inv must be float32 scalars")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Triton delayed FP8 Adam requires contiguous tensors")
        if any(tensor.numel() != master.numel() for tensor in tensors[:5]):
            raise ValueError("gradient, parameter, optimizer state, and master sizes must match")

        num_blocks = triton.cdiv(master.numel(), block_size)
        num_blocks_per_tensor.append(num_blocks)
        total_blocks += num_blocks
        signature.append(tuple(tensor.data_ptr() for tensor in tensors) + (master.numel(),))

    if total_blocks == 0:
        return

    num_vector_cores = driver.active.utils.get_device_properties(device.index)["num_vectorcore"]
    cache_slot = (
        cache_key,
        block_size,
        tasks_per_core,
        num_vector_cores,
        grad_dtype,
        moment_dtype,
        fp8_dtype,
    )
    signature = tuple(signature)
    cached = metadata_cache.get(cache_slot) if metadata_cache is not None else None
    if cached is not None and cached[0] == signature and cached[1].device == device:
        metadata, total_tasks = cached[1:]
    else:
        blocks_per_task = max(1, triton.cdiv(total_blocks, num_vector_cores * tasks_per_core))
        metadata_parts = [0] * (int(_NUM_TENSOR_METADATA_FIELDS) * num_tensors)
        for tensor_idx, tensors in enumerate(
            zip(
                grads,
                params,
                exp_avgs,
                exp_avg_sqs,
                masters,
                scales,
                amaxes,
                scale_invs,
            )
        ):
            for list_idx, tensor in enumerate(tensors):
                metadata_parts[list_idx * num_tensors + tensor_idx] = tensor.data_ptr()
            metadata_parts[8 * num_tensors + tensor_idx] = masters[tensor_idx].numel()
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
        # total_tasks is a runtime (non-constexpr) kernel argument; Triton's
        # argument binder cannot mangle a constexpr object for such parameters.
        total_tasks = int(
            (len(metadata_parts) - _NUM_TENSOR_METADATA_FIELDS * num_tensors)
            // _NUM_TASK_METADATA_FIELDS
        )
        metadata = torch.tensor(metadata_parts, dtype=torch.int64, device=device)
        if metadata_cache is not None:
            metadata_cache[cache_slot] = (signature, metadata, total_tasks)

    lr = _to_float32(lr)
    beta1 = _to_float32(beta1)
    beta2 = _to_float32(beta2)
    eps = _to_float32(eps)
    weight_decay = _to_float32(weight_decay)
    if bias_correction:
        beta1_power = _to_float32(beta1**step)
        beta2_power = _to_float32(beta2**step)
        step_size = _to_float32(lr / _to_float32(1.0 - beta1_power))
        bias_correction2_inv_sqrt = _to_float32(1.0 / math.sqrt(_to_float32(1.0 - beta2_power)))
    else:
        step_size = lr
        bias_correction2_inv_sqrt = 1.0

    _multi_tensor_adam_fp8_kernel[(min(total_tasks, num_vector_cores),)](
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
        FP8_DTYPE=triton_fp8_dtype,
        FP8_MAX=torch.finfo(fp8_dtype).max,
        ADAM_W_MODE=adam_w_mode,
        BLOCK_SIZE=block_size,
    )
