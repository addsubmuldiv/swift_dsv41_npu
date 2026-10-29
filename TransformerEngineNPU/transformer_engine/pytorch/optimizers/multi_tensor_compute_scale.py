# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor scale computation for FP8 / MXFP8 quantization on Ascend NPU."""

from typing import List

import torch


def multi_tensor_compute_scale_and_scale_inv(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    max_fp8: float,
    force_pow_2_scales: bool,
    epsilon: float,
) -> None:
    """Compute quantization scale and inverse scale from amax.

    Operates in-place on tensor_lists.

    .. warning::
        This API is **experimental** and subject to change.

    Parameters
    ----------
    chunk_size: int
        Number of tensor elements processed at a time (ignored on NPU).
    noop_flag: torch.Tensor
        Single-element int32 tensor, kernel exits if non-zero.
    tensor_lists: List[List[torch.Tensor]]
        Three lists: [amax_list, scale_list, scale_inv_list].
    max_fp8: float
        Maximum representable value in the target FP8 format.
    force_pow_2_scales: bool
        If True, force scales to be powers of 2.
    epsilon: float
        Term added to amax for numerical stability.
    """
    if len(tensor_lists) != 3:
        raise AssertionError(f"tensor_lists must have exactly 3 elements, got {len(tensor_lists)}")

    amax_list, scale_list, scale_inv_list = tensor_lists
    if not (len(amax_list) == len(scale_list) == len(scale_inv_list)):
        raise AssertionError("amax, scale, and scale_inv lists must have the same length")
    if not amax_list:
        return

    with torch.no_grad():
        if getattr(amax_list[0], "is_npu", False):
            # Match upstream's fused multi-tensor implementation with one Triton launch.
            num_tensors = len(amax_list)
            metadata_parts = [0] * (4 * num_tensors)
            block_size = 1024
            total_blocks = 0
            device = amax_list[0].device
            for i, (amax, scale, scale_inv) in enumerate(
                zip(amax_list, scale_list, scale_inv_list)
            ):
                if amax.device != device or scale.device != device or scale_inv.device != device:
                    raise ValueError("all tensors must be on the same device")
                if amax.dtype != torch.float32:
                    raise ValueError(f"amax must be float32, got {amax.dtype}")
                if scale.dtype != torch.float32 or scale_inv.dtype != torch.float32:
                    raise ValueError("scale and scale_inv must be float32")
                if (
                    not amax.is_contiguous()
                    or not scale.is_contiguous()
                    or not scale_inv.is_contiguous()
                ):
                    raise ValueError("amax, scale, and scale_inv must be contiguous")
                if amax.numel() != scale.numel() or amax.numel() != scale_inv.numel():
                    raise ValueError("amax, scale, and scale_inv sizes must match")

                metadata_parts[i] = amax.data_ptr()
                metadata_parts[num_tensors + i] = scale.data_ptr()
                metadata_parts[2 * num_tensors + i] = scale_inv.data_ptr()
                metadata_parts[3 * num_tensors + i] = amax.numel()
                num_blocks = (amax.numel() + block_size - 1) // block_size
                for block_idx in range(num_blocks):
                    metadata_parts.extend((i, block_idx))
                total_blocks += num_blocks

            if total_blocks == 0:
                return

            from ..triton.multi_tensor_compute_scale import (
                _multi_tensor_compute_scale_and_scale_inv_kernel,
            )

            metadata = torch.tensor(metadata_parts, dtype=torch.int64, device=device)
            _multi_tensor_compute_scale_and_scale_inv_kernel[(total_blocks,)](
                metadata,
                max_fp8,
                epsilon,
                NUM_TENSORS=num_tensors,
                FORCE_POW_2_SCALES=force_pow_2_scales,
                BLOCK_SIZE=block_size,
            )
            return

        for amax, scale, scale_inv in zip(amax_list, scale_list, scale_inv_list):
            clamped = torch.max(
                amax, torch.tensor(epsilon, dtype=torch.float32, device=amax.device)
            )
            scale_tmp = torch.div(
                torch.tensor(max_fp8, dtype=torch.float32, device=amax.device), clamped
            )
            scale_tmp = torch.where(
                torch.isinf(scale_tmp),
                torch.tensor(
                    torch.finfo(torch.float32).max,
                    dtype=torch.float32,
                    device=amax.device,
                ),
                scale_tmp,
            )

            if force_pow_2_scales:
                _, exp = torch.frexp(scale_tmp)
                exp = exp - 1
                unity = torch.tensor(1.0, device=exp.device)
                torch.ldexp(unity, exp, out=scale_tmp)
                scale_tmp = torch.where(amax == float("inf"), 0.0, scale_tmp)

            scale_tmp = torch.where(amax == 0.0, 1.0, scale_tmp)
            scale.copy_(scale_tmp)
            torch.reciprocal(scale_tmp, out=scale_inv)


def multi_tensor_compute_scale_inv_e8m0(
    chunk_size: int,
    dummy: object,
    tensor_lists: List[List[torch.Tensor]],
) -> None:
    """Compute E8M0 scale_inv from amax for MXFP8 scaling.

    Operates in-place on tensor_lists.

    .. warning::
        This API is **experimental** and subject to change.

    Parameters
    ----------
    chunk_size: int
        Number of tensor elements processed at a time (ignored on NPU).
    dummy: object
        Ignored; kept for API compatibility (must be None).
    tensor_lists: List[List[torch.Tensor]]
        Two lists: [amax_list, scale_inv_list].
    """
    if dummy is not None:
        raise RuntimeError("No-op flag is not supported.")

    amax_list, scale_inv_list = tensor_lists

    with torch.no_grad():
        for amax, scale_inv in zip(amax_list, scale_inv_list):
            amax_fp32 = amax.float()
            exponent = torch.floor(torch.log2(amax_fp32))
            # Correct NPU log2 boundary errors against the reconstructed exponent interval.
            power_of_two = torch.exp2(exponent)
            exponent += (amax_fp32 >= power_of_two * 2).to(exponent.dtype)
            exponent -= (amax_fp32 < power_of_two).to(exponent.dtype)
            exponent = exponent.to(torch.int32) - 8 + 127
            exponent.clamp_(0, 254)
            exponent = torch.where(amax > 0, exponent, 0)
            scale_inv.copy_(exponent.to(torch.uint8))
