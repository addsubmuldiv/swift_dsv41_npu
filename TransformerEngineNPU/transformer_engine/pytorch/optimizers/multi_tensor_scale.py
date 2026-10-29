# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor scale operations for Ascend NPU."""

from typing import List

import torch


def multi_tensor_scale(
    chunk_size: int,
    is_infinite: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: float,
) -> None:
    """Overflow check + scale for a list of contigual tensors.

    .. warning::
        This API is **experimental** and subject to change.

    Parameters
    ----------
    chunk_size: int
        Number of tensor elements processed at a time (ignored on NPU).
    is_infinite: torch.Tensor
        Single-element int32 tensor, set to 1 if any input is non-finite.
    tensor_lists: List[List[torch.Tensor]]
        Two lists: [input_list, output_list], each with N tensors.
    scale: float
        Scaling factor.
    """
    if len(tensor_lists) != 2:
        raise AssertionError(f"tensor_lists must have exactly 2 elements, got {len(tensor_lists)}")
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError("input and output lists must have the same number of tensors")

    with torch.no_grad():
        for in_t, out_t in zip(tensor_lists[0], tensor_lists[1]):
            # NVTE parity (multi_tensor/scale.cu): the overflow flag is
            # computed from the INPUT values (isfinite before scaling), not
            # from the scaled output — scale-induced overflow is not flagged.
            if not torch.isfinite(in_t).all():
                is_infinite.fill_(1)
            out_t.copy_(in_t)
            out_t.mul_(scale)


def multi_tensor_scale_tensor(
    chunk_size: int,
    is_infinite: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    scale: torch.Tensor,
) -> None:
    """Overflow check + scale for a list of contiguous tensors.

    scale is passed as a tensor.

    .. warning::
        This API is **experimental** and subject to change.

    Parameters
    ----------
    chunk_size: int
        Number of tensor elements processed at a time (ignored on NPU).
    is_infinite: torch.Tensor
        Single-element int32 tensor, set to 1 if any input is non-finite.
    tensor_lists: List[List[torch.Tensor]]
        Two lists: [input_list, output_list], each with N tensors.
    scale: torch.Tensor
        Tensor containing the scaling factor.
    """
    multi_tensor_scale(chunk_size, is_infinite, tensor_lists, scale.item())
