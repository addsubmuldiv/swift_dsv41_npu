# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor L2 norm operations for Ascend NPU."""

from typing import List, Optional, Tuple

import torch


def multi_tensor_l2norm(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    per_tensor: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute L2 norm for a list of contiguous tensors.

    .. warning::
        This API is **experimental** and subject to change.

    Parameters
    ----------
    chunk_size: int
        Number of tensor elements processed at a time (ignored on NPU).
    noop_flag: torch.Tensor
        Single-element int32 tensor, kernel exits if non-zero.
    tensor_lists: List[List[torch.Tensor]]
        One list: [grad_list] with N tensors.
    per_tensor: bool, optional
        If True, also return per-tensor norms. Default: False.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        (total_norm, per_tensor_norms) where per_tensor_norms is None
        if per_tensor is False.
    """
    per_tensor = bool(per_tensor)

    norms: List[torch.Tensor] = []
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            if grad.numel() == 0:
                norms.append(torch.tensor(0.0, dtype=torch.float32, device=grad.device))
            else:
                norm = grad.float().norm()
                # NVTE parity (multi_tensor/l2norm.cu L2NormFunctor): the
                # noop flag is set to 1 when the fp32 sum-of-squares
                # reduction is non-finite (inf/nan grads or reduction
                # overflow). sqrt is monotone, so checking the norm value
                # is equivalent to checking the block-level partial sums.
                if not torch.isfinite(norm):
                    noop_flag.fill_(1)
                norms.append(norm)

    if not norms:
        device = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else "cpu"
        ret = torch.zeros(1, dtype=torch.float32, device=device)
        ret_per_tensor = (
            torch.stack(norms) if per_tensor else torch.empty(0, dtype=torch.float32, device=device)
        )
        return ret, ret_per_tensor if per_tensor else None

    stacked = torch.stack(norms)
    total_norm = stacked.norm()

    if per_tensor:
        return total_norm, stacked
    return total_norm, None


def multi_tensor_unscale_l2norm(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: List[List[torch.Tensor]],
    inv_scale: torch.Tensor,
    per_tensor: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute L2 norm after unscaling.

    Unscaling is only for L2 norm computation; tensors are not updated.

    .. warning::
        This API is **experimental** and subject to change.

    Parameters
    ----------
    chunk_size: int
        Number of tensor elements processed at a time (ignored on NPU).
    noop_flag: torch.Tensor
        Single-element int32 tensor, kernel exits if non-zero.
    tensor_lists: List[List[torch.Tensor]]
        One list: [grad_list] with N tensors.
    inv_scale: torch.Tensor
        Tensor containing the inverse scaling factor.
    per_tensor: bool, optional
        If True, also return per-tensor norms. Default: False.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.Tensor]]
        (total_norm, per_tensor_norms) where per_tensor_norms is None
        if per_tensor is False.
    """
    per_tensor = bool(per_tensor)
    inv_scale_val = inv_scale.item()

    norms: List[torch.Tensor] = []
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            if grad.numel() == 0:
                norms.append(torch.tensor(0.0, dtype=torch.float32, device=grad.device))
            else:
                norm = (grad.float() * inv_scale_val).norm()
                # NVTE parity (multi_tensor/l2norm.cu UnscaleL2NormFunctor):
                # set the noop flag when the fp32 reduction is non-finite.
                if not torch.isfinite(norm):
                    noop_flag.fill_(1)
                norms.append(norm)

    if not norms:
        device = tensor_lists[0][0].device if tensor_lists and tensor_lists[0] else "cpu"
        ret = torch.zeros(1, dtype=torch.float32, device=device)
        ret_per_tensor = (
            torch.stack(norms) if per_tensor else torch.empty(0, dtype=torch.float32, device=device)
        )
        return ret, ret_per_tensor if per_tensor else None

    stacked = torch.stack(norms)
    total_norm = stacked.norm()

    if per_tensor:
        return total_norm, stacked
    return total_norm, None
