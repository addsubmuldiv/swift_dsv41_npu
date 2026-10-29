# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Derived from NVIDIA Megatron-LM (core_v0.13.0rc3), which is licensed under
# the Apache License, Version 2.0. You may obtain a copy of the License at
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0rc3/LICENSE
# Modified by Huawei Technologies Co., Ltd. for the TransformerEngineNPU
# ATK CPU baselines. See LICENSE for license information.

import warnings
import torch
from typing import Optional

# ===================== cpu baseline ops =====================


def _permute_op_cpu(
    tokens,  # [num_tokens, hidden]
    routing_map,  # [num_tokens, num_experts], bool
    probs: Optional[torch.Tensor] = None,  # [num_tokens, num_experts]
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    CPU baseline of operator torch_npu.npu_moe_token_permute_with_routing_map() and
    torch_npu.npu_moe_token_permute_with_routing_map_grad()
    fused mode is not used in the moe permute and unpermute tests
    """

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        sorted_indices = (
            routing_map.argsort(dim=-1, descending=True, stable=True)[:, :capacity]
            .contiguous()
            .to(dtype=torch.int32)
        )
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            probs_T_1D = probs.T.contiguous().view(-1)
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        routing_map = routing_map.bool().T.contiguous()  # [num_experts, num_tokens]
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)

    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def _unpermute_op_cpu(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    CPU baseline of operator torch_npu.npu_moe_token_unpermute_with_routing_map() and
    torch_npu.npu_moe_token_unpermute_with_routing_map_grad()
    fused mode is not used.
    """
    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            probs_T_1D = probs.T.contiguous().view(-1)

            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            # Integer 0/1 routing maps are supported by the adapter, but
            # masked_select requires a boolean mask.
            permuted_probs = probs.T.contiguous().masked_select(routing_map.bool().T.contiguous())

        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    if not torch.is_floating_point(sorted_indices):
        sorted_indices = sorted_indices.to(torch.long)
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens.to(dtype=input_dtype)


# ===================== cpu baseline API-like interfaces =====================


def mp_pad_probs_cpu(
    inp,  # [num_tokens, hidden]
    probs: torch.Tensor,  # [num_tokens, num_experts]
    routing_map: torch.Tensor,  # [num_tokens, num_experts], bool
    tokens_per_expert: torch.Tensor,
    align_size: int,
):
    """CPU implementation of the ``moe_permute_and_pad_with_probs`` API surface.

    Reimplements the public API on CPU: the fused device operator is replaced
    with the CPU baseline ``_permute_op_cpu``, keeping the signature and the
    returned tensor layout unchanged.
    """
    if tokens_per_expert is None:
        raise ValueError(
            "tokens_per_expert must be provided to the fused permute padding function."
        )
    if align_size <= 0:
        raise ValueError(f"align_size must be positive, got {align_size}.")

    # Ensure tokens_per_expert is on the same device as input to avoid device transfers
    if tokens_per_expert.device != inp.device:
        tokens_per_expert = tokens_per_expert.to(inp.device)

    # Calculate aligned token counts per expert
    target_tokens_per_expert = (torch.ceil(tokens_per_expert / align_size) * align_size).long()

    # Calculate pad_offsets (cumulative padding offsets per expert)
    pad_lengths = target_tokens_per_expert - tokens_per_expert
    cum_pad = torch.cumsum(pad_lengths, dim=0)
    pad_offsets = torch.cat([torch.zeros(1, dtype=cum_pad.dtype, device=inp.device), cum_pad[:-1]])
    if torch.all(pad_offsets == 0):
        pad_offsets = None

    # Use cpu baseline instead of nvte CUDA baseline
    output, permuted_probs, row_id_map = _permute_op_cpu(
        inp, routing_map, probs, target_tokens_per_expert.sum().item(), drop_and_pad=True
    )
    return output, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert


def moe_unpermute_pad_probs_cpu(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
    map_type: str = "mask",
    probs: Optional[torch.Tensor] = None,
    pad_offsets: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
):
    """CPU implementation of the mask-map ``moe_unpermute`` path with padding.

    Reimplements the public ``moe_unpermute`` interface (``map_type="mask"``
    with optional ``pad_offsets``) on CPU by replacing the device operator
    with the CPU baseline ``_unpermute_op_cpu``. When ``merging_probs`` is
    supplied, the matching ``routing_map`` is required and forwarded to the
    baseline.
    """
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    if map_type == "index":
        raise ValueError("moe_unpermute_and_pad_with_probs requires map_type as 'mask'")
    if map_type == "mask":
        # Check that routing_map is provided when merging_probs is not None
        if merging_probs is not None and routing_map is None:
            raise ValueError("Mask must be provided to permute the probs. ")

        if restore_shape is None:
            restore_shape = inp.shape
        drop_and_pad = pad_offsets is not None

        unpermuted_output = _unpermute_op_cpu(
            inp,
            row_id_map,
            restore_shape,
            merging_probs,
            routing_map,
            drop_and_pad=drop_and_pad,
        )
        return unpermuted_output
    raise ValueError("map_type should be one of 'mask' or 'index'")
