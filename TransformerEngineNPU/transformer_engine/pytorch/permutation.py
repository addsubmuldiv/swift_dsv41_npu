# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoE Permutation API for NPU"""

import warnings
from typing import List, Optional, Tuple
import torch
import torch_npu

__all__ = [
    "moe_permute",
    "moe_unpermute",
    "moe_sort_chunks_by_index",
    "moe_sort_chunks_by_index_with_probs",
    "moe_permute_with_probs",
    "moe_permute_and_pad_with_probs",
]


# ===================== Helper Functions =====================


def _convert_tensors_to_fp32_if_needed(
    tensor1: Optional[torch.Tensor], tensor2: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.dtype]]:
    """Convert tensors to fp32 if they have different dtypes and one is already fp32."""
    if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
        return tensor1, tensor2, None

    dtype1, dtype2 = tensor1.dtype, tensor2.dtype

    if (dtype1 == torch.float32) ^ (dtype2 == torch.float32):
        original_dtype = dtype2 if dtype1 == torch.float32 else dtype1
        tensor1 = tensor1.to(torch.float32)
        tensor2 = tensor2.to(torch.float32)
        return tensor1, tensor2, original_dtype

    return tensor1, tensor2, None


def _restore_original_dtype(
    tensor1: Optional[torch.Tensor],
    tensor2: Optional[torch.Tensor],
    original_dtype: Optional[torch.dtype],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Restore tensors to their original dtype if needed."""
    if original_dtype is None:
        return tensor1, tensor2

    return (
        tensor1.to(original_dtype) if tensor1 is not None else tensor1,
        tensor2.to(original_dtype) if tensor2 is not None else tensor2,
    )


# ===================== _moe_permute_index_map custom ops =====================


@torch.library.custom_op("te_moe::permute_index_map", mutates_args=[])
def moe_permute_index_map_forward(
    inp: torch.Tensor,
    index: torch.Tensor,
    num_out_tokens: int,
    max_token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for MoE permute with index router map."""
    if not inp.numel():
        # Empty input: return empty tensors with correct shapes
        # row_id_map shape should be [num_tokens * topK]
        num_tokens = inp.shape[0]
        topK = index.shape[1]
        return inp.clone(), torch.empty((num_tokens * topK,), dtype=torch.int32, device=inp.device)

    if not inp.is_npu:
        raise ValueError(f"inp must be a NPU tensor, but got tensor on {inp.device}.")
    if not index.is_npu:
        raise ValueError(f"index must be a NPU tensor, but got tensor on {index.device}.")
    if inp.size(0) != index.size(0):
        raise ValueError(
            f"Permute not possible: inp.size(0) ({inp.size(0)}) must match "
            f"index.size(0) ({index.size(0)})."
        )
    assert num_out_tokens >= 0, (
        f"moe_permute (index map) requires num_out_tokens >= 0, got {num_out_tokens}."
    )

    # index supports INT32 and INT64 according to documentation
    if index.dtype not in (torch.int32, torch.int64):
        warnings.warn(
            f"The data type of the input `index` of Permute is {index.dtype}! "
            "The supported types are torch.int32 and torch.int64."
        )
        index = index.to(torch.int32)

    topK = index.size(1)

    # Call NPU-specific operator
    # torch_npu.npu_moe_token_permute expects:
    # - tokens: [num_tokens, hidden_size]
    # - indices: [num_tokens, topK]
    # - num_out_tokens: int (optional, default 0)
    # - padded_mode: bool (optional, default False)
    # Returns: permuted_tokens, sorted_indices
    permuted_act, row_id_map = torch_npu.npu_moe_token_permute(
        inp, index, num_out_tokens=num_out_tokens, padded_mode=False
    )

    return permuted_act, row_id_map


@moe_permute_index_map_forward.register_fake
def _moe_permute_index_map_fake(
    inp: torch.Tensor,
    index: torch.Tensor,
    num_out_tokens: int,
    max_token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference."""
    num_tokens = inp.shape[0]
    topK = index.shape[1]
    if num_tokens > 0:
        assert num_out_tokens >= 0, (
            f"moe_permute (index map) requires num_out_tokens >= 0, got {num_out_tokens}."
        )

    # Infer output shape
    output_tokens = num_out_tokens if num_out_tokens > 0 else num_tokens * topK

    # row_id_map is 1D with size = num_tokens * topK
    fake_output = torch.empty((output_tokens, inp.shape[1]), dtype=inp.dtype, device=inp.device)
    fake_row_id_map = torch.empty((num_tokens * topK,), dtype=torch.int32, device=inp.device)

    return fake_output, fake_row_id_map


@torch.library.custom_op("te_moe::permute_index_map_bwd", mutates_args=[])
def moe_permute_index_map_backward(
    tokens: torch.Tensor,
    grad_permuted_act: torch.Tensor,
    indices: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for MoE permute with index router map."""
    # NPU backward operator: torch_npu.npu_moe_token_permute_grad
    # Parameters: tokens, grad_permuted_tokens, indices, sorted_indices, padded_mode
    act_grad = torch_npu.npu_moe_token_permute_grad(
        tokens, grad_permuted_act, indices, sorted_indices, padded_mode=False
    )
    return act_grad


@moe_permute_index_map_backward.register_fake
def _moe_permute_index_map_backward_fake(
    tokens: torch.Tensor,
    grad_permuted_act: torch.Tensor,
    indices: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for shape inference of backward."""
    return torch.empty(
        (tokens.size(0), grad_permuted_act.shape[1]),
        dtype=grad_permuted_act.dtype,
        device=grad_permuted_act.device,
    )


def _moe_permute_index_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, index, _num_out_tokens, _max_token_num = inputs
    permuted_act, sorted_indices = output
    ctx.empty_input = inp.size(0) == 0
    ctx.save_for_backward(inp, index, sorted_indices)


def _moe_permute_index_map_backward_wrapper(ctx, grad_permuted_act, grad_row_id_map):
    """Backward pass wrapper that calls the custom backward op."""
    if ctx.empty_input:
        return grad_permuted_act, None, None, None

    if not grad_permuted_act.is_contiguous():
        grad_permuted_act = grad_permuted_act.contiguous()

    tokens, indices, sorted_indices = ctx.saved_tensors
    act_grad = torch.ops.te_moe.permute_index_map_bwd(
        tokens, grad_permuted_act, indices, sorted_indices
    )

    return act_grad, None, None, None


moe_permute_index_map_forward.register_autograd(
    _moe_permute_index_map_backward_wrapper,
    setup_context=_moe_permute_index_map_setup_context,
)


# ===================== _moe_unpermute_index_map custom ops =====================


@torch.library.custom_op("te_moe::unpermute_index_map_fwd", mutates_args=[])
def moe_unpermute_index_map_forward(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    topK: int,
) -> torch.Tensor:
    """Forward pass for MoE unpermute with index router map."""
    if not inp.numel():
        # Empty input: return empty tensor with correct shape
        # unpermuted_output shape should be [num_tokens, hidden_size]
        # Note: inp.shape may not match [num_tokens, hidden_size] when inp.numel() == 0
        hidden_size = inp.shape[1] if inp.shape[1] > 0 else 0
        return torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)

    # Use torch_npu.npu_moe_token_unpermute for index mode
    # Parameters: permutedTokens, sortedIndices, probsOptional, paddedMode, restoreShapeOptional
    # Returns: out
    if probs is not None and probs.numel() > 0:
        unpermuted_output = torch_npu.npu_moe_token_unpermute(
            inp, row_id_map, probs=probs, padded_mode=False, restore_shape=None
        )
    else:
        # Basic unpermute without probs
        unpermuted_output = torch_npu.npu_moe_token_unpermute(
            inp, row_id_map, probs=None, padded_mode=False, restore_shape=None
        )

    return unpermuted_output


@moe_unpermute_index_map_forward.register_fake
def _moe_unpermute_index_map_forward_fake(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    topK: int,
) -> torch.Tensor:
    """Fake implementation for shape inference."""
    # Output shape: (num_tokens, hidden_size)
    return torch.empty((num_tokens, inp.shape[1]), dtype=inp.dtype, device=inp.device)


@torch.library.custom_op("te_moe::unpermute_index_map_bwd", mutates_args=[])
def moe_unpermute_index_map_backward(
    permuted_tokens: torch.Tensor,
    grad_unpermuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE unpermute with index router map."""
    # Use torch_npu.npu_moe_token_unpermute_grad for backward
    # Parameters: permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs, padded_mode, restore_shape
    # Returns: grad_permuted_tokens, grad_probs
    act_grad, prob_grad = torch_npu.npu_moe_token_unpermute_grad(
        permuted_tokens,
        grad_unpermuted_tokens,
        sorted_indices,
        probs=probs,
        padded_mode=False,
        restore_shape=None,
    )

    return act_grad, prob_grad


@moe_unpermute_index_map_backward.register_fake
def _moe_unpermute_index_map_backward_fake(
    permuted_tokens: torch.Tensor,
    grad_unpermuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference of backward."""
    # act_grad shape: (permuted_tokens.size(0), hidden_size)
    # prob_grad shape: (num_tokens, topK)
    topK = probs.size(1) if probs is not None and probs.numel() > 0 else 1
    num_tokens = (
        probs.size(0) if probs is not None and probs.numel() > 0 else sorted_indices.size(0)
    )
    act_grad = torch.empty(
        (permuted_tokens.size(0), grad_unpermuted_tokens.shape[1]),
        dtype=grad_unpermuted_tokens.dtype,
        device=grad_unpermuted_tokens.device,
    )
    prob_grad = torch.empty(
        (num_tokens, topK), dtype=torch.float32, device=grad_unpermuted_tokens.device
    )
    return act_grad, prob_grad


def _moe_unpermute_index_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, sorted_indices, probs, _num_tokens, _topK = inputs
    ctx.empty_input = inp.size(0) == 0
    ctx.save_for_backward(inp, sorted_indices, probs)
    ctx.needs_probs_grad = probs.requires_grad if probs is not None else False


def _moe_unpermute_index_map_backward_wrapper(ctx, unpermuted_act_grad):
    """Backward pass wrapper that calls the custom backward op."""
    if ctx.empty_input:
        prob_grad = torch.zeros_like(ctx.saved_tensors[2]) if ctx.needs_probs_grad else None
        return unpermuted_act_grad, None, prob_grad, None, None

    if not unpermuted_act_grad.is_contiguous():
        unpermuted_act_grad = unpermuted_act_grad.contiguous()

    permuted_tokens, sorted_indices, probs = ctx.saved_tensors

    act_grad, prob_grad = torch.ops.te_moe.unpermute_index_map_bwd(
        permuted_tokens, unpermuted_act_grad, sorted_indices, probs
    )

    if not ctx.needs_probs_grad:
        prob_grad = None

    return act_grad, None, prob_grad, None, None


moe_unpermute_index_map_forward.register_autograd(
    _moe_unpermute_index_map_backward_wrapper,
    setup_context=_moe_unpermute_index_map_setup_context,
)


# ===================== _moe_permute_mask_map custom ops =====================


@torch.library.custom_op("te_moe::permute_mask_map_fwd", mutates_args=[])
def moe_permute_mask_map_forward(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    probs: Optional[torch.Tensor],
    drop_and_pad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for MoE permute with mask router map."""
    if not inp.numel():
        # Empty input: return empty tensors with correct shapes
        # row_id_map and permuted_probs shapes should be [num_out_tokens]
        return (
            inp.clone(),
            torch.empty((num_out_tokens,), dtype=torch.int32, device=inp.device),
            torch.empty((num_out_tokens,), dtype=torch.float32, device=inp.device),
        )

    if not inp.is_npu:
        raise ValueError(f"inp must be a NPU tensor, but got tensor on {inp.device}.")
    if not routing_map.is_npu:
        raise ValueError(
            f"routing_map must be a NPU tensor, but got tensor on {routing_map.device}."
        )
    if probs is not None:
        if not probs.is_npu:
            raise ValueError(f"probs must be a NPU tensor, but got tensor on {probs.device}.")

    if inp.size(0) != routing_map.size(0):
        raise ValueError(
            f"Permute not possible: inp.size(0) ({inp.size(0)}) must match "
            f"routing_map.size(0) ({routing_map.size(0)})."
        )
    assert num_out_tokens > 0, (
        f"moe_permute (mask map) requires num_out_tokens > 0, got {num_out_tokens}. "
        "Use int(routing_map.sum()) or num_tokens * top_k."
    )

    inp, probs, original_dtype = _convert_tensors_to_fp32_if_needed(inp, probs)

    permuted_act, permuted_probs, row_id_map = torch_npu.npu_moe_token_permute_with_routing_map(
        inp, routing_map, probs=probs, num_out_tokens=num_out_tokens, drop_and_pad=drop_and_pad
    )

    if permuted_probs is None:
        permuted_probs = torch.empty(0, device=inp.device)

    permuted_act, permuted_probs = _restore_original_dtype(
        permuted_act, permuted_probs, original_dtype
    )

    return permuted_act, row_id_map, permuted_probs


@moe_permute_mask_map_forward.register_fake
def _moe_permute_mask_map_forward_fake(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    probs: Optional[torch.Tensor],
    drop_and_pad: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference."""
    num_tokens = inp.shape[0]
    hidden_size = inp.shape[1]
    if num_tokens > 0:
        assert num_out_tokens > 0, (
            f"moe_permute (mask map) requires num_out_tokens > 0, got {num_out_tokens}. "
            "Use int(routing_map.sum()) or num_tokens * top_k."
        )
        out_rows = num_out_tokens
    else:
        # Match `moe_permute_mask_map_forward` empty-input fast path (ignores num_out_tokens).
        out_rows = 0

    fake_row_id_map = torch.empty((out_rows,), dtype=torch.int32, device=inp.device)

    fake_output = torch.empty((out_rows, hidden_size), dtype=inp.dtype, device=inp.device)
    if probs is not None:
        fake_permuted_probs = (
            torch.empty((out_rows,), dtype=probs.dtype, device=inp.device)
            if out_rows > 0
            else torch.empty(0, device=inp.device)
        )
    else:
        fake_permuted_probs = torch.empty(0, device=inp.device)
    return fake_output, fake_row_id_map, fake_permuted_probs


@torch.library.custom_op("te_moe::permute_mask_map_bwd", mutates_args=[])
def moe_permute_mask_map_backward(
    permuted_token_out_grad: torch.Tensor,
    probs_grad: Optional[torch.Tensor],
    sorted_indices: torch.Tensor,
    routing_map: torch.Tensor,
    experts_num: int,
    tokens_num: int,
    drop_and_pad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE permute with mask router map."""
    # NPU backward operator: torch_npu.npu_moe_token_permute_with_routing_map_grad
    # Parameters: permuted_token_out_grad, probs_grad, sorted_indices, routing_map, experts_num, tokens_num, drop_and_pad
    act_grad, routing_map_grad = torch_npu.npu_moe_token_permute_with_routing_map_grad(
        permuted_token_out_grad,
        probs_grad,
        sorted_indices,
        routing_map,
        experts_num,
        tokens_num,
        drop_and_pad,
    )
    if routing_map_grad is None:
        routing_map_grad = torch.empty(0, device=permuted_token_out_grad.device)
    return act_grad, routing_map_grad


@moe_permute_mask_map_backward.register_fake
def _moe_permute_mask_map_backward_fake(
    permuted_token_out_grad: torch.Tensor,
    probs_grad: Optional[torch.Tensor],
    sorted_indices: torch.Tensor,
    routing_map: torch.Tensor,
    experts_num: int,
    tokens_num: int,
    drop_and_pad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake for backward shape inference."""
    hidden_size = permuted_token_out_grad.shape[1]
    act_grad = torch.empty(
        (tokens_num, hidden_size),
        dtype=permuted_token_out_grad.dtype,
        device=permuted_token_out_grad.device,
    )
    if probs_grad is not None:
        routing_map_grad = torch.empty(
            (tokens_num, experts_num),
            dtype=probs_grad.dtype,
            device=permuted_token_out_grad.device,
        )
    else:
        routing_map_grad = torch.empty(0, device=permuted_token_out_grad.device)
    return act_grad, routing_map_grad


def _moe_permute_mask_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, routing_map, _num_out_tokens, probs, drop_and_pad = inputs
    _output_tensor, sorted_indices, _permuted_probs = output
    ctx.empty_input = inp.size(0) == 0
    ctx.save_for_backward(sorted_indices, routing_map)
    ctx.num_experts = routing_map.size(1)
    ctx.num_tokens = inp.size(0)
    ctx.needs_probs_grad = probs is not None and probs.requires_grad
    ctx.drop_and_pad = drop_and_pad


def _moe_permute_mask_map_backward_wrapper(ctx, grad_output, grad_row_id_map, grad_permuted_probs):
    """Backward wrapper calling the custom backward op."""
    if ctx.empty_input:
        if ctx.needs_probs_grad:
            routing_map_grad = torch.zeros(
                (ctx.num_tokens, ctx.num_experts),
                dtype=grad_permuted_probs.dtype,
                device=grad_permuted_probs.device,
            )
        else:
            routing_map_grad = None
        return grad_output, None, None, routing_map_grad, None

    sorted_indices, routing_map = ctx.saved_tensors

    # Pass probs_grad only if it has content
    probs_grad_input = grad_permuted_probs if grad_permuted_probs.numel() > 0 else None

    act_grad, routing_map_grad = torch.ops.te_moe.permute_mask_map_bwd(
        grad_output,
        probs_grad_input,
        sorted_indices,
        routing_map,
        ctx.num_experts,
        ctx.num_tokens,
        ctx.drop_and_pad,
    )

    if not ctx.needs_probs_grad or routing_map_grad.numel() == 0:
        routing_map_grad = None

    return act_grad, None, None, routing_map_grad, None


moe_permute_mask_map_forward.register_autograd(
    _moe_permute_mask_map_backward_wrapper,
    setup_context=_moe_permute_mask_map_setup_context,
)


# ===================== _moe_unpermute_mask_map custom ops =====================


@torch.library.custom_op("te_moe::unpermute_mask_map_fwd", mutates_args=[])
def moe_unpermute_mask_map_forward(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    num_tokens: int,
    hidden_size: int,
    pad_offsets: Optional[torch.Tensor],
    routing_map: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for MoE unpermute with mask router map.

    Returns:
        unpermuted_output: The unpermuted tensor
        out_index: Index tensor needed for backward (when drop_and_pad=True)
        permuted_token_id: Token ID tensor needed for backward (when drop_and_pad=True)
    """
    if not inp.numel():
        # Empty input: return empty tensors with correct shapes
        # inp.shape[0] is the number of permuted tokens (could be 0 or non-zero depending on which dimension is 0)
        permute_token_size = inp.shape[0]
        return (
            inp.clone(),
            torch.empty((permute_token_size,), dtype=torch.int32, device=inp.device),
            torch.empty((permute_token_size,), dtype=torch.int32, device=inp.device),
        )

    drop_and_pad = pad_offsets is not None
    restore_shape = (num_tokens, hidden_size)

    # 使用 _npu_moe_token_unpermute_with_routing_map（参考 MindSpeed）
    # Parameters: permutedTokens, sortedIndices, restoreShape, probsOptional, routingMapOptional, dropAndPad
    # Returns: unpermutedTokens, outIndex, permuteTokenId, permuteProbs
    unpermuted_output, out_index, permuted_token_id, _ = (
        torch_npu._npu_moe_token_unpermute_with_routing_map(
            inp,
            row_id_map,
            restore_shape,
            probs=merging_probs,
            routing_map=routing_map,
            drop_and_pad=drop_and_pad,
        )
    )
    return unpermuted_output, out_index, permuted_token_id


@moe_unpermute_mask_map_forward.register_fake
def _moe_unpermute_mask_map_forward_fake(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    num_tokens: int,
    hidden_size: int,
    pad_offsets: Optional[torch.Tensor],
    routing_map: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference."""
    unpermuted_output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)

    permute_token_size = inp.shape[0]
    out_index = torch.empty((permute_token_size,), dtype=torch.int32, device=inp.device)
    permuted_token_id = torch.empty((permute_token_size,), dtype=torch.int32, device=inp.device)

    return unpermuted_output, out_index, permuted_token_id


@torch.library.custom_op("te_moe::unpermute_mask_map_bwd_with_probs", mutates_args=[])
def moe_unpermute_mask_map_backward_with_probs(
    unpermuted_tokens_grad: torch.Tensor,
    out_index: torch.Tensor,
    permuted_token_id: torch.Tensor,
    routing_map: Optional[torch.Tensor],
    permuted_tokens: Optional[torch.Tensor],
    probs: Optional[torch.Tensor],
    drop_and_pad: bool,
    restore_shape: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE unpermute with merging probs."""
    act_grad, probs_grad = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
        unpermuted_tokens_grad,
        out_index,
        permuted_token_id=permuted_token_id,
        routing_map=routing_map,
        permuted_tokens=permuted_tokens,
        probs=probs,
        drop_and_pad=drop_and_pad,
        restore_shape=restore_shape,
    )
    return act_grad, probs_grad


@moe_unpermute_mask_map_backward_with_probs.register_fake
def _moe_unpermute_mask_map_bwd_with_probs_fake(
    unpermuted_tokens_grad: torch.Tensor,
    out_index: torch.Tensor,
    permuted_token_id: torch.Tensor,
    routing_map: Optional[torch.Tensor],
    permuted_tokens: Optional[torch.Tensor],
    probs: Optional[torch.Tensor],
    drop_and_pad: bool,
    restore_shape: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake for backward shape inference with merging probs."""
    num_permuted_tokens = permuted_token_id.size(0) if permuted_token_id is not None else 0
    hidden_size = restore_shape[1] if restore_shape else unpermuted_tokens_grad.shape[1]
    num_tokens = restore_shape[0] if restore_shape else 0
    num_experts = probs.size(1) if probs is not None else 0

    act_grad = torch.empty(
        (num_permuted_tokens, hidden_size),
        dtype=unpermuted_tokens_grad.dtype,
        device=unpermuted_tokens_grad.device,
    )
    if probs is not None:
        probs_grad = torch.empty(
            (num_tokens, num_experts),
            dtype=probs.dtype,
            device=unpermuted_tokens_grad.device,
        )
    else:
        probs_grad = torch.empty(0, device=unpermuted_tokens_grad.device)
    return act_grad, probs_grad


@torch.library.custom_op("te_moe::unpermute_mask_map_bwd_no_probs", mutates_args=[])
def moe_unpermute_mask_map_backward_no_probs(
    unpermuted_tokens_grad: torch.Tensor,
    out_index: torch.Tensor,
    permuted_token_id: torch.Tensor,
    routing_map: Optional[torch.Tensor],
    drop_and_pad: bool,
    restore_shape: List[int],
) -> torch.Tensor:
    """Backward pass for MoE unpermute without merging probs (permute grad back)."""
    act_grad, _ = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
        unpermuted_tokens_grad,
        out_index,
        permuted_token_id,
        routing_map=routing_map,
        permuted_tokens=None,
        probs=None,
        drop_and_pad=drop_and_pad,
        restore_shape=restore_shape,
    )
    return act_grad


@moe_unpermute_mask_map_backward_no_probs.register_fake
def _moe_unpermute_mask_map_bwd_no_probs_fake(
    unpermuted_tokens_grad: torch.Tensor,
    out_index: torch.Tensor,
    permuted_token_id: torch.Tensor,
    routing_map: Optional[torch.Tensor],
    drop_and_pad: bool,
    restore_shape: List[int],
) -> torch.Tensor:
    """Fake for backward shape inference without probs."""
    num_permuted_tokens = permuted_token_id.size(0) if permuted_token_id is not None else 0
    hidden_size = restore_shape[1] if restore_shape else unpermuted_tokens_grad.shape[1]
    return torch.empty(
        (num_permuted_tokens, hidden_size),
        dtype=unpermuted_tokens_grad.dtype,
        device=unpermuted_tokens_grad.device,
    )


def _moe_unpermute_mask_map_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, sorted_indices, merging_probs, num_tokens, hidden_size, pad_offsets, routing_map = inputs
    unpermuted_output, out_index, permuted_token_id = output

    ctx.empty_input = inp.size(0) == 0
    ctx.restore_shape = (num_tokens, hidden_size)
    ctx.drop_and_pad = pad_offsets is not None
    ctx.with_probs = merging_probs is not None

    # Save tensors needed for backward
    if not ctx.empty_input:
        if ctx.with_probs:
            ctx.save_for_backward(
                inp,
                sorted_indices,
                merging_probs,
                pad_offsets,
                routing_map,
                out_index,
                permuted_token_id,
            )
            ctx.needs_probs_grad = merging_probs.requires_grad
        else:
            ctx.save_for_backward(
                sorted_indices, pad_offsets, routing_map, out_index, permuted_token_id
            )
            ctx.needs_probs_grad = False
    else:
        ctx.needs_probs_grad = False


def _moe_unpermute_mask_map_backward_wrapper(
    ctx, unpermuted_act_grad, grad_out_index, grad_permuted_token_id
):
    """Backward wrapper calling the appropriate custom backward op.

    Args:
        unpermuted_act_grad: Gradient w.r.t. unpermuted_output
        grad_out_index: Gradient w.r.t. out_index (always None, not used)
        grad_permuted_token_id: Gradient w.r.t. permuted_token_id (always None, not used)
    """
    if ctx.empty_input:
        if ctx.with_probs:
            _, _, merging_probs, _, _, _, _ = ctx.saved_tensors
            probs_grad = torch.zeros_like(merging_probs) if ctx.needs_probs_grad else None
            return unpermuted_act_grad, None, probs_grad, None, None, None, None
        return unpermuted_act_grad, None, None, None, None, None, None

    act_grad = None
    probs_grad = None

    if ctx.with_probs:
        (
            fwd_input,
            sorted_indices,
            merging_probs,
            pad_offsets,
            routing_map,
            out_index,
            permuted_token_id,
        ) = ctx.saved_tensors
        permuted_tokens = fwd_input
        probs = merging_probs
    else:
        sorted_indices, pad_offsets, routing_map, out_index, permuted_token_id = ctx.saved_tensors
        permuted_tokens = None
        probs = None

    if ctx.drop_and_pad:
        act_grad, probs_grad = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
            unpermuted_act_grad,
            out_index,
            permuted_token_id=permuted_token_id,
            routing_map=routing_map,
            permuted_tokens=permuted_tokens,
            probs=probs,
            drop_and_pad=ctx.drop_and_pad,
            restore_shape=ctx.restore_shape,
        )
    else:
        act_grad, probs_grad = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
            unpermuted_act_grad,
            sorted_indices,
            sorted_indices,
            routing_map=routing_map,
            permuted_tokens=permuted_tokens,
            probs=probs,
            drop_and_pad=ctx.drop_and_pad,
            restore_shape=ctx.restore_shape,
        )

    if not ctx.needs_probs_grad:
        probs_grad = None

    # Return gradients for 7 inputs: inp, row_id_map, merging_probs, num_tokens, hidden_size, pad_offsets, routing_map
    return act_grad, None, probs_grad, None, None, None, None


moe_unpermute_mask_map_forward.register_autograd(
    _moe_unpermute_mask_map_backward_wrapper,
    setup_context=_moe_unpermute_mask_map_setup_context,
)


# ===================== _moe_chunk_sort custom ops =====================


@torch.library.custom_op("te_moe::chunk_sort_fwd", mutates_args=[])
def moe_chunk_sort_forward(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for MoE chunk sort. Returns (output, permuted_probs, row_id_map)."""
    if not inp.numel():
        # Empty input: return empty tensors with correct shapes
        # output shape: [num_tokens, hidden_size]
        # permuted_probs shape: [num_tokens] (if probs is not None) or [0]
        # row_id_map shape: [num_tokens]
        num_tokens = inp.shape[0]
        hidden_size = inp.shape[1]
        probs_out = (
            torch.empty((num_tokens,), dtype=probs.dtype, device=inp.device)
            if probs is not None
            else torch.empty(0, device=inp.device)
        )
        return (
            torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device),
            probs_out,
            torch.empty((num_tokens,), dtype=torch.int32, device=inp.device),
        )

    num_tokens, hidden_size = inp.shape
    num_splits = split_sizes.size(0)

    # Keep Triton optional until a non-empty chunk sort is executed.
    import transformer_engine.pytorch.triton.sort_chunks_by_idx as triton_permutation

    row_id_map = triton_permutation.make_chunk_sort_map(
        split_sizes,
        sorted_idxs,
        num_tokens,
        num_splits,
    )
    output, permuted_probs = triton_permutation.sort_chunks_by_map(
        inp,
        row_id_map,
        probs,
        num_tokens,
        hidden_size,
        is_forward=True,
    )

    if permuted_probs is None:
        permuted_probs = torch.empty(0, device=output.device)

    return output, permuted_probs, row_id_map


@moe_chunk_sort_forward.register_fake
def _moe_chunk_sort_forward_fake(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake for shape inference."""
    num_tokens = inp.shape[0]
    hidden_size = inp.shape[1]
    fake_output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=inp.device)
    if probs is not None:
        fake_probs = torch.empty((num_tokens,), dtype=probs.dtype, device=inp.device)
    else:
        fake_probs = torch.empty(0, device=inp.device)
    # row_id_map: 1D, size num_tokens
    fake_row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device=inp.device)
    return fake_output, fake_probs, fake_row_id_map


@torch.library.custom_op("te_moe::chunk_sort_bwd", mutates_args=[])
def moe_chunk_sort_backward(
    permuted_act_grad: torch.Tensor,
    permuted_probs_grad: Optional[torch.Tensor],
    row_id_map: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for MoE chunk sort."""
    import transformer_engine.pytorch.triton.sort_chunks_by_idx as triton_permutation

    act_grad, probs_grad = triton_permutation.sort_chunks_by_map(
        permuted_act_grad,
        row_id_map,
        permuted_probs_grad,
        num_tokens,
        hidden_size,
        is_forward=False,
    )

    if probs_grad is None:
        probs_grad = torch.empty(0, device=act_grad.device)

    return act_grad, probs_grad


@moe_chunk_sort_backward.register_fake
def _moe_chunk_sort_backward_fake(
    permuted_act_grad: torch.Tensor,
    permuted_probs_grad: Optional[torch.Tensor],
    row_id_map: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake for backward shape inference."""
    fake_act_grad = torch.empty(
        (num_tokens, hidden_size),
        dtype=permuted_act_grad.dtype,
        device=permuted_act_grad.device,
    )
    if permuted_probs_grad is not None:
        fake_probs_grad = torch.empty(
            (num_tokens,),
            dtype=permuted_probs_grad.dtype,
            device=permuted_act_grad.device,
        )
    else:
        fake_probs_grad = torch.empty(0, device=permuted_act_grad.device)
    return fake_act_grad, fake_probs_grad


def _moe_chunk_sort_setup_context(ctx, inputs, output):
    """Save context for backward pass."""
    inp, _split_sizes, _sorted_idxs, probs = inputs
    _output_tensor, _permuted_probs, row_id_map = output
    ctx.empty_input = inp.size(0) == 0
    ctx.save_for_backward(row_id_map)
    ctx.num_tokens = inp.size(0)
    ctx.hidden_size = inp.size(1) if not ctx.empty_input else 0
    ctx.needs_probs_grad = probs is not None and probs.requires_grad


def _moe_chunk_sort_backward_wrapper(ctx, permuted_act_grad, permuted_probs_grad, _row_id_map_grad):
    """Backward wrapper calling the custom backward op."""
    if ctx.empty_input:
        probs_grad = permuted_probs_grad if ctx.needs_probs_grad else None
        return permuted_act_grad, None, None, probs_grad

    (row_id_map,) = ctx.saved_tensors

    probs_grad_input = permuted_probs_grad if permuted_probs_grad.numel() > 0 else None

    act_grad, probs_grad = torch.ops.te_moe.chunk_sort_bwd(
        permuted_act_grad,
        probs_grad_input,
        row_id_map,
        ctx.num_tokens,
        ctx.hidden_size,
    )

    if not ctx.needs_probs_grad or probs_grad.numel() == 0:
        probs_grad = None

    return act_grad, None, None, probs_grad


moe_chunk_sort_forward.register_autograd(
    _moe_chunk_sort_backward_wrapper,
    setup_context=_moe_chunk_sort_setup_context,
)


# ===================== Public API =====================


def moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    max_token_num: int = -1,
    map_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    routing_map : torch.Tensor
        The token to expert mapping tensor.
        If map_type is 'mask', routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
        If map_type is 'index', routing_map is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.
    num_out_tokens : int
        Number of output tokens (rows in the permuted buffer).
        mask map: must be > 0, e.g. int(routing_map.sum()) or num_tokens * top_k.
        index map: must be >= 0; 0 means infer as num_tokens * top_k.
    max_token_num : int, default = -1
        Workspace sizing hint, only used for map_type='index'. Ignored for 'mask'.

    map_type : str, default = 'mask'
        Type of the routing map tensor.
        Options are: 'mask', 'index'.
        Refer to `routing_map` for more details.
    """
    if map_type == "index":
        return torch.ops.te_moe.permute_index_map(inp, routing_map, num_out_tokens, max_token_num)
    if map_type == "mask":
        output, row_id_map, _ = torch.ops.te_moe.permute_mask_map_fwd(
            inp, routing_map, num_out_tokens, None, False
        )
        return output, row_id_map
    raise ValueError("map_type should be one of 'mask' or 'index'")


def moe_permute_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens and probs based on the routing_map.
    Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs : torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens, num_experts]. It will be permuted with the tokens
        according to the routing_map.
    routing_map : torch.Tensor
        The token to expert mapping tensor of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
    num_out_tokens : int
        Number of output tokens (rows in the permuted buffer). Must be > 0,
        e.g. int(routing_map.sum()) or num_tokens * top_k.
    """
    output, row_id_map, permuted_probs = torch.ops.te_moe.permute_mask_map_fwd(
        inp, routing_map, num_out_tokens, probs, False
    )
    return output, permuted_probs, row_id_map


def moe_permute_and_pad_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    align_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Permute the tokens and probs based on the routing_map.
    Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs: torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens, num_experts]. It will be permuted with the tokens
        according to the routing_map.
    routing_map: torch.Tensor
        The token to expert mapping tensor of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
    tokens_per_expert : torch.Tensor
        Tensor of shape `[num_experts]` containing actual token counts per expert.
    align_size : int
        the alignment size for the input tensor.
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

    # Calculate pad_offsets (cumulative padding offsets per expert). None when no
    # padding is needed, matching the NVTE semantics (NVTE permutation.py returns
    # pad_offsets=None if tokens_per_expert already equals the aligned target).
    if torch.equal(tokens_per_expert, target_tokens_per_expert):
        pad_offsets = None
    else:
        pad_lengths = target_tokens_per_expert - tokens_per_expert
        cum_pad = torch.cumsum(pad_lengths, dim=0)
        pad_offsets = torch.cat(
            [torch.zeros(1, dtype=cum_pad.dtype, device=inp.device), cum_pad[:-1]]
        )

    # Always use drop_and_pad=True for expert-clustered layout (needed for All-to-All)
    output, row_id_map, permuted_probs = torch.ops.te_moe.permute_mask_map_fwd(
        inp, routing_map, target_tokens_per_expert.sum().item(), probs, True
    )
    return output, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert


def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
    map_type: str = "mask",
    probs: Optional[torch.Tensor] = None,
    pad_offsets: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map : torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    merging_probs : torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape : torch.Size, default = None
        The output shape after the unpermute operation.
    map_type : str, default = 'mask'
        Type of the routing map tensor. Should be the same as the value passed to moe_permute.
        Options are: 'mask', 'index'.
    probs : torch.Tensor, default = None
        Renamed to merging_probs. Keep for backward compatibility.
    pad_offsets : torch.Tensor, default = None
        Tensor of per-expert cumulative padding offsets used to remove padding added
        during permutation. This is the fourth output of `moe_permute_and_pad_with_probs`
        and is required when unpermuting padded outputs.
    routing_map : torch.Tensor, default = None
        The routing map tensor used for backward pass. This is NPU-specific parameter
        and should be the same routing_map used in the corresponding moe_permute call.
    """
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    if map_type == "index":
        # Normalize probs
        if merging_probs is not None:
            if merging_probs.dtype != torch.float32:
                warnings.warn(
                    f"The data type of the input `probs` of Unpermute is {merging_probs.dtype}! "
                    "The recommended type is torch.float32."
                )
                merging_probs = merging_probs.to(torch.float32)
            num_tokens = merging_probs.size(0)
            topK = merging_probs.size(1)
        else:
            num_tokens = row_id_map.size(0)
            topK = 1
            merging_probs = None

        return torch.ops.te_moe.unpermute_index_map_fwd(
            inp, row_id_map, merging_probs, num_tokens, topK
        )
    if map_type == "mask":
        # Check that routing_map is provided when merging_probs is not None
        if merging_probs is not None and routing_map is None:
            raise ValueError("Mask must be provided to permute the probs. ")

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape

        unpermuted_output, _, _ = torch.ops.te_moe.unpermute_mask_map_fwd(
            inp,
            row_id_map,
            merging_probs,
            num_tokens,
            hidden_size,
            pad_offsets,
            routing_map,
        )
        return unpermuted_output
    raise ValueError("map_type should be one of 'mask' or 'index'")


def moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    split_sizes : torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices : torch.Tensor
        Chunk indices used to permute the chunks.
    """
    output, _, _ = torch.ops.te_moe.chunk_sort_fwd(inp, split_sizes, sorted_index, None)
    return output


def moe_sort_chunks_by_index_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor and probs based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    probs : torch.Tensor
        The tensor of probabilities corresponding to the permuted tokens and is
        of shape [num_tokens]. It will be permuted with the tokens according to
        the split_sizes and sorted_indices.
    split_sizes : torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices : torch.Tensor
        Chunk indices used to permute the chunks.
    """
    output, permuted_probs, _ = torch.ops.te_moe.chunk_sort_fwd(
        inp, split_sizes, sorted_index, probs
    )
    return output, permuted_probs
