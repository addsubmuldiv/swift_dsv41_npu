# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""THD (token-head-dim) context-parallel index utilities for NPU.

Mirrors ``thd_get_partitioned_indices`` from the CUDA extension
(``transformer_engine/common/fused_attn/context_parallel.cu``): every packed
sequence, padded to a multiple of ``2 * world_size`` tokens, is cut into
``2 * world_size`` equal chunks and rank ``r`` receives chunk ``r`` (from the
front) followed by the mirrored chunk ``2 * world_size - 1 - r`` of every
sequence, so that causal-attention work is balanced across context-parallel
ranks.
"""

from __future__ import annotations

import torch

__all__ = ["thd_get_partitioned_indices"]


def thd_get_partitioned_indices(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    """Compute the token indices assigned to one context-parallel rank.

    The concatenation of the results over all ranks in rank order defines the
    load-balanced global token layout that Transformer Engine's THD
    context-parallel attention kernels assume, so the ordering of the returned
    indices is part of the contract.

    Args:
        cu_seqlens: 1-D ``torch.int32`` cumulative sequence lengths of the
            padded THD pack, ``[batch + 1]`` with ``cu_seqlens[0] == 0``. Every
            entry must be divisible by ``2 * world_size``. Zero-length
            sequences contribute no tokens.
        total_tokens: total (padded) token count; must equal ``cu_seqlens[-1]``
            and be divisible by ``2 * world_size``.
        world_size: context-parallel world size, positive.
        rank: context-parallel rank in ``[0, world_size)``.

    Returns:
        1-D ``torch.int32`` tensor of length ``total_tokens // world_size`` on
        ``cu_seqlens.device``: indices into the token (first) dimension of the
        full padded THD tensor, ascending, covering chunk ``rank`` and chunk
        ``2 * world_size - 1 - rank`` of every sequence.
    """
    total_tokens = int(total_tokens)
    world_size = int(world_size)
    rank = int(rank)

    if not isinstance(cu_seqlens, torch.Tensor):
        raise ValueError(f"cu_seqlens must be a torch.Tensor, got {type(cu_seqlens)}.")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}.")
    if cu_seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens must be 1-D, got {cu_seqlens.dim()}-D.")
    if cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens needs at least 2 entries, got {cu_seqlens.numel()}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}.")
    if total_tokens % (2 * world_size) != 0:
        raise ValueError(
            f"total_tokens must be divisible by 2 * world_size = {2 * world_size}, "
            f"got {total_tokens}."
        )
    # One small host sync per call; cu_seqlens is [batch + 1] entries.
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError(
            f"total_tokens ({total_tokens}) must match cu_seqlens[-1] "
            f"({int(cu_seqlens[-1].item())})."
        )
    if bool((cu_seqlens % (2 * world_size)).any().item()):
        raise ValueError(
            f"every cu_seqlens entry must be divisible by 2 * world_size = {2 * world_size}."
        )

    device = cu_seqlens.device
    # int64 internally: repeat_interleave wants matching dtypes and the output
    # is cast back to int32 below.
    cu = cu_seqlens.to(torch.int64)
    lengths = cu[1:] - cu[:-1]
    per_seq_take = lengths // world_size  # tokens this rank takes from each sequence (2 chunks)

    take = total_tokens // world_size
    seq_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), per_seq_take)
    seq_starts = torch.cumsum(per_seq_take, dim=0) - per_seq_take  # exclusive cumsum
    # Local index within the rank's take of its sequence, in [0, 2 * chunk).
    local = torch.arange(take, device=device) - seq_starts[seq_ids]
    chunk = lengths[seq_ids] // (2 * world_size)
    is_front = local < chunk
    # Front half: chunk `rank`; mirrored half: chunk `2 * world_size - 1 - rank`.
    chunk_id = rank + (2 * world_size - 1 - 2 * rank) * (~is_front).to(torch.int64)
    out = cu[seq_ids] + chunk * chunk_id + local % chunk
    return out.to(torch.int32)
