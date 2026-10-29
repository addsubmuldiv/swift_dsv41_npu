# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for ``transformer_engine_torch.thd_get_partitioned_indices``.

Golden cases are derived from the CUDA kernel in
``transformer_engine/common/fused_attn/context_parallel.cu`` (rank ``r`` takes
chunk ``r`` and the mirrored chunk ``2 * world_size - 1 - r`` of every packed
sequence); the reference implementation mirrors the per-sequence zigzag split
that Megatron-LM uses as ground truth in ``test_thd_correctness.py``.
"""

from __future__ import annotations

import pytest
import torch

from utils import npu_available

from transformer_engine_torch import thd_get_partitioned_indices

# (cu_seqlens, world_size, {rank: expected indices})
GOLDEN_CASES = [
    # Two equal sequences of length 8, W=2 (2W=4 chunks of 2 tokens each).
    ([0, 8, 16], 2, {0: [0, 1, 6, 7, 8, 9, 14, 15], 1: [2, 3, 4, 5, 10, 11, 12, 13]}),
    # Variable lengths (4 and 8).
    ([0, 4, 12], 2, {0: [0, 3, 4, 5, 10, 11], 1: [1, 2, 6, 7, 8, 9]}),
    # Empty middle sequence is skipped, same as the case above.
    ([0, 4, 4, 12], 2, {0: [0, 3, 4, 5, 10, 11], 1: [1, 2, 6, 7, 8, 9]}),
    # W=4 (8 chunks of 1 token per sequence).
    (
        [0, 8, 16],
        4,
        {
            0: [0, 7, 8, 15],
            1: [1, 6, 9, 14],
            2: [2, 5, 10, 13],
            3: [3, 4, 11, 12],
        },
    ),
    # W=1: rank 0 owns the entire pack.
    ([0, 4, 12], 1, {0: list(range(12))}),
]


def _reference_indices(cu_seqlens: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """Independent per-sequence reference (Megatron ``_zigzag_split`` style)."""
    indices = []
    for seq in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[seq])
        length = int(cu_seqlens[seq + 1]) - start
        chunk = length // (2 * world_size)
        indices.extend(range(start + rank * chunk, start + (rank + 1) * chunk))
        mirror = 2 * world_size - 1 - rank
        indices.extend(range(start + mirror * chunk, start + (mirror + 1) * chunk))
    return torch.tensor(indices, dtype=torch.int32)


def _random_aligned_cu(batch: int, world_size: int, seed: int) -> torch.Tensor:
    """Random cu_seqlens whose lengths are multiples of 2*world_size (0 allowed)."""
    gen = torch.Generator().manual_seed(seed)
    while True:
        lengths = torch.randint(0, 5, (batch,), generator=gen) * (2 * world_size)
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)]
        )
        if int(cu_seqlens[-1]) > 0:
            return cu_seqlens


@pytest.mark.parametrize("cu_seqlens,world_size,expected", GOLDEN_CASES)
def test_golden_kernel_examples(cu_seqlens, world_size, expected):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    total = int(cu[-1])
    for rank, want in expected.items():
        out = thd_get_partitioned_indices(cu, total, world_size, rank)
        assert out.dtype == torch.int32
        assert out.dim() == 1
        assert out.numel() == total // world_size
        assert torch.equal(out, torch.tensor(want, dtype=torch.int32))


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("batch", [1, 3, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_random(world_size, batch, seed):
    cu = _random_aligned_cu(batch, world_size, seed)
    total = int(cu[-1])
    for rank in range(world_size):
        out = thd_get_partitioned_indices(cu, total, world_size, rank)
        assert torch.equal(out, _reference_indices(cu, world_size, rank))


@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("seed", [3, 4])
def test_partition_invariants(world_size, seed):
    cu = _random_aligned_cu(5, world_size, seed)
    total = int(cu[-1])
    parts = [thd_get_partitioned_indices(cu, total, world_size, r) for r in range(world_size)]
    for part in parts:
        assert part.dtype == torch.int32
        assert part.numel() == total // world_size
        assert part.device == cu.device
    # Deterministic ordering: the LB layout is re-derived identically by
    # callers (mamba undo/redo), so repeated calls must agree exactly.
    for rank in range(world_size):
        assert torch.equal(parts[rank], thd_get_partitioned_indices(cu, total, world_size, rank))
    # Ranks partition the token axis exactly: disjoint, complete, ascending.
    merged = torch.sort(torch.cat(parts)).values
    assert torch.equal(merged, torch.arange(total, dtype=torch.int32))


def test_load_balanced_layout_round_trip():
    """Replicates the mamba/GDN undo/redo pattern that consumes the indices."""
    world_size, seed = 4, 5
    cu = _random_aligned_cu(5, world_size, seed)
    total = int(cu[-1])
    tokens = torch.randn(total, 3, 8)

    # _redo_attention_load_balancing: gather per-rank slices into LB order.
    index = torch.empty(total, dtype=torch.int32)
    per_rank = total // world_size
    for rank in range(world_size):
        index[rank * per_rank : (rank + 1) * per_rank] = thd_get_partitioned_indices(
            cu, total, world_size, rank
        )
    lb = tokens.index_select(0, index)

    # _undo_attention_load_balancing: scatter back to sequential order.
    restored = torch.empty_like(tokens)
    for rank in range(world_size):
        part = thd_get_partitioned_indices(cu, total, world_size, rank)
        restored[part] = lb[rank * per_rank : (rank + 1) * per_rank]
    assert torch.equal(restored, tokens)


@pytest.mark.parametrize(
    "cu_seqlens,total,world_size,rank",
    [
        ([0, 8], 10, 2, 0),  # total not divisible by 2*world_size
        ([0, 8], 12, 2, 0),  # total != cu_seqlens[-1]
        ([0, 5, 12], 12, 2, 0),  # cu_seqlens entry not divisible by 2*world_size
        ([0, 8], 8, 2, -1),  # negative rank
        ([0, 8], 8, 2, 2),  # rank == world_size
        ([0, 8], 8, 0, 0),  # non-positive world_size
        ([0, 8], 0, 2, 0),  # non-positive total
    ],
)
def test_validation_errors(cu_seqlens, total, world_size, rank):
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    with pytest.raises(ValueError):
        thd_get_partitioned_indices(cu, total, world_size, rank)


def test_validation_rejects_bad_tensor_inputs():
    with pytest.raises(ValueError):  # wrong dtype
        thd_get_partitioned_indices(torch.tensor([0, 8], dtype=torch.int64), 8, 2, 0)
    with pytest.raises(ValueError):  # wrong ndim
        thd_get_partitioned_indices(torch.zeros(1, 2, dtype=torch.int32), 8, 2, 0)
    with pytest.raises(ValueError):  # fewer than 2 entries
        thd_get_partitioned_indices(torch.tensor([0], dtype=torch.int32), 8, 2, 0)
    with pytest.raises(ValueError):  # not a tensor
        thd_get_partitioned_indices([0, 8], 8, 2, 0)


@pytest.mark.skipif(not npu_available(), reason="NPU device is required")
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("seed", [6, 7])
def test_npu_device_matches_reference(world_size, seed):
    cu = _random_aligned_cu(4, world_size, seed).to("npu")
    total = int(cu[-1])
    for rank in range(world_size):
        out = thd_get_partitioned_indices(cu, total, world_size, rank)
        assert out.device.type == "npu"
        assert out.dtype == torch.int32
        assert torch.equal(out.cpu(), _reference_indices(cu.cpu(), world_size, rank))
