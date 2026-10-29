# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Distributed correctness tests for Swap CP Attention and Swap Flash Attention.

The Swap Flash test runs with one NPU and covers both the full-sequence fast
path and chunked path. The Swap CP test runs with two NPUs, uses striped input
ownership, and compares forward plus dQ with a full-sequence causal FA
reference, including dQ/dK/dV gradients.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from distributed_testing import DistributedTest
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention.swap_cp_attn.swap_cp_attn_utils import (
    run_fa,
)
from transformer_engine.pytorch.attention.dot_product_attention.swap_cp_attn.swap_flash_attn import (
    SwapFlashAttention,
)
from utils import assert_close, npu_available


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_DTYPE = torch.bfloat16
_HEADS = 4
_HEAD_DIM = 16
_ATOL = 1e-2
_RTOL = 1e-2


def _rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _make_global_qkv(seq_len: int) -> tuple[torch.Tensor, ...]:
    shape = (seq_len, 1, _HEADS, _HEAD_DIM)
    base = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    scale = float(base.numel())
    return tuple((base + offset).div(scale).to(_DTYPE) for offset in range(3))


def _striped_positions(global_seq: int, local_seq: int, chunk_size: int, rank: int, world: int):
    positions = []
    for chunk_start in range(0, global_seq, chunk_size * world):
        rank_start = chunk_start + rank * chunk_size
        positions.append(torch.arange(rank_start, rank_start + chunk_size))
    result = torch.cat(positions).long()
    assert result.numel() == local_seq
    return result


def _reference_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return run_fa(q, k, v, _HEADS, True, None, _HEAD_DIM**-0.5)[0].view_as(q)


def _assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert_close(actual.cpu(), expected.cpu(), rtol=_RTOL, atol=_ATOL)


class TestSwapFlashAttention(DistributedTest):
    """Single-rank SwapFlashAttention correctness coverage."""

    world_size = 1
    reuse_dist_env = False

    @pytest.mark.parametrize("seq_chunk_size", [8, 32])
    def test_forward_backward_matches_flash_attention(self, seq_chunk_size: int) -> None:
        """Chunked and full-sequence paths match causal FA forward/backward."""
        rank = _rank()
        torch.npu.set_device(rank)
        device = torch.device(f"npu:{rank}")
        q_cpu, k_cpu, v_cpu = _make_global_qkv(32)

        q = q_cpu.to(device).detach().requires_grad_(True)
        k = k_cpu.to(device).detach().requires_grad_(True)
        v = v_cpu.to(device).detach().requires_grad_(True)
        q_ref = q_cpu.to(device).detach().requires_grad_(True)
        k_ref = k_cpu.to(device).detach().requires_grad_(True)
        v_ref = v_cpu.to(device).detach().requires_grad_(True)
        grad_output = torch.arange(q.numel(), dtype=torch.float32, device=device).reshape_as(q)
        grad_output = (grad_output / grad_output.numel()).to(_DTYPE)

        attention = SwapFlashAttention()
        output = attention(
            q,
            k,
            v,
            seq_chunk_size=seq_chunk_size,
            head_chunk_size=_HEADS,
            kv_chunk_cache_slots=1,
            compute_device=device,
            is_cpu_output=False,
            is_forward_only=False,
            q_chunk_cache_slots=1,
            backward_mode="high_accuracy",
        )
        reference = _reference_forward(q_ref, k_ref, v_ref)
        output.backward(grad_output)
        reference.backward(grad_output)

        _assert_tensor_close(output, reference)
        _assert_tensor_close(q.grad, q_ref.grad)
        _assert_tensor_close(k.grad, k_ref.grad)
        _assert_tensor_close(v.grad, v_ref.grad)


class TestSwapCPAttention(DistributedTest):
    """Two-rank Swap CP correctness coverage through DotProductAttention."""

    world_size = 2
    reuse_dist_env = False

    def test_forward_backward_matches_global_flash_attention(self) -> None:
        """Striped Swap CP output and all input gradients match global causal FA."""
        rank = _rank()
        world = dist.get_world_size()
        torch.npu.set_device(rank)
        device = torch.device(f"npu:{rank}")
        local_seq = 16
        global_seq = local_seq * world
        seq_chunk_size = 8
        q_cpu, k_cpu, v_cpu = _make_global_qkv(global_seq)
        positions = _striped_positions(global_seq, local_seq, seq_chunk_size, rank, world)
        positions_device = positions.to(device)

        q = q_cpu.index_select(0, positions).to(device).detach().requires_grad_(True)
        k = k_cpu.index_select(0, positions).to(device).detach().requires_grad_(True)
        v = v_cpu.index_select(0, positions).to(device).detach().requires_grad_(True)
        q_ref = q_cpu.to(device).detach().requires_grad_(True)
        k_ref = k_cpu.to(device).detach().requires_grad_(True)
        v_ref = v_cpu.to(device).detach().requires_grad_(True)
        grad_global = torch.arange(q_cpu.numel(), dtype=torch.float32).reshape_as(q_cpu)
        grad_global = (grad_global / grad_global.numel()).to(_DTYPE)
        grad_local = grad_global.index_select(0, positions).to(device)

        attention = DotProductAttention(
            num_attention_heads=_HEADS,
            kv_channels=_HEAD_DIM,
            num_gqa_groups=_HEADS,
            qkv_format="sbhd",
            attn_mask_type="causal",
            cp_group=dist.group.WORLD,
            cp_global_ranks=list(range(world)),
            cp_stream=torch.npu.Stream(),
            cp_comm_type="swapcpattn_algo",
            softmax_scale=_HEAD_DIM**-0.5,
        )
        output = attention(
            q,
            k,
            v,
            qkv_format="sbhd",
            attn_mask_type="causal",
            seq_chunk_size=seq_chunk_size,
            head_chunk_size=_HEADS,
            seq_layout_mode="striped",
            kv_chunk_cache_slots=1,
            q_chunk_cache_slots=1,
            backward_mode="high_accuracy",
        )
        reference = _reference_forward(q_ref, k_ref, v_ref)
        output.backward(grad_local)
        reference.backward(grad_global.to(device))

        expected_output = reference.detach().index_select(0, positions_device)
        expected_q_grad = q_ref.grad.detach().index_select(0, positions_device)
        expected_k_grad = k_ref.grad.detach().index_select(0, positions_device)
        expected_v_grad = v_ref.grad.detach().index_select(0, positions_device)
        _assert_tensor_close(output, expected_output)
        _assert_tensor_close(q.grad, expected_q_grad)
        _assert_tensor_close(k.grad, expected_k_grad)
        _assert_tensor_close(v.grad, expected_v_grad)
