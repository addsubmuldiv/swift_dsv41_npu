# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for RoPE (Rotary Position Embedding) on NPU.

Covers:
  - RotaryPositionEmbedding module
  - apply_rotary_pos_emb (fused & unfused, sbhd/bshd/thd)
  - apply_fused_qkv_rotary_pos_emb
  - _rotate_half / _rotate_half_thd
  - _npu_rotary_mul wrapper
  - _get_freqs_on_this_cp_rank
"""

from typing import Union, List

import pytest
import torch
import torch_npu  # noqa: F401

from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    apply_fused_qkv_rotary_pos_emb,
    _rotate_half,
    _rotate_half_thd,
    _npu_rotary_mul,
    _get_freqs_on_this_cp_rank,
)


def _seed() -> None:
    seed = 42
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def _device() -> torch.device:
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


def _overlapping_grad(output: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(output, List):
        return sum(t.sum() * 2 for t in output)
    return output.sum() * 2


def _non_overlapping_grad(output: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(output, List):
        return sum(torch.sum(t * torch.ones_like(t)) for t in output)
    return torch.sum(output * torch.ones_like(output))


def _fwd_atol_rtol(dtype):
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 1e-3, 1e-3


def _bwd_atol_rtol(dtype):
    return 1e-5, 1e-5


# ===================== _rotate_half / _rotate_half_thd =====================


class TestRotateHalf:
    def test_half_mode_4d(self):
        x = torch.randn(2, 3, 4, 8)
        out = _rotate_half(x, interleaved=False)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        expected = torch.cat((-x2, x1), dim=-1)
        torch.testing.assert_close(out, expected)

    def test_interleave_mode_4d(self):
        x = torch.randn(2, 3, 4, 8)
        out = _rotate_half(x, interleaved=True)
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        expected = torch.stack((-x2, x1), dim=-1).view(x.shape)
        torch.testing.assert_close(out, expected)


class TestRotateHalfThd:
    def test_half_mode_3d(self):
        x = torch.randn(6, 4, 8)
        out = _rotate_half_thd(x, interleaved=False)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        expected = torch.cat((-x2, x1), dim=-1)
        torch.testing.assert_close(out, expected)

    def test_interleave_mode_3d(self):
        x = torch.randn(6, 4, 8)
        out = _rotate_half_thd(x, interleaved=True)
        x1 = x[:, :, ::2]
        x2 = x[:, :, 1::2]
        expected = torch.stack((-x2, x1), dim=-1).view(x.shape)
        torch.testing.assert_close(out, expected)


# ===================== _npu_rotary_mul =====================


class TestNpuRotaryMul:
    def test_half_mode_sbhd(self):
        dev = _device()
        x = torch.randn(8, 2, 4, 64, device=dev, dtype=torch.float16)
        freqs = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        cos_ = torch.cos(freqs).to(torch.float16)
        sin_ = torch.sin(freqs).to(torch.float16)
        out = _npu_rotary_mul(x.contiguous(), cos_, sin_, interleaved=False)
        assert out.shape == x.shape

        x_cpu = x.cpu().float()
        cos_cpu = cos_.cpu().float()
        sin_cpu = sin_.cpu().float()
        x1, x2 = torch.chunk(x_cpu, 2, dim=-1)
        expected = (x_cpu * cos_cpu) + (torch.cat((-x2, x1), dim=-1) * sin_cpu)
        torch.testing.assert_close(out.cpu().float(), expected, atol=1e-3, rtol=1e-3)

    def test_interleave_mode_sbhd(self):
        dev = _device()
        x = torch.randn(8, 2, 4, 64, device=dev, dtype=torch.float16)
        freqs = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        cos_ = torch.cos(freqs).to(torch.float16)
        sin_ = torch.sin(freqs).to(torch.float16)
        out = _npu_rotary_mul(x.contiguous(), cos_, sin_, interleaved=True)
        assert out.shape == x.shape

    def test_bshd_layout(self):
        dev = _device()
        x = torch.randn(2, 8, 4, 64, device=dev, dtype=torch.float16)
        freqs = torch.randn(1, 8, 1, 64, device=dev, dtype=torch.float32)
        cos_ = torch.cos(freqs).to(torch.float16)
        sin_ = torch.sin(freqs).to(torch.float16)
        out = _npu_rotary_mul(x.contiguous(), cos_, sin_, interleaved=False)
        assert out.shape == x.shape

    def test_backward(self):
        dev = _device()
        x = torch.randn(8, 2, 4, 64, device=dev, dtype=torch.float16, requires_grad=True)
        freqs = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        cos_ = torch.cos(freqs).to(torch.float16)
        sin_ = torch.sin(freqs).to(torch.float16)
        out = _npu_rotary_mul(x.contiguous(), cos_, sin_, interleaved=False)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_bfloat16(self):
        dev = _device()
        x = torch.randn(8, 2, 4, 64, device=dev, dtype=torch.bfloat16)
        freqs = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        cos_ = torch.cos(freqs).to(torch.bfloat16)
        sin_ = torch.sin(freqs).to(torch.bfloat16)
        out = _npu_rotary_mul(x.contiguous(), cos_, sin_, interleaved=False)
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16


# ===================== _get_freqs_on_this_cp_rank =====================


class TestGetFreqsOnThisCpRank:
    def test_cp_size_1(self):
        freqs = torch.randn(16, 1, 1, 64)
        out = _get_freqs_on_this_cp_rank(freqs, seqlen=8, cp_size=1, cp_rank=0)
        assert out.shape == (8, 1, 1, 64)
        torch.testing.assert_close(out, freqs[:8])

    def test_cp_size_2_rank0(self):
        freqs = torch.randn(16, 1, 1, 64)
        out = _get_freqs_on_this_cp_rank(freqs, seqlen=8, cp_size=2, cp_rank=0)
        assert out.shape == (8, 1, 1, 64)
        expected = torch.cat([freqs[0:4], freqs[12:16]])
        torch.testing.assert_close(out, expected)

    def test_cp_size_2_rank1(self):
        freqs = torch.randn(16, 1, 1, 64)
        out = _get_freqs_on_this_cp_rank(freqs, seqlen=8, cp_size=2, cp_rank=1)
        assert out.shape == (8, 1, 1, 64)
        expected = torch.cat([freqs[4:8], freqs[8:12]])
        torch.testing.assert_close(out, expected)


# ===================== RotaryPositionEmbedding =====================


class TestRotaryPositionEmbedding:
    def test_output_shape(self):
        dev = _device()
        rope = RotaryPositionEmbedding(dim=128, rotary_base=10000.0).to(dev)
        emb = rope(max_seq_len=16)
        assert emb.shape == (16, 1, 1, 128)

    def test_rotary_percent(self):
        dev = _device()
        rope = RotaryPositionEmbedding(dim=128, rotary_percent=0.5).to(dev)
        emb = rope(max_seq_len=16)
        assert emb.shape == (16, 1, 1, 64)

    def test_interleaved(self):
        dev = _device()
        rope = RotaryPositionEmbedding(dim=128, interleaved=True).to(dev)
        emb = rope(max_seq_len=16)
        assert emb.shape == (16, 1, 1, 128)

    def test_autocast_same_result(self):
        dev = _device()
        rope = RotaryPositionEmbedding(128).to(dev)
        emb_no_autocast = rope(max_seq_len=1024)
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            emb_autocast = rope(max_seq_len=1024)
        torch.testing.assert_close(
            emb_no_autocast.to(dtype=torch.bfloat16),
            emb_autocast.to(dtype=torch.bfloat16),
            atol=1e-8,
            rtol=1e-8,
        )

    def test_inv_freq_on_npu(self):
        dev = _device()
        rope = RotaryPositionEmbedding(dim=64).to(dev)
        assert rope.inv_freq.device.type == "npu"

    def test_seq_len_interpolation(self):
        dev = _device()
        rope = RotaryPositionEmbedding(
            dim=64, seq_len_interpolation_factor=2, pretrained_max_position_embeddings=512
        ).to(dev)
        emb = rope(max_seq_len=256)
        assert emb.shape == (256, 1, 1, 64)


# ===================== apply_rotary_pos_emb (fused vs unfused) =====================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
class TestFusedRoPESbhdBshd:
    def test_fused_matches_unfused(self, dtype, tensor_format, interleaved, loss_func):
        dev = _device()
        _seed()
        seq_length, batch_size, head_num, hidden_size = 128, 2, 4, 64
        fwd_atol, fwd_rtol = _fwd_atol_rtol(dtype)
        bwd_atol, bwd_rtol = _bwd_atol_rtol(dtype)

        t = torch.rand(
            (seq_length, batch_size, head_num, hidden_size),
            dtype=dtype,
            device=dev,
        )
        if tensor_format == "bshd":
            t = t.transpose(0, 1).contiguous()
        t.requires_grad = True

        rope = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb = rope(seq_length)

        output_unfused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format=tensor_format,
            interleaved=interleaved,
            fused=False,
        )
        loss_unfused = loss_func(output_unfused)
        loss_unfused.backward()
        grad_unfused = t.grad.detach().clone()
        t.grad = None

        output_fused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format=tensor_format,
            interleaved=interleaved,
            fused=True,
        )
        loss_fused = loss_func(output_fused)
        loss_fused.backward()
        grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(output_fused, output_unfused, atol=fwd_atol, rtol=fwd_rtol)
        torch.testing.assert_close(grad_fused, grad_unfused, atol=bwd_atol, rtol=bwd_rtol)

    def test_fused_with_start_positions(self, dtype, tensor_format, interleaved, loss_func):
        dev = _device()
        _seed()
        seq_length, batch_size, head_num, hidden_size = 64, 2, 4, 64
        margin = 10
        fwd_atol, fwd_rtol = _fwd_atol_rtol(dtype)
        bwd_atol, bwd_rtol = _bwd_atol_rtol(dtype)

        t = torch.rand(
            (seq_length - margin, batch_size, head_num, hidden_size),
            dtype=dtype,
            device=dev,
        )
        if tensor_format == "bshd":
            t = t.transpose(0, 1).contiguous()
        t.requires_grad = True

        start_positions = torch.randint(0, margin, (batch_size,), dtype=torch.int32, device=dev)

        rope = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb = rope(seq_length)

        output_unfused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=False,
        )
        loss_unfused = loss_func(output_unfused)
        loss_unfused.backward()
        grad_unfused = t.grad.detach().clone()
        t.grad = None

        output_fused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=True,
        )
        loss_fused = loss_func(output_fused)
        loss_fused.backward()
        grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(output_fused, output_unfused, atol=fwd_atol, rtol=fwd_rtol)
        torch.testing.assert_close(grad_fused, grad_unfused, atol=bwd_atol, rtol=bwd_rtol)

    def test_fused_with_cp(self, dtype, tensor_format, interleaved, loss_func):
        dev = _device()
        _seed()
        seq_length, batch_size, head_num, hidden_size = 128, 2, 4, 64
        cp_size = 2
        fwd_atol, fwd_rtol = _fwd_atol_rtol(dtype)
        bwd_atol, bwd_rtol = _bwd_atol_rtol(dtype)

        t = torch.rand(
            (seq_length, batch_size, head_num, hidden_size),
            dtype=dtype,
            device=dev,
        )
        if tensor_format == "bshd":
            t = t.transpose(0, 1).contiguous()
        t.requires_grad = True

        rope = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb = rope(seq_length * cp_size)

        for cp_rank in range(cp_size):
            t_grad_holder = t.detach().clone().requires_grad_(True)

            output_unfused = apply_rotary_pos_emb(
                t_grad_holder,
                emb,
                tensor_format=tensor_format,
                interleaved=interleaved,
                fused=False,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
            loss_unfused = loss_func(output_unfused)
            loss_unfused.backward()
            grad_unfused = t_grad_holder.grad.detach().clone()

            t_grad_holder2 = t.detach().clone().requires_grad_(True)
            output_fused = apply_rotary_pos_emb(
                t_grad_holder2,
                emb,
                tensor_format=tensor_format,
                interleaved=interleaved,
                fused=True,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
            loss_fused = loss_func(output_fused)
            loss_fused.backward()
            grad_fused = t_grad_holder2.grad.detach().clone()

            torch.testing.assert_close(output_fused, output_unfused, atol=fwd_atol, rtol=fwd_rtol)
            torch.testing.assert_close(grad_fused, grad_unfused, atol=bwd_atol, rtol=bwd_rtol)


# ===================== apply_rotary_pos_emb thd =====================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
class TestFusedRoPEThd:
    def test_fused_thd_matches_unfused(self, dtype, interleaved, loss_func):
        dev = _device()
        _seed()
        head_num, hidden_size = 4, 64
        fwd_atol, fwd_rtol = _fwd_atol_rtol(dtype)
        bwd_atol, bwd_rtol = _bwd_atol_rtol(dtype)
        cu_seqlens = torch.tensor([0, 3, 8, 12, 20], dtype=torch.int32, device=dev)
        total_tokens = cu_seqlens[-1].item()

        t = torch.rand((total_tokens, head_num, hidden_size), dtype=dtype, device=dev)
        t.requires_grad = True

        rope = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb = rope(total_tokens)

        output_unfused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format="thd",
            interleaved=interleaved,
            fused=False,
            cu_seqlens=cu_seqlens,
        )
        loss_unfused = loss_func(output_unfused)
        loss_unfused.backward()
        grad_unfused = t.grad.detach().clone()
        t.grad = None

        output_fused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format="thd",
            interleaved=interleaved,
            fused=True,
            cu_seqlens=cu_seqlens,
        )
        loss_fused = loss_func(output_fused)
        loss_fused.backward()
        grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(output_fused, output_unfused, atol=fwd_atol, rtol=fwd_rtol)
        torch.testing.assert_close(grad_fused, grad_unfused, atol=bwd_atol, rtol=bwd_rtol)

    def test_fused_thd_with_start_positions(self, dtype, interleaved, loss_func):
        dev = _device()
        _seed()
        head_num, hidden_size = 4, 64
        fwd_atol, fwd_rtol = _fwd_atol_rtol(dtype)
        bwd_atol, bwd_rtol = _bwd_atol_rtol(dtype)
        cu_seqlens = torch.tensor([0, 5, 10, 15, 20], dtype=torch.int32, device=dev)
        total_tokens = cu_seqlens[-1].item()
        num_seqs = len(cu_seqlens) - 1

        t = torch.rand((total_tokens, head_num, hidden_size), dtype=dtype, device=dev)
        t.requires_grad = True

        start_positions = torch.randint(0, 5, (num_seqs,), dtype=torch.int32, device=dev)

        rope = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb = rope(total_tokens + 5)

        output_unfused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format="thd",
            start_positions=start_positions,
            interleaved=interleaved,
            fused=False,
            cu_seqlens=cu_seqlens,
        )
        loss_unfused = loss_func(output_unfused)
        loss_unfused.backward()
        grad_unfused = t.grad.detach().clone()
        t.grad = None

        output_fused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format="thd",
            start_positions=start_positions,
            interleaved=interleaved,
            fused=True,
            cu_seqlens=cu_seqlens,
        )
        loss_fused = loss_func(output_fused)
        loss_fused.backward()
        grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(output_fused, output_unfused, atol=fwd_atol, rtol=fwd_rtol)
        torch.testing.assert_close(grad_fused, grad_unfused, atol=bwd_atol, rtol=bwd_rtol)

    def test_fused_thd_with_cp(self, dtype, interleaved, loss_func):
        dev = _device()
        _seed()
        head_num, hidden_size = 4, 64
        cp_size = 2
        fwd_atol, fwd_rtol = _fwd_atol_rtol(dtype)
        bwd_atol, bwd_rtol = _bwd_atol_rtol(dtype)
        cu_seqlens = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=dev)
        total_tokens = cu_seqlens[-1].item()

        rope = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb = rope(total_tokens * cp_size)

        for cp_rank in range(cp_size):
            t = torch.rand(
                (total_tokens // cp_size, head_num, hidden_size), dtype=dtype, device=dev
            )
            t.requires_grad = True

            output_unfused = apply_rotary_pos_emb(
                t,
                emb,
                tensor_format="thd",
                interleaved=interleaved,
                fused=False,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
            loss_unfused = loss_func(output_unfused)
            loss_unfused.backward()
            grad_unfused = t.grad.detach().clone()

            t2 = t.detach().clone().requires_grad_(True)
            output_fused = apply_rotary_pos_emb(
                t2,
                emb,
                tensor_format="thd",
                interleaved=interleaved,
                fused=True,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
            loss_fused = loss_func(output_fused)
            loss_fused.backward()
            grad_fused = t2.grad.detach().clone()

            torch.testing.assert_close(output_fused, output_unfused, atol=fwd_atol, rtol=fwd_rtol)
            torch.testing.assert_close(grad_fused, grad_unfused, atol=bwd_atol, rtol=bwd_rtol)


# ===================== apply_fused_qkv_rotary_pos_emb =====================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
class TestFusedQKVRoPE:
    def test_qkv_rope_matches_separate_rope(self, dtype, tensor_format, interleaved, loss_func):
        dev = _device()
        _seed()
        seq_length, batch_size, head_num, hidden_size = 64, 2, 4, 64
        fwd_atol, fwd_rtol = _fwd_atol_rtol(dtype)
        bwd_atol, bwd_rtol = _bwd_atol_rtol(dtype)

        qkv = torch.rand(
            (seq_length, batch_size, head_num, hidden_size * 3),
            dtype=dtype,
            device=dev,
        )
        if tensor_format == "bshd":
            qkv = qkv.transpose(0, 1).contiguous()
        qkv.requires_grad = True

        rope_q = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_q = rope_q(seq_length)
        rope_k = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_k = rope_k(seq_length)

        out_fused = apply_fused_qkv_rotary_pos_emb(
            qkv,
            emb_q,
            emb_k,
            qkv_split_arg_list=[hidden_size, hidden_size, hidden_size],
            tensor_format=tensor_format,
            interleaved=interleaved,
        )
        q_fused, k_fused, v_fused = out_fused
        loss_fused = loss_func(list(out_fused))
        loss_fused.backward()
        grad_fused = qkv.grad.detach().clone()
        qkv.grad = None

        q, k, v = torch.split(qkv.detach().clone(), [hidden_size, hidden_size, hidden_size], dim=-1)
        q.requires_grad = True
        k.requires_grad = True

        q_out = apply_rotary_pos_emb(
            q,
            emb_q,
            tensor_format=tensor_format,
            interleaved=interleaved,
            fused=True,
        )
        k_out = apply_rotary_pos_emb(
            k,
            emb_k,
            tensor_format=tensor_format,
            interleaved=interleaved,
            fused=True,
        )
        out_separate = torch.cat([q_out, k_out, v], dim=-1)
        loss_separate = loss_func(out_separate)
        loss_separate.backward()

        torch.testing.assert_close(q_fused, q_out.detach(), atol=fwd_atol, rtol=fwd_rtol)
        torch.testing.assert_close(k_fused, k_out.detach(), atol=fwd_atol, rtol=fwd_rtol)
        torch.testing.assert_close(v_fused, v)

        grad_q, grad_k, _ = torch.split(grad_fused, [hidden_size, hidden_size, hidden_size], dim=-1)
        torch.testing.assert_close(grad_q, q.grad, atol=bwd_atol, rtol=bwd_rtol)
        torch.testing.assert_close(grad_k, k.grad, atol=bwd_atol, rtol=bwd_rtol)

    def test_qkv_rope_with_start_positions(self, dtype, tensor_format, interleaved, loss_func):
        dev = _device()
        _seed()
        seq_length, batch_size, head_num, hidden_size = 64, 2, 4, 64
        margin = 5

        qkv = torch.rand(
            (seq_length - margin, batch_size, head_num, hidden_size * 3),
            dtype=dtype,
            device=dev,
        )
        if tensor_format == "bshd":
            qkv = qkv.transpose(0, 1).contiguous()
        qkv.requires_grad = True

        start_positions = torch.randint(0, margin, (batch_size,), dtype=torch.int32, device=dev)

        rope_q = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_q = rope_q(seq_length + margin)
        rope_k = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_k = rope_k(seq_length + margin)

        out = apply_fused_qkv_rotary_pos_emb(
            qkv,
            emb_q,
            emb_k,
            qkv_split_arg_list=[hidden_size, hidden_size, hidden_size],
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
        )
        q_out, k_out, v_out = out
        assert q_out.shape == qkv.shape[:3] + (hidden_size,)
        assert k_out.shape == qkv.shape[:3] + (hidden_size,)
        assert v_out.shape == qkv.shape[:3] + (hidden_size,)
        loss = loss_func(list(out))
        loss.backward()
        assert qkv.grad is not None

    def test_qkv_rope_with_cp(self, dtype, tensor_format, interleaved, loss_func):
        dev = _device()
        _seed()
        seq_length, batch_size, head_num, hidden_size = 64, 2, 4, 64
        cp_size = 2

        rope_q = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_q = rope_q(seq_length * cp_size)
        rope_k = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_k = rope_k(seq_length * cp_size)

        for cp_rank in range(cp_size):
            qkv = torch.rand(
                (seq_length, batch_size, head_num, hidden_size * 3),
                dtype=dtype,
                device=dev,
            )
            if tensor_format == "bshd":
                qkv = qkv.transpose(0, 1).contiguous()
            qkv.requires_grad = True

            out = apply_fused_qkv_rotary_pos_emb(
                qkv,
                emb_q,
                emb_k,
                qkv_split_arg_list=[hidden_size, hidden_size, hidden_size],
                tensor_format=tensor_format,
                interleaved=interleaved,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
            loss = loss_func(list(out))
            loss.backward()
            assert qkv.grad is not None

    def test_qkv_rope_v_unchanged(self, dtype, tensor_format, interleaved, loss_func):
        dev = _device()
        _seed()
        seq_length, batch_size, head_num, hidden_size = 32, 2, 4, 64

        qkv = torch.rand(
            (seq_length, batch_size, head_num, hidden_size * 3),
            dtype=dtype,
            device=dev,
        )
        if tensor_format == "bshd":
            qkv = qkv.transpose(0, 1).contiguous()

        _, _, v_orig = torch.split(qkv, [hidden_size, hidden_size, hidden_size], dim=-1)

        rope_q = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_q = rope_q(seq_length)
        rope_k = RotaryPositionEmbedding(hidden_size, interleaved=interleaved).to(dev)
        emb_k = rope_k(seq_length)

        out = apply_fused_qkv_rotary_pos_emb(
            qkv,
            emb_q,
            emb_k,
            qkv_split_arg_list=[hidden_size, hidden_size, hidden_size],
            tensor_format=tensor_format,
            interleaved=interleaved,
        )
        _, _, v_out = out
        torch.testing.assert_close(v_out, v_orig)


# ===================== Cross-format consistency =====================


class TestCrossFormatConsistency:
    def test_unfused_thd_vs_bshd_vs_sbhd(self):
        dev = _device()
        _seed()
        seqlen, batch_size, head_num, hidden_size = 16, 4, 4, 64
        cp_size = 2

        seq_lens = torch.tensor([seqlen for _ in range(batch_size)], dtype=torch.int32)
        cu_seqlens = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.int32), seq_lens]), dim=0
        ).to(device=dev, dtype=torch.int32)

        thd = torch.rand(
            (cu_seqlens[-1].item() // cp_size, head_num, hidden_size),
            dtype=torch.float16,
            device=dev,
        )
        bshd = thd.view(batch_size, -1, head_num, hidden_size).clone()
        sbhd = bshd.transpose(1, 0).clone()

        rope = RotaryPositionEmbedding(hidden_size).to(dev)
        emb = rope(seqlen * cp_size)

        for cp_rank in range(cp_size):
            out_thd = apply_rotary_pos_emb(
                thd,
                emb,
                tensor_format="thd",
                fused=False,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )

            out_bshd = apply_rotary_pos_emb(
                bshd,
                emb,
                tensor_format="bshd",
                fused=False,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )

            out_sbhd = apply_rotary_pos_emb(
                sbhd,
                emb,
                tensor_format="sbhd",
                fused=False,
                cu_seqlens=cu_seqlens,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )

            torch.testing.assert_close(
                out_bshd.reshape(out_thd.shape), out_thd, atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                out_sbhd.transpose(1, 0).reshape(out_thd.shape), out_thd, atol=1e-5, rtol=1e-5
            )

    def test_fused_thd_vs_bshd_vs_sbhd(self):
        dev = _device()
        _seed()
        seqlen, batch_size, head_num, hidden_size = 16, 4, 4, 64

        seq_lens = torch.tensor([seqlen for _ in range(batch_size)], dtype=torch.int32)
        cu_seqlens = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.int32), seq_lens]), dim=0
        ).to(device=dev, dtype=torch.int32)

        thd = torch.rand(
            (cu_seqlens[-1].item(), head_num, hidden_size),
            dtype=torch.float16,
            device=dev,
        )
        bshd = thd.view(batch_size, -1, head_num, hidden_size).clone()
        sbhd = bshd.transpose(1, 0).clone()

        rope = RotaryPositionEmbedding(hidden_size).to(dev)
        emb = rope(seqlen)

        out_thd = apply_rotary_pos_emb(
            thd,
            emb,
            tensor_format="thd",
            fused=True,
            cu_seqlens=cu_seqlens,
        )
        out_bshd = apply_rotary_pos_emb(
            bshd,
            emb,
            tensor_format="bshd",
            fused=True,
        )
        out_sbhd = apply_rotary_pos_emb(
            sbhd,
            emb,
            tensor_format="sbhd",
            fused=True,
        )

        torch.testing.assert_close(out_bshd.reshape(out_thd.shape), out_thd, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            out_sbhd.transpose(1, 0).reshape(out_thd.shape), out_thd, atol=1e-3, rtol=1e-3
        )


# ===================== Error / edge case tests =====================


class TestEdgeCases:
    def test_thd_requires_cu_seqlens(self):
        dev = _device()
        t = torch.randn(8, 4, 64, device=dev, dtype=torch.float16)
        freqs = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        with pytest.raises(AssertionError, match="cu_seqlens must not be None"):
            apply_rotary_pos_emb(t, freqs, tensor_format="thd", fused=True)

    def test_unsupported_tensor_format(self):
        dev = _device()
        t = torch.randn(8, 2, 4, 64, device=dev, dtype=torch.float16)
        freqs = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        with pytest.raises(ValueError, match="Unsupported tensor_format"):
            apply_rotary_pos_emb(t, freqs, tensor_format="thbd", fused=False)

    def test_qkv_thd_not_supported(self):
        dev = _device()
        qkv = torch.randn(8, 2, 4, 192, device=dev, dtype=torch.float16)
        qf = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        kf = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        with pytest.raises(AssertionError, match="'thd' tensor_format not supported"):
            apply_fused_qkv_rotary_pos_emb(qkv, qf, kf, [64, 64, 64], tensor_format="thd")

    def test_qkv_cp_with_start_positions_not_supported(self):
        dev = _device()
        qkv = torch.randn(8, 2, 4, 192, device=dev, dtype=torch.float16)
        qf = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        kf = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        sp = torch.tensor([0, 0], device=dev, dtype=torch.int32)
        with pytest.raises(AssertionError, match="start_positions != None with CP SIZE > 1"):
            apply_fused_qkv_rotary_pos_emb(
                qkv,
                qf,
                kf,
                [64, 64, 64],
                start_positions=sp,
                cp_size=2,
            )

    def test_rot_dim_less_than_head_dim(self):
        dev = _device()
        hidden_size = 128
        rot_dim = 64
        t = torch.randn(8, 2, 4, hidden_size, device=dev, dtype=torch.float16)
        freqs = torch.randn(8, 1, 1, rot_dim, device=dev, dtype=torch.float32)
        out = apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=True)
        assert out.shape == t.shape

    def test_float32_input(self):
        dev = _device()
        t = torch.randn(8, 2, 4, 64, device=dev, dtype=torch.float32)
        freqs = torch.randn(8, 1, 1, 64, device=dev, dtype=torch.float32)
        out = apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", fused=True)
        assert out.shape == t.shape
        assert out.dtype == torch.float32
