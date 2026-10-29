# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Black-box tests for the MLA (multi-latent attention) RoPE entry points.

These two functions apply the same rotation as Megatron-LM's
``megatron.core.fusions.fused_mla_yarn_rope_apply``:

* ``fused_apply_mla_rope_for_q`` rotates only the trailing ``emb_dim`` channels of
  ``q`` in the *interleaved* (GPT-J style) layout and leaves the leading
  ``qk_head_dim`` channels untouched;
* ``fused_apply_mla_rope_for_kv`` splits ``kv`` into key/value, rotates the shared
  ``k_pos_emb`` (a single head) and broadcasts it into the key.

Layout note: the outputs are element-wise identical to Megatron's Triton kernel — the
rotated channels are written back in stacked (first-half/second-half) order — so the
per-call fallback between the two implementations is layout-transparent.

Table note: the rope tables must be Megatron's duplicated-half layout
(``cat(freqs, freqs)``), because the native kernel's registered backward is the exact
adjoint only for duplicated angles; any other layout raises
``MLARopeUnsupportedError`` so callers fall back to the upstream implementation.

The reference implementations below are written with plain torch ops and follow the
documented interleaved-pair formulas, so the tests only rely on documented behavior.
"""

from typing import Tuple

import pytest
import torch
import torch_npu  # noqa: F401

from transformer_engine.pytorch.attention.rope import (
    MLARopeUnsupportedError,
    fused_apply_mla_rope_for_q,
    fused_apply_mla_rope_for_kv,
)


def _seed() -> None:
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


def _device() -> torch.device:
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    return torch.device("npu:0")


def _tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 1e-5, 1e-5


def _duplicated_freqs(
    seq: int, emb: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin the way Megatron's RoPE cache does: ``cat(freqs, freqs)``.

    The duplicated halves are required by both the kernel backward (exact adjoint only
    for duplicated angles) and the table-layout guard. Angles are drawn over several
    radians so gradient comparisons stay discriminating in bf16.
    """
    freqs = torch.rand(seq, 1, 1, emb // 2, device=device) * 6.0
    emb_full = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb_full), torch.sin(emb_full)


def _ref_mla_rope_q(
    q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, qk_head_dim: int, emb_dim: int
) -> torch.Tensor:
    """Reference: rotate ``q[..., qk_head_dim:]`` as interleaved pairs, keep the rest."""
    seq = q.shape[0]
    cos, sin = cos[:seq], sin[:seq]
    q_no_pe = q[..., :qk_head_dim]
    x = q[..., qk_head_dim:]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    cos_l, sin_l = cos[..., : emb_dim // 2], sin[..., : emb_dim // 2]
    cos_r, sin_r = cos[..., emb_dim // 2 :], sin[..., emb_dim // 2 :]
    out_even = x_even * cos_l - x_odd * sin_l
    out_odd = x_odd * cos_r + x_even * sin_r
    # Megatron writes first-half/second-half (stacked), not interleaved.
    x_rot = torch.cat((out_even, out_odd), dim=-1)
    return torch.cat((q_no_pe, x_rot), dim=-1)


def _ref_mla_rope_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    emb_dim: int,
    k_dim: int,
    v_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference: rotate ``k_pos_emb`` and broadcast it into the key; value is a view."""
    seq, _, heads, _ = kv.shape
    key = _ref_mla_rope_q(k_pos_emb, cos, sin, 0, emb_dim).expand(-1, -1, heads, -1)
    return torch.cat((kv[..., :k_dim], key), dim=-1), kv[..., k_dim:]


class TestMLARopeQ:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_matches_reference(self, dtype):
        device = _device()
        _seed()
        seq, batch, heads, qk, emb = 16, 2, 4, 8, 8
        q = torch.randn(seq, batch, heads, qk + emb, dtype=dtype, device=device)
        freqs = torch.rand(seq, 1, 1, emb // 2, device=device)
        emb_full = torch.cat((freqs, freqs), dim=-1)
        cos, sin = torch.cos(emb_full), torch.sin(emb_full)

        out = fused_apply_mla_rope_for_q(q, cos, sin, qk, emb)
        ref = _ref_mla_rope_q(q, cos, sin, qk, emb)

        atol, rtol = _tolerances(dtype)
        assert out.shape == q.shape and out.dtype == q.dtype
        # Leading (non-rotary) channels must be passed through untouched.
        assert torch.equal(out[..., :qk], q[..., :qk])
        assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)

    def test_non_duplicated_table_is_rejected(self):
        """The kernel backward is exact only for duplicated angles; other tables fall back."""
        device = _device()
        _seed()
        seq, batch, heads, qk, emb = 8, 1, 3, 4, 8
        q = torch.randn(seq, batch, heads, qk + emb, device=device)
        freqs = torch.rand(seq, 1, 1, emb // 2, device=device)
        cos = torch.cos(torch.cat((freqs, freqs * 2), dim=-1))
        sin = torch.sin(torch.cat((freqs, freqs * 2), dim=-1))

        with pytest.raises(MLARopeUnsupportedError):
            fused_apply_mla_rope_for_q(q, cos, sin, qk, emb)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_matches_reference(self, dtype):
        device = _device()
        _seed()
        seq, batch, heads, qk, emb = 16, 2, 4, 8, 8
        base = torch.randn(seq, batch, heads, qk + emb, dtype=dtype, device=device)
        cos, sin = _duplicated_freqs(seq, emb, device)
        grad = torch.randn_like(base)

        q_leaf = base.detach().clone().requires_grad_(True)
        fused_apply_mla_rope_for_q(q_leaf, cos, sin, qk, emb).backward(grad)

        q_ref = base.detach().clone().requires_grad_(True)
        _ref_mla_rope_q(q_ref, cos, sin, qk, emb).backward(grad)

        atol, rtol = _tolerances(dtype)
        assert torch.allclose(q_leaf.grad.float(), q_ref.grad.float(), atol=atol, rtol=rtol)

    def test_cos_sin_longer_than_seq_is_accepted(self):
        """Context parallel slicing may leave a longer cos/sin cache (Megatron behavior)."""
        device = _device()
        _seed()
        seq, batch, heads, qk, emb = 8, 1, 2, 4, 8
        q = torch.randn(seq, batch, heads, qk + emb, device=device)
        freqs = torch.rand(seq * 2, 1, 1, emb // 2, device=device)
        emb_full = torch.cat((freqs, freqs), dim=-1)
        cos, sin = torch.cos(emb_full), torch.sin(emb_full)

        out = fused_apply_mla_rope_for_q(q, cos, sin, qk, emb)
        ref = _ref_mla_rope_q(q, cos[:seq], sin[:seq], qk, emb)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_unsupported_inputs_are_rejected(self):
        """packed/thd, interleaved and 3D inputs (absorbed_mla) fall back to Megatron."""
        device = _device()
        _seed()
        seq, heads, qk, emb = 4, 2, 4, 8
        q = torch.randn(seq, 1, heads, qk + emb, device=device)
        cos = torch.randn(seq, 1, 1, emb, device=device)
        sin = torch.randn(seq, 1, 1, emb, device=device)

        with pytest.raises(MLARopeUnsupportedError):  # 3D input, e.g. absorbed_mla
            fused_apply_mla_rope_for_q(q.squeeze(1), cos, sin, qk, emb)

        with pytest.raises(MLARopeUnsupportedError):
            fused_apply_mla_rope_for_q(q, cos, sin, qk, emb, cu_seqlens_q=torch.tensor([0, seq]))
        with pytest.raises(MLARopeUnsupportedError):
            fused_apply_mla_rope_for_q(q, cos, sin, qk, emb, rotary_interleaved=True)


class TestMLARopeKV:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_matches_reference(self, dtype):
        device = _device()
        _seed()
        seq, batch, heads, k_dim, v_dim, emb = 16, 2, 4, 8, 8, 8
        kv = torch.randn(seq, batch, heads, k_dim + v_dim, dtype=dtype, device=device)
        k_pos_emb = torch.randn(seq, batch, 1, emb, dtype=dtype, device=device)
        freqs = torch.rand(seq, 1, 1, emb // 2, device=device)
        emb_full = torch.cat((freqs, freqs), dim=-1)
        cos, sin = torch.cos(emb_full), torch.sin(emb_full)

        key, value = fused_apply_mla_rope_for_kv(kv, k_pos_emb, cos, sin, emb, k_dim, v_dim)
        ref_key, ref_value = _ref_mla_rope_kv(kv, k_pos_emb, cos, sin, emb, k_dim, v_dim)

        atol, rtol = _tolerances(dtype)
        assert key.shape == (seq, batch, heads, emb + k_dim) and key.dtype == kv.dtype
        assert value.shape == (seq, batch, heads, v_dim) and value.dtype == kv.dtype
        # Non-rotary key channels and the whole value must be passed through untouched.
        assert torch.equal(key[..., :k_dim], kv[..., :k_dim])
        assert torch.equal(value, kv[..., k_dim:])
        assert torch.allclose(key.float(), ref_key.float(), atol=atol, rtol=rtol)

    def test_value_is_a_view_of_kv(self):
        """The value tensor must not be materialized (Megatron relies on it being a view)."""
        device = _device()
        _seed()
        kv = torch.randn(4, 1, 2, 16, device=device)
        k_pos_emb = torch.randn(4, 1, 1, 8, device=device)
        cos, sin = _duplicated_freqs(4, 8, device)
        _, value = fused_apply_mla_rope_for_kv(kv, k_pos_emb, cos, sin, 8, 8, 8)
        assert value.untyped_storage().data_ptr() == kv.untyped_storage().data_ptr()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_backward_matches_reference(self, dtype):
        device = _device()
        _seed()
        seq, batch, heads, k_dim, v_dim, emb = 8, 2, 3, 8, 8, 8
        kv = torch.randn(seq, batch, heads, k_dim + v_dim, dtype=dtype, device=device)
        k_pos_emb = torch.randn(seq, batch, 1, emb, dtype=dtype, device=device)
        cos, sin = _duplicated_freqs(seq, emb, device)

        grad_key = torch.randn(seq, batch, heads, emb + k_dim, dtype=dtype, device=device)
        grad_value = torch.randn(seq, batch, heads, v_dim, dtype=dtype, device=device)

        kv_leaf = kv.detach().clone().requires_grad_(True)
        kpe_leaf = k_pos_emb.detach().clone().requires_grad_(True)
        key, value = fused_apply_mla_rope_for_kv(kv_leaf, kpe_leaf, cos, sin, emb, k_dim, v_dim)
        torch.autograd.backward([key, value], [grad_key, grad_value])

        kv_ref = kv.detach().clone().requires_grad_(True)
        kpe_ref = k_pos_emb.detach().clone().requires_grad_(True)
        ref_key, ref_value = _ref_mla_rope_kv(kv_ref, kpe_ref, cos, sin, emb, k_dim, v_dim)
        torch.autograd.backward([ref_key, ref_value], [grad_key, grad_value])

        atol, rtol = _tolerances(dtype)
        assert torch.allclose(kv_leaf.grad.float(), kv_ref.grad.float(), atol=atol, rtol=rtol)
        assert torch.allclose(kpe_leaf.grad.float(), kpe_ref.grad.float(), atol=atol, rtol=rtol)

    def test_packed_and_interleaved_are_rejected(self):
        device = _device()
        _seed()
        kv = torch.randn(4, 1, 2, 16, device=device)
        k_pos_emb = torch.randn(4, 1, 1, 8, device=device)
        cos, sin = _duplicated_freqs(4, 8, device)

        with pytest.raises(MLARopeUnsupportedError):  # 3D input, e.g. absorbed_mla
            fused_apply_mla_rope_for_kv(kv.squeeze(1), k_pos_emb, cos, sin, 8, 8, 8)
        with pytest.raises(MLARopeUnsupportedError):
            fused_apply_mla_rope_for_kv(
                kv, k_pos_emb, cos, sin, 8, 8, 8, cu_seqlens_kv=torch.tensor([0, 4])
            )
        with pytest.raises(MLARopeUnsupportedError):
            fused_apply_mla_rope_for_kv(kv, k_pos_emb, cos, sin, 8, 8, 8, rotary_interleaved=True)

    def test_non_duplicated_table_is_rejected(self):
        device = _device()
        _seed()
        kv = torch.randn(4, 1, 2, 16, device=device)
        k_pos_emb = torch.randn(4, 1, 1, 8, device=device)
        freqs = torch.rand(4, 1, 1, 4, device=device)
        cos = torch.cos(torch.cat((freqs, freqs * 2), dim=-1))
        sin = torch.sin(torch.cat((freqs, freqs * 2), dim=-1))

        with pytest.raises(MLARopeUnsupportedError):
            fused_apply_mla_rope_for_kv(kv, k_pos_emb, cos, sin, 8, 8, 8)
