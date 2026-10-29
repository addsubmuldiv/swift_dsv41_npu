# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Tests for GDN (GatedDeltaNet) NPU-compatible torch-native FLA interfaces.

The tested functions are thin wrappers around existing PyTorch / Megatron
primitives (``F.conv1d``, ``F.normalize``, ``torch_chunk_gated_delta_rule``).
Accuracy comparisons against "reference" implementations that call the same
underlying primitives would be circular.  This file therefore focuses on
*contract* tests: smoke (runs without error, correct shapes), gradient
flow, input validation, and mathematical invariants where applicable.

Run with:
    pytest test/unit_tests/patches/megatron/test_gdn_packed_seq.py -v -s
"""

import logging

import pytest
import torch
import torch.nn.functional as F

from megatron_adaptor.features_manager.megatron import megatron_basic
from megatron_adaptor.patches.megatron.gdn_packed_seq import (
    _npu_causal_conv1d,
    _npu_chunk_gated_delta_rule,
    _npu_l2norm,
)

logger = logging.getLogger("test_gdn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")


DEVICE = "npu:0"
DTYPE = torch.bfloat16


class _PatchRecorder:
    def __init__(self):
        self.calls = []

    def register_patch(self, path, function, create_dummy=False):
        self.calls.append((path, function, create_dummy))


def test_register_patches_skips_fla_patches_when_fla_is_available(monkeypatch):
    feature = megatron_basic.MegatronBasicFeature()
    fla_patch_calls = []
    monkeypatch.setattr(megatron_basic, "HAVE_FLA", True)
    monkeypatch.setattr(feature, "register_mcore_basic_patches", lambda patch_manager, args: None)
    monkeypatch.setattr(feature, "register_fla_patches", lambda patch_manager, args: fla_patch_calls.append(True))
    monkeypatch.setattr(
        "megatron_adaptor.patches.patch_utils.is_megatron_training_available",
        lambda: False,
    )

    feature.register_patches(_PatchRecorder(), None)

    assert not fla_patch_calls


def test_register_patches_registers_fla_patches_when_fla_is_unavailable(monkeypatch):
    feature = megatron_basic.MegatronBasicFeature()
    fla_patch_calls = []
    monkeypatch.setattr(megatron_basic, "HAVE_FLA", False)
    monkeypatch.setattr(feature, "register_mcore_basic_patches", lambda patch_manager, args: None)
    monkeypatch.setattr(feature, "register_fla_patches", lambda patch_manager, args: fla_patch_calls.append(True))
    monkeypatch.setattr(
        "megatron_adaptor.patches.patch_utils.is_megatron_training_available",
        lambda: False,
    )

    feature.register_patches(_PatchRecorder(), None)

    assert fla_patch_calls == [True]


def test_register_fla_patches_registers_fla_fallbacks():
    recorder = _PatchRecorder()

    megatron_basic.MegatronBasicFeature().register_fla_patches(recorder, None)

    assert recorder.calls == [
        ("fla.modules.convolution.causal_conv1d", _npu_causal_conv1d, True),
        ("fla.ops.gated_delta_rule.chunk_gated_delta_rule", _npu_chunk_gated_delta_rule, True),
        ("fla.modules.l2norm.l2norm", _npu_l2norm, True),
    ]


def _make_gdn_inputs(B, T, H, D, requires_grad=False):
    """Create a full set of GDN inputs with optional gradient tracking."""
    q = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=requires_grad)
    k = F.normalize(torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE), p=2, dim=-1)
    if requires_grad:
        k = k.detach().requires_grad_(True)
    v = torch.randn(B, T, H, D, device=DEVICE, dtype=DTYPE, requires_grad=requires_grad)
    beta = torch.rand(B, T, H, device=DEVICE, dtype=DTYPE).sigmoid()
    if requires_grad:
        beta = beta.detach().requires_grad_(True)
    g = F.logsigmoid(torch.rand(B, T, H, device=DEVICE, dtype=DTYPE))
    if requires_grad:
        g = g.detach().requires_grad_(True)
    return q, k, v, beta, g


class TestCausalConv1d:
    """Tests for ``_npu_causal_conv1d`` (wrapper around ``F.conv1d``)."""

    @pytest.mark.parametrize("activation", [None, "silu"])
    def test_smoke(self, activation):
        """Runs without error and produces correct output shape."""
        B, S, D = 1, 64, 32
        x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
        w = torch.randn(D, 4, device=DEVICE, dtype=DTYPE)
        b = torch.randn(D, device=DEVICE, dtype=DTYPE)

        out, final_state = _npu_causal_conv1d(x, w, bias=b, activation=activation)
        assert out.shape == (B, S, D)
        assert out.dtype == DTYPE
        assert final_state is None

    @pytest.mark.parametrize("activation", [None, "silu"])
    def test_varlen(self, activation):
        """Packed-sequence path: per-sequence unpack → compute → repack."""
        D = 32
        cu_seqlens = torch.tensor([0, 10, 30, 45], dtype=torch.long, device=DEVICE)
        x = torch.randn(1, 45, D, device=DEVICE, dtype=DTYPE)
        w = torch.randn(D, 4, device=DEVICE, dtype=DTYPE)
        b = torch.randn(D, device=DEVICE, dtype=DTYPE)

        out, final_state = _npu_causal_conv1d(x, w, bias=b, activation=activation, cu_seqlens=cu_seqlens)
        assert out.shape == (1, 45, D)
        assert out.dtype == DTYPE
        assert final_state is None

    def test_gradient(self):
        """Backward produces finite gradients for input and weight."""
        B, S, D = 1, 64, 32
        x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        w = torch.randn(D, 4, device=DEVICE, dtype=DTYPE, requires_grad=True)

        out, _ = _npu_causal_conv1d(x, w, activation="silu")
        out.sum().backward()

        assert x.grad is not None and x.grad.isfinite().all(), "x.grad missing or non-finite"
        assert w.grad is not None and w.grad.isfinite().all(), "w.grad missing or non-finite"


class TestChunkGatedDeltaRule:
    """Tests for ``_npu_chunk_gated_delta_rule`` (delegates to Megatron built-in).

    Accuracy comparisons against a reference are intentionally omitted: the
    non-packed path is a direct pass-through to
    ``torch_chunk_gated_delta_rule``, so any "reference" would be testing
    the same function against itself.
    """

    @pytest.mark.parametrize("use_l2norm", [False, True])
    def test_smoke(self, use_l2norm):
        """Runs without error and produces correct output shape."""
        B, T, H, D = 1, 64, 2, 32
        q, k, v, beta, g = _make_gdn_inputs(B, T, H, D)

        o, final_state = _npu_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=64, use_qk_l2norm_in_kernel=use_l2norm
        )

        assert o.shape == v.shape
        assert o.dtype == DTYPE
        assert final_state is None

    def test_varlen(self):
        """Packed-sequence path: per-sequence unpack → compute → repack."""
        H, D = 2, 32
        total = 64
        cu_seqlens = torch.tensor([0, 16, 48, 64], dtype=torch.long, device=DEVICE)
        q, k, v, beta, g = _make_gdn_inputs(1, total, H, D)

        o, final_state = _npu_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=64, use_qk_l2norm_in_kernel=False, cu_seqlens=cu_seqlens
        )

        assert o.shape == v.shape
        assert final_state is None

    def test_gradient_finite(self):
        """Backward produces finite gradients for all input tensors."""
        B, T, H, D = 1, 64, 2, 32
        q, k, v, beta, g = _make_gdn_inputs(B, T, H, D, requires_grad=True)

        o, _ = _npu_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, chunk_size=64, use_qk_l2norm_in_kernel=False)
        o.sum().backward()

        for nm, t in [("q", q), ("k", k), ("v", v), ("beta", beta), ("g", g)]:
            assert t.grad is not None and t.grad.isfinite().all(), f"{nm}.grad invalid"


class TestDegeneratePackedSequence:
    """A single-element ``cu_seqlens`` (zero segments) must not crash.

    CPU-only: the degenerate path never reaches the underlying NPU primitives.
    """

    @pytest.mark.parametrize("activation", [None, "silu"])
    def test_causal_conv1d_empty_packed_batch(self, activation):
        D = 8
        x = torch.randn(1, 0, D, dtype=DTYPE)
        w = torch.randn(D, 4, dtype=DTYPE)
        b = torch.randn(D, dtype=DTYPE)

        out, final_state = _npu_causal_conv1d(x, w, bias=b, activation=activation, cu_seqlens=torch.tensor([0]))

        assert out.shape == (1, 0, D)
        assert out.dtype == DTYPE
        assert final_state is None

    def test_chunk_gated_delta_rule_empty_packed_batch(self):
        B, H, D = 1, 2, 8
        q = torch.randn(B, 0, H, D, dtype=DTYPE)
        k = torch.randn(B, 0, H, D, dtype=DTYPE)
        v = torch.randn(B, 0, H, D, dtype=DTYPE)
        beta = torch.rand(B, 0, H, dtype=DTYPE)
        g = torch.randn(B, 0, H, dtype=DTYPE)

        o, final_state = _npu_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=64, use_qk_l2norm_in_kernel=False, cu_seqlens=torch.tensor([0])
        )

        assert o.shape == v.shape
        assert final_state is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
