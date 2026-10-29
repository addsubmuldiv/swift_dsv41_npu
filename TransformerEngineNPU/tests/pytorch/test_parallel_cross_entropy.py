# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for parallel_cross_entropy on Ascend NPU."""

from __future__ import annotations

import os
import random

import pytest
import torch

import torch_npu  # noqa: F401

os.environ.setdefault("TE_PARALLEL_CE_FORCE_TRITON", "1")

from transformer_engine.pytorch import parallel_cross_entropy
from transformer_engine.pytorch.cross_entropy import set_force_triton
import transformer_engine.pytorch.cross_entropy as _ce_mod

_HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
requires_npu = pytest.mark.skipif(not _HAS_NPU, reason="NPU required")


def _device():
    return torch.device("npu:0") if _HAS_NPU else torch.device("cpu")


def _dtype_tols(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-5, "rtol": 1e-5}
    if dtype == torch.bfloat16:
        return {"atol": 2e-2, "rtol": 2e-2}
    return {"atol": 1e-3, "rtol": 1e-3}


def _pytorch_ce(logits, target, label_smoothing, reduce_loss, ignore_idx):
    n, v = logits.shape[0] * logits.shape[1], logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(n, v).float(),
        target.reshape(n),
        ignore_index=ignore_idx,
        label_smoothing=label_smoothing,
        reduction="mean" if reduce_loss else "none",
    )
    if not reduce_loss:
        loss = loss.reshape(logits.shape[0], logits.shape[1])
    return loss


@pytest.fixture(autouse=True)
def _npu_seed():
    torch.manual_seed(0)
    random.seed(0)
    if _HAS_NPU:
        torch.npu.manual_seed_all(0)
    yield


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduce_loss", [True, False])
@pytest.mark.parametrize("swap_dim", [False, True])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_single_device_fwd_bwd(dtype, reduce_loss, swap_dim, label_smoothing):
    device = _device()
    batch, seq, vocab = 2, 32, 1024
    if swap_dim:
        logits = torch.randn(seq, batch, vocab, dtype=dtype, device=device)
        target = torch.randint(0, vocab, (seq, batch), device=device)
    else:
        logits = torch.randn(batch, seq, vocab, dtype=dtype, device=device)
        target = torch.randint(0, vocab, (batch, seq), device=device)

    logits_t = logits.clone().detach().requires_grad_(True)
    logits_r = logits.clone().detach().requires_grad_(True)
    loss_t = parallel_cross_entropy(
        logits_t, target, label_smoothing, reduce_loss, None, ignore_idx=-100
    )
    loss_r = _pytorch_ce(logits_r, target, label_smoothing, reduce_loss, -100)
    if reduce_loss:
        (loss_t**2).backward()
        (loss_r**2).backward()
    else:
        loss_t.square().sum().backward()
        loss_r.square().sum().backward()

    tols = _dtype_tols(dtype)
    torch.testing.assert_close(
        loss_t.float().cpu(), loss_r.float().cpu().reshape_as(loss_t), **tols
    )
    torch.testing.assert_close(
        logits_t.grad.float().cpu(), logits_r.grad.float().cpu().reshape_as(logits_t), **tols
    )


@requires_npu
@pytest.mark.parametrize("reduce_loss", [True, False])
def test_ignore_idx(reduce_loss):
    device = _device()
    batch, seq, vocab = 2, 16, 512
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32, device=device)
    target = torch.randint(0, vocab, (batch, seq), device=device)
    target[0, :3] = -100
    logits_t = logits.clone().detach().requires_grad_(True)
    logits_r = logits.clone().detach().requires_grad_(True)
    loss_t = parallel_cross_entropy(logits_t, target, 0.0, reduce_loss, None, -100)
    loss_r = _pytorch_ce(logits_r, target, 0.0, reduce_loss, -100)
    if reduce_loss:
        loss_t.backward()
        loss_r.backward()
    else:
        loss_t.sum().backward()
        loss_r.sum().backward()
    torch.testing.assert_close(loss_t.cpu(), loss_r.cpu().reshape_as(loss_t), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(logits_t.grad.cpu(), logits_r.grad.cpu(), atol=1e-5, rtol=1e-5)


@requires_npu
def test_non_contiguous_transposed_input():
    device = _device()
    s, b, v = 4, 2, 128
    torch.manual_seed(42)
    logits = torch.randn(s, b, v, device=device)
    target = torch.randint(0, v, (b, s), device=device)
    logits_t = logits.transpose(0, 1)
    logits_c = logits_t.contiguous()
    assert logits_t.stride(-1) == 1
    assert logits_t.stride(-2) != logits_t.shape[-1]
    loss_t = parallel_cross_entropy(logits_t, target, 0.0, False, None)
    loss_c = parallel_cross_entropy(logits_c, target, 0.0, False, None)
    assert torch.allclose(loss_t, loss_c)


@requires_npu
def test_empty_vocab_raises():
    device = _device()
    logits = torch.randn(2, 4, 8, device=device)
    target = torch.randint(0, 8, (2, 3), device=device)
    with pytest.raises(ValueError):
        parallel_cross_entropy(logits, target)


@requires_npu
def test_invalid_rank_shape_raises():
    device = _device()
    logits = torch.randn(2, 4, device=device)
    target = torch.randint(0, 4, (2, 4), device=device)
    with pytest.raises(ValueError):
        parallel_cross_entropy(logits, target)


def _peak_allocated_bytes():
    """Peak since last reset_peak_memory_stats (not live memory_allocated)."""
    torch.npu.synchronize()
    return int(torch.npu.max_memory_allocated())


def _live_allocated_bytes():
    torch.npu.synchronize()
    return int(torch.npu.memory_allocated())


@requires_npu
def test_memory_peak_vs_pytorch():
    """Peak allocated memory should not grow >5% vs PyTorch CE (same shape)."""
    device = _device()
    batch, seq, vocab = 2, 128, 16384
    torch.npu.empty_cache()

    def _run(use_te: bool):
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        logits = torch.randn(
            batch, seq, vocab, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        target = torch.randint(0, vocab, (batch, seq), device=device)
        if use_te:
            loss = parallel_cross_entropy(logits, target, 0.0, True, None)
        else:
            n, v = batch * seq, vocab
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(n, v).float(), target.reshape(n), reduction="mean"
            )
        loss.backward()
        peak = _peak_allocated_bytes()
        del logits, target, loss
        torch.npu.empty_cache()
        return peak

    for _ in range(2):
        _run(False)
        _run(True)
    torch.npu.empty_cache()
    pt_peak = _run(False)
    te_peak = _run(True)
    if pt_peak <= 0:
        return
    growth = (te_peak - pt_peak) / float(pt_peak)
    assert growth <= 0.05 + 1e-6, f"memory growth {growth:.3%} > 5% (te={te_peak}, pt={pt_peak})"


@requires_npu
def test_memory_no_leak_repeated():
    """Repeated fwd/bwd should not monotonically grow allocated memory."""
    device = _device()
    batch, seq, vocab = 2, 64, 8192
    torch.npu.empty_cache()

    def _once():
        logits = torch.randn(
            batch, seq, vocab, dtype=torch.float32, device=device, requires_grad=True
        )
        target = torch.randint(0, vocab, (batch, seq), device=device)
        loss = parallel_cross_entropy(logits, target, 0.0, True, None)
        loss.backward()
        del logits, target, loss
        torch.npu.empty_cache()
        return _live_allocated_bytes()

    for _ in range(3):
        _once()
    samples = [_once() for _ in range(8)]
    ref = sorted(samples[:4])[len(samples[:4]) // 2]
    last = samples[-1]
    if ref <= 0:
        return
    assert last <= ref * 1.05 + 16 * 1024 * 1024, (
        f"possible leak: ref={ref} last={last} samples={samples}"
    )


@requires_npu
def test_logits_not_overwritten():
    """Caller logits must remain unchanged after Triton forward."""
    device = _device()
    batch, seq, vocab = 2, 16, 512
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32, device=device, requires_grad=True)
    before = logits.detach().clone()
    target = torch.randint(0, vocab, (batch, seq), device=device)
    loss = parallel_cross_entropy(logits, target, 0.1, True, None)
    loss.backward()
    assert torch.equal(logits.detach(), before), "caller logits were overwritten"


@requires_npu
@pytest.mark.parametrize("force_triton", [True, False])
def test_oob_target_raises(force_triton):
    """OOB targets must raise ValueError on Triton and on ACL-with-check.

    ACL default hot path skips host target sync for task-book latency; UT enables
    TE_PARALLEL_CE_CHECK_TARGETS when exercising the ACL route.
    """
    prev_force = _ce_mod._FORCE_TRITON
    set_force_triton(force_triton)
    prev_check = _ce_mod.ACL_CHECK_TARGETS
    if not force_triton:
        _ce_mod.ACL_CHECK_TARGETS = True
    try:
        device = _device()
        batch, seq, vocab = 2, 8, 128
        logits = torch.randn(batch, seq, vocab, dtype=torch.float32, device=device)
        target = torch.randint(0, vocab, (batch, seq), device=device)
        target[0, 0] = vocab + 5
        with pytest.raises(ValueError, match="out-of-range"):
            parallel_cross_entropy(logits, target, 0.0, True, None)
    finally:
        set_force_triton(prev_force)
        _ce_mod.ACL_CHECK_TARGETS = prev_check


@pytest.mark.parametrize("force_triton", [False, True])
@requires_npu
def test_all_ignored_reduce_loss_is_zero(force_triton):
    """Mean CE with every label ignored must be 0 (not NaN) on default entry."""
    prev_force = _ce_mod._FORCE_TRITON
    set_force_triton(force_triton)
    # Do not flip ACL_ZERO_ALL_IGNORED — exercise the shipped default path.
    try:
        device = torch.device("npu:0")
        logits = torch.randn(1, 1, 64, device=device, dtype=torch.float32, requires_grad=True)
        target = torch.full((1, 1), -100, device=device, dtype=torch.long)
        loss = parallel_cross_entropy(logits, target, 0.0, True, None, -100)
        assert torch.isfinite(loss).item()
        assert float(loss.item()) == 0.0
        loss.backward()
        assert logits.grad is not None
        assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))
    finally:
        set_force_triton(prev_force)


@requires_npu
def test_repeated_backward_retain_graph():
    """Saved grad template must not be mutated across retain_graph backwards."""
    device = _device()
    batch, seq, vocab = 2, 16, 256
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32, device=device, requires_grad=True)
    target = torch.randint(0, vocab, (batch, seq), device=device)
    loss = parallel_cross_entropy(logits, target, 0.0, True, None)
    g1 = torch.autograd.grad(
        loss, logits, grad_outputs=torch.tensor(2.0, device=device), retain_graph=True
    )[0]
    g2 = torch.autograd.grad(loss, logits, grad_outputs=torch.tensor(2.0, device=device))[0]
    torch.testing.assert_close(g1, g2, atol=1e-5, rtol=1e-5)


@requires_npu
def test_batch_slice_noncontiguous_rows():
    """Batch slice base[::2] must match contiguous clone (fwd+bwd)."""
    device = _device()
    torch.manual_seed(7)
    base = torch.randn(4, 3, 16, dtype=torch.float32, device=device)
    sliced = base[::2]
    assert sliced.shape == (2, 3, 16)
    assert sliced.stride() == (96, 16, 1)
    target = torch.randint(0, 16, (2, 3), device=device)

    logits_s = sliced.clone().detach().requires_grad_(True)
    logits_c = sliced.contiguous().clone().detach().requires_grad_(True)
    loss_s = parallel_cross_entropy(logits_s, target, 0.0, True, None)
    loss_c = parallel_cross_entropy(logits_c, target, 0.0, True, None)
    torch.testing.assert_close(loss_s, loss_c, atol=1e-5, rtol=1e-5)
    loss_s.backward()
    loss_c.backward()
    torch.testing.assert_close(logits_s.grad, logits_c.grad, atol=1e-5, rtol=1e-5)
