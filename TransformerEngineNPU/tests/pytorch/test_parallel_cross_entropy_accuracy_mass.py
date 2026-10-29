# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""pytest mass accuracy validation for parallel_cross_entropy."""

from __future__ import annotations

import os

import pytest
import torch

import torch_npu  # noqa: F401

os.environ["TE_PARALLEL_CE_FORCE_TRITON"] = "1"

from case_grid import generate_accuracy_cases, summarize_shape_coverage
from transformer_engine.pytorch import parallel_cross_entropy
from transformer_engine.pytorch.cross_entropy import set_force_triton

set_force_triton(True)

_HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
requires_npu = pytest.mark.skipif(not _HAS_NPU, reason="NPU required")

_MIN_CASES = int(os.environ.get("TE_MASS_MIN_CASES", "1500"))
_CASES = generate_accuracy_cases(min_count=_MIN_CASES)
_COVERAGE = summarize_shape_coverage([(c[0], c[1], c[2]) for c in _CASES])


def _dtype(name: str):
    return torch.float32 if name == "float32" else torch.bfloat16


def _tols(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 2e-2, 2e-2


def _ref(logits, target, label_smoothing, reduce_loss, ignore_idx):
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
    return torch.nan_to_num(loss, nan=0.0)


def _run_case(case, device):
    b, s, v, dtype_name, reduce_loss, ls, use_ignore, swap, noncontig = case
    dtype = _dtype(dtype_name)
    if swap and not noncontig:
        logits = torch.randn(s, b, v, device=device, dtype=dtype)
        target = torch.randint(0, v, (s, b), device=device)
    elif swap:
        base = torch.randn(s, b, v, device=device, dtype=dtype)
        target = torch.randint(0, v, (s, b), device=device)
        logits = base.transpose(0, 1)
        target = target.transpose(0, 1).contiguous()
    else:
        logits = torch.randn(b, s, v, device=device, dtype=dtype)
        target = torch.randint(0, v, (b, s), device=device)

    if use_ignore:
        flat = target.reshape(-1).clone()
        # Keep >=1 valid label so mean-CE is defined (all-ignore covered by unit UT).
        n_ignore = min(max(1, flat.numel() // 8), max(0, flat.numel() - 1))
        if n_ignore > 0:
            flat[:n_ignore] = -100
            target = flat.view_as(target)

    x_t = logits.detach().clone().requires_grad_(True)
    x_r = x_t.detach().contiguous().clone().requires_grad_(True)
    tgt_r = target.contiguous()

    loss_t = parallel_cross_entropy(x_t, target, ls, reduce_loss, None, -100)
    loss_r = _ref(x_r, tgt_r, ls, reduce_loss, -100)
    if reduce_loss:
        loss_t.backward()
        loss_r.backward()
    else:
        loss_t.sum().backward()
        loss_r.sum().backward()

    atol, rtol = _tols(dtype)
    assert torch.allclose(
        loss_t.float(), loss_r.float().reshape_as(loss_t), atol=atol, rtol=rtol
    ), "loss_mismatch"
    assert x_t.grad is not None and x_r.grad is not None, "missing_grad"
    g_t = x_t.grad.float().contiguous()
    g_r = x_r.grad.float().contiguous().reshape_as(g_t)
    assert torch.allclose(g_t, g_r, atol=atol, rtol=rtol), "grad_mismatch"


@requires_npu
def test_accuracy_coverage_meta():
    assert _COVERAGE["count"] >= _MIN_CASES
    assert _COVERAGE["small_V_lt_1k"] > 0
    assert _COVERAGE["mid_V_1k_16k"] > 0
    assert _COVERAGE["large_V_ge_16k"] > 0


@requires_npu
@pytest.mark.parametrize("case", _CASES, ids=[f"c{i}" for i in range(len(_CASES))])
def test_accuracy_case(case):
    _run_case(case, torch.device("npu:0"))
