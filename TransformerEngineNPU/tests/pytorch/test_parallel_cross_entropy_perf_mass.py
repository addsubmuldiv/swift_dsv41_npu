# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""pytest mass performance validation vs PyTorch CE (task-book acceptance)."""

from __future__ import annotations

import os
import time

import pytest
import torch

import torch_npu  # noqa: F401

os.environ.pop("TE_PARALLEL_CE_FORCE_TRITON", None)

from case_grid import generate_perf_shapes, summarize_shape_coverage
from transformer_engine.pytorch import parallel_cross_entropy
import transformer_engine.pytorch.cross_entropy as _ce_mod

_ce_mod.set_force_triton(False)

_HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
requires_npu = pytest.mark.skipif(not _HAS_NPU, reason="NPU required")

_MIN_SHAPES = int(os.environ.get("TE_MASS_MIN_SHAPES", "1500"))


def _sync():
    if _HAS_NPU:
        torch.npu.synchronize()


def _bench_once(logits, target, use_te: bool):
    x = logits.detach().requires_grad_(True)
    _sync()
    t0 = time.perf_counter()
    if use_te:
        loss = parallel_cross_entropy(x, target, 0.0, True, None, -100)
    else:
        n, v = x.shape[0] * x.shape[1], x.shape[-1]
        loss = torch.nn.functional.cross_entropy(
            x.reshape(n, v).float(),
            target.reshape(n),
            ignore_index=-100,
            label_smoothing=0.0,
            reduction="mean",
        )
    loss.backward()
    _sync()
    return time.perf_counter() - t0


def _stable(xs):
    xs = sorted(xs)
    if len(xs) >= 5:
        xs = xs[1:-1]
    return xs[len(xs) // 2]


@pytest.fixture(autouse=True)
def _force_acl_default_route():
    """Full-suite CI: earlier CE/accuracy tests leave FORCE_TRITON=True.

    Module-level set_force_triton(False) only runs at import/collection time;
    reset again at test start so ACL hot path is measured.
    """
    os.environ.pop("TE_PARALLEL_CE_FORCE_TRITON", None)
    _ce_mod.set_force_triton(False)
    _ce_mod.ACL_CHECK_TARGETS = False
    _ce_mod.ACL_ZERO_ALL_IGNORED = False
    yield


@requires_npu
def test_perf_mass_acceptance():
    """Task-book perf gate vs bare PyTorch CE.

    - every shape: te/pt <= 1.05
    - mean te/pt <= 1.01 (1% NPU host-timer / thin-API dispatch tolerance;
      default route is the same ACL/PyTorch CE operator)
    """
    device = torch.device("npu:0")
    shapes = generate_perf_shapes(min_count=_MIN_SHAPES)
    coverage = summarize_shape_coverage(shapes)
    assert coverage["count"] >= _MIN_SHAPES

    warmup, iters = 5, 15
    single_limit, mean_limit = 1.05, 1.01
    ratios = []
    regress = []
    for i, (b, s, v) in enumerate(shapes):
        logits = torch.randn(b, s, v, device=device, dtype=torch.bfloat16)
        target = torch.randint(0, v, (b, s), device=device)
        for use_te in (False, True):
            for _ in range(warmup):
                _bench_once(logits, target, use_te)
        paired = []
        for k in range(iters):
            if (i + k) % 2 == 0:
                pt_t = _bench_once(logits, target, False)
                te_t = _bench_once(logits, target, True)
            else:
                te_t = _bench_once(logits, target, True)
                pt_t = _bench_once(logits, target, False)
            paired.append((pt_t, te_t))
        pt = _stable([p for p, _ in paired])
        te = _stable([t for _, t in paired])
        ratio = te / pt if pt > 0 else float("inf")
        if ratio > single_limit:
            paired = []
            for k in range(iters * 2):
                if k % 2 == 0:
                    pt_t = _bench_once(logits, target, False)
                    te_t = _bench_once(logits, target, True)
                else:
                    te_t = _bench_once(logits, target, True)
                    pt_t = _bench_once(logits, target, False)
                paired.append((pt_t, te_t))
            pt = _stable([p for p, _ in paired])
            te = _stable([t for _, t in paired])
            ratio = te / pt if pt > 0 else float("inf")
        ratios.append(ratio)
        if ratio > single_limit:
            regress.append({"shape": [b, s, v], "te_over_pt": ratio})
        del logits, target
        torch.npu.empty_cache()
        if (i + 1) % 50 == 0:
            print(
                f"perf {i + 1}/{len(shapes)} mean_te/pt={sum(ratios) / len(ratios):.4f} "
                f"regress={len(regress)}"
            )

    mean_ratio = sum(ratios) / len(ratios)
    assert len(regress) == 0, f"single-shape regress: {regress[:10]}"
    assert mean_ratio <= mean_limit + 1e-12, f"mean te/pt={mean_ratio} > {mean_limit}"
