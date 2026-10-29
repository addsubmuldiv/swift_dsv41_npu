#!/usr/bin/env python3
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Benchmark mHC aggregate / expand_combine Triton kernels vs PyTorch reference.

Usage (on remote NPU box):
    source /home/Temp_room/te_npu_30/env.sh
    export LIBRARY_PATH=/home/Temp_room/te_npu_30/tools/linklib:/usr/lib64:$LIBRARY_PATH
    cd /home/Temp_room/te_npu_31/src/TransformerEngineNPU
    python tests/pytorch/benchmark_mhc_aggregate_expand.py --warmup 5 --iters 30 \
        --json-out /tmp/perf_aggregate_expand.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

import torch_npu  # noqa: F401

from transformer_engine.pytorch.triton.mhc import (
    mhc_fused_aggregate,
    mhc_fused_expand_combine,
)

os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
os.environ.setdefault("NVTE_DISABLE_TRITON_AUTOTUNING", "1")
os.environ.setdefault("TE_NPU_ALLOW_NONDETERMINISTIC_ALGO", "1")


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------


def aggregate_ref(x, H_pre):
    return (x * H_pre.unsqueeze(-2)).sum(dim=-1)


def expand_combine_ref(f, bias, H_post, x, H_res):
    if bias is not None:
        f = f + bias.view(1, 1, -1)
    term_post = f.unsqueeze(-1) * H_post.unsqueeze(-2)
    term_res = torch.einsum("sbci,sbij->sbcj", x, H_res)
    return term_post + term_res


# ---------------------------------------------------------------------------
# H generator
# ---------------------------------------------------------------------------


def random_legal_H(s, b, dtype, device):
    H_pre = torch.sigmoid(torch.randn(s, b, 4, device=device, dtype=torch.float32)).to(dtype)
    H_post = 2.0 * torch.sigmoid(torch.randn(s, b, 4, device=device, dtype=torch.float32)).to(dtype)
    raw = torch.randn(s, b, 4, 4, device=device, dtype=torch.float32)
    H_res = F.softmax(raw, dim=-1)
    H_res = H_res / H_res.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    H_res = H_res.to(dtype)
    return H_pre, H_post, H_res


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    shape: str
    dtype: str
    bias: bool
    mode: str  # "triton" or "pytorch"
    mean_ms: float
    peak_mem_mb: float


def _sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _peak_mb():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.npu.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _reset_peak():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats()
        torch.npu.empty_cache()


def bench(fn, warmup: int, iters: int) -> tuple[float, float]:
    """Returns (mean_ms, peak_mem_mb)."""
    for _ in range(warmup):
        fn()
    _sync()

    _reset_peak()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    elapsed = time.perf_counter() - start
    peak = _peak_mb()
    return (elapsed / iters) * 1000.0, peak


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    dtypes = [torch.float32, torch.bfloat16]
    shapes = [
        (8, 2, 128),
        (16, 1, 256),
        (4, 4, 512),
        (32, 2, 128),
        (16, 4, 256),
    ]

    results: list[Result] = []

    for s, b, C in shapes:
        for dtype in dtypes:
            shape_str = f"s={s},b={b},C={C}"
            device = torch.device("npu:0")

            x = torch.randn(s, b, C, 4, device=device, dtype=dtype)
            H_pre, H_post, H_res = random_legal_H(s, b, dtype, device)
            f = torch.randn(s, b, C, device=device, dtype=dtype)
            bias = torch.randn(C, device=device, dtype=dtype)

            # --- aggregate ---
            mean_triton, peak_triton = bench(
                lambda x=x, H_pre=H_pre: mhc_fused_aggregate(x, H_pre, n=4), args.warmup, args.iters
            )
            mean_pytorch, peak_pytorch = bench(
                lambda x=x, H_pre=H_pre: aggregate_ref(x, H_pre), args.warmup, args.iters
            )
            results.append(
                Result(
                    "aggregate", shape_str, str(dtype), False, "triton", mean_triton, peak_triton
                )
            )
            results.append(
                Result(
                    "aggregate", shape_str, str(dtype), False, "pytorch", mean_pytorch, peak_pytorch
                )
            )

            # --- expand_combine (no bias) ---
            mean_triton, peak_triton = bench(
                lambda f=f, H_post=H_post, x=x, H_res=H_res: mhc_fused_expand_combine(
                    f, None, H_post, x, H_res, n=4
                ),
                args.warmup,
                args.iters,
            )
            mean_pytorch, peak_pytorch = bench(
                lambda f=f, H_post=H_post, x=x, H_res=H_res: expand_combine_ref(
                    f, None, H_post, x, H_res
                ),
                args.warmup,
                args.iters,
            )
            results.append(
                Result(
                    "expand_combine",
                    shape_str,
                    str(dtype),
                    False,
                    "triton",
                    mean_triton,
                    peak_triton,
                )
            )
            results.append(
                Result(
                    "expand_combine",
                    shape_str,
                    str(dtype),
                    False,
                    "pytorch",
                    mean_pytorch,
                    peak_pytorch,
                )
            )

            # --- expand_combine (with bias) ---
            mean_triton, peak_triton = bench(
                lambda f=f, bias=bias, H_post=H_post, x=x, H_res=H_res: mhc_fused_expand_combine(
                    f, bias, H_post, x, H_res, n=4
                ),
                args.warmup,
                args.iters,
            )
            mean_pytorch, peak_pytorch = bench(
                lambda f=f, bias=bias, H_post=H_post, x=x, H_res=H_res: expand_combine_ref(
                    f, bias, H_post, x, H_res
                ),
                args.warmup,
                args.iters,
            )
            results.append(
                Result(
                    "expand_combine_bias",
                    shape_str,
                    str(dtype),
                    True,
                    "triton",
                    mean_triton,
                    peak_triton,
                )
            )
            results.append(
                Result(
                    "expand_combine_bias",
                    shape_str,
                    str(dtype),
                    True,
                    "pytorch",
                    mean_pytorch,
                    peak_pytorch,
                )
            )

    # Print table
    print(f"\n{'=' * 100}")
    print(
        f"{'Operation':<22} {'Shape':<20} {'Dtype':<12} {'Bias':<6} {'Mode':<8} {'Mean(ms)':<12} {'Peak(MB)':<10}"
    )
    print(f"{'=' * 100}")
    for r in results:
        print(
            f"{r.name:<22} {r.shape:<20} {r.dtype:<12} {str(r.bias):<6} {r.mode:<8} {r.mean_ms:<12.4f} {r.peak_mem_mb:<10.2f}"
        )

    # Speedup summary
    print(f"\n{'=' * 60}")
    print("Speedup summary (Triton vs PyTorch):")
    for i in range(0, len(results), 2):
        triton = results[i]
        pytorch = results[i + 1]
        speedup = pytorch.mean_ms / triton.mean_ms if triton.mean_ms > 0 else float("inf")
        print(f"  {triton.name} {triton.shape} {triton.dtype} bias={triton.bias}: {speedup:.2f}x")

    # JSON output
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump([asdict(r) for r in results], fp, indent=2)
        print(f"\nResults written to {args.json_out}")


if __name__ == "__main__":
    main()
