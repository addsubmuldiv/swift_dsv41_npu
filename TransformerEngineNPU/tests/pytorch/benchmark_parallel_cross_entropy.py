# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark parallel_cross_entropy vs PyTorch CE on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from transformer_engine.pytorch import parallel_cross_entropy


def _sync():
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def bench_once(logits, target, reduce_loss, label_smoothing, use_te: bool):
    logits = logits.detach().requires_grad_(True)
    _sync()
    t0 = time.perf_counter()
    if use_te:
        loss = parallel_cross_entropy(logits, target, label_smoothing, reduce_loss, None)
    else:
        n, v = logits.shape[0] * logits.shape[1], logits.shape[-1]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(n, v).float(),
            target.reshape(n),
            label_smoothing=label_smoothing,
            reduction="mean" if reduce_loss else "none",
        )
        if not reduce_loss:
            loss = loss.reshape(logits.shape[0], logits.shape[1])
    if reduce_loss:
        loss.backward()
    else:
        loss.sum().backward()
    _sync()
    return time.perf_counter() - t0


def run_suite(shapes, warmup, iters, device, label: str):
    rows = []
    for b, s, v in shapes:
        logits = torch.randn(b, s, v, device=device, dtype=torch.bfloat16)
        target = torch.randint(0, v, (b, s), device=device)
        for use_te in (False, True):
            for _ in range(warmup):
                bench_once(logits, target, True, 0.0, use_te)
            times = [bench_once(logits, target, True, 0.0, use_te) for _ in range(iters)]
            avg = sum(times) / len(times)
            rows.append(
                {
                    "suite": label,
                    "shape": [b, s, v],
                    "impl": "te" if use_te else "pytorch",
                    "avg_s": avg,
                    "ms": avg * 1000,
                }
            )
            print(f"[{label}] {'te' if use_te else 'pt'} shape={(b, s, v)} avg={avg * 1000:.3f} ms")
    summary = []
    for i in range(0, len(rows), 2):
        pt, te = rows[i], rows[i + 1]
        ratio = te["avg_s"] / pt["avg_s"] if pt["avg_s"] > 0 else float("inf")
        summary.append(
            {
                "suite": label,
                "shape": pt["shape"],
                "pytorch_ms": pt["ms"],
                "te_ms": te["ms"],
                "te_over_pt": ratio,
                "regress_gt_5pct": ratio > 1.05,
            }
        )
        print(f"[{label}] shape={pt['shape']} te/pt={ratio:.3f} regress>{5}%={ratio > 1.05}")
    avg_ratio = sum(s["te_over_pt"] for s in summary) / len(summary)
    print(f"[{label}] mean te/pt={avg_ratio:.3f}")
    return rows, summary, avg_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--device", type=str, default="npu:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    shapes = [
        (2, 128, 4096),
        (4, 256, 8192),
        (2, 512, 32000),
        (2, 256, 65536),
        (1, 512, 128000),
    ]

    import transformer_engine.pytorch.cross_entropy as ce_mod

    # Default routing: clear env + module flag so import-time FORCE cannot pollute.
    os.environ.pop("TE_PARALLEL_CE_FORCE_TRITON", None)
    ce_mod.set_force_triton(False)
    rows_d, sum_d, avg_d = run_suite(shapes, args.warmup, args.iters, device, "default_route")

    # Forced Triton: kernel-only comparison.
    os.environ["TE_PARALLEL_CE_FORCE_TRITON"] = "1"
    ce_mod.set_force_triton(True)
    rows_f, sum_f, avg_f = run_suite(shapes, args.warmup, args.iters, device, "force_triton")

    out = {
        "default_route": {"rows": rows_d, "summary": sum_d, "mean_te_over_pt": avg_d},
        "force_triton": {"rows": rows_f, "summary": sum_f, "mean_te_over_pt": avg_f},
        "acceptance_default_mean_ok": avg_d <= 1.0,
        "acceptance_default_single_ok": all(not s["regress_gt_5pct"] for s in sum_d),
    }
    print(
        "ACCEPT default mean<=1:",
        out["acceptance_default_mean_ok"],
        "single<=5%:",
        out["acceptance_default_single_ok"],
    )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
