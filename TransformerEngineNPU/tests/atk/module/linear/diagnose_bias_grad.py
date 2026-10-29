# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""One-shot root-cause discriminator for the linear case-9 bias_grad failure.

REPORT atk 3#17: case 9 ([4096,512], bias=True) fails only on output_4
(bias_grad) with matched_ratio 0.9570 against the NVTE golden, while every
other output of the same case and every smaller-row bias case passes. The
first hypothesis (bf16 ``torch.sum`` blocked-reduction divergence vs NVTE's
wgrad-GEMM fp32 accumulator, commit d0ca7d3) could not be reproduced on the
local stack (CANN 26.0.rc1: bf16 sum is bitwise identical to fp32-accumulate-
then-round), so the fix was reverted in 90d52eb and the intranet re-run
reproduced the exact same 0.9570 — deterministic, but non-discriminating.

This script runs on the intranet NPU host (where input.bin and the golden
tree live) and compares three db variants against the actual golden:

  A  dy.sum(dim=0)                      — current (post-revert) TENPU path
  B  dy.sum(dim=0, dtype=fp32).to(bf16) — the d0ca7d3 fix
  C  dy.to(fp32).sum(dim=0).to(bf16)    — exact reference (golden's math)

Verdict table:
  * A fails, B/C pass        → intranet stack's bf16 sum IS lossy → re-land
                               d0ca7d3 (revert the revert).
  * A == B == C, all fail    → the golden db is not a plain fp32 column sum
                               (NVTE epilogue / stale golden) → investigate
                               the golden side, do not blame torch.sum.
  * A == B == C, all pass    → not reproducible in isolation → the divergence
                               enters elsewhere (reopen with ATK artifacts).

Usage (on the NPU host, alongside the golden tree):
  python diagnose_bias_grad.py \
    --input-bin ./linear_inputs/9/input.bin \
    --golden ./linear_goldens/cpu_nvte/all_linear/9/output_4.pt
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

# Standard mixed-tolerance gate from linear.yaml (primary ratio gate).
RTOL = 0.0156
ATOL = 0.0156
REQUIRED_RATIO = 0.99


def _gate(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, int, float]:
    """Return (matched_ratio, failing_count, max_abs_err) for the gate."""
    a = actual.to(torch.float32)
    e = expected.to(torch.float32)
    ok = (a - e).abs() <= ATOL + RTOL * e.abs()
    return ok.float().mean().item(), int((~ok).sum().item()), (a - e).abs().max().item()


def _describe(name: str, tensor: torch.Tensor, golden: torch.Tensor) -> Dict:
    ratio, fails, max_err = _gate(tensor, golden)
    print(
        "  {:<44} bitwise={:<5} ratio={:.4f} failing={:<4d} max_abs_err={:.6f}".format(
            name, str(torch.equal(tensor, golden)), ratio, fails, max_err
        )
    )
    return {"name": name, "ratio": ratio, "fails": fails, "bitwise": torch.equal(tensor, golden)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument(
        "--input-bin",
        required=True,
        help="ATK canonical input artifact, e.g. <input-dir>/9/input.bin",
    )
    parser.add_argument(
        "--golden",
        required=True,
        help="NVTE golden bias_grad, e.g. <golden-dir>/cpu_nvte/all_linear/9/output_4.pt",
    )
    parser.add_argument(
        "--device",
        default="npu",
        help="Device to run the sums on (must be npu to probe the NPU stack; default npu)",
    )
    cli = parser.parse_args()

    for label, path in (("input", cli.input_bin), ("golden", cli.golden)):
        if not Path(path).exists():
            raise FileNotFoundError("{} not found: {}".format(label, path))

    # run_linear_case signature: inp, weight, bias, grad_output, grad_returned_bias, ...
    saved = torch.load(cli.input_bin, map_location="cpu", weights_only=True)
    dy_cpu = saved[3]
    golden = torch.load(cli.golden, map_location="cpu", weights_only=True)

    dy = dy_cpu.to(cli.device)
    inner = dy.shape[-1]
    dy2d = dy.reshape(-1, inner)

    print("grad_output: shape={} dtype={} device={}".format(tuple(dy.shape), dy.dtype, cli.device))
    print(
        "golden db:   shape={} dtype={} |db| median={:.4f} min={:.6f}".format(
            tuple(golden.shape),
            golden.dtype,
            golden.abs().median().item(),
            golden.abs().min().item(),
        )
    )
    print("gate: rtol={} atol={} required_ratio={}".format(RTOL, ATOL, REQUIRED_RATIO))
    print("variants vs golden:")

    variants = {
        "A old   dy.sum(dim=0)": dy2d.sum(dim=0),
        "B fix   sum(dtype=fp32).to(bf16)": dy2d.sum(dim=0, dtype=torch.float32).to(dy.dtype),
        "C ref   fp32 sum then round": dy2d.to(torch.float32).sum(dim=0).to(dy.dtype),
    }
    results = {name: _describe(name, tensor.cpu(), golden) for name, tensor in variants.items()}

    a, b, c = (results[name] for name in variants)
    # H2 signature: all three variants agree with each other (the sum path is
    # deterministic and identical across formulations) yet none matches golden.
    abc_identical = all(
        torch.equal(variants[n1].cpu(), variants[n2].cpu())
        for n1, n2 in (
            ("A old   dy.sum(dim=0)", "B fix   sum(dtype=fp32).to(bf16)"),
            ("B fix   sum(dtype=fp32).to(bf16)", "C ref   fp32 sum then round"),
        )
    )

    print("verdict:")
    if a["ratio"] < REQUIRED_RATIO <= b["ratio"]:
        print(
            "  A fails / B passes -> intranet bf16 sum is lossy: RE-LAND d0ca7d3 (revert the revert)."
        )
    elif abc_identical and c["ratio"] < REQUIRED_RATIO:
        print(
            "  A == B == C but golden disagrees -> golden db is not a plain fp32 column sum;\n"
            "  investigate the NVTE-side golden (epilogue/stale) — torch.sum is NOT the cause."
        )
    elif a["ratio"] >= REQUIRED_RATIO and b["ratio"] >= REQUIRED_RATIO:
        print(
            "  all variants pass in isolation -> divergence enters elsewhere; reopen with\n"
            "  the actual ATK npu_output artifacts."
        )
    else:
        print("  mixed outcome — see the table above and REPORT atk 3#17.")


if __name__ == "__main__":
    main()
