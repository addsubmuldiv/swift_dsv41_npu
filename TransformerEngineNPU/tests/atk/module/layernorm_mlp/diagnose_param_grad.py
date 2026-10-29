# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Diagnose the real-tier LayerNorm/RMSNorm param-grad mismatches (NPU host).

First golden-loop run of the ``layernorm_mlp`` guard: every small-tier case
passes and every real-tier case fails **only** on ``output_2``/``output_3``
(``ln_weight_grad`` / ``ln_bias_grad``) with ``matched_ratio`` 0.94-0.98 and
3-30 elements over the absolute/ULP backstop, while forward output, input
grad, all fc1/fc2 grads and the API contract pass everywhere.

Mechanism under test: the param grads are sums over 1024-4096 tokens; the two
implementations are independently-rounded bf16 pipelines, so the incoming
``dgrad`` already differs from NVTE's at ~1-ULP-per-element scale, and any
element where the sum cancels to near zero shows that noise as a large
*relative* error. TENPU's hand-written ``_norm_bwd`` (bf16 stats, bf16
products) adds a comparable extra term on ``dgamma``.

This script measures, per failing case and per output, three pairwise
comparisons under the exact mixed-tolerance criteria:

    T-vs-N   : TENPU (bf16)   vs NVTE golden   (reproduces the ATK verdict)
    T-vs-R   : TENPU (bf16)   vs fp32 autograd reference (TENPU's own noise)
    N-vs-R   : NVTE golden    vs fp32 autograd reference (NVTE's own noise)

If T-vs-R ~= T-vs-N while N-vs-R is clean, TENPU's bf16 backward is the gap
(core fix viable); if both T-vs-R and N-vs-R show the same noise, the
difference is mutual bf16 pipeline noise (guard-side: per-output thresholds,
see ``../_comparators.py``). ``--fp32-norm-bwd`` A/B-compares against the
pre-fix hand-written backward re-implemented in fp32 (the module itself now
uses the fused kernels: native_layer_norm_backward / npu_rms_norm_backward).

Run on the NPU host that has the input/golden artifacts, e.g.:

    python tests/atk/module/layernorm_mlp/diagnose_param_grad.py \
        --cases result/layernorm_mlp/json/all_layernorm_mlp.json \
        --input-dir ./layernorm_mlp_inputs \
        --golden-dir ./layernorm_mlp_goldens \
        --case-ids 6 7 8 9 10
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from function_layernorm_mlp import RESULT_SCHEMA, run_layernorm_mlp_case  # noqa: E402

_TENSOR_COUNT = 8  # x, dy, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b


def _mt_stats(actual, expected, rtol, atol, ratio_req, max_abs, max_ulp):
    """Mirror MixedToleranceAccuracy.compute_accuracy_result on CPU tensors."""
    a = actual.detach().to(torch.float64)
    e = expected.detach().to(torch.float64)
    err = (a - e).abs()
    close = err <= atol + rtol * e.abs()
    ratio = float(close.to(torch.float32).mean().item()) if e.numel() else 1.0
    if expected.dtype in (torch.float16, torch.bfloat16, torch.float32):
        ulp_dtype = expected.dtype
    else:
        ulp_dtype = torch.float32
    remote = expected.detach().to(ulp_dtype)
    spacing = (torch.nextafter(remote, torch.full_like(remote, float("inf"))) - remote).to(
        torch.float64
    )
    hard = (err > max_abs) & (err > max_ulp * spacing)
    unmatched = ~close
    ref_abs = e.abs()
    return {
        "ratio": ratio,
        "hard": int(hard.sum().item()),
        "err_max": float(err.max().item()) if e.numel() else 0.0,
        "unmatched_ref_abs_median": float(ref_abs[unmatched].median().item())
        if int(unmatched.sum().item())
        else float("nan"),
        "matched_ref_abs_median": float(ref_abs[close].median().item())
        if int(close.sum().item())
        else float("nan"),
    }


def _fmt(stats):
    if stats is None:
        return "        --        "
    return "ratio={:.4f} hard={:<3d} errmax={:8.3f}".format(
        stats["ratio"], stats["hard"], stats["err_max"]
    )


def _load_case_args(input_dir, case_id):
    input_file = Path(input_dir) / str(case_id) / "input.bin"
    args = torch.load(input_file, map_location="cpu", weights_only=True)
    return list(args)


def _run_side(args, device, dtype):
    tensors = [t.to(dtype).to(device) for t in args[:_TENSOR_COUNT]]
    attrs = args[_TENSOR_COUNT:]
    return run_layernorm_mlp_case(SimpleNamespace(args=tensors + list(attrs)), device)


def _load_golden(golden_dir, save_name, case_id, count):
    base = Path(golden_dir) / "cpu_nvte" / save_name / str(case_id)
    outputs = []
    for index in range(count):
        outputs.append(torch.load(base / "output_{}.pt".format(index), weights_only=True))
    return outputs


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="ATK all_layernorm_mlp.json")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--case-ids", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--rtol", type=float, default=0.05, help="matched-ratio rtol (module yaml)")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--ratio-req", type=float, default=0.99)
    parser.add_argument("--max-abs", type=float, default=1.0)
    parser.add_argument("--max-ulp", type=float, default=32.0)
    parser.add_argument(
        "--fp32-norm-bwd",
        action="store_true",
        help="Also re-measure with _apply_norm/_norm_bwd patched to fp32 precision",
    )
    return parser.parse_args()


def _patch_fp32_norm():
    """Patch LayerNormMLP's norm fwd/bwd to fp32 stats and fp32 products."""
    import transformer_engine.pytorch.module.layernorm_mlp as lnm
    import torch_npu

    def _apply_norm_fp32(
        inputmat,
        ln_weight,
        ln_bias,
        eps,
        normalization,
        zero_centered_gamma,
        activation_dtype=None,
    ):
        if normalization == "LayerNorm":
            gamma = 1.0 + ln_weight if zero_centered_gamma else ln_weight
            if activation_dtype is not None:
                gamma = gamma.to(activation_dtype)
                if ln_bias is not None:
                    ln_bias = ln_bias.to(activation_dtype)
            ln_out, mu, rsigma = torch.native_layer_norm(
                inputmat, [inputmat.shape[-1]], gamma, ln_bias, eps
            )
            return ln_out, mu, rsigma
        if normalization == "RMSNorm":
            gamma = 1.0 + ln_weight if zero_centered_gamma else ln_weight
            if activation_dtype is not None:
                gamma = gamma.to(activation_dtype)
            ln_out, rrsigma = torch_npu.npu_rms_norm(inputmat, gamma, epsilon=eps)
            return ln_out, None, rrsigma
        raise ValueError("Unknown normalization: {}".format(normalization))

    def _norm_bwd_fp32(
        dgrad, inputmat, ln_weight, ln_bias, mu, rsigma, zero_centered_gamma, normalization
    ):
        # NOTE: the module's real _norm_bwd now uses the fused kernels
        # (native_layer_norm_backward / npu_rms_norm_backward); this patch
        # re-implements the pre-fix hand-written math in fp32 for A/B triage.
        d = dgrad.float()
        x = inputmat.float()
        w = ln_weight.float()
        n = x.shape[-1]
        if mu is not None:
            mu = mu.float()
        rs = rsigma.float()
        x_hat = (x - mu) * rs if mu is not None else x * rs
        dx_hat = d * ((1.0 + w) if zero_centered_gamma else w)
        reduce_dims = tuple(range(len(d.shape) - 1))
        if normalization == "LayerNorm":
            dvar = (dx_hat * (x - mu) * (-0.5) * rs.pow(3)).sum(dim=-1, keepdim=True)
            dmean = (-dx_hat * rs).sum(dim=-1, keepdim=True) + dvar * (-2.0 / n) * (x - mu).sum(
                dim=-1, keepdim=True
            )
            dx = dx_hat * rs + dvar * 2.0 / n * (x - mu) + dmean / n
            dbeta = d.sum(dim=reduce_dims)
        elif normalization == "RMSNorm":
            dvar = (dx_hat * x).sum(dim=-1, keepdim=True) * (-0.5) * rs.pow(3)
            dx = dx_hat * rs + dvar * 2.0 * x / n
            dbeta = None
        else:
            raise ValueError("Unknown normalization: {}".format(normalization))
        dgamma = (d * x_hat).sum(dim=reduce_dims)
        return (
            dx.to(dgrad.dtype),
            dgamma.to(ln_weight.dtype),
            (dbeta.to(ln_weight.dtype) if dbeta is not None else None),
        )

    lnm._apply_norm = _apply_norm_fp32
    lnm._norm_bwd = _norm_bwd_fp32


def main():
    args = _parse_args()
    if args.device == "npu" and not torch.npu.is_available():
        raise RuntimeError("NPU device required (run on the NPU host)")

    with open(args.cases, "r", encoding="utf-8") as stream:
        cases = {case["id"]: case for case in json.load(stream)}
    save_name = Path(args.cases).stem

    patched_note = ""
    if args.fp32_norm_bwd:
        _patch_fp32_norm()
        patched_note = "  [fp32-norm-bwd patched]"

    for case_id in args.case_ids:
        case = cases[case_id]
        specs = case["inputs"]
        shape = specs[0]["shape"]
        attrs = [spec["range_values"] for spec in specs[_TENSOR_COUNT:]]
        print(
            "\n== case {id}  shape={shape} {norm}/{act} bias={bias} zc={zc} eps={eps}"
            "{patched}".format(
                id=case_id,
                shape=shape,
                norm=attrs[1],
                act=attrs[2],
                bias=attrs[3],
                zc=attrs[4],
                eps=attrs[0],
                patched=patched_note,
            )
        )
        raw_args = _load_case_args(args.input_dir, case_id)

        tenpu = [out.detach().to("cpu") for out in _run_side(raw_args, args.device, torch.bfloat16)]
        reference = [
            out.detach().to("cpu") for out in _run_side(raw_args, args.device, torch.float32)
        ]
        golden = _load_golden(args.golden_dir, save_name, case_id, len(RESULT_SCHEMA))

        print(
            "   {:<28s} {:<24s} {:<24s} {:<24s}".format(
                "output", "T-vs-NVTE", "T-vs-fp32", "NVTE-vs-fp32"
            )
        )
        for index, name in enumerate(RESULT_SCHEMA):
            if name == "api_contract_json_utf8":
                continue
            row = []
            for left, right in ((tenpu, golden), (tenpu, reference), (golden, reference)):
                if left[index].numel() == 0 and right[index].numel() == 0:
                    row.append(None)
                    continue
                row.append(
                    _mt_stats(
                        left[index],
                        right[index],
                        args.rtol,
                        args.atol,
                        args.ratio_req,
                        args.max_abs,
                        args.max_ulp,
                    )
                )
            print(
                "   {:<28s} {} | {} | {}".format(
                    "o{} {}".format(index, name), _fmt(row[0]), _fmt(row[1]), _fmt(row[2])
                )
            )
            stats = row[0]
            if stats and stats["hard"]:
                print(
                    "        near-zero check (T-vs-NVTE): unmatched |ref| median "
                    "{:.4g} vs matched {:.4g}".format(
                        stats["unmatched_ref_abs_median"],
                        stats["matched_ref_abs_median"],
                    )
                )

    print(
        "\nReading the table:\n"
        "  - T-vs-NVTE ~= the ATK verdict (sanity: should match the csv row).\n"
        "  - T-vs-fp32 noisy + NVTE-vs-fp32 clean  -> TENPU backward precision gap;\n"
        "    try --fp32-norm-bwd (core-fix preview).\n"
        "  - both T-vs-fp32 and NVTE-vs-fp32 noisy -> mutual bf16 pipeline noise;\n"
        "    resolve guard-side with per-output thresholds (bf16_atol_output_2 ...).\n"
        "  - unmatched |ref| median << matched median -> failures sit on\n"
        "    near-zero-cancellation elements (noise, not systematic error)."
    )


if __name__ == "__main__":
    main()
