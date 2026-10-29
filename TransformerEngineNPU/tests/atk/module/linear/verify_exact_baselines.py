# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Negative controls for the exact baselines of the ``linear`` guard.

The change this verifies replaces ``forward_output``, ``weight_grad`` and
``bias_grad`` in the golden with exact fp64 references (see
``../_exact_reference.py``). A baseline that is synthesised rather than measured
can fail in two opposite ways, and only the second is dangerous:

* the baseline is wrong -> the device is judged against nonsense (visible: every
  case red, or a real defect washed out);
* the *device under test* is also replaced -> the ATK comparison degenerates into
  reference-vs-reference, every case is green forever, and a real regression
  ships silently. This is the failure mode the whole design is shaped around.

So these checks are deliberately adversarial: they try to make the guard fail
loudly, and they assert that a deliberately corrupted device output still goes
red. Per the project's convention for guards (see the quantization modules'
``-x_ref`` negative control), an assertion that cannot fail is not evidence.

Checks C1, C2, C7, C7c, C8 and C9a/C9b are device-free and always run. C3-C6 need
a device (``--device cuda`` for the golden-side claims; ``npu`` is accepted but
only the raw-output half is meaningful there) and are reported as SKIP, never as
a silent PASS, when no device is available.
"""

import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_HERE = Path(__file__).resolve().parent
_MODULE_ROOT = _HERE.parent
for _path in (str(_MODULE_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _comparators import MixedToleranceAccuracy, load_design_acc_kwargs  # noqa: E402
from _exact_reference import (  # noqa: E402
    bias_grad_reference,
    chunked_bgrad,
    chunked_wgrad,
    concatenated_bias,
    concatenated_weight,
    forward_reference,
)
from function_linear import (  # noqa: E402
    _BGRAD_SLOT,
    _FORWARD_SLOT,
    _WGRAD_SLOT,
    _golden_side_guard,
    run_linear_case,
)

_DESIGN = _HERE / "linear.yaml"

# Small shape that already separates an fp32 reference from an fp64 one on fp16
# (coarser dtypes hide the drift under their own ULP, which is itself a finding).
_DRIFT_SHAPE = (16384, 96, 512)

_RESULTS = []


def _record(name, passed, detail=""):
    _RESULTS.append((name, passed, detail))
    print("  {:<34} {:<5} {}".format(name, "PASS" if passed else "FAIL", detail))


def _skip(name, detail):
    print("  {:<34} {:<5} {}".format(name, "SKIP", detail))


def _comparator():
    return MixedToleranceAccuracy(None, **load_design_acc_kwargs(_DESIGN))


def _ratio(local, remote, comparator):
    """Matched ratio under the same thresholds ATK would apply to this dtype."""
    thresholds = comparator.thresholds[local.dtype]
    local64, remote64 = local.to(torch.float64), remote.to(torch.float64)
    close = (local64 - remote64).abs() <= thresholds["atol"] + thresholds["rtol"] * remote64.abs()
    return float(close.to(torch.float32).mean().item())


def _judge(local, remote, comparator, index):
    return bool(
        comparator.compute_accuracy_result(local, remote, "output_{}.pt".format(index)).result
    )


def _make_case(dtype, inp_shape, out_features, device):
    in_features = inp_shape[-1]
    torch.manual_seed(0)
    x = torch.rand(*inp_shape, dtype=dtype) * 10 - 5
    weight = torch.rand(out_features, in_features, dtype=dtype) - 0.5
    bias = torch.rand(out_features, dtype=dtype) - 0.5
    dy = torch.rand(*inp_shape[:-1], out_features, dtype=dtype) - 0.5
    holder = SimpleNamespace(
        args=(
            x.to(device),
            weight.to(device),
            bias.to(device),
            dy.to(device),
            torch.zeros(out_features, dtype=dtype).to(device),
            True,  # bias_enabled
            False,  # return_bias
            False,  # consume_returned_bias
            "none",  # parameters_split
        )
    )
    return holder, x, dy, weight, bias


def _switch_sites():
    """Every ``exact_baselines=`` keyword across the module tree, by file.

    AST rather than text search: docstrings and this file's own search strings
    must not count, and a *positional* argument would not be seen here at all --
    which is the point, the switch has to stay keyword-only.
    """
    sites = {}
    for path in sorted(_MODULE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "exact_baselines":
                    on = isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                    key = path.relative_to(_MODULE_ROOT).as_posix()
                    sites.setdefault(key, []).append((node.lineno, on))
    return sites


def check_policy_lint():
    """C1: exactly the intended call sites turn the switch on."""
    sites = _switch_sites()
    on_counts = {
        name: count
        for name, count in (
            (name, sum(1 for _, on in entries if on)) for name, entries in sites.items()
        )
        if count
    }
    expected = {
        "linear/function_linear.py": 1,  # the gpu branch of __call__
        "linear/collect_nvte_golden.py": 1,  # offline golden collection
        "linear/verify_exact_baselines.py": 1,  # this file's baseline-mode check
        # layernorm_linear: its Linear shares te.Linear's cuBLASLt BGRADB
        # path, so its wgrad/bgrad slots carry the same golden-side references.
        "layernorm_linear/function_layernorm_linear.py": 1,  # lnl gpu branch
        "layernorm_linear/collect_nvte_golden.py": 1,  # lnl golden collection
        # grouped_linear: the grouped wgrad comes out of the same cuBLASLt
        # wgrad family, so its grad slots get the same golden-side treatment.
        # grouped_linear: the grouped wgrad comes out of the same cuBLASLt
        # wgrad family, so its grad slots get the same golden-side treatment.
        "grouped_linear/function_grouped_linear.py": 1,  # grouped gpu branch
        "grouped_linear/collect_nvte_golden.py": 1,  # grouped golden collection
        "grouped_linear/verify_exact_baselines.py": 1,  # grouped baseline-mode check
        # layernorm_mlp: NVTE's forward departs from the materialized chain
        # (fused gelu epilogue, skinny-N GEMM shapes) while TENPU reproduces
        # it, so the forward slot gets the same golden-side treatment.
        "layernorm_mlp/function_layernorm_mlp.py": 1,  # mlp gpu branch
        "layernorm_mlp/collect_nvte_golden.py": 1,  # mlp golden collection
        "layernorm_mlp/verify_exact_baselines.py": 1,  # mlp baseline-mode check
    }
    _record("C1 policy lint (11 golden sites)", on_counts == expected, "on: {}".format(on_counts))

    off_files = {name for name, entries in sites.items() if any(not on for _, on in entries)}
    # The grouped/mlp judges live outside the repo (atk_experiments) —
    # outside this lint's tree; the in-tree anchor is linear's.
    judges_off = "linear/diagnose_slot_vs_fp64.py" in off_files
    _record(
        "C1b judges keep the switch off",
        judges_off,
        "off in: {}".format(sorted(off_files)),
    )
    _record(
        "C1b judges keep the switch off",
        judges_off,
        "off in: {}".format(sorted(off_files)),
    )

    adapter = (_HERE / "function_linear.py").read_text(encoding="utf-8")
    npu_branch = [
        line for line in adapter.splitlines() if 'run_linear_case(input_data, "npu")' in line
    ]
    clean_npu = len(npu_branch) == 1 and "exact_baselines" not in npu_branch[0].replace(" ", "")
    _record("C1c NPU branch passes no switch", clean_npu, "device under test stays raw")


def check_guard_raises():
    """C2: the guard rejects anything that is not the golden host."""
    cpu = torch.zeros(2)
    cases = [
        ("npu", {"t": cpu}),
        ("npu:0", {"t": cpu}),
        ("cuda", {"t": cpu}),
    ]
    for device, tensors in cases:
        try:
            _golden_side_guard(device, tensors)
        except RuntimeError:
            _record("C2 guard rejects {!r}".format(device), True, "")
        else:
            _record("C2 guard rejects {!r}".format(device), False, "returned instead of raising")


def check_bias_truth_table():
    """C7: the four db shapes."""
    dy = torch.full((4, 6), 0.25, dtype=torch.bfloat16)
    grb = torch.arange(6, dtype=torch.bfloat16).to(torch.float32).to(torch.bfloat16)
    column_sum = chunked_bgrad(dy)
    expected = {
        (False, False, False): "empty",
        (False, False, True): "empty",
        (False, True, False): "empty",
        (False, True, True): "empty",
        (True, False, False): "column_sum",
        (True, False, True): "column_sum",
        (True, True, False): "empty",
        (True, True, True): "identity",
    }
    bad = []
    for (bias, ret, consume), kind in expected.items():
        got = bias_grad_reference(dy, grb, bias, ret, consume)
        if kind == "empty":
            ok = got is None
        elif kind == "column_sum":
            ok = got is not None and torch.equal(got, column_sum)
        else:
            ok = got is not None and torch.equal(got, grb.to(torch.float64))
        if not ok:
            bad.append((bias, ret, consume, kind))
    _record("C7 bias_grad truth table (8 combos)", not bad, "bad: {}".format(bad))

    wrong_pair = bias_grad_reference(dy, grb, True, False, True)
    _record(
        "C7b (return_bias=F, consume=T) is a column sum",
        wrong_pair is not None and torch.equal(wrong_pair, column_sum),
        "the shape the old collector mis-wrote as grad_returned_bias",
    )


def check_forward_truth_table():
    """C7c/C7d: the forward reference applies the bias iff the module does."""
    dtype = torch.float16
    tokens, in_features, out_features = 64, 32, 24
    torch.manual_seed(0)
    x = torch.rand(tokens, in_features, dtype=dtype) * 10 - 5
    weight = torch.rand(out_features, in_features, dtype=dtype) - 0.5
    bias = torch.rand(out_features, dtype=dtype) - 0.5
    slot = torch.empty(tokens, out_features, dtype=dtype)

    gemm = (x.double() @ weight.double().t()).to(dtype)
    with_bias = (x.double() @ weight.double().t() + bias.double()).to(dtype)
    applied = forward_reference(slot, x, weight, bias, True)
    withheld = forward_reference(slot, x, weight, bias, False)

    _record(
        "C7c forward truth table",
        torch.equal(applied, with_bias) and torch.equal(withheld, gemm),
        "apply_bias=True -> x@W^T+b, False -> x@W^T",
    )
    # Guards C7c against being vacuous: if the two agreed, the check above would
    # pass no matter which one the reference built.
    spread = float((applied.float() - withheld.float()).abs().max().item())
    _record(
        "C7d the two forward shapes differ",
        spread > 0.0,
        "max |+b - no b| = {:.4f} (bias magnitude)".format(spread),
    )


def check_split_concatenation():
    """C8: split parameters are rebuilt in order, and a bad split fails closed."""
    dtype = torch.float16
    out_features, in_features = 48, 16
    full_w = torch.rand(out_features, in_features, dtype=dtype)
    full_b = torch.rand(out_features, dtype=dtype)

    def _params(weight_rows, bias_rows):
        params, offset = [], 0
        for index, rows in enumerate(weight_rows):
            params.append(
                (
                    "p{}_weight".format(index),
                    torch.nn.Parameter(full_w[offset : offset + rows].clone()),
                )
            )
            offset += rows
        offset = 0
        for index, rows in enumerate(bias_rows):
            params.append(
                (
                    "p{}_bias".format(index),
                    torch.nn.Parameter(full_b[offset : offset + rows].clone()),
                )
            )
            offset += rows
        return params

    split = _params([18, 30], [18, 30])
    rebuilt_w = concatenated_weight(split, out_features)
    rebuilt_b = concatenated_bias(split, out_features)
    _record(
        "C8 split rows rebuilt in order",
        torch.equal(rebuilt_w.to(dtype), full_w) and torch.equal(rebuilt_b.to(dtype), full_b),
        "an 18+30 split equals the unsplit tensor",
    )

    try:
        concatenated_weight(_params([18, 29], [48]), out_features)
    except RuntimeError:
        _record("C8b untileable rows raise", True, "18+29 != 48")
    else:
        _record("C8b untileable rows raise", False, "returned a short tensor instead")


def check_fp32_forward_reference_drifts():
    """C9b: why the forward reference is fp64 and not fp32."""
    dtype = torch.float16
    tokens, in_features, out_features = 16384, 96, 512
    torch.manual_seed(0)
    x = torch.rand(tokens, in_features, dtype=dtype) * 10 - 5
    weight = torch.rand(out_features, in_features, dtype=dtype) - 0.5
    bias = torch.rand(out_features, dtype=dtype) - 0.5
    slot = torch.empty(tokens, out_features, dtype=dtype)
    ref64 = forward_reference(slot, x, weight, bias, True)
    ref32 = ((x.float() @ weight.float().t()) + bias.float()).to(dtype)
    n_diff = int((ref32 != ref64).sum())
    _record(
        "C9b fp32 forward ref drifts (fp16)",
        n_diff > 0,
        "{} of {} elements differ from the fp64 reference".format(n_diff, ref64.numel()),
    )


def check_fp32_reference_drifts():
    """C9a: why the wgrad reference has to be fp64 at all."""
    tokens, in_features, out_features = _DRIFT_SHAPE
    torch.manual_seed(0)
    x = torch.rand(tokens, in_features, dtype=torch.float16) * 10 - 5
    dy = torch.rand(tokens, out_features, dtype=torch.float16) - 0.5
    ref64 = chunked_wgrad(dy, x).to(torch.float16)
    acc32 = torch.zeros(out_features, in_features)
    for start in range(0, tokens, 32768):
        stop = min(start + 32768, tokens)
        acc32 += dy[start:stop].float().t() @ x[start:stop].float()
    ref32 = acc32.to(torch.float16)
    n_diff = int((ref32 != ref64).sum())
    _record(
        "C9a fp32 wgrad ref drifts (fp16)",
        n_diff > 0,
        "{} of {} elements differ from the fp64 reference".format(n_diff, ref32.numel()),
    )


def check_device_side(device, device_label):
    """C3-C6: raw vs exact, baseline mode, and two corrupted-output controls."""
    dtype = torch.float16
    inp_shape = [16384, 96]
    out_features = 512
    in_features = inp_shape[-1]
    holder, x, dy, weight, bias = _make_case(dtype, inp_shape, out_features, device)
    comparator = _comparator()

    raw = run_linear_case(holder, device, exact_baselines=False)
    ref_wgrad = chunked_wgrad(dy.reshape(-1, out_features), x.reshape(-1, in_features))
    ref_bgrad = chunked_bgrad(dy.reshape(-1, out_features))
    # bias_enabled=True / return_bias=False in _make_case, so the forward output
    # carries the bias and apply_bias is True. ``forward_reference`` writes onto
    # the slot's device; bring it back to host to compare with the other slots.
    ref_forward = (
        forward_reference(raw[_FORWARD_SLOT], x.reshape(-1, in_features), weight, bias, True)
        .detach()
        .float()
        .cpu()
        .reshape(-1, out_features)
    )

    raw_w = raw[_WGRAD_SLOT].detach().float().cpu().reshape(out_features, in_features)
    raw_b = raw[_BGRAD_SLOT].detach().float().cpu()
    raw_f = raw[_FORWARD_SLOT].detach().float().cpu().reshape(-1, out_features)
    ratio_w = _ratio(raw_w.to(dtype), ref_wgrad.to(dtype), comparator)
    ratio_b = _ratio(raw_b.to(dtype), ref_bgrad.to(dtype), comparator)
    ratio_f = _ratio(raw_f.to(dtype), ref_forward.to(dtype), comparator)
    detail = "wgrad {:.4f} bias_grad {:.4f} forward {:.4f}".format(ratio_w, ratio_b, ratio_f)

    # The golden side is the one documented to deviate; the device side must not.
    if device_label == "cuda":
        _record("C3 raw NVTE deviates from exact", ratio_w < 0.999, detail)
    else:
        _record("C3 raw device agrees with exact", ratio_w > 0.999 and ratio_f > 0.999, detail)

    if device_label != "cuda":
        _skip("C4 baseline mode", "golden-side only; run with --device cuda")
        _skip("C4b forward baseline", "golden-side only; run with --device cuda")
        _skip("C5 scale error still red", "needs the golden-side baseline")
        _skip("C5b forward scale still red", "needs the golden-side baseline")
        _skip("C6 axis error still red", "needs the golden-side baseline")
        _skip("C6b forward bias still red", "needs the golden-side baseline")
        return

    substituted = run_linear_case(holder, device, exact_baselines=True)
    sub_w = substituted[_WGRAD_SLOT].detach().float().cpu().reshape(out_features, in_features)
    sub_b = substituted[_BGRAD_SLOT].detach().float().cpu()
    sub_f = substituted[_FORWARD_SLOT].detach().float().cpu().reshape(-1, out_features)
    _record(
        "C4 baseline mode reproduces exact",
        _ratio(sub_w.to(dtype), ref_wgrad.to(dtype), comparator) == 1.0
        and _ratio(sub_b.to(dtype), ref_bgrad.to(dtype), comparator) == 1.0,
        "slots {} and {} are the reference itself".format(_WGRAD_SLOT, _BGRAD_SLOT),
    )
    _record(
        "C4b forward baseline reproduces exact",
        torch.equal(sub_f.to(dtype), ref_forward.to(dtype)),
        "slot {} is bitwise the fp64 reference rounded once".format(_FORWARD_SLOT),
    )

    scaled = (sub_w * 1.05).to(dtype)
    _record(
        "C5 scale error still red",
        not _judge(scaled, ref_wgrad.to(dtype), comparator, _WGRAD_SLOT),
        "x1.05 on wgrad must not pass an exact baseline",
    )
    scaled_f = (sub_f * 1.05).to(dtype)
    _record(
        "C5b forward scale error still red",
        not _judge(scaled_f, ref_forward.to(dtype), comparator, _FORWARD_SLOT),
        "x1.05 on forward must not pass an exact baseline",
    )

    transposed = sub_w.t().contiguous().reshape(-1)
    _record(
        "C6 axis error still red",
        not _judge(transposed, ref_wgrad.to(dtype).reshape(-1), comparator, _WGRAD_SLOT),
        "transposed wgrad must not pass an exact baseline",
    )
    # The forward reference's own failure mode: forgetting that return_bias cases
    # carry no bias. Dropping it here mimics exactly the mistake the `apply_bias`
    # argument exists to prevent, so it has to be red.
    without_bias = (sub_f - bias.float()).to(dtype)
    _record(
        "C6b forward missing-bias still red",
        not _judge(without_bias, ref_forward.to(dtype), comparator, _FORWARD_SLOT),
        "a bias-dropped golden must not pass",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, choices=["npu", "cuda"])
    args = parser.parse_args()

    print("exact baseline negative controls")
    check_policy_lint()
    check_guard_raises()
    check_bias_truth_table()
    check_forward_truth_table()
    check_split_concatenation()
    check_fp32_reference_drifts()
    check_fp32_forward_reference_drifts()

    if args.device is None:
        for name in (
            "C3 raw vs exact",
            "C4 baseline mode",
            "C4b forward baseline",
            "C5 scale",
            "C5b forward scale",
            "C6 axis",
            "C6b forward bias",
        ):
            _skip(name, "no --device given")
    else:
        check_device_side(args.device, args.device)

    failed = [name for name, passed, _ in _RESULTS if not passed]
    print(
        "\n{} checks, {} failed{}".format(
            len(_RESULTS), len(failed), (": " + ", ".join(failed)) if failed else ""
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
