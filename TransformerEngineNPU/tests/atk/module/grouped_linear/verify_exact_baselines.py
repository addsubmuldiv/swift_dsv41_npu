# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Negative controls for the exact baselines of the ``grouped_linear`` guard.

The change this verifies replaces ``weights_grad_concatenated`` (and
``biases_grad_concatenated_or_empty`` when bias is on) in the golden with exact
fp64 references (see ``../_exact_reference.py`` and
``function_grouped_linear.py``). A baseline that is synthesised rather than
measured can fail in two opposite ways, and only the second is dangerous:

* the baseline is wrong -> the device is judged against nonsense (visible: every
  case red, or a real defect washed out);
* the *device under test* is also replaced -> the ATK comparison degenerates into
  reference-vs-reference, every case is green forever, and a real regression
  ships silently. This is the failure mode the whole design is shaped around.

So these checks are deliberately adversarial: they try to make the guard fail
loudly, and they assert that a deliberately corrupted device output still goes
red. Per the project's convention for guards (see the quantization modules'
``-x_ref`` negative control), an assertion that cannot fail is not evidence.

Checks C1, C2, C7-C10 are device-free and always run. C3-C6 need a device
(``--device cuda`` for the golden-side claims; ``npu`` is accepted but only the
raw-output half is meaningful there) and are reported as SKIP, never as a silent
PASS, when no device is available.
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
from function_grouped_linear import (  # noqa: E402
    _BGRAD_SLOT,
    _WGRAD_SLOT,
    RESULT_SCHEMA,
    _golden_side_guard,
    _row_spans,
    grouped_bgrad_reference,
    grouped_wgrad_reference,
    run_grouped_linear_case,
)

_DESIGN = _HERE / "grouped_linear.yaml"

_RESULTS = []


def _record(name, passed, detail=""):
    _RESULTS.append((name, passed, detail))
    print("  {:<38} {:<5} {}".format(name, "PASS" if passed else "FAIL", detail))


def _skip(name, detail):
    print("  {:<38} {:<5} {}".format(name, "SKIP", detail))


def _comparator():
    return MixedToleranceAccuracy(None, **load_design_acc_kwargs(_DESIGN))


def _judge(local, remote, comparator, index):
    return bool(
        comparator.compute_accuracy_result(local, remote, "output_{}.pt".format(index)).result
    )


def _make_case(dtype, rows, in_features, out_features, m_splits, bias_enabled, device):
    torch.manual_seed(0)
    x = torch.rand(rows, in_features, dtype=dtype) * 10 - 5
    dy = torch.rand(rows, out_features, dtype=dtype) - 0.5
    weights = [torch.rand(out_features, in_features, dtype=dtype) - 0.5 for _ in m_splits]
    biases = [torch.rand(out_features, dtype=dtype) - 0.5 for _ in m_splits]
    holder = SimpleNamespace(
        args=(
            (x.to(device), dy.to(device))
            + tuple(w.to(device) for w in weights)
            + tuple(b.to(device) for b in biases)
            + (len(m_splits), ",".join(str(m) for m in m_splits), bias_enabled)
        )
    )
    return holder, x, dy, weights, biases


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
        "linear/verify_exact_baselines.py": 1,  # that file's baseline-mode check
        # layernorm_linear: its Linear shares te.Linear's cuBLASLt BGRADB
        # path, so its wgrad/bgrad slots carry the same golden-side references.
        "layernorm_linear/function_layernorm_linear.py": 1,  # lnl gpu branch
        "layernorm_linear/collect_nvte_golden.py": 1,  # lnl golden collection
        # grouped_linear: the grouped wgrad comes out of the same cuBLASLt
        # wgrad family, so its grad slots get the same golden-side treatment.
        "grouped_linear/function_grouped_linear.py": 1,  # grouped gpu branch
        "grouped_linear/collect_nvte_golden.py": 1,  # grouped golden collection
        "grouped_linear/verify_exact_baselines.py": 1,  # this file's C4 check
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
    _record("C1b judges keep the switch off", judges_off, "off in: {}".format(sorted(off_files)))

    adapter = (_HERE / "function_grouped_linear.py").read_text(encoding="utf-8")
    npu_branch = [
        line
        for line in adapter.splitlines()
        if 'run_grouped_linear_case(input_data, "npu")' in line
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


def check_reference_vs_autograd():
    """C7: the grouped references equal independent fp64 autograd, rounded once.

    The independent path is torch autograd over per-expert ``F.linear`` modules
    in fp64 — a different formulation than the chunked accumulation the baseline
    uses, so agreement (bitwise, after the single rounding into the case dtype)
    is evidence and not a tautology. Per-block comparison also pins the expert
    ORDER of the concatenated slot: expert e's block must be expert e's own
    gradient, not a neighbour's.
    """
    dtype = torch.float16
    rows, in_features, out_features = 512, 32, 96
    m_splits = [301, 197, 14]
    torch.manual_seed(0)
    x = torch.rand(rows, in_features, dtype=dtype) * 10 - 5
    dy = torch.rand(rows, out_features, dtype=dtype) - 0.5
    weights = [torch.rand(out_features, in_features, dtype=dtype) - 0.5 for _ in m_splits]
    biases = [torch.rand(out_features, dtype=dtype) - 0.5 for _ in m_splits]

    # Distinguishable experts: if the reference ever mixed expert blocks, the
    # per-block comparison below would fail by a factor of 3, not by an ULP.
    weights[1] = weights[1] * 3.0
    weights[2] = weights[2] * 0.1

    x64 = x.double().requires_grad_(True)
    w64 = [w.double().clone().requires_grad_(True) for w in weights]
    b64 = [b.double().clone().requires_grad_(True) for b in biases]
    outs = []
    offset = 0
    for index, count in enumerate(m_splits):
        outs.append(
            torch.nn.functional.linear(x64[offset : offset + count], w64[index], b64[index])
        )
        offset += count
    loss = (torch.cat(outs) * dy.double()).sum()
    loss.backward()
    autograd_wgrad = torch.cat([w.grad.reshape(-1) for w in w64])
    autograd_bgrad = torch.cat([b.grad for b in b64])

    ref_wgrad = grouped_wgrad_reference(x, dy, m_splits)
    ref_bgrad = grouped_bgrad_reference(dy, m_splits)
    _record(
        "C7 wgrad == fp64 autograd (bitwise after one rounding)",
        torch.equal(ref_wgrad.to(dtype), autograd_wgrad.to(dtype)),
        "max |ref - autograd| = {:.3e} (fp64)".format(
            (ref_wgrad - autograd_wgrad).abs().max().item()
        ),
    )
    _record(
        "C7b bgrad == fp64 autograd (bitwise after one rounding)",
        torch.equal(ref_bgrad.to(dtype), autograd_bgrad.to(dtype)),
        "colsum path",
    )

    # Per-block: expert order in the concatenated slot.
    offset = 0
    order_ok = True
    for index, w in enumerate(w64):
        count = w.grad.numel()
        block = ref_wgrad[offset : offset + count]
        if not torch.equal(block.to(dtype), w.grad.to(dtype).reshape(-1)):
            order_ok = False
        offset += count
    _record("C7c expert blocks in m_splits order", order_ok, "scaled experts cannot swap silently")


def check_row_spans():
    """C8: the expert row spans tile, and a negative split fails closed."""
    spans = _row_spans([3, 0, 4])
    ok = spans == [(0, 3), (3, 3), (3, 7)]
    _record("C8 row spans tile [0, rows)", ok, "spans: {}".format(spans))
    try:
        _row_spans([3, -1])
    except ValueError:
        _record("C8b negative split raises", True, "")
    else:
        _record("C8b negative split raises", False, "returned instead of raising")


def check_fp32_reference_drifts():
    """C9: why the grouped wgrad reference has to be fp64 and not fp32.

    The per-expert K is the m_splits entry, up to thousands on the generalized
    set; an fp32 GEMM reference drifts by ~a dtype ULP over a long K, while the
    fp16 gate only allows ~2 ULP (rtol 1.95e-3 against an fp16 ULP of 9.77e-4).
    """
    dtype = torch.float16
    rows, in_features, out_features = 16384, 96, 512
    m_splits = [8029, 8135]
    torch.manual_seed(0)
    x = torch.rand(rows, in_features, dtype=dtype) * 10 - 5
    dy = torch.rand(rows, out_features, dtype=dtype) - 0.5
    ref64 = grouped_wgrad_reference(x, dy, m_splits).to(dtype)
    blocks32 = []
    offset = 0
    for count in m_splits:
        stop = offset + count
        blocks32.append((dy[offset:stop].float().t() @ x[offset:stop].float()).reshape(-1))
        offset = stop
    ref32 = torch.cat(blocks32).to(dtype)
    n_diff = int((ref32 != ref64).sum())
    _record(
        "C9 fp32 grouped wgrad ref drifts (fp16)",
        n_diff > 0,
        "{} of {} elements differ from the fp64 reference".format(n_diff, ref64.numel()),
    )


def check_slot_mapping():
    """C10: the substituted indices are the slots the schema says they are."""
    ok = (
        RESULT_SCHEMA[_WGRAD_SLOT] == "weights_grad_concatenated"
        and RESULT_SCHEMA[_BGRAD_SLOT] == "biases_grad_concatenated_or_empty"
        and _WGRAD_SLOT == 2
        and _BGRAD_SLOT == 3
    )
    _record("C10 slot mapping (wgrad=2, bgrad=3)", ok, "schema: {}".format(RESULT_SCHEMA))


def check_device_side(device, device_label):
    """C3-C6: raw vs exact, baseline mode, and three corrupted-output controls."""
    dtype = torch.float16
    rows, in_features, out_features = 16384, 96, 512
    m_splits, bias_enabled = [8029, 8355], True
    holder, x, dy, weights, biases = _make_case(
        dtype, rows, in_features, out_features, m_splits, bias_enabled, device
    )
    comparator = _comparator()

    raw = run_grouped_linear_case(holder, device, exact_baselines=False)
    ref_wgrad = grouped_wgrad_reference(x, dy, m_splits)
    ref_bgrad = grouped_bgrad_reference(dy, m_splits)
    raw_w = raw[_WGRAD_SLOT].detach().float().cpu()
    ratio_w = _ratio(raw_w.to(dtype), ref_wgrad.to(dtype), comparator)
    detail = "wgrad ratio_vs_fp64 {:.4f}".format(ratio_w)

    # The golden side is the one documented to deviate; the device side must not.
    if device_label == "cuda":
        _record("C3 raw NVTE deviates from exact", ratio_w < 0.999, detail)
    else:
        _record("C3 raw device agrees with exact", ratio_w > 0.999, detail)

    if device_label != "cuda":
        _skip("C4 baseline mode", "golden-side only; run with --device cuda")
        _skip("C5 scale error still red", "needs the golden-side baseline")
        _skip("C5b expert swap still red", "needs the golden-side baseline")
        _skip("C6 axis error still red", "needs the golden-side baseline")
        return

    substituted = run_grouped_linear_case(holder, device, exact_baselines=True)
    sub_w = substituted[_WGRAD_SLOT].detach().float().cpu()
    sub_b = substituted[_BGRAD_SLOT].detach().float().cpu()
    _record(
        "C4 baseline mode reproduces exact",
        _ratio(sub_w.to(dtype), ref_wgrad.to(dtype), comparator) == 1.0
        and _ratio(sub_b.to(dtype), ref_bgrad.to(dtype), comparator) == 1.0,
        "slots {} and {} are the reference itself".format(_WGRAD_SLOT, _BGRAD_SLOT),
    )

    scaled = (sub_w * 1.05).to(dtype)
    _record(
        "C5 scale error still red",
        not _judge(scaled, ref_wgrad.to(dtype), comparator, _WGRAD_SLOT),
        "x1.05 on wgrad must not pass an exact baseline",
    )

    # The grouped-specific failure mode: the experts' gradient blocks in the
    # wrong order (an m_splits/weight pairing mistake). A mean-only or
    # tolerance-only guard could miss it; an exact reference cannot.
    block = out_features * in_features
    swapped = torch.cat([sub_w[block:].reshape(-1), sub_w[:block].reshape(-1)]).reshape(-1)
    _record(
        "C5b expert swap still red",
        not _judge(swapped, ref_wgrad.to(dtype), comparator, _WGRAD_SLOT),
        "expert blocks 0<->1 swapped must not pass",
    )

    transposed = sub_w.reshape(-1, out_features, in_features)
    transposed = transposed.transpose(1, 2).reshape(-1)
    _record(
        "C6 axis error still red",
        not _judge(transposed, ref_wgrad.to(dtype), comparator, _WGRAD_SLOT),
        "per-expert transposed wgrad must not pass",
    )


def _ratio(local, remote, comparator):
    """Matched ratio under the same thresholds ATK would apply to this dtype."""
    thresholds = comparator.thresholds[local.dtype]
    local64, remote64 = local.to(torch.float64), remote.to(torch.float64)
    close = (local64 - remote64).abs() <= thresholds["atol"] + thresholds["rtol"] * remote64.abs()
    return float(close.to(torch.float32).mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, choices=["npu", "cuda"])
    args = parser.parse_args()

    print("exact baseline negative controls (grouped_linear)")
    check_policy_lint()
    check_guard_raises()
    check_reference_vs_autograd()
    check_row_spans()
    check_fp32_reference_drifts()
    check_slot_mapping()

    if args.device is None:
        for name in (
            "C3 raw vs exact",
            "C4 baseline mode",
            "C5 scale",
            "C5b expert swap",
            "C6 axis",
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
