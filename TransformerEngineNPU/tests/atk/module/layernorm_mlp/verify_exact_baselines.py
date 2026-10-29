# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Negative controls for the exact baseline of the ``layernorm_mlp`` guard.

The change this verifies replaces ``forward_output`` in the golden with an
exact fp64 materialized-chain reference (``mlp_forward_reference``: ln_out,
fc1 pre-activation and activation output each rounded once into the case
dtype; the fc2 output rounds once). A baseline that is synthesised rather
than measured can fail in two opposite ways, and only the second is
dangerous:

* the baseline is wrong -> the device is judged against nonsense (visible:
  every case red, or a real defect washed out);
* the *device under test* is also replaced -> the ATK comparison degenerates
  into reference-vs-reference, every case green forever, a real regression
  shipped silently.

So these checks are deliberately adversarial. The mlp-specific trap the
controls guard: the **gelu variant** — TENPU computes the tanh approximation,
and a reference built on the erf form (or on silu where the case ran gelu)
would put a systematic bias into the golden itself.

Checks C1, C2, C7-C10 are device-free and always run. C3-C6 need a device
(``--device cuda`` for the golden-side claims; ``npu`` is accepted but only
the raw-output half is meaningful there) and are reported as SKIP, never as a
silent PASS, when no device is available.
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
    activation_forward_reference,
    mlp_forward_reference,
    norm_forward_reference,
)
from function_layernorm_mlp import (  # noqa: E402
    _FORWARD_SLOT,
    RESULT_SCHEMA,
    _golden_side_guard,
    run_layernorm_mlp_case,
)

_DESIGN = _HERE / "layernorm_mlp.yaml"

_RESULTS = []


def _record(name, passed, detail=""):
    _RESULTS.append((name, passed, detail))
    print("  {:<40} {:<5} {}".format(name, "PASS" if passed else "FAIL", detail))


def _skip(name, detail):
    print("  {:<40} {:<5} {}".format(name, "SKIP", detail))


def _comparator():
    return MixedToleranceAccuracy(None, **load_design_acc_kwargs(_DESIGN))


def _judge(local, remote, comparator, index):
    return bool(
        comparator.compute_accuracy_result(local, remote, "output_{}.pt".format(index)).result
    )


def _make_case(dtype, shape, ffn, eps, normalization, activation, bias_on, zc, device, seed=0):
    torch.manual_seed(seed)
    hidden = shape[-1]
    x = torch.rand(*shape, dtype=dtype) * 5.2 - 2.6
    dy = torch.rand(*shape, dtype=dtype) - 0.5
    ln_w = torch.rand(hidden, dtype=dtype) + 0.5
    ln_b = torch.rand(hidden, dtype=dtype) - 0.5
    w1 = torch.rand(ffn, hidden, dtype=dtype) - 0.5
    b1 = torch.rand(ffn, dtype=dtype) - 0.5
    w2 = torch.rand(hidden, ffn, dtype=dtype) - 0.5
    b2 = torch.rand(hidden, dtype=dtype) - 0.5
    holder = SimpleNamespace(
        args=(
            x.to(device),
            dy.to(device),
            ln_w.to(device),
            ln_b.to(device),
            w1.to(device),
            b1.to(device),
            w2.to(device),
            b2.to(device),
            eps,
            normalization,
            activation,
            bias_on,
            zc,
        )
    )
    return holder, x, ln_w, ln_b, w1, b1, w2, b2


def _switch_sites():
    """Every ``exact_baselines=`` keyword across the module tree, by file."""
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
        "linear/function_linear.py": 1,
        "linear/collect_nvte_golden.py": 1,
        "linear/verify_exact_baselines.py": 1,
        "layernorm_linear/function_layernorm_linear.py": 1,
        "layernorm_linear/collect_nvte_golden.py": 1,
        "grouped_linear/function_grouped_linear.py": 1,
        "grouped_linear/collect_nvte_golden.py": 1,
        "grouped_linear/verify_exact_baselines.py": 1,
        # layernorm_mlp: NVTE's forward departs from the materialized chain
        # (fused gelu epilogue, skinny-N GEMM shapes) while TENPU reproduces
        # it, so the forward slot gets the same golden-side treatment.
        "layernorm_mlp/function_layernorm_mlp.py": 1,
        "layernorm_mlp/collect_nvte_golden.py": 1,
        "layernorm_mlp/verify_exact_baselines.py": 1,
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

    adapter = (_HERE / "function_layernorm_mlp.py").read_text(encoding="utf-8")
    npu_branch = [
        line for line in adapter.splitlines() if 'run_layernorm_mlp_case(input_data, "npu")' in line
    ]
    clean_npu = len(npu_branch) == 1 and "exact_baselines" not in npu_branch[0].replace(" ", "")
    _record("C1c NPU branch passes no switch", clean_npu, "device under test stays raw")


def check_guard_raises():
    """C2: the guard rejects anything that is not the golden host."""
    cpu = torch.zeros(2)
    for device, tensors in (("npu", {"t": cpu}), ("npu:0", {"t": cpu}), ("cuda", {"t": cpu})):
        try:
            _golden_side_guard(device, tensors)
        except RuntimeError:
            _record("C2 guard rejects {!r}".format(device), True, "")
        else:
            _record("C2 guard rejects {!r}".format(device), False, "returned instead of raising")


def _independent_chain(x, ln_w, ln_b, w1, b1, w2, b2, eps, norm, act, bias_on, zc, dtype):
    """The same materialized dataflow through independent torch ops (F.* / manual)."""
    hidden = x.shape[-1]
    gamma64 = ln_w.double() + (1.0 if zc else 0.0)
    if norm == "RMSNorm":
        ln64 = x.double() * x.double().pow(2).mean(dim=-1, keepdim=True).add_(eps).rsqrt() * gamma64
    else:
        ln64 = torch.nn.functional.layer_norm(
            x.double(), (hidden,), gamma64, ln_b.double() if ln_b is not None else None, eps
        )
    ln_r = ln64.to(dtype).double()
    z = torch.nn.functional.linear(ln_r, w1.double(), b1.double() if bias_on else None)
    z = z.to(dtype).double()
    if act == "silu":
        a = torch.nn.functional.silu(z)
    else:
        a = torch.nn.functional.gelu(z, approximate="tanh")
    a = a.to(dtype).double()
    out = torch.nn.functional.linear(a, w2.double(), b2.double() if bias_on else None)
    return out.to(dtype)


def check_reference_vs_independent():
    """C7: the reference equals an independently formulated chain, bitwise.

    The independent path uses ``F.layer_norm`` / ``F.linear`` / ``F.silu`` /
    ``F.gelu(tanh)`` — different ops, same materialized semantics — so bitwise
    agreement after the per-stage roundings is evidence, not a tautology.
    Sweeps the knob matrix (norm x activation x bias x zc).
    """
    dtype = torch.bfloat16
    shape, ffn = [64, 96], 256
    bad = []
    for norm in ("LayerNorm", "RMSNorm"):
        for act in ("silu", "gelu"):
            for bias_on in (True, False):
                for zc in (True, False):
                    torch.manual_seed(0)
                    x = torch.rand(*shape, dtype=dtype) * 5.2 - 2.6
                    ln_w = torch.rand(shape[-1], dtype=dtype) + 0.5
                    ln_b = torch.rand(shape[-1], dtype=dtype) - 0.5
                    w1 = torch.rand(ffn, shape[-1], dtype=dtype) - 0.5
                    b1 = torch.rand(ffn, dtype=dtype) - 0.5
                    w2 = torch.rand(shape[-1], ffn, dtype=dtype) - 0.5
                    b2 = torch.rand(shape[-1], dtype=dtype) - 0.5
                    slot = torch.empty(*shape, dtype=dtype)
                    ref = mlp_forward_reference(
                        slot,
                        x.reshape(-1, shape[-1]),
                        ln_w,
                        ln_b,
                        w1,
                        b1,
                        w2,
                        b2,
                        1e-5,
                        norm,
                        act,
                        bias_on,
                        zc,
                    )
                    ind = _independent_chain(
                        x, ln_w, ln_b, w1, b1, w2, b2, 1e-5, norm, act, bias_on, zc, dtype
                    )
                    if not torch.equal(ref.reshape(shape), ind):
                        bad.append((norm, act, bias_on, zc))
    _record("C7 reference == independent chain (16 combos)", not bad, "bad: {}".format(bad))


def check_activation_table():
    """C8: the activation formulas, and a garbage name fails closed."""
    z = torch.linspace(-6, 6, 101, dtype=torch.float64)
    silu_ok = torch.equal(activation_forward_reference(z, "silu"), z * torch.sigmoid(z))
    tanh_ref = 0.5 * z * (1.0 + torch.tanh(0.7978845608028654 * (z + 0.044715 * z.pow(3))))
    gelu_ok = torch.equal(activation_forward_reference(z, "gelu"), tanh_ref)
    _record("C8 activation formulas (silu, gelu-tanh)", silu_ok and gelu_ok, "")
    try:
        activation_forward_reference(z, "relu")
    except RuntimeError:
        _record("C8b unknown activation raises", True, "")
    else:
        _record("C8b unknown activation raises", False, "returned instead of raising")


def check_variant_negative():
    """C9: the gelu variant is pinned to the device's (tanh approximation).

    TENPU computes tanh-gelu (``npu_gelu(approximate="tanh")``) and C7 pins
    the reference bitwise to that formulation. What a wrong-variant (erf)
    golden would cost is *measured* here rather than assumed: the chain
    difference stays under the module's 0.05 gate (ratio 1.0000, and 1.0000
    at the tight 0.0156 gate too — the difference concentrates on large
    elements where rtol covers it), so it is a **margin hazard**, not a
    verdict hazard: it consumes the absolute-tolerance budget of real
    elements while the guard stays green. The control keeps that visible
    instead of silently green.
    """
    dtype = torch.bfloat16
    shape, ffn = [48, 1024], 2048
    torch.manual_seed(0)
    x = torch.rand(*shape, dtype=dtype) * 5.2 - 2.6
    ln_w = torch.rand(shape[-1], dtype=dtype) + 0.5
    ln_b = torch.rand(shape[-1], dtype=dtype) - 0.5
    w1 = torch.rand(ffn, shape[-1], dtype=dtype) - 0.5
    b1 = torch.rand(ffn, dtype=dtype) - 0.5
    w2 = torch.rand(shape[-1], ffn, dtype=dtype) - 0.5
    b2 = torch.rand(shape[-1], dtype=dtype) - 0.5
    slot = torch.empty(*shape, dtype=dtype)

    tanh_chain = mlp_forward_reference(
        slot,
        x.reshape(-1, shape[-1]),
        ln_w,
        ln_b,
        w1,
        b1,
        w2,
        b2,
        1e-6,
        "LayerNorm",
        "gelu",
        True,
        True,
    )
    ln64 = norm_forward_reference(x.reshape(-1, shape[-1]), ln_w, ln_b, 1e-6, "LayerNorm", True)
    z = (ln64.to(dtype).double() @ w1.double().t() + b1.double()).to(dtype).double()
    erf_act = (0.5 * z * (1.0 + torch.erf(z / 2**0.5))).to(dtype).double()
    erf_chain = (erf_act @ w2.double().t() + b2.double()).to(dtype).reshape(shape)

    diff = (erf_chain.double() - tanh_chain.double()).abs()
    n_floor = int((diff > 0.05).sum())
    _record(
        "C9 variant difference exceeds the atol floor",
        n_floor > 0,
        "{} elements: a wrong-variant golden eats real margin".format(n_floor),
    )
    _record(
        "C9b the two variants differ",
        diff.max().item() > 0.0,
        "max |erf - tanh| chain difference = {:.4f}".format(diff.max().item()),
    )


def check_slot_mapping():
    """C10: the substituted index is the slot the schema says it is."""
    ok = _FORWARD_SLOT == 0 and RESULT_SCHEMA[_FORWARD_SLOT] == "forward_output"
    _record("C10 slot mapping (forward=0)", ok, "schema[0]: {}".format(RESULT_SCHEMA[0]))


def check_device_side(device, device_label):
    """C3-C6: raw vs exact, baseline mode, and two corrupted-output controls."""
    dtype = torch.bfloat16
    # g83-like: the shape family where NVTE's own output measurably departs.
    shape, ffn = [48, 1024], 2048
    holder, x, ln_w, ln_b, w1, b1, w2, b2 = _make_case(
        dtype, shape, ffn, 1e-6, "LayerNorm", "gelu", True, True, device
    )
    comparator = _comparator()

    raw = run_layernorm_mlp_case(holder, device, exact_baselines=False)
    slot = torch.empty(*shape, dtype=dtype)
    ref = mlp_forward_reference(
        slot,
        x.reshape(-1, shape[-1]),
        ln_w,
        ln_b,
        w1,
        b1,
        w2,
        b2,
        1e-6,
        "LayerNorm",
        "gelu",
        True,
        True,
    )
    raw_f = raw[_FORWARD_SLOT].detach().cpu()
    result = comparator.compute_accuracy_result(raw_f, ref, "output_0.pt")
    detail = "forward vs exact: {}".format(result.error_info)

    # Post the general_gemm bias fusion fix TENPU reproduces the chain; the
    # golden side is the one documented to depart (fused gelu / skinny-N GEMM).
    if device_label == "cuda":
        _record("C3 raw NVTE depiates from exact", not result.result, detail)
    else:
        _record("C3 raw device agrees with exact", result.result, detail)

    if device_label != "cuda":
        _skip("C4 baseline mode", "golden-side only; run with --device cuda")
        _skip("C5 scale error still red", "needs the golden-side baseline")
        _skip("C6 missing-bias still red", "needs the golden-side baseline")
        return

    substituted = run_layernorm_mlp_case(holder, device, exact_baselines=True)
    sub_f = substituted[_FORWARD_SLOT].detach().cpu()
    _record(
        "C4 baseline mode reproduces exact",
        torch.equal(sub_f, ref),
        "slot {} is the reference itself".format(_FORWARD_SLOT),
    )

    scaled = (sub_f.double() * 1.05).to(dtype)
    _record(
        "C5 scale error still red",
        not _judge(scaled, ref, comparator, _FORWARD_SLOT),
        "x1.05 on the forward must not pass an exact baseline",
    )
    # The forward reference's own trap: dropping the fc2 bias. Mimics exactly
    # the mistake the bias handling exists to prevent.
    no_bias = (sub_f.double() - b2.double()).to(dtype)
    _record(
        "C6 missing-bias still red",
        not _judge(no_bias, ref, comparator, _FORWARD_SLOT),
        "a bias-dropped golden must not pass",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, choices=["npu", "cuda"])
    args = parser.parse_args()

    print("exact baseline negative controls (layernorm_mlp)")
    check_policy_lint()
    check_guard_raises()
    check_reference_vs_independent()
    check_activation_table()
    check_variant_negative()
    check_slot_mapping()

    if args.device is None:
        for name in ("C3 raw vs exact", "C4 baseline mode", "C5 scale", "C6 missing bias"):
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
