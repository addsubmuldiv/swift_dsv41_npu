# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Verify the LayerNormLinear exact grad-slot baselines WITHOUT a CUDA host.

Checks (all pure CPU fp64, runnable on the NPU dev box):

  C1  wgrad/bgrad references == an INDEPENDENT fp64 autograd composition of
      LayerNorm/RMSNorm + Linear (plain torch, no shared code with
      ``_exact_reference``) over the full knob matrix
      {LayerNorm, RMSNorm} x {zc on/off} x {bias on/off}, several shapes.
      Both sides round the LN output to the case dtype once (the module
      materializes it in the case dtype before the GEMM).
  C2  beta is in the wgrad reference: a beta-dropped reference must FAIL the
      module comparator against the correct one (the LayerNormLinear
      counterpart of linear's apply_bias trap).
  C3  RMS-vs-LayerNorm mixup must FAIL the comparator (wrong normalization
      formula must not sneak through).
  C4  zero_centered_gamma flip must FAIL the comparator.
  C5  a 1.05 scale error on the wgrad reference must FAIL the comparator
      (guards against an assertion that cannot fail).
  C6  the golden-side guard: ``_apply_exact_baselines`` must refuse to run on
      a non-CUDA device (reference-vs-reference can never fail).
  C7  bias_grad == exact column sum for every configuration; empty for
      bias-off.
  C8  the A-operand dtype rounding is load-bearing: on fp16 data a
      rounding-less wgrad reference must FAIL the comparator (the
      generalized run's slot-4 failures at 0.94-0.99 were exactly this
      omission).
  C9  lngamma/lnbeta references == an INDEPENDENT fp64 autograd of the norm
      alone, fed the same dln, over the knob matrix; both sides take dln =
      round_dtype(dy @ W) (the module's dgrad buffer dtype), and the
      independent side derives it without ``_exact_reference``.
  C10 the dln dtype rounding is load-bearing for lngamma (the 3#31 analogue
      of C8): a rounding-less lngamma reference must FAIL the comparator.
  C11 a case-dtype xhat is NOT what the device does: an xhat-rounded
      reference must FAIL (it moves matched_ratio to the observed 0.9x band).
  C12 a 1.05 scale error on lngamma must FAIL the comparator (the new slots'
      "cannot be vacuous" guard, C5's analogue).
  C13 RMSNorm leaves the ln_bias_grad slot empty (arity, not a number).
      zc invariance of the two new slots needs no separate probe: C9 compares
      both zc settings against the one reference, so a gamma leak into these
      sums would surface there as a factor-gamma divergence.

The end-to-end substitution glue (NVTE run -> slot replacement on CUDA) cannot
be exercised on this host; the collector's ``device_vs_baseline`` manifest
field is the on-golden-host evidence for that, exactly as in linear's flow.
"""

import sys
from pathlib import Path

import torch

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = str(Path(_HERE).resolve().parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from _exact_reference import (  # noqa: E402
    LngradAccumulator,
    chunked_bgrad,
    chunked_lngrad,
    chunked_wgrad,
    dln_chunk_reference,
    forward_reference,
    norm_forward_reference,
)

from _comparators import MixedToleranceAccuracy, load_design_acc_kwargs  # noqa: E402

RTOL_TIGHT = 1e-12  # fp64 order-of-accumulation noise; any real bug is >> this

SHAPES = [(64, 512, 256), (257, 1024, 512), (4096, 512, 512)]  # (tokens, hidden, out)
KNOBS = [
    ("LayerNorm", False, True),
    ("LayerNorm", True, True),
    ("LayerNorm", True, False),
    ("RMSNorm", False, True),
    ("RMSNorm", True, True),
    ("RMSNorm", True, False),
]  # (normalization, zc, bias)


def independent_reference(x, dy, ln_w, ln_b, eps, normalization, zc, bias_vec, weight_mat):
    """Plain fp64 autograd: norm -> linear, nothing shared with _exact_reference.

    The LN output is rounded to the case dtype once (then upcast back to
    fp64), mirroring the module's dataflow — the LN output tensor feeds the
    GEMM in the case dtype, so the wgrad A operand carries that rounding;
    the baseline reference rounds for the same reason.
    """
    x64 = x.double().requires_grad_(True)
    w64 = weight_mat.double().requires_grad_(True)
    gamma = ln_w.double()
    gamma_eff = (1.0 + gamma) if zc else gamma
    if normalization == "RMSNorm":
        norm_out = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * gamma_eff
    else:
        mu = x64.mean(-1, keepdim=True)
        var = x64.var(-1, unbiased=False, keepdim=True)
        norm_out = (x64 - mu) * torch.rsqrt(var + eps) * gamma_eff + ln_b.double()
    norm_out = norm_out.to(dtype=x.dtype).double()
    out = norm_out @ w64.t()
    if bias_vec is not None:
        out = out + bias_vec.double()
    out.backward(dy.double())
    return w64.grad, x64


def independent_lngrad(x, dln64, ln_w, eps, normalization, zc):
    """Plain fp64 autograd of the norm ALONE, fed the same ``dln``.

    Nothing shared with ``_exact_reference``: torch derives dgamma from its own
    LayerNorm/RMSNorm backward. ``dln64`` is ``round_dtype(dy @ W)``, derived by
    the caller with a plain matmul -- it is the *input* of the module's LN
    backward, so no dY-through-the-GEMM path is involved here (unlike C1, where
    the LN output is the GEMM's operand). ``zc`` deliberately enters this
    side's forward: if the reference wrongly applied ``gamma_eff`` to the
    parameter grads, C9 would show it as a factor-gamma divergence.
    """
    x64 = x.double().requires_grad_(True)
    gamma = ln_w.double().requires_grad_(True)
    gamma_eff = (1.0 + gamma) if zc else gamma
    if normalization == "RMSNorm":
        norm_out = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps) * gamma_eff
    else:
        mu = x64.mean(-1, keepdim=True)
        var = x64.var(dim=-1, unbiased=False, keepdim=True)
        norm_out = (x64 - mu) * torch.rsqrt(var + eps) * gamma_eff
    norm_out.backward(dln64)
    return gamma.grad


def make_comparator():
    acc_kwargs = load_design_acc_kwargs(Path(_HERE) / "layernorm_linear.yaml")
    return MixedToleranceAccuracy(None, **acc_kwargs)


def judged_ok(comparator, actual, expected, slot):
    result = comparator.compute_accuracy_result(
        actual.float(), expected.float(), "output_{}.pt".format(slot)
    )
    return bool(result.result)


def main():
    failures = []

    def check(name, cond, detail=""):
        print(
            "  [{}] {}{}".format(
                "PASS" if cond else "FAIL", name, (" -- " + detail) if (detail and not cond) else ""
            )
        )
        if not cond:
            failures.append(name)

    torch.manual_seed(20260923)
    comparator = make_comparator()

    total = 0
    for tokens, hidden, out_features in SHAPES:
        for normalization, zc, bias_on in KNOBS:
            total += 1
            x = torch.randn(tokens, hidden)
            dy = torch.rand(tokens, out_features) - 0.5
            ln_w = torch.rand(hidden) + 0.5
            ln_b = torch.rand(hidden) - 0.5
            weight = torch.randn(out_features, hidden) * 0.05
            bias_vec = torch.rand(out_features) - 0.5 if bias_on else None
            eps = 1e-5

            # ours
            ln_out = norm_forward_reference(
                x,
                ln_w,
                ln_b if normalization != "RMSNorm" else None,
                eps,
                normalization,
                zc,
            ).to(dtype=x.dtype)
            wgrad = chunked_wgrad(dy, ln_out)
            bgrad = chunked_bgrad(dy) if bias_on else None

            # independent
            wgrad_ref, _ = independent_reference(
                x, dy, ln_w, ln_b, eps, normalization, zc, bias_vec, weight
            )

            ok = torch.allclose(wgrad, wgrad_ref, rtol=RTOL_TIGHT, atol=1e-9)
            if not ok:
                rel = ((wgrad - wgrad_ref).abs() / wgrad_ref.abs().clamp_min(1e-12)).max()
                check(
                    "C1 wgrad {} zc={} bias={} [{}]".format(
                        normalization, zc, bias_on, (tokens, hidden, out_features)
                    ),
                    False,
                    "max rel dev {:.3e}".format(rel.item()),
                )
                continue
            if bias_on:
                db_ref = dy.double().sum(0)
                check(
                    "C7 bgrad {} [{}]".format(normalization, (tokens, hidden, out_features)),
                    torch.allclose(bgrad, db_ref, rtol=RTOL_TIGHT, atol=1e-9),
                )
    check("C1 wgrad == independent fp64 autograd ({} configs)".format(total), not failures)

    # --- C9: lngamma/lnbeta == independent fp64 autograd (fp16 case dtype) ---
    # Same knob matrix; the shared reference is driven through LngradAccumulator
    # by chunked_lngrad, the independent side by torch's own norm backward.
    lngrad_bad = []
    for tokens, hidden, out_features in SHAPES:
        for normalization, zc, _bias_on in KNOBS:
            torch.manual_seed(1000 + tokens + hidden + out_features)
            x16 = (torch.randn(tokens, hidden) * 0.7).half()
            dy16 = (torch.rand(tokens, out_features) - 0.5).half()
            w16 = (torch.randn(out_features, hidden) * 0.05).half()
            g16 = (torch.rand(hidden) + 0.5).half()
            eps = 1e-5
            dln = (dy16.double() @ w16.double()).to(torch.float16).double()
            want_gamma, want_beta = chunked_lngrad(
                x16, dy16, w16, eps, normalization, round_dln_dtype=torch.float16
            )
            got_gamma = independent_lngrad(x16, dln, g16, eps, normalization, zc).double()
            got_beta = dln.sum(0)
            label = "C9 {} zc={} [{}]".format(normalization, zc, (tokens, hidden, out_features))
            if not torch.allclose(want_gamma, got_gamma, rtol=RTOL_TIGHT, atol=1e-9):
                lngrad_bad.append(label + " lngamma")
            if not torch.allclose(want_beta, got_beta, rtol=RTOL_TIGHT, atol=1e-9):
                lngrad_bad.append(label + " lnbeta")
    check(
        "C9 lngamma/lnbeta == independent fp64 autograd ({} configs)".format(
            len(SHAPES) * len(KNOBS)
        ),
        not lngrad_bad,
        "; ".join(lngrad_bad[:3]),
    )

    # --- C15-C17: the forward_output baseline (slot 0) ----------------------
    # C15 compares it against the independent fp64 autograd of norm -> linear
    # (the same composition C1 uses, reading this time the *forward* output),
    # with both sides rounding the LN output once into the case dtype.
    fwd_bad = []
    for tokens, hidden, out_features in SHAPES:
        torch.manual_seed(2000 + tokens + hidden + out_features)
        x16 = (torch.randn(tokens, hidden) * 0.7).half()
        w16 = (torch.randn(out_features, hidden) * 0.05).half()
        g16 = (torch.rand(hidden) + 0.5).half()
        b16 = (torch.rand(out_features) - 0.5).half()
        bias_flag = bool(hidden % 2)  # exercise both bias paths across shapes
        eps = 1e-5
        slot = torch.empty(tokens, out_features, dtype=torch.float16)
        ln_out = norm_forward_reference(
            x16, g16, None, eps, "RMSNorm", False, out_dtype=torch.float16
        )
        ref = forward_reference(slot, ln_out, w16, b16 if bias_flag else None, apply_bias=bias_flag)
        # independent: autograd through norm in fp64, then the GEMM by hand on
        # the once-rounded LN output
        x64 = x16.double()
        norm_out = x64 * torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps) * g16.double()
        want = (norm_out.to(torch.float16).double() @ w16.double().t()).to(torch.float16)
        if bias_flag:
            want = (want.double() + b16.double()).to(torch.float16)
        if not torch.equal(ref, want):
            fwd_bad.append("C15 forward [{}]".format((tokens, hidden, out_features)))
    check(
        "C15 forward ref == independent fp64 GEMM ({} shapes)".format(len(SHAPES)),
        not fwd_bad,
        "; ".join(fwd_bad[:3]),
    )

    tokens, hidden, out_features = 4096, 512, 512
    torch.manual_seed(33)
    x16 = (torch.randn(tokens, hidden) * 2).half()
    w16 = (torch.randn(out_features, hidden) * 0.2).half()
    g16 = (torch.rand(hidden) + 0.5).half()
    b16 = (torch.rand(out_features) - 0.5).half()
    eps = 1e-5
    slot = torch.empty(tokens, out_features, dtype=torch.float16)
    ln_out16 = norm_forward_reference(
        x16, g16, None, eps, "LayerNorm", True, out_dtype=torch.float16
    )
    fwd = forward_reference(slot, ln_out16, w16, b16, apply_bias=True)
    # C16: the apply_bias trap -- dropping the bias must FAIL the comparator
    # (this adapter pins return_bias=False, so the bias is always in the output)
    check(
        "C16 bias-dropped forward FAILS comparator",
        not judged_ok(
            comparator, forward_reference(slot, ln_out16, w16, None, apply_bias=False), fwd, 0
        ),
        "bias-less reference passed -- the apply_bias trap is open",
    )
    # C17: a scale error on slot 0 must FAIL (the baseline is not a rubber stamp)
    check(
        "C17 forward x1.05 FAILS comparator",
        not judged_ok(comparator, fwd * 1.05, fwd, 0),
    )

    # --- negative controls on one representative config (LayerNorm, zc, bias) ---
    tokens, hidden, out_features = 4096, 512, 512
    x = torch.randn(tokens, hidden)
    dy = torch.rand(tokens, out_features) - 0.5
    ln_w = torch.rand(hidden) + 0.5
    ln_b = torch.rand(hidden) - 0.5
    eps = 1e-5

    ln_out = norm_forward_reference(x, ln_w, ln_b, eps, "LayerNorm", True)
    wgrad = chunked_wgrad(dy, ln_out.to(dtype=x.dtype))

    # C8: the A-operand dtype rounding is load-bearing. On fp16 data (the
    # case dtype where the generalized run failed slot 4 at 0.94-0.99), a
    # rounding-less wgrad reference must FAIL the comparator against the
    # correct one — the fp16 ln_out rounding leaves ~sqrt(T)*|dy|*eps_fp16
    # (~0.005 at 4096 tokens) of absolute offset on near-zero elements.
    x16 = (torch.randn(tokens, hidden) * 3).half()
    dy16 = (torch.rand(tokens, out_features) - 0.5).half()
    ln16_w = (torch.rand(hidden) + 0.5).half()
    ln16_b = (torch.rand(hidden) - 0.5).half()
    ln_out16 = norm_forward_reference(x16, ln16_w, ln16_b, eps, "LayerNorm", True)
    wgrad16_correct = chunked_wgrad(dy16, ln_out16.to(dtype=torch.float16))
    wgrad16_noround = chunked_wgrad(dy16, ln_out16)
    check(
        "C8 rounding-less wgrad FAILS comparator (fp16)",
        not judged_ok(comparator, wgrad16_noround, wgrad16_correct, 4),
        "rounding-less reference passed -- the A-operand rounding is not load-bearing",
    )

    # C2: beta dropped
    ln_out_nobeta = norm_forward_reference(x, ln_w, None, eps, "LayerNorm", True)
    wgrad_nobeta = chunked_wgrad(dy, ln_out_nobeta)
    check(
        "C2 beta-dropped wgrad FAILS comparator",
        not judged_ok(comparator, wgrad_nobeta, wgrad, 4),
        "beta-less reference passed -- the apply_bias-class trap is open",
    )
    # C3: RMS formula used for a LayerNorm case
    ln_out_rms = norm_forward_reference(x, ln_w, None, eps, "RMSNorm", True)
    wgrad_rms = chunked_wgrad(dy, ln_out_rms)
    check(
        "C3 RMS-vs-LN mixup FAILS comparator",
        not judged_ok(comparator, wgrad_rms, wgrad, 4),
    )
    # C4: zc flip
    ln_out_nozc = norm_forward_reference(x, ln_w, ln_b, eps, "LayerNorm", False)
    wgrad_nozc = chunked_wgrad(dy, ln_out_nozc)
    check(
        "C4 zc-flip wgrad FAILS comparator",
        not judged_ok(comparator, wgrad_nozc, wgrad, 4),
    )
    # C5: scale error
    check(
        "C5 wgrad x1.05 FAILS comparator",
        not judged_ok(comparator, wgrad * 1.05, wgrad, 4),
        "comparator cannot fail -- the assertions above are vacuous",
    )
    # C5b: scale error on bgrad
    bgrad = chunked_bgrad(dy)
    check(
        "C5b bgrad x1.05 FAILS comparator",
        not judged_ok(comparator, bgrad * 1.05, bgrad, 5),
    )

    # --- C10-C13: the LayerNorm parameter-grad slots ---
    tokens, hidden, out_features = 4096, 512, 512
    torch.manual_seed(31)
    x16 = (torch.randn(tokens, hidden) * 2).half()
    dy16 = (torch.rand(tokens, out_features) - 0.5).half()
    w16 = (torch.randn(out_features, hidden) * 0.2).half()
    gamma16, beta16 = chunked_lngrad(
        x16, dy16, w16, eps, "LayerNorm", round_dln_dtype=torch.float16
    )
    # C10: the dln dtype rounding is load-bearing, exactly as in C8 for the
    # wgrad A operand. The module's LN backward reads a dgrad buffer of the
    # case dtype; a reference that skips that rounding is off by ~sqrt(T) * ULP
    # on the sum, i.e. on the same 0.9x band the generalized run reported.
    gamma_noround, _ = chunked_lngrad(x16, dy16, w16, eps, "LayerNorm", round_dln_dtype=None)
    check(
        "C10 rounding-less lngamma FAILS comparator (fp16)",
        not judged_ok(comparator, gamma_noround, gamma16, 2),
        "rounding-less lngamma passed -- the dln dtype rounding is not load-bearing",
    )
    # C11: a case-dtype xhat is not what the device does (its normalized input
    # stays fp32; measured on the device: the fp64-xhat reference matches, the
    # case-dtype-xhat one lands on the same 0.9x band).
    weight64 = w16.double()
    xhat16_acc = LngradAccumulator(hidden, eps, "LayerNorm", xhat_dtype=torch.float16)
    for start in range(0, tokens, 512):
        xhat16_acc.update(
            x16[start : start + 512],
            dln_chunk_reference(dy16[start : start + 512], weight64, torch.float16),
        )
    gamma_xhat16, _ = xhat16_acc.value()
    check(
        "C11 case-dtype-xhat lngamma FAILS comparator",
        not judged_ok(comparator, gamma_xhat16, gamma16, 2),
    )
    # C12: a real axis/scale defect must still be red against the exact
    # reference (the baseline is not a rubber stamp). Transposing the two
    # operands of the dgamma sum is the analogue of linear's axis-swap probe.
    check(
        "C12 lngamma x1.05 FAILS comparator",
        not judged_ok(comparator, gamma16 * 1.05, gamma16, 2),
        "comparator cannot fail -- the C10/C11 assertions are vacuous",
    )
    # C13: RMSNorm has no LN bias parameter, so the slot the adapter leaves
    # untouched must pass through `to_slot` empty (arity, not a number).
    from _exact_reference import to_slot

    _, rms_beta = chunked_lngrad(x16, dy16, w16, eps, "RMSNorm", round_dln_dtype=torch.float16)
    check(
        "C13 RMSNorm ln_bias_grad stays an empty passthrough",
        to_slot(rms_beta, torch.empty(0), "ln_bias_grad_or_empty").numel() == 0,
    )

    # C14: the four exact slots point at the right RESULT_SCHEMA names. The
    # slot wiring itself can only be executed on a CUDA host (the guard
    # refuses anything else), so this pins the one thing a wiring change could
    # silently get wrong.
    try:
        from function_layernorm_linear import (  # noqa: F401
            RESULT_SCHEMA,
            _BGRAD_SLOT,
            _FORWARD_SLOT,
            _LNBETA_SLOT,
            _LNGAMMA_SLOT,
            _WGRAD_SLOT,
        )

        check(
            "C14 exact slots map to the right RESULT_SCHEMA names",
            RESULT_SCHEMA[_FORWARD_SLOT] == "forward_output"
            and RESULT_SCHEMA[_LNGAMMA_SLOT] == "ln_weight_grad"
            and RESULT_SCHEMA[_LNBETA_SLOT] == "ln_bias_grad_or_empty"
            and RESULT_SCHEMA[_WGRAD_SLOT] == "weight_grad"
            and RESULT_SCHEMA[_BGRAD_SLOT] == "bias_grad_or_empty",
            "slot indices drifted: {}".format(
                [_FORWARD_SLOT, _LNGAMMA_SLOT, _LNBETA_SLOT, _WGRAD_SLOT, _BGRAD_SLOT]
            ),
        )
    except ImportError as exc:  # pragma: no cover - ATK import surface
        check("C14 exact slots map to the right RESULT_SCHEMA names", False, str(exc))

    # C6: the golden-side guard
    try:
        from function_layernorm_linear import _apply_exact_baselines

        _apply_exact_baselines(
            slots=[torch.randn(2, 2)] * 7,
            x=x,
            dy=dy,
            ln_w=ln_w,
            ln_b=ln_b,
            weight_mat=torch.randn(2, 2),
            bias_vec=torch.randn(2),
            normalization="LayerNorm",
            bias_enabled=True,
            zc=True,
            eps=eps,
            device="npu",
        )
        check("C6 guard refuses non-CUDA device", False, "no exception raised")
    except RuntimeError as exc:
        check(
            "C6 guard refuses non-CUDA device",
            "GOLDEN-SIDE ONLY" in str(exc),
            "unexpected error: {}".format(exc),
        )

    print()
    if failures:
        print("FAILED: {}".format(failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
