# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Per-slot accuracy judge for ``te.Linear``: NPU/NVTE output vs exact fp64.

Companion to ``diagnose_bias_grad.py``. Where that one judges the ``db``
baseline, this one answers **which side deviates** for any slot, by comparing a
single device's output against a reference that is far more accurate than
either implementation — a chunked fp64 accumulation on CPU.

Why fp64 is exact here: for bf16/fp16 operands the products are exact in fp32
(8x8 and 11x11 mantissa bits fit the 24-bit fp32 mantissa), so the only error in
the reference is the accumulation order, which at fp64 is ~1e-16 — orders of
magnitude below the bf16/fp16 ULP. A ratio of 1.0 against this reference means
"this device reduces as accurately as the dtype allows".

Run on the **NPU host** (``--device npu``) and on the **CUDA/golden host**
(``--device cuda``) and compare the two reports:

* both sides 1.0 -> the residual ATK failure is a comparator/baseline artefact,
  not a numerical defect of either implementation.
* one side < 0.99 -> that side is the deviant; the slot and the shape tell you
  which kernel to look at.

Everything is compared row-chunk by row-chunk, so peak host memory stays at one
chunk of fp64 — the listed shapes go up to 4M tokens, where materialising a
full fp64 ``[tokens, N]`` reference would need ~9 GB.

Usage::

    python tests/atk/module/linear/diagnose_slot_vs_fp64.py --device npu
    python tests/atk/module/linear/diagnose_slot_vs_fp64.py --device cuda

The default shape list mirrors the fp16/bf16 ``all_linear`` cases whose
``output_3`` (weight_grad) failed at matched_ratio 0.76-0.98: weight_grad is the
slot whose reduction dimension is the token count, so the list sweeps token
counts up to 4M with small out_features.

The ``id73-like`` / ``id137-like`` entries were added for the two cases still red
on ``output_0`` (``forward_output``). Both were measured clean on the NPU host
(ratio_vs_fp64 1.0000, 99.9% of elements correctly rounded, 0 elements beyond
2 ULP), so judging the other side is what remains. They carry explicit ``attrs``
because their forward outputs do **not** include the bias: ``id73-like`` has
``return_bias=True`` and ``id137-like`` has ``bias_enabled=False``. NVTE sets
``self.apply_bias = bias and not return_bias`` (``pytorch/module/linear.py:1497``),
so a reference that always adds ``bias64`` would report a spurious ~0.5 absolute
error (the bias magnitude) and blame the device. Keep ``apply_bias`` honest.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_HERE = Path(__file__).resolve().parent
# The shared comparators and exact-reference helpers live one level up.
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _exact_reference import (  # noqa: E402
    BgradAccumulator,
    DEFAULT_CHUNK_ROWS,
    WgradAccumulator,
)
from function_linear import RESULT_SCHEMA, run_linear_case  # noqa: E402

BF16 = torch.bfloat16
FP16 = torch.float16

# Shape spec: (label, dtype, input shape, out_features, attrs); in_features =
# last dim. ``attrs`` is None for the historical entries, which all mean
# bias_enabled=True / return_bias=False / consume=False / split "none" -- the
# only combination under which the forward output carries the bias.
SHAPES = [
    ("id32-like", BF16, [24, 24, 48], 128, None),
    ("id87-like", BF16, [64, 64, 16, 48], 48, None),
    ("id95-like", FP16, [1024, 16, 16], 1024, None),
    ("id126-like", FP16, [16, 256, 64, 1024], 1024, None),
    ("id133-like", FP16, [2048, 1024, 96], 512, None),
    ("id136-like", BF16, [2048, 2048, 128], 256, None),
    # The two still-red output_0 cases (x range [-5, 5]).
    (
        "id73-like",
        FP16,
        [24, 24, 2048],
        96,
        dict(return_bias=True, parameters_split="dict2"),
    ),
    (
        "id137-like",
        FP16,
        [256, 64, 24, 2048],
        2048,
        dict(bias_enabled=False, consume_returned_bias=True, parameters_split="tuple2"),
    ),
]

# What the historical entries mean, used when ``attrs`` is absent.
_DEFAULT_ATTRS = dict(
    bias_enabled=True,
    return_bias=False,
    consume_returned_bias=False,
    parameters_split="none",
)

_CHUNK = DEFAULT_CHUNK_ROWS  # rows per step; bounds the fp64 workspace to ~one chunk


class _Stats:
    """Streaming error statistics over chunks (never holds the full tensor)."""

    def __init__(self, tolerance):
        self.rtol, self.atol = tolerance
        self.total = 0
        self.passed = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.chunk_medians = []

    def update(self, got, ref):
        """Accumulate one chunk; ``ref`` is fp64, ``got`` any device dtype."""
        got = got.detach().float().double().cpu().reshape(ref.shape)
        abs_error = (got - ref).abs()
        rel_error = abs_error / ref.abs().clamp_min(1e-30)
        self.total += ref.numel()
        self.passed += int((abs_error <= self.atol + self.rtol * ref.abs()).sum())
        self.max_abs = max(self.max_abs, abs_error.max().item())
        self.max_rel = max(self.max_rel, rel_error.max().item())
        self.chunk_medians.append(rel_error.median().item())

    def update_plain(self, got, ref):
        """Same, for a reference accumulated whole (weight_grad, bias_grad)."""
        self.update(got, ref)

    def line(self, tag):
        if self.total == 0:
            return "  {:<22s} EMPTY (numel=0)".format(tag)
        med = sorted(self.chunk_medians)[len(self.chunk_medians) // 2]
        return "  {:<22s} med_rel={:.3e} max_rel={:.3e} max_abs={:.3e} ratio_vs_fp64={:.4f}".format(
            tag, med, self.max_rel, self.max_abs, self.passed / self.total
        )


def _tolerance(dtype):
    """ATK mixed_tolerance defaults for this dtype (rtol, atol)."""
    if dtype == BF16:
        return 1.56e-2, 1.56e-2
    if dtype == FP16:
        return 1.95e-3, 1.95e-3
    return 9.77e-4, 1.53e-5


def run_one(label, dtype, inp_shape, out_features, attrs, device):
    attrs = dict(_DEFAULT_ATTRS, **(attrs or {}))
    bias_enabled = attrs["bias_enabled"]
    return_bias = attrs["return_bias"]
    # NVTE: apply_bias = bias and not return_bias (linear.py:1497). With
    # return_bias=True the bias is *returned* for the caller to fuse, so the
    # forward output does NOT contain it -- adding bias64 here would manufacture
    # a bias-sized error and wrongly blame the device under test.
    apply_bias = bias_enabled and not return_bias

    in_features = inp_shape[-1]
    torch.manual_seed(0)
    x = torch.rand(*inp_shape, dtype=dtype) * 10 - 5
    weight = torch.rand(out_features, in_features, dtype=dtype) - 0.5
    bias = torch.rand(out_features, dtype=dtype) - 0.5
    dy = torch.rand(*inp_shape[:-1], out_features, dtype=dtype) - 0.5
    grad_returned_bias = torch.zeros(out_features, dtype=dtype)
    weight64, bias64 = weight.double(), bias.double()

    # ``run_linear_case`` only reads ``input_data.args``; a namespace is enough.
    holder = SimpleNamespace(
        args=(
            x.to(device),
            weight.to(device),
            bias.to(device),
            dy.to(device),
            grad_returned_bias.to(device),
            bias_enabled,
            return_bias,
            attrs["consume_returned_bias"],
            attrs["parameters_split"],
        )
    )
    # This script is the arbiter for "which side deviates", so it must judge the
    # device's RAW output: exact_baselines stays off, or it would compare a
    # reference against itself and report 1.0000 unconditionally.
    outputs = run_linear_case(holder, device, exact_baselines=False)

    x2d = x.reshape(-1, in_features)
    dy2d = dy.reshape(-1, out_features)
    # The adapter returns the forward/input-grad outputs with the input's leading
    # dimensions, so flatten to [tokens, :] before slicing rows.
    forward_out = outputs[0].reshape(-1, out_features)
    input_grad_out = outputs[2].reshape(-1, in_features)
    tokens = x2d.shape[0]
    tolerance = _tolerance(dtype)

    forward = _Stats(tolerance)
    dgrad = _Stats(tolerance)
    # Same accumulators the golden baseline uses (_exact_reference.py), so the
    # judge and the baseline cannot drift into two definitions.
    wgrad_acc = WgradAccumulator(out_features, in_features)
    bgrad_acc = BgradAccumulator(out_features)

    for start in range(0, tokens, _CHUNK):
        stop = min(start + _CHUNK, tokens)
        xc, dyc = x2d[start:stop].double(), dy2d[start:stop].double()
        # forward: out = x @ W^T [+ b iff apply_bias]; input_grad: dx = dy @ W
        forward_ref = xc @ weight64.t()
        if apply_bias:
            forward_ref = forward_ref + bias64
        forward.update(forward_out[start:stop], forward_ref)
        dgrad.update(input_grad_out[start:stop], dyc @ weight64)
        wgrad_acc.update(x2d[start:stop], dy2d[start:stop])
        bgrad_acc.update(dy2d[start:stop])

    wgrad = _Stats(tolerance)
    wgrad.update_plain(outputs[3], wgrad_acc.value())
    bgrad = _Stats(tolerance)
    # An empty slot is a legitimate arity, not a numeric claim: with
    # return_bias=True and consume_returned_bias=False the bias reaches the loss
    # through no path at all, so the device returns [0] (all_linear case 73 does).
    # Reshaping [0] into the reference's shape would raise instead of reporting.
    if outputs[4].numel():
        bgrad.update_plain(outputs[4], bgrad_acc.value())

    print(
        "\n--- {}  dtype={} in={} N={} tokens={} attrs={} ---".format(
            label,
            str(dtype).rsplit(".", maxsplit=1)[-1],
            list(inp_shape),
            out_features,
            tokens,
            {k: v for k, v in attrs.items() if v != _DEFAULT_ATTRS[k]} or "default",
        )
    )
    print(forward.line(RESULT_SCHEMA[0]))
    print(dgrad.line(RESULT_SCHEMA[2]))
    print(wgrad.line(RESULT_SCHEMA[3]))
    print(bgrad.line(RESULT_SCHEMA[4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="npu",
        choices=["npu", "cuda"],
        help="device whose adapter output is judged",
    )
    parser.add_argument(
        "--only", default=None, help="run a single shape by label prefix, e.g. id136"
    )
    args = parser.parse_args()

    print("judging '{}' output against an exact fp64 reference".format(args.device))
    for label, dtype, inp_shape, out_features, attrs in SHAPES:
        if args.only and not label.startswith(args.only):
            continue
        run_one(label, dtype, inp_shape, out_features, attrs, args.device)

    print("\nboth sides ratio_vs_fp64 == 1.0  ->  residual failure is not a")
    print("numerical defect of either implementation; check the ATK baseline.")
    print("one side < 0.99                    ->  that side is the deviant.")


if __name__ == "__main__":
    main()
