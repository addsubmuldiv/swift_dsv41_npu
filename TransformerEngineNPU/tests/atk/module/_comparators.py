# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Shared ATK accuracy comparators for the Megatron TE API consistency guards.

Every ATK guard module (``tests/atk/module/<name>/function_<name>.py``) used to
carry its own copy of the ``*_allclose`` and ``*_mare`` comparator classes.
This module extracts the logic so future modules reuse it with a single call:

    from _comparators import register_module_comparators
    register_module_comparators("<name>")

which registers ``<name>_allclose``, ``<name>_mare`` and
``<name>_mixed_tolerance`` in the ATK ``ACCURACY_REGISTRY``. The yaml
``standard.acc`` block then selects one comparator and passes its thresholds
as kwargs. The benchmark (reference) is always the NVTE golden collected on a
CUDA host via ``collect_nvte_golden.py`` — ATK's native single-benchmark task
mechanism is not used because its ``BenchmarkDevice`` enum has no GPU device.

Three standards are provided:

- ``*_allclose``: per-element ``|actual - expected| <= atol + rtol * |expected|``
  (legacy TE-oriented tolerances).
- ``*_mare``: ATK single-benchmark MERE/MARE standard, pass iff
  ``mean_rel_err < threshold`` and ``max_rel_err < mare_ratio * threshold``
  (thresholds 2**-10 / 2**-7 / 2**-13 for fp16 / bf16 / fp32).
- ``*_mixed_tolerance``: ATK mixed-tolerance standard, pass iff
  ``matched_ratio >= 0.99`` (element-wise ``torch.isclose``) and every element's
  abs error is within ``max_abs_error_limit`` or ``max_ulp_multiple`` ULPs —
  the most robust option for non-deterministic kernels.

This module is intentionally self-contained: it only imports the ATK symbols
that exist across all ATK versions (``AccuracyConfig``, ``ACCURACY_REGISTRY``,
``BaseAccuracyCompare``). It does NOT depend on ATK's newer
``mixed_tolerance_benchmark_compare`` / ``single_benchmark_compare`` modules,
so it works with older installed ATK builds (e.g. a pip-installed ATK in
``site-packages`` that predates the mixed-tolerance comparator).
"""

import os
import re
from pathlib import Path

import torch
import yaml

from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare


def load_design_acc_kwargs(design_path) -> dict:
    """Comparator kwargs from a module design yaml's ``standard.acc`` block.

    ATK bakes the ``standard`` block into the case JSON, so a collector or a
    verifier that wants to reproduce ATK's verdict exactly must read the same
    kwargs. Returns ``{}`` when the file cannot be read: this only ever supplies
    provenance/threshold defaults, never a pass/fail decision on its own.
    """
    try:
        with Path(design_path).open("r", encoding="utf-8") as stream:
            design = yaml.safe_load(stream)
        for name, kwargs in (design.get("standard", {}).get("acc", {}) or {}).items():
            if str(name).endswith("mixed_tolerance"):
                return dict(kwargs or {})
    except (OSError, yaml.YAMLError, AttributeError, TypeError):
        return {}
    return {}


class AllCloseAccuracy(BaseAccuracyCompare):
    """Per-element allclose comparison with TE-oriented fixed tolerances."""

    def __init__(self, case_config, **kwargs):
        super().__init__(case_config, **kwargs)
        self.fp16_rtol = kwargs.get("fp16_rtol", 0.03)
        self.fp16_atol = kwargs.get("fp16_atol", 0.03)
        self.bf16_rtol = kwargs.get("bf16_rtol", 0.04)
        self.bf16_atol = kwargs.get("bf16_atol", 0.04)

    def _tolerances(self, dtype):
        if dtype == torch.float16:
            return self.fp16_rtol, self.fp16_atol
        if dtype == torch.bfloat16:
            return self.bf16_rtol, self.bf16_atol
        if dtype in (torch.float32, torch.float64):
            return 1.0e-5, 1.0e-6
        return 0.0, 0.0

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        if local_output.numel() == 0 and remote_output.numel() == 0:
            return AccuracyConfig(
                result=True,
                filename=data_file,
                error_info="both outputs are empty",
            )
        if not local_output.dtype.is_floating_point:
            passed = torch.equal(local_output, remote_output)
            return AccuracyConfig(
                result=bool(passed),
                filename=data_file,
                error_info="exact comparison for contract/integer tensor",
            )

        rtol, atol = self._tolerances(local_output.dtype)
        actual = local_output.to(torch.float64)
        expected = remote_output.to(torch.float64)
        finite = torch.isfinite(actual).all() and torch.isfinite(expected).all()
        diff = torch.abs(actual - expected)
        allowed = atol + rtol * torch.abs(expected)
        passed = bool(finite and torch.all(diff <= allowed))
        max_abs = float(diff.max().item())
        denom = torch.clamp(torch.abs(expected), min=1.0e-12)
        max_rel = float((diff / denom).max().item())
        return AccuracyConfig(
            result=passed,
            filename=data_file,
            error_info="rtol={}, atol={}, max_abs={}, max_rel={}".format(
                rtol, atol, max_abs, max_rel
            ),
            max_abs=max_abs,
            max_rel_err=max_rel,
        )


class MareAccuracy(BaseAccuracyCompare):
    """MERE/MARE relative-error comparison (ATK single-benchmark standard).

    Pass iff ``mean_rel_err < threshold`` AND
    ``max_rel_err < mare_ratio * threshold``, where
    ``rel_err = |actual - expected| / (|expected| + eps)``. Per-dtype
    thresholds mirror ATK's ``single_bm`` defaults (bf16 2**-7 ~ 0.78%) and are
    overridable from the yaml ``standard.acc`` block via ``fp16_thd`` /
    ``bf16_thd`` / ``fp32_thd`` / ``fp8e4m3_thd`` / ``fp8e5m2_thd`` and
    ``mare_ratio``.
    """

    def __init__(self, case_config, **kwargs):
        super().__init__(case_config, **kwargs)
        self.threshold = {
            torch.float16: 2.0**-10,
            torch.bfloat16: 2.0**-7,
            torch.float32: 2.0**-13,
            torch.float8_e4m3fn: 2.0**-3,
            torch.float8_e5m2: 2.0**-2,
        }
        self.mare_ratio = float(kwargs.get("mare_ratio", 10))
        for key, dtype in (
            ("fp16_thd", torch.float16),
            ("bf16_thd", torch.bfloat16),
            ("fp32_thd", torch.float32),
            ("fp8e4m3_thd", torch.float8_e4m3fn),
            ("fp8e5m2_thd", torch.float8_e5m2),
        ):
            if key in kwargs:
                self.threshold[dtype] = float(kwargs[key])

    def _threshold(self, dtype):
        if dtype == torch.float64:
            return self.threshold.get(torch.float32)
        return self.threshold.get(dtype)

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        if local_output.numel() == 0 and remote_output.numel() == 0:
            return AccuracyConfig(
                result=True,
                filename=data_file,
                error_info="both outputs are empty",
            )
        if not local_output.dtype.is_floating_point:
            passed = torch.equal(local_output, remote_output)
            return AccuracyConfig(
                result=bool(passed),
                filename=data_file,
                error_info="exact comparison for contract/integer tensor",
            )

        eps = 1.0e-7
        actual = local_output.to(torch.float64)
        expected = remote_output.to(torch.float64)
        finite = torch.isfinite(actual).all() and torch.isfinite(expected).all()
        if not finite:
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info="nan/inf in output",
            )

        rel_err = torch.abs(actual - expected) / (torch.abs(expected) + eps)
        mean_rel_err = float(rel_err.mean().item())
        max_rel_err = float(rel_err.max().item())

        threshold = self._threshold(local_output.dtype)
        if threshold is None:
            passed = bool(torch.equal(local_output, remote_output))
            return AccuracyConfig(
                result=passed,
                filename=data_file,
                error_info="no threshold for dtype; exact equality used",
            )

        mare_limit = threshold * self.mare_ratio
        passed = bool(mean_rel_err < threshold and max_rel_err < mare_limit)
        return AccuracyConfig(
            result=passed,
            filename=data_file,
            error_info="threshold={:.4g}, mare_ratio={:.1f}, mean_rel_err={:.4g}, "
            "max_rel_err={:.4g}".format(threshold, self.mare_ratio, mean_rel_err, max_rel_err),
            mean_rel_err=mean_rel_err,
            max_rel_err=max_rel_err,
        )


class MixedToleranceAccuracy(BaseAccuracyCompare):
    """ATK mixed-tolerance benchmark standard (self-contained).

    Mirrors ATK's ``mixed_tolerance_bm`` comparator without depending on the
    ATK module that ships it. Pass iff:

    1. ``matched_ratio >= required_matched_ratio`` (default 0.99), where
       ``matched_ratio`` is the fraction of elements with
       ``|actual - expected| <= atol + rtol * |expected|``, AND
    2. every element's absolute error is within ``max_abs_error_limit`` or
       ``max_ulp_multiple`` ULPs of the expected value (the ULP backstop keeps
       near-zero elements from failing on inflated relative error).

    Thresholds are configurable via flat prefixed kwargs
    (``bf16_rtol``, ``bf16_atol``, ``bf16_required_matched_ratio``,
    ``bf16_max_abs_error_limit``, ``bf16_max_ulp_multiple``, and the same for
    ``fp16`` / ``fp32``). Defaults mirror ATK's mixed-tolerance standard.

    **Per-output overrides**: batch-summed parameter gradients (e.g. LayerNorm
    dgamma/dbeta over thousands of tokens) accumulate irreducible cross-kernel
    bf16 noise that lands on near-zero elements, while the other outputs of
    the same case (forward, input grad, GEMM weight grads) stay well inside
    the default tolerance. A module-wide threshold hike would blind the guard on
    those other outputs, so this comparator also accepts per-output-slot
    overrides keyed by the ATK output file index:

        bf16_atol_output_2: 4.0
        bf16_max_abs_error_limit_output_2: 8.0

    Output slot 2/3 here means ``output_2.pt`` / ``output_3.pt`` as produced by
    the module adapter (see its ``RESULT_SCHEMA``). Overrides are opt-in per
    yaml and leave every other output at the module-wide thresholds.

    **Non-judged outputs** (``non_judged_outputs: [2, 4]``): slots whose
    cross-implementation divergence is irreducible *against an NVTE golden* —
    batch-summed parameter gradients at generalized token counts are mutual
    sqrt(N) noise (neither side is wrong given its own operands) — can be
    excluded from the verdict without losing the measurement:
    their matched_ratio requirement and hard caps are opened, the isclose
    ratio and errmax are still computed at the slot's configured rtol/atol,
    and their ``error_info`` reads ``not judged (...)`` so a green row is
    never mistaken for a judged green. Judged slots are unaffected, and the
    list applies to every dtype (the mutual-noise mechanism is
    dtype-independent).

    **Chunked evaluation**: every predicate below is elementwise, so the
    tensors are walked in ``_CHUNK_ELEMENTS`` slices instead of being upcast
    to fp64 whole — verdicts are identical to whole-tensor evaluation while
    peak memory stays bounded (a 1G-element output pair no longer allocates
    ~40 GB of fp64 temporaries; pre-chunking that case blew the post-process
    celery SoftTimeLimitExceeded). ``matched_ratio`` is counted exactly in
    int64 (the whole-tensor form computed it as an fp32 mean).
    """

    _ULP_DTYPE = {
        torch.float16: torch.float16,
        torch.bfloat16: torch.bfloat16,
        torch.float32: torch.float32,
        torch.float64: torch.float64,
    }

    # Comparison slice budget: ~0.5 GB per fp64 tensor view, a few of which
    # are live at once (~3 GB peak) — independent of the output size.
    _CHUNK_ELEMENTS = 64 * 1024 * 1024

    _DEFAULTS = {
        torch.float16: dict(
            rtol=1.95e-3,
            atol=1.95e-3,
            required_matched_ratio=0.99,
            max_abs_error_limit=1e-1,
            max_ulp_multiple=32,
        ),
        torch.bfloat16: dict(
            rtol=1.56e-2,
            atol=1.56e-2,
            required_matched_ratio=0.99,
            max_abs_error_limit=1.0,
            max_ulp_multiple=32,
        ),
        torch.float32: dict(
            rtol=9.77e-4,
            atol=1.53e-5,
            required_matched_ratio=0.99,
            max_abs_error_limit=1e-2,
            max_ulp_multiple=32,
        ),
    }
    _PREFIX = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }

    def __init__(self, case_config, **kwargs):
        super().__init__(case_config, **kwargs)
        self.thresholds = {}
        self.output_overrides = {}
        _attrs = (
            "rtol",
            "atol",
            "required_matched_ratio",
            "max_abs_error_limit",
            "max_ulp_multiple",
        )
        for dtype, default in self._DEFAULTS.items():
            th = dict(default)
            prefix = self._PREFIX[dtype]
            for attr in _attrs:
                key = "{}_{}".format(prefix, attr)
                if key in kwargs:
                    th[attr] = float(kwargs[key])
            self.thresholds[dtype] = th
            # Per-output overrides: "<prefix>_<attr>_output_<i>" (e.g.
            # "bf16_atol_output_2") relax only that output slot.
            pattern = re.compile(
                r"^{prefix}_({attrs})_output_(\d+)$".format(prefix=prefix, attrs="|".join(_attrs))
            )
            overrides = {}
            for key, value in kwargs.items():
                match = pattern.match(str(key))
                if match:
                    index = int(match.group(2))
                    overrides.setdefault(index, {})[match.group(1)] = float(value)
            if overrides:
                self.output_overrides[dtype] = overrides
        # Non-judged outputs ("non_judged_outputs: [2, 4]"): verdict opened,
        # measurement kept — see the class docstring. Parsed once, fail-closed:
        # a malformed slot list aborts the comparison rather than silently
        # freeing the wrong slots.
        self.non_judged_outputs = self._parse_non_judged_outputs(kwargs.get("non_judged_outputs"))

    @staticmethod
    def _parse_non_judged_outputs(value):
        """Output indices from the ``non_judged_outputs`` kwarg.

        Accepts the yaml-native list (``[2, 4]``) or a comma string
        (``"2,4"`` — for design files that pass scalars only).
        """
        if value is None:
            return frozenset()
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                "non_judged_outputs expects a list of output indices, got {!r}".format(value)
            )
        return frozenset(int(index) for index in value)

    def _threshold(self, dtype):
        if dtype in self.thresholds:
            return self.thresholds[dtype]
        if dtype == torch.float64:
            return self.thresholds.get(torch.float32)
        return None

    @staticmethod
    def _output_index(data_file):
        """Slot index of an ATK output filename (``output_2.pt`` -> 2), or None.

        The index is the position in the adapter's returned tuple /
        ``RESULT_SCHEMA``.
        """
        match = re.search(r"output_(\d+)", os.path.basename(str(data_file)))
        return int(match.group(1)) if match else None

    def _output_override(self, dtype, data_file):
        """Return the per-output threshold override for ``data_file``, if any."""
        overrides = self.output_overrides.get(dtype)
        if not overrides:
            return None
        index = self._output_index(data_file)
        if index is None:
            return None
        return overrides.get(index)

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        if local_output.numel() == 0 and remote_output.numel() == 0:
            return AccuracyConfig(
                result=True,
                filename=data_file,
                error_info="both outputs are empty",
            )
        if not local_output.dtype.is_floating_point:
            passed = torch.equal(local_output, remote_output)
            return AccuracyConfig(
                result=bool(passed),
                filename=data_file,
                error_info="exact comparison for contract/integer tensor",
            )

        threshold = self._threshold(local_output.dtype)
        if threshold is None:
            passed = bool(torch.equal(local_output, remote_output))
            return AccuracyConfig(
                result=passed,
                filename=data_file,
                error_info="no mixed-tolerance threshold for dtype; exact equality used",
            )
        threshold = dict(threshold)
        output_override = self._output_override(local_output.dtype, data_file)
        if output_override:
            threshold.update(output_override)
        output_index = self._output_index(data_file)
        judged = output_index not in self.non_judged_outputs
        if not judged:
            # Verdict opened, measurement unchanged: the slot's configured
            # rtol/atol still drive matched_ratio below — only the gates stop
            # binding on it.
            threshold["required_matched_ratio"] = 0.0
            threshold["max_abs_error_limit"] = float("inf")
            threshold["max_ulp_multiple"] = float("inf")

        local_numel = local_output.numel()
        if local_numel != remote_output.numel():
            # The whole-tensor form died on a broadcast error here; a count
            # mismatch is a failed comparison, not an exception.
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info="element count mismatch: local {} vs remote {}".format(
                    local_numel, remote_output.numel()
                ),
            )
        flat_local = local_output.contiguous().view(-1)
        flat_remote = remote_output.contiguous().view(-1)
        ulp_dtype = self._ULP_DTYPE.get(local_output.dtype, torch.float32)

        # Chunked walk: every predicate below is elementwise, so slicing
        # changes nothing about the verdict — it only bounds the fp64
        # temporaries. Non-finite short-circuits exactly like the
        # whole-tensor form did (verdict "nan/inf in output" regardless of
        # how the earlier chunks scored).
        matched_count = 0
        abs_fail_count = 0
        errmax = 0.0
        finite = True
        for start in range(0, local_numel, self._CHUNK_ELEMENTS):
            stop = min(start + self._CHUNK_ELEMENTS, local_numel)
            actual = flat_local[start:stop].to(torch.float64)
            expected = flat_remote[start:stop].to(torch.float64)
            if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                finite = False
                break

            abs_error = torch.abs(actual - expected)
            close_mask = abs_error <= threshold["atol"] + threshold["rtol"] * torch.abs(expected)
            matched_count += int(close_mask.sum().item())
            errmax = max(errmax, float(abs_error.max().item()))

            remote_ulp = flat_remote[start:stop].to(dtype=ulp_dtype)
            spacing = (
                torch.nextafter(remote_ulp, torch.full_like(remote_ulp, torch.inf)) - remote_ulp
            ).to(torch.float64)
            abs_error_mask = (abs_error <= threshold["max_abs_error_limit"]) | (
                abs_error <= threshold["max_ulp_multiple"] * spacing
            )
            abs_fail_count += int((~abs_error_mask).sum().item())

        if not finite:
            return AccuracyConfig(
                result=False,
                filename=data_file,
                error_info="nan/inf in output",
            )

        matched_ratio = matched_count / local_numel
        abs_error_ok = abs_fail_count == 0
        abs_error_fail_count = abs_fail_count

        result = bool(matched_ratio >= threshold["required_matched_ratio"] and abs_error_ok)
        # Stats ride along on every numeric verdict — passing slots included:
        # the errmax-vs-token scatter for re-calibration comes from green runs,
        # not just reds.
        stats = "matched_ratio={:.4f}, errmax={:.3e}".format(matched_ratio, errmax)
        if not judged:
            error_info = "not judged ({})".format(stats)
        else:
            parts = []
            if matched_ratio < threshold["required_matched_ratio"]:
                parts.append(
                    "matched_ratio={:.4f} < {}".format(
                        matched_ratio, threshold["required_matched_ratio"]
                    )
                )
            if not abs_error_ok:
                parts.append(
                    "{} elements exceed max_abs_error_limit={} and {}xULP".format(
                        abs_error_fail_count,
                        threshold["max_abs_error_limit"],
                        threshold["max_ulp_multiple"],
                    )
                )
            error_info = "OK" if not parts else "; ".join(parts)
            error_info += " ({})".format(stats)
        return AccuracyConfig(
            result=result,
            filename=data_file,
            error_info=error_info,
        )


def register_module_comparators(prefix: str) -> None:
    """Register the shared comparators under ``<prefix>_<standard>`` names.

    Uses the ``ACCURACY_REGISTRY.register("key")`` decorator form (the same
    pattern the guard modules have always used) so it works on older ATK builds
    that may lack ``Registry.register_with_key``.
    """
    ACCURACY_REGISTRY.register("{prefix}_allclose".format(prefix=prefix))(AllCloseAccuracy)
    ACCURACY_REGISTRY.register("{prefix}_mare".format(prefix=prefix))(MareAccuracy)
    ACCURACY_REGISTRY.register("{prefix}_mixed_tolerance".format(prefix=prefix))(
        MixedToleranceAccuracy
    )
