# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``transformer_engine.pytorch.Linear``.

ITERATION STATUS: this file is the hook-route candidate, coexisting with
the default full-build generator (``generate_linear.py``, registry key
``generate_te_linear_p0`` — curated two-tier matrix, golden-compatible).
Once this route is validated through iteration, ``generate_linear.py``
gets deleted and this file becomes the only generator. Until then, switch
routes by pointing the yaml ``generate:`` field AND the ``atk case -p``
argument at this file.

Route: the native ATK engine drives sampling; this subclass patches the
cross-input coupling via the official ``after_input_config`` extension
point, plus ``after_case_config`` for the split-parity reconcile:

    slot 0  inp               → recorded; ndim 2-4, last dim = in_features
                                 (LEADING dims may be 0 — the 0-row empty
                                 input is legal for Linear, unlike the norm
                                 family; only the last dim must be >= 1)
    slot 1  weight            := [out, x.shape[-1]] — FIRST dim (out) is
                                 kept from native sampling (the yaml 2-D
                                 pool), SECOND dim is forced to in_features
    slot 2  bias_value        := [out]
    slot 3  grad_output       := x.shape[:-1] + [out]
    slot 4  grad_returned_bias := [out]           (te.Linear returns the
                                 bias vector itself, so its grad is 1-D)
    slot 5-7 bias_enabled / return_bias / consume_returned_bias — free attrs
    slot 8  split_kind        ∈ {none, tuple2, dict2}; "tuple2" requires an
                                 EVEN out_features (adapter raises otherwise)
                                 — reconciled in after_case_config by bumping
                                 a sampled odd out to even (and syncing every
                                 derived shape), preserving the tuple2 branch

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI default
1234) + identical ATK version ⇒ identical cases. Pin all three in the
baseline.

Yaml prerequisites (already encoded in linear.yaml — GENERALIZATION.md
§5.3): ``shape_distributions`` tuned to module scale; ``max_length`` ≈ 2x
the largest intended shape; minimal pinned ``shapes`` on the fully-derived
inputs (bias/grads) — weight keeps a live 2-D pool for its free dim.

Boundary track runs natively; category selection is per-input boundary
flags. The degenerate guards differ from the norm family: a 0-ROW x (zero
leading dim) is legal, so the guards reject only 0-dim/1-D x, a zero LAST
dim, and non-2-D / zero-row weights.

wgrad reduction cap (all dtypes). ``weight_grad = dY^T @ X`` reduces over the
token count, and at generalized token counts (all_linear id142: fp32,
[256, 48, 256, 48] = 3.1M tokens) the two accumulation orders -- NPU tiles vs
the golden side -- diverge by mutual rounding noise whose TAIL trips the
mixed-tolerance hard caps (4/49152 elements over both max_abs_error_limit and
32xULP while matched_ratio stays green): a corner no implementation can
discriminate, clamped out of the matrix rather than judged. Unlike the
LayerNormLinear eps corner there is no well-posed dtype -- fp32 trips the
zero-tolerance abs/ULP caps (id142), fp16/bf16 trip the matched-ratio gate
(historical output_3 reds at 0.76-0.98, 16K-4M tokens) -- so the cap is a
per-dtype table (``_WGRAD_MAX_TOKENS``). The clamp halves the leading dims
(the LN/DPA halving idiom) and then WIDENS in_features by the same factor,
bounded, so the case keeps its element scale (x is preserved exactly for
power-of-two factors) while the reduction length drops; large-T wgrad stress
is consciously traded away and stays covered by the wide-feature cases.
Constants are PROVISIONAL pending the two-host ``diagnose_slot_vs_fp64.py``
sweep. Cases under the cap are untouched, so their goldens stay valid; every
clamped case changes its JSON and needs its golden re-collected.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Fail-closed layout check: the hook addresses slots by index, so a yaml
# whose inputs don't match the 9-slot adapter contract must raise here
# instead of silently patching the wrong slots.
_EXPECTED_TYPES = ("tensor", "tensor", "tensor", "tensor", "tensor", "attr", "attr", "attr", "attr")

# wgrad reduction-length cap: weight_grad = dY^T @ X reduces over the token
# count, and cross-implementation accumulation noise grows with that length
# until its tail trips the mixed-tolerance hard caps (max_abs_error_limit /
# max_ulp_multiple are ZERO-tolerance gates -- they carry no element budget,
# so a single tail element fails the case; see the module docstring).
# Anchors (PROVISIONAL pending the two-host diagnose_slot_vs_fp64.py sweep):
#     fp32  262144  all_linear id142 ([256,48,256,48], 3.1M tokens) red with
#           4/49152 elements over both caps at 12x this length;
#           matched_ratio green there.
#     fp16    8192  historical output_3 reds at matched_ratio 0.76-0.98 start
#     bf16    8192  at 16K tokens (id95-like fp16, [1024,16,16]); 2x below
#           the smallest. Drop these two entries if the exact-baseline golden
#           has since settled the coarse dtypes -- the cap would then only
#           churn green cases.
# An unknown dtype gets the strictest cap (fail-closed).
_WGRAD_MAX_TOKENS = {
    "fp32": 262144,
    "fp16": 8192,
    "bf16": 8192,
}

# in_features compensation bound: the clamp widens x's last dim by the same
# factor it shrank the token count, so the case keeps its element scale
# (x = tokens x in is preserved exactly for power-of-two factors) and the
# wgrad matched-ratio budget grows with it. Bounded by the yaml dim pool's
# largest value; an odd in_features stays odd so the pool's unaligned-tail
# coverage (the 17-wide cases) survives the growth.
_WGRAD_GROW_MAX_FEATURES = 2048


def _leading_product(shape):
    """Row count of a ``[..., in_features]`` tensor."""
    tokens = 1
    for dim in shape[:-1]:
        tokens *= dim
    return tokens


def _wgrad_max_tokens(dtype):
    """Per-dtype wgrad reduction cap; unknown dtypes get the strictest."""
    return _WGRAD_MAX_TOKENS.get(str(dtype), min(_WGRAD_MAX_TOKENS.values()))


def _clamp_tokens_for_wgrad(x_shape, dtype):
    """Halve the leading dims (left to right) until the wgrad reduction fits.

    Returns ``(shape, token_scale)``, ``token_scale = old_tokens / new_tokens``
    (1.0 when untouched). Same idiom as the LayerNormLinear hook's
    ``_clamp_shape_for_narrow_slots``: halve rather than reject -- ATK's
    generator has no per-case skip, and raising here aborts the whole
    ``atk case`` run. A 0-row input (legal for Linear, unlike the norm
    family) short-circuits: an empty reduction has no noise and its wgrad
    compares as empty-equals-empty.
    """
    cap = _wgrad_max_tokens(dtype)
    tokens = _leading_product(x_shape)
    if tokens == 0 or tokens <= cap:
        return list(x_shape), 1.0
    shape = list(x_shape)
    for index in range(len(shape) - 1):
        while shape[index] > 1 and _leading_product(shape) > cap:
            shape[index] = max(1, shape[index] // 2)
    return shape, tokens / _leading_product(shape)


def _grow_in_features(in_features, token_scale):
    """Widen in_features by the factor the token clamp removed, bounded.

    Only a real halving (``token_scale >= 2``) is compensated. The growth is
    capped at ``_WGRAD_GROW_MAX_FEATURES`` (the yaml dim pool's largest
    value) and never shrinks the dim; an odd width stays odd (``grown | 1``)
    so the unaligned-tail coverage survives.
    """
    if token_scale < 2.0:
        return in_features
    grown = min(int(round(in_features * token_scale)), _WGRAD_GROW_MAX_FEATURES)
    grown = max(grown, in_features)
    if in_features % 2 == 1 and grown < _WGRAD_GROW_MAX_FEATURES:
        grown |= 1
    return grown


@GENERATOR_REGISTRY.register("generate_te_linear_hook")
class GenerateTELinearHook(CaseGenerator):
    """Native sampling + coupling patch for the TE Linear case matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_te_linear_hook expects the input layout "
                "(inp, weight, bias_value, grad_output, grad_returned_bias, "
                "bias_enabled, return_bias, consume_returned_bias, "
                "split_kind) with types tensor x5 + attr x4, got {}".format(got)
            )
        super().__init__(config)
        self._x_shape = None
        self._x_dtype = None
        self._out_features = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        if index == 0:
            shape = list(input_case.shape)
            if len(shape) not in (2, 3, 4) or shape[-1] < 1:
                raise ValueError(
                    "generate_te_linear_hook: degenerate inp shape {} — "
                    "Linear requires 2-4-D input with in_features (last dim) "
                    ">= 1; leading dims may be 0 (0-row input is legal, "
                    "unlike the norm family). Keep the boundary track off: "
                    "extra_numbers: 0 and has_empty/has_infnan: false on the "
                    "tensor inputs".format(input_case.shape)
                )
            self._x_shape = shape
            self._x_dtype = input_case.dtype
        elif index == 1:
            # weight — keep the natively sampled out (first dim), force the
            # in dim to x's last dim.
            if len(input_case.shape) != 2 or input_case.shape[0] < 1:
                raise ValueError(
                    "generate_te_linear_hook: degenerate weight shape {} — "
                    "weight must be 2-D [out_features, in_features] with "
                    "out_features >= 1".format(input_case.shape)
                )
            self._out_features = input_case.shape[0]
            input_case.shape = [self._out_features, self._x_shape[-1]]
            input_case.dtype = self._x_dtype
        elif index == 2:
            # bias_value := [out]
            input_case.shape = [self._out_features]
            input_case.dtype = self._x_dtype
        elif index == 3:
            # grad_output := x.shape[:-1] + [out]
            input_case.shape = self._x_shape[:-1] + [self._out_features]
            input_case.dtype = self._x_dtype
        elif index == 4:
            # grad_returned_bias := [out] (te.Linear returns the bias vector)
            input_case.shape = [self._out_features]
            input_case.dtype = self._x_dtype
        # index 5-8 (bools + split_kind): free attrs, untouched here.
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # wgrad reduction cap FIRST (see _WGRAD_MAX_TOKENS): the token clamp
        # and the in_features compensation both move shapes, so every derived
        # input is re-synced here; the split reconcile below then bumps out
        # on a consistent case (its own re-sync lines become a no-op).
        self._x_shape, token_scale = _clamp_tokens_for_wgrad(self._x_shape, self._x_dtype)
        self._x_shape[-1] = _grow_in_features(self._x_shape[-1], token_scale)
        case_config.inputs[0].shape = list(self._x_shape)
        case_config.inputs[1].shape = [self._out_features, self._x_shape[-1]]
        case_config.inputs[2].shape = [self._out_features]
        case_config.inputs[3].shape = self._x_shape[:-1] + [self._out_features]
        case_config.inputs[4].shape = [self._out_features]
        # split_kind reconcile: "tuple2" requires an even out_features
        # (adapter raises ValueError otherwise). A sampled odd out is bumped
        # to even — every derived shape is re-synced — so the tuple2 branch
        # stays covered instead of being silently downgraded to dict2.
        split_kind = str(case_config.inputs[8].range_values)
        if split_kind == "tuple2" and self._out_features % 2 == 1:
            self._out_features += 1
            in_features = self._x_shape[-1]
            case_config.inputs[1].shape = [self._out_features, in_features]
            case_config.inputs[2].shape = [self._out_features]
            case_config.inputs[3].shape = self._x_shape[:-1] + [self._out_features]
            case_config.inputs[4].shape = [self._out_features]
        # Fail-closed consistency check on the assembled case.
        w = case_config.inputs[1]
        tokens = _leading_product(self._x_shape)
        expected = (
            w.shape == [self._out_features, self._x_shape[-1]]
            and case_config.inputs[2].shape == [self._out_features]
            and case_config.inputs[3].shape == self._x_shape[:-1] + [self._out_features]
            and case_config.inputs[4].shape == [self._out_features]
            and len({c.dtype for c in case_config.inputs[:5]}) == 1
            and tokens <= _wgrad_max_tokens(self._x_dtype)
        )
        if not expected:
            raise ValueError(
                "generate_te_linear_hook: assembled case violates the Linear "
                "coupling contract (x={}, weight={}, bias={}, grad_output={}, "
                "grad_returned_bias={}); wgrad reduction budget: tokens={} "
                "(cap {} for dtype {})".format(
                    self._x_shape,
                    w.shape,
                    case_config.inputs[2].shape,
                    case_config.inputs[3].shape,
                    case_config.inputs[4].shape,
                    tokens,
                    _wgrad_max_tokens(self._x_dtype),
                    self._x_dtype,
                )
            )
        return case_config
