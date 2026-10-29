# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``transformer_engine.pytorch.LayerNormLinear``.

ITERATION STATUS: this file is the hook-route candidate, coexisting with
the default full-build generator (``generate_layernorm_linear.py``,
registry key ``generate_layernorm_linear_p0`` — curated two-tier matrix,
golden-compatible). Once this route is validated through iteration, the
full-build file gets deleted and this one becomes the only generator.
Until then, switch routes by pointing the yaml ``generate:`` field AND
the ``atk case -p`` argument at this file.

Route: the native ATK engine drives sampling; this subclass patches the
cross-input coupling via the official ``after_input_config`` extension
point, plus ``after_case_config`` for the one slot whose dependency comes
later in the input order:

    slot 0  x          → recorded; ndim 2-4, all dims >= 1 (norm family:
                         empty inputs are illegal, unlike plain Linear)
    slot 1  dy         := x.shape[:-1] + [out] — DEFERRED to
                         after_case_config (out comes from weight_mat at
                         slot 4, after dy in the adapter's slot order)
    slot 2  ln_w       := [x.shape[-1]]            (layer-norm weight)
    slot 3  ln_b       := [x.shape[-1]]            (dead input under
                         normalization="RMSNorm" — patched anyway for
                         input faithfulness)
    slot 4  weight_mat := [out, x.shape[-1]] — FIRST dim (out) kept from
                         native sampling (yaml 2-D pool), SECOND dim forced
    slot 5  bias_vec   := [out]                    (dead input under
                         bias_enabled=False)
    slot 6  eps / slot 7 normalization ("LayerNorm"/"RMSNorm") /
    slot 8  bias_enabled / slot 9 zero_centered_gamma — free attrs

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI default
1234) + identical ATK version ⇒ identical cases. Pin all three in the
baseline.

Yaml prerequisites (already encoded in layernorm_linear.yaml —
GENERALIZATION.md §5.3): ``shape_distributions`` tuned to module scale;
``max_length`` ≈ 2x the largest intended shape; minimal pinned ``shapes``
on the derived inputs — weight_mat keeps a live 2-D pool for its free dim.

Boundary track runs natively; category selection is per-input boundary
flags. Degenerate x (0-dim / any zero dim) is rejected by the guard below
with an actionable error.

eps/data-scale coupling (coarse dtypes). ``after_case_config`` also holds the
sampled x to ``rsigma = 1/sqrt(var + eps) <= _MAX_RSIGMA`` by scaling up any
bucket the eps floor would otherwise dominate -- the shipped instance is the
nd bucket with std = 1e-3 against eps = 1e-5 (rsigma ~ 300), where the LN
backward's ``dln`` quantization gets amplified past the comparator's absolute
cap and two implementations that are each exact given their own ``dln`` still
disagree. The clamp is per bucket, about the bucket's own
center, and the case JSON records the clamped spec, so the golden and the
device see the same data. fp32 is exempt (its ULP is 10^4 finer and the corner
is well-posed there); the measurements behind the bound are at ``_MAX_RSIGMA``.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Fail-closed layout check: the hook addresses slots by index, so a yaml
# whose inputs don't match the 10-slot adapter contract must raise here
# instead of silently patching the wrong slots.
_EXPECTED_TYPES = (
    "tensor",
    "tensor",
    "tensor",
    "tensor",
    "tensor",
    "tensor",
    "attr",
    "attr",
    "attr",
    "attr",
)

# The eps-dominated corner is ill-posed for a coarse dtype. ``dX`` is
# ``rsigma * (two-cancellation bracket)`` and the LN backward's input
# ``dln = dY @ W`` is materialized in the case dtype, so its +-1 ULP
# quantization is multiplied by ``rsigma = 1/sqrt(var + eps)`` on the way into
# ``dX``. When the sampled x sits below the eps floor that factor reaches ~300
# and the resulting absolute noise floor (measured 7-64 on bf16, case 136 with
# std 1e-3 vs eps 1e-5) exceeds the comparator's absolute cap: two
# implementations that are each exact *given their own dln* still disagree, so
# the case cannot discriminate. Measured floor vs base std (same shape/knobs as
# case 136, ids 9xx in the project-local judge
# ``atk_experiments/layernorm_linear/diagnose_slot23_vs_fp64.py``, outside this repo):
#
#     std        1e-3    1e-2    3.2e-2   1e-1
#     rsigma     302     95      31       9.5
#     abs_fail   7       0       0        0
#     max_abs    64      16      8        2      (bf16)
#
# 3.2e-2 passes only by sitting exactly on the 32-ULP backstop, and the real
# comparison runs against NVTE's own dln, which diverges ~7x more than the
# exact rounded one (53 vs 7 abs failures on case 136), so the bound is set
# where the measured floor is a further ~4x clear of it. FP32 is exempt: its
# ULP is ~10^4 finer, the amplification never reaches the cap, and the corner
# is well-posed there (case 110 runs it clean at var/eps ~ 0.3).
_MAX_RSIGMA = 10.0
_WELL_POSED_DTYPES = ("fp32",)
_COARSE_FALLBACK = ("fp16", "bf16")

# Narrow-slot / token-count coupling. The LN parameter-grad slots are 1-D
# ``[in_features]``, so the comparator's ratio budget is
# ``floor(0.01 * in_features)`` **failing elements** -- zero for every width
# below 100, one at 128, two at 256. The noise on those elements is an
# absolute floor set by the *reduction length* (the token count), so the two
# are decoupled: a narrow slot is decided by whether a single near-zero
# element happens to exist. Measured on the NPU (fp32, in=64, out=96, the
# tightest-atol dtype):
#
#     tokens      1e4      3e4      1e5      2e5      4.2e5
#     max_abs     3.2e-5   1.1e-4   3.5e-4   6.3e-4   1.4e-3     <- noise ~ T
#
# (the 4.2e5 row is case 110's shape, which flips red on ~1 draw in 5). The
# elementwise gate for a near-zero element bottoms out at the fp32 atol
# (1.53e-5), so the cap is set where that floor lands back inside the gate:
# 8192 rows ~ 2.7e-5, within 2x of the atol. The coarse dtypes shrink by
# sqrt(T) instead (their floor is the +-1 ULP of the intermediate dln), so
# they gain ~10x fewer flips from the same cap. This is a probability
# reduction, not a guarantee (a near-zero element can always exist) -- it
# keeps the odd-width coverage (17/24/48... exercise the unaligned tail
# paths) and the native comparator, and the large-reduction stress stays
# covered by the wide in_features cases, which are not capped.
_NARROW_IN_FEATURES = 128
_NARROW_MAX_TOKENS = 8192


def _leading_product(shape):
    """Row count of a ``[..., in_features]`` tensor."""
    tokens = 1
    for dim in shape[:-1]:
        tokens *= dim
    return tokens


def _clamp_shape_for_narrow_slots(x_shape):
    """Halve the leading dims (left to right) until a narrow slot's reduction fits.

    Returns ``(shape, clamped)``. Only the leading dims move: ``in_features``
    is x's last dim, which is what the slots are keyed on and what every
    derived input mirrors (dy / ln_w / ln_b / weight / bias keep their shape
    contracts). Same idiom as the DPA guard's ``_clamp_batch_for_dq_bytes``:
    halve the dim rather than reject the case — ATK's generator has no
    per-case skip, and raising here aborts the whole ``atk case`` run.
    """
    shape = list(x_shape)
    if shape[-1] >= _NARROW_IN_FEATURES or _leading_product(shape) <= _NARROW_MAX_TOKENS:
        return shape, False
    changed = False
    for index in range(len(shape) - 1):
        while shape[index] > 1 and _leading_product(shape) > _NARROW_MAX_TOKENS:
            shape[index] = max(1, shape[index] // 2)
            changed = True
    return shape, changed


def _implied_std(bucket):
    """Std implied by ONE sampled range bucket, or None if not modelled.

    Two bucket forms come out of ATK's value spec: the uniform interval
    ``[lo, hi]`` and the normal draw ``{"name": "nd", "mean": [...], "std":
    [...]}`` (``DefaultCustomDistConfig.default_range_values == [[-5, 5]]``,
    and the recorded per-input value is the bucket list the case drew from).
    The outlier style records its *base* spec here -- the amplification lives
    in ``outlier_values`` -- which is the conservative one to judge: the rows
    without an outlier are exactly the eps-dominated ones.
    """
    if isinstance(bucket, dict):
        if str(bucket.get("name")) == "nd":
            std = bucket.get("std")
            if isinstance(std, (list, tuple)) and len(std) == 1:
                return abs(float(std[0]))
        return None
    if isinstance(bucket, (list, tuple)):
        if len(bucket) == 2 and all(isinstance(v, (int, float)) for v in bucket):
            return abs(float(bucket[1]) - float(bucket[0])) / (12.0**0.5)
        return None
    return None


def _scale_bucket(bucket, scale):
    """Scale a bucket's deviation about its center by ``scale`` (> 1)."""
    if isinstance(bucket, dict):
        out = dict(bucket)
        std = out.get("std")
        out["std"] = [abs(float(std[0])) * scale]
        return out
    lo, hi = float(bucket[0]), float(bucket[1])
    center = 0.5 * (lo + hi)
    half = 0.5 * abs(hi - lo) * scale
    return [center - half, center + half]


def _clamp_x_range_for_eps(input_case, dtype, eps):
    """Scale up every sampled x bucket that the eps floor would dominate.

    The bound is ``rsigma = 1/sqrt(var + eps) <= _MAX_RSIGMA``; with this
    module's eps pool (<= 1e-4) it is effectively "coarse dtypes need
    ``std >= 0.1``", i.e. the rule does couple to eps but the data scale is
    what binds. Buckets are clamped one by one (scaling a case's already-fine
    bucket would distort a legal range for nothing), about their own center,
    so the case JSON records the clamped spec and the golden and the device
    see the same data.

    Two things are deliberately left alone, both non-amplifying: a bucket this
    parser does not model (an unrecognised form is skipped rather than
    rejected -- ATK's generator has no per-case skip and raising aborts the
    whole ``atk case`` run; the cost of a gap is a red case, which is visible,
    never a silently wrong range), and a zero-width bucket (the deliberate
    constant-x boundary probe: xhat is identically 0 there, so ``lngamma`` is
    exactly 0 on both sides and the amplification has no product to act on).
    """
    if str(dtype) in _WELL_POSED_DTYPES:
        return False
    floor_var = (1.0 / _MAX_RSIGMA) ** 2 - float(eps)
    if floor_var <= 0.0:
        return False

    def _clamp(bucket):
        std = _implied_std(bucket)
        if std is None or std <= 0.0 or std * std >= floor_var:
            return bucket, False
        return _scale_bucket(bucket, (floor_var**0.5) / std), True

    current = input_case.range_values
    if _implied_std(current) is not None:  # the value is itself one bucket
        new_bucket, hit = _clamp(current)
        if hit:
            input_case.range_values = new_bucket
        return hit
    if isinstance(current, (list, tuple)) and not isinstance(current, str):
        clamped, changed = [], False
        for bucket in current:
            new_bucket, hit = _clamp(bucket)
            clamped.append(new_bucket)
            changed = changed or hit
        if changed:
            input_case.range_values = clamped
        return changed
    return False


def _iter_bucket_stds(range_values):
    """Stds of every modelled bucket in a sampled range spec (see _implied_std)."""
    std = _implied_std(range_values)
    if std is not None:
        yield std
        return
    if isinstance(range_values, (list, tuple)) and not isinstance(range_values, str):
        for bucket in range_values:
            bucket_std = _implied_std(bucket)
            if bucket_std is not None:
                yield bucket_std


@GENERATOR_REGISTRY.register("generate_layernorm_linear_hook")
class GenerateLayerNormLinearHook(CaseGenerator):
    """Native sampling + coupling patch for the TE LayerNormLinear matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_layernorm_linear_hook expects the input layout "
                "(x, dy, ln_w, ln_b, weight_mat, bias_vec, eps, "
                "normalization, bias_enabled, zero_centered_gamma) with "
                "types tensor x6 + attr x4, got {}".format(got)
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
            if not input_case.shape or not all(d > 0 for d in input_case.shape):
                raise ValueError(
                    "generate_layernorm_linear_hook: degenerate x shape {} — "
                    "scalar/empty inputs are illegal for LayerNormLinear "
                    "(norm family: 2-4-D contract, backward rejects an empty "
                    "dy). Keep the boundary track off: extra_numbers: 0 and "
                    "has_empty/has_infnan: false on the tensor "
                    "inputs".format(input_case.shape)
                )
            self._x_shape = list(input_case.shape)
            self._x_dtype = input_case.dtype
        elif index in (2, 3):
            # ln_w / ln_b := [x.shape[-1]] (ln_b is a dead input under
            # RMSNorm; patched anyway so the case files stay faithful)
            input_case.shape = [self._x_shape[-1]]
            input_case.dtype = self._x_dtype
        elif index == 4:
            # weight_mat — keep the natively sampled out (first dim), force
            # the in dim to x's last dim.
            if len(input_case.shape) != 2 or input_case.shape[0] < 1:
                raise ValueError(
                    "generate_layernorm_linear_hook: degenerate weight_mat "
                    "shape {} — weight must be 2-D [out_features, "
                    "in_features] with out_features >= 1".format(input_case.shape)
                )
            self._out_features = input_case.shape[0]
            input_case.shape = [self._out_features, self._x_shape[-1]]
            input_case.dtype = self._x_dtype
        elif index == 5:
            # bias_vec := [out] (dead input under bias_enabled=False)
            input_case.shape = [self._out_features]
            input_case.dtype = self._x_dtype
        # index 1 (dy) is patched in after_case_config — its out dependency
        # (weight_mat, slot 4) comes later in the input order.
        # index 6-9 (eps / normalization / bias_enabled / zc): free attrs.
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # Narrow-slot / token-count coupling (see _NARROW_MAX_TOKENS): only the
        # leading dims move, so the derived inputs patched in after_input_config
        # (ln_w/ln_b = [x.shape[-1]], weight = [out, x.shape[-1]]) stay valid.
        self._x_shape, clamped = _clamp_shape_for_narrow_slots(self._x_shape)
        if clamped:
            case_config.inputs[0].shape = list(self._x_shape)
        # dy := x.shape[:-1] + [out] — deferred here because out comes from
        # weight_mat (slot 4), which fires after dy (slot 1).
        case_config.inputs[1].shape = self._x_shape[:-1] + [self._out_features]
        case_config.inputs[1].dtype = self._x_dtype
        # eps is sampled at slot 6, so the eps/data-scale coupling can only be
        # decided here, with both in hand. Clamped rather than rejected: ATK's
        # generator has no per-case skip, and raising here aborts the whole
        # `atk case` run (the DPA head_dim lesson).
        _clamp_x_range_for_eps(
            case_config.inputs[0], self._x_dtype, case_config.inputs[6].range_values
        )
        # Fail-closed consistency check on the assembled case.
        in_features = self._x_shape[-1]
        out = self._out_features
        eps = float(case_config.inputs[6].range_values)
        x_stds = list(_iter_bucket_stds(case_config.inputs[0].range_values))
        # Fail-closed on the invariant the clamp above establishes (and that a
        # future parser gap would silently drop). Zero-width buckets are the
        # constant-x probe and carry no amplification -- see the clamp's
        # docstring for both exemptions.
        rsigma_ok = str(self._x_dtype) in _WELL_POSED_DTYPES or all(
            std <= 0.0 or std * std + eps >= (1.0 / _MAX_RSIGMA) ** 2 - 1e-12 for std in x_stds
        )
        # ... and on the token budget the narrow-slot clamp establishes.
        tokens = 1
        for dim in self._x_shape[:-1]:
            tokens *= dim
        narrow_ok = in_features >= _NARROW_IN_FEATURES or tokens <= _NARROW_MAX_TOKENS
        expected = (
            case_config.inputs[1].shape == self._x_shape[:-1] + [out]
            and case_config.inputs[2].shape == [in_features]
            and case_config.inputs[3].shape == [in_features]
            and case_config.inputs[4].shape == [out, in_features]
            and case_config.inputs[5].shape == [out]
            and len({c.dtype for c in case_config.inputs[:6]}) == 1
            and rsigma_ok
            and narrow_ok
        )
        if not expected:
            raise ValueError(
                "generate_layernorm_linear_hook: assembled case violates the "
                "LayerNormLinear coupling contract (x={}, dy={}, ln_w={}, "
                "ln_b={}, weight={}, bias={}); x range {} (stds={}) vs eps={} "
                "gives rsigma up to {} (max {} for dtype {}); narrow-slot token "
                "budget: in_features={} tokens={} (cap {} below {})".format(
                    self._x_shape,
                    case_config.inputs[1].shape,
                    case_config.inputs[2].shape,
                    case_config.inputs[3].shape,
                    case_config.inputs[4].shape,
                    case_config.inputs[5].shape,
                    case_config.inputs[0].range_values,
                    x_stds,
                    eps,
                    max(
                        [(std * std + eps) ** -0.5 for std in x_stds if std * std + eps > 0]
                        or [float("inf")]
                    ),
                    _MAX_RSIGMA,
                    self._x_dtype,
                    in_features,
                    tokens,
                    _NARROW_MAX_TOKENS,
                    _NARROW_IN_FEATURES,
                )
            )
        return case_config
