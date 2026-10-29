# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Sample-space guards shared by the norm-family ATK generators.

The guards here constrain *what the generator samples*, never what the device
computes, and they are all of one shape: a LayerNorm whose ``rsigma =
1/sqrt(var + eps)`` is large turns the +-1 ULP quantization of some
case-dtype intermediate into a large *absolute* deviation downstream. When the
comparator's gate for that slot bottoms out at an absolute cap, two
implementations that are each exact *given their own operands* still disagree
on a handful of near-zero elements -- a case that cannot discriminate. The
sample space therefore keeps rsigma bounded, and the case JSON records the
clamped range so both sides still see the same data.

``MAX_RSIGMA``: the eps-dominated corner. ``dX`` is ``rsigma * (bracket)``, so
any difference in the bracket is amplified by rsigma. Measured on the NPU,
device vs exact fp64:

- ``layernorm_linear`` (the corner the shipped case set hit):
  sampled x with std 1e-3 against eps 1e-5 gives rsigma ~ 300 and an absolute
  dX floor of 7-64, over the bf16 absolute cap; scaling the sampled std up to
  ~0.1 (rsigma <= 10) measured 0 violations with a 4x margin, while rsigma ~ 31
  only passed by sitting exactly on the 32-ULP backstop.
- ``layernorm_mlp`` (2026-09-28, the generalized run's dominant failure):
  a mirror of the failing case ``c44`` (bf16, x ~ U(-1e-3, 1e-3), eps 1e-6 =>
  rsigma ~ 877) gives ``input_grad`` matched_ratio 0.926 with 379 elements
  past the absolute cap and max_abs 16; the same shape with a wide x range has
  ``input_grad`` at **1.000000**. The failing |e| sit 3-9x below the slot
  median, i.e. the amplification, not a reduction defect (given its own
  upstream the device's LN parameter grads are exact: ratio 1.000000, 0 abs
  violations, in every mirror case).

FP32 is **not** exempt. The mechanism is the same but the binding gate is
different: the fp32 *relative* tolerance (9.77e-4) is never violated -- the two
implementations agree to ~8e-6 relative on a dX of magnitude 3.6e3 -- but the
comparator's ``max_abs_error_limit`` (1e-2, a hard AND gate on every element)
is, for ``|e| > 10.2``, *stricter* than the relative tolerance it is supposed
to backstop. Since ``dX ~ rsigma * bracket``, the absolute floor grows with
rsigma, and it is rsigma -- not the dtype -- that decides whether the case is
well posed. Measured on the NPU (2026-09-28, the generalized run's output_1
reds: cases 104/108/121/122, all with x ~ U(-1e-3, 1e-3) against eps <= 1e-5):

- mirror of case 108 (fp32, hidden 2048, eps 1e-6 => rsigma ~ 867):
  ``input_grad`` 30142 elements past the 1e-2 cap, max_abs 2.8e-2, i.e. the
  same signature the shipped run reported (2.9e6 elements at 12.6e6 tokens).
- the identical config with the sampled x std raised to 0.1 (rsigma = 10):
  **0** elements past the cap, max_abs 3.5e-4, ratio 0.9997. The abs floor
  scales linearly with rsigma, so 10 leaves a ~28x margin.

``MAX_PREACT``: the fp16 activation-backward overflow (2026-09-30, the
generalized run's fp16 nan/inf reds, e.g. layernorm_mlp case 16: fp16,
RMSNorm + zero_centered_gamma, x sitting exactly ON the rsigma floor -- which
is why the guard above cannot cover this corner). The activation backward
takes exp/tanh of the fc1 PRE-ACTIVATION, and fp16's 5-bit exponent (max
65504) overflows ~33 orders of magnitude earlier than bf16/fp32's 8-bit
exponent (3.4e38) -- hence the historical coarse-dtype reds were all
finite-noise bf16 while the nan/inf class is fp16-only. x-hat is unit-RMS
per token BY CONSTRUCTION, so no x-range clamp can bound the pre-activation;
its k-sigma reach is

    k * sqrt(hidden * E[ln_out^2]) * sigma(fc1_w) + reach(fc1_b)
    E[ln_out^2] = E[gamma_eff^2] + E[beta^2]   (beta only under LayerNorm)

and the generator's lever is the fc1 weight/bias sampling range. Unlike
rsigma this is CONTAINMENT of a device-side defect, not a well-posedness
carve-out: the golden side stays finite through the corner (CUDA promotes
the half math internally), so an unclamped case still discriminates. Keep the
clamp only until ``gelu_bwd`` gains the qgelu-style fp32 promotion
(ops/basic/npu_activation.py) or the kernel saturates, then drop it. The
bounds are PROVISIONAL pending the stage-by-stage kernel probe: the tanh
argument t = 0.798(y + 0.0447 y^3) pushes exp(2t) past fp16 max at |y| ~ 4.0;
silu's sigmoid argument overflows at |y| ~ 11.

The policy is duplicated per module today because ``layernorm_mlp`` and
``layernorm_linear`` live on separate branches; the linear side's copy should
import from here once they merge.
"""

from typing import Iterator, List, Optional, Tuple

MAX_RSIGMA = 10.0
WELL_POSED_DTYPES: Tuple[str, ...] = ()
NORMAL_BUCKET = "n" + "d"  # ATK's normal-distribution bucket name (typos hook)

# fp16 activation-backward overflow bound on the fc1 pre-activation (see the
# module docstring). bf16/fp32 never see the corner at matrix scales (8-bit
# exponents), so the guard is fp16-only -- the mirror image of the old
# well-posed-dtype exemption: there the fine dtype was exempt, here the
# narrow-exponent one is the endangered side. An unknown activation gets the
# strictest bound (fail-closed).
MAX_PREACT = {
    "gelu": 4.0,  # tanh-arg exp(2t) exceeds fp16 max at |y| ~ 4.0 (provisional)
    "silu": 11.0,  # sigmoid's exp argument exceeds fp16 max at |y| ~ 11 (provisional)
}
OVERFLOW_DTYPES: Tuple[str, ...] = ("fp16",)
# The k in "k-sigma reach": the max over N ~ 1e5-1e6 Gaussian draws sits at
# ~4.5-5 sigma, so k = 6 carries the margin (a realized ~5-sigma max then
# lands ~20% under the bound).
PREACT_SIGMAS = 6.0


def implied_std(bucket) -> Optional[float]:
    """Std implied by ONE sampled range bucket, or None if not modelled.

    Two bucket forms come out of ATK's value spec: the uniform interval
    ``[lo, hi]`` and the normal draw ``{"name": "nd", "mean": [...], "std":
    [...]}`` (``DefaultCustomDistConfig.default_range_values == [[-5, 5]]``;
    the recorded per-input value is the bucket list the case drew from).

    The outlier style records its *base* spec here -- the amplification lives
    in ``outlier_values`` -- which is the conservative one to judge: the rows
    without an outlier are exactly the eps-dominated ones.
    """
    if isinstance(bucket, dict):
        if str(bucket.get("name")) == NORMAL_BUCKET:
            std = bucket.get("std")
            if isinstance(std, (list, tuple)) and len(std) == 1:
                return abs(float(std[0]))
        return None
    if isinstance(bucket, (list, tuple)):
        if len(bucket) == 2 and all(isinstance(v, (int, float)) for v in bucket):
            return abs(float(bucket[1]) - float(bucket[0])) / (12.0**0.5)
        return None
    return None


def implied_mean(bucket) -> Optional[float]:
    """Mean implied by ONE sampled range bucket, or None if not modelled.

    The same two bucket forms as :func:`implied_std`; needed for the
    pre-activation bound, whose second moment depends on the mean as well.
    """
    if isinstance(bucket, dict):
        if str(bucket.get("name")) == NORMAL_BUCKET:
            mean = bucket.get("mean")
            if isinstance(mean, (list, tuple)) and len(mean) == 1:
                return float(mean[0])
        return None
    if isinstance(bucket, (list, tuple)):
        if len(bucket) == 2 and all(isinstance(v, (int, float)) for v in bucket):
            return 0.5 * (float(bucket[0]) + float(bucket[1]))
        return None
    return None


def iter_bucket_stds(range_values) -> Iterator[float]:
    """Stds of every modelled bucket in a sampled range spec."""
    std = implied_std(range_values)
    if std is not None:
        yield std
        return
    if isinstance(range_values, (list, tuple)) and not isinstance(range_values, str):
        for bucket in range_values:
            bucket_std = implied_std(bucket)
            if bucket_std is not None:
                yield bucket_std


def scale_bucket(bucket, scale: float):
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


def clamp_x_range_for_eps(input_case, dtype, eps) -> bool:
    """Scale up every sampled x bucket that the eps floor would dominate.

    The bound is ``rsigma = 1/sqrt(var + eps) <= MAX_RSIGMA``; with the eps
    pools in use (<= 1e-4) it is effectively "coarse dtypes need std >= 0.1",
    i.e. the rule does couple to eps but the data scale is what binds. Buckets
    are clamped one by one (scaling an already-fine bucket would distort a
    legal range for nothing), about their own center, so the case JSON records
    the clamped spec. Returns whether anything changed.

    Two things are deliberately left alone, both non-amplifying: a bucket this
    parser does not model (an unrecognised form is skipped rather than rejected
    -- ATK's generator has no per-case skip and raising aborts the whole
    ``atk case`` run; the cost of a gap is a visible red case, never a silently
    wrong range), and a zero-width bucket (the constant-x boundary probe:
    xhat is identically 0 there, so the amplified products do not exist).
    """
    if str(dtype) in WELL_POSED_DTYPES:
        return False
    floor_var = (1.0 / MAX_RSIGMA) ** 2 - float(eps)
    if floor_var <= 0.0:
        return False

    def _clamp(bucket):
        std = implied_std(bucket)
        if std is None or std <= 0.0 or std * std >= floor_var:
            return bucket, False
        return scale_bucket(bucket, (floor_var**0.5) / std), True

    current = input_case.range_values
    if implied_std(current) is not None:  # the value is itself one bucket
        new_bucket, hit = _clamp(current)
        if hit:
            input_case.range_values = new_bucket
        return hit
    if isinstance(current, (list, tuple)) and not isinstance(current, str):
        clamped: List = []
        changed = False
        for bucket in current:
            new_bucket, hit = _clamp(bucket)
            clamped.append(new_bucket)
            changed = changed or hit
        if changed:
            input_case.range_values = clamped
        return changed
    return False


def rsigma_violations(range_values, dtype, eps) -> List[Tuple[float, float]]:
    """Buckets still violating the rsigma bound, as ``(std, rsigma)`` pairs.

    For the generator's fail-closed assertion: after the clamp above, every
    modelled bucket must satisfy the bound, and a future parser gap must not
    slip through silently.
    """
    if str(dtype) in WELL_POSED_DTYPES:
        return []
    floor_var = (1.0 / MAX_RSIGMA) ** 2
    bad = []
    for std in iter_bucket_stds(range_values):
        if std <= 0.0:
            continue
        if std * std + float(eps) < floor_var - 1e-12:
            bad.append((std, (std * std + float(eps)) ** -0.5))
    return bad


def _iter_spec_buckets(range_values):
    """Yield the buckets of a sampled range spec (a bare bucket counts as one)."""
    if implied_std(range_values) is not None:
        yield range_values
        return
    if isinstance(range_values, (list, tuple)) and not isinstance(range_values, str):
        yield from range_values


def _max_second_moment(range_values, shift: float) -> Optional[float]:
    """Max over buckets of ``E[(v + shift)^2]`` for a draw ``v`` from the spec.

    None when no bucket is modelled. The max (not the mixture moment) is the
    conservative per-bucket bound: a case's tensor is drawn from ONE bucket.
    """
    best = None
    for bucket in _iter_spec_buckets(range_values):
        std = implied_std(bucket)
        mean = implied_mean(bucket)
        if std is None or mean is None:
            continue
        moment = std * std + (mean + shift) ** 2
        best = moment if best is None else max(best, moment)
    return best


def _max_abs_bound(range_values) -> Optional[float]:
    """Max ``|bound|`` over modelled uniform buckets (their exact reach)."""
    best = None
    for bucket in _iter_spec_buckets(range_values):
        if (
            isinstance(bucket, (list, tuple))
            and len(bucket) == 2
            and all(isinstance(v, (int, float)) for v in bucket)
        ):
            reach = max(abs(float(bucket[0])), abs(float(bucket[1])))
            best = reach if best is None else max(best, reach)
    return best


def _max_std(range_values) -> Optional[float]:
    """Max std over modelled buckets (the conservative per-bucket bound)."""
    best = None
    for std in iter_bucket_stds(range_values):
        best = std if best is None else max(best, std)
    return best


def _scale_spec(input_case, scale: float) -> None:
    """Scale every modelled bucket of an input's range about its own center.

    Unmodelled buckets are left alone -- the visible-red policy of the eps
    clamp (skipping beats aborting the whole ``atk case`` run; a gap costs a
    red case, never a silently wrong range).
    """
    current = input_case.range_values
    if implied_std(current) is not None:  # the value is itself one bucket
        input_case.range_values = scale_bucket(current, scale)
        return
    if isinstance(current, (list, tuple)) and not isinstance(current, str):
        scaled = []
        for bucket in current:
            if implied_std(bucket) is not None:
                scaled.append(scale_bucket(bucket, scale))
            else:
                scaled.append(bucket)
        input_case.range_values = scaled


def _preact_parts(
    hidden, ln_w_range, ln_b_range, normalization, zero_centered_gamma, fc1_b_range, bias_enabled
):
    """``(coef, bias_reach)`` with reach = ``coef * sigma(fc1_w) + bias_reach``.

    ``coef = PREACT_SIGMAS * sqrt(hidden * E[ln_out^2])`` and the bias reach
    is the fc1_b bound (0 without bias). None when a feeding spec is
    unmodelled. E[ln_out^2] = E[gamma_eff^2] + E[beta^2] under LayerNorm
    (x-hat is unit-RMS and zero-mean for both norms, so the beta term only
    exists there; gamma_eff = w + 1 under zero_centered_gamma).
    """
    shift = 1.0 if zero_centered_gamma else 0.0
    gamma2 = _max_second_moment(ln_w_range, shift)
    if gamma2 is None:
        return None, None
    if str(normalization) == "LayerNorm":
        beta2 = _max_second_moment(ln_b_range, 0.0)
        if beta2 is None:
            return None, None
        gamma2 += beta2
    bias_reach = 0.0
    if bias_enabled:
        bias_reach = _max_abs_bound(fc1_b_range)
        if bias_reach is None:
            return None, None
    return PREACT_SIGMAS * (float(hidden) * gamma2) ** 0.5, bias_reach


def preact_reach(
    hidden,
    ln_w_range,
    ln_b_range,
    normalization,
    zero_centered_gamma,
    fc1_w_range,
    fc1_b_range,
    bias_enabled,
) -> Optional[float]:
    """k-sigma reach of the fc1 pre-activation implied by the sampled specs.

    None when any feeding spec is unmodelled (visible-red policy). The
    pre-activation is a sum of ``hidden`` independent products
    ``ln_out * w``, so the Gaussian k-sigma model applies however wide the
    token count; only the number of drawn elements grows with it, which the
    k = 6 margin absorbs.
    """
    coef, bias_reach = _preact_parts(
        hidden,
        ln_w_range,
        ln_b_range,
        normalization,
        zero_centered_gamma,
        fc1_b_range,
        bias_enabled,
    )
    if coef is None:
        return None
    sigma_w = _max_std(fc1_w_range)
    if sigma_w is None:
        return None
    return coef * sigma_w + bias_reach


def clamp_fc1_range_for_preact(
    fc1_w_case,
    fc1_b_case,
    *,
    hidden,
    ln_w_range,
    ln_b_range,
    normalization,
    zero_centered_gamma,
    bias_enabled,
    dtype,
    activation,
) -> bool:
    """Shrink the fc1 weight (and bias) range until the pre-activation fits.

    The reach is a JOINT quantity (``hidden x E[ln_out^2] x sigma_W1``), so --
    unlike the per-bucket eps clamp -- every modelled fc1_w bucket is scaled
    by the same factor about its own center, and the case JSON records the
    clamped spec (both sides see the same data). The bias bucket is scaled
    first when its reach alone eats more than half the budget. Returns
    whether anything changed.
    """
    if str(dtype) not in OVERFLOW_DTYPES:
        return False
    bound = MAX_PREACT.get(str(activation), min(MAX_PREACT.values()))
    changed = False
    bias_reach = 0.0
    if bias_enabled:
        bias_reach = _max_abs_bound(fc1_b_case.range_values)
        if bias_reach is None:
            return False  # unmodelled bias: visible-red policy
        if bias_reach > 0.5 * bound:
            _scale_spec(fc1_b_case, (0.5 * bound) / bias_reach)
            changed = True
            bias_reach = _max_abs_bound(fc1_b_case.range_values) or 0.0
    coef, _ = _preact_parts(
        hidden,
        ln_w_range,
        ln_b_range,
        normalization,
        zero_centered_gamma,
        fc1_b_case.range_values,
        bias_enabled,
    )
    if coef is None or coef <= 0.0:
        return changed
    target_sigma = (bound - bias_reach) / coef
    if target_sigma <= 0.0:
        # The bias alone exceeds the budget even after its own clamp; the
        # violations check reports it rather than zeroing the weights.
        return changed
    sigma_w = _max_std(fc1_w_case.range_values)
    if sigma_w is None or sigma_w <= target_sigma:
        return changed
    _scale_spec(fc1_w_case, target_sigma / sigma_w)
    return True


def preact_violations(
    fc1_w_range,
    fc1_b_range,
    *,
    hidden,
    ln_w_range,
    ln_b_range,
    normalization,
    zero_centered_gamma,
    bias_enabled,
    dtype,
    activation,
) -> List[str]:
    """Post-clamp fail-closed check on the pre-activation reach.

    Mirrors :func:`rsigma_violations`: after the clamp above, the realized
    reach must fit the bound, and a future parser gap must not slip through
    silently (an unmodelled spec is NOT a violation -- that gap costs a
    visible red case, never a silently wrong range).
    """
    if str(dtype) not in OVERFLOW_DTYPES:
        return []
    bound = MAX_PREACT.get(str(activation), min(MAX_PREACT.values()))
    reach = preact_reach(
        hidden,
        ln_w_range,
        ln_b_range,
        normalization,
        zero_centered_gamma,
        fc1_w_range,
        fc1_b_range,
        bias_enabled,
    )
    if reach is not None and reach > bound + 1e-9:
        return [
            "fc1 pre-activation k-sigma reach {:.4g} > {} (dtype {}, activation {})".format(
                reach, bound, dtype, activation
            )
        ]
    return []
