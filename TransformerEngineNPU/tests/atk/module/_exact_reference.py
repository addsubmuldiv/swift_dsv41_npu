# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Exact fp64 references for the gradient slots of the ATK module guards.

Why this module exists
----------------------

For some slots the NVTE golden is *less* accurate than the device under test,
so benchmarking against it reports a failure that belongs to the reference, not
to TENPU. This was established for ``te.Linear``'s ``bias_grad``
(NVTE's non-FP8 db comes from the cuBLASLt ``CUBLASLT_EPILOGUE_BGRADB`` fused
into the wgrad GEMM, whose tiled in-GEMM accumulation carries truncation noise
on long columns). The generalized ``all_linear`` case set then showed the *main*
wgrad output of that same GEMM is affected too, on the large-token shapes:

    slot vs exact fp64        NPU side   GPU/NVTE side
    input_grad                1.0000     1.0000
    weight_grad               1.0000     0.8802 .. 0.9689
    bias_grad                 1.0000     0.9033 .. 1.0000

The two sides agree bit-for-bit on the shapes where they are both clean, so the
inputs and the forward pass are identical; only the slots produced by NVTE's
coarser kernels diverge. Hence the same remedy 3#17 already applied to ``db``
-- **choose an exact baseline instead of the kernel output** -- and the same
argument that it is not a tolerance relaxation: a real defect (wrong scale,
wrong axis, swapped slots) is still red against an exact reference.

``forward_output`` joined that list later, and is the reason
``forward_reference`` exists. The last two red cases of the generalized
150-case ``all_linear`` set (both fp16, ``id73`` and ``id137``) sit on that slot,
and the per-slot judge puts the deviation on the golden side again::

    forward_output vs exact fp64        NPU      CUDA/NVTE
    id73   [24,24,2048] -> 96           1.0000   0.9616
    id137  [256,64,24,2048] -> 2048     1.0000   0.9851
    both   input_grad                   1.0000   1.0000

The mechanism is visible because ``id137``'s forward and ``input_grad`` are the
*same* GEMM shape (M=393216, N=2048, K=2048, both fp16): only the transpose
flags differ (``x @ W^T`` vs ``dy @ W``), and so does the fp32-accumulation
signature ``med_rel`` -- 2.169e-04 for the forward against 1.692e-04 for
``input_grad``, the latter being the "accumulate in fp32, round once" value the
NPU produces as well. So it is the kernel cuBLASLt picks for the ``transB=T``
forward on those shapes that is coarse, not the reduction depth by itself; on
the shapes where NVTE stays clean the substitution is a measured no-op.

``LayerNormLinear``'s own LayerNorm parameter grads (``dgamma``/``dbeta``)
joined the list through the generalized ``all_layernorm_linear`` run (9 red
cases). They are batch reductions over every token, so the
same question applies, and the per-slot judge answered it the same way: fed the
device's *own* ``dln = dY @ W`` the exact fp64 reduction reproduces the
device's ``dgamma``/``dbeta`` at matched_ratio 1.000000 with zero absolute
violations in all 9 cases, and without the one dtype rounding of ``dln`` the
reference reproduces the observed failures (0.76-0.96, on the near-cancelling
elements only). So the divergence lives in the *operand* -- two accurate fp32
accumulators rounding the same dgrad to the case dtype differently -- and the
remedy is again an exact baseline (``chunked_lngrad`` /
``LngradAccumulator``), with the same "one rounding, mirroring the dataflow"
contract.

Why fp64 and not fp32
---------------------

3#17 implemented the ``db`` baseline as a CPU **fp32** column sum. That is
adequate for a pure reduction but *not* for the wgrad slot, measured on the real
failing shapes (NPU output judged against each reference, ``chunk_rows=32768``)::

    shape                  tokens    fp32 ref vs fp64 ref    NPU vs fp32 ref
    id32-like  (bf16)         576    bitwise identical       1.0000
    id126-like (fp16)     262 144    differ on 41% elements  0.9393  <-- false red
    id133-like (fp16)   2 097 152    differ on 43% elements  0.9334  <-- false red

An fp32 **GEMM** reference drifts by ~1 dtype ULP over a long K, while the fp16
gate only allows ~2 ULP (``rtol`` 1.95e-3 against an fp16 ULP of 9.77e-4). The
drift eats the margin and reds a clean device. The same measurement on ``db``
(a ``torch.sum`` pairwise reduction, far more accurate than a naive fp32 sum)
showed fp32 and fp64 bitwise identical up to 4 194 304 tokens, so moving ``db``
to fp64 as well is a measured no-op that keeps one definition for both slots and
lets the baseline and ``linear/diagnose_slot_vs_fp64.py`` share one formula.

Bitwise exactness of the reference
----------------------------------

For bf16/fp16 operands the products are exact in fp32 (8x8 and 11x11 mantissa
bits fit the 24-bit fp32 mantissa), so the only error in these references is the
accumulation order -- ~1e-16 at fp64, orders of magnitude below the bf16/fp16
ULP. "Exact" here means exactly that.

Contract
--------

* Only torch is imported: this module is used by the golden-side adapter path,
  by the collectors, and directly by the standalone diagnostic scripts.
* ``to_slot`` never changes a slot's dtype or element count -- ATK requires the
  local and remote ``output_*.pt`` sets to match in both
  (``ATK/atk/tasks/post_process/base_compare.py:191-213,230-238``). An empty
  device slot stays empty: its emptiness is an arity claim, not a numeric one.
* Nothing here may run on the device under test. Substituting the reference on
  both sides turns the ATK comparison into reference-vs-reference: every case
  green, guard blind. The golden-side guard lives in the adapter
  (``linear/function_linear.py::_golden_side_guard``); keep it that way.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import torch

DEFAULT_CHUNK_ROWS = 32768

# Host-RAM ceiling for ONE fp64 working buffer of the references. The golden
# runs these on the CUDA host, inside the same long-lived worker process that
# also holds the case's device tensors, and the generalized shapes reach
# in/out = 2048 with millions of tokens: at the old fixed 32768-row chunk one
# fp64 buffer is 32768 * 8192 * 8 = 2.1 GiB, and `norm_forward_reference` held
# *three* whole-tensor ones (measured: +6.0 GiB for a [524288, 512] x, +12.0
# GiB for a [262144, 2048] x, on top of the inputs). The workers on the golden
# host were SIGKILLed for memory during the online run, so
# the row chunk is now capped by this budget; the accumulator scratch buffers
# are allocated once per accumulator and reused, which also stops the
# alloc/free churn that inflates RSS across cases in one worker.
_CHUNK_BYTES_BUDGET = 256 * 1024**2


def chunk_rows_for(working_width: int, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> int:
    """Cap a row chunk so the fp64 working set of one chunk fits the budget.

    ``working_width`` is the total column count of the buffers a path keeps
    alive for one chunk (not just the widest one), so the returned chunk keeps
    their *sum* under ``_CHUNK_BYTES_BUDGET``.
    """
    if working_width <= 0:
        return chunk_rows
    affordable = max(1, _CHUNK_BYTES_BUDGET // (working_width * 8))
    return max(1, min(chunk_rows, affordable))


def _to_host_fp64(chunk: torch.Tensor) -> torch.Tensor:
    """Move a chunk to host *first*, then widen to fp64.

    Order matters: ``.to("cuda", torch.float64)`` would do the fp64 conversion
    on device, which runs at 1/64 rate on consumer parts.
    """
    return chunk.detach().cpu().to(torch.float64)


class WgradAccumulator:
    """Exact fp64 accumulator for ``dw = dy^T @ x``, fed one row chunk at a time.

    ``update`` is the single shared definition of the reference; both the
    golden-side baseline and ``diagnose_slot_vs_fp64.py`` call it, so the judge
    and the baseline cannot drift apart.
    """

    def __init__(self, out_features: int, in_features: int):
        self._acc = torch.zeros(out_features, in_features, dtype=torch.float64)
        self._rows = 0

    def update(self, x_chunk: torch.Tensor, dy_chunk: torch.Tensor) -> None:
        x64 = _to_host_fp64(x_chunk)
        dy64 = _to_host_fp64(dy_chunk)
        if x64.shape[0] != dy64.shape[0]:
            raise ValueError(
                "x and dy chunks disagree on row count: {} vs {}".format(
                    x64.shape[0], dy64.shape[0]
                )
            )
        self._acc += dy64.t() @ x64
        self._rows += x64.shape[0]

    @property
    def rows(self) -> int:
        return self._rows

    def value(self) -> torch.Tensor:
        return self._acc


class BgradAccumulator:
    """Exact fp64 accumulator for ``db = sum(dy, dim=0)``, one row chunk at a time.

    Keeps an fp32 mirror alongside so callers can check the property 3#17 relied
    on (an fp32 column sum agreeing bitwise with the exact one). A divergence
    means the fp32 formulation has started drifting on this case set and the
    equivalence must be re-derived rather than assumed.
    """

    def __init__(self, out_features: int):
        self._acc64 = torch.zeros(out_features, dtype=torch.float64)
        self._acc32 = torch.zeros(out_features, dtype=torch.float32)
        self._rows = 0

    def update(self, dy_chunk: torch.Tensor) -> None:
        dy64 = _to_host_fp64(dy_chunk)
        self._acc64 += dy64.sum(dim=0)
        self._acc32 += dy_chunk.detach().cpu().to(torch.float32).sum(dim=0)
        self._rows += dy64.shape[0]

    @property
    def rows(self) -> int:
        return self._rows

    def value(self) -> torch.Tensor:
        return self._acc64

    def fp32_matches_fp64(self, dtype: torch.dtype) -> bool:
        """True when the fp32 mirror round-trips to the same slot values."""
        return torch.equal(self._acc32.to(dtype), self._acc64.to(dtype))


def chunked_wgrad(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> torch.Tensor:
    """Exact fp64 ``dw`` for already-2D operands: ``[tokens, out]`` x ``[tokens, in]``."""
    out_features, in_features = dy2d.shape[1], x2d.shape[1]
    chunk_rows = chunk_rows_for(in_features + out_features, chunk_rows)
    accumulator = WgradAccumulator(out_features, in_features)
    for start in range(0, x2d.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x2d.shape[0])
        accumulator.update(x2d[start:stop], dy2d[start:stop])
    return accumulator.value()


def chunked_bgrad(dy2d: torch.Tensor, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> torch.Tensor:
    """Exact fp64 ``db`` column sum for an already-2D ``grad_output``."""
    chunk_rows = chunk_rows_for(dy2d.shape[1], chunk_rows)
    accumulator = BgradAccumulator(dy2d.shape[1])
    for start in range(0, dy2d.shape[0], chunk_rows):
        accumulator.update(dy2d[start : min(start + chunk_rows, dy2d.shape[0])])
    return accumulator.value()


def _concat_parameter_rows(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    suffix: str,
    out_features: int,
    dims: int,
) -> torch.Tensor:
    """Host-fp64 concatenation of every ``*<suffix>`` parameter, on the out axis.

    ``parameters_split`` may hold the weight (and the bias) as several
    parameters; ``named_parameters`` order is the concatenation order the
    adapter's own slot assembly uses, so the reference and the slots agree by
    construction. Note this deliberately does **not** filter on
    ``parameter.grad is not None`` the way ``row_spans_from_grad_parameters``
    does: the forward reference needs the *values*, which exist whether or not
    the parameter received a gradient.

    Raises unless the rows tile ``[0, out_features)`` exactly -- the same
    fail-closed rule as the gradient spans, so a split layout this cannot
    reproduce is an error rather than a silently mis-shaped golden.
    """
    blocks: List[torch.Tensor] = []
    rows = 0
    for name, parameter in named_parameters:
        if not name.endswith(suffix):
            continue
        block = _to_host_fp64(parameter)
        if block.dim() != dims:
            raise RuntimeError(
                "parameter {!r} is {}D, expected {}D for suffix {!r}".format(
                    name, block.dim(), dims, suffix
                )
            )
        blocks.append(block)
        rows += block.shape[0]
    if not blocks:
        raise RuntimeError("no parameter named *{} found; cannot build a reference".format(suffix))
    if rows != out_features:
        raise RuntimeError(
            "parameters *{!r} cover {} rows, expected {}; the concatenation "
            "layout assumption no longer holds".format(suffix, rows, out_features)
        )
    widths = {block.shape[1] for block in blocks} if dims == 2 else set()
    if len(widths) > 1:
        raise RuntimeError("parameters *{!r} disagree on width: {}".format(suffix, sorted(widths)))
    return torch.cat(blocks, dim=0)


def concatenated_weight(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    out_features: int,
) -> torch.Tensor:
    """Host-fp64 ``[out_features, in_features]`` weight as the forward consumes it."""
    return _concat_parameter_rows(named_parameters, "weight", out_features, dims=2)


def concatenated_bias(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    out_features: int,
) -> torch.Tensor:
    """Host-fp64 ``[out_features]`` bias as the forward consumes it."""
    return _concat_parameter_rows(named_parameters, "bias", out_features, dims=1)


def forward_reference(
    slot: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    apply_bias: bool,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> torch.Tensor:
    """Exact fp64 ``x @ W^T [+ b]``, cast into the forward slot's shape and dtype.

    ``apply_bias`` is the caller's ``bias_enabled and not return_bias`` -- NVTE's
    ``Linear.apply_bias``. Passing the plain ``bias_enabled`` here would make the
    golden itself wrong by the bias magnitude on every ``return_bias=True`` case
    (~0.5 on the ``all_linear`` set) and blame the device for it.

    Row-chunked on purpose: a whole fp64 ``[tokens, N]`` reference is what the
    judge avoids (~9 GB at 4M tokens), and here it would be live twice (fp64, then
    cast). Peak host memory is one fp64 chunk plus the slot-sized output.
    """
    if slot.numel() == 0:
        return slot
    tokens, in_features = x2d.shape
    out_features = weight.shape[0]
    if weight.shape[1] != in_features:
        raise RuntimeError(
            "weight is [{} , {}] but the input's last dim is {}".format(
                out_features, weight.shape[1], in_features
            )
        )
    if slot.numel() != tokens * out_features:
        raise RuntimeError(
            "forward reference has {} elements, device slot has {}; ATK compares "
            "element counts across the two sides".format(tokens * out_features, slot.numel())
        )

    weight64 = _to_host_fp64(weight)
    bias64 = _to_host_fp64(bias) if (apply_bias and bias is not None) else None
    out = torch.empty(slot.numel(), dtype=slot.dtype, device=slot.device)
    out2d = out.view(tokens, out_features)
    for start in range(0, tokens, chunk_rows):
        stop = min(start + chunk_rows, tokens)
        reference = _to_host_fp64(x2d[start:stop]) @ weight64.t()
        if bias64 is not None:
            reference = reference + bias64
        # One rounding, into the slot's dtype -- same contract as ``to_slot``.
        out2d[start:stop] = reference.to(slot.dtype)
    return out.view(slot.shape)


def activation_forward_reference(z: torch.Tensor, activation: str) -> torch.Tensor:
    """Host-fp64 activation for the MLP forward chain.

    ``silu = z * sigmoid(z)`` has no variants. ``gelu`` here is the **tanh
    approximation** — TENPU's ``gelu_fwd`` calls ``npu_gelu(approximate=
    "tanh")``, and the baseline is judged against the device that runs that
    formula, so the reference must mirror it (the erf form would build a
    bias-sized systematic difference into the golden).
    """
    z = z.double()
    if activation == "silu":
        return z * torch.sigmoid(z)
    if activation == "gelu":
        return 0.5 * z * (1.0 + torch.tanh(0.7978845608028654 * (z + 0.044715 * z.pow(3))))
    raise RuntimeError("unsupported activation: {!r}".format(activation))


def mlp_forward_reference(
    slot: torch.Tensor,
    x2d: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Optional[torch.Tensor],
    fc1_weight: torch.Tensor,
    fc1_bias: Optional[torch.Tensor],
    fc2_weight: torch.Tensor,
    fc2_bias: Optional[torch.Tensor],
    eps: float,
    normalization: str,
    activation: str,
    bias_enabled: bool,
    zero_centered_gamma: bool,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> torch.Tensor:
    """Exact fp64 LayerNormMLP forward, cast into the slot's shape and dtype.

    The **materialized dataflow**: every intermediate the module writes in the
    case dtype — the norm output, the fc1 pre-activation (with its bias), the
    activation output — is rounded once on the way through; the fc2 output
    (+bias) rounds once. That is the chain both module *sources* describe
    (TENPU's separate ops on materialized tensors; NVTE's silu path), and it
    is what TENPU reproduces bitwise-median after the ``general_gemm`` bias
    fusion fix (measured ratio 1.0000 on all twelve mirror shapes). NVTE's
    *fused* gelu epilogue skips the pre-activation rounding and measurably
    departs from this chain (0.9677-0.9796 at the o0 level on the rerun's red
    cases; its skinny-N GEMM shapes drift the same way), which is why the
    golden carries this reference instead of the kernel output — the same
    baseline-not-tolerance argument as the linear wgrad slots.

    Row-chunked like ``forward_reference``: the generalized shapes reach
    2e9 elements and whole-tensor fp64 intermediates are what the golden
    host gets killed for. ``ln_bias`` is ignored for RMSNorm (dead parameter
    by construction, same as the module's).
    """
    if slot.numel() == 0:
        return slot
    tokens, in_features = x2d.shape
    ffn = fc1_weight.shape[0]
    out_features = fc2_weight.shape[0]
    if fc2_weight.shape[1] != ffn:
        raise RuntimeError(
            "fc2_weight is [{}, {}] but ffn is {}".format(out_features, fc2_weight.shape[1], ffn)
        )
    if slot.numel() != tokens * out_features:
        raise RuntimeError(
            "forward reference has {} elements, device slot has {}; ATK compares "
            "element counts across the two sides".format(tokens * out_features, slot.numel())
        )

    # Per-chunk fp64 working set: x copy + ln_out + z + act + out.
    chunk_rows = chunk_rows_for(in_features + in_features + 3 * ffn + out_features, chunk_rows)
    w1_64 = _to_host_fp64(fc1_weight)
    w2_64 = _to_host_fp64(fc2_weight)
    b1_64 = _to_host_fp64(fc1_bias) if bias_enabled else None
    b2_64 = _to_host_fp64(fc2_bias) if bias_enabled else None
    ln_bias_or_none = ln_bias if normalization != "RMSNorm" else None

    out = torch.empty(slot.numel(), dtype=slot.dtype, device=slot.device)
    out2d = out.view(tokens, out_features)
    for start in range(0, tokens, chunk_rows):
        stop = min(start + chunk_rows, tokens)
        ln64 = norm_forward_reference(
            x2d[start:stop],
            ln_weight,
            ln_bias_or_none,
            eps,
            normalization,
            zero_centered_gamma,
        )
        ln_r = ln64.to(slot.dtype).double()
        z = ln_r @ w1_64.t()
        if b1_64 is not None:
            z = z + b1_64
        z = z.to(slot.dtype).double()
        act = activation_forward_reference(z, activation).to(slot.dtype).double()
        reference = act @ w2_64.t()
        if b2_64 is not None:
            reference = reference + b2_64
        # One rounding into the slot's dtype at the final stage only.
        out2d[start:stop] = reference.to(slot.dtype)
    return out.view(slot.shape)


def norm_forward_reference(
    x2d: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor],
    eps: float,
    normalization: str,
    zero_centered_gamma: bool,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Exact LayerNorm/RMSNorm forward output on host — the Linear's A operand
    inside ``LayerNormLinear``.

    This is the fully affine-transformed normalization output, so for
    LayerNorm the **beta is included**: it enters the wgrad GEMM's A operand,
    and dropping it skews the whole wgrad reference by
    ``outer(column_sum(dy), beta)`` — the LayerNormLinear counterpart of
    Linear's ``apply_bias`` trap. RMSNorm has no beta by construction.

    ``zero_centered_gamma`` applies to both normalizations
    (``gamma_eff = 1 + gamma``); variance is biased and eps sits inside the
    sqrt — both matching ``F.layer_norm`` / TENPU ``npu_rms_norm``.

    Row-chunked, and the one caller that does not need fp64 output passes
    ``out_dtype``: the whole-tensor form kept *three* ``[tokens,
    in_features]`` fp64 buffers alive at once (measured +6.0 GiB for a
    [524288, 512] input), which is what the golden host was being killed
    for on the generalized shapes. Every row's mean/variance is independent of
    the other rows, so chunking is **bit-identical** to the whole-tensor form.
    """
    if x2d.dim() != 2:
        raise RuntimeError(
            "normalization reference needs 2-D input, got {}".format(tuple(x2d.shape))
        )
    # x64 + (x - mu) + the affine product are the fp64 working buffers here.
    chunk_rows = chunk_rows_for(3 * x2d.shape[1], chunk_rows)
    tokens = x2d.shape[0]
    gamma64 = _to_host_fp64(gamma)
    gamma_eff = (1.0 + gamma64) if zero_centered_gamma else gamma64
    beta64 = None
    if beta is not None:
        beta64 = _to_host_fp64(beta)
    if normalization not in ("LayerNorm", "RMSNorm"):
        raise RuntimeError("unsupported normalization: {!r}".format(normalization))

    out = torch.empty(
        (tokens, x2d.shape[1]), dtype=out_dtype if out_dtype is not None else torch.float64
    )
    for start in range(0, tokens, chunk_rows):
        stop = min(start + chunk_rows, tokens)
        x = _to_host_fp64(x2d[start:stop])
        if normalization == "RMSNorm":
            reference = x * x.pow(2).mean(dim=-1, keepdim=True).add_(eps).rsqrt() * gamma_eff
        else:
            mu = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            reference = (x - mu) * torch.rsqrt(var + eps) * gamma_eff
            if beta64 is not None:
                reference = reference + beta64
        # One rounding into the output dtype when one was asked for -- the same
        # "cast once" contract the callers apply to the fp64 form.
        out[start:stop] = reference if out_dtype is None else reference.to(out_dtype)
    return out


def norm_stats_chunk(
    x_chunk: torch.Tensor, eps: float, normalization: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Host-fp64 ``(xhat, rsigma)`` for one row chunk.

    One definition of the normalization math for every consumer: the LN
    parameter-grad reference (``normalized_input_chunk``), and the judge's
    ``dX`` reference, which needs ``rsigma`` on its own because ``dX`` is
    ``rsigma`` times a two-cancellation bracket (that factor is what makes an
    eps-dominated row amplify any input difference).

    Conventions match ``norm_forward_reference``: biased variance, eps inside
    the sqrt (LayerNorm) / inside the mean square (RMSNorm).
    """
    x = _to_host_fp64(x_chunk)
    if normalization == "RMSNorm":
        rsigma = x.pow(2).mean(dim=-1, keepdim=True).add_(eps).rsqrt()
        return x * rsigma, rsigma
    if normalization != "LayerNorm":
        raise RuntimeError("unsupported normalization: {!r}".format(normalization))
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rsigma = torch.rsqrt(var + eps)
    return (x - mu) * rsigma, rsigma


def normalized_input_chunk(x_chunk: torch.Tensor, eps: float, normalization: str) -> torch.Tensor:
    """Host-fp64 ``xhat`` — the *unscaled* normalized input — for one row chunk.

    ``dgamma``/``dbeta`` of a LayerNorm are ``colsum(xhat * dln)`` and
    ``colsum(dln)``: neither the affine scale nor the shift reaches those
    sums, so ``zero_centered_gamma`` is deliberately absent from this file's
    LN-grad path (measured: the same reference matches the device for zc on
    and off alike, while a wrong gamma placement in ``dX`` is a factor-gamma
    error). Only ``dX`` carries ``gamma_eff`` — see ``dx_reference`` in the
    project-local judge (``atk_experiments/layernorm_linear/diagnose_slot23_vs_fp64.py``,
    outside this repo).
    """
    return norm_stats_chunk(x_chunk, eps, normalization)[0]


def dln_chunk_reference(
    dy_chunk: torch.Tensor,
    weight64: torch.Tensor,
    round_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Host-fp64 ``dY @ W`` for one row chunk, rounded once into the case dtype.

    This is the gradient the module's LN backward actually consumes: the
    Linear's dgrad, written into a buffer of the case dtype before the norm
    backward reads it (``weight64`` is the canonical ``[out_features,
    in_features]`` weight, already on the host in fp64 via ``_to_host_fp64``).

    ``round_dtype`` is what makes the reference match the device. Two fp32
    accumulators that are both accurate still land on opposite sides of that
    rounding boundary now and then (~0.02-0.1% of elements, at most a few
    ULP), and the LN backward turns that input-level difference into an
    absolute noise floor on its parameter grads; the reference mirrors the
    rounding so it reproduces the device's *operand* rather than an
    idealization the device can never hit. Measured: with
    the rounding the reference matches the device bitwise on the small cases
    (``max_abs == 0``) and inside the ratio gate on the large ones; without
    it, it reproduces the generalized run's slot-2/3 failures (0.76-0.96).
    """
    dln = _to_host_fp64(dy_chunk) @ weight64
    if round_dtype is not None:
        dln = dln.to(round_dtype).to(torch.float64)
    return dln


class LngradAccumulator:
    """Exact fp64 accumulator for LayerNormLinear's LayerNorm parameter grads.

    ``update`` is the single shared definition of the reference; the
    golden-side baseline (``layernorm_linear/function_layernorm_linear.py``)
    and the project-local per-slot judge
    (``atk_experiments/layernorm_linear/diagnose_slot23_vs_fp64.py``, outside this repo)
    both call it, so the baseline and the arbiter cannot drift apart. The
    ``xhat_dtype`` and pre-rounded-``dln`` variants are what the judge needs
    to attribute a deviation to one operand; they are diagnostics, never the
    baseline.

    ``dln`` is passed in rather than computed here because the caller owns
    the GEMM (and its one dtype rounding — see ``dln_chunk_reference``) and
    may need to feed several accumulators from one fp64 GEMM.
    """

    def __init__(
        self,
        in_features: int,
        eps: float,
        normalization: str,
        xhat_dtype: Optional[torch.dtype] = None,
    ):
        if normalization not in ("LayerNorm", "RMSNorm"):
            raise RuntimeError("unsupported normalization: {!r}".format(normalization))
        self.in_features = int(in_features)
        self._eps = float(eps)
        self._normalization = normalization
        self._xhat_dtype = xhat_dtype
        self._gamma = torch.zeros(self.in_features, dtype=torch.float64)
        self._beta = torch.zeros(self.in_features, dtype=torch.float64)
        # Reused across chunks: one allocation per accumulator instead of one
        # per chunk, which is what keeps the RSS of a long case list flat.
        self._scratch = None
        self._rows = 0

    def update(self, x_chunk: torch.Tensor, dln_chunk: torch.Tensor) -> None:
        xhat = normalized_input_chunk(x_chunk, self._eps, self._normalization)
        dln = _to_host_fp64(dln_chunk)
        if dln.shape != xhat.shape:
            raise ValueError(
                "dln chunk {} does not match the normalized input {}: dln is "
                "the dgrad of the LN output, so it carries the LN output's own "
                "shape".format(tuple(dln.shape), tuple(xhat.shape))
            )
        if xhat.shape[1] != self.in_features:
            raise ValueError(
                "chunk width {} does not match the accumulator's in_features {}".format(
                    xhat.shape[1], self.in_features
                )
            )
        if self._xhat_dtype is not None:
            xhat = xhat.to(self._xhat_dtype).to(torch.float64)
        if self._scratch is None or self._scratch.shape != dln.shape:
            self._scratch = torch.empty(dln.shape, dtype=torch.float64)
        torch.mul(xhat, dln, out=self._scratch)
        self._gamma += self._scratch.sum(dim=0)
        self._beta += dln.sum(dim=0)
        self._rows += xhat.shape[0]

    @property
    def rows(self) -> int:
        return self._rows

    def value(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._gamma, self._beta


def chunked_lngrad(
    x2d: torch.Tensor,
    dy2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    normalization: str,
    round_dln_dtype: Optional[torch.dtype] = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact fp64 ``(dgamma, dbeta)`` for the norm inside ``LayerNormLinear``.

    ``round_dln_dtype`` is the case dtype the module's dgrad is materialized
    in; it is a required part of the definition, not an option — see
    ``dln_chunk_reference``. ``dbeta`` is returned for RMSNorm too (it is
    simply unused there: the module has no LN bias parameter, so the slot is
    empty by construction).
    """
    in_features = x2d.shape[1]
    # One fp64 chunk holds: the dy copy (out), dln (in), xhat (in) and the
    # accumulator scratch (in). Budgeted, because at the generalized
    # in/out = 2048 shapes a fixed 32768-row chunk is a 3.8 GiB peak.
    chunk_rows = chunk_rows_for(3 * in_features + weight.shape[0], chunk_rows)
    accumulator = LngradAccumulator(
        in_features=in_features, eps=float(eps), normalization=normalization
    )
    weight64 = _to_host_fp64(weight)
    for start in range(0, x2d.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x2d.shape[0])
        accumulator.update(
            x2d[start:stop],
            dln_chunk_reference(dy2d[start:stop], weight64, round_dln_dtype),
        )
    return accumulator.value()


def bias_grad_reference(
    dy2d: torch.Tensor,
    grad_returned_bias: torch.Tensor,
    bias_enabled: bool,
    return_bias: bool,
    consume_returned_bias: bool,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Optional[torch.Tensor]:
    """Exact reference for the ``bias_grad`` slot, or ``None`` for "keep the device slot".

    ``None`` means the slot is empty by construction and must be passed through
    unchanged (arity, not a numeric claim). The four shapes follow from
    ``Linear.apply_bias = bias and not return_bias`` (NVTE ``module/linear.py``):
    with ``return_bias`` the bias never enters the forward output and is only
    reachable through the ``consume_returned_bias`` term of the adapter's loss.

    =========== =========== =========== ==========================
    bias        return_bias consume     device slot / reference
    =========== =========== =========== ==========================
    False       any         any         empty
    True        False       any         column sum of grad_output
    True        True        False       empty (bias grad is None)
    True        True        True        grad_returned_bias, identity
    =========== =========== =========== ==========================
    """
    if not bias_enabled:
        return None
    if not return_bias:
        return chunked_bgrad(dy2d, chunk_rows=chunk_rows)
    if not consume_returned_bias:
        return None
    return _to_host_fp64(grad_returned_bias)


def row_spans_from_grad_parameters(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    suffix: str,
    out_features: int,
) -> List[Tuple[int, int]]:
    """Row spans of the parameters ``_concatenate_grads`` gathers, in its order.

    ``_concatenate_grads`` flattens each parameter's grad and concatenates, so
    the slot is the whole ``[out_features, in_features]`` gradient flattened --
    *provided* the parameters tile the row axis contiguously and in order. This
    derives the spans from the live parameter shapes (never from ``split_kind``)
    so the assumption is checked against the module rather than re-derived.

    Raises if the spans do not exactly tile ``[0, out_features)``: that is the
    "some parameter's grad was None and got skipped" case, where the device slot
    is silently shorter than the reference and writing a golden anyway would
    persist a shape-mismatched artifact.
    """
    spans: List[Tuple[int, int]] = []
    offset = 0
    for name, parameter in named_parameters:
        if not name.endswith(suffix) or parameter.grad is None:
            continue
        rows = parameter.shape[0]
        spans.append((offset, offset + rows))
        offset += rows
    if offset != out_features:
        raise RuntimeError(
            "parameter rows for suffix {!r} cover {} rows, expected {}; the "
            "flatten-and-concatenate layout assumption no longer holds".format(
                suffix, offset, out_features
            )
        )
    return spans


def to_slot(reference: torch.Tensor, slot: torch.Tensor, name: str) -> torch.Tensor:
    """Cast an fp64 reference onto an existing device slot, preserving its contract.

    Empty slots pass through untouched. Otherwise the reference must match the
    slot element-for-element; a mismatch is raised, never silently reshaped --
    ATK compares element counts and dtypes across the two sides.
    """
    if slot.numel() == 0:
        return slot
    if reference.numel() != slot.numel():
        raise RuntimeError(
            "{}: reference has {} elements, device slot has {}".format(
                name, reference.numel(), slot.numel()
            )
        )
    return reference.to(device=slot.device, dtype=slot.dtype)


def flatten_row_blocks(full2d: torch.Tensor, spans: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """``cat([full[r0:r1].reshape(-1) ...])`` -- equals ``full.reshape(-1)``."""
    return torch.cat([full2d[start:stop].reshape(-1) for start, stop in spans])
