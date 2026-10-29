# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Functional tests for the NVTE-compatible ``backward_override`` feature."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from utils import (
    assert_close,
    make_reference_and_test_tensors,
    npu_available,
    quantization_tols,
)

from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    GroupedLinear,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    autocast,
)
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    is_fp8_available,
    is_mxfp8_available,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_TOLS = quantization_tols("mxfp8_block_scaling")
# Plain (non-FP8) wgrad tolerances: a bf16 GEMM output compared against an fp32
# reference, and an fp32 accumulation of exact bf16 products.
_BF16_WGRAD_TOLS = dict(rtol=5e-2, atol=5e-2)
_FP32_WGRAD_TOLS = dict(rtol=1e-3, atol=1e-3)
_NUM_GEMMS = 2
_IN_FEATURES = 16
_OUT_FEATURES = 32
_SEQ_LEN = 32


def _reset_fp8_state() -> None:
    FP8GlobalStateManager.reset()


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Deterministic data across runs: tolerance margins are calibrated on
    fixed inputs, so random drift must not eat into them.
    """
    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)


def _fp32_pair(shape: int | tuple[int, ...], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair of float32 CPU reference and bf16 NPU test tensors."""
    kwargs.setdefault("test_dtype", torch.bfloat16)
    kwargs.setdefault("test_device", torch.device("npu"))
    return make_reference_and_test_tensors(shape, ref_dtype=torch.float32, **kwargs)


def _layernorm_backward_reference(
    dgrad_2d: torch.Tensor,
    input_2d: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate the LayerNorm backward math of ``layernorm_linear.py``.

    Returns ``(dx, dgamma)`` for the given upstream gradient, input, and norm
    weight; dtypes are promoted to the widest input automatically.
    """
    mean = input_2d.mean(dim=-1, keepdim=True)
    var = input_2d.var(dim=-1, unbiased=False, keepdim=True)
    rsigma = torch.rsqrt(var + eps)
    x_hat = (input_2d - mean) * rsigma

    dgamma = (dgrad_2d * x_hat).sum(dim=0)
    dx_hat = dgrad_2d * gamma

    n = input_2d.shape[-1]
    dvar = (dx_hat * (input_2d - mean) * (-0.5) * (rsigma**3)).sum(dim=-1, keepdim=True)
    dmean = (-dx_hat * rsigma).sum(dim=-1, keepdim=True) + dvar * (-2.0 / n) * (
        input_2d - mean
    ).sum(dim=-1, keepdim=True)
    dx = dx_hat * rsigma + dvar * 2.0 / n * (input_2d - mean) + dmean / n
    return dx, dgamma


def _grouped_linear_reference(
    ref_inp: torch.Tensor,
    ref_weights: list[tuple[torch.Tensor, torch.Tensor]],
    m_splits: list[int],
) -> torch.Tensor:
    splits = torch.split(ref_inp, m_splits, dim=0)
    return torch.cat(
        [F.linear(x, w) for x, (w, _) in zip(splits, ref_weights)],
        dim=0,
    )


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_linear_backward_override_matches_reference(override: str) -> None:
    """Linear backward uses the requested precision mode without FP8 operands."""
    ref_inp, inp = _fp32_pair(
        (_SEQ_LEN, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    ref_weight, weight = _fp32_pair(
        (_OUT_FEATURES, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    ref_grad, grad = _fp32_pair(
        (_SEQ_LEN, _OUT_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
        requires_grad=False,
    )

    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    try:
        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp)
        out.backward(grad)

        ref_out = F.linear(ref_inp, ref_weight)
        ref_out.backward(ref_grad)

        assert_close(out, ref_out, **_TOLS)
        assert_close(inp.grad, ref_inp.grad, **_TOLS)
        assert_close(layer.weight.grad, ref_weight.grad, **_TOLS)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_layernorm_linear_backward_override_matches_reference(override: str) -> None:
    """LayerNormLinear override keeps forward/backward numerically correct."""
    _, ln_weight = _fp32_pair((_IN_FEATURES,))
    _, weight = _fp32_pair((_OUT_FEATURES, _IN_FEATURES))
    _, inp = _fp32_pair((_SEQ_LEN, _IN_FEATURES))
    _, grad = _fp32_pair((_SEQ_LEN, _OUT_FEATURES), requires_grad=False)

    layer = LayerNormLinear(
        _IN_FEATURES,
        _OUT_FEATURES,
        eps=1e-5,
        bias=False,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    layer_ref = LayerNormLinear(
        _IN_FEATURES,
        _OUT_FEATURES,
        eps=layer.eps,
        bias=False,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.layer_norm_weight.copy_(ln_weight)
        layer.weight.copy_(weight)
        layer_ref.layer_norm_weight.copy_(ln_weight)
        layer_ref.weight.copy_(weight)

    try:
        # Forward is unaffected by the flag: compare bitwise against a
        # default-recipe run of an identical layer (same quantization path).
        with autocast(enabled=True, recipe=recipe.MXFP8BlockScaling()):
            y_ref = layer_ref(inp)

        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp)
        out.backward(grad)

        assert_close(out, y_ref, rtol=0, atol=0, check_dtype=True)

        # Plain-math references built from the actual bf16 operands (not the
        # float originals, whose bf16 rounding differs from what ran).
        ln_out_f = F.layer_norm(
            inp.detach().float(),
            (_IN_FEATURES,),
            ln_weight.detach().float(),
            None,
            layer.eps,
        )
        expected_dw = grad.detach().float().t() @ ln_out_f
        d_ln_out = grad.detach().float() @ weight.detach().float()
        expected_dx, expected_dgamma = _layernorm_backward_reference(
            d_ln_out, inp.detach().float(), ln_weight.detach().float(), layer.eps
        )

        # LayerNorm backward amplifies operand noise through cancellation, so
        # a handful of near-zero outputs always deviate beyond any per-element
        # bound. Validate the two quantities that separate real defects from
        # benign cancellation tails: the relative L2 error (systematic
        # operand/shape/sign mistakes blow this up) and the error relative to
        # the reference peak (localized spikes).
        def _assert_grad_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
            err = (actual.float() - expected.float()).abs()
            ref_mag = expected.float().abs().max()
            rel_l2 = (err.norm() / expected.float().norm()).item()
            rel_peak = (err.max() / ref_mag).item()
            l2_bound, peak_bound = (0.02, 0.05)
            if override == "dequantized":
                # One extra MXFP8 round trip on the saved wgrad operand and
                # the dgrad weight.
                l2_bound, peak_bound = (0.04, 0.10)
            assert rel_l2 <= l2_bound, f"rel L2 error {rel_l2:.4f} > {l2_bound}"
            assert rel_peak <= peak_bound, f"rel peak error {rel_peak:.4f} > {peak_bound}"

        _assert_grad_close(inp.grad, expected_dx)
        _assert_grad_close(layer.weight.grad, expected_dw)
        _assert_grad_close(layer.layer_norm_weight.grad, expected_dgamma)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("single_grouped_weight", [False, True])
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_grouped_linear_backward_override_matches_reference(
    override: str,
    single_grouped_weight: bool,
) -> None:
    """GroupedLinear override matches per-expert dense reference gradients."""
    m_splits = [_SEQ_LEN // _NUM_GEMMS] * _NUM_GEMMS
    ref_weights = [
        _fp32_pair(
            (_OUT_FEATURES, _IN_FEATURES),
            test_dtype=torch.bfloat16,
            test_device=torch.device("npu"),
        )
        for _ in range(_NUM_GEMMS)
    ]
    ref_inp, inp = _fp32_pair(
        (_SEQ_LEN, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    ref_grad, grad = _fp32_pair(
        (_SEQ_LEN, _OUT_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
        requires_grad=False,
    )

    layer = GroupedLinear(
        _NUM_GEMMS,
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
        single_grouped_weight=single_grouped_weight,
    )
    with torch.no_grad():
        if single_grouped_weight:
            layer.weight.copy_(torch.stack([w for _, w in ref_weights], dim=0))
        else:
            for i, (_, w) in enumerate(ref_weights):
                getattr(layer, f"weight{i}").copy_(w)

    ref_out = _grouped_linear_reference(ref_inp, ref_weights, m_splits)

    # The module quantizes each expert's input rows to MXFP8 before the GEMM;
    # route the forward expectation through the same quantization round trip
    # (rowwise blocks tile K and rows are independent, so quantizing the full
    # tensor equals per-expert quantization).
    inp_operand = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(inp.detach()).dequantize()
    operand_splits = torch.split(inp_operand.float(), m_splits, dim=0)
    expected_out = torch.cat(
        [F.linear(s, w.detach().float()) for s, (_, w) in zip(operand_splits, ref_weights)],
        dim=0,
    )

    try:
        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp, m_splits)
        out.backward(grad)
        ref_out.backward(ref_grad)

        assert_close(out, expected_out, **_TOLS)
        assert_close(inp.grad, ref_inp.grad, **_TOLS)
        if single_grouped_weight:
            expected_wgrad = torch.stack([w.grad for w, _ in ref_weights], dim=0)
            assert_close(layer.weight.grad, expected_wgrad, **_TOLS)
        else:
            for i, (ref_w, _) in enumerate(ref_weights):
                assert_close(getattr(layer, f"weight{i}").grad, ref_w.grad, **_TOLS)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_linear_fuse_wgrad_accumulation_main_grad(override: str) -> None:
    """Fused Linear wgrad is accumulated into main_grad instead of param.grad."""
    _, weight = _fp32_pair(
        (_OUT_FEATURES, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    _, inp = _fp32_pair(
        (_SEQ_LEN, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    _, grad = _fp32_pair(
        (_SEQ_LEN, _OUT_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
        requires_grad=False,
    )

    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        fuse_wgrad_accumulation=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)

    try:
        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp, is_first_microbatch=None)
        out.backward(grad)

        expected = grad.detach().float().t() @ inp.detach().float()
        assert layer.weight.grad is None
        assert_close(layer.weight.main_grad, expected, **_TOLS)
    finally:
        _reset_fp8_state()


def test_linear_non_fp32_main_grad_falls_back_to_plain_wgrad() -> None:
    """A non-FP32 ``main_grad`` must disable fused accumulation in backward.

    The FP32 gate on ``main_grad`` is evaluated in backward (the buffer is only
    guaranteed to exist there), so a non-FP32 buffer has to fall back to the
    plain wgrad path: the gradient lands in ``weight.grad`` and the pre-existing
    ``main_grad`` buffer stays untouched.
    """
    _, weight = _fp32_pair((_OUT_FEATURES, _IN_FEATURES))
    _, inp = _fp32_pair((_SEQ_LEN, _IN_FEATURES))
    _, grad = _fp32_pair((_SEQ_LEN, _OUT_FEATURES), requires_grad=False)

    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        fuse_wgrad_accumulation=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
    main_grad = torch.zeros_like(layer.weight, dtype=torch.bfloat16)
    layer.weight.main_grad = main_grad

    out = layer(inp, is_first_microbatch=None)
    out.backward(grad)

    expected = grad.detach().float().t() @ inp.detach().float()
    assert layer.weight.grad is not None
    assert_close(layer.weight.grad, expected, **_BF16_WGRAD_TOLS)
    assert_close(main_grad, torch.zeros_like(main_grad), rtol=0, atol=0)


@pytest.mark.parametrize(
    "main_grad_dtype, expect_fused",
    [(torch.float32, True), (torch.bfloat16, False)],
    ids=["fp32-main-grad", "bf16-main-grad"],
)
def test_linear_fsdp_lazy_main_grad_dtype_gate(
    main_grad_dtype: torch.dtype,
    expect_fused: bool,
) -> None:
    """FSDP params expose ``main_grad`` only in backward, via ``get_main_grad()``.

    Forward must not probe ``weight.main_grad`` (it does not exist yet); the
    FP32 dtype gate then runs in backward once the buffer is materialized,
    selecting the fused path for an FP32 buffer and the plain path otherwise.
    """
    _, weight = _fp32_pair((_OUT_FEATURES, _IN_FEATURES))
    _, inp = _fp32_pair((_SEQ_LEN, _IN_FEATURES))
    _, grad = _fp32_pair((_SEQ_LEN, _OUT_FEATURES), requires_grad=False)

    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        fuse_wgrad_accumulation=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    # Simulate an FSDP-managed parameter: ``main_grad`` is not set eagerly and is
    # only reachable through ``get_main_grad()``. Probing ``weight.main_grad`` in
    # forward (the pre-fix behavior) would raise AttributeError here.
    main_grad = torch.zeros(
        (_OUT_FEATURES, _IN_FEATURES),
        dtype=main_grad_dtype,
        device=torch.device("npu"),
    )

    def _get_main_grad() -> torch.Tensor:
        return main_grad

    layer.weight.__fsdp_param__ = True
    layer.weight.get_main_grad = _get_main_grad
    assert not hasattr(layer.weight, "main_grad")

    out = layer(inp, is_first_microbatch=None)
    out.backward(grad)

    expected = grad.detach().float().t() @ inp.detach().float()
    if expect_fused:
        # wgrad accumulates into ``main_grad``; ``weight.grad`` stays None.
        assert layer.weight.grad is None
        assert_close(main_grad, expected, **_FP32_WGRAD_TOLS)
    else:
        # Non-FP32 buffer: fused accumulation is skipped entirely.
        assert layer.weight.grad is not None
        assert_close(layer.weight.grad, expected, **_BF16_WGRAD_TOLS)
        assert_close(main_grad, torch.zeros_like(main_grad), rtol=0, atol=0)


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("delay_wgrad_compute", [False, True])
@pytest.mark.parametrize("fuse_wgrad_accumulation", [False, True])
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_linear_backward_override_multi_microbatch(
    override: str,
    fuse_wgrad_accumulation: bool,
    delay_wgrad_compute: bool,
) -> None:
    """Override stays correct across gradient-accumulation microbatches.

    Drives explicit ``is_first_microbatch`` bookkeeping over several
    forward/backward passes and validates:

    - per-microbatch input gradients against plain-math references;
    - accumulated weight gradient in ``main_grad`` (fuse path, first
      microbatch intentionally excluded per upstream semantics) or ``grad``
      (non-fuse path);
    - delayed wgrad compute: per-microbatch ``backward_dw()`` results match
      the per-microbatch references.
    """
    num_microbatches = 3
    _, weight = _fp32_pair(
        (_OUT_FEATURES, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    _, inp_full = _fp32_pair(
        (_SEQ_LEN, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    _, grad_full = _fp32_pair(
        (_SEQ_LEN, _OUT_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
        requires_grad=False,
    )
    inp_chunks = list(inp_full.chunk(num_microbatches))
    grad_chunks = list(grad_full.chunk(num_microbatches))

    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        delay_wgrad_compute=delay_wgrad_compute,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
    if fuse_wgrad_accumulation:
        layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)

    try:
        run_inputs = []
        run_dw = []
        for i in range(num_microbatches):
            x = inp_chunks[i].detach().clone().requires_grad_(True)
            run_inputs.append(x)
            with autocast(
                enabled=True,
                recipe=recipe.MXFP8BlockScaling(backward_override=override),
            ):
                out = layer(x, is_first_microbatch=(i == 0))
            out.backward(grad_chunks[i])
            if delay_wgrad_compute:
                layer.backward_dw()
                if not fuse_wgrad_accumulation:
                    run_dw.append(layer.weight.grad.detach().clone())
                    layer.weight.grad = None

        # Per-microbatch input gradients.
        for i, x in enumerate(run_inputs):
            expected_dx = grad_chunks[i].detach().float() @ weight.detach().float()
            assert_close(x.grad, expected_dx, **_TOLS)

        def _expected_dw(idx: int) -> torch.Tensor:
            return grad_chunks[idx].detach().float().t() @ inp_chunks[idx].detach().float()

        def _expected_dw_sum(indices) -> torch.Tensor:
            total = torch.zeros(
                (_OUT_FEATURES, _IN_FEATURES),
                dtype=torch.float32,
                device=torch.device("npu"),
            )
            for idx in indices:
                total += _expected_dw(idx)
            return total

        if fuse_wgrad_accumulation:
            # Upstream semantics: with an explicit ``is_first_microbatch``
            # protocol the first microbatch's wgrad bypasses main_grad
            # accumulation (ctx.fuse_wgrad_accumulation is gated on
            # ``not is_first_microbatch``) and lands in ``weight.grad``
            # instead; the remaining ones add up inside ``main_grad``.
            if delay_wgrad_compute:
                # Delayed path drops the first microbatch's wgrad (stored via
                # a non-fusing closure whose result is never consumed).
                assert layer.weight.grad is None
            else:
                assert_close(
                    layer.weight.grad,
                    _expected_dw(0).to(weight.dtype),
                    **_TOLS,
                )
            assert_close(
                layer.weight.main_grad,
                _expected_dw_sum(range(1, num_microbatches)),
                **_TOLS,
            )
        elif delay_wgrad_compute:
            # backward_dw overwrites .grad with the last stored result; the
            # per-microbatch clones taken above are the real check.
            for i in range(num_microbatches):
                assert_close(run_dw[i], _expected_dw(i), **_TOLS)
        else:
            assert_close(layer.weight.grad, _expected_dw_sum(range(num_microbatches)), **_TOLS)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_grouped_linear_fuse_wgrad_accumulation_main_grad(override: str) -> None:
    """Performance GroupedLinear fused wgrad accumulates into packed main_grad."""
    m_splits = [4, 6]
    ref_weights = [
        _fp32_pair(
            (16, 16),
            test_dtype=torch.bfloat16,
            test_device=torch.device("npu"),
        )
        for _ in range(_NUM_GEMMS)
    ]
    _, inp = _fp32_pair(
        (sum(m_splits), 16),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    _, grad = _fp32_pair(
        (sum(m_splits), 16),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
        requires_grad=False,
    )

    layer = GroupedLinear(
        _NUM_GEMMS,
        16,
        16,
        bias=False,
        fuse_wgrad_accumulation=True,
        single_grouped_weight=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(torch.stack([w for _, w in ref_weights], dim=0))
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)

    try:
        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp, m_splits)
        out.backward(grad)

        inp_splits = torch.split(inp.detach().float(), m_splits, dim=0)
        grad_splits = torch.split(grad.detach().float(), m_splits, dim=0)
        expected = torch.stack(
            [g.t() @ x for x, g in zip(inp_splits, grad_splits)],
            dim=0,
        )
        assert layer.weight.grad is None
        assert_close(layer.weight.main_grad, expected, **_TOLS)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_grouped_linear_frozen_weight_backward_matches_reference(override: str) -> None:
    """Frozen grouped weights still produce dgrad with a stable saved-tensor layout."""
    m_splits = [_SEQ_LEN // _NUM_GEMMS] * _NUM_GEMMS
    ref_weights = [
        _fp32_pair(
            (_OUT_FEATURES, _IN_FEATURES),
            test_dtype=torch.bfloat16,
            test_device=torch.device("npu"),
        )
        for _ in range(_NUM_GEMMS)
    ]
    _, inp = _fp32_pair(
        (_SEQ_LEN, _IN_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
    )
    _, grad = _fp32_pair(
        (_SEQ_LEN, _OUT_FEATURES),
        test_dtype=torch.bfloat16,
        test_device=torch.device("npu"),
        requires_grad=False,
    )

    layer = GroupedLinear(
        _NUM_GEMMS,
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        single_grouped_weight=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(torch.stack([w for _, w in ref_weights], dim=0))
    layer.weight.requires_grad_(False)

    try:
        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp, m_splits)
        out.backward(grad)

        grad_splits = torch.split(grad.detach().float(), m_splits, dim=0)
        expected = torch.cat(
            [
                grad_split.cpu() @ weight
                for grad_split, (weight, _) in zip(grad_splits, ref_weights)
            ],
            dim=0,
        )
        assert_close(inp.grad, expected, **_TOLS)
        assert layer.weight.grad is None
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_layernorm_mlp_backward_override_not_implemented(override: str) -> None:
    """LayerNormMLP rejects backward override, matching upstream NVTE behavior."""
    layer = LayerNormMLP(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    inp = torch.randn(
        (8, _IN_FEATURES),
        dtype=torch.bfloat16,
        device=torch.device("npu"),
        requires_grad=True,
    )

    try:
        with (
            autocast(
                enabled=True,
                recipe=recipe.MXFP8BlockScaling(backward_override=override),
            ),
            pytest.raises(NotImplementedError, match="LayerNormMLP"),
        ):
            layer(inp)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_linear_backward_override_with_bias(override: str) -> None:
    """bias=True exercises the unfused bgrad path under the override."""
    _, inp = _fp32_pair((_SEQ_LEN, _IN_FEATURES))
    _, weight = _fp32_pair((_OUT_FEATURES, _IN_FEATURES))
    _, bias = _fp32_pair((_OUT_FEATURES,))
    _, grad = _fp32_pair((_SEQ_LEN, _OUT_FEATURES), requires_grad=False)

    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)

    try:
        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp)
        out.backward(grad)

        expected_out = F.linear(
            inp.detach().float(), weight.detach().float(), bias.detach().float()
        )
        expected_dx = grad.detach().float() @ weight.detach().float()
        expected_dw = grad.detach().float().t() @ inp.detach().float()
        assert_close(out, expected_out, **_TOLS)
        assert_close(inp.grad, expected_dx, **_TOLS)
        assert_close(layer.weight.grad, expected_dw, **_TOLS)
        assert_close(layer.bias.grad, grad.detach().sum(dim=0), **_TOLS)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(not is_fp8_available(), reason="FP8 requires an FP8-capable NPU")
def test_linear_save_original_input_falls_back_on_delayed_scaling() -> None:
    """save_original_input is gracefully disabled under delayed scaling."""
    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        save_original_input=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    inp = torch.randn((_SEQ_LEN, _IN_FEATURES), dtype=torch.bfloat16, device=torch.device("npu"))

    try:
        with (
            pytest.warns(UserWarning, match="DelayedScaling recipe is not supported"),
            autocast(enabled=True, recipe=recipe.DelayedScaling()),
        ):
            out = layer(inp)
        assert out.shape == (_SEQ_LEN, _OUT_FEATURES)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
def test_linear_save_original_input_matches_reference() -> None:
    """save_original_input keeps forward bitwise and backward within tolerance."""
    _, inp = _fp32_pair((_SEQ_LEN, _IN_FEATURES))
    _, weight = _fp32_pair((_OUT_FEATURES, _IN_FEATURES))
    _, grad = _fp32_pair((_SEQ_LEN, _OUT_FEATURES), requires_grad=False)

    layer_soi = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        save_original_input=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    layer_def = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    for layer in (layer_soi, layer_def):
        with torch.no_grad():
            layer.weight.copy_(weight)
    x_soi = inp.detach().clone().requires_grad_(True)
    x_def = inp.detach().clone().requires_grad_(True)

    try:
        # Forward is unaffected: identical quantization operands.
        with autocast(enabled=True, recipe=recipe.MXFP8BlockScaling()):
            out_soi = layer_soi(x_soi)
            out_def = layer_def(x_def)
        assert_close(out_soi, out_def, rtol=0, atol=0, check_dtype=True)

        # Backward re-quantizes the saved original input into a columnwise
        # view; numerics align with the default path within tolerances.
        out_soi.backward(grad)
        out_def.backward(grad)

        expected_dx = grad.detach().float() @ weight.detach().float()
        expected_dw = grad.detach().float().t() @ inp.detach().float()
        assert_close(x_soi.grad, expected_dx, **_TOLS)
        assert_close(layer_soi.weight.grad, expected_dw, **_TOLS)
        assert_close(layer_soi.weight.grad, layer_def.weight.grad, rtol=0.25, atol=0.09375)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(not is_fp8_available(), reason="FP8 requires an FP8-capable NPU")
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_linear_backward_override_current_scaling_matches_reference(
    override: str,
) -> None:
    """Float8CurrentScaling honors the override like MXFP8 block scaling."""
    ref_inp, inp = _fp32_pair((_SEQ_LEN, _IN_FEATURES))
    ref_weight, weight = _fp32_pair((_OUT_FEATURES, _IN_FEATURES))
    _, grad = _fp32_pair((_SEQ_LEN, _OUT_FEATURES), requires_grad=False)

    layer = Linear(
        _IN_FEATURES,
        _OUT_FEATURES,
        bias=False,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    try:
        with autocast(
            enabled=True,
            recipe=recipe.Float8CurrentScaling(backward_override=override),
        ):
            out = layer(inp)
        out.backward(grad)

        expected_out = F.linear(ref_inp, ref_weight)
        expected_dx = grad.detach().float() @ weight.detach().float()
        expected_dw = grad.detach().float().t() @ inp.detach().float()
        assert_close(out, expected_out, **_TOLS)
        assert_close(inp.grad, expected_dx, **_TOLS)
        assert_close(layer.weight.grad, expected_dw, **_TOLS)
    finally:
        _reset_fp8_state()


@pytest.mark.skipif(
    not is_mxfp8_available(),
    reason="MXFP8 requires an FP8-capable NPU",
)
@pytest.mark.parametrize("override", ["high_precision", "dequantized"])
def test_grouped_linear_delay_wgrad_compute_under_override(override: str) -> None:
    """Delayed grouped wgrad lands in weight.grad after backward_dw()."""
    m_splits = [4, 6]
    _, weight = _fp32_pair((_NUM_GEMMS, 16, 16))
    _, inp = _fp32_pair((sum(m_splits), 16))
    _, grad = _fp32_pair((sum(m_splits), 16), requires_grad=False)

    layer = GroupedLinear(
        _NUM_GEMMS,
        16,
        16,
        bias=False,
        delay_wgrad_compute=True,
        single_grouped_weight=True,
        params_dtype=torch.bfloat16,
        device=torch.device("npu"),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    try:
        with autocast(
            enabled=True,
            recipe=recipe.MXFP8BlockScaling(backward_override=override),
        ):
            out = layer(inp, m_splits)
        out.backward(grad)
        layer.backward_dw()

        inp_splits = torch.split(inp.detach().float(), m_splits, dim=0)
        grad_splits = torch.split(grad.detach().float(), m_splits, dim=0)
        expected = torch.stack(
            [g.t() @ x for x, g in zip(inp_splits, grad_splits)],
            dim=0,
        )
        assert_close(layer.weight.grad, expected, **_TOLS)
    finally:
        _reset_fp8_state()


def test_nvte_backward_override_env_semantics() -> None:
    """NVTE_BACKWARD_OVERRIDE seeds recipe defaults at import and fails fast.

    The value is read when the dataclass default is evaluated (import time),
    so it must survive into instances; invalid values must abort the process
    instead of silently mis-quantizing.
    """
    import os
    import subprocess
    import sys

    code = (
        "import transformer_engine.common.recipe as r; "
        "print(r.MXFP8BlockScaling().backward_override)"
    )
    env_ok = {**os.environ, "NVTE_BACKWARD_OVERRIDE": "high_precision"}
    result = subprocess.run(  # pylint: disable=subprocess-run-check
        [sys.executable, "-c", code],
        env=env_ok,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "high_precision"

    env_bad = {**os.environ, "NVTE_BACKWARD_OVERRIDE": "invalid_mode"}
    result = subprocess.run(  # pylint: disable=subprocess-run-check
        [sys.executable, "-c", code],
        env=env_bad,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "NVTE_BACKWARD_OVERRIDE" in (result.stderr + result.stdout)
