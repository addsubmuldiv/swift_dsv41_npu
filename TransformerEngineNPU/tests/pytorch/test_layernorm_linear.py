# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tests for ``LayerNormLinear`` covering the lazy-wgrad and refactored paths.

These tests focus on the changes introduced by the alignment with
``linear.py``: dataclass-based args, ``WeightGradStore``-driven delayed
wgrad, ``_layernorm_linear_forward_impl`` / ``_layernorm_linear_backward``
split, and the ``fsdp_group`` plumbing for cached FP8 workspaces.

The numerical path is exercised at the ``LayerNormLinear`` user API level
so the entire forward / backward / ``backward_dw`` orchestration is
covered end-to-end without depending on NPU-specific quantization
kernels that are not available in this CI environment.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from transformer_engine.pytorch import LayerNormLinear
from transformer_engine.pytorch.module._common import WeightGradStore

from utils import npu_available


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_BF16_FORWARD_TOLS = {"rtol": 2e-2, "atol": 2e-2}
_BF16_GRAD_TOLS = {"rtol": 5e-2, "atol": 2.5e-1}


def _npu_device() -> torch.device:
    return torch.device("npu")


def _make_inputs(
    *,
    batch: int = 4,
    seqlen: int = 8,
    in_features: int = 16,
    out_features: int = 32,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    inp = torch.randn(batch, seqlen, in_features, dtype=dtype, device=_npu_device())
    weight = torch.randn(out_features, in_features, dtype=dtype, device=_npu_device())
    bias = torch.randn(out_features, dtype=dtype, device=_npu_device())
    return inp, weight, bias


def _assert_close(name, actual, expected, tolerances):
    torch.testing.assert_close(
        actual,
        expected,
        msg=lambda msg: f"{name} mismatch\n{msg}",
        **tolerances,
    )


# ---------------------------------------------------------------------------
# Forward / Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
def test_forward_backward_matches_reference(
    dtype,
    normalization,
    zero_centered_gamma,
    use_bias,
):
    """Forward + backward correctness in non-fp8 path against PyTorch reference."""
    inp, weight, bias = _make_inputs(dtype=dtype)
    inp_test = inp.detach().clone().requires_grad_(True)
    inp_ref = inp.detach().clone().requires_grad_(True)
    layer = LayerNormLinear(
        16,
        32,
        eps=1e-5,
        bias=use_bias,
        normalization=normalization,
        zero_centered_gamma=zero_centered_gamma,
        params_dtype=dtype,
        device=_npu_device(),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
        if use_bias:
            layer.bias.copy_(bias)

    ref_weight = layer.weight.detach().clone().requires_grad_(True)
    ref_bias = layer.bias.detach().clone().requires_grad_(True) if use_bias else None
    ref_ln_weight = layer.layer_norm_weight.detach().clone().requires_grad_(True)
    ref_ln_bias = (
        layer.layer_norm_bias.detach().clone().requires_grad_(True)
        if layer.layer_norm_bias is not None
        else None
    )
    gamma = 1 + ref_ln_weight if zero_centered_gamma else ref_ln_weight
    if normalization == "LayerNorm":
        ref_ln_out = torch.nn.functional.layer_norm(
            inp_ref,
            [inp_ref.shape[-1]],
            weight=gamma,
            bias=ref_ln_bias,
            eps=layer.eps,
        )
    else:
        # npu_rms_norm accumulates the normalization statistics in FP32 even
        # when its input and weight are BF16. Match that numerical model in the
        # reference, then cast back to the kernel output dtype before GEMM.
        inp_ref_fp32 = inp_ref.float()
        ref_ln_out = inp_ref_fp32 * torch.rsqrt(
            inp_ref_fp32.square().mean(dim=-1, keepdim=True) + layer.eps
        )
        ref_ln_out = (ref_ln_out * gamma.float()).to(dtype)
    ref_out = torch.nn.functional.linear(ref_ln_out, ref_weight, ref_bias)

    out = layer(inp_test)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    ref_out.backward(grad_out)

    _assert_close("output", out, ref_out, _BF16_FORWARD_TOLS)
    _assert_close("input gradient", inp_test.grad, inp_ref.grad, _BF16_GRAD_TOLS)
    _assert_close("linear weight gradient", layer.weight.grad, ref_weight.grad, _BF16_GRAD_TOLS)
    _assert_close(
        "normalization weight gradient",
        layer.layer_norm_weight.grad,
        ref_ln_weight.grad,
        _BF16_GRAD_TOLS,
    )
    if use_bias:
        _assert_close("linear bias gradient", layer.bias.grad, ref_bias.grad, _BF16_GRAD_TOLS)
    if normalization == "LayerNorm":
        _assert_close(
            "normalization bias gradient",
            layer.layer_norm_bias.grad,
            ref_ln_bias.grad,
            _BF16_GRAD_TOLS,
        )


# ---------------------------------------------------------------------------
# Delayed wgrad
# ---------------------------------------------------------------------------


def test_backward_dw_assigns_wgrad_after_delay():
    """When ``delay_wgrad_compute=True``, ``backward_dw`` must populate wgrad."""
    inp, weight, bias = _make_inputs()
    layer = LayerNormLinear(
        16,
        32,
        bias=True,
        delay_wgrad_compute=True,
        params_dtype=torch.bfloat16,
        device=_npu_device(),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    assert isinstance(layer.wgrad_store, WeightGradStore)
    for name, param in layer.named_parameters():
        if name in layer.weight_names or name in layer.bias_names:
            assert getattr(param, "skip_backward_post_hook", False) is True

    out = layer(inp)
    grad_out = torch.randn_like(out)
    with torch.no_grad():
        ln_out = torch.nn.functional.layer_norm(
            inp,
            [inp.shape[-1]],
            weight=layer.layer_norm_weight,
            bias=layer.layer_norm_bias,
            eps=layer.eps,
        )
        expected_wgrad = (
            grad_out.reshape(-1, grad_out.shape[-1])
            .float()
            .t()
            .matmul(ln_out.reshape(-1, ln_out.shape[-1]).float())
        )
    out.backward(grad_out)

    # With skip_backward_post_hook set, autograd skips materializing .grad on
    # the weight. ``backward_dw`` must do it explicitly via the WeightGradStore.
    assert layer.weight.grad is None
    assert layer.need_backward_dw()

    layer.backward_dw()
    _assert_close(
        "delayed weight gradient",
        layer.weight.grad.float(),
        expected_wgrad,
        _BF16_GRAD_TOLS,
    )


def test_delayed_fused_wgrad_first_microbatch_overwrites_main_grad():
    """The first delayed microbatch must write, rather than discard, ``main_grad``."""
    inp, weight, _ = _make_inputs()
    inp.requires_grad_(True)
    layer = LayerNormLinear(
        16,
        32,
        bias=False,
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=True,
        params_dtype=torch.bfloat16,
        device=_npu_device(),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.weight.main_grad = torch.full_like(layer.weight, 7.0, dtype=torch.float32)

    out = layer(inp, is_first_microbatch=True)
    grad_out = torch.randn_like(out)
    with torch.no_grad():
        ln_out = torch.nn.functional.layer_norm(
            inp,
            [inp.shape[-1]],
            weight=layer.layer_norm_weight,
            bias=layer.layer_norm_bias,
            eps=layer.eps,
        )
        expected_wgrad = (
            grad_out.reshape(-1, grad_out.shape[-1])
            .float()
            .t()
            .matmul(ln_out.reshape(-1, ln_out.shape[-1]).float())
        )

    out.backward(grad_out)
    torch.testing.assert_close(
        layer.weight.main_grad,
        torch.full_like(layer.weight.main_grad, 7.0),
    )

    layer.backward_dw()
    _assert_close(
        "fused main gradient",
        layer.weight.main_grad,
        expected_wgrad,
        _BF16_GRAD_TOLS,
    )


def test_delayed_wgrad_fallback_materializes_weight_grad():
    """Unsupported fused accumulation must fall back without dropping delayed wgrad."""
    inp, weight, _ = _make_inputs()
    layer = LayerNormLinear(
        16,
        32,
        bias=False,
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=True,
        params_dtype=torch.bfloat16,
        device=_npu_device(),
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
    # Non-FP8 fused accumulation only supports an FP32 main_grad. This forces
    # the ordinary-wgrad fallback while the public fuse flag remains enabled.
    layer.weight.main_grad = torch.zeros_like(layer.weight)

    out = layer(inp)
    grad_out = torch.randn_like(out)
    with torch.no_grad():
        ln_out = torch.nn.functional.layer_norm(
            inp,
            [inp.shape[-1]],
            weight=layer.layer_norm_weight,
            bias=layer.layer_norm_bias,
            eps=layer.eps,
        )
        expected_wgrad = (
            grad_out.reshape(-1, grad_out.shape[-1])
            .float()
            .t()
            .matmul(ln_out.reshape(-1, ln_out.shape[-1]).float())
        )
    out.backward(grad_out)
    layer.backward_dw()
    _assert_close(
        "fallback weight gradient",
        layer.weight.grad.float(),
        expected_wgrad,
        _BF16_GRAD_TOLS,
    )


# ---------------------------------------------------------------------------
# return_layernorm_output
# ---------------------------------------------------------------------------


def test_return_layernorm_output_gathered_preserves_backward_at_tp1():
    """The gathered flag must not discard the returned normalization gradient at TP=1."""
    inp, weight, bias = _make_inputs()
    inp_gathered = inp.detach().clone().requires_grad_(True)
    inp_local = inp.detach().clone().requires_grad_(True)

    gathered = LayerNormLinear(
        16,
        32,
        return_layernorm_output=True,
        return_layernorm_output_gathered=True,
        params_dtype=torch.bfloat16,
        device=_npu_device(),
    )
    local = LayerNormLinear(
        16,
        32,
        return_layernorm_output=True,
        return_layernorm_output_gathered=False,
        params_dtype=torch.bfloat16,
        device=_npu_device(),
    )
    with torch.no_grad():
        gathered.weight.copy_(weight)
        gathered.bias.copy_(bias)
        local.load_state_dict(gathered.state_dict())

    gathered_out, gathered_ln = gathered(inp_gathered)
    local_out, local_ln = local(inp_local)
    (gathered_out.float().sum() + gathered_ln.float().sum()).backward()
    (local_out.float().sum() + local_ln.float().sum()).backward()

    torch.testing.assert_close(inp_gathered.grad, inp_local.grad, rtol=1e-2, atol=1e-2)
    for gathered_param, local_param in zip(gathered.parameters(), local.parameters()):
        torch.testing.assert_close(
            gathered_param.grad,
            local_param.grad,
            rtol=1e-2,
            atol=1e-2,
        )
