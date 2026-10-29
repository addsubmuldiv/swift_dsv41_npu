# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Module-level sanity coverage adapted from NVIDIA's PyTorch sanity tests.

These tests intentionally exercise user-facing PyTorch modules instead of
``transformer_engine_torch``/``tex`` single operators.  The NPU port keeps a few
backend-specific layout choices, so this file validates the high-level training
path: module construction, autocast recipe wiring, forward/backward execution,
gradient materialization, zero-token handling, and quantized parameter init.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    GroupedLinear,
    Linear,
    autocast,
    quantized_model_init,
)
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
    is_w4a8_available,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor import (
    Float8BlockwiseQTensor,
    Float8Tensor,
    MXFP8Tensor,
)
from transformer_engine.pytorch.utils import init_method_normal

from utils import npu_available


@dataclass(frozen=True)
class ModelConfig:
    max_seqlen_q: int
    batch_size: int
    hidden_size: int
    num_layers: int = 2


_MODEL_CONFIGS = {
    "small": ModelConfig(max_seqlen_q=32, batch_size=2, hidden_size=32),
    "weird": ModelConfig(max_seqlen_q=37, batch_size=3, hidden_size=23),
}

_DTYPES = [torch.float16, torch.bfloat16]
_GRAD_CASES = [
    pytest.param(False, False, id="all_grads"),
    pytest.param(True, False, id="skip_wgrad"),
    pytest.param(False, True, id="skip_dgrad"),
]

_fp8_available, _reason_for_no_fp8 = is_fp8_available(return_reason=True)
_fp8_block_scaling_available, _reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
    return_reason=True
)
_mxfp8_available, _reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
_w4a8_available, _reason_for_no_w4a8 = is_w4a8_available(return_reason=True)


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


def _npu_device() -> torch.device:
    return torch.device("npu")


def _randn_npu(shape, *, dtype: torch.dtype, requires_grad: bool = False) -> torch.Tensor:
    """Generate random test inputs without exercising NPU random kernels."""
    return (
        torch.randn(shape, dtype=dtype, device="cpu")
        .to(device=_npu_device())
        .requires_grad_(requires_grad)
    )


def _seed() -> None:
    seed = 1234
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def _sync() -> None:
    torch.npu.synchronize()


def _scaled_init_method_normal(
    sigma: float, num_layers: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Match NVIDIA sanity test intent without depending on CUDA-only helpers."""

    return init_method_normal(sigma / (2.0 * num_layers) ** 0.5)


def _module_recipe_cases() -> list[pytest.ParameterSet]:
    cases = [
        pytest.param("small", None, id="small_no_fp8"),
        pytest.param("weird", None, id="weird_no_fp8"),
    ]
    if _fp8_available:
        cases.extend(
            [
                pytest.param(
                    "small",
                    recipe.DelayedScaling,
                    id="small_delayed",
                    marks=pytest.mark.skipif(
                        not _fp8_available,
                        reason=_reason_for_no_fp8,
                    ),
                ),
                pytest.param(
                    "small",
                    recipe.Float8CurrentScaling,
                    id="small_current",
                    marks=pytest.mark.skipif(
                        not _fp8_available,
                        reason=_reason_for_no_fp8,
                    ),
                ),
            ]
        )
    if _mxfp8_available:
        cases.append(
            pytest.param(
                "small",
                recipe.MXFP8BlockScaling,
                id="small_mxfp8",
                marks=pytest.mark.skipif(
                    not _mxfp8_available,
                    reason=_reason_for_no_mxfp8,
                ),
            )
        )
    return cases


def _fp8_primary_weight_cases() -> list[pytest.ParameterSet]:
    return [
        pytest.param(
            recipe.DelayedScaling,
            Float8Tensor,
            marks=pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8),
            id="delayed",
        ),
        pytest.param(
            recipe.Float8CurrentScaling,
            Float8Tensor,
            marks=pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8),
            id="current",
        ),
        pytest.param(
            recipe.Float8BlockScaling,
            Float8BlockwiseQTensor,
            marks=pytest.mark.skipif(
                not _fp8_block_scaling_available,
                reason=_reason_for_no_fp8_block_scaling,
            ),
            id="fp8_blockwise",
        ),
        pytest.param(
            recipe.MXFP8BlockScaling,
            MXFP8Tensor,
            marks=pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8),
            id="mxfp8",
        ),
    ]


def _make_recipe(recipe_cls: Optional[type[recipe.Recipe]]) -> Optional[recipe.Recipe]:
    return None if recipe_cls is None else recipe_cls()


def _is_fp8_supported(config: ModelConfig) -> bool:
    return config.max_seqlen_q * config.batch_size % 16 == 0 and config.hidden_size % 16 == 0


def _skip_unsupported_fp8_config(config: ModelConfig, fp8_recipe: Optional[recipe.Recipe]) -> None:
    if fp8_recipe is None:
        return
    if not _is_fp8_supported(config):
        pytest.skip("Model config does not support FP8")


def _disable_wgrads(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


@pytest.fixture(autouse=True)
def _reset_global_fp8_state():
    _seed()
    yield
    FP8GlobalStateManager.reset()


def _first_tensor(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _assert_quantized_storage_equal(
    actual: QuantizedTensor,
    expected: QuantizedTensor,
    *,
    primary_only: bool = False,
) -> None:
    assert actual.__class__ is expected.__class__
    assert actual.origin_shape == expected.origin_shape
    actual_metadata = actual.get_metadata()
    for name, expected_value in expected.get_metadata().items():
        if name == "quantizer":
            continue
        if primary_only and name in (
            "data_transpose",
            "columnwise_data",
            "columnwise_scale_inv",
        ):
            continue
        actual_value = actual_metadata[name]
        if expected_value is None:
            assert actual_value is None
        elif isinstance(expected_value, torch.Tensor):
            assert actual_value is not None
            assert actual_value.dtype == expected_value.dtype
            assert actual_value.shape == expected_value.shape
            assert torch.equal(
                actual_value.detach().contiguous().view(torch.uint8).cpu(),
                expected_value.detach().contiguous().view(torch.uint8).cpu(),
            )
        else:
            assert actual_value == expected_value


def _clone_parameter_grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone all materialized parameter gradients."""

    return {
        name: param.grad.detach().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }


def _assert_parameter_grads_close(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare gradients for every parameter in two equivalent modules."""

    actual_grads = _clone_parameter_grads(actual)
    expected_grads = _clone_parameter_grads(expected)
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        torch.testing.assert_close(
            actual_grads[name],
            expected_grads[name],
            rtol=rtol,
            atol=atol,
            msg=lambda msg, name=name: f"Gradient mismatch for {name}: {msg}",
        )


def _assert_common_backward(
    module: torch.nn.Module,
    inp: torch.Tensor,
    *,
    skip_wgrad: bool,
    skip_dgrad: bool,
) -> None:
    if not skip_dgrad:
        assert inp.grad is not None, "Input gradient should be materialized."
    if not skip_wgrad:
        missing = [
            name
            for name, param in module.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        assert not missing, f"Parameter gradients should be materialized: {missing}"


def _run_common_module_sanity(
    module: torch.nn.Module,
    *,
    dtype: torch.dtype,
    config: ModelConfig,
    fp8_recipe: Optional[recipe.Recipe],
    skip_wgrad: bool,
    skip_dgrad: bool,
    microbatching: bool,
) -> None:
    if skip_dgrad and skip_wgrad:
        pytest.skip("No gradient computation; skipping to avoid PyTorch RuntimeError.")
    _skip_unsupported_fp8_config(config, fp8_recipe)

    inp = _randn_npu(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=dtype,
        requires_grad=not skip_dgrad,
    )

    if skip_wgrad:
        _disable_wgrads(module)

    with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
        if microbatching:
            _ = module(inp, is_first_microbatch=True)
            out = module(inp, is_first_microbatch=False)
        else:
            out = module(inp)

    out = _first_tensor(out)
    assert out.shape[:-1] == inp.shape[:-1]
    loss = out.sum()
    loss.backward()
    _sync()

    _assert_common_backward(
        module,
        inp,
        skip_wgrad=skip_wgrad,
        skip_dgrad=skip_dgrad,
    )


def _make_linear(config: ModelConfig, dtype: torch.dtype) -> Linear:
    return Linear(
        config.hidden_size,
        config.hidden_size,
        init_method=_scaled_init_method_normal(0.023, config.num_layers),
        params_dtype=dtype,
        device="npu",
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(("model", "fp8_recipe_cls"), _module_recipe_cases())
@pytest.mark.parametrize(("skip_wgrad", "skip_dgrad"), _GRAD_CASES)
@pytest.mark.parametrize("microbatching", [False, True])
def test_sanity_linear(
    dtype: torch.dtype,
    fp8_recipe_cls: Optional[type[recipe.Recipe]],
    model: str,
    skip_wgrad: bool,
    skip_dgrad: bool,
    microbatching: bool,
) -> None:
    """Adapted from NVIDIA test_sanity_linear."""

    config = _MODEL_CONFIGS[model]
    fp8_recipe = _make_recipe(fp8_recipe_cls)
    block = _make_linear(config, dtype)

    _run_common_module_sanity(
        block,
        dtype=dtype,
        config=config,
        fp8_recipe=fp8_recipe,
        skip_wgrad=skip_wgrad,
        skip_dgrad=skip_dgrad,
        microbatching=microbatching,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("parallel_mode", [None, "row"])
@pytest.mark.parametrize("return_bias", [False, True])
def test_sanity_linear_bias_path(
    dtype: torch.dtype,
    parallel_mode: Optional[str],
    return_bias: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check Linear bias routing: fused GEMM, unfused add, or dense fast path.

    When the GEMM runs without bias (returned bias or row-parallel unfused
    add) the non-fp8 dense fast path calls torch.matmul directly, so
    general_gemm may legitimately not be called at all.
    """

    from transformer_engine.pytorch.module import linear as linear_module

    config = _MODEL_CONFIGS["small"]
    block = Linear(
        config.hidden_size,
        config.hidden_size,
        bias=True,
        return_bias=return_bias,
        parallel_mode=parallel_mode,
        params_dtype=dtype,
        device="npu",
    )
    assert block.apply_bias == (not return_bias)
    assert block.gemm_bias_unfused_add == (parallel_mode == "row" and not return_bias)

    seen_gemm_biases = []
    seen_dense_matmuls = []
    original_general_gemm = linear_module.general_gemm
    original_matmul = torch.matmul

    def _record_general_gemm(*args, **kwargs):
        seen_gemm_biases.append(kwargs.get("bias"))
        return original_general_gemm(*args, **kwargs)

    def _record_dense_matmul(*args, **kwargs):
        seen_dense_matmuls.append(args)
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(linear_module, "general_gemm", _record_general_gemm)
    monkeypatch.setattr(torch, "matmul", _record_dense_matmul)

    inp = _randn_npu(
        (config.max_seqlen_q, config.hidden_size),
        dtype=dtype,
        requires_grad=True,
    )
    result = block(inp)
    dense_fast_path = bool(seen_dense_matmuls)
    if return_bias:
        out, returned_bias = result
        assert returned_bias.shape == block.bias.shape
        assert returned_bias.dtype == dtype
    else:
        out = result

    expected = torch.matmul(inp, block.weight.t())
    if not return_bias:
        expected = expected + block.bias
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    expect_gemm_bias = block.apply_bias and not block.gemm_bias_unfused_add
    if expect_gemm_bias:
        assert seen_gemm_biases, "Fused-bias forward must route through general_gemm."
        assert seen_gemm_biases[0] is not None, "GEMM should receive the fused bias."
    elif seen_gemm_biases:
        assert seen_gemm_biases[0] is None, "GEMM must stay bias-less when unfused."
    else:
        assert dense_fast_path, "Bias-less GEMM should take the dense fast path."

    out.sum().backward()
    _sync()
    if return_bias:
        assert block.bias.grad is None
    else:
        assert block.bias.grad is not None


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("return_bias", [False, True])
def test_sanity_layernorm_mlp_bias_semantics(
    dtype: torch.dtype,
    return_bias: bool,
) -> None:
    """Returned FC2 bias must be numerically equivalent to applying it internally."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        bias=True,
        return_bias=return_bias,
        params_dtype=dtype,
        device="npu",
    )
    reference = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        bias=True,
        return_bias=not return_bias,
        params_dtype=dtype,
        device="npu",
    )
    reference.load_state_dict(block.state_dict())

    inp = _randn_npu(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=dtype,
        requires_grad=True,
    )
    reference_inp = inp.detach().clone().requires_grad_(True)
    result = block(inp)
    reference_result = reference(reference_inp)
    if return_bias:
        out, returned_bias = result
        torch.testing.assert_close(returned_bias, block.fc2_bias, rtol=0, atol=0)
        reference_out = reference_result
        actual_with_bias = out + returned_bias
    else:
        out = result
        reference_out, reference_bias = reference_result
        actual_with_bias = out
        reference_out = reference_out + reference_bias

    torch.testing.assert_close(actual_with_bias, reference_out, rtol=1e-2, atol=1e-2)

    grad = _randn_npu(actual_with_bias.shape, dtype=dtype)
    actual_with_bias.backward(grad)
    reference_out.backward(grad)
    _sync()
    torch.testing.assert_close(inp.grad, reference_inp.grad, rtol=1e-2, atol=1e-2)
    _assert_parameter_grads_close(block, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "fp8_recipe_cls",
    [
        pytest.param(None, id="no_fp8"),
        pytest.param(
            recipe.DelayedScaling,
            marks=pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8),
            id="fp8",
        ),
    ],
)
def test_sanity_layernorm_mlp_return_layernorm_output_grad(
    fp8_recipe_cls: Optional[type[recipe.Recipe]],
) -> None:
    """LayerNorm auxiliary output must contribute to gradients in every compute mode."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    fp8_recipe = _make_recipe(fp8_recipe_cls)
    _skip_unsupported_fp8_config(config, fp8_recipe)
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        return_layernorm_output=True,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    inp = _randn_npu(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    reference_inp = inp.detach().float().cpu().requires_grad_(True)
    reference_weight = block.layer_norm_weight.detach().float().cpu().requires_grad_(True)
    reference_bias = block.layer_norm_bias.detach().float().cpu().requires_grad_(True)

    with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
        _, ln_out = block(inp)
        coefficients = torch.linspace(
            0.5,
            1.5,
            config.hidden_size,
            dtype=ln_out.dtype,
            device=ln_out.device,
        )
        (ln_out * coefficients).sum().backward()

    reference_coefficients = coefficients.detach().float().cpu()
    reference_out = torch.nn.functional.layer_norm(
        reference_inp,
        [config.hidden_size],
        reference_weight,
        reference_bias,
        block.eps,
    )
    (reference_out * reference_coefficients).sum().backward()

    _sync()
    torch.testing.assert_close(inp.grad.float().cpu(), reference_inp.grad, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(
        block.layer_norm_weight.grad.float().cpu(),
        reference_weight.grad,
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        block.layer_norm_bias.grad.float().cpu(),
        reference_bias.grad,
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.parametrize("leading_shape", [(32,), (2, 3, 4)])
def test_sanity_layernorm_mlp_accepts_arbitrary_leading_dims(
    leading_shape: tuple[int, ...],
) -> None:
    """Match NVTE behavior for inputs with arbitrary leading dimensions."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    reference = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    reference.load_state_dict(block.state_dict())
    inp = _randn_npu(
        (*leading_shape, config.hidden_size),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_inp = inp.detach().clone().reshape(-1, config.hidden_size).requires_grad_(True)
    out = block(inp)
    reference_out = reference(reference_inp).reshape_as(out)
    torch.testing.assert_close(out, reference_out, rtol=1e-2, atol=1e-2)

    grad = _randn_npu(out.shape, dtype=out.dtype)
    out.backward(grad)
    reference_out.backward(grad)
    _sync()
    torch.testing.assert_close(
        inp.grad,
        reference_inp.grad.reshape_as(inp),
        rtol=1e-2,
        atol=1e-2,
    )
    _assert_parameter_grads_close(block, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8)
def test_sanity_layernorm_mlp_fp8_norm_grads_without_input_grad() -> None:
    """FP8 dgrad data is still required when only normalization parameters need gradients."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    fp8_recipe = _make_recipe(recipe.DelayedScaling)
    _skip_unsupported_fp8_config(config, fp8_recipe)
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    reference = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    reference.load_state_dict(block.state_dict())
    inp = _randn_npu(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=torch.bfloat16,
        requires_grad=False,
    )
    reference_inp = inp.detach().clone().requires_grad_(True)
    grad = _randn_npu(inp.shape, dtype=torch.bfloat16)

    with autocast(enabled=True, recipe=fp8_recipe):
        out = block(inp)
        reference_out = reference(reference_inp)
        out.backward(grad)
        reference_out.backward(grad)

    _sync()
    torch.testing.assert_close(out, reference_out, rtol=0.125, atol=0.0675)
    _assert_parameter_grads_close(block, reference, rtol=0.125, atol=0.0675)


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_sanity_layernorm_mlp_layer_norm_weight_init(zero_centered_gamma: bool) -> None:
    """LayerNorm weight initialization must produce an effective gamma of one."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        zero_centered_gamma=zero_centered_gamma,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    expected = (
        torch.zeros_like(block.layer_norm_weight)
        if zero_centered_gamma
        else torch.ones_like(block.layer_norm_weight)
    )
    torch.testing.assert_close(block.layer_norm_weight, expected, rtol=0, atol=0)


@pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8)
def test_sanity_layernorm_mlp_fp8_fc2_bias_grad_uses_high_precision_grad() -> None:
    """FP8 GEMM quantization must not be applied to the FC2 bias reduction."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    fp8_recipe = _make_recipe(recipe.DelayedScaling)
    _skip_unsupported_fp8_config(config, fp8_recipe)
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    inp = _randn_npu(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    with autocast(enabled=True, recipe=fp8_recipe):
        out = block(inp)
        grad = torch.linspace(
            0.25,
            1.75,
            config.hidden_size,
            dtype=out.dtype,
            device=out.device,
        )
        (out * grad).sum().backward()

    _sync()
    expected = grad * (config.max_seqlen_q * config.batch_size)
    torch.testing.assert_close(block.fc2_bias.grad, expected, rtol=0, atol=0)


def test_sanity_clamped_swiglu_matches_nvte_semantics() -> None:
    """Validate NVTE limit, alpha, and linear-offset semantics on NPU."""

    from transformer_engine.pytorch.ops.basic import ClampedSwiGLU
    from transformer_engine.pytorch.ops.basic.npu_activation import (
        clamped_swiglu_bwd,
        clamped_swiglu_fwd,
    )

    params = {"limit": 2.0, "alpha": 1.5, "glu_linear_offset": 0.75}
    x = _randn_npu((8, 16), dtype=torch.float32)
    dy = _randn_npu((8, 8), dtype=torch.float32)
    glu, linear = x.chunk(2, dim=-1)
    clamped_glu = torch.clamp(glu, max=params["limit"])
    clamped_linear = (
        torch.clamp(linear, min=-params["limit"], max=params["limit"]) + params["glu_linear_offset"]
    )
    sigmoid = torch.sigmoid(params["alpha"] * clamped_glu)
    expected_out = clamped_glu * sigmoid * clamped_linear

    dactivated_glu = sigmoid + params["alpha"] * clamped_glu * sigmoid * (1.0 - sigmoid)
    dactivated_glu = dactivated_glu * (glu <= params["limit"]).to(dactivated_glu.dtype)
    expected_dglu = dy * clamped_linear * dactivated_glu
    expected_dlinear = dy * clamped_glu * sigmoid
    linear_in_range = (linear >= -params["limit"]) & (linear <= params["limit"])
    expected_dlinear = expected_dlinear * linear_in_range.to(expected_dlinear.dtype)
    expected_dx = torch.cat([expected_dglu, expected_dlinear], dim=-1)

    torch.testing.assert_close(clamped_swiglu_fwd(x, **params), expected_out)
    torch.testing.assert_close(clamped_swiglu_bwd(x, dy, **params), expected_dx)

    op_input = x.detach().clone().requires_grad_(True)
    op_out = ClampedSwiGLU(**params)(op_input)
    op_out.backward(dy)
    torch.testing.assert_close(op_out, expected_out)
    torch.testing.assert_close(op_input.grad, expected_dx)


def test_sanity_layernorm_mlp_init_method_semantics() -> None:
    """A custom FC1 initializer must not implicitly override the FC2 initializer."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        init_method=lambda tensor: torch.nn.init.constant_(tensor, 3.0),
        output_layer_init_method=lambda tensor: torch.nn.init.constant_(tensor, 5.0),
        params_dtype=torch.float32,
        device="npu",
    )
    torch.testing.assert_close(block.fc1_weight, torch.full_like(block.fc1_weight, 3.0))
    torch.testing.assert_close(block.fc2_weight, torch.full_like(block.fc2_weight, 5.0))


@pytest.mark.parametrize(
    "fp8_recipe_cls",
    [
        pytest.param(None, id="no_fp8"),
        pytest.param(
            recipe.DelayedScaling,
            marks=pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8),
            id="delayed_scaling",
        ),
        pytest.param(
            recipe.MXFP8BlockScaling,
            marks=pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8),
            id="mxfp8",
        ),
        pytest.param(
            recipe.W4A8BlockScaling,
            marks=pytest.mark.skipif(not _w4a8_available, reason=_reason_for_no_w4a8),
            id="w4a8",
        ),
    ],
)
def test_sanity_layernorm_mlp_internal_checkpoint(
    fp8_recipe_cls: Optional[type[recipe.Recipe]],
) -> None:
    """Internal activation checkpointing must preserve tensors required by backward."""

    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    config = _MODEL_CONFIGS["small"]
    fp8_recipe = _make_recipe(fp8_recipe_cls)
    _skip_unsupported_fp8_config(config, fp8_recipe)
    block = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        checkpoint=True,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    reference = LayerNormMLP(
        config.hidden_size,
        4 * config.hidden_size,
        checkpoint=False,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    reference.load_state_dict(block.state_dict())
    inp = _randn_npu(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_inp = inp.detach().clone().requires_grad_(True)
    grad = _randn_npu(inp.shape, dtype=torch.bfloat16)

    with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
        reference_out = reference(reference_inp)
        reference_out.backward(grad)
    reference_out = reference_out.detach()
    reference_input_grad = reference_inp.grad.detach().clone()
    reference_grads = _clone_parameter_grads(reference)

    with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
        out = block(inp)
        out.backward(grad)

    _sync()
    if fp8_recipe is None:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 0.125, 0.0675
    torch.testing.assert_close(out, reference_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(inp.grad, reference_input_grad, rtol=rtol, atol=atol)
    actual_grads = _clone_parameter_grads(block)
    assert actual_grads.keys() == reference_grads.keys()
    for name, actual_grad in actual_grads.items():
        torch.testing.assert_close(
            actual_grad,
            reference_grads[name],
            rtol=rtol,
            atol=atol,
            msg=lambda msg, name=name: f"Gradient mismatch for {name}: {msg}",
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [32])
@pytest.mark.parametrize(
    "fp8_recipe_cls",
    [
        pytest.param(None, id="no_fp8"),
        pytest.param(
            recipe.MXFP8BlockScaling,
            marks=pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8),
            id="mxfp8",
        ),
    ],
)
@pytest.mark.parametrize("fp8_model_params", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
def test_sanity_linear_with_fp8_model_params(
    dtype: torch.dtype,
    num_tokens: int,
    fp8_recipe_cls: Optional[type[recipe.Recipe]],
    fp8_model_params: bool,
    use_bias: bool,
) -> None:
    """Exercise Linear with optional MXFP8 activation and model-parameter quantization."""

    config = _MODEL_CONFIGS["small"]
    fp8_recipe = _make_recipe(fp8_recipe_cls)
    _skip_unsupported_fp8_config(config, fp8_recipe)
    ffn_hidden_size = 4 * config.hidden_size

    with quantized_model_init(
        enabled=fp8_recipe is not None and fp8_model_params,
        recipe=fp8_recipe,
    ):
        te_linear = Linear(
            config.hidden_size,
            ffn_hidden_size,
            bias=use_bias,
            params_dtype=dtype,
            device="npu",
        )

    inp = _randn_npu(
        (num_tokens, config.hidden_size),
        dtype=dtype,
        requires_grad=True,
    )
    with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
        out = te_linear(inp)
    loss = out.sum()
    loss.backward()
    _sync()

    assert out.shape == (num_tokens, ffn_hidden_size)


@pytest.mark.parametrize(
    "fp8_recipe_cls,weight_cls",
    _fp8_primary_weight_cases(),
)
def test_fp8_primary_weight_checkpoint_roundtrip(fp8_recipe_cls, weight_cls) -> None:
    """Preserve FP8 primary-weight payloads across a model checkpoint."""
    fp8_recipe = fp8_recipe_cls()
    with quantized_model_init(enabled=True, recipe=fp8_recipe):
        source = Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="npu")

    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    state_dict = torch.load(buffer, weights_only=False)

    assert "_extra_state" in state_dict
    assert state_dict["weight"].__class__ is weight_cls
    with quantized_model_init(enabled=True, recipe=fp8_recipe_cls()):
        restored = Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="npu")
    restored.load_state_dict(state_dict)

    assert restored.weight.__class__ is weight_cls
    _assert_quantized_storage_equal(restored.weight, source.weight)
    inp = _randn_npu((16, 128), dtype=torch.bfloat16)
    with torch.no_grad(), autocast(enabled=True, recipe=fp8_recipe):
        expected_output = source(inp).clone()
    with torch.no_grad(), autocast(enabled=True, recipe=fp8_recipe_cls()):
        actual_output = restored(inp)
    _sync()
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fp8_recipe_cls,weight_cls",
    _fp8_primary_weight_cases(),
)
def test_fp8_primary_weight_dcp_checkpoint(fp8_recipe_cls, weight_cls, tmp_path) -> None:
    """Preserve an FP8 primary weight through torch.distributed.checkpoint."""
    import torch.distributed.checkpoint as dcp

    fp8_recipe = fp8_recipe_cls()
    with quantized_model_init(enabled=True, recipe=fp8_recipe):
        source = Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="npu")
    assert source.weight.__class__ is weight_cls

    dcp.save({"weight": source.weight.detach()}, checkpoint_id=str(tmp_path))

    with quantized_model_init(enabled=True, recipe=fp8_recipe_cls()):
        restored = Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="npu")
    state_dict = {"weight": restored.weight.detach()}
    dcp.load(state_dict, checkpoint_id=str(tmp_path))
    restored.load_state_dict(state_dict, strict=False)

    assert state_dict["weight"].__class__ is weight_cls
    assert restored.weight.__class__ is weight_cls
    _assert_quantized_storage_equal(restored.weight, source.weight, primary_only=True)
    torch.testing.assert_close(
        restored.weight.dequantize(), source.weight.dequantize(), rtol=0, atol=0
    )
    inp = _randn_npu((16, 128), dtype=torch.bfloat16)
    with torch.no_grad(), autocast(enabled=True, recipe=fp8_recipe):
        expected_output = source(inp).clone()
    with torch.no_grad(), autocast(enabled=True, recipe=fp8_recipe_cls()):
        actual_output = restored(inp)
    _sync()
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_sanity_linear_preserves_all_leading_dimensions(dtype: torch.dtype) -> None:
    """Linear restores arbitrary leading dimensions in forward and backward."""

    input_shape = (2, 3, 4, 16)
    output_shape = (*input_shape[:-1], 12)
    layer = Linear(
        input_shape[-1],
        output_shape[-1],
        bias=True,
        params_dtype=dtype,
        device="npu",
    )
    inp = _randn_npu(input_shape, dtype=dtype, requires_grad=True)

    out = layer(inp)
    assert out.shape == output_shape

    grad_output = _randn_npu(output_shape, dtype=dtype)
    out.backward(grad_output)
    _sync()

    assert inp.grad is not None
    assert inp.grad.shape == input_shape
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == layer.bias.shape


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("num_tokens_per_nonempty_group", [0, 16])
@pytest.mark.parametrize("empty_split", ["first", "last", "middle"])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("fp8_recipe_cls", [pytest.param(None, id="no_fp8")])
def test_sanity_grouped_linear(
    dtype: torch.dtype,
    num_tokens_per_nonempty_group: int,
    empty_split: str,
    use_bias: bool,
    fp8_recipe_cls: Optional[type[recipe.Recipe]],
) -> None:
    """Adapted from NVIDIA test_sanity_grouped_linear."""

    config = _MODEL_CONFIGS["small"]
    num_gemms = 4
    ffn_hidden_size = 4 * config.hidden_size
    fp8_recipe = _make_recipe(fp8_recipe_cls)

    te_grouped_linear = GroupedLinear(
        num_gemms,
        config.hidden_size,
        ffn_hidden_size,
        bias=use_bias,
        params_dtype=dtype,
        device="npu",
    )

    m_splits = [num_tokens_per_nonempty_group] * num_gemms
    if empty_split == "first":
        m_splits[0] = 0
    elif empty_split == "last":
        m_splits[-1] = 0
    else:
        m_splits[num_gemms // 2] = 0
    num_tokens = sum(m_splits)

    inp = _randn_npu(
        (num_tokens, config.hidden_size),
        dtype=dtype,
        requires_grad=True,
    )

    with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
        out = te_grouped_linear(inp, m_splits)
    loss = out.sum()
    loss.backward()
    _sync()

    assert out.shape == (num_tokens, ffn_hidden_size)


@pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8)
def test_grouped_linear_tensorwise_primary_weight_init() -> None:
    """Allocate enough quantizers for every ungrouped FP8 primary weight."""
    num_gemms = 4
    with quantized_model_init(enabled=True, recipe=recipe.Float8CurrentScaling()):
        module = GroupedLinear(
            num_gemms,
            32,
            64,
            bias=False,
            params_dtype=torch.bfloat16,
            device="npu",
        )

    assert len(module.quantizers["scaling_fwd"]) == 3 * num_gemms
    for i in range(num_gemms):
        assert isinstance(getattr(module, f"weight{i}"), Float8Tensor)


def test_delayed_fused_grouped_wgrad_first_microbatch_overwrites_then_accumulates() -> None:
    """Guard first-microbatch zeroing and later accumulation of packed ``main_grad``."""
    num_gemms = 2
    in_features = 16
    out_features = 16
    m_splits = [3, 5]
    num_tokens = sum(m_splits)
    layer = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=False,
        delay_wgrad_compute=True,
        fuse_wgrad_accumulation=True,
        single_grouped_weight=True,
        params_dtype=torch.bfloat16,
        device="npu",
    )
    assert layer._use_performance_grouped_linear()

    stale_main_grad = torch.full_like(layer.weight, 7.0, dtype=torch.float32)
    layer.weight.main_grad = stale_main_grad

    def run_microbatch(is_first_microbatch: bool) -> torch.Tensor:
        inp = _randn_npu((num_tokens, in_features), dtype=torch.bfloat16)
        out = layer(inp, m_splits, is_first_microbatch=is_first_microbatch)
        grad_out = _randn_npu(out.shape, dtype=torch.bfloat16)
        expected = torch.stack(
            [
                dy.float().t().matmul(x.float())
                for x, dy in zip(
                    torch.split(inp, m_splits),
                    torch.split(grad_out, m_splits),
                )
            ]
        )
        out.backward(grad_out)
        assert layer.need_backward_dw()
        return expected

    expected_first = run_microbatch(is_first_microbatch=True)
    torch.testing.assert_close(
        layer.weight.main_grad,
        torch.full_like(layer.weight.main_grad, 7.0),
        rtol=0,
        atol=0,
    )
    layer.backward_dw()
    _sync()
    torch.testing.assert_close(
        layer.weight.main_grad,
        expected_first,
        rtol=5e-2,
        atol=2.5e-1,
    )

    expected_second = run_microbatch(is_first_microbatch=False)
    torch.testing.assert_close(
        layer.weight.main_grad,
        expected_first,
        rtol=5e-2,
        atol=2.5e-1,
    )
    layer.backward_dw()
    _sync()
    torch.testing.assert_close(
        layer.weight.main_grad,
        expected_first + expected_second,
        rtol=5e-2,
        atol=2.5e-1,
    )


def test_model_multiple_cast() -> None:
    """Adapted from NVIDIA test_model_multiple_cast."""

    inp = torch.zeros((16, 16), device=_npu_device())
    module = Linear(16, 32, device="npu")

    out = module(inp)
    assert out.dtype == torch.float32

    module.half()
    inp = inp.half()
    out = module(inp)
    assert out.dtype == torch.float16


@pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8)
def test_quantized_model_init_high_precision_init_val() -> None:
    """Adapted from NVIDIA test_quantized_model_init_high_precision_init_val."""

    with quantized_model_init(preserve_high_precision_init_val=True):
        module = Linear(128, 128, params_dtype=torch.bfloat16, device="npu")

    weight = module.weight
    assert isinstance(weight, QuantizedTensor)
    assert hasattr(weight, "_high_precision_init_val")
    assert hasattr(weight, "get_high_precision_init_val")
    assert hasattr(weight, "clear_high_precision_init_val")

    high_precision = weight.get_high_precision_init_val()
    assert high_precision.device.type == "cpu"

    new_weight = weight._get_quantizer().make_empty(
        shape=weight.shape,
        dtype=weight.dtype,
        device=weight.device,
    )
    weight._get_quantizer().update_quantized(high_precision.to(weight.device), new_weight)

    torch.testing.assert_close(
        new_weight.dequantize(dtype=weight.dtype),
        weight.dequantize(dtype=weight.dtype),
        rtol=0,
        atol=0,
    )


_NORMALIZATIONS = ["LayerNorm", "RMSNorm"]
_ACTIVATIONS = ["gelu", "swiglu"]


def _make_layernorm_linear(
    config: ModelConfig,
    dtype: torch.dtype,
    *,
    zero_centered_gamma: bool = False,
    normalization: str = "LayerNorm",
) -> "LayerNormLinear":  # noqa: F821
    from transformer_engine.pytorch.module.layernorm_linear import LayerNormLinear

    sigma = 0.023
    init_method = init_method_normal(sigma)
    return LayerNormLinear(
        config.hidden_size,
        config.hidden_size * 3,
        init_method=init_method,
        zero_centered_gamma=zero_centered_gamma,
        normalization=normalization,
        params_dtype=dtype,
        device="npu",
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(("model", "fp8_recipe_cls"), _module_recipe_cases())
@pytest.mark.parametrize(("skip_wgrad", "skip_dgrad"), _GRAD_CASES)
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("normalization", _NORMALIZATIONS)
@pytest.mark.parametrize("microbatching", [False, True])
def test_sanity_layernorm_linear(
    dtype: torch.dtype,
    fp8_recipe_cls: Optional[type[recipe.Recipe]],
    model: str,
    skip_wgrad: bool,
    skip_dgrad: bool,
    zero_centered_gamma: bool,
    normalization: str,
    microbatching: bool,
) -> None:
    """Adapted from NVIDIA test_sanity_layernorm_linear.

    Exercises LayerNormLinear forward/backward with:
    - LayerNorm and RMSNorm normalizations
    - Optional zero_centered_gamma
    - BF16 and FP8 (MXFP8) compute paths
    - Microbatching (is_first_microbatch)
    - Selective gradient skip (wgrad / dgrad)
    """
    config = _MODEL_CONFIGS[model]
    fp8_recipe = _make_recipe(fp8_recipe_cls)
    _skip_unsupported_fp8_config(config, fp8_recipe)

    block = _make_layernorm_linear(
        config,
        dtype,
        zero_centered_gamma=zero_centered_gamma,
        normalization=normalization,
    )

    _run_common_module_sanity(
        block,
        dtype=dtype,
        config=config,
        fp8_recipe=fp8_recipe,
        skip_wgrad=skip_wgrad,
        skip_dgrad=skip_dgrad,
        microbatching=microbatching,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 16])
def test_sanity_single_grouped_linear(dtype: torch.dtype, use_bias: bool, num_tokens: int) -> None:
    """GroupedLinear must retain the group dimension when num_gemms is one."""
    config = _MODEL_CONFIGS["small"]
    ffn_hidden_size = 4 * config.hidden_size
    layer = GroupedLinear(
        1,
        config.hidden_size,
        ffn_hidden_size,
        bias=use_bias,
        params_dtype=dtype,
        device="npu",
    )
    inp = _randn_npu(
        (num_tokens, config.hidden_size),
        dtype=dtype,
        requires_grad=True,
    )

    out = layer(inp, [num_tokens])
    out.sum().backward()
    _sync()

    assert out.shape == (num_tokens, ffn_hidden_size)
    assert inp.grad is not None
    assert layer.weight0.grad is not None
