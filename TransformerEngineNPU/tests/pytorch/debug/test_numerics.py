# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Precision probes for TE debug quantization controls."""

from __future__ import annotations

import contextlib
from pathlib import Path
import textwrap

import pytest
import torch

from utils import a5_npu_available


pytest.importorskip("nvdlfw_inspect.api")


requires_a5_quantization = pytest.mark.skipif(
    not a5_npu_available(),
    reason="Atlas A5 NPU is required for FP8 quantization tests",
)


def _reset_debug_api() -> None:
    import nvdlfw_inspect.api as debug_api

    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    if debug_api.DEBUG_MANAGER is not None:
        debug_api.end_debug()
    TEDebugState._reset()


@contextlib.contextmanager
def _debug_session(
    tmp_path: Path,
    *,
    name: str = "debug_precision",
    layer_name_regex_pattern: str = ".*",
    disabled_gemms: list[str] | None = None,
):
    import nvdlfw_inspect.api as debug_api

    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    _reset_debug_api()
    feature_dirs = Path(__file__).parents[3] / "transformer_engine" / "debug" / "features"
    if disabled_gemms is None:
        config_file = ""
    else:
        config_file = tmp_path / f"{name}.yaml"
        config_file.write_text(
            textwrap.dedent(
                f"""
                {name}:
                  enabled: True
                  layers:
                    layer_name_regex_pattern: {layer_name_regex_pattern}
                  transformer_engine:
                    DisableQuantizationGEMM:
                      enabled: True
                      gemms: [{", ".join(disabled_gemms)}]
                """
            ),
            encoding="utf-8",
        )

    TEDebugState._reset()
    debug_api.initialize(
        config_file=config_file,
        feature_dirs=[str(feature_dirs)],
        log_dir=tmp_path,
    )
    try:
        yield debug_api
    finally:
        _reset_debug_api()


def _make_input() -> torch.Tensor:
    values = torch.linspace(-3.0, 3.0, steps=64 * 64, dtype=torch.float32)
    return values.reshape(64, 64).to(dtype=torch.bfloat16, device="npu")


def _make_mlp_input() -> torch.Tensor:
    values = torch.linspace(-3.0, 3.0, steps=8 * 8 * 64, dtype=torch.float32)
    return values.reshape(8, 8, 64).to(dtype=torch.bfloat16, device="npu")


def _make_linear():
    from transformer_engine.pytorch import Linear

    module = Linear(64, 64, bias=False, params_dtype=torch.bfloat16, device="npu")
    weight = torch.linspace(-2.0, 2.0, steps=64 * 64, dtype=torch.float32)
    weight = weight.reshape(64, 64).to(dtype=torch.bfloat16, device="npu")
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


def _make_layernorm_linear():
    from transformer_engine.pytorch import LayerNormLinear

    module = LayerNormLinear(
        64,
        64,
        bias=False,
        params_dtype=torch.bfloat16,
        device="npu",
        name="layernorm_linear_debug_precision",
    )
    weight = torch.linspace(-2.0, 2.0, steps=64 * 64, dtype=torch.float32)
    weight = weight.reshape(64, 64).to(dtype=torch.bfloat16, device="npu")
    with torch.no_grad():
        module.weight.copy_(weight)
        module.layer_norm_weight.fill_(1.0)
        module.layer_norm_bias.zero_()
    return module


def _make_layernorm_mlp():
    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    module = LayerNormMLP(
        64,
        64,
        bias=False,
        params_dtype=torch.bfloat16,
        device="npu",
        activation="relu",
        name="layernorm_mlp_debug_precision",
    )
    fc1_weight = torch.linspace(-2.0, 2.0, steps=64 * 64, dtype=torch.float32)
    fc1_weight = fc1_weight.reshape(64, 64).to(dtype=torch.bfloat16, device="npu")
    fc2_weight = torch.linspace(2.0, -2.0, steps=64 * 64, dtype=torch.float32)
    fc2_weight = fc2_weight.reshape(64, 64).to(dtype=torch.bfloat16, device="npu")
    with torch.no_grad():
        module.fc1_weight.copy_(fc1_weight)
        module.fc2_weight.copy_(fc2_weight)
        module.layer_norm_weight.fill_(1.0)
        module.layer_norm_bias.zero_()
    return module


def _make_grouped_linear():
    from transformer_engine.pytorch import GroupedLinear

    module = GroupedLinear(
        2,
        64,
        64,
        bias=False,
        params_dtype=torch.bfloat16,
        device="npu",
        single_grouped_weight=True,
        name="grouped_linear_debug_precision",
    )
    weight = torch.linspace(-2.0, 2.0, steps=2 * 64 * 64, dtype=torch.float32)
    weight = weight.reshape(2, 64, 64).to(dtype=torch.bfloat16, device="npu")
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


def _run_linear(module, inp: torch.Tensor, *, fp8_enabled: bool) -> torch.Tensor:
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import autocast

    with torch.no_grad(), autocast(enabled=fp8_enabled, recipe=recipe.DelayedScaling()):
        out = module(inp)
    torch.npu.synchronize()
    return out.detach()


def _run_linear_with_grads(module, inp: torch.Tensor, *, fp8_enabled: bool):
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import autocast

    module.zero_grad(set_to_none=True)
    inp = inp.detach().clone().requires_grad_(True)
    with autocast(enabled=fp8_enabled, recipe=recipe.DelayedScaling()):
        out = module(inp)
        out.float().sum().backward()
    torch.npu.synchronize()
    return out.detach(), inp.grad.detach(), module.weight.grad.detach()


def _run_layernorm_linear(module, inp: torch.Tensor, *, fp8_enabled: bool):
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import autocast

    module.zero_grad(set_to_none=True)
    inp = inp.detach().clone().requires_grad_(True)
    with autocast(enabled=fp8_enabled, recipe=recipe.DelayedScaling()):
        out = module(inp)
        out.float().sum().backward()
    torch.npu.synchronize()
    return (
        out.detach(),
        inp.grad.detach(),
        module.weight.grad.detach(),
    )


def _run_layernorm_mlp(module, inp: torch.Tensor, *, fp8_enabled: bool):
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import autocast

    module.zero_grad(set_to_none=True)
    inp = inp.detach().clone().requires_grad_(True)
    with autocast(enabled=fp8_enabled, recipe=recipe.DelayedScaling()):
        out = module(inp)
        out.float().sum().backward()
    torch.npu.synchronize()
    return (
        out.detach(),
        inp.grad.detach(),
        module.fc1_weight.grad.detach(),
        module.fc2_weight.grad.detach(),
    )


def _run_grouped_linear(module, inp: torch.Tensor, *, fp8_enabled: bool):
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import autocast

    module.zero_grad(set_to_none=True)
    inp = inp.detach().clone().requires_grad_(True)
    with autocast(enabled=fp8_enabled, recipe=recipe.DelayedScaling()):
        out = module(inp, [32, 32])
        out.float().sum().backward()
    torch.npu.synchronize()
    return out.detach(), inp.grad.detach(), module.weight.grad.detach()


def test_unmatched_debug_feature_preserves_non_fp8_forward_backward(tmp_path) -> None:
    import nvdlfw_inspect.api as debug_api

    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    if debug_api.DEBUG_MANAGER is not None:
        debug_api.end_debug()
    TEDebugState._reset()

    inp = _make_input()
    mlp_inp = _make_mlp_input()
    baseline_tensors = (
        _run_linear_with_grads(_make_linear(), inp, fp8_enabled=False),
        _run_layernorm_linear(_make_layernorm_linear(), inp, fp8_enabled=False),
        _run_layernorm_mlp(_make_layernorm_mlp(), mlp_inp, fp8_enabled=False),
    )

    config_file = tmp_path / "unmatched_layer.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            unmatched_layer:
              enabled: True
              layers:
                layer_name_regex_pattern: ^does_not_match_any_layer$
              transformer_engine:
                DisableQuantizationGEMM:
                  enabled: True
                  gemms: [fprop, dgrad, wgrad]
            """
        ),
        encoding="utf-8",
    )
    feature_dirs = Path(__file__).parents[3] / "transformer_engine" / "debug" / "features"

    TEDebugState._reset()
    debug_api.initialize(
        config_file=config_file,
        feature_dirs=[str(feature_dirs)],
        log_dir=tmp_path,
    )
    try:
        debug_tensors = (
            _run_linear_with_grads(_make_linear(), inp, fp8_enabled=False),
            _run_layernorm_linear(_make_layernorm_linear(), inp, fp8_enabled=False),
            _run_layernorm_mlp(_make_layernorm_mlp(), mlp_inp, fp8_enabled=False),
        )
        debug_api.step()
    finally:
        debug_api.end_debug()

    for baseline_module_tensors, debug_module_tensors in zip(baseline_tensors, debug_tensors):
        for baseline_tensor, debug_tensor in zip(baseline_module_tensors, debug_module_tensors):
            torch.testing.assert_close(debug_tensor, baseline_tensor)


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    return diff.max().item(), diff.mean().item()


def _emulate_gemm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    lhs_usage,
    rhs_usage,
    lhs_fp8_dtype: torch.dtype,
    rhs_fp8_dtype: torch.dtype,
    fp8_enabled: bool,
) -> torch.Tensor:
    from transformer_engine.pytorch.ops.gemm import general_gemm
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    if fp8_enabled:
        lhs_scale = torch.ones(1, dtype=torch.float32, device=lhs.device)
        rhs_scale = torch.ones(1, dtype=torch.float32, device=rhs.device)
        lhs = Float8Quantizer(lhs_scale, lhs_scale.clone(), lhs_fp8_dtype)(lhs)
        rhs = Float8Quantizer(rhs_scale, rhs_scale.clone(), rhs_fp8_dtype)(rhs)

    return general_gemm(
        lhs,
        rhs,
        usage_a=lhs_usage,
        usage_b=rhs_usage,
        out_dtype=torch.bfloat16,
    )


def _assert_debug_reduces_forward_error(
    fp8_out: torch.Tensor,
    no_quant_out: torch.Tensor,
    debug_disabled_out: torch.Tensor,
    *,
    prefix: str,
) -> None:
    fp8_max_abs, fp8_mean_abs = _error_metrics(fp8_out, no_quant_out)
    debug_max_abs, debug_mean_abs = _error_metrics(debug_disabled_out, no_quant_out)
    print(
        f"{prefix}: "
        f"fp8_enabled_max_abs={fp8_max_abs:.8f}, "
        f"fp8_enabled_mean_abs={fp8_mean_abs:.8f}, "
        f"debug_disable_fprop_max_abs={debug_max_abs:.8f}, "
        f"debug_disable_fprop_mean_abs={debug_mean_abs:.8f}"
    )

    assert debug_max_abs <= fp8_max_abs
    assert debug_mean_abs <= fp8_mean_abs


def _assert_debug_reduces_error(
    fp8_tensor: torch.Tensor,
    no_quant_tensor: torch.Tensor,
    debug_disabled_tensor: torch.Tensor,
    *,
    prefix: str,
    metric_name: str,
) -> None:
    fp8_max_abs, fp8_mean_abs = _error_metrics(fp8_tensor, no_quant_tensor)
    debug_max_abs, debug_mean_abs = _error_metrics(debug_disabled_tensor, no_quant_tensor)
    print(
        f"{prefix}_{metric_name}: "
        f"fp8_enabled_max_abs={fp8_max_abs:.8f}, "
        f"fp8_enabled_mean_abs={fp8_mean_abs:.8f}, "
        f"debug_disable_max_abs={debug_max_abs:.8f}, "
        f"debug_disable_mean_abs={debug_mean_abs:.8f}"
    )

    assert debug_max_abs <= fp8_max_abs
    assert debug_mean_abs <= fp8_mean_abs


@pytest.mark.parametrize("fprop_fp8", [True, False])
@pytest.mark.parametrize("dgrad_fp8", [True, False])
@pytest.mark.parametrize("wgrad_fp8", [True, False])
@requires_a5_quantization
def test_debug_disable_gemms_matches_emulated_linear(
    tmp_path,
    fprop_fp8: bool,
    dgrad_fp8: bool,
    wgrad_fp8: bool,
) -> None:
    from transformer_engine.pytorch.constants import TensorUsage
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_input()
    disabled_gemms = [
        gemm
        for gemm, enabled in (
            ("fprop", fprop_fp8),
            ("dgrad", dgrad_fp8),
            ("wgrad", wgrad_fp8),
        )
        if not enabled
    ]

    with _debug_session(
        tmp_path,
        name="disable_linear_gemms",
        disabled_gemms=disabled_gemms,
    ) as debug_api:
        module = _make_linear()
        weight = module.weight.detach().clone()
        output, input_grad, weight_grad = _run_linear_with_grads(
            module,
            inp,
            fp8_enabled=True,
        )
        debug_api.step()

    expected_output = _emulate_gemm(
        inp,
        weight,
        lhs_usage=TensorUsage.LHS,
        rhs_usage=TensorUsage.RHS_TRANS,
        lhs_fp8_dtype=torch.float8_e4m3fn,
        rhs_fp8_dtype=torch.float8_e4m3fn,
        fp8_enabled=fprop_fp8,
    )
    grad_output = torch.ones_like(expected_output)
    expected_input_grad = _emulate_gemm(
        grad_output,
        weight,
        lhs_usage=TensorUsage.LHS,
        rhs_usage=TensorUsage.RHS,
        lhs_fp8_dtype=torch.float8_e5m2,
        rhs_fp8_dtype=torch.float8_e4m3fn,
        fp8_enabled=dgrad_fp8,
    )
    expected_weight_grad = _emulate_gemm(
        grad_output,
        inp,
        lhs_usage=TensorUsage.LHS_TRANS,
        rhs_usage=TensorUsage.RHS,
        lhs_fp8_dtype=torch.float8_e5m2,
        rhs_fp8_dtype=torch.float8_e4m3fn,
        fp8_enabled=wgrad_fp8,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(input_grad, expected_input_grad)
    torch.testing.assert_close(weight_grad, expected_weight_grad)


@requires_a5_quantization
def test_debug_disable_fprop_with_yaml_reduces_linear_fp8_forward_error(tmp_path) -> None:
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_input()
    no_quant_linear = _make_linear()
    fp8_linear = _make_linear()
    no_quant_out = _run_linear(no_quant_linear, inp, fp8_enabled=False)
    fp8_out = _run_linear(fp8_linear, inp, fp8_enabled=True)

    with _debug_session(
        tmp_path, name="disable_linear_fprop", disabled_gemms=["fprop"]
    ) as debug_api:
        debug_disabled_linear = _make_linear()
        debug_disabled_out = _run_linear(debug_disabled_linear, inp, fp8_enabled=True)
        debug_api.step()

    _assert_debug_reduces_forward_error(
        fp8_out,
        no_quant_out,
        debug_disabled_out,
        prefix="linear_debug_precision_yaml",
    )


@requires_a5_quantization
def test_debug_disable_fprop_with_yaml_reduces_layernorm_mlp_fp8_error(tmp_path) -> None:
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_mlp_input()
    no_quant_layer = _make_layernorm_mlp()
    fp8_layer = _make_layernorm_mlp()
    no_quant_tensors = _run_layernorm_mlp(no_quant_layer, inp, fp8_enabled=False)
    fp8_tensors = _run_layernorm_mlp(fp8_layer, inp, fp8_enabled=True)

    with _debug_session(
        tmp_path,
        name="disable_layernorm_mlp_fprop",
        layer_name_regex_pattern="layernorm_mlp_debug_precision",
        disabled_gemms=["fprop"],
    ) as debug_api:
        debug_disabled_layer = _make_layernorm_mlp()
        debug_disabled_tensors = _run_layernorm_mlp(debug_disabled_layer, inp, fp8_enabled=True)
        debug_api.step()

    _assert_debug_reduces_error(
        fp8_tensors[0],
        no_quant_tensors[0],
        debug_disabled_tensors[0],
        prefix="layernorm_mlp_debug_precision_yaml_fprop",
        metric_name="output",
    )


@pytest.mark.parametrize(
    ("disabled_gemm", "target_idx", "metric_name"),
    [
        ("fprop", 0, "output"),
        ("dgrad", 1, "input_grad"),
        ("wgrad", 2, "weight_grad"),
    ],
)
@requires_a5_quantization
def test_debug_disable_gemm_reduces_performance_grouped_linear_fp8_error(
    tmp_path,
    disabled_gemm: str,
    target_idx: int,
    metric_name: str,
) -> None:
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_input()
    no_quant_tensors = _run_grouped_linear(_make_grouped_linear(), inp, fp8_enabled=False)
    fp8_tensors = _run_grouped_linear(_make_grouped_linear(), inp, fp8_enabled=True)

    with _debug_session(
        tmp_path,
        name=f"disable_grouped_linear_{disabled_gemm}",
        layer_name_regex_pattern="grouped_linear_debug_precision",
        disabled_gemms=[disabled_gemm],
    ) as debug_api:
        debug_tensors = _run_grouped_linear(_make_grouped_linear(), inp, fp8_enabled=True)
        debug_api.step()

    _assert_debug_reduces_error(
        fp8_tensors[target_idx],
        no_quant_tensors[target_idx],
        debug_tensors[target_idx],
        prefix=f"performance_grouped_linear_debug_precision_{disabled_gemm}",
        metric_name=metric_name,
    )


@pytest.mark.parametrize(
    ("disabled_gemm", "target_idx", "metric_name"),
    [
        ("fprop", 0, "output"),
        ("dgrad", 1, "input_grad"),
        ("wgrad", 2, "weight_grad"),
    ],
)
@requires_a5_quantization
def test_debug_disable_gemm_with_yaml_reduces_layernorm_linear_fp8_error(
    tmp_path,
    disabled_gemm: str,
    target_idx: int,
    metric_name: str,
) -> None:
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_input()
    no_quant_layer = _make_layernorm_linear()
    fp8_layer = _make_layernorm_linear()
    no_quant_tensors = _run_layernorm_linear(no_quant_layer, inp, fp8_enabled=False)
    fp8_tensors = _run_layernorm_linear(fp8_layer, inp, fp8_enabled=True)

    with _debug_session(
        tmp_path,
        name=f"disable_layernorm_linear_{disabled_gemm}",
        layer_name_regex_pattern="layernorm_linear_debug_precision",
        disabled_gemms=[disabled_gemm],
    ) as debug_api:
        debug_disabled_layer = _make_layernorm_linear()
        debug_disabled_tensors = _run_layernorm_linear(debug_disabled_layer, inp, fp8_enabled=True)
        debug_api.step()

    _assert_debug_reduces_error(
        fp8_tensors[target_idx],
        no_quant_tensors[target_idx],
        debug_disabled_tensors[target_idx],
        prefix=f"layernorm_linear_debug_precision_yaml_{disabled_gemm}",
        metric_name=metric_name,
    )


# ---------------------------------------------------------------------------
# Weight workspace caching tests (Step 5)
# ---------------------------------------------------------------------------


def _run_linear_microbatched(
    module, inp: torch.Tensor, *, fp8_enabled: bool, is_first_microbatch=None
) -> torch.Tensor:
    """Run Linear.forward with explicit is_first_microbatch for workspace caching."""
    from transformer_engine.common import recipe
    from transformer_engine.pytorch import autocast

    with torch.no_grad(), autocast(enabled=fp8_enabled, recipe=recipe.DelayedScaling()):
        out = module(inp, is_first_microbatch=is_first_microbatch)
    torch.npu.synchronize()
    return out.detach()


@requires_a5_quantization
def test_debug_weight_workspace_caching_produces_consistent_output(tmp_path) -> None:
    """Microbatch workspace cache should honour debug decisions across steps."""
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_input()

    # Baseline: no quantization (debug does not need to be installed)
    no_quant_linear = _make_linear()
    no_quant_out = _run_linear(no_quant_linear, inp, fp8_enabled=False)

    with _debug_session(
        tmp_path, name="disable_workspace_fprop", disabled_gemms=["fprop"]
    ) as debug_api:
        debug_linear = _make_linear()

        out_first = _run_linear_microbatched(
            debug_linear, inp, fp8_enabled=True, is_first_microbatch=True
        )

        out_second = _run_linear_microbatched(
            debug_linear, inp, fp8_enabled=True, is_first_microbatch=False
        )
        debug_api.step()

    # Both outputs should be close to BF16 baseline (debug disabled fprop)
    first_max, first_mean = _error_metrics(out_first, no_quant_out)
    second_max, second_mean = _error_metrics(out_second, no_quant_out)
    print(
        f"linear_workspace_caching: "
        f"first_max={first_max:.8f}, first_mean={first_mean:.8f}, "
        f"second_max={second_max:.8f}, second_mean={second_mean:.8f}"
    )

    # Workspace caching should not change the output
    consistency_max, consistency_mean = _error_metrics(out_first, out_second)
    print(
        f"linear_workspace_consistency: "
        f"max_diff={consistency_max:.8f}, mean_diff={consistency_mean:.8f}"
    )
    assert consistency_max <= 1e-3, (
        f"Cached workspace output diverged from fresh output: max_diff={consistency_max:.8f}"
    )


@requires_a5_quantization
def test_debug_weight_workspace_caching_matches_no_cache_path(tmp_path) -> None:
    """Forward with workspace caching should match forward without caching."""
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_input()

    with _debug_session(
        tmp_path, name="disable_workspace_cache_fprop", disabled_gemms=["fprop"]
    ) as debug_api:
        linear_no_cache = _make_linear()
        out_no_cache = _run_linear_microbatched(
            linear_no_cache, inp, fp8_enabled=True, is_first_microbatch=None
        )

        linear_cached = _make_linear()
        _run_linear_microbatched(linear_cached, inp, fp8_enabled=True, is_first_microbatch=True)
        out_cached = _run_linear_microbatched(
            linear_cached, inp, fp8_enabled=True, is_first_microbatch=False
        )
        debug_api.step()

    max_diff, mean_diff = _error_metrics(out_no_cache, out_cached)
    print(f"linear_workspace_cache_vs_no_cache: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
    assert max_diff <= 1e-3, f"Cached path differs from non-cached path: max_diff={max_diff:.8f}"


@requires_a5_quantization
def test_debug_weight_workspace_caching_without_any_disable_is_noop(tmp_path) -> None:
    """When no GEMM is disabled, workspace caching should not change behaviour."""
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager, is_fp8_available

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)

    torch.manual_seed(1234)
    torch.npu.manual_seed(1234)
    FP8GlobalStateManager.reset()
    _reset_debug_api()

    inp = _make_input()

    with _debug_session(tmp_path) as debug_api:
        linear = _make_linear()
        out_first = _run_linear_microbatched(
            linear, inp, fp8_enabled=True, is_first_microbatch=True
        )
        out_second = _run_linear_microbatched(
            linear, inp, fp8_enabled=True, is_first_microbatch=False
        )
        debug_api.step()

    max_diff, mean_diff = _error_metrics(out_first, out_second)
    print(f"linear_workspace_noop_debug: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
    assert max_diff <= 1e-3, f"No-op debug caching diverged: max_diff={max_diff:.8f}"
