# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Sanity tests for TE debug features with real debug initialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch import autocast
from utils import a5_npu_available, npu_available


debug_api = pytest.importorskip("nvdlfw_inspect.api")


B, S, H, D = 4, 4, 16, 16

MODEL_KEYS = ["linear", "layernorm_linear", "layernorm_mlp"]

CONFIGS = {
    "": "",
    "disable_gemm": """
disable_gemm:
  enabled: True
  layers:
    layer_name_regex_pattern: .*
  transformer_engine:
    DisableQuantizationGEMM:
      enabled: True
      gemms: [fprop, dgrad, wgrad]
""",
    "disable_layer": """
disable_layer:
  enabled: True
  layers:
    layer_name_regex_pattern: .*
  transformer_engine:
    DisableQuantizationLayer:
      enabled: True
""",
    "log_fp8": """
log_fp8:
  enabled: True
  layers:
    layer_name_regex_pattern: .*
  transformer_engine:
    LogFp8TensorStats:
      enabled: True
      tensors: [activation, gradient, weight]
      stats: [underflows%]
      start_step: 0
      end_step: 1
""",
}

FP8_REQUIRED_CONFIGS = {"log_fp8"}


def _reset_debug_api() -> None:
    if debug_api.DEBUG_MANAGER is not None:
        debug_api.end_debug()
    TEDebugState._reset()


def _config_file(tmp_path: Path, config_key: str, config: str) -> str:
    if config == "":
        return ""
    path = tmp_path / f"{config_key}.yaml"
    path.write_text(config, encoding="utf-8")
    return str(path)


def _get_model(model_key: str, *, fuse_wgrad_accumulation: bool = False):
    kwargs = {
        "params_dtype": torch.bfloat16,
        "device": "npu",
        "name": f"{model_key}_debug_sanity",
        "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
    }
    if model_key == "linear":
        return te.Linear(D, D, **kwargs)
    if model_key == "layernorm_linear":
        return te.LayerNormLinear(D, D, **kwargs)
    if model_key == "layernorm_mlp":
        return te.LayerNormMLP(D, D, **kwargs)
    raise ValueError(f"Unsupported model key: {model_key}")


def _run_forward_backward(model, *, fp8: bool) -> None:
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        inp = torch.randn((S, B, H), dtype=torch.bfloat16, device="npu", requires_grad=True)
        with autocast(enabled=fp8, recipe=recipe.DelayedScaling()):
            out = model(inp)
            out.float().sum().backward()
        debug_api.step()
    torch.npu.synchronize()


def _run_test(model_key: str, fp8: bool, config_key: str, feature_dirs, tmp_path: Path) -> None:
    config_file = _config_file(tmp_path, config_key, CONFIGS[config_key])
    _reset_debug_api()
    debug_api.initialize(config_file=config_file, feature_dirs=feature_dirs, log_dir=tmp_path)
    try:
        model = _get_model(model_key)
        _run_forward_backward(model, fp8=fp8)
    finally:
        _reset_debug_api()


@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("config_key", CONFIGS.keys())
def test_sanity_debug(model_key, fp8, config_key, feature_dirs, tmp_path):
    if (fp8 or config_key in FP8_REQUIRED_CONFIGS) and not a5_npu_available():
        pytest.skip("Atlas A5 NPU is required for FP8 quantization tests")
    if not fp8 and config_key in FP8_REQUIRED_CONFIGS:
        pytest.skip(f"Config '{config_key}' requires FP8")
    if fp8:
        from transformer_engine.pytorch.quantization import is_fp8_available

        fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
    _run_test(model_key, fp8, config_key, feature_dirs, tmp_path)


@pytest.mark.skipif(not npu_available(), reason="NPU device is required")
@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize("fuse_wgrad_accumulation", [False, True])
def test_non_fp8_debug_inspects_all_gemm_tensors(
    model_key,
    fuse_wgrad_accumulation,
    configs_dir,
    feature_dirs,
):
    """Test non-FP8 debug hooks, including fused wgrad accumulation outputs."""
    _reset_debug_api()
    debug_api.initialize(
        config_file=str(Path(configs_dir) / "dummy_feature.yaml"),
        feature_dirs=feature_dirs,
    )
    try:
        import transformer_engine.debug.features._test_dummy_feature as dummy_feature

        dummy_feature._inspect_tensor_enabled_call_count = 0
        dummy_feature._inspect_tensor_call_count = 0
        dummy_feature._inspect_tensor_calls = []

        model = _get_model(
            model_key,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )
        if fuse_wgrad_accumulation:
            for name in ("weight", "fc1_weight", "fc2_weight"):
                if hasattr(model, name):
                    weight = getattr(model, name)
                    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)

        inp = torch.randn(
            (S, B, H),
            dtype=torch.bfloat16,
            device="npu",
            requires_grad=True,
        )
        model(inp).float().sum().backward()
        debug_api.step()
        torch.npu.synchronize()

        expected_call_count = 12 if model_key == "layernorm_mlp" else 6
        assert dummy_feature._inspect_tensor_enabled_call_count == expected_call_count
        assert dummy_feature._inspect_tensor_call_count == expected_call_count
    finally:
        _reset_debug_api()


@pytest.mark.skipif(not npu_available(), reason="NPU device is required")
@pytest.mark.parametrize(
    ("single_grouped_weight", "fuse_wgrad_accumulation"),
    [(False, False), (True, False), (True, True)],
)
def test_grouped_linear_debug_routes_each_gemm(
    single_grouped_weight,
    fuse_wgrad_accumulation,
    configs_dir,
    feature_dirs,
):
    """Test that each grouped GEMM has independent debug tensor hooks."""
    _reset_debug_api()
    debug_api.initialize(
        config_file=str(Path(configs_dir) / "dummy_feature.yaml"),
        feature_dirs=feature_dirs,
    )
    try:
        import transformer_engine.debug.features._test_dummy_feature as dummy_feature

        dummy_feature._inspect_tensor_enabled_call_count = 0
        dummy_feature._inspect_tensor_call_count = 0
        dummy_feature._inspect_tensor_calls = []

        model = te.GroupedLinear(
            2,
            D,
            D,
            params_dtype=torch.bfloat16,
            device="npu",
            name="grouped_linear_debug_sanity",
            single_grouped_weight=single_grouped_weight,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )
        if fuse_wgrad_accumulation:
            model.weight.main_grad = torch.zeros_like(model.weight, dtype=torch.float32)

        inp = torch.randn(
            (S, B, H),
            dtype=torch.bfloat16,
            device="npu",
            requires_grad=True,
        )
        model(inp, [S * B // 2, S * B // 2]).float().sum().backward()
        debug_api.step()
        torch.npu.synchronize()

        expected_calls = {
            (f"grouped_linear_debug_sanity.gemm_{gemm}", tensor_name)
            for gemm in range(2)
            for tensor_name in (
                "activation",
                "weight",
                "output",
                "gradient",
                "dgrad",
                "wgrad",
            )
        }
        assert dummy_feature._inspect_tensor_enabled_call_count == len(expected_calls)
        assert dummy_feature._inspect_tensor_call_count == len(expected_calls)
        assert set(dummy_feature._inspect_tensor_calls) == expected_calls
    finally:
        _reset_debug_api()


@pytest.mark.skipif(not a5_npu_available(), reason="Atlas A5 NPU is required")
def test_grouped_linear_mxfp8_debug_handles_empty_group(configs_dir, feature_dirs):
    """Test that MXFP8 debug skips GEMM for an expert with no tokens."""
    _reset_debug_api()
    debug_api.initialize(
        config_file=str(Path(configs_dir) / "dummy_feature.yaml"),
        feature_dirs=feature_dirs,
    )
    try:
        model = te.GroupedLinear(
            2,
            32,
            32,
            params_dtype=torch.bfloat16,
            device="npu",
            name="grouped_linear_empty_group_debug_sanity",
        )
        inp = torch.randn(
            (32, 32),
            dtype=torch.bfloat16,
            device="npu",
            requires_grad=True,
        )
        with autocast(enabled=True, recipe=recipe.MXFP8BlockScaling()):
            model(inp, [0, 32]).float().sum().backward()
        debug_api.step()
        torch.npu.synchronize()
    finally:
        _reset_debug_api()
