# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""API feature tests for the Transformer Engine debug namespace."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformer_engine.debug.pytorch.debug_state import TEDebugState
from utils import a5_npu_available


debug_api = pytest.importorskip("nvdlfw_inspect.api")


requires_a5_quantization = pytest.mark.skipif(
    not a5_npu_available(),
    reason="Atlas A5 NPU is required for FP8 quantization tests",
)


def _config_path(configs_dir: str, filename: str) -> str:
    return str(Path(configs_dir) / filename)


def _reset_debug_api() -> None:
    from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS

    if debug_api.DEBUG_MANAGER is not None:
        debug_api.end_debug()
    TEDebugState._reset()
    STATS_BUFFERS.reset()


def _float8_quantizer():
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    return Float8Quantizer(
        scale=torch.ones(1, device="npu"),
        amax=torch.zeros(1, device="npu"),
        fp8_dtype=torch.float8_e4m3fn,
    )


def test_transformer_engine_no_config(feature_dirs):
    _reset_debug_api()
    debug_api.initialize(config_file="", feature_dirs=feature_dirs)
    try:
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="fprop", iteration=0
        )[0]
        assert not debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.attn.qkv",
            gemm="fprop",
            tensor_name="activation",
            iteration=0,
        )[0]
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.1.attn.qkv", tensor_name="activation", iteration=0
        )[0]
    finally:
        _reset_debug_api()


def test_disable_quantization_gemm(configs_dir, feature_dirs):
    _reset_debug_api()
    debug_api.initialize(
        config_file=_config_path(configs_dir, "disable_quantization_gemms.yaml"),
        feature_dirs=feature_dirs,
    )
    try:
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="fprop", iteration=0
        )[0]
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="dgrad", iteration=0
        )[0]
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="wgrad", iteration=0
        )[0]

        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="fprop", iteration=0
        )[0]
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="dgrad", iteration=0
        )[0]
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="wgrad", iteration=0
        )[0]
    finally:
        _reset_debug_api()


def test_disable_quantization_layer(configs_dir, feature_dirs):
    _reset_debug_api()
    debug_api.initialize(
        config_file=_config_path(configs_dir, "disable_quantization_layer.yaml"),
        feature_dirs=feature_dirs,
    )
    try:
        for gemm in ("fprop", "dgrad", "wgrad"):
            assert debug_api.transformer_engine.fp8_gemm_enabled(
                "decoder.1.mlp.fc1", gemm=gemm, iteration=0
            )[0]
            assert not debug_api.transformer_engine.fp8_gemm_enabled(
                "decoder.1.attn.qkv", gemm=gemm, iteration=0
            )[0]
    finally:
        _reset_debug_api()


def test_disable_quantization_layer_and_gemm_can_coexist(configs_dir, feature_dirs):
    _reset_debug_api()
    debug_api.initialize(
        config_file=_config_path(configs_dir, "disable_quantization_layer_and_gemm.yaml"),
        feature_dirs=feature_dirs,
    )
    try:
        for gemm in ("fprop", "dgrad", "wgrad"):
            assert not debug_api.transformer_engine.fp8_gemm_enabled(
                "decoder.1.mlp.fc1", gemm=gemm, iteration=0
            )[0]
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.mlp.fc2", gemm="fprop", iteration=0
        )[0]
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.mlp.fc2", gemm="dgrad", iteration=0
        )[0]
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.mlp.fc2", gemm="wgrad", iteration=0
        )[0]
    finally:
        _reset_debug_api()


def test_log_fp8_tensor_stats_enabled_from_yaml(configs_dir, feature_dirs):
    _reset_debug_api()
    debug_api.initialize(
        config_file=_config_path(configs_dir, "log_fp8_tensor_stats.yaml"),
        feature_dirs=feature_dirs,
        default_logging_enabled=False,
    )
    try:
        assert debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.1.mlp.fc1", tensor_name="activation", iteration=0
        )[0]
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.1.mlp.fc1", tensor_name="weight", iteration=0
        )[0]
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.2.mlp.fc1", tensor_name="activation", iteration=0
        )[0]
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.1.mlp.fc1", tensor_name="activation", iteration=3
        )[0]
    finally:
        _reset_debug_api()


@requires_a5_quantization
def test_log_fp8_tensor_stats_collects_through_namespace(configs_dir, feature_dirs):
    from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS

    _reset_debug_api()
    debug_api.initialize(
        config_file=_config_path(configs_dir, "log_fp8_tensor_stats.yaml"),
        feature_dirs=feature_dirs,
        default_logging_enabled=False,
    )
    try:
        tensor = torch.tensor([0.0, 1.0, 2.0], device="npu")
        quantizer = _float8_quantizer()
        tensor_fp8 = quantizer(tensor)
        debug_api.transformer_engine.inspect_tensor(
            "decoder.1.mlp.fc1",
            tensor=tensor,
            tensor_name="activation",
            iteration=0,
            tp_group=None,
            quantizer=quantizer,
            rowwise_quantized_tensor=tensor_fp8,
            columnwise_quantized_tensor=tensor_fp8,
        )
        stats = STATS_BUFFERS.log_stats()
        assert ("decoder.1.mlp.fc1", "activation", "underflows%", 0) in stats
    finally:
        _reset_debug_api()
