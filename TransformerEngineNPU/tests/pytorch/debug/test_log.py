# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tests for debug tensor-stat logging."""

from pathlib import Path
import textwrap

import pytest
import torch
import torch.nn.functional as F

from utils import a5_npu_available


pytest.importorskip("nvdlfw_inspect.api")


requires_a5_quantization = pytest.mark.skipif(
    not a5_npu_available(),
    reason="Atlas A5 NPU is required for FP8 quantization tests",
)


LOG_QUANTIZED_CONFIG_BASE = """
log:
  layers:
    layer_name_regex_pattern: .*
  enabled: True
  transformer_engine:
    LogFp8TensorStats:
      enabled: True
      stats: [{stats}]
      tensors: [activation, gradient, weight]
      freq: 2
      start_step: 0
      end_step: 10
"""

fp8_recipes = [
    "mxfp8",
    "fp8_delayed_scaling",
    "fp8_current_scaling",
    "fp8_block_scaling",
]

bare_stats = [
    "underflows%",
    "scale_inv_min",
    "scale_inv_max",
    "mse",
]


@pytest.fixture(autouse=True)
def _feature_import_environment(tmp_path):
    from nvdlfw_inspect.logging import initialize_logger

    initialize_logger(tmp_path)


def _reset_debug_api():
    import nvdlfw_inspect.api as debug_api

    from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    if debug_api.DEBUG_MANAGER is not None:
        debug_api.end_debug()
    TEDebugState._reset()
    STATS_BUFFERS.reset()


def _feature_dirs():
    return Path(__file__).parents[3] / "transformer_engine" / "debug" / "features"


def _float8_quantizer():
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    return Float8Quantizer(
        scale=torch.ones(1, device="npu"),
        amax=torch.zeros(1, device="npu"),
        fp8_dtype=torch.float8_e4m3fn,
    )


def _quantize_float8(tensor):
    quantizer = _float8_quantizer()
    return quantizer, quantizer(tensor)


def test_get_reduction_params_uses_tp_group_only_for_weight(monkeypatch):
    import nvdlfw_inspect.api as debug_api

    from transformer_engine.debug.features.utils import get_reduction_params
    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    default_group = object()
    tp_group = object()
    monkeypatch.setattr(debug_api, "get_tensor_reduction_group", lambda: default_group)
    monkeypatch.setattr(TEDebugState, "weight_tensor_tp_group_reduce", True)

    assert get_reduction_params("activation", tp_group, 2) == (False, default_group, True)
    assert get_reduction_params("weight", tp_group, 2) == (False, tp_group, False)
    assert get_reduction_params("weight", None, 2) == (True, default_group, False)


def test_upstream_stats_registry_is_available():
    from transformer_engine.debug.features.utils.stats_computation import (
        DEPENDENCIES,
        STATS,
        stats_to_num,
    )

    assert "mean" in STATS
    assert DEPENDENCIES["mean"] == {"sum", "numel"}
    assert "mxfp8_scale_inv_min_columnwise" in stats_to_num


def test_stats_buffer_initializes_storage_on_explicit_device():
    from transformer_engine.debug.features.utils.stats_buffer import STATS, _Buffer

    buffer = _Buffer(
        "linear",
        "activation",
        ["mse"],
        None,
        True,
        device=torch.device("npu"),
    )

    assert buffer._buffer.shape == (len(STATS),)
    assert buffer._buffer.device.type == "npu"
    assert buffer._new_buffer is not buffer._buffer
    assert buffer._tmp_buffer is not buffer._buffer
    assert not buffer.modified[0]


def test_stats_buffers_isolate_schedule_options():
    from transformer_engine.debug.features.utils.stats_buffer import StatsBuffers

    buffers = StatsBuffers()
    buffers.try_add_buffer(
        "linear",
        "activation",
        ["mse"],
        (0, 4, None, "fp8"),
        None,
        True,
        device=torch.device("npu"),
    )
    buffers.try_add_buffer(
        "linear",
        "activation",
        ["mse"],
        (6, 10, None, "fp8"),
        None,
        True,
        device=torch.device("npu"),
    )

    assert len(buffers.buffers) == 2


@requires_a5_quantization
def test_stats_buffer_excludes_inactive_rank(monkeypatch):
    from transformer_engine.debug.features.utils import stats_buffer
    from transformer_engine.debug.features.utils.stats_computation import stats_to_num

    tensor = torch.ones(3, device="npu")
    quantizer, quantized = _quantize_float8(tensor)
    buffer = stats_buffer._Buffer(
        "linear", "activation", ["scale_inv_min"], object(), True, device=tensor.device
    )
    buffer.feed(tensor, 3, aux_dict={"": quantized})
    inactive_buffer = buffer._buffer.clone()
    inactive_buffer[stats_to_num["scale_inv_min"]] = 100.0
    gathered = iter(
        (
            (torch.tensor([True, False], device="npu"), None),
            (torch.stack((buffer._buffer, inactive_buffer)), None),
        )
    )
    monkeypatch.setattr(
        stats_buffer,
        "gather_along_first_dim",
        lambda *_args, **_kwargs: next(gathered),
    )

    output = buffer.log()
    expected = quantized._scale_inv.float().min()
    torch.testing.assert_close(output[("linear", "activation", "scale_inv_min", 3)], expected)
    assert quantizer is not None


@requires_a5_quantization
def test_log_fp8_tensor_stats_collects_real_float8_stats_with_debug_init(tmp_path):
    import nvdlfw_inspect.api as debug_api
    from nvdlfw_inspect.logging import BaseLogger

    class CaptureStatsLogger(BaseLogger):
        def __init__(self):
            self.records = {}

        def log_scalar(self, name, value, iteration, **kwargs):
            self.records[(name, iteration)] = value

    _reset_debug_api()
    config_file = tmp_path / "fp8_stats.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            fp8_stats:
              enabled: True
              layers:
                layer_name_regex_pattern: ^linear$
              transformer_engine:
                LogFp8TensorStats:
                  enabled: True
                  tensors: [activation]
                  stats: [scale_inv_min, scale_inv_max, mse, underflows%]
                  start_step: 3
                  end_step: 3
                  freq: 1
            """
        ),
        encoding="utf-8",
    )
    tensor = torch.tensor([1.0, 2.0, 3.0], device="npu")
    quantizer, quantized = _quantize_float8(tensor)
    stats_logger = CaptureStatsLogger()

    debug_api.initialize(
        config_file=config_file,
        feature_dirs=[str(_feature_dirs())],
        log_dir=tmp_path,
        statistics_logger=stats_logger,
        default_logging_enabled=True,
    )
    try:
        enabled, next_iter = debug_api.transformer_engine.inspect_tensor_enabled(
            layer_name="linear",
            tensor_name="activation",
            iteration=3,
        )
        assert enabled is True
        assert next_iter is None
        debug_api.transformer_engine.inspect_tensor(
            layer_name="linear",
            tensor_name="activation",
            iteration=3,
            tp_group=None,
            tensor=tensor,
            rowwise_quantized_tensor=quantized,
            columnwise_quantized_tensor=quantized,
            quantizer=quantizer,
        )
        debug_api.step()
    finally:
        _reset_debug_api()

    records = stats_logger.records
    assert ("linear_activation_scale_inv_min", 3) in records
    assert ("linear_activation_scale_inv_max", 3) in records
    torch.testing.assert_close(
        records[("linear_activation_mse", 3)],
        F.mse_loss(tensor.float(), quantized.dequantize().float()),
    )
    assert records[("linear_activation_underflows%", 3)] >= 0


@pytest.mark.parametrize("fp8_recipe", fp8_recipes)
@requires_a5_quantization
def test_log_quantized_stats_numerics(fp8_recipe, tmp_path):
    import nvdlfw_inspect.api as debug_api

    from transformer_engine.common import recipe
    from transformer_engine.pytorch.quantization import (
        RecipeState,
        is_fp8_available,
        is_fp8_block_scaling_available,
        is_mxfp8_available,
    )

    fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
    mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
    fp8_block_scaling_available, reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
        return_reason=True
    )
    recipe_by_name = {
        "mxfp8": recipe.MXFP8BlockScaling(),
        "fp8_delayed_scaling": recipe.DelayedScaling(),
        "fp8_current_scaling": recipe.Float8CurrentScaling(),
        "fp8_block_scaling": recipe.Float8BlockScaling(),
    }
    fp8_recipe = recipe_by_name[fp8_recipe]

    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if not mxfp8_available and fp8_recipe == recipe.MXFP8BlockScaling():
        pytest.skip(reason_for_no_mxfp8)
    if not fp8_block_scaling_available and fp8_recipe == recipe.Float8BlockScaling():
        pytest.skip(reason_for_no_fp8_block_scaling)

    config_file = tmp_path / "log_quantized_stats_numerics.yaml"
    config_file.write_text(
        LOG_QUANTIZED_CONFIG_BASE.format(stats=", ".join(bare_stats)),
        encoding="utf-8",
    )
    _reset_debug_api()
    debug_api.initialize(
        config_file=config_file,
        feature_dirs=[str(_feature_dirs())],
        log_dir=tmp_path,
        default_logging_enabled=True,
    )
    try:
        recipe_state = RecipeState.create(
            fp8_recipe,
            mode="forward",
            num_quantizers=3,
            device=torch.device("npu"),
        )

        tensor = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
        tensor[0, 100:200] = -0.0
        quantizer = recipe_state.make_quantizers()[0]
        quantized_tensor = quantizer(tensor)

        debug_api.transformer_engine.inspect_tensor(
            layer_name="layer_name",
            tensor_name="activation",
            iteration=0,
            tp_group=None,
            tensor=tensor,
            quantizer=quantizer,
            rowwise_quantized_tensor=quantized_tensor,
            columnwise_quantized_tensor=quantized_tensor,
        )
        debug_api.step()

        dequantized_tensor = quantized_tensor.dequantize()
    finally:
        _reset_debug_api()

    stats_log_dir = tmp_path / "nvdlfw_inspect_statistics_logs"
    output = "\n".join(path.read_text(encoding="utf-8") for path in stats_log_dir.glob("*.log"))
    for line in output.splitlines():
        if "underflows%" in line:
            underflows = float(line.split("value=")[1])
            expected = (
                ((dequantized_tensor == 0).sum() - (tensor == 0).sum()) / tensor.numel() * 100
            )
            assert underflows == pytest.approx(expected.cpu(), abs=1e-4)
        if "mse" in line:
            mse = float(line.split("value=")[1])
            expected = F.mse_loss(dequantized_tensor, tensor, reduction="mean")
            assert mse == pytest.approx(expected.float().cpu(), abs=1e-4)
        if "overflows%" in line:
            overflows = float(line.split("value=")[1])
            expected = (
                (abs(dequantized_tensor) > abs(tensor)).sum() / dequantized_tensor.numel() * 100
            )
            assert overflows == pytest.approx(expected.cpu(), abs=1e-4)


@requires_a5_quantization
def test_log_fp8_tensor_stats_rejects_mixed_rowwise_columnwise_tensors():
    from transformer_engine.debug.features.log_fp8_tensor_stats import LogFp8TensorStats

    tensor = torch.ones(2, device="npu")
    quantizer = _float8_quantizer()

    with pytest.raises(AssertionError):
        LogFp8TensorStats().inspect_tensor(
            {"stats": ["scale_inv_min"]},
            layer_name="linear",
            tensor_name="activation",
            iteration=3,
            tp_group=None,
            tensor=tensor,
            rowwise_quantized_tensor=tensor,
            columnwise_quantized_tensor=tensor.clone(),
            quantizer=quantizer,
        )


@requires_a5_quantization
def test_log_fp8_tensor_stats_skips_dense_tensor_like_upstream(monkeypatch):
    import nvdlfw_inspect.api as debug_api

    from transformer_engine.debug.features.log_fp8_tensor_stats import LogFp8TensorStats
    from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    monkeypatch.setattr(debug_api, "get_tensor_reduction_group", lambda: None)
    monkeypatch.setattr(debug_api, "log_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(TEDebugState, "get_iteration", classmethod(lambda cls: 3))
    STATS_BUFFERS.reset()

    dense_tensor = torch.ones(2, device="npu")
    with pytest.warns(UserWarning, match="incompatible precision"):
        LogFp8TensorStats().inspect_tensor(
            {"stats": ["scale_inv_min"]},
            layer_name="linear",
            tensor_name="activation",
            iteration=3,
            tp_group=None,
            tensor=dense_tensor,
            rowwise_quantized_tensor=dense_tensor,
            columnwise_quantized_tensor=dense_tensor,
            quantizer=_float8_quantizer(),
        )

    assert not STATS_BUFFERS.log_stats()


@pytest.mark.parametrize(
    ("iteration", "expected"),
    (
        (3, (False, 4)),
        (4, (True, 6)),
    ),
)
def test_next_enabled_iter_uses_global_frequency_alignment(iteration, expected):
    from transformer_engine.debug.features.utils import next_enabled_iter

    assert next_enabled_iter(3, 8, None, 2, iteration) == expected


def test_next_enabled_iter_finds_distant_future_interval():
    from transformer_engine.debug.features.utils import next_enabled_iter

    assert next_enabled_iter(None, None, [[10, 12]], 2, 1) == (False, 10)
