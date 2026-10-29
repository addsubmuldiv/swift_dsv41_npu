# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tests for debug FP8 stats computation with GroupedTensor."""

from pathlib import Path
import textwrap

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch.quantization import is_mxfp8_available

mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)


def _compute_fp8_stat(stat, tensor, quantized_tensor):
    from transformer_engine.debug.features.utils.stats_computation import STATS

    return STATS[stat][0](tensor, {"": quantized_tensor})


def _make_float8_grouped(
    rowwise_data,
    rowwise_scale_inv,
    *,
    columnwise_data=None,
    columnwise_scale_inv=None,
    num_tensors=2,
):
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    quantizer = Float8Quantizer(
        scale=torch.ones(1),
        amax=torch.zeros(1),
        fp8_dtype=torch.float8_e4m3fn,
        columnwise=columnwise_data is not None or columnwise_scale_inv is not None,
    )
    return GroupedTensor(
        shape=tuple(rowwise_data.shape),
        dtype=torch.float32,
        num_tensors=num_tensors,
        quantizer=quantizer,
        data=rowwise_data,
        scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
    )


def test_grouped_float8_dequantize_uses_packed_group_layout():
    data = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float8_e4m3fn,
    )
    grouped = _make_float8_grouped(data, torch.tensor([1.0, 2.0]))

    result = grouped.dequantize(dtype=torch.float32)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 4.0], [6.0, 8.0]])
    torch.testing.assert_close(result, expected)


def test_grouped_tensor_get_data_tensors_selects_orientation():
    rowwise = torch.ones((4, 2), dtype=torch.float8_e4m3fn)
    columnwise = torch.full((4, 2), 2.0, dtype=torch.float8_e4m3fn)
    grouped = _make_float8_grouped(
        rowwise,
        torch.ones(2),
        columnwise_data=columnwise,
        columnwise_scale_inv=torch.ones(2),
    )

    assert grouped.get_data_tensors(rowwise_data=True, columnwise_data=False) is rowwise
    assert grouped.get_data_tensors(rowwise_data=False, columnwise_data=True) is columnwise
    assert grouped.get_data_tensors() == (rowwise, columnwise)


def test_grouped_float8_stats_aggregate_complete_packed_storage():
    rowwise = (
        torch.tensor(
            [0, 128, 56, 64, 56, 56, 56, 56],
            dtype=torch.uint8,
        )
        .view(torch.float8_e4m3fn)
        .reshape(4, 2)
    )
    columnwise = torch.ones((4, 2), dtype=torch.float8_e4m3fn)
    grouped = _make_float8_grouped(
        rowwise,
        torch.tensor([1.0, 4.0]),
        columnwise_data=columnwise,
        columnwise_scale_inv=torch.tensor([2.0, 2.0]),
    )
    rowwise_dequantized = torch.tensor([[0.0, -0.0], [1.0, 2.0], [4.0, 4.0], [4.0, 4.0]])
    original = rowwise_dequantized + 1.0

    assert _compute_fp8_stat("scale_inv_min", original, grouped).item() == 1.0
    assert _compute_fp8_stat("scale_inv_max", original, grouped).item() == 4.0
    assert _compute_fp8_stat("mse", original, grouped).item() == pytest.approx(1.0)
    assert _compute_fp8_stat("underflows%", original, grouped).item() == pytest.approx(25.0)
    assert _compute_fp8_stat("scale_inv_min_columnwise", original, grouped).item() == 2.0
    assert _compute_fp8_stat("scale_inv_max_columnwise", original, grouped).item() == 2.0
    expected_columnwise_mse = F.mse_loss(original, torch.full_like(original, 2.0))
    torch.testing.assert_close(
        _compute_fp8_stat("mse_columnwise", original, grouped),
        expected_columnwise_mse,
    )


def test_grouped_mxfp8_scale_stats_decode_e8m0():
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    data = torch.ones((32, 32), dtype=torch.float8_e4m3fn)
    grouped = GroupedTensor(
        shape=tuple(data.shape),
        dtype=torch.float32,
        num_tensors=2,
        quantizer=MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn),
        data=data,
        scale_inv=torch.tensor([126, 127, 128], dtype=torch.uint8),
    )

    assert _compute_fp8_stat("scale_inv_min", None, grouped).item() == pytest.approx(0.5)
    assert _compute_fp8_stat("scale_inv_max", None, grouped).item() == pytest.approx(2.0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_grouped_mxfp8_dequantize_delegates_storage_layout():
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    rowwise = torch.ones((32, 32), dtype=torch.float8_e4m3fn).npu()
    columnwise = torch.full((32, 32), 2.0, dtype=torch.float8_e4m3fn).npu()
    rowwise_scale = torch.full((32, 1), 127, dtype=torch.uint8).npu()
    columnwise_scale = torch.full((1, 32), 127, dtype=torch.uint8).npu()
    grouped = GroupedTensor(
        shape=tuple(rowwise.shape),
        dtype=torch.float32,
        num_tensors=2,
        quantizer=MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn),
        data=rowwise,
        scale_inv=rowwise_scale,
        columnwise_data=columnwise,
        columnwise_scale_inv=columnwise_scale,
    )

    torch.testing.assert_close(grouped.dequantize().cpu(), torch.ones((32, 32)))
    torch.testing.assert_close(
        grouped.dequantize(columnwise=True).cpu(),
        torch.full((32, 32), 2.0),
    )


def test_grouped_blockwise_dequantize_delegates_storage_layout():
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    data = torch.ones((128, 128), dtype=torch.float8_e4m3fn)
    grouped = GroupedTensor(
        shape=tuple(data.shape),
        dtype=torch.float32,
        num_tensors=2,
        quantizer=Float8BlockQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            block_scaling_dim=2,
        ),
        data=data,
        scale_inv=torch.tensor([[2.0]]),
    )

    torch.testing.assert_close(grouped.dequantize(), torch.full((128, 128), 2.0))


def test_grouped_blockwise_mse_dequantizes_to_float32():
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    data = torch.ones((128, 128), dtype=torch.float8_e4m3fn)
    grouped = GroupedTensor(
        shape=tuple(data.shape),
        dtype=torch.float8_e4m3fn,
        num_tensors=2,
        quantizer=Float8BlockQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            block_scaling_dim=2,
            force_pow_2_scales=False,
        ),
        data=data,
        scale_inv=torch.tensor([[1.1]]),
    )
    original = torch.full((128, 128), 1.2)

    assert _compute_fp8_stat("mse", original, grouped).item() == pytest.approx(0.01)


def test_grouped_dequantize_rejects_missing_requested_scale():
    data = torch.ones((4, 2), dtype=torch.float8_e4m3fn)
    grouped = _make_float8_grouped(data, None)

    with pytest.raises(ValueError, match="rowwise scale_inv"):
        grouped.dequantize()


def test_log_fp8_tensor_stats_collects_grouped_packed_storage_with_real_debug_init(tmp_path):
    debug_api = pytest.importorskip("nvdlfw_inspect.api")
    from nvdlfw_inspect.logging import BaseLogger

    from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
    from transformer_engine.debug.pytorch.debug_state import TEDebugState

    class CaptureStatsLogger(BaseLogger):
        def __init__(self):
            self.records = {}

        def log_scalar(self, name, value, iteration, **kwargs):
            self.records[(name, iteration)] = value

    if debug_api.DEBUG_MANAGER is not None:
        debug_api.end_debug()
    TEDebugState._reset()

    config_file = tmp_path / "grouped_fp8_stats.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            grouped_fp8_stats:
              enabled: True
              layers:
                layer_name_regex_pattern: ^grouped_linear$
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
    feature_dirs = Path(__file__).parents[3] / "transformer_engine" / "debug" / "features"

    try:
        data = torch.ones((4, 2), dtype=torch.float8_e4m3fn)
        grouped = _make_float8_grouped(data, torch.tensor([1.0, 2.0]))
        original = torch.tensor([[2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0]])
        STATS_BUFFERS.reset()
        stats_logger = CaptureStatsLogger()

        debug_api.initialize(
            config_file=config_file,
            feature_dirs=[str(feature_dirs)],
            log_dir=tmp_path,
            statistics_logger=stats_logger,
            default_logging_enabled=True,
        )
        enabled, next_iter = debug_api.transformer_engine.inspect_tensor_enabled(
            layer_name="grouped_linear",
            tensor_name="activation",
            iteration=3,
        )
        assert enabled is True
        assert next_iter is None

        debug_api.transformer_engine.inspect_tensor(
            layer_name="grouped_linear",
            tensor_name="activation",
            iteration=3,
            tp_group=None,
            tensor=original,
            rowwise_quantized_tensor=grouped,
            columnwise_quantized_tensor=grouped,
            quantizer=grouped.quantizer,
        )

        debug_api.step()
        records = stats_logger.records
        assert records[("grouped_linear_activation_scale_inv_min", 3)].item() == 1.0
        assert records[("grouped_linear_activation_scale_inv_max", 3)].item() == 2.0
        assert records[("grouped_linear_activation_mse", 3)].item() == 1.0
        assert records[("grouped_linear_activation_underflows%", 3)].item() == 0.0
        stats_log_dir = tmp_path / "nvdlfw_inspect_statistics_logs"
        stats_log_text = "\n".join(
            path.read_text(encoding="utf-8") for path in stats_log_dir.glob("*.log")
        )
        stats_log_lines = stats_log_text.splitlines()
        expected_log_values = {
            "grouped_linear_activation_scale_inv_min": 1.0,
            "grouped_linear_activation_scale_inv_max": 2.0,
            "grouped_linear_activation_mse": 1.0,
            "grouped_linear_activation_underflows%": 0.0,
        }
        for metric_name, expected_value in expected_log_values.items():
            assert any(
                metric_name in line and f"value={expected_value:.4e}" in line
                for line in stats_log_lines
            )
    finally:
        if debug_api.DEBUG_MANAGER is not None:
            debug_api.end_debug()
        TEDebugState._reset()
        STATS_BUFFERS.reset()
