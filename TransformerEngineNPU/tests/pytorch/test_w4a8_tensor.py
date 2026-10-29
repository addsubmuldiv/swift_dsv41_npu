# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch

from transformer_engine.pytorch.constants import TensorUsage
from transformer_engine.pytorch.quantization import is_w4a8_available
from transformer_engine.pytorch.tensor import (
    MXFP8Quantizer,
    W4A8WeightQuantizer,
)
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

from utils import npu_available


_w4a8_available, _reason_for_no_w4a8 = is_w4a8_available(return_reason=True)
pytestmark = pytest.mark.skipif(
    not npu_available() or not _w4a8_available,
    reason=_reason_for_no_w4a8 or "NPU device is required",
)


def _uniform_npu(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        steps=math.prod(shape),
        dtype=torch.float32,
        device="cpu",
    ).reshape(shape)
    return values.to(device=torch.device("npu"), dtype=dtype).contiguous()


@pytest.mark.parametrize(
    "usage_pair",
    [
        pytest.param(("fprop", TensorUsage.LHS, TensorUsage.RHS_TRANS), id="fprop"),
        pytest.param(("dgrad", TensorUsage.LHS, TensorUsage.RHS), id="dgrad"),
    ],
)
def test_w4a8_matmul_invokes_quant_matmul(usage_pair) -> None:
    tag, usage_a, usage_b = usage_pair
    if tag == "fprop":
        lhs = _uniform_npu((64, 64), dtype=torch.bfloat16)
        rhs = _uniform_npu((96, 64), dtype=torch.bfloat16)
        lhs_q = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(lhs)
        rhs_q = W4A8WeightQuantizer(mxfp8_dtype=torch.float8_e4m3fn)(rhs)
        out = lhs_q.matmul(rhs_q, usage_a, usage_b, torch.bfloat16)
        expected_shape = (lhs.shape[0], rhs.shape[0])
    else:
        lhs = _uniform_npu((64, 96), dtype=torch.bfloat16)
        rhs = _uniform_npu((96, 64), dtype=torch.bfloat16)
        lhs_q = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(lhs)
        rhs_q = W4A8WeightQuantizer(mxfp8_dtype=torch.float8_e4m3fn)(rhs)
        out = lhs_q.matmul(rhs_q, usage_a, usage_b, torch.bfloat16)
        expected_shape = (lhs.shape[0], rhs.shape[1])
    torch.npu.synchronize()
    assert tuple(out.shape) == expected_shape
    assert out.dtype == torch.bfloat16
    assert out.device.type == "npu"


def test_w4a8_dequantize_grouped_roundtrip() -> None:
    """Grouped W4A8 dequantize uses the MXFP8 backward view, like the dense path.

    The MXFP4 rowwise buffers carry an NPU-internal tensor format that
    pure-torch dequantization cannot consume (mirrors
    ``W4A8WeightTensorStorage.dequantize``), so both orientations must return
    the columnwise-view reconstruction.
    """
    num_gemms = 2
    rows_per_group = 32
    in_features = 64
    weight = _uniform_npu((rows_per_group * num_gemms, in_features), dtype=torch.bfloat16)
    quantizer = W4A8WeightQuantizer(mxfp8_dtype=torch.float8_e4m3fn)
    quantizer.set_usage(rowwise=True, columnwise=True)

    group_list = torch.arange(1, num_gemms + 1, device="npu", dtype=torch.int32) * rows_per_group
    grouped = quantizer.grouped_quantize_impl(group_list, weight)
    assert isinstance(grouped, GroupedTensor)
    assert grouped.quantizer is quantizer

    rowwise_dq = grouped.dequantize(dtype=torch.float32)
    columnwise_dq = grouped.dequantize(dtype=torch.float32, columnwise=True)

    torch.testing.assert_close(rowwise_dq, columnwise_dq, rtol=0, atol=0)
    for dq in (rowwise_dq, columnwise_dq):
        assert tuple(dq.shape) == tuple(weight.shape)
        assert dq.dtype == torch.float32
        assert torch.isfinite(dq).all()
        torch.testing.assert_close(
            dq,
            weight.float(),
            rtol=0.25,
            atol=1.0,
            msg="W4A8 grouped dequantize diverged from source weight",
        )
