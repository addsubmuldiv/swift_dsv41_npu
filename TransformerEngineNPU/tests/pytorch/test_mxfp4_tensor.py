# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch

from transformer_engine.pytorch.constants import TensorUsage
from transformer_engine.pytorch.quantization import is_mxfp4_available
from transformer_engine.pytorch.tensor import MXFP4Quantizer, MXFP4Tensor

from utils import npu_available


_mxfp4_available, _reason_for_no_mxfp4 = is_mxfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(
    not npu_available() or not _mxfp4_available,
    reason=_reason_for_no_mxfp4 or "NPU device is required",
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


def _dequantize_mxfp4_oracle(
    data: torch.Tensor,
    scale_inv: torch.Tensor,
    shape: tuple[int, int],
    *,
    axis: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    packed = data.contiguous().view(torch.uint8)
    codes = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).flatten(-2)
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=packed.device,
        dtype=torch.float32,
    )
    values = table[codes.long()].reshape(shape)
    scales = torch.pow(2.0, scale_inv.contiguous().view(torch.uint8).float() - 127.0)

    if axis == -1:
        num_blocks = shape[1] // 32
        scales = scales.reshape(shape[0], -1)[:, :num_blocks]
        result = values.reshape(shape[0], num_blocks, 32) * scales.unsqueeze(-1)
    else:
        assert axis == -2
        num_blocks = shape[0] // 32
        scales = scales.permute(0, 2, 1).reshape(-1, shape[1])[:num_blocks]
        result = values.reshape(num_blocks, 32, shape[1]) * scales.unsqueeze(1)
    return result.reshape(shape).to(dtype)


def test_mxfp4_quantize_invokes_dynamic_mx_quant() -> None:
    dense = _uniform_npu((64, 64), dtype=torch.bfloat16)
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=True, with_rht=True)

    quantized = quantizer(dense)
    torch.npu.synchronize()

    assert isinstance(quantized, MXFP4Tensor)
    assert tuple(quantized.shape) == tuple(dense.shape)
    assert quantized.dtype == dense.dtype
    assert quantized._rowwise_data is not None
    assert quantized._rowwise_scale_inv is not None
    assert quantized._columnwise_data is not None
    assert quantized._columnwise_scale_inv is not None
    assert quantized._rowwise_data.device.type == "npu"
    assert quantized._columnwise_data.device.type == "npu"


def test_mxfp4_quantized_matmul_invokes_quant_matmul() -> None:
    inp = _uniform_npu((64, 64), dtype=torch.bfloat16)
    weight = _uniform_npu((96, 64), dtype=torch.bfloat16)
    quantizer = MXFP4Quantizer(rowwise=True, columnwise=True, with_rht=False)

    inp_mxfp4 = quantizer(inp)
    weight_mxfp4 = quantizer(weight)
    out = inp_mxfp4.matmul(
        weight_mxfp4,
        TensorUsage.LHS,
        TensorUsage.RHS_TRANS,
        torch.bfloat16,
    )
    torch.npu.synchronize()

    assert tuple(out.shape) == (inp.shape[0], weight.shape[0])
    assert out.dtype == torch.bfloat16
    assert out.device.type == "npu"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mxfp4_columnwise_dequantize(dtype: torch.dtype) -> None:
    dense = _uniform_npu((64, 96), dtype=dtype)
    quantizer = MXFP4Quantizer(rowwise=False, columnwise=True, with_rht=False)
    quantized = quantizer(dense)

    actual = quantized.dequantize(dtype=dtype)
    expected = (
        _dequantize_mxfp4_oracle(
            quantized._columnwise_data.t().contiguous(),
            quantized._columnwise_scale_inv.transpose(0, 1).contiguous(),
            (dense.shape[1], dense.shape[0]),
            axis=-1,
            dtype=dtype,
        )
        .t()
        .contiguous()
    )
    torch.npu.synchronize()

    assert actual.shape == dense.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_mxfp4_columnwise_dequantize(dtype: torch.dtype) -> None:
    dense = _uniform_npu((64, 96), dtype=dtype)
    group_list = torch.tensor([32, 64], device="npu", dtype=torch.int32)
    quantizer = MXFP4Quantizer(rowwise=False, columnwise=True, with_rht=False)
    quantized = quantizer.grouped_quantize(dense, group_list)

    actual = quantized.dequantize(dtype=dtype, columnwise=True)
    expected = _dequantize_mxfp4_oracle(
        quantized.columnwise_data,
        quantized.columnwise_scale_inv,
        tuple(dense.shape),
        axis=-2,
        dtype=dtype,
    )
    torch.npu.synchronize()

    assert actual.shape == dense.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
