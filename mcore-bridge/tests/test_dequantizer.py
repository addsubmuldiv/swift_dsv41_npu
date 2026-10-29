# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from unittest.mock import patch

from mcore_bridge.bridge.gpt_bridge import GPTBridge
from mcore_bridge.model.gpts.deepseek_v4 import DeepseekV4Bridge
from mcore_bridge.utils.dequantizer import Fp8Dequantizer, fp4_to_fp8, fp4_to_fp32

UE8M0 = pytest.param(
    getattr(torch, 'float8_e8m0fnu', None),
    marks=pytest.mark.skipif(
        not hasattr(torch, 'float8_e8m0fnu'), reason='UE8M0 is unavailable in this PyTorch version'))


@pytest.fixture(params=['cpu', 'npu'])
def device(request):
    if request.param == 'npu' and (not hasattr(torch, 'npu') or not torch.npu.is_available()):
        pytest.skip('NPU unavailable')
    return torch.device(request.param)


@pytest.mark.parametrize('layout', ['block', 'row', 'padded', 'strided'])
@pytest.mark.parametrize('as_bytes', [False, True])
def test_fp8_all_encodings(device, layout, as_bytes):
    raw = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    block_size = 'auto'
    if layout == 'strided':
        raw = raw.t()
    elif layout == 'padded':
        raw = raw[:13, :15]
        block_size = (8, 8)
    scale_rows = 16 if layout == 'row' else 2
    scales = torch.arange(1, scale_rows * 2 + 1, dtype=torch.float32).reshape(scale_rows, 2) / 8
    converter = Fp8Dequantizer(block_size=block_size)
    expected = converter.convert(raw, scales)
    weight = raw.to(device)
    if not as_bytes:
        weight = weight.view(torch.float8_e4m3fn)
    with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('unexpected CPU fallback')):
        actual = converter.convert(weight, scales.to(device))
    assert actual.device.type == device.type
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    finite = ~torch.isnan(expected)
    assert torch.equal(torch.signbit(actual.cpu()[finite]), torch.signbit(expected[finite]))


def test_fp4_all_encodings(device):
    packed = torch.arange(256, dtype=torch.int32).to(torch.int8).reshape(16, 16)
    expected = fp4_to_fp8(packed).float()
    scales = torch.tensor([[0.5, 1.], [2., 4.]])
    converter = Fp8Dequantizer(block_size='auto')
    with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('unexpected CPU fallback')):
        unpacked = fp4_to_fp32(packed.to(device))
        actual = converter.convert(unpacked, scales.to(device))
    assert unpacked.dtype == torch.float32
    assert actual.device.type == device.type
    torch.testing.assert_close(unpacked.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(unpacked.cpu()), torch.signbit(expected))
    torch.testing.assert_close(actual.cpu(), converter.convert(expected, scales), rtol=0, atol=0)


@pytest.mark.parametrize('packed_fp4', [False, True])
@pytest.mark.parametrize('source_device', ['cpu', 'npu'])
@pytest.mark.parametrize('scale_dtype', [torch.float32, UE8M0])
def test_npu_bridge_loading(device, packed_fp4, source_device, scale_dtype):
    if device.type != 'npu':
        pytest.skip('NPU bridge path')
    scales = torch.tensor([[0.5, 1.], [2., 4.]]).to(scale_dtype)
    if packed_fp4:
        weight = torch.arange(256, dtype=torch.int32).to(torch.int8).reshape(16, 16)
        decoded = fp4_to_fp8(weight).float()
        bridge = object.__new__(DeepseekV4Bridge)
    else:
        weight = (torch.arange(256).reshape(16, 16) % 31 - 15).to(torch.float8_e4m3fn)
        decoded = weight.float()
        bridge = object.__new__(GPTBridge)
    param = torch.nn.Parameter(torch.empty(decoded.shape, dtype=torch.bfloat16, device=device))
    expected = Fp8Dequantizer(block_size='auto').convert(decoded, scales).bfloat16()
    convert = Fp8Dequantizer.convert

    def on_device(self, tensor, scale):
        assert tensor.device.type == scale.device.type == 'npu'
        return convert(self, tensor, scale)

    weight, scales = weight.to(source_device), scales.to(source_device)
    with patch.object(Fp8Dequantizer, 'convert', on_device), patch.object(
            torch.Tensor, 'cpu', side_effect=AssertionError('unexpected CPU fallback')):
        bridge._set_param(param, weight, scales)
    torch.testing.assert_close(param.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.float8_e5m2, UE8M0])
def test_fp8_scale_encodings(device, dtype):
    scales = torch.arange(256, dtype=torch.uint8).reshape(16, 16).view(dtype)
    weight = torch.ones(16, 16, dtype=torch.float8_e4m3fn)
    expected = Fp8Dequantizer(block_size='auto').convert(weight, scales)
    with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('unexpected CPU fallback')):
        actual = Fp8Dequantizer(block_size='auto').convert(weight.to(device), scales.to(device))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
