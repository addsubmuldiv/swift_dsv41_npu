# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for ``transformer_engine_torch.DType``.

The enum must mirror the GPU pybind enum (bound from the C++ ``DType`` in
``transformer_engine/common/include/transformer_engine/transformer_engine.h``)
value-for-value, compare by integer like the pybind version does, and be
accepted by TENPU's quantizers, which natively take ``torch.dtype``.
"""

from __future__ import annotations

import pytest
import torch

from utils import npu_available

from transformer_engine_torch import DType

GPU_ENUM_VALUES = {
    "kByte": 0,
    "kInt16": 1,
    "kInt32": 2,
    "kInt64": 3,
    "kFloat32": 4,
    "kFloat16": 5,
    "kBFloat16": 6,
    "kFloat8E4M3": 7,
    "kFloat8E5M2": 8,
    "kFloat8E8M0": 9,
    "kFloat4E2M1": 10,
}


def test_member_values_match_gpu_enum():
    assert {member.name: member.value for member in DType} == GPU_ENUM_VALUES


def test_integer_comparison_parity():
    # The GPU pybind enum registers custom __eq__/__ne__ comparing by integer
    # (csrc/extensions/pybind.cpp); IntEnum provides the same semantics natively.
    assert DType.kFloat8E4M3 == 7
    assert DType.kFloat8E5M2 == 8
    assert int(DType.kBFloat16) == 6
    assert DType(7) is DType.kFloat8E4M3
    assert DType.kFloat8E4M3 != DType.kFloat8E5M2


@pytest.mark.parametrize(
    "member,expected",
    [
        (DType.kByte, torch.uint8),
        (DType.kInt16, torch.int16),
        (DType.kInt32, torch.int32),
        (DType.kInt64, torch.int64),
        (DType.kFloat32, torch.float32),
        (DType.kFloat16, torch.float16),
        (DType.kBFloat16, torch.bfloat16),
        (DType.kFloat8E4M3, torch.float8_e4m3fn),
        (DType.kFloat8E5M2, torch.float8_e5m2),
    ],
)
def test_torch_dtype_mapping(member, expected):
    assert member.torch_dtype == expected


def test_torch_dtype_unmapped_members_raise():
    # E8M0 (MX scale type) and FP4 have no plain torch.dtype equivalent.
    with pytest.raises(NotImplementedError):
        _ = DType.kFloat8E8M0.torch_dtype
    with pytest.raises(NotImplementedError):
        _ = DType.kFloat4E2M1.torch_dtype


@pytest.mark.skipif(not npu_available(), reason="NPU device is required")
def test_quantizers_accept_dtype_enum():
    # Mirrors Megatron-LM's usage: test_fp8.py / test_mxfp8_refit.py pass
    # tex.DType members straight into the quantizers.
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    float8 = Float8Quantizer(scale=torch.ones(1), amax=torch.zeros(1), fp8_dtype=DType.kFloat8E4M3)
    assert float8.dtype == torch.float8_e4m3fn

    mxfp8 = MXFP8Quantizer(fp8_dtype=DType.kFloat8E5M2, rowwise=True, columnwise=False)
    assert mxfp8.dtype == torch.float8_e5m2
    assert mxfp8.rowwise_usage and not mxfp8.columnwise_usage

    # Plain torch.dtype keeps working unchanged.
    plain = Float8Quantizer(scale=torch.ones(1), amax=torch.zeros(1), fp8_dtype=torch.float8_e5m2)
    assert plain.dtype == torch.float8_e5m2


@pytest.mark.skipif(not npu_available(), reason="NPU device is required")
def test_quantizer_rejects_unknown_dtype():
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    with pytest.raises(TypeError):
        Float8Quantizer(scale=torch.ones(1), amax=torch.zeros(1), fp8_dtype="float8_e4m3")


@pytest.mark.skipif(not npu_available(), reason="NPU device is required")
def test_quantize_end_to_end_with_dtype_enum():
    from transformer_engine.pytorch.quantization import is_fp8_available
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

    fp8_available, reason = is_fp8_available(return_reason=True)
    if not fp8_available:
        pytest.skip(reason)

    quantizer = Float8Quantizer(
        scale=torch.ones(1, device="npu"),
        amax=torch.zeros(1, device="npu"),
        fp8_dtype=DType.kFloat8E4M3,
    )
    x = torch.randn(4, 32, device="npu", dtype=torch.bfloat16)
    fp8_t = quantizer.quantize(x)
    dequant = fp8_t.dequantize(dtype=torch.float32)
    # scale=1.0 quantizes values directly; e4m3 relative precision is ~2^-3.
    torch.testing.assert_close(dequant, x.to(torch.float32), rtol=0.2, atol=0.05)
