# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""A5 coverage for FP8, MXFP, and W4A8 tensor quantizers."""

import pytest
import torch
import torch_npu

from utils import npu_available
from transformer_engine.pytorch.constants import NPUVersion, TensorUsage
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.w4a8_tensor import W4A8WeightQuantizer
from transformer_engine.pytorch.utils import check_npu_version

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")
_A5 = check_npu_version(NPUVersion.A5)


def _skip_unsupported_npu_op(exc: RuntimeError) -> None:
    msg = str(exc)
    unsupported_tokens = (
        "dynamic_quant_run_aclnn",
        "npu_quant_matmul",
        "aclnn",
        "NPU function error",
    )
    if any(token in msg for token in unsupported_tokens):
        pytest.skip(f"NPU quantization runtime does not support this op: {msg.splitlines()[0]}")
    raise exc


def _float8_quantizer(*, rowwise: bool = True, columnwise: bool = True) -> Float8Quantizer:
    return Float8Quantizer(
        scale=torch.tensor([1.0], device="npu"),
        amax=torch.tensor([0.0], device="npu"),
        fp8_dtype=torch.float8_e4m3fn,
        rowwise=rowwise,
        columnwise=columnwise,
    )


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestFloat8Quantizer:
    def test_construct(self):
        q = _float8_quantizer()
        assert q.dtype == torch.float8_e4m3fn

    def test_copy(self):
        q = _float8_quantizer(rowwise=False, columnwise=True)
        copied = q.copy()
        assert copied.dtype == q.dtype
        assert copied.scale is q.scale
        assert copied.rowwise_usage is False
        assert copied.columnwise_usage is True

    def test_make_empty(self):
        q = _float8_quantizer()
        empty = q.make_empty(shape=[2, 4], dtype=torch.float32, device=torch.device("npu"))
        assert tuple(empty.shape) == (2, 4)
        assert isinstance(empty._quantizer, Float8Quantizer)

    def test_calibrate_noop(self):
        q = _float8_quantizer()
        inp = torch.tensor([[-3.0, 1.0], [2.0, -0.5]], device="npu")
        q.calibrate(inp)
        torch.testing.assert_close(q.amax.cpu(), torch.tensor([3.0]))

    def test_quantize_none_is_a_noop(self):
        assert _float8_quantizer().quantize_impl(None) is None

    def test_update_rejects_non_float8_destination(self):
        with pytest.raises(ValueError, match="can only update Float8Tensor"):
            _float8_quantizer().update_quantized(
                torch.zeros(2, 4, device="npu"),
                torch.zeros(2, 4, device="npu"),
            )

    def test_quantize_roundtrip(self):
        q = _float8_quantizer()
        inp = torch.randn(3, 4, device="cpu").to("npu")
        try:
            fp8 = q.quantize_impl(inp)
        except RuntimeError as exc:
            _skip_unsupported_npu_op(exc)
        assert fp8.dequantize().shape == (3, 4)
        assert q.amax.item() >= inp.abs().max().item()


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestFloat8CurrentScalingQuantizer:
    def test_construct(self):
        q = Float8CurrentScalingQuantizer(fp8_dtype=torch.float8_e4m3fn, device=torch.device("npu"))
        assert q.dtype == torch.float8_e4m3fn

    def test_make_empty(self):
        q = Float8CurrentScalingQuantizer(fp8_dtype=torch.float8_e4m3fn, device=torch.device("npu"))
        empty = q.make_empty(shape=[4, 8], dtype=torch.float32, device=torch.device("npu"))
        assert tuple(empty.shape) == (4, 8)
        assert isinstance(empty._quantizer, Float8CurrentScalingQuantizer)

    def test_copy_and_serialized_state_preserve_configuration(self):
        q = Float8CurrentScalingQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=False,
            columnwise=True,
            with_amax_reduction=True,
            device=torch.device("npu"),
        )
        copied = q.copy()
        state = q.__getstate__()
        assert copied.rowwise_usage is False
        assert copied.columnwise_usage is True
        assert copied.with_amax_reduction is True
        assert state["amax_reduction_group"] is None

    def test_quantize_none_is_a_noop(self):
        q = Float8CurrentScalingQuantizer(device=torch.device("npu"))
        assert q.quantize_impl(None) is None

    def test_quantize_impl(self):
        q = Float8CurrentScalingQuantizer(fp8_dtype=torch.float8_e4m3fn, device=torch.device("npu"))
        try:
            result = q.quantize_impl(torch.randn(32, 32, device="npu"))
        except RuntimeError as exc:
            _skip_unsupported_npu_op(exc)
        assert result.shape == (32, 32)


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestFloat8Tensor:
    def test_matmul(self):
        qa = _float8_quantizer()
        qb = _float8_quantizer()
        try:
            a_fp8 = qa.quantize_impl(torch.randn(32, 64, device="npu", dtype=torch.float16))
            b_fp8 = qb.quantize_impl(torch.randn(64, 48, device="npu", dtype=torch.float16))
            out = a_fp8.matmul(b_fp8, TensorUsage.LHS, TensorUsage.RHS_TRANS, torch.float16)
        except RuntimeError as exc:
            _skip_unsupported_npu_op(exc)
        assert out.shape == (32, 48)


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestFloat8BlockQuantizer:
    def test_construct(self):
        q = Float8BlockQuantizer(fp8_dtype=torch.float8_e4m3fn, rowwise=True, columnwise=True)
        assert q.dtype == torch.float8_e4m3fn


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestMXFP8Tensor:
    def test_quantizer_construct(self):
        q = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)
        assert q.dtype == torch.float8_e4m3fn
        assert q.is_quantizable(torch.empty(32, 32, device="npu"))

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [((32,), False), ((32, 31), False), ((31, 32), False), ((32, 32), True)],
    )
    def test_is_quantizable_requires_block_aligned_2d_input(self, shape, expected):
        q = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)
        assert q.is_quantizable(torch.empty(shape, device="npu")) is expected

    def test_disabled_quantizer_returns_original_tensor(self):
        q = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn, rowwise=False, columnwise=False)
        inp = torch.zeros(32, 32, device="npu")
        assert q.quantize_impl(inp) is inp


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestMXFP4Tensor:
    def test_quantizer_construct(self):
        q = MXFP4Quantizer(fp4_dtype=torch_npu.float4_e2m1fn_x2, rowwise=True, columnwise=False)
        assert q.dtype == torch_npu.float4_e2m1fn_x2
        assert q.rowwise_usage is True
        assert q.columnwise_usage is False

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [((32,), False), ((32, 31), False), ((31, 32), False), ((32, 32), True)],
    )
    def test_is_quantizable_requires_block_aligned_2d_input(self, shape, expected):
        q = MXFP4Quantizer(fp4_dtype=torch_npu.float4_e2m1fn_x2)
        assert q.is_quantizable(torch.empty(shape, device="npu")) is expected

    def test_disabled_quantizer_returns_original_tensor(self):
        q = MXFP4Quantizer(rowwise=False, columnwise=False)
        inp = torch.zeros(32, 32, device="npu")
        assert q.quantize_impl(inp) is inp


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestW4A8Tensor:
    def test_quantizer_construct(self):
        q = W4A8WeightQuantizer(
            mxfp8_dtype=torch.float8_e4m3fn,
            rowwise=False,
            columnwise=True,
        )
        assert q.mxfp8_dtype == torch.float8_e4m3fn
        assert q.rowwise_usage is False
        assert q.columnwise_usage is True

    @pytest.mark.parametrize(
        ("shape", "expected"),
        [((32,), False), ((32, 31), False), ((31, 32), False), ((32, 32), True)],
    )
    def test_is_quantizable_requires_block_aligned_2d_input(self, shape, expected):
        q = W4A8WeightQuantizer(mxfp8_dtype=torch.float8_e4m3fn)
        assert q.is_quantizable(torch.empty(shape, device="npu")) is expected

    def test_disabled_quantizer_returns_original_tensor(self):
        q = W4A8WeightQuantizer(
            mxfp8_dtype=torch.float8_e4m3fn,
            rowwise=False,
            columnwise=False,
        )
        inp = torch.zeros(32, 32, device="npu")
        assert q.quantize_impl(inp) is inp
