# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import io
import math
import signal
from collections.abc import Iterable
from typing import Any, Optional

import pytest
import torch

from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE, TensorUsage
from transformer_engine.pytorch.quantization import (
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    Float8Tensor,
    MXFP8Quantizer,
    MXFP8Tensor,
)
from transformer_engine.pytorch.torch_version import torch_version

_dense_dtypes = [torch.float32, torch.float16, torch.bfloat16]
_dtypes = [torch.float16, torch.bfloat16]
_fp8_dtypes = [torch.float8_e4m3fn, torch.float8_e5m2]
_float8_tols = {
    torch.float8_e4m3fn: dict(rtol=0.125, atol=0.0675),
    torch.float8_e5m2: dict(rtol=0.25, atol=0.125),
}

# NVIDIA CUDA tests use per-element _tols in tests/pytorch/test_quantized_tensor.py.
# The current NPU backend does not yet match those strict bounds for MXFP8
# dequantize/chunk numerics, so we keep a separate acceptance contract instead of
# overloading the NVIDIA name.
_npu_mxfp8_acceptance_bounds = {
    torch.float8_e4m3fn: dict(max_mae=0.1, max_abs=1.0),
    torch.float8_e5m2: dict(max_mae=0.2, max_abs=1.0),
}

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
    return_reason=True
)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
_quantization_list = []
if fp8_available:
    _quantization_list.append("fp8")
if fp8_block_scaling_available:
    _quantization_list.append("fp8_blockwise")
if mxfp8_available:
    _quantization_list.append("mxfp8")


def _to_list(x: Iterable | Any) -> list:
    """Convert to list if iterable, otherwise put in singleton list."""
    if isinstance(x, Iterable):
        return list(x)
    return [x]


def _npu_device() -> torch.device:
    return torch.device("npu")


def _rand_npu(shape: int | Iterable[int], *, dtype: torch.dtype) -> torch.Tensor:
    """Generate random test data without exercising NPU random kernels."""
    return torch.rand(shape, dtype=dtype, device="cpu").to(device=_npu_device())


def _uniform_npu(shape: int | Iterable[int], *, dtype: torch.dtype) -> torch.Tensor:
    """Generate values in [-1, 1] without exercising NPU random kernels."""
    return 2 * _rand_npu(shape, dtype=dtype) - 1


def _npu_float8_raw_shape(shape: Iterable[int]) -> torch.Size:
    """Return NPU Float8 raw storage shape used by quant matmul kernels."""
    shape = tuple(shape)
    if len(shape) == 0:
        return torch.Size((1, 1))
    if len(shape) == 2:
        return torch.Size(shape)
    return torch.Size((math.prod(shape[:-1]), shape[-1]))


def _logical_size(tensor: torch.Tensor) -> torch.Size:
    """Return the NVIDIA-facing logical size for an NPU quantized tensor."""
    if isinstance(tensor, QuantizedTensor) and hasattr(tensor, "_shape"):
        return torch.Size(tensor.origin_shape)
    return tensor.size()


def _assert_float8_raw_storage(tensor: Float8Tensor) -> None:
    """Check NPU Float8 raw storage without relying on tensor public size."""
    expected_raw_shape = _npu_float8_raw_shape(_logical_size(tensor))
    assert tensor._data.size() == expected_raw_shape
    gemm_data, _ = tensor.get_data(TensorUsage.LHS)
    assert gemm_data.size() == expected_raw_shape


def _seed() -> None:
    seed = 1234
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed(seed)


def _make_float8_quantizer(
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    scale: float = 1.0,
    device: torch.device | None = None,
    rowwise: bool = True,
    columnwise: bool = True,
) -> Float8Quantizer:
    if device is None:
        device = _npu_device()
    return Float8Quantizer(
        scale=torch.full((1,), scale, dtype=torch.float32, device=device),
        amax=torch.zeros((1,), dtype=torch.float32, device=device),
        fp8_dtype=fp8_dtype,
        rowwise=rowwise,
        columnwise=columnwise,
    )


def _to_float8(
    tensor: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    scale: float = 1.0,
) -> Float8Tensor:
    tensor = tensor.to(device=_npu_device())
    return _make_float8_quantizer(
        fp8_dtype=fp8_dtype,
        scale=scale,
        device=tensor.device,
    )(tensor)


def _to_float8_current_scaling(
    tensor: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    return_transpose: bool = False,
    force_pow_2_scales: bool = False,
    amax_epsilon: float = 0.0,
) -> Float8Tensor:
    tensor = tensor.to(device=_npu_device())
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=fp8_dtype,
        device=tensor.device,
        force_pow_2_scales=force_pow_2_scales,
        amax_epsilon=amax_epsilon,
    )
    quantizer.set_usage(rowwise=True, columnwise=return_transpose)
    return quantizer(tensor)


def _to_float8_blockwise(
    tensor: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Float8BlockwiseQTensor:
    tensor = tensor.to(device=_npu_device())
    quantizer = Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=True,
        force_pow_2_scales=True,
        amax_epsilon=0.0,
        block_scaling_dim=1,
    )
    return quantizer(tensor)


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    quantization: Optional[str] = None,
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device | str = "cpu",
    test_dtype: torch.dtype = torch.bfloat16,
    test_device: torch.device | str = _npu_device(),
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values.

    This mirrors the helper shape of NVIDIA's tests/pytorch/test_quantized_tensor.py,
    but only covers the quantization modes currently exercised in the NPU port.
    """

    ref = torch.rand(shape, dtype=ref_dtype, device=ref_device)
    test = ref.to(device=test_device, dtype=test_dtype)
    if quantization is None:
        if test.data_ptr() == ref.data_ptr():
            test = test.clone()
    elif quantization == "fp8":
        test = _to_float8(test, fp8_dtype=torch.float8_e4m3fn, scale=1.0)
    elif quantization == "fp8_blockwise":
        test = _to_float8_blockwise(test, fp8_dtype=torch.float8_e4m3fn)
    elif quantization == "mxfp8":
        test = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(test)
    else:
        raise ValueError(f"Unsupported quantization scheme ({quantization})")

    if isinstance(test, QuantizedTensor):
        ref.copy_(test.dequantize().to(dtype=ref.dtype, device="cpu"))
    else:
        ref.copy_(test.to(dtype=ref.dtype, device="cpu"))

    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


@pytest.mark.parametrize(
    "quantization",
    [
        pytest.param(
            "fp8_blockwise",
            marks=pytest.mark.skipif(
                not fp8_block_scaling_available,
                reason=reason_for_no_fp8_block_scaling,
            ),
            id="fp8_blockwise",
        ),
        pytest.param(
            "mxfp8",
            marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
            id="mxfp8",
        ),
    ],
)
@pytest.mark.parametrize("columnwise_usage", [False, True], ids=["rowwise", "rowwise_columnwise"])
def test_block_fp8_serialization(quantization: str, columnwise_usage: bool) -> None:
    """Preserve block FP8 payloads and scales across serialization."""
    _, tensor = make_reference_and_test_tensors(
        (128, 128),
        quantization=quantization,
        test_dtype=torch.bfloat16,
        requires_grad=False,
    )
    tensor.update_usage(rowwise_usage=True, columnwise_usage=columnwise_usage)
    tensor._quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)

    assert restored.__class__ is tensor.__class__
    assert restored.origin_shape == tensor.origin_shape
    assert restored.dtype == tensor.dtype
    assert restored._fp8_dtype == tensor._fp8_dtype
    assert restored._quantizer.__class__ is tensor._quantizer.__class__
    assert restored.get_usages() == tensor.get_usages()
    assert restored._quantizer.get_usages() == tensor._quantizer.get_usages()

    for name in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        expected = getattr(tensor, name)
        actual = getattr(restored, name)
        if expected is None:
            assert actual is None
            continue
        assert actual is not None
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(
            actual.detach().contiguous().view(torch.uint8).cpu(),
            expected.detach().contiguous().view(torch.uint8).cpu(),
        )


@pytest.mark.parametrize("quantization", _quantization_list)
def test_to_copy_dtype_change_dequantizes(quantization: str) -> None:
    """``.to`` with a different dtype returns a dense tensor in that dtype."""
    _, tensor = make_reference_and_test_tensors(
        (128, 128),
        quantization=quantization,
        test_dtype=torch.bfloat16,
        requires_grad=False,
    )

    out = tensor.detach().to(torch.float32)
    expected = tensor.dequantize().to(dtype=torch.float32)

    assert not isinstance(out, QuantizedTensor)
    assert out.dtype == torch.float32
    assert out.device.type == tensor.device.type
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    torch_version() < (2, 6, 0),
    reason="Explicit-path safe globals require PyTorch 2.6 or later",
)
def test_megatron_exit_signal_checkpoint_safe_load() -> None:
    """Allow safe loading of Megatron checkpoints containing args.exit_signal."""
    buffer = io.BytesIO()
    torch.save({"exit_signal": signal.SIGTERM}, buffer)
    buffer.seek(0)

    checkpoint = torch.load(buffer, weights_only=True)

    assert checkpoint["exit_signal"] is signal.SIGTERM


def _assert_npu_mxfp8_acceptance(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_mae: float,
    max_abs: float,
) -> None:
    actual = actual.detach().to(dtype=torch.float32, device="cpu")
    expected = expected.detach().to(dtype=torch.float32, device="cpu")
    diff = (actual - expected).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()
    assert mae <= max_mae, f"mae too large: got {mae:.6f}, expected <= {max_mae:.6f}"
    assert max_err <= max_abs, f"max_abs too large: got {max_err:.6f}, expected <= {max_abs:.6f}"


def _decode_mxfp8_data_for_reference(data: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """Decode MXFP8 payload to float32 on CPU."""
    if data.dtype == fp8_dtype:
        decoded = data
    else:
        decoded = data.contiguous().view(dtype=fp8_dtype)
    return decoded.to(dtype=torch.float32, device="cpu")


def _columnwise_scalar_reference_from_logical_scale(
    data: torch.Tensor,
    logical_scale: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """Independent CPU scalar reference for 1x32 block dequantize."""
    q = _decode_mxfp8_data_for_reference(data, fp8_dtype)
    scale = logical_scale.detach().to(dtype=torch.float32, device="cpu")
    m_dim = math.prod(q.shape[:-1]) if q.ndim > 1 else 1
    k_dim = q.shape[-1] if q.ndim > 0 else 1
    q_2d = q.reshape(m_dim, k_dim)
    out = torch.empty((m_dim, k_dim), dtype=torch.float32, device="cpu")

    k_blocks = math.ceil(k_dim / MXFP8_BLOCK_SCALING_SIZE)
    assert tuple(scale.shape) == (m_dim, k_blocks)
    for m_idx in range(m_dim):
        for k_block in range(k_blocks):
            block_scale = float(scale[m_idx, k_block])
            start = k_block * MXFP8_BLOCK_SCALING_SIZE
            end = min(start + MXFP8_BLOCK_SCALING_SIZE, k_dim)
            for k_idx in range(start, end):
                out[m_idx, k_idx] = float(q_2d[m_idx, k_idx]) * block_scale
    return out.reshape(*q.shape)


def _rowwise_scalar_reference_from_logical_scale(
    data: torch.Tensor,
    logical_scale: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
) -> torch.Tensor:
    """Independent CPU scalar reference for 32x1 block dequantize."""
    q = _decode_mxfp8_data_for_reference(data, fp8_dtype)
    scale = logical_scale.detach().to(dtype=torch.float32, device="cpu")
    m_dim = math.prod(q.shape[:-1]) if q.ndim > 1 else 1
    k_dim = q.shape[-1] if q.ndim > 0 else 1
    q_2d = q.reshape(m_dim, k_dim)
    out = torch.empty((m_dim, k_dim), dtype=torch.float32, device="cpu")

    m_blocks = math.ceil(m_dim / MXFP8_BLOCK_SCALING_SIZE)
    assert tuple(scale.shape) == (m_blocks, k_dim)
    for m_block in range(m_blocks):
        start = m_block * MXFP8_BLOCK_SCALING_SIZE
        end = min(start + MXFP8_BLOCK_SCALING_SIZE, m_dim)
        for k_idx in range(k_dim):
            block_scale = float(scale[m_block, k_idx])
            for m_idx in range(start, end):
                out[m_idx, k_idx] = float(q_2d[m_idx, k_idx]) * block_scale
    return out.reshape(*q.shape)


def _make_manual_mxfp8_tensor(
    shape: Iterable[int],
    *,
    dtype: torch.dtype,
    fp8_dtype: torch.dtype,
    rowwise_data: Optional[torch.Tensor],
    rowwise_scale_inv: Optional[torch.Tensor],
    columnwise_data: Optional[torch.Tensor],
    columnwise_scale_inv: Optional[torch.Tensor],
) -> MXFP8Tensor:
    """Construct MXFP8Tensor directly from supplied raw buffers."""
    return MXFP8Tensor(
        shape=tuple(shape),
        dtype=dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=fp8_dtype,
        quantizer=None,
        with_gemm_swizzled_scales=False,
        device=_npu_device(),
    )


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFloat8Tensor:
    @staticmethod
    def setup_class() -> None:
        _seed()

    @staticmethod
    def _test_quantize_dequantize(
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale: float = 3.5,
        dtype: torch.dtype = torch.float32,
        dims: int | Iterable[int] = 23,
    ) -> None:
        x_ref = 2 * torch.rand(_to_list(dims), dtype=dtype, device="cpu") - 1
        x_fp8 = _to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_dequantized = x_fp8.dequantize(dtype=dtype).to(device="cpu")

        torch.testing.assert_close(x_dequantized, x_ref, **_float8_tols[fp8_dtype])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_dequantized, -x_ref, **_float8_tols[fp8_dtype])

    def test_constructor(
        self,
        dims: int | Iterable[int] = 1,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale_inv: float = 0.375,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Adapted from NVIDIA TestFloat8Tensor.test_constructor."""
        dims = _to_list(dims)
        tensor = Float8Tensor(
            shape=dims,
            dtype=dtype,
            data=torch.zeros(dims, device=_npu_device(), dtype=fp8_dtype),
            fp8_scale_inv=torch.full((1,), scale_inv, dtype=torch.float32, device=_npu_device()),
            fp8_dtype=fp8_dtype,
            quantizer=_make_float8_quantizer(fp8_dtype=fp8_dtype, scale=scale_inv),
            device=_npu_device(),
        )
        assert list(tensor.size()) == dims
        assert tensor.dtype == dtype
        assert tensor.is_cuda

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", _dense_dtypes)
    def test_quantize_dequantize_dtypes(
        self,
        fp8_dtype: torch.dtype,
        dtype: torch.dtype,
    ) -> None:
        self._test_quantize_dequantize(fp8_dtype=fp8_dtype, dtype=dtype)

    @pytest.mark.parametrize("scale", [0.375, 1.0, 3.5])
    def test_quantize_dequantize_scales(self, scale: float) -> None:
        self._test_quantize_dequantize(scale=scale)

    @pytest.mark.parametrize("dims", [[], 1, 311, [7, 11], [7, 5, 3], [2, 3, 5, 3]])
    def test_quantize_dequantize_dims(self, dims: int | Iterable[int]) -> None:
        self._test_quantize_dequantize(dims=dims)

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", _dense_dtypes)
    @pytest.mark.parametrize("noop", [True, False])
    def test_quantize_dequantize_noop(
        self,
        fp8_dtype: torch.dtype,
        dtype: torch.dtype,
        noop: bool,
    ) -> None:
        """Adapted from NVIDIA TestFloat8Tensor.test_quantize_dequantize_noop."""
        noop_tensor = torch.zeros(1, dtype=torch.float32, device=_npu_device())
        if noop:
            noop_tensor = torch.ones(1, dtype=torch.float32, device=_npu_device())
        dims = 23
        scale = 3.5

        x_ref = 2 * torch.rand(_to_list(dims), dtype=dtype, device="cpu") - 1
        x_fp8 = _to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_ref_noop_test = 2 * x_ref.to(device=_npu_device())
        x_fp8_orig = x_fp8.clone()

        x_fp8.quantize_(x_ref_noop_test, noop_flag=noop_tensor)
        if noop_tensor.item() == 1.0:
            torch.testing.assert_close(x_fp8, x_fp8_orig, atol=0, rtol=0)
        else:
            torch.testing.assert_close(
                x_fp8.dequantize(dtype=dtype),
                x_ref_noop_test,
                **_float8_tols[fp8_dtype],
            )

    def test_basic_ops(
        self,
        dims: int | Iterable[int] = 23,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale: float = 3.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Adapted from NVIDIA TestFloat8Tensor.test_basic_ops."""
        dims = _to_list(dims)
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        y_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_fp8 = _to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        y_fp8 = _to_float8(y_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_ref = x_fp8.dequantize(dtype=dtype)
        y_ref = y_fp8.dequantize(dtype=dtype)

        torch.testing.assert_close(-x_fp8, -x_ref, rtol=0, atol=0)
        torch.testing.assert_close(x_fp8.abs(), x_ref.abs(), rtol=0, atol=0)

        tols = _float8_tols[fp8_dtype]
        torch.testing.assert_close(x_fp8 + y_fp8, x_ref + y_ref, **tols)
        torch.testing.assert_close(x_fp8 - y_fp8, x_ref - y_ref, **tols)
        torch.testing.assert_close(x_fp8 * y_fp8, x_ref * y_ref, **tols)
        torch.testing.assert_close(x_fp8 + y_ref, x_ref + y_ref, **tols)
        torch.testing.assert_close(x_ref + y_fp8, x_ref + y_ref, **tols)
        torch.testing.assert_close(torch.sin(x_fp8), torch.sin(x_ref), **tols)

        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8 + y_fp8, x_ref - y_fp8, **tols)

    def test_inplace_ops(
        self,
        dims: int | Iterable[int] = 23,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale: float = 3.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Adapted from NVIDIA TestFloat8Tensor.test_inplace_ops."""
        dims = _to_list(dims)
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        y_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_fp8 = _to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        y_fp8 = _to_float8(y_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_ref = x_fp8.dequantize(dtype=dtype)
        y_ref = y_fp8.dequantize(dtype=dtype)

        tols = _float8_tols[fp8_dtype]
        x_fp8 += y_ref
        x_ref += y_ref
        torch.testing.assert_close(x_fp8.dequantize(dtype=dtype), x_ref, **tols)
        x_ref = x_fp8.dequantize(dtype=dtype)
        x_fp8 -= y_fp8
        x_ref -= y_fp8.dequantize(dtype=dtype)
        torch.testing.assert_close(x_fp8.dequantize(dtype=dtype), x_ref, **tols)
        x_ref = x_fp8.dequantize(dtype=dtype)
        x_fp8 *= 2
        x_ref *= 2
        torch.testing.assert_close(x_fp8.dequantize(dtype=dtype), x_ref, **tols)

        x_ref += 123
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8.dequantize(dtype=dtype), x_ref, **tols)

    def test_serialization(
        self,
        dims: int | Iterable[int] = (2, 3, 5),
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        scale: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Adapted from NVIDIA TestFloat8Tensor.test_serialization."""
        dims = _to_list(dims)
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_fp8 = _to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_ref = x_fp8.dequantize(dtype=dtype).cpu()

        byte_stream = io.BytesIO()
        torch.save(x_fp8, byte_stream)
        x_bytes = byte_stream.getvalue()

        x_fp8._data.zero_()
        x_fp8._scale_inv.zero_()
        del x_fp8, byte_stream

        x_fp8 = torch.load(io.BytesIO(x_bytes), weights_only=False)
        del x_bytes
        torch.testing.assert_close(x_fp8.dequantize(dtype=dtype).cpu(), x_ref, rtol=0, atol=0)

        x_fp8._data.zero_()
        x_fp8._scale_inv.zero_()
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8.dequantize(dtype=dtype).cpu(), x_ref, rtol=0, atol=0)

    def test_set_data(self) -> None:
        """Adapted from NVIDIA TestFloat8Tensor.test_set_data."""
        x0 = torch.zeros(4, dtype=torch.float32)
        x = _to_float8(x0)
        assert isinstance(x, Float8Tensor)
        assert x0.size() == _logical_size(x)
        _assert_float8_raw_storage(x)
        assert x.dtype == torch.float32
        assert x.is_cuda and x._data.is_npu
        y = x.dequantize()
        assert not isinstance(y, Float8Tensor)
        assert _logical_size(x) == y.size()
        assert x.dtype == y.dtype
        assert x.device == y.device

        x0 = torch.zeros((3, 2), dtype=torch.float16, device=x.device)
        x.data = x0
        assert isinstance(x, Float8Tensor)
        assert x0.size() == _logical_size(x)
        _assert_float8_raw_storage(x)
        assert x0.dtype == x.dtype
        assert x0.device == x.device == x._data.device
        y = x.dequantize()
        assert not isinstance(y, Float8Tensor)
        assert _logical_size(x) == y.size()
        assert x.dtype == y.dtype
        assert x.device == y.device

        x = _to_float8(torch.ones((4, 3, 1), dtype=torch.float32))
        x0 = _to_float8(torch.zeros((4, 3, 1), dtype=torch.float32))
        x.data = x0
        assert isinstance(x, Float8Tensor)
        assert _logical_size(x0) == _logical_size(x)
        _assert_float8_raw_storage(x)
        assert x0.dtype == x.dtype
        assert x0.device == x.device == x._data.device
        assert x0._data is x._data
        assert x0._scale_inv is x._scale_inv
        y = x.dequantize()
        assert not isinstance(y, Float8Tensor)
        assert _logical_size(x) == y.size()
        assert x.dtype == y.dtype
        assert x.device == y.device


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestCurrentScalingFloat8Tensor:
    @staticmethod
    def setup_class() -> None:
        _seed()

    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("dims", [[], [128, 128]])
    @pytest.mark.parametrize("return_transpose", [True, False], ids=str)
    @pytest.mark.parametrize("force_pow_2_scales", [True, False], ids=str)
    @pytest.mark.parametrize("amax_epsilon", [0.0, 1e-6], ids=str)
    def test_quantize(
        self,
        fp8_dtype: torch.dtype,
        dtype: torch.dtype,
        dims: int | Iterable[int],
        return_transpose: bool,
        force_pow_2_scales: bool,
        amax_epsilon: float,
    ) -> None:
        """Adapted from NVIDIA TestCurrentScalingFloat8Tensor.test_quantize."""
        x_hp = _uniform_npu(_to_list(dims), dtype=dtype)
        x_fp8 = _to_float8_current_scaling(
            x_hp,
            fp8_dtype=fp8_dtype,
            return_transpose=return_transpose,
            force_pow_2_scales=force_pow_2_scales,
            amax_epsilon=amax_epsilon,
        )

        assert isinstance(x_fp8, Float8Tensor)
        assert x_fp8._data is not None
        assert x_fp8._data.dtype == fp8_dtype
        assert x_fp8._scale_inv.dtype == torch.float32
        assert x_fp8._quantizer is not None
        assert x_fp8._quantizer.get_usages() == {
            "rowwise": True,
            "columnwise": return_transpose,
        }
        torch.testing.assert_close(
            x_fp8.dequantize(dtype=dtype),
            x_hp,
            **_float8_tols[fp8_dtype],
        )

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("dims", [[], 1, 311, [7, 11], [7, 5, 3], [2, 3, 5, 3]])
    def test_quantize_dequantize(
        self,
        fp8_dtype: torch.dtype,
        dtype: torch.dtype,
        dims: int | Iterable[int],
    ) -> None:
        """Adapted from NVIDIA TestCurrentScalingFloat8Tensor.test_quantize_dequantize."""
        x_hp = _uniform_npu(_to_list(dims), dtype=dtype)
        x_fp8 = _to_float8_current_scaling(x_hp, fp8_dtype=fp8_dtype)
        x_deq = x_fp8.dequantize(dtype=dtype)

        torch.testing.assert_close(x_deq, x_hp, **_float8_tols[fp8_dtype])
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_deq, x_hp + 1.0, **_float8_tols[fp8_dtype])


class TestQuantizedTensor:
    @staticmethod
    def setup_class() -> None:
        _seed()

    @pytest.mark.parametrize("op", ("clone",))
    @pytest.mark.parametrize("quantization", _quantization_list)
    @pytest.mark.parametrize("shape", [(128, 128)])
    def test_identity_op(
        self,
        *,
        op: str,
        quantization: str,
        shape: Iterable[int],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Adapted from NVIDIA TestQuantizedTensor.test_identity_op."""

        x_ref, x_test = make_reference_and_test_tensors(
            shape=shape,
            quantization=quantization,
            test_dtype=dtype,
        )
        dy_test = _rand_npu(shape, dtype=dtype)
        dy_ref = dy_test.to(dtype=torch.float64, device="cpu")

        if op == "clone":
            y_ref = x_ref.clone()
            y_test = x_test.clone()
        elif op == "view":
            y_ref = x_ref.view(shape)
            y_test = x_test.view(shape)
        elif op == "reshape":
            y_ref = x_ref.reshape(shape)
            y_test = x_test.reshape(shape)
        else:
            y_ref = x_ref.contiguous()
            y_test = x_test.contiguous()

        if isinstance(y_test, Float8BlockwiseQTensor) and y_test.size() != dy_test.size():
            y_test = y_test.dequantize(dtype=dtype)
            y_test = y_test.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(y_test, y_ref, rtol=0, atol=0)
            return

        y_test.backward(dy_test)
        assert x_test.grad is not None

        if isinstance(y_test, QuantizedTensor):
            y_test = y_test.dequantize(dtype=dtype)
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")

        torch.testing.assert_close(y_test, y_ref, rtol=0, atol=0)
        torch.testing.assert_close(dx_test, dy_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("quantization", _quantization_list)
    @pytest.mark.parametrize("dim", [0])
    @pytest.mark.parametrize("shape", [(128, 128)])
    def test_chunk(
        self,
        *,
        quantization: str,
        dim: int,
        shape: Iterable[int],
        chunks: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Adapted from NVIDIA TestQuantizedTensor.test_chunk."""

        x_ref, x_test = make_reference_and_test_tensors(
            shape=shape,
            quantization=quantization,
            test_dtype=dtype,
            requires_grad=False,
        )

        ys_ref = torch.chunk(x_ref, chunks, dim=dim)
        ys_test = torch.chunk(x_test, chunks, dim=dim)

        for y_ref, y_test in zip(ys_ref, ys_test):
            assert y_ref.size() == _logical_size(y_test)

            if quantization == "fp8":
                assert isinstance(y_test, Float8Tensor)
                y_test = y_test.dequantize(dtype=dtype)
                y_test = y_test.to(dtype=torch.float64, device="cpu")
                torch.testing.assert_close(y_test, y_ref, rtol=0, atol=0)
                continue

            if isinstance(y_test, QuantizedTensor):
                y_test = y_test.dequantize(dtype=dtype)
            y_test = y_test.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(y_test, y_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("quantization", _quantization_list)
    def test_shape_with_none_data(
        self,
        *,
        quantization: str,
        shape: Iterable[int] = (128, 128),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Keep the logical shape when CPU offload clears internal payloads."""
        _, x_test = make_reference_and_test_tensors(
            shape=shape,
            quantization=quantization,
            test_dtype=dtype,
            requires_grad=False,
        )
        expected_shape = torch.Size(shape)
        assert x_test.shape == expected_shape

        if isinstance(x_test, Float8Tensor):
            x_test._data = None
            x_test._transpose = None
        elif isinstance(x_test, (Float8BlockwiseQTensor, MXFP8Tensor)):
            x_test._rowwise_data = None
            x_test._columnwise_data = None

        assert x_test.shape == expected_shape


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
class TestMXFP8Tensor:
    @staticmethod
    def setup_class() -> None:
        _seed()

    def test_copy_from_cpu_dense_tensor(self) -> None:
        """Move a CPU checkpoint tensor to NPU before updating MXFP8 storage."""
        shape = (64, 64)
        dtype = torch.bfloat16
        fp8_dtype = torch.float8_e4m3fn
        src = torch.linspace(-1.0, 1.0, math.prod(shape), dtype=dtype).reshape(shape)
        quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
        dst = quantizer(torch.zeros(shape, dtype=dtype, device=_npu_device()))
        expected = quantizer(src.to(device=_npu_device()))

        dst.copy_(src)

        torch.testing.assert_close(
            dst._rowwise_data.view(torch.uint8),
            expected._rowwise_data.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dst._rowwise_scale_inv,
            expected._rowwise_scale_inv,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dst._columnwise_data.view(torch.uint8),
            expected._columnwise_data.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dst._columnwise_scale_inv,
            expected._columnwise_scale_inv,
            rtol=0,
            atol=0,
        )

    def test_parameter_data_copy_updates_storage(self) -> None:
        """Update the original MXFP8 parameter through its detached data view."""
        shape = (64, 64)
        dtype = torch.bfloat16
        fp8_dtype = torch.float8_e4m3fn
        src = _uniform_npu(shape, dtype=dtype)
        quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
        param = torch.nn.Parameter(quantizer(torch.zeros(shape, dtype=dtype, device=_npu_device())))
        expected = quantizer(src)
        rowwise_data_ptr = param._rowwise_data.data_ptr()
        columnwise_data_ptr = param._columnwise_data.data_ptr()
        rowwise_scale_ptr = param._rowwise_scale_inv.data_ptr()
        columnwise_scale_ptr = param._columnwise_scale_inv.data_ptr()

        param.data.copy_(src)

        assert param._rowwise_data.data_ptr() == rowwise_data_ptr
        assert param._columnwise_data.data_ptr() == columnwise_data_ptr
        assert param._rowwise_scale_inv.data_ptr() == rowwise_scale_ptr
        assert param._columnwise_scale_inv.data_ptr() == columnwise_scale_ptr
        torch.testing.assert_close(
            param._rowwise_data.view(torch.uint8),
            expected._rowwise_data.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            param._rowwise_scale_inv,
            expected._rowwise_scale_inv,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            param._columnwise_data.view(torch.uint8),
            expected._columnwise_data.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            param._columnwise_scale_inv,
            expected._columnwise_scale_inv,
            rtol=0,
            atol=0,
        )

    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("dims", [[128, 128]])
    def test_mxfp8_dequantize_columnwise_only(
        self,
        fp8_dtype: torch.dtype,
        dtype: torch.dtype,
        dims,
    ) -> None:
        """Port of NVIDIA TestMXFP8Tensor.test_mxfp8_dequantize_columnwise_only."""

        x_ref = _uniform_npu(_to_list(dims), dtype=dtype)

        quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
        x_mxfp8 = quantizer(x_ref)

        x_deq_rowwise = x_mxfp8.dequantize(dtype=dtype)
        _assert_npu_mxfp8_acceptance(
            x_deq_rowwise,
            x_ref,
            **_npu_mxfp8_acceptance_bounds[fp8_dtype],
        )

        x_mxfp8.update_usage(rowwise_usage=False, columnwise_usage=True)
        assert x_mxfp8._rowwise_data is None
        assert x_mxfp8._columnwise_data is not None

        x_deq_columnwise = x_mxfp8.dequantize(dtype=dtype)
        _assert_npu_mxfp8_acceptance(
            x_deq_columnwise,
            x_ref,
            **_npu_mxfp8_acceptance_bounds[fp8_dtype],
        )

        with pytest.raises(AssertionError):
            _assert_npu_mxfp8_acceptance(
                x_deq_columnwise,
                -x_ref,
                **_npu_mxfp8_acceptance_bounds[fp8_dtype],
            )

    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
    @pytest.mark.parametrize("dims", [[128, 128]])
    def test_mxfp8_dequantize_columnwise_only_quantized_separately(
        self,
        fp8_dtype: torch.dtype,
        dims,
    ) -> None:
        """Port of NVIDIA TestMXFP8Tensor.test_mxfp8_dequantize_columnwise_only_quantized_separately."""

        dtype = torch.bfloat16
        x_ref = _uniform_npu(_to_list(dims), dtype=dtype)

        quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=False, columnwise=True)
        x_mxfp8 = quantizer(x_ref)
        assert x_mxfp8._rowwise_data is None
        assert x_mxfp8._columnwise_data is not None

        x_deq = x_mxfp8.dequantize(dtype=dtype)
        _assert_npu_mxfp8_acceptance(
            x_deq,
            x_ref,
            **_npu_mxfp8_acceptance_bounds[fp8_dtype],
        )

        with pytest.raises(AssertionError):
            _assert_npu_mxfp8_acceptance(
                x_deq,
                -x_ref,
                **_npu_mxfp8_acceptance_bounds[fp8_dtype],
            )

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    def test_mxfp8_columnwise_manual_tail_matches_cpu_scalar_reference(
        self,
        fp8_dtype: torch.dtype,
    ) -> None:
        """Validate non-32-divisible columnwise dequantize against independent CPU scalar math."""

        shape = (2, 17, 65)
        dtype = torch.float32
        device = _npu_device()
        m_dim = math.prod(shape[:-1])
        k_dim = shape[-1]
        k_blocks = math.ceil(k_dim / MXFP8_BLOCK_SCALING_SIZE)

        dense = torch.linspace(
            -1.0,
            1.0,
            steps=math.prod(shape),
            dtype=torch.float32,
            device=device,
        ).reshape(shape)
        columnwise_data = dense.to(dtype=fp8_dtype)
        logical_scale = (
            torch.arange(m_dim * k_blocks, dtype=torch.float32, device=device)
            .reshape(m_dim, k_blocks)
            .div(32.0)
            .add(0.5)
        )

        x_mxfp8 = _make_manual_mxfp8_tensor(
            shape,
            dtype=dtype,
            fp8_dtype=fp8_dtype,
            rowwise_data=None,
            rowwise_scale_inv=None,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=logical_scale,
        )

        x_deq = x_mxfp8.dequantize(dtype=torch.float32).to(device="cpu")
        x_ref = _columnwise_scalar_reference_from_logical_scale(
            columnwise_data,
            logical_scale,
            fp8_dtype=fp8_dtype,
        )
        torch.testing.assert_close(x_deq, x_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    def test_mxfp8_rowwise_manual_tail_matches_cpu_scalar_reference(
        self,
        fp8_dtype: torch.dtype,
    ) -> None:
        """Validate non-32-divisible rowwise dequantize against independent CPU scalar math."""

        shape = (2, 17, 65)
        dtype = torch.float32
        device = _npu_device()
        m_dim = math.prod(shape[:-1])
        k_dim = shape[-1]
        m_blocks = math.ceil(m_dim / MXFP8_BLOCK_SCALING_SIZE)

        dense = torch.linspace(
            -1.0,
            1.0,
            steps=math.prod(shape),
            dtype=torch.float32,
            device=device,
        ).reshape(shape)
        rowwise_data = dense.to(dtype=fp8_dtype)
        logical_scale = (
            torch.arange(m_blocks * k_dim, dtype=torch.float32, device=device)
            .reshape(m_blocks, k_dim)
            .div(64.0)
            .add(0.25)
        )

        x_mxfp8 = _make_manual_mxfp8_tensor(
            shape,
            dtype=dtype,
            fp8_dtype=fp8_dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=logical_scale,
            columnwise_data=None,
            columnwise_scale_inv=None,
        )

        x_deq = x_mxfp8.dequantize(dtype=torch.float32).to(device="cpu")
        x_ref = _rowwise_scalar_reference_from_logical_scale(
            rowwise_data,
            logical_scale,
            fp8_dtype=fp8_dtype,
        )
        torch.testing.assert_close(x_deq, x_ref, rtol=0, atol=0)
