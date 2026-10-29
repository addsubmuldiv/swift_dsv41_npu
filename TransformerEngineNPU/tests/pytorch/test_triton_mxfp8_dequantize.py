# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tests for Triton MXFP8 dequantization kernels."""

import math

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
    MXFP8TensorStorage,
    _dequantize_mxfp8_impl,
)
from transformer_engine.pytorch.triton.mxfp8_dequantize import mxfp8_dequantize
from transformer_engine.pytorch.utils import check_npu_version


def _encode_e8m0_scale(scale_float: torch.Tensor) -> torch.Tensor:
    """Encode float32 scale to E8M0 format (uint8)."""
    log2_scale = torch.log2(scale_float)
    exponent = (log2_scale + 127.0).round().clamp(0, 255)
    return exponent.to(torch.uint8)


def _create_mxfp8_storage(
    m_dim: int,
    k_dim: int,
    block_axis: str = "k",
    device: str = "npu",
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> MXFP8TensorStorage:
    """Create MXFP8 test storage with random data (optimized with vectorized operations)."""
    data_float = torch.randn(m_dim, k_dim, dtype=torch.float32, device=device)

    if block_axis == "k":
        k_blocks = math.ceil(k_dim / MXFP8_BLOCK_SCALING_SIZE)
        scales_float = torch.rand(m_dim, k_blocks, dtype=torch.float32, device=device) * 0.1 + 0.01

        # Pad K dimension if needed
        k_padded = k_blocks * MXFP8_BLOCK_SCALING_SIZE
        if k_padded > k_dim:
            data_padded = torch.nn.functional.pad(data_float, (0, k_padded - k_dim), value=0.0)
        else:
            data_padded = data_float

        # Vectorized quantization: reshape to [M, K_blocks, BLOCK_SIZE]
        data_blocked = data_padded.reshape(m_dim, k_blocks, MXFP8_BLOCK_SCALING_SIZE)
        scales_broadcast = scales_float.unsqueeze(-1)
        quantized_blocked = (data_blocked / scales_broadcast).to(fp8_dtype)
        quantized_data = quantized_blocked.reshape(m_dim, k_padded)[:, :k_dim]
    else:
        m_blocks = math.ceil(m_dim / MXFP8_BLOCK_SCALING_SIZE)
        scales_float = torch.rand(m_blocks, k_dim, dtype=torch.float32, device=device) * 0.1 + 0.01

        # Pad M dimension if needed
        m_padded = m_blocks * MXFP8_BLOCK_SCALING_SIZE
        if m_padded > m_dim:
            data_padded = torch.nn.functional.pad(
                data_float, (0, 0, 0, m_padded - m_dim), value=0.0
            )
        else:
            data_padded = data_float

        # Vectorized quantization: reshape to [M_blocks, BLOCK_SIZE, K]
        data_blocked = data_padded.reshape(m_blocks, MXFP8_BLOCK_SCALING_SIZE, k_dim)
        scales_broadcast = scales_float.unsqueeze(1)
        quantized_blocked = (data_blocked / scales_broadcast).to(fp8_dtype)
        quantized_data = quantized_blocked.reshape(m_padded, k_dim)[:m_dim, :]

    scale_inv_e8m0 = _encode_e8m0_scale(scales_float)

    return MXFP8TensorStorage(
        rowwise_data=quantized_data,
        rowwise_scale_inv=scale_inv_e8m0,
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=fp8_dtype,
        quantizer=None,
        with_gemm_swizzled_scales=False,
        fake_dtype=torch.float32,
    )


class TestMXFP8DequantizeTritonCorrectness:
    """Correctness tests for Triton MXFP8 dequantization."""

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="quantized types tests require Atlas A5"
    )
    @pytest.mark.parametrize("m_dim", [64, 128])
    @pytest.mark.parametrize("k_dim", [64, 128])
    @pytest.mark.parametrize("block_axis", ["k", "m"])
    def test_dequantize_aligned_shapes(self, m_dim, k_dim, block_axis):
        """Test dequantization with shapes that are multiples of 32."""
        storage = _create_mxfp8_storage(m_dim, k_dim, block_axis)

        output_pytorch = _dequantize_mxfp8_impl(storage, torch.float32)
        output_triton = mxfp8_dequantize(storage, torch.float32)

        torch.testing.assert_close(
            output_triton,
            output_pytorch,
            rtol=1e-5,
            atol=1e-6,
            msg=f"Mismatch for shape ({m_dim}, {k_dim}), axis={block_axis}",
        )
