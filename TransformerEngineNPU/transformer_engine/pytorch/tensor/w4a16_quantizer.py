# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""MXFP4 weight fake quantization for W4A16 QAT."""

import math
from typing import Optional

import torch

from transformer_engine.common.recipe import Recipe, W4A16BlockScaling
from transformer_engine.common.recipe.base import QParams

from ..quantized_tensor import Quantizer
from .dummy_tensor import DummyTensor
from .mxfp4_tensor import MXFP4Quantizer


class FakeMXFP4Quantizer(Quantizer):
    """Fake-quantize weights to MXFP4 E2M1 and return them in the input dtype."""

    def __init__(self, *, block_size: int = 32, qparams: Optional[QParams] = None) -> None:
        if block_size != 32:
            raise ValueError("FakeMXFP4Quantizer currently supports block_size=32 only")
        super().__init__(rowwise=True, columnwise=True)
        self.block_size = block_size
        self._mxfp4_quantizer = MXFP4Quantizer(
            rowwise=True,
            columnwise=False,
            block_size=block_size,
            qparams=qparams,
            with_rht=False,
        )

    @property
    def dtype(self) -> torch.dtype:
        """MXFP4 payload dtype used during fake quantization."""
        return self._mxfp4_quantizer.dtype

    def copy(self) -> "FakeMXFP4Quantizer":
        """Create a shallow copy."""
        quantizer = FakeMXFP4Quantizer(
            block_size=self.block_size,
            qparams=self._mxfp4_quantizer.qparams,
        )
        quantizer.rowwise_usage = self.rowwise_usage
        quantizer.columnwise_usage = self.columnwise_usage
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        return quantizer

    def _fake_quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return tensor
        quantized = self._mxfp4_quantizer.quantize_impl(tensor)
        return quantized.dequantize(dtype=tensor.dtype)

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: DummyTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> DummyTensor:
        """Refresh an existing fake-quantized tensor."""
        if not isinstance(dst, DummyTensor):
            raise ValueError("FakeMXFP4Quantizer can only update DummyTensor")
        return dst._update_data(self._fake_quantize(src), grouped=False, noop_flag=noop_flag)

    def quantize_impl(self, tensor: torch.Tensor) -> DummyTensor:
        """Apply MXFP4 quantize-dequantize to a weight tensor."""
        if tensor is None:
            return tensor
        return DummyTensor(
            tensor.shape,
            tensor.dtype,
            data=self._fake_quantize(tensor),
            quantizer=self,
        )

    def update_grouped_quantized(
        self,
        src: torch.Tensor,
        dst: DummyTensor,
        group_list,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> DummyTensor:
        """Refresh an existing grouped fake-quantized tensor."""
        if not isinstance(dst, DummyTensor):
            raise ValueError("FakeMXFP4Quantizer can only update DummyTensor")
        return dst._update_data(self._fake_quantize(src), grouped=True, noop_flag=noop_flag)

    def grouped_quantize_impl(  # pylint: disable=unused-argument
        self,
        group_list,
        tensor: torch.Tensor,
    ) -> DummyTensor:
        """Apply MXFP4 quantize-dequantize to grouped weights."""
        if tensor is None:
            return tensor
        return DummyTensor(
            tensor.shape,
            tensor.dtype,
            data=self._fake_quantize(tensor),
            quantizer=self,
            grouped=True,
        )

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Return whether the weight shape satisfies MXFP4 block alignment."""
        if inp.ndim < 2 or inp.shape[-1] % self.block_size != 0:
            return False
        return math.prod(inp.shape[:-1]) % self.block_size == 0

    def _get_compatible_recipe(self) -> type[Recipe]:
        return W4A16BlockScaling


__all__ = ["FakeMXFP4Quantizer"]
