# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from ...constants import TensorUsage
from ...quantized_tensor import QuantizedTensorStorage, transpose_mx_data
from .mxfp8_tensor_storage import MXFP8TensorStorage

if TYPE_CHECKING:
    from ..w4a8_tensor import W4A8WeightQuantizer


class _FromW4A8WeightFunc(torch.autograd.Function):
    """Cast from W4A8 hybrid storage-backed tensor to a dense dtype."""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],
        tensor: "W4A8WeightTensorStorage",
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor._dtype
        return tensor.dequantize(dtype=dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return grad, None


class W4A8WeightTensorStorage(QuantizedTensorStorage):
    """Hybrid weight storage for W4A8.

    The same logical weight stores MXFP4 data for forward RHS-transposed
    GEMMs and MXFP8 data for backward dgrad GEMMs. Wgrad is not computed
    with this tensor; it is MXFP8 grad-output @ MXFP8 input.
    """

    _mxfp4_rowwise_data: Optional[torch.Tensor]
    _mxfp4_rowwise_scale_inv: Optional[torch.Tensor]
    _mxfp8_columnwise_data: Optional[torch.Tensor]
    _mxfp8_columnwise_scale_inv: Optional[torch.Tensor]
    _quantizer: Optional["W4A8WeightQuantizer"]
    _mxfp4_dtype: torch.dtype
    _mxfp8_dtype: torch.dtype
    _block_size: int

    def __new__(
        cls,
        mxfp4_rowwise_data: Optional[torch.Tensor],
        mxfp4_rowwise_scale_inv: Optional[torch.Tensor],
        mxfp8_columnwise_data: Optional[torch.Tensor],
        mxfp8_columnwise_scale_inv: Optional[torch.Tensor],
        mxfp4_dtype: torch.dtype,
        mxfp8_dtype: torch.dtype,
        quantizer: Optional["W4A8WeightQuantizer"],
        block_size: int,
        *args,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is W4A8WeightTensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._mxfp4_rowwise_data = mxfp4_rowwise_data
        instance._mxfp4_rowwise_scale_inv = mxfp4_rowwise_scale_inv
        instance._mxfp8_columnwise_data = mxfp8_columnwise_data
        instance._mxfp8_columnwise_scale_inv = mxfp8_columnwise_scale_inv
        instance._mxfp4_dtype = mxfp4_dtype
        instance._mxfp8_dtype = mxfp8_dtype
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._block_size = block_size
        return instance

    def get_data(self, usage):
        if usage == TensorUsage.RHS_TRANS:
            return transpose_mx_data(self._mxfp4_rowwise_data, self._mxfp4_rowwise_scale_inv)
        if usage == TensorUsage.RHS:
            return self._mxfp8_columnwise_data, self._mxfp8_columnwise_scale_inv
        raise RuntimeError(
            "W4A8WeightTensorStorage only supports RHS_TRANS for fprop and RHS for dgrad"
        )

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        if not isinstance(src, W4A8WeightTensorStorage):
            raise TypeError("copy_from_storage expects W4A8WeightTensorStorage")
        if self._mxfp4_dtype != src._mxfp4_dtype or self._mxfp8_dtype != src._mxfp8_dtype:
            raise RuntimeError("W4A8 dtype mismatch in copy_from_storage")
        if self._block_size != src._block_size:
            raise RuntimeError("W4A8 block size mismatch in copy_from_storage")

        def _copy_optional(dst: Optional[torch.Tensor], src_tensor: Optional[torch.Tensor]):
            if dst is not None and src_tensor is not None:
                dst.copy_(src_tensor)

        _copy_optional(self._mxfp4_rowwise_data, src._mxfp4_rowwise_data)
        _copy_optional(self._mxfp4_rowwise_scale_inv, src._mxfp4_rowwise_scale_inv)
        _copy_optional(self._mxfp8_columnwise_data, src._mxfp8_columnwise_data)
        _copy_optional(self._mxfp8_columnwise_scale_inv, src._mxfp8_columnwise_scale_inv)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "mxfp4_rowwise_data": self._mxfp4_rowwise_data,
            "mxfp4_rowwise_scale_inv": self._mxfp4_rowwise_scale_inv,
            "mxfp8_columnwise_data": self._mxfp8_columnwise_data,
            "mxfp8_columnwise_scale_inv": self._mxfp8_columnwise_scale_inv,
            "mxfp4_dtype": self._mxfp4_dtype,
            "mxfp8_dtype": self._mxfp8_dtype,
            "quantizer": self._quantizer,
            "block_size": self._block_size,
            "fake_dtype": self._dtype,
        }

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self._dtype
        if self._mxfp8_columnwise_data is None:
            raise RuntimeError("W4A8 dequantize requires the MXFP8 backward weight view")
        tensor = MXFP8TensorStorage(
            rowwise_data=None,
            rowwise_scale_inv=None,
            columnwise_data=self._mxfp8_columnwise_data,
            columnwise_scale_inv=self._mxfp8_columnwise_scale_inv,
            fp8_dtype=self._mxfp8_dtype,
            quantizer=None,
            with_gemm_swizzled_scales=False,
            fake_dtype=self._dtype,
        )
        return tensor.dequantize(dtype=dtype)

    def prepare_for_saving(self) -> Tuple[list[Optional[torch.Tensor]], "W4A8WeightTensorStorage"]:
        tensors = [
            self._mxfp4_rowwise_data,
            self._mxfp4_rowwise_scale_inv,
            self._mxfp8_columnwise_data,
            self._mxfp8_columnwise_scale_inv,
        ]
        self._mxfp4_rowwise_data = None
        self._mxfp4_rowwise_scale_inv = None
        self._mxfp8_columnwise_data = None
        self._mxfp8_columnwise_scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        self._mxfp4_rowwise_data = tensors[0]
        self._mxfp4_rowwise_scale_inv = tensors[1]
        self._mxfp8_columnwise_data = tensors[2]
        self._mxfp8_columnwise_scale_inv = tensors[3]
        return tensors[4:]

    def get_data_tensors(self, fprop_data: bool = True, dgrad_data: bool = True):
        tensors = []
        if fprop_data and self._mxfp4_rowwise_data is not None:
            tensors.append(self._mxfp4_rowwise_data)
        if dgrad_data and self._mxfp8_columnwise_data is not None:
            tensors.append(self._mxfp8_columnwise_data)
        if not tensors:
            raise ValueError("W4A8WeightTensorStorage has no data tensors")
        return tuple(tensors)

    def size(self, *args, **kwargs):
        if self._mxfp4_rowwise_data is not None:
            return self._mxfp4_rowwise_data.size(*args, **kwargs)
        return self._mxfp8_columnwise_data.size(*args, **kwargs)

    def clear(self):
        for tensor in (
            self._mxfp4_rowwise_data,
            self._mxfp4_rowwise_scale_inv,
            self._mxfp8_columnwise_data,
            self._mxfp8_columnwise_scale_inv,
        ):
            if tensor is not None:
                tensor.untyped_storage().resize_(0)

    def clear_wise(self, rowwise=False, colwise=False):
        tensors = []
        if rowwise:
            tensors.extend((self._mxfp4_rowwise_data, self._mxfp4_rowwise_scale_inv))
        if colwise:
            tensors.extend((self._mxfp8_columnwise_data, self._mxfp8_columnwise_scale_inv))
        for tensor in tensors:
            if tensor is not None:
                tensor.untyped_storage().resize_(0)

    def to(self, *args, **kwargs):
        for tensor in (
            self._mxfp4_rowwise_data,
            self._mxfp4_rowwise_scale_inv,
            self._mxfp8_columnwise_data,
            self._mxfp8_columnwise_scale_inv,
        ):
            if tensor is not None:
                tensor.to(*args, **kwargs)

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        if rowwise_usage is None:
            rowwise_usage = self._mxfp4_rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._mxfp8_columnwise_data is not None

        if rowwise_usage:
            if self._mxfp4_rowwise_data is None or self._mxfp4_rowwise_scale_inv is None:
                raise RuntimeError("Requested W4A8 forward usage, but MXFP4 data is missing")
        else:
            self._mxfp4_rowwise_data = None
            self._mxfp4_rowwise_scale_inv = None

        if columnwise_usage:
            if self._mxfp8_columnwise_data is None or self._mxfp8_columnwise_scale_inv is None:
                raise RuntimeError("Requested W4A8 dgrad usage, but MXFP8 data is missing")
        else:
            self._mxfp8_columnwise_data = None
            self._mxfp8_columnwise_scale_inv = None

    def get_usages(self) -> Dict[str, bool]:
        return {
            "rowwise": self._mxfp4_rowwise_data is not None,
            "columnwise": self._mxfp8_columnwise_data is not None,
        }

    @staticmethod
    def scale_shape_for_rowwise(shape, block_size: int):
        m_dim = math.prod(shape[:-1]) if len(shape) > 1 else 1
        k_dim = shape[-1] if len(shape) > 0 else 1
        return k_dim, math.ceil(m_dim / block_size)

    @staticmethod
    def scale_shape_for_columnwise(shape, block_size: int):
        m_dim = math.prod(shape[:-1]) if len(shape) > 1 else 1
        k_dim = shape[-1] if len(shape) > 0 else 1
        return m_dim, math.ceil(k_dim / block_size)

    def __repr__(self):
        return (
            "W4A8WeightTensorStorage("
            f"mxfp4_dtype={self._mxfp4_dtype}, "
            f"mxfp8_dtype={self._mxfp8_dtype}, "
            f"block_size={self._block_size}, "
            f"usages={self.get_usages()}"
            ")"
        )


__all__ = ["W4A8WeightTensorStorage", "_FromW4A8WeightFunc"]
