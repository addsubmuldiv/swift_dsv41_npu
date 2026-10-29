# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from ...constants import TensorUsage
from ...quantized_tensor import QuantizedTensorStorage, transpose_mx_data

if TYPE_CHECKING:
    from ..mxfp8_tensor import MXFP8Quantizer


class _FromMXFP8Func(torch.autograd.Function):
    """Cast from MXFP8 storage-backed tensor to other dtype."""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],
        tensor: "MXFP8TensorStorage",
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor._dtype
        return _dequantize_mxfp8_impl(tensor, dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return grad, None


class MXFP8TensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of MXFP8Tensor.

    MXFP8Tensor inherits from the PyTorch tensor class and this mixin
    class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.

    """

    # Row-scaled FP8 data
    _rowwise_data: Optional[torch.Tensor]
    # Column-scaled FP8 data
    _columnwise_data: Optional[torch.Tensor]
    # Scaling factors for row-scaled FP8 data
    _rowwise_scale_inv: torch.Tensor
    # Scaling factors for column-scaled FP8 data
    _columnwise_scale_inv: torch.Tensor

    # Builder class for casting to MXFP8
    _quantizer: Optional["MXFP8Quantizer"]
    # FP8 data type
    _fp8_dtype: torch.dtype
    # Whether scaling factors are in the swizzled format expected by
    # GEMM
    _with_gemm_swizzled_scales: bool

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: torch.dtype,
        quantizer: Optional["MXFP8Quantizer"],
        with_gemm_swizzled_scales: bool,
        *args,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is MXFP8TensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._fp8_dtype = fp8_dtype
        instance._with_gemm_swizzled_scales = with_gemm_swizzled_scales
        return instance

    @property
    def row(self):
        return self._rowwise_data, self._rowwise_scale_inv

    @property
    def col(self):
        return self._columnwise_data, self._columnwise_scale_inv

    def get_data(self, usage):
        if self._quantizer is not None and getattr(self._quantizer, "use_2d_block_scaling", False):
            if usage == TensorUsage.LHS:
                return self._rowwise_data, self._rowwise_scale_inv
            if usage == TensorUsage.RHS_TRANS:
                if self._columnwise_data is None:
                    return transpose_mx_data(self._rowwise_data, self._rowwise_scale_inv)
                return self._columnwise_data, self._columnwise_scale_inv
            if usage == TensorUsage.RHS:
                if self._rowwise_data is None:
                    return self._columnwise_data, self._columnwise_scale_inv
                return self._rowwise_data, self._rowwise_scale_inv
            assert usage == TensorUsage.LHS_TRANS
            return self._columnwise_data, self._columnwise_scale_inv
        if usage == TensorUsage.LHS:
            return self._rowwise_data, self._rowwise_scale_inv
        if usage == TensorUsage.RHS_TRANS:
            return transpose_mx_data(self._rowwise_data, self._rowwise_scale_inv)
        if usage == TensorUsage.RHS:
            return self._columnwise_data, self._columnwise_scale_inv
        assert usage == TensorUsage.LHS_TRANS
        return transpose_mx_data(self._columnwise_data, self._columnwise_scale_inv)

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another MXFP8TensorStorage."""
        if not isinstance(src, MXFP8TensorStorage):
            raise TypeError("copy_from_storage expects MXFP8TensorStorage")
        if self._fp8_dtype != src._fp8_dtype:
            raise RuntimeError("FP8 dtype mismatch in copy_from_storage")
        if self._with_gemm_swizzled_scales != src._with_gemm_swizzled_scales:
            raise RuntimeError("Scale layout mismatch in copy_from_storage")

        def _copy_optional(dst: Optional[torch.Tensor], src_tensor: Optional[torch.Tensor]):
            if dst is not None and src_tensor is not None:
                dst.copy_(src_tensor)

        _copy_optional(self._rowwise_data, src._rowwise_data)
        _copy_optional(self._columnwise_data, src._columnwise_data)
        _copy_optional(self._rowwise_scale_inv, src._rowwise_scale_inv)
        _copy_optional(self._columnwise_scale_inv, src._columnwise_scale_inv)

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "quantizer": self._quantizer,
            "with_gemm_swizzled_scales": self._with_gemm_swizzled_scales,
            "fake_dtype": self._dtype,
        }

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a higher precision."""
        if dtype is None:
            dtype = self._dtype
        return _dequantize_mxfp8_impl(self, dtype)

    def dequantize_for_usage(
        self,
        usage: TensorUsage,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Dequantize the exact payload and scale selected for a GEMM usage.

        MXFP8 keeps independently scaled rowwise and columnwise representations.
        In particular, standard MXFP8 may also keep independently quantized
        payloads, while 32x32 MXFP8 shares one payload with two scale layouts.
        Calling :meth:`dequantize` and transposing the result therefore does not
        necessarily reproduce the operand selected by :meth:`get_data`.
        """
        if dtype is None:
            dtype = self._dtype

        data, scale_inv = self.get_data(usage)
        if data is None or scale_inv is None:
            raise RuntimeError(f"MXFP8 data for GEMM usage {usage!s} is not available")

        orig_shape = tuple(data.shape)
        _, m_dim, k_dim = self._flatten_to_2d(self._decode_quantized_data(data))
        matches_k_blocks = self._scale_matches_axis_k_blocks(
            scale_inv,
            m_dim=m_dim,
            k_dim=k_dim,
        )
        matches_m_blocks = self._scale_matches_axis_m_blocks(
            scale_inv,
            m_dim=m_dim,
            k_dim=k_dim,
        )

        if matches_k_blocks and not matches_m_blocks:
            result = self._dequantize_axis_k_blocks(data, scale_inv)
        elif matches_m_blocks and not matches_k_blocks:
            result = self._dequantize_axis_m_blocks(data, scale_inv)
        elif usage in (TensorUsage.LHS, TensorUsage.LHS_TRANS):
            # Ambiguous square/small layouts are resolved using the logical
            # direction selected by get_data(). LHS uses K-wise scaling and
            # LHS_TRANS is the transposed columnwise representation.
            result = self._dequantize_axis_k_blocks(data, scale_inv)
        elif usage in (TensorUsage.RHS, TensorUsage.RHS_TRANS):
            # RHS uses M-wise scaling and RHS_TRANS is the transposed rowwise
            # representation. Both are already laid out as GEMM operands.
            result = self._dequantize_axis_m_blocks(data, scale_inv)
        else:
            raise ValueError(f"Unsupported MXFP8 GEMM usage: {usage!r}")

        result = result.to(dtype)
        if len(orig_shape) == 0:
            return result.reshape([])
        return result.reshape(*orig_shape).contiguous()

    @staticmethod
    def _decode_scale_inv(scale_inv: torch.Tensor) -> torch.Tensor:
        """Decode E8M0 scale tensor to float32 values."""
        if scale_inv.dtype != torch.uint8:
            try:
                return scale_inv.to(torch.float32)
            except (TypeError, RuntimeError):
                pass

        exponent = scale_inv.to(torch.float32)
        return torch.pow(2.0, exponent - 127.0)

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], "MXFP8TensorStorage"]:
        """Prepare the tensor base for saving for backward"""
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        return tensors[4:]

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Get this Tensor's data."""
        if rowwise_data and columnwise_data:
            return self._rowwise_data, self._columnwise_data
        if rowwise_data:
            return self._rowwise_data
        if columnwise_data:
            return self._columnwise_data
        raise ValueError("No data to get, both rowwise_data and columnwise_data are False")

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._rowwise_data is not None:
            return self._rowwise_data.size(*args, **kwargs)
        return self._columnwise_data.size(*args, **kwargs)

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for tensor in (
            self._rowwise_data,
            self._rowwise_scale_inv,
            self._columnwise_data,
            self._columnwise_scale_inv,
        ):
            if tensor is not None:
                tensor.untyped_storage().resize_(0)

    def clear_wise(self, rowwise=False, colwise=False):
        tensors = []
        if rowwise:
            tensors.append(self._rowwise_data)
            tensors.append(self._rowwise_scale_inv)
        if colwise:
            tensors.append(self._columnwise_data)
            tensors.append(self._columnwise_scale_inv)
        for tensor in tensors:
            tensor.untyped_storage().resize_(0)

    def __repr__(self):
        data_rowwise = self.dequantize()
        # TODO @Muu 这里看着不太对的样子, 原生实现既如此
        return (
            "MXFP8TensorStorage("
            f"fp8_dtype={self._fp8_dtype}, "
            f"rowwise_scaled_data={data_rowwise}, "
            f"rowwise_scale_inv={self._rowwise_scale_inv}, "
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """
        For MXFP8, columnwise scaled output is only produced by x2
        scaling kernels, so this function only disables usages.
        """
        # Default usage is based on available data
        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None

        # Update row-scaled data
        if rowwise_usage:
            if self._rowwise_data is None:
                raise RuntimeError(
                    "Requested row-wise usage, but MXFP8Tensor is missing row-scaled FP8 data"
                )
            if self._rowwise_scale_inv is None:
                raise RuntimeError(
                    "Requested row-wise usage, but MXFP8Tensor is missing row-scaled scale-inverses"
                )
        else:
            self._rowwise_data = None
            self._rowwise_scale_inv = None

        # Update column-scaled data
        if columnwise_usage:
            if self._columnwise_data is None:
                raise RuntimeError(
                    "Requested column-wise usage, but MXFP8Tensor is missing column-scaled FP8 data"
                )
            if self._columnwise_scale_inv is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but MXFP8Tensor is missing column-scaled scale-inverses"
                )
        else:
            self._columnwise_data = None
            self._columnwise_scale_inv = None

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }


def _dequantize_mxfp8_impl(tensor: MXFP8TensorStorage, dtype: torch.dtype) -> torch.Tensor:
    """Shared MXFP8 dequantize implementation for forward and autograd apply."""
    from ...triton.mxfp8_dequantize import mxfp8_dequantize

    return mxfp8_dequantize(tensor, dtype)
