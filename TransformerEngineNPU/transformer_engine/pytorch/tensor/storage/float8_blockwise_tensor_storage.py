# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import math

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from ...constants import TensorUsage
from ...quantized_tensor import QuantizedTensorStorage, transpose_quantized_tensor

if TYPE_CHECKING:
    from ..float8_blockwise_tensor import Float8BlockQuantizer


class _FromFloat8BlockwiseFunc(torch.autograd.Function):
    """Cast from blockwise FP8 storage-backed tensor to other dtype."""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],
        tensor: "Float8BlockwiseQTensorStorage",
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor._dtype
        return Float8BlockwiseQTensorStorage.dequantize(tensor, dtype=dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return grad, None


class Float8BlockwiseQTensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of Float8BlockwiseQTensor.

    Float8BlockwiseQTensor inherits from the PyTorch tensor class and this mixin
    class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.

    """

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Optional["Float8BlockQuantizer"]
    _fp8_dtype: torch.dtype
    _rowwise_scale_inv: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]
    _is_2D_scaled: bool

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: torch.dtype,
        quantizer: Optional["Float8BlockQuantizer"],
        is_2D_scaled: bool,
        *args,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is Float8BlockwiseQTensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._fp8_dtype = fp8_dtype
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._is_2D_scaled = is_2D_scaled

        return instance

    @property
    def row(self):
        return self._rowwise_data, self._rowwise_scale_inv

    @property
    def col(self):
        return self._columnwise_data, self._columnwise_scale_inv

    def get_data(self, usage: TensorUsage):
        if usage == TensorUsage.LHS:
            return self._rowwise_data, self._rowwise_scale_inv
        if usage == TensorUsage.RHS_TRANS:
            return transpose_quantized_tensor(self._rowwise_data, self._rowwise_scale_inv)
        if usage == TensorUsage.RHS:
            return self._columnwise_data, self._columnwise_scale_inv
        assert usage == TensorUsage.LHS_TRANS
        return self._columnwise_data, self._columnwise_scale_inv

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ):
            if t is not None:
                t.untyped_storage().resize_(0)

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another Float8BlockwiseQTensorStorage."""
        if not isinstance(src, Float8BlockwiseQTensorStorage):
            raise TypeError("copy_from_storage expects Float8BlockwiseQTensorStorage")
        if self._fp8_dtype != src._fp8_dtype:
            raise RuntimeError("FP8 dtype mismatch in copy_from_storage")
        if self._is_2D_scaled != src._is_2D_scaled:
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
            "is_2D_scaled": self._is_2D_scaled,
            "device": self.device,
            "fake_dtype": self._dtype,
        }

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], "Float8BlockwiseQTensorStorage"]:
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

    @staticmethod
    def _flatten_to_2d(data: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Flatten tensor to [M, K] using the last dimension as K."""
        q_M, q_K = 1, 1
        if len(data.shape) >= 1:
            q_K = data.shape[-1]
        for i in range(len(data.shape) - 1):
            q_M *= data.shape[i]
        return data.reshape(q_M, q_K), q_M, q_K

    def _get_2d_block_dims(self, is_rowwise_data: bool) -> Tuple[int, int]:
        """Get logical block dimensions for the current blockwise layout."""
        if self._quantizer is None:
            return 128, 128
        block_dim = self._quantizer.colwise_dim if is_rowwise_data else self._quantizer.rowwise_dim
        return block_dim["row_block_size"], block_dim["col_block_size"]

    @staticmethod
    def _format_2d_scales(
        q_M: int,
        q_K: int,
        scales: torch.Tensor,
        m_block_len: int,
        k_block_len: int,
    ) -> torch.Tensor:
        """Slice padded scales down to the logical 2D block layout."""
        derived_scale_m_shape = math.ceil(q_M / m_block_len)
        derived_scale_k_shape = math.ceil(q_K / k_block_len)
        scale_M, scale_K = scales.shape
        if scale_M < derived_scale_m_shape or scale_K < derived_scale_k_shape:
            raise RuntimeError(
                "Float8 blockwise scale tensor has insufficient logical shape "
                f"for dequantize: got {tuple(scales.shape)}, "
                f"expected at least ({derived_scale_m_shape}, {derived_scale_k_shape})"
            )
        if derived_scale_m_shape == scale_M and derived_scale_k_shape == scale_K:
            return scales
        return scales[:derived_scale_m_shape, :derived_scale_k_shape].contiguous()

    @staticmethod
    def _transpose_last_axis_to_front(data: torch.Tensor) -> torch.Tensor:
        """Move the leading matrix axis to the end to undo rowwise transposed storage."""
        if data.dim() < 2:
            return data
        permute_dims = list(range(1, data.dim()))
        permute_dims.append(0)
        return torch.permute(data, tuple(permute_dims)).contiguous()

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Construct plain PyTorch tensor from Float8BlockwiseQTensor"""
        if dtype is None:
            dtype = self._dtype
        transpose_output = False
        if self._rowwise_data is not None:
            q = self._rowwise_data
            scale_inv = self._rowwise_scale_inv
            m_block_len, k_block_len = self._get_2d_block_dims(is_rowwise_data=True)
            q, q_M, q_K = self._flatten_to_2d(q)
            orig_shape = tuple(self._rowwise_data.shape)
            # For 1x128 rowwise scaling, quantization is applied to tensor_2d.T.
            transpose_output = m_block_len == 1
        else:
            assert self._columnwise_data is not None, "No data to dequantize"
            q = self._columnwise_data
            scale_inv = self._columnwise_scale_inv
            m_block_len, k_block_len = self._get_2d_block_dims(is_rowwise_data=False)
            q, q_M, q_K = self._flatten_to_2d(q)
            orig_shape = tuple(self._columnwise_data.shape)

        formatted_scales = self._format_2d_scales(q_M, q_K, scale_inv, m_block_len, k_block_len)
        m_tiles, k_tiles = formatted_scales.shape
        unpadded_m, unpadded_k = q_M, q_K
        if q_M % m_block_len != 0 or q_K % k_block_len != 0:
            m_pad_amount = (m_block_len - (q_M % m_block_len)) % m_block_len
            k_pad_amount = (k_block_len - (q_K % k_block_len)) % k_block_len
            q = torch.nn.functional.pad(
                q, (0, k_pad_amount, 0, m_pad_amount), mode="constant", value=0
            ).contiguous()
        padded_M, padded_K = q.shape
        q_tiled = q.reshape(m_tiles, m_block_len, k_tiles, k_block_len)

        result = q_tiled.to(torch.float32) * formatted_scales.to(torch.float32).view(
            m_tiles, 1, k_tiles, 1
        )
        result = result.view(padded_M, padded_K).to(dtype)
        if padded_M != unpadded_m or padded_K != unpadded_k:
            result = result[:unpadded_m, :unpadded_k]
        if len(orig_shape) == 0:
            result = result.reshape([])
        else:
            result = result.reshape(*orig_shape).contiguous()
        if transpose_output:
            return self._transpose_last_axis_to_front(result)
        return result

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._rowwise_data is not None:
            return self._rowwise_data.size(*args, **kwargs)
        dims = list(self._columnwise_data.size(*args, **kwargs))
        reordered = []
        for i in range(1, len(dims)):
            reordered.append(dims[i])
        reordered.append(dims[0])
        return torch.Size(reordered)

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("Float8BlockwiseQTensorStorage has no data!")

    def __repr__(self):
        if self._rowwise_data is not None:
            data = self.dequantize()
            descriptor = "rowwise"
        else:
            data = self.dequantize()
            descriptor = "columnwise"
        return (
            "Float8BlockwiseQTensorStorage("
            f"fp8_dtype={self._fp8_dtype}, "
            f"{descriptor}_scaled_data={data})"
        )

    def _create_columnwise(self):
        """Create columnwise data and scales from 2D-scaled rowwise storage."""
        assert self._is_2D_scaled, "Cannot create columnwise data when not using 2D scaling."
        assert self._rowwise_data is not None and self._rowwise_scale_inv is not None, (
            "Cannot create columnwise data because rowwise data is None."
        )
        if self._quantizer is not None and self._quantizer.block_scaling_dim not in (
            2,
            (2, 2),
        ):
            raise RuntimeError("Cannot create columnwise data when not using 2D block scaling.")

        self._columnwise_data = self._rowwise_data.clone(memory_format=torch.contiguous_format)
        self._columnwise_scale_inv = self._rowwise_scale_inv.clone(
            memory_format=torch.contiguous_format
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """update_usage can be used to clear out one of two possible copies of the data."""
        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None
        assert columnwise_usage or rowwise_usage, (
            "Must retain some data either columnwise or rowwise"
        )

        if columnwise_usage and rowwise_usage:
            if self._columnwise_data is None or self._columnwise_scale_inv is None:
                self._create_columnwise()
            assert (
                self._rowwise_data is not None
                and self._rowwise_scale_inv is not None
                and self._columnwise_data is not None
                and self._columnwise_scale_inv is not None
            ), "Cannot update to rowwise and columnwise usage."
            return

        if rowwise_usage:
            assert self._rowwise_data is not None and self._rowwise_scale_inv is not None, (
                "Cannot update to rowwise usage."
            )
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            return
        if columnwise_usage:
            assert self._columnwise_data is not None and self._columnwise_scale_inv is not None, (
                "Cannot update to columnwise usage."
            )
            self._rowwise_data = None
            self._rowwise_scale_inv = None
            return

        return

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }
