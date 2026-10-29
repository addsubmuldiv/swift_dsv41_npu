# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import math
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch_npu

from ...constants import MXFP4_BLOCK_SCALING_SIZE, TensorUsage
from ...quantized_tensor import QuantizedTensorStorage, transpose_mx_data

if TYPE_CHECKING:
    from ..mxfp4_tensor import MXFP4Quantizer


_NPU_ANTI_MX_QUANT_DST_TYPES = {
    torch.float16: 5,  # c10::ScalarType::Half
    torch.float32: 6,  # c10::ScalarType::Float
    torch.bfloat16: 15,  # c10::ScalarType::BFloat16
}


@lru_cache(maxsize=None)
def _get_mxfp4_lookup_table(device: torch.device) -> torch.Tensor:
    """Get the MXFP4 E2M1 lookup table for a device."""
    return torch.tensor(
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
        device=device,
        dtype=torch.float32,
    )


class _FromMXFP4Func(torch.autograd.Function):
    """Cast from MXFP4 storage-backed tensor to other dtype."""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],
        tensor: "MXFP4TensorStorage",
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = tensor._dtype
        return _dequantize_mxfp4_impl(tensor, dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return grad, None


class MXFP4TensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of MXFP4Tensor."""

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _rowwise_scale_inv: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]
    _quantizer: Optional["MXFP4Quantizer"]
    _fp4_dtype: torch.dtype
    _block_size: int

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp4_dtype: torch.dtype,
        quantizer: Optional["MXFP4Quantizer"],
        *args,
        block_size: int = MXFP4_BLOCK_SCALING_SIZE,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is MXFP4TensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._fp4_dtype = fp4_dtype
        instance._block_size = block_size
        return instance

    @property
    def row(self):
        return self._rowwise_data, self._rowwise_scale_inv

    @property
    def col(self):
        return self._columnwise_data, self._columnwise_scale_inv

    def get_data(self, usage):
        if usage == TensorUsage.LHS:
            return self._rowwise_data, self._rowwise_scale_inv
        if usage == TensorUsage.RHS_TRANS:
            return transpose_mx_data(self._rowwise_data, self._rowwise_scale_inv)
        if usage == TensorUsage.RHS:
            return self._columnwise_data, self._columnwise_scale_inv
        assert usage == TensorUsage.LHS_TRANS
        return transpose_mx_data(self._columnwise_data, self._columnwise_scale_inv)

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another MXFP4TensorStorage."""
        if not isinstance(src, MXFP4TensorStorage):
            raise TypeError("copy_from_storage expects MXFP4TensorStorage")
        if self._fp4_dtype != src._fp4_dtype:
            raise RuntimeError("MXFP4 dtype mismatch in copy_from_storage")
        if self._block_size != src._block_size:
            raise RuntimeError("MXFP4 block size mismatch in copy_from_storage")

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
            "fp4_dtype": self._fp4_dtype,
            "quantizer": self._quantizer,
            "block_size": self._block_size,
            "fake_dtype": self._dtype,
        }

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Dequantize to a higher precision."""
        if dtype is None:
            dtype = self._dtype
        return _dequantize_mxfp4_impl(self, dtype)

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

    def _decode_quantized_data(self, data: torch.Tensor) -> torch.Tensor:
        """Decode packed MXFP4 data to its logical values."""
        if data.dtype not in (torch.uint8, self._fp4_dtype):
            return data
        packed = data.contiguous().view(torch.uint8)
        table = _get_mxfp4_lookup_table(packed.device)
        low = table[(packed & 0x0F).long()]
        high = table[((packed >> 4) & 0x0F).long()]
        return torch.stack((low, high), dim=-1).flatten(-2)

    @staticmethod
    def _flatten_to_2d(data: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Flatten a tensor into [M, K]."""
        q_m, q_k = 1, 1
        if len(data.shape) >= 1:
            q_k = data.shape[-1]
        for i in range(len(data.shape) - 1):
            q_m *= data.shape[i]
        return data.reshape(q_m, q_k), q_m, q_k

    @staticmethod
    def _reshape_scale_inv(
        scale_inv: torch.Tensor, *, keep_first_dim: bool = False
    ) -> torch.Tensor:
        """Reshape decoded scale tensors into a 2D logical view."""
        if scale_inv.ndim == 0:
            return scale_inv.reshape(1, 1)
        if scale_inv.ndim == 1:
            return scale_inv.reshape(-1, 1)
        if scale_inv.ndim == 2:
            return scale_inv
        if not keep_first_dim:
            return scale_inv.reshape(-1, scale_inv.shape[-1])
        return scale_inv.reshape(scale_inv.shape[0], -1)

    @staticmethod
    def _reshape_logical_matrix(
        scale_matrix: torch.Tensor,
        rows: int,
        cols: int,
    ) -> torch.Tensor:
        """Flatten and rebuild a logical 2D scale matrix when enough data exists."""
        required_numel = rows * cols
        if scale_matrix.numel() < required_numel:
            return scale_matrix
        return scale_matrix.reshape(-1)[:required_numel].reshape(rows, cols)

    @classmethod
    def _coerce_scale_matrix_shape(
        cls,
        scale_matrix: torch.Tensor,
        *,
        logical_dim0: int,
        logical_dim1: int,
    ) -> torch.Tensor:
        if scale_matrix.shape[0] >= logical_dim0 and scale_matrix.shape[1] >= logical_dim1:
            return scale_matrix
        return cls._reshape_logical_matrix(scale_matrix, logical_dim0, logical_dim1)

    @classmethod
    def _coerce_rowwise_scale_shape(
        cls,
        scale_matrix: torch.Tensor,
        *,
        m_blocks: int,
        k_dim: int,
    ) -> torch.Tensor:
        if scale_matrix.shape[0] == m_blocks and scale_matrix.shape[1] >= k_dim:
            return scale_matrix[:, :k_dim]
        if scale_matrix.shape[0] == k_dim and scale_matrix.shape[1] >= m_blocks:
            return scale_matrix[:k_dim, :m_blocks].transpose(-2, -1).contiguous()
        return cls._reshape_logical_matrix(scale_matrix, m_blocks, k_dim)

    @staticmethod
    def _scale_matches_axis_k_blocks(
        scale_inv: torch.Tensor,
        *,
        m_dim: int,
        k_dim: int,
        block_size: int,
    ) -> bool:
        if scale_inv.ndim < 2:
            return False
        k_blocks = math.ceil(k_dim / block_size)
        return scale_inv.shape[0] == m_dim and math.prod(scale_inv.shape[1:]) >= k_blocks

    @staticmethod
    def _scale_matches_axis_m_blocks(
        scale_inv: torch.Tensor,
        *,
        m_dim: int,
        k_dim: int,
        block_size: int,
    ) -> bool:
        if scale_inv.ndim < 2:
            return False
        m_blocks = math.ceil(m_dim / block_size)
        if (
            scale_inv.ndim >= 3
            and scale_inv.shape[0] * scale_inv.shape[-1] >= m_blocks
            and scale_inv.shape[1] >= k_dim
        ):
            return True
        return (scale_inv.shape[0] == m_blocks and scale_inv.shape[1] >= k_dim) or (
            scale_inv.ndim == 2 and scale_inv.shape[0] == k_dim and scale_inv.shape[1] >= m_blocks
        )

    def _dequantize_axis_k_blocks(self, q: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
        """Dequantize data quantized along the last dimension in 1x32 blocks."""
        if q.dtype in (torch.uint8, self._fp4_dtype):
            q = q.contiguous().view(torch.uint8)
            table = _get_mxfp4_lookup_table(q.device)
            low = q & 0x0F
            high = (q >> 4) & 0x0F
            q = torch.stack((table[low.long()], table[high.long()]), dim=-1).flatten(-2)
        else:
            q = self._decode_quantized_data(q)
        q, q_m, q_k = self._flatten_to_2d(q)
        block_len = self._block_size
        scales_tiled_dim = math.ceil(q_k / block_len)
        scale_matrix = self._reshape_scale_inv(
            self._decode_scale_inv(scale_inv), keep_first_dim=True
        )
        scale_matrix = self._coerce_scale_matrix_shape(
            scale_matrix,
            logical_dim0=q_m,
            logical_dim1=scales_tiled_dim,
        )
        if scale_matrix.shape[0] < q_m or scale_matrix.shape[1] < scales_tiled_dim:
            raise RuntimeError(
                "MXFP4 rowwise scale tensor has insufficient logical shape for dequantize: "
                f"got {tuple(scale_matrix.shape)}, expected at least ({q_m}, {scales_tiled_dim})"
            )
        valid_scale = scale_matrix[:q_m, :scales_tiled_dim].contiguous()

        if q_k % block_len != 0:
            k_pad_amount = (block_len - (q_k % block_len)) % block_len
            q = torch.nn.functional.pad(q, (0, k_pad_amount, 0, 0), mode="constant", value=0)
        padded_m, padded_k = q.shape
        q_tiled = q.reshape(q_m, scales_tiled_dim, block_len)
        dq_scale = valid_scale.reshape(q_m, scales_tiled_dim, 1)
        result = q_tiled.to(torch.float32) * dq_scale
        if padded_m != q_m or padded_k != q_k:
            result = result.reshape(padded_m, padded_k)[:q_m, :q_k]
        return result

    def _dequantize_axis_m_blocks(self, q: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
        """Dequantize data quantized along the flattened leading dimension in 32x1 blocks."""
        q, q_m, q_k = self._flatten_to_2d(self._decode_quantized_data(q))
        block_len = self._block_size
        scales_tiled_dim = math.ceil(q_m / block_len)
        decoded_scale = self._decode_scale_inv(scale_inv)
        if (
            decoded_scale.ndim >= 3
            and decoded_scale.shape[0] * decoded_scale.shape[-1] >= scales_tiled_dim
            and decoded_scale.shape[1] >= q_k
        ):
            scale_matrix = decoded_scale.permute(0, 2, 1).reshape(-1, decoded_scale.shape[1])
        else:
            scale_matrix = self._reshape_scale_inv(decoded_scale, keep_first_dim=True)
        scale_matrix = self._coerce_rowwise_scale_shape(
            scale_matrix,
            m_blocks=scales_tiled_dim,
            k_dim=q_k,
        )
        if scale_matrix.shape[0] < scales_tiled_dim or scale_matrix.shape[1] < q_k:
            raise RuntimeError(
                "MXFP4 columnwise scale tensor has insufficient logical shape for dequantize: "
                f"got {tuple(scale_matrix.shape)}, "
                f"expected at least ({scales_tiled_dim}, {q_k})"
            )
        valid_scale = scale_matrix[:scales_tiled_dim, :q_k].contiguous()

        if q_m % block_len != 0:
            m_pad_amount = (block_len - (q_m % block_len)) % block_len
            q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_amount), mode="constant", value=0)
        padded_m, padded_k = q.shape
        q_tiled = q.reshape(scales_tiled_dim, block_len, q_k)
        dq_scale = valid_scale.reshape(scales_tiled_dim, 1, q_k)
        result = q_tiled.to(torch.float32) * dq_scale
        if padded_m != q_m or padded_k != q_k:
            result = result.reshape(padded_m, padded_k)[:q_m, :q_k]
        return result

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], "MXFP4TensorStorage"]:
        """Prepare the tensor base for saving for backward."""
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
            if tensor is not None:
                tensor.untyped_storage().resize_(0)

    def __repr__(self):
        return (
            "MXFP4TensorStorage("
            f"fp4_dtype={self._fp4_dtype}, "
            f"block_size={self._block_size}, "
            f"usages={self.get_usages()}"
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None

        if rowwise_usage:
            if self._rowwise_data is None:
                raise RuntimeError(
                    "Requested row-wise usage, but MXFP4Tensor is missing row-scaled FP4 data"
                )
            if self._rowwise_scale_inv is None:
                raise RuntimeError(
                    "Requested row-wise usage, but MXFP4Tensor is missing row-scaled scale-inverses"
                )
        else:
            self._rowwise_data = None
            self._rowwise_scale_inv = None

        if columnwise_usage:
            if self._columnwise_data is None:
                raise RuntimeError(
                    "Requested column-wise usage, but MXFP4Tensor is missing column-scaled FP4 data"
                )
            if self._columnwise_scale_inv is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but MXFP4Tensor is missing column-scaled scale-inverses"
                )
        else:
            self._columnwise_data = None
            self._columnwise_scale_inv = None

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor."""
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }

    @staticmethod
    def scale_shape_for_rowwise(shape, block_size: int = MXFP4_BLOCK_SCALING_SIZE):
        m_dim = math.prod(shape[:-1]) if len(shape) > 1 else 1
        k_dim = shape[-1] if len(shape) > 0 else 1
        return k_dim, math.ceil(m_dim / block_size)

    @staticmethod
    def scale_shape_for_columnwise(shape, block_size: int = MXFP4_BLOCK_SCALING_SIZE):
        m_dim = math.prod(shape[:-1]) if len(shape) > 1 else 1
        k_dim = shape[-1] if len(shape) > 0 else 1
        return m_dim, math.ceil(k_dim / block_size)


def _dequantize_mxfp4_impl(tensor: MXFP4TensorStorage, dtype: torch.dtype) -> torch.Tensor:
    """Shared MXFP4 dequantize implementation for forward and autograd apply."""
    logical_shape = getattr(tensor, "origin_shape", None)
    if tensor._rowwise_data is not None:
        orig_shape = (
            tuple(logical_shape) if logical_shape is not None else tuple(tensor._rowwise_data.shape)
        )
        result = torch_npu.npu_anti_mx_quant(
            tensor._rowwise_data,
            tensor._rowwise_scale_inv,
            axis=-1,
            dst_type=_NPU_ANTI_MX_QUANT_DST_TYPES[dtype],
            src_type=296,
        )
    else:
        assert tensor._columnwise_data is not None, "No data to dequantize"
        orig_shape = (
            tuple(logical_shape)
            if logical_shape is not None
            else tuple(tensor._columnwise_data.shape)
        )
        if (
            orig_shape
            and tensor._columnwise_data.dtype in (torch.uint8, tensor._fp4_dtype)
            and tensor._columnwise_data.shape[-1] == orig_shape[-1]
        ):
            # Ordinary columnwise storage is a transposed axis-K quantized pair.
            result = torch_npu.npu_anti_mx_quant(
                tensor._columnwise_data.t().contiguous(),
                tensor._columnwise_scale_inv.transpose(0, 1).contiguous(),
                axis=-1,
                dst_type=_NPU_ANTI_MX_QUANT_DST_TYPES[dtype],
                src_type=296,
            )
            result = result.t().contiguous()
        else:
            decoded_data = tensor._decode_quantized_data(tensor._columnwise_data)
            _, m_dim, k_dim = tensor._flatten_to_2d(decoded_data)
            if tensor._scale_matches_axis_m_blocks(
                tensor._columnwise_scale_inv,
                m_dim=m_dim,
                k_dim=k_dim,
                block_size=tensor._block_size,
            ):
                result = tensor._dequantize_axis_m_blocks(
                    decoded_data, tensor._columnwise_scale_inv
                )
            else:
                result = tensor._dequantize_axis_k_blocks(
                    decoded_data, tensor._columnwise_scale_inv
                )
    result = result.to(dtype)
    if len(orig_shape) == 0:
        return result.reshape([])
    return result.reshape(*orig_shape).contiguous()


__all__ = ["MXFP4TensorStorage", "_FromMXFP4Func"]
