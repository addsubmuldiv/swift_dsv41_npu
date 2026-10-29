# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import MutableSequence, Optional, Tuple, TypedDict, Union

import torch
import torch_npu

from transformer_engine.common.recipe import (
    Float8BlockScaling,
    Recipe,
)
from ._quantization_helpers import (
    _IdentityFunc,
    _pack_fp8_tensor_for_pickle,
    _pack_tensor_for_pickle,
    _restore_fp8_tensor_from_pickle,
)
from .storage.float8_blockwise_tensor_storage import (
    Float8BlockwiseQTensorStorage,
    _FromFloat8BlockwiseFunc,
)
from ..constants import TensorUsage
from ..quantized_tensor import QuantizedTensor, Quantizer
from ..utils import view_as_n_dim

aten = torch.ops.aten


class BlockDim(TypedDict):
    row_block_size: int
    col_block_size: int


ROWWISE_BLOCK_DIM = BlockDim(row_block_size=1, col_block_size=128)
COLWISE_BLOCK_DIM = BlockDim(row_block_size=128, col_block_size=128)


def get_block_dim(scaling_dim):
    if scaling_dim == 1:
        return ROWWISE_BLOCK_DIM
    elif scaling_dim == 2:
        return COLWISE_BLOCK_DIM
    else:
        raise RuntimeError(f"not supported block scaling dim: {scaling_dim}")


class Float8BlockQuantizer(Quantizer):
    """Builder class for FP8 tensors with MX block scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized by
    dividing them into groups of 32 elements, each scaled and cast
    separately using current data.

    """

    dtype: torch.dtype

    def __init__(
        self,
        fp8_dtype: torch.dtype,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
        amax_epsilon: float = 0.0,
        force_pow_2_scales: bool = True,
        block_scaling_dim: Union[int, Tuple[int, int]] = 2,
        columnwise_group_quant_lhs_trans: bool = False,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype
        self.block_len = 128
        self.block_scaling_dim = block_scaling_dim
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon
        self.columnwise_group_quant_lhs_trans = columnwise_group_quant_lhs_trans

        if isinstance(block_scaling_dim, tuple):
            rowwise_dim, colwise_dim = block_scaling_dim
        else:
            rowwise_dim, colwise_dim = block_scaling_dim, block_scaling_dim

        self.rowwise_dim = get_block_dim(rowwise_dim)
        self.colwise_dim = get_block_dim(colwise_dim)

    def copy(self) -> Float8BlockQuantizer:
        """Create shallow copy"""
        quantizer = Float8BlockQuantizer(
            fp8_dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            amax_epsilon=self.amax_epsilon,
            force_pow_2_scales=self.force_pow_2_scales,
            block_scaling_dim=self.block_scaling_dim,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.rowwise_dim = self.rowwise_dim
        quantizer.colwise_dim = self.colwise_dim
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        quantizer.columnwise_group_quant_lhs_trans = self.columnwise_group_quant_lhs_trans
        return quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: Float8BlockwiseQTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8BlockwiseQTensor:
        if src is None:
            return src
        if not self.rowwise_usage and not self.columnwise_usage:
            return src
        col_data, col_scale, row_data, row_scale = self._quantize_data(src)
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        dst._columnwise_data = col_data
        dst._columnwise_scale_inv = col_scale
        dst._rowwise_data = row_data
        dst._rowwise_scale_inv = row_scale
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> Float8BlockwiseQTensor:
        """Quantize tensor implementation"""
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor
        col_data, col_scale, row_data, row_scale = self._quantize_data(tensor)
        return Float8BlockwiseQTensor(
            tensor.shape,
            tensor.dtype,
            rowwise_data=row_data,
            rowwise_scale_inv=row_scale,
            columnwise_data=col_data,
            columnwise_scale_inv=col_scale,
            fp8_dtype=self.dtype,
            quantizer=self,
            is_2D_scaled=True,
            device=tensor.device,
        )

    def _quantize_data(self, tensor: torch.Tensor):
        col_data, col_scale, row_data, row_scale = None, None, None, None
        tensor_2d = view_as_n_dim(tensor)
        if self.columnwise_usage:
            col_data, col_scale = torch_npu.npu_dynamic_block_quant(
                tensor_2d.T if self.block_scaling_dim == 1 else tensor_2d,
                dst_type=self.dtype,
                **self.colwise_dim,
            )
        if self.rowwise_usage:
            row_data, row_scale = torch_npu.npu_dynamic_block_quant(
                tensor_2d,
                dst_type=self.dtype,
                **self.rowwise_dim,
            )
        return col_data, col_scale, row_data, row_scale

    def _quantize_grouped_data(self, tensor: torch.Tensor, group_list):
        col_data, col_scale, row_data, row_scale = None, None, None, None

        if self.columnwise_usage:
            quant_tensor = tensor
            scale_group_dim = -2
            if self.colwise_dim == ROWWISE_BLOCK_DIM:
                quant_tensor = tensor.transpose(-1, -2)
                scale_group_dim = -1

            col_data, col_scale = torch_npu.npu_dynamic_block_quant(
                quant_tensor,
                dst_type=self.dtype,
                **self.colwise_dim,
            )
            if self.columnwise_use_group_quant:
                group_ends = (
                    group_list.tolist() if isinstance(group_list, torch.Tensor) else group_list
                )
                scale_parts = []
                start = 0
                separator_index = [slice(None)] * col_scale.ndim
                separator_index[scale_group_dim] = slice(0, 1)
                separator = torch.zeros_like(col_scale[tuple(separator_index)])

                for group_end in group_ends:
                    end = min(int(group_end) // self.block_len, col_scale.shape[scale_group_dim])
                    scale_index = [slice(None)] * col_scale.ndim
                    scale_index[scale_group_dim] = slice(start, end)
                    scale_parts.append(col_scale[tuple(scale_index)])
                    scale_parts.append(separator)
                    start = end

                col_scale = torch.cat(scale_parts, dim=scale_group_dim)
                if self.columnwise_group_quant_lhs_trans:
                    col_data = col_data.transpose(-1, -2).contiguous()
                    col_scale = col_scale.transpose(-1, -2).contiguous()

        if self.rowwise_usage:
            row_data, row_scale = torch_npu.npu_dynamic_block_quant(
                tensor,
                dst_type=self.dtype,
                **self.rowwise_dim,
            )
        return col_data, col_scale, row_data, row_scale

    def update_grouped_quantized(self, src, dst, group_list, *, noop_flag=None):
        if src is None:
            return src
        if not self.rowwise_usage and not self.columnwise_usage:
            return src

        col_data, col_scale, row_data, row_scale = self._quantize_grouped_data(src, group_list)
        dst.logical_shape = tuple(src.shape)
        dst.columnwise_data = col_data
        dst.columnwise_scale_inv = col_scale
        dst.rowwise_data = row_data
        dst.rowwise_scale_inv = row_scale
        dst.scale_inv = row_scale
        return dst

    def grouped_quantize_impl(self, group_list, tensor: torch.Tensor):
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor

        col_data, col_scale, row_data, row_scale = self._quantize_grouped_data(tensor, group_list)
        # pylint: disable=cyclic-import
        from .grouped_tensor import GroupedTensor

        return GroupedTensor(
            shape=tensor.shape,
            dtype=self.dtype,
            num_tensors=len(group_list),
            quantizer=self,
            columnwise_data=col_data,
            columnwise_scale_inv=col_scale,
            data=row_data,
            scale_inv=row_scale,
        )

    def dequantize_grouped_tensor(self, tensor, *, dtype=None, columnwise=False):
        """Dequantize one orientation of packed blockwise-scaled grouped buffers."""
        storage = Float8BlockwiseQTensorStorage(
            rowwise_data=None if columnwise else tensor.rowwise_data,
            rowwise_scale_inv=None if columnwise else tensor.rowwise_scale_inv,
            columnwise_data=tensor.columnwise_data if columnwise else None,
            columnwise_scale_inv=tensor.columnwise_scale_inv if columnwise else None,
            fp8_dtype=self.dtype,
            quantizer=self,
            is_2D_scaled=True,
            fake_dtype=tensor.fake_dtype,
        )
        return storage.dequantize(dtype=dtype if dtype is not None else tensor.fake_dtype)

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % self.block_len != 0:
            return False
        if math.prod(inp.shape[:-1]) % self.block_len != 0:
            return False
        return True

    def get_scale_shape(
        self, shape: MutableSequence[int], *, columnwise: bool
    ) -> tuple[int, int, list]:
        """Logical scale tensor shape for blockwise data."""
        dim0 = math.prod(shape[:-1]) if len(shape) > 1 else 1
        dim1 = shape[-1] if len(shape) > 0 else 1
        if self.block_scaling_dim == 2:
            scale_dim0 = math.ceil(dim0 / self.block_len)
            scale_dim1 = math.ceil(dim1 / self.block_len)
            if columnwise:
                return scale_dim1, scale_dim0
            return scale_dim0, scale_dim1
        if columnwise:
            return math.ceil(dim0 / self.block_len), dim1
        return dim1, math.ceil(dim0 / self.block_len)

    @staticmethod
    def get_columnwise_shape(shape: Iterable[int]) -> tuple[int, ...]:
        """Columnwise data is stored transposed relative to logical shape."""
        shape = tuple(shape)
        if len(shape) == 0:
            return ()
        return shape[-1], *shape[:-1]

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "Float8BlockwiseQTensor":
        if device is None:
            device = torch.device("npu" if hasattr(torch, "npu") else "cpu")
        shape = tuple(shape)
        rowwise_data = None
        rowwise_scale_inv = None
        columnwise_data = None
        columnwise_scale_inv = None
        if self.rowwise_usage:
            rowwise_shape = (
                self.get_columnwise_shape(shape) if self.block_scaling_dim == 1 else shape
            )
            rowwise_data = torch.empty(
                rowwise_shape, dtype=self.dtype, device=device, pin_memory=pin_memory
            )
            rowwise_scale_inv = torch.empty(
                self.get_scale_shape(shape, columnwise=False),
                dtype=torch.float32,
                device=device,
                pin_memory=pin_memory,
            )
        if self.columnwise_usage:
            columnwise_data = torch.empty(
                shape, dtype=self.dtype, device=device, pin_memory=pin_memory
            )
            columnwise_scale_inv = torch.empty(
                self.get_scale_shape(shape, columnwise=True),
                dtype=torch.float32,
                device=device,
                pin_memory=pin_memory,
            )
        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=self.dtype,
            quantizer=self,
            is_2D_scaled=True,
            requires_grad=requires_grad,
            device=device,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return Float8BlockScaling


class Float8BlockwiseQTensor(Float8BlockwiseQTensorStorage, QuantizedTensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    """

    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: torch.dtype,
        quantizer: Quantizer,
        is_2D_scaled: bool,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp8_dtype,
            quantizer,
            is_2D_scaled,
            *args,
            **kwargs,
        )

        return instance

    def __repr__(self, *, tensor_contents=None):
        return (
            f"Float8BlockwiseQTensor(fp8_dtype={self._fp8_dtype},"
            f" is_2D_scaled={self._is_2D_scaled},"
            f" data={self.dequantize()})"
        )

    @property
    def shape(self):
        """Return the shape of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.shape
        if self._columnwise_data is not None:
            return self._columnwise_data.shape
        return torch.Tensor.size(self)

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA/NPU-backed device."""
        if self._rowwise_data is not None:
            return self._rowwise_data.is_npu
        if self._columnwise_data is not None:
            return self._columnwise_data.is_npu
        raise RuntimeError("Float8BlockwiseQTensor has no data!")

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Construct plain PyTorch tensor from Float8BlockwiseQTensor."""
        if dtype is None:
            dtype = self.dtype
        if torch.is_grad_enabled():
            result = _FromFloat8BlockwiseFunc.apply(self, dtype)
        else:
            result = _FromFloat8BlockwiseFunc.forward(None, self, dtype)
        if tuple(result.shape) != tuple(self.origin_shape):
            result = result.reshape(self.origin_shape)
        return result

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8BlockwiseQTensor:
        """Update blockwise FP8 data in-place from a dense tensor."""
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        return super().quantize_(tensor, noop_flag=noop_flag)

    def view(self, *shape):
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape):
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def detach(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return Float8BlockwiseQTensor.make_like(self)

    def clone(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        rowwise_data = None
        if self._rowwise_data is not None:
            rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
                "rowwise_scale_inv": (
                    self._rowwise_scale_inv.detach().clone()
                    if self._rowwise_scale_inv is not None
                    else None
                ),
                "columnwise_scale_inv": (
                    self._columnwise_scale_inv.detach().clone()
                    if self._columnwise_scale_inv is not None
                    else None
                ),
            },
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Serialize blockwise FP8 payload and scale metadata."""
        return (
            _make_float8_blockwise_tensor_in_reduce_ex,
            (
                torch.Size(self.origin_shape),
                _pack_fp8_tensor_for_pickle(self._rowwise_data),
                _pack_tensor_for_pickle(self._rowwise_scale_inv),
                _pack_fp8_tensor_for_pickle(self._columnwise_data),
                _pack_tensor_for_pickle(self._columnwise_scale_inv),
                self._fp8_dtype,
                self.dtype,
                self._quantizer,
                self._is_2D_scaled,
            ),
        )

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: torch.dtype,
        dtype: torch.dtype,
        quantizer: Optional[Quantizer],
        is_2D_scaled: bool,
    ) -> Float8BlockwiseQTensor:
        """Reconstruct a blockwise FP8 tensor from an older pickle payload."""
        return _make_float8_blockwise_tensor_in_reduce_ex(
            shape,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp8_dtype,
            dtype,
            quantizer,
            is_2D_scaled,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == aten.copy_.default:
            dst, src = args[0], args[1]
            if isinstance(dst, Float8BlockwiseQTensor) and isinstance(src, Float8BlockwiseQTensor):
                src_usages = src.get_usages()
                dst_usages = dst.get_usages()
                if all(not enabled or dst_usages[name] for name, enabled in src_usages.items()):
                    dst.copy_from_storage(src)
                    dst.update_usage(
                        rowwise_usage=src_usages["rowwise"],
                        columnwise_usage=src_usages["columnwise"],
                    )
                    dst.origin_shape = src.origin_shape
                    dst.dtype = src.dtype
                    return dst

        if func == aten.view.default:
            tensor = args[0]
            return tensor.view(*args[1])

        if func == aten.as_strided.default:
            tensor = args[0]
            shape = args[1]
            strides = args[2]
            if (
                len(shape) == len(strides) == 2
                and tuple(strides) == (shape[-1], 1)
                and tuple(shape) == tuple(tensor.size())
            ):
                return Float8BlockwiseQTensor.make_like(tensor)

        if func == aten.slice.Tensor:
            tensor = args[0]
            dim = args[1]
            start = args[2]
            end = args[3]
            step = args[4] if len(args) > 4 else 1
            if start == 0 and end == tensor.size(dim) and step == 1:
                return Float8BlockwiseQTensor.make_like(tensor)

        if func == aten.new_zeros.default:
            tensor = args[0]
            shape = torch.Size(args[1])
            out = tensor._quantizer.make_empty(
                shape=shape,
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=False,
            )
            for internal in (
                out._rowwise_data,
                out._rowwise_scale_inv,
                out._columnwise_data,
                out._columnwise_scale_inv,
            ):
                if internal is not None:
                    internal.zero_()
            return out

        return super().__torch_dispatch__(func, types, args, kwargs)

    def untyped_storage(self) -> torch.UntypedStorage:
        """Return the rowwise or columnwise data storage backing this tensor."""
        data = self._rowwise_data if self._rowwise_data is not None else self._columnwise_data
        if data is not None:
            return data.untyped_storage()
        return torch.UntypedStorage(0, device=self.device)

    def _build_default_quantizer(self) -> Optional[Quantizer]:
        """Build default quantizer for the tensor"""
        return Float8BlockQuantizer(fp8_dtype=self.dtype)

    def matmul(self, B: Float8BlockwiseQTensor, usage: TensorUsage, usage_b, out_dtype):
        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)
        out = torch_npu.npu_quant_matmul(
            a,
            b,
            b_scale,
            pertoken_scale=a_scale,
            output_dtype=out_dtype,
            group_sizes=[1, 128, 128],
        )
        if len(out.shape) == 2:
            return out
        return out.view(*self.origin_shape[:-1], *out.shape[1:])

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8BlockwiseQTensor:
        """Return tensor with contiguous buffers in the requested format."""
        if self._rowwise_data is not None and not self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            pass
        elif self._columnwise_data is not None and not self._columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            pass
        elif self._rowwise_scale_inv is not None and not self._rowwise_scale_inv.is_contiguous():
            pass
        elif (
            self._columnwise_scale_inv is not None
            and not self._columnwise_scale_inv.is_contiguous()
        ):
            pass
        else:
            return self

        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": (
                    self._rowwise_data.contiguous(memory_format=memory_format)
                    if self._rowwise_data is not None
                    else None
                ),
                "columnwise_data": (
                    self._columnwise_data.contiguous(memory_format=memory_format)
                    if self._columnwise_data is not None
                    else None
                ),
                "rowwise_scale_inv": (
                    self._rowwise_scale_inv.contiguous()
                    if self._rowwise_scale_inv is not None
                    else None
                ),
                "columnwise_scale_inv": (
                    self._columnwise_scale_inv.contiguous()
                    if self._columnwise_scale_inv is not None
                    else None
                ),
            },
        )

    def _view_or_reshape(self, shape, *, reshape: bool):
        """Return a blockwise tensor view when layout still matches, else dense fallback."""
        cur_shape = torch.Size(self.origin_shape)
        shape = _canonicalize_shape(cur_shape, shape)
        if tuple(shape) == tuple(cur_shape):
            return self

        if not _supports_blockwise_shape(self, cur_shape, shape):
            op_name = "reshape" if reshape else "view"
            warnings.warn(
                "Float8BlockwiseQTensor does not support "
                f"{op_name} that changes the required blockwise layout "
                f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)}). "
                f"Falling back to dequantize().{op_name}(...) in higher precision.",
                stacklevel=2,
            )
            dense = self.dequantize()
            return dense.reshape(*shape) if reshape else dense.view(*shape)

        new_rowwise_data = None
        new_columnwise_data = None
        if self._rowwise_data is not None:
            rowwise_shape = _rowwise_storage_shape(self, shape)
            new_rowwise_data = (
                self._rowwise_data.reshape(*rowwise_shape)
                if reshape
                else self._rowwise_data.view(*rowwise_shape)
            )
        if self._columnwise_data is not None:
            new_columnwise_data = (
                self._columnwise_data.reshape(*shape)
                if reshape
                else self._columnwise_data.view(*shape)
            )

        return Float8BlockwiseQTensor(
            shape,
            self.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=self._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=self._columnwise_scale_inv,
            fp8_dtype=self._fp8_dtype,
            quantizer=self._quantizer,
            is_2D_scaled=self._is_2D_scaled,
            device=self.device,
        )


def _make_float8_blockwise_tensor_in_reduce_ex(
    shape: torch.Size,
    rowwise_data: Optional[torch.Tensor],
    rowwise_scale_inv: Optional[torch.Tensor],
    columnwise_data: Optional[torch.Tensor],
    columnwise_scale_inv: Optional[torch.Tensor],
    fp8_dtype: torch.dtype,
    dtype: torch.dtype,
    quantizer: Optional[Quantizer],
    is_2D_scaled: bool,
) -> Float8BlockwiseQTensor:
    """Reconstruct a ``Float8BlockwiseQTensor`` from its pickle payload."""
    rowwise_data = _restore_fp8_tensor_from_pickle(rowwise_data, fp8_dtype)
    columnwise_data = _restore_fp8_tensor_from_pickle(columnwise_data, fp8_dtype)
    device = None
    if rowwise_data is not None:
        device = rowwise_data.device
    elif columnwise_data is not None:
        device = columnwise_data.device
    return Float8BlockwiseQTensor(
        shape,
        dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=fp8_dtype,
        quantizer=quantizer,
        is_2D_scaled=is_2D_scaled,
        device=device,
    )


def _reshape_grad_to_input(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reshape view/reshape gradients back to the input tensor shape."""
    if isinstance(grad, QuantizedTensor) and hasattr(grad, "_view_or_reshape"):
        return grad._view_or_reshape(shape, reshape=True)
    return grad.reshape(shape)


class _ViewFunc(torch.autograd.Function):
    """Autograd-aware view for Float8BlockwiseQTensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        ctx.shape = torch.Size(tensor.origin_shape)
        if shape is None:
            return tensor.detach()
        return tensor._view_or_reshape(shape, reshape=False)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        return _reshape_grad_to_input(grad, ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Autograd-aware reshape for Float8BlockwiseQTensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        ctx.shape = torch.Size(tensor.origin_shape)
        if shape is None:
            return tensor.detach()
        return tensor._view_or_reshape(shape, reshape=True)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        return _reshape_grad_to_input(grad, ctx.shape), None


def _canonicalize_shape(cur_shape, shape) -> torch.Size:
    """Normalize view/reshape arguments and resolve a single inferred dim."""
    if not isinstance(shape, Iterable):
        shape = [shape]
    elif len(shape) == 1 and isinstance(shape[0], Iterable):
        shape = list(shape[0])
    else:
        shape = list(shape)

    infer_index = None
    known_product = 1
    for i, dim in enumerate(shape):
        if dim == -1:
            if infer_index is not None:
                raise RuntimeError("Only one dimension can be inferred")
            infer_index = i
            continue
        known_product *= dim

    if infer_index is not None:
        inferred = math.prod(cur_shape) // known_product
        shape[infer_index] = inferred

    return torch.Size(shape)


def _supports_blockwise_shape(
    tensor: Float8BlockwiseQTensor,
    cur_shape: torch.Size,
    new_shape: torch.Size,
) -> bool:
    """Whether blockwise layout can be reused for the requested logical shape."""
    block_scaling_dim = getattr(tensor._quantizer, "block_scaling_dim", 2)
    if block_scaling_dim == 1:
        return len(new_shape) >= 1 and len(cur_shape) >= 1 and new_shape[-1] == cur_shape[-1]
    return (
        len(new_shape) >= 2
        and len(cur_shape) >= 2
        and new_shape[-1] == cur_shape[-1]
        and new_shape[-2] == cur_shape[-2]
    )


def _rowwise_storage_shape(
    tensor: Float8BlockwiseQTensor,
    logical_shape: torch.Size,
) -> torch.Size:
    """Map logical shape to rowwise buffer shape for blockwise storage."""
    block_scaling_dim = getattr(tensor._quantizer, "block_scaling_dim", 2)
    if block_scaling_dim == 1:
        return torch.Size([logical_shape[-1], *logical_shape[:-1]])
    return logical_shape


__all__ = ["Float8BlockQuantizer", "Float8BlockwiseQTensor"]
