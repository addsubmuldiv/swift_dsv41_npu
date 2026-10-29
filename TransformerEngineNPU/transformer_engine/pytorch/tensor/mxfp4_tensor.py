# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from dataclasses import replace
import math
import warnings
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import torch
import torch_npu

from transformer_engine.common.recipe import MXFP4BlockScaling, Recipe
from transformer_engine.common.recipe.base import QParams
from .grouped_tensor import GroupedTensor

from ..constants import MXFP4_BLOCK_SCALING_SIZE
from ..quantized_tensor import QuantizedTensor, Quantizer, transpose_mx_data
from ..utils import devices_match, view_as_n_dim
from ._quantization_helpers import _IdentityFunc
from .utils import noop_requested
from .storage.mxfp4_tensor_storage import MXFP4TensorStorage, _FromMXFP4Func


def _canonicalize_shape(cur_shape, shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
        shape = tuple(shape[0])
    else:
        shape = tuple(shape)
    if shape.count(-1) > 1:
        raise RuntimeError(f"only one dimension can be inferred, got shape={shape}")
    if -1 not in shape:
        return torch.Size(shape)
    known = 1
    for dim in shape:
        if dim != -1:
            known *= dim
    numel = math.prod(cur_shape)
    if known == 0 or numel % known != 0:
        raise RuntimeError(f"shape {shape} is invalid for input of size {numel}")
    return torch.Size(numel // known if dim == -1 else dim for dim in shape)


@lru_cache
def hadamard16(device, dtype):
    H = torch.ones((1, 1), device=device, dtype=dtype)
    while H.shape[0] < 16:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    H = H * 0.25  # 1 / sqrt(16)
    signs = torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        device=device,
        dtype=dtype,
    )
    return H, signs


def rht_last_dim(x):
    """TE-style 16-wide RHT on the last dimension."""
    assert x.shape[-1] % 16 == 0

    H, signs = hadamard16(x.device, x.dtype)

    orig_shape = x.shape
    x16 = x.contiguous().reshape(-1, 16)

    y16 = (x16 * signs) @ H
    return y16.reshape(orig_shape)


class MXFP4Quantizer(Quantizer):
    """Builder class for MXFP4 tensors with block scaling."""

    def __init__(
        self,
        fp4_dtype: Optional[torch.dtype] = None,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
        block_size: int = MXFP4_BLOCK_SCALING_SIZE,
        qparams: Optional[QParams] = None,
        with_rht: Optional[bool] = None,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.fp4_dtype = fp4_dtype
        self.block_size = block_size
        if qparams is None:
            qparams = QParams(
                power_2_scale=True,
                random_hadamard_transform=bool(with_rht),
            )
        elif with_rht is not None and qparams.random_hadamard_transform != with_rht:
            qparams = replace(qparams, random_hadamard_transform=with_rht)
        self.qparams = qparams
        self.with_rht = self.qparams.random_hadamard_transform

    @property
    def dtype(self) -> torch.dtype:
        return torch_npu.float4_e2m1fn_x2

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        self.fp4_dtype = value

    def copy(self) -> "MXFP4Quantizer":
        """Create shallow copy."""
        quantizer = MXFP4Quantizer(
            fp4_dtype=self.fp4_dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            block_size=self.block_size,
            qparams=self.qparams,
            with_rht=self.with_rht,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        return quantizer

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rowwise_usage={self.rowwise_usage}, "
            f"columnwise_usage={self.columnwise_usage}, "
            f"internal={self.internal}, "
            f"with_rht={self.with_rht}, "
            ")"
        )

    def _quantize_tensor(self, tensor: torch.Tensor, with_rht: bool = False):
        tensor_2d = view_as_n_dim(tensor)
        col_data, col_scale, row_data, row_scale = None, None, None, None
        if self.columnwise_usage:
            if with_rht:
                # 先做转置，再做rht，沿尾轴做rht
                quant_tensor = rht_last_dim(tensor_2d.t())
            else:
                quant_tensor = tensor_2d.t()
            col_data, col_scale = torch_npu.npu_dynamic_mx_quant(quant_tensor)
            col_data, col_scale = col_data.t(), col_scale.transpose(0, 1)
        if self.rowwise_usage:
            row_data, row_scale = torch_npu.npu_dynamic_mx_quant(tensor_2d)
        return col_data, col_scale, row_data, row_scale

    @staticmethod
    def _group_list_for_quant(group_list, device: torch.device):
        if isinstance(group_list, torch.Tensor):
            if group_list.device != device:
                group_list = group_list.to(device=device)
            return group_list.to(torch.int32)
        return torch.tensor(group_list, device=device, dtype=torch.int32)

    @staticmethod
    def _apply_grouped_columnwise_rht(
        tensor_2d: torch.Tensor,
        group_list,
    ) -> torch.Tensor:
        if isinstance(group_list, torch.Tensor):
            group_boundaries = group_list.detach().cpu().tolist()
        else:
            group_boundaries = group_list
        start = 0
        outputs = []
        for end in group_boundaries:
            group_tensor = tensor_2d[start:end]
            outputs.append(rht_last_dim(group_tensor.t()).t())
            start = end
        return torch.cat(outputs, dim=0)

    def _grouped_quantize_tensor(self, group_list, tensor: torch.Tensor):
        col_data, col_scale, row_data, row_scale = None, None, None, None
        if self.columnwise_usage:
            if self.with_rht:
                col_data, col_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                    tensor,
                    group_list.to(torch.int32),
                    round_mode="rint",
                    dst_type=torch.float8_e4m3fn,
                    blocksize=32,
                )
            else:
                col_data, col_scale = torch_npu.npu_dynamic_mx_quant(tensor, axis=-2)
        if self.rowwise_usage:
            row_data, row_scale = torch_npu.npu_dynamic_mx_quant(tensor)
        return col_data, col_scale, row_data, row_scale

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: "MXFP4Tensor",
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> "MXFP4Tensor":
        if src is None:
            return src
        if noop_requested(noop_flag):
            return dst
        if not self.rowwise_usage and not self.columnwise_usage:
            return src
        if not isinstance(dst, MXFP4TensorStorage):
            raise ValueError("MXFP4Quantizer can only update MXFP4Tensor")
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()
        col_data, col_scale, row_data, row_scale = self._quantize_tensor(src, self.with_rht)
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        dst._columnwise_data = col_data
        dst._columnwise_scale_inv = col_scale
        dst._rowwise_data = row_data
        dst._rowwise_scale_inv = row_scale
        dst._fp4_dtype = self.dtype
        dst._block_size = self.block_size
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> "MXFP4Tensor":
        """Quantize tensor implementation."""
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor
        col_data, col_scale, row_data, row_scale = self._quantize_tensor(tensor, self.with_rht)
        return MXFP4Tensor(
            tensor.shape,
            tensor.dtype,
            rowwise_data=row_data,
            rowwise_scale_inv=row_scale,
            columnwise_data=col_data,
            columnwise_scale_inv=col_scale,
            fp4_dtype=self.dtype,
            quantizer=self,
            block_size=self.block_size,
            device=tensor.device,
        )

    def update_grouped_quantized(self, src, dst: "GroupedTensor", group_list, *, noop_flag=None):
        if src is None:
            return src
        if noop_requested(noop_flag):
            return dst
        if not self.rowwise_usage and not self.columnwise_usage:
            return src

        if not isinstance(dst, GroupedTensor):
            raise ValueError("MXFP4Quantizer can only update GroupedTensor")
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        col_data, col_scale, row_data, row_scale = self._grouped_quantize_tensor(
            group_list,
            src,
        )
        dst.logical_shape = tuple(src.shape)
        dst.fake_dtype = src.dtype
        dst.num_tensors = len(group_list)
        dst.quantizer = self
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

        col_data, col_scale, row_data, row_scale = self._grouped_quantize_tensor(
            group_list,
            tensor,
        )

        return GroupedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            num_tensors=len(group_list),
            quantizer=self,
            columnwise_data=col_data,
            columnwise_scale_inv=col_scale,
            data=row_data,
            scale_inv=row_scale,
        )

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized."""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % self.block_size != 0:
            return False
        if math.prod(inp.shape[:-1]) % self.block_size != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "MXFP4Tensor":
        if device is None:
            device = torch.device("npu" if hasattr(torch, "npu") else "cpu")
        shape = tuple(shape)
        rowwise_data = None
        rowwise_scale_inv = None
        columnwise_data = None
        columnwise_scale_inv = None
        if self.rowwise_usage:
            rowwise_data = torch.empty(
                shape, dtype=self.dtype, device=device, pin_memory=pin_memory
            )
            rowwise_scale_inv = torch.empty(
                MXFP4TensorStorage.scale_shape_for_rowwise(shape, self.block_size),
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
        if self.columnwise_usage:
            columnwise_data = torch.empty(
                shape, dtype=self.dtype, device=device, pin_memory=pin_memory
            )
            columnwise_scale_inv = torch.empty(
                MXFP4TensorStorage.scale_shape_for_columnwise(shape, self.block_size),
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
        return MXFP4Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp4_dtype=self.dtype,
            quantizer=self,
            block_size=self.block_size,
            requires_grad=requires_grad,
            device=device,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return MXFP4BlockScaling

    def transpose(self, data, scale):
        return transpose_mx_data(data, scale)

    def dequantize_grouped_tensor(self, tensor, *, dtype=None, columnwise=False):
        """Dequantize one orientation of packed MXFP4 grouped buffers."""
        storage = MXFP4TensorStorage(
            rowwise_data=None if columnwise else tensor.rowwise_data,
            rowwise_scale_inv=None if columnwise else tensor.rowwise_scale_inv,
            columnwise_data=tensor.columnwise_data if columnwise else None,
            columnwise_scale_inv=tensor.columnwise_scale_inv if columnwise else None,
            fp4_dtype=self.dtype,
            quantizer=self,
            fake_dtype=tensor.fake_dtype,
        )
        storage.origin_shape = tensor.logical_shape
        return storage.dequantize(dtype=dtype if dtype is not None else tensor.fake_dtype)


class MXFP4Tensor(MXFP4TensorStorage, QuantizedTensor):
    """Tensor class with MXFP4 data and E8M0 block scales."""

    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp4_dtype: torch.dtype,
        quantizer: Optional[Quantizer],
        block_size: int,
        **kwargs,
    ):
        return super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp4_dtype,
            quantizer,
            *args,
            block_size=block_size,
            **kwargs,
        )

    def _build_default_quantizer(self) -> Optional[Quantizer]:
        """Build default quantizer for the tensor."""
        qparams = getattr(self._quantizer, "qparams", None)
        with_rht = getattr(self._quantizer, "with_rht", None)
        return MXFP4Quantizer(
            fp4_dtype=self._fp4_dtype,
            block_size=self._block_size,
            qparams=qparams,
            with_rht=with_rht,
        )

    @property
    def device(self):
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("MXFP4Tensor has no data!")

    @property
    def shape(self):
        return torch.Size(self.origin_shape)

    def dim(self):
        return len(self.origin_shape)

    @property
    def is_cuda(self):
        if self._rowwise_data is not None:
            return self._rowwise_data.is_npu
        if self._columnwise_data is not None:
            return self._columnwise_data.is_npu
        raise RuntimeError("MXFP4Tensor has no data!")

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Construct plain PyTorch tensor from MXFP4Tensor."""
        if dtype is None:
            dtype = self.dtype
        if torch.is_grad_enabled():
            result = _FromMXFP4Func.apply(self, dtype)
        else:
            result = _FromMXFP4Func.forward(None, self, dtype)
        if tuple(result.shape) != tuple(self.origin_shape):
            result = result.reshape(self.origin_shape)
        return result

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> "MXFP4Tensor":
        """Update MXFP4 data in-place from a dense tensor."""
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        return super().quantize_(tensor, noop_flag=noop_flag)

    def detach(self) -> "MXFP4Tensor":
        return MXFP4Tensor.make_like(self)

    def clone(self) -> "MXFP4Tensor":
        rowwise_data = None
        columnwise_data = None
        rowwise_scale_inv = None
        columnwise_scale_inv = None
        if self._rowwise_data is not None:
            rowwise_data = self._rowwise_data.detach().clone()
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        if self._rowwise_scale_inv is not None:
            rowwise_scale_inv = self._rowwise_scale_inv.detach().clone()
        if self._columnwise_scale_inv is not None:
            columnwise_scale_inv = self._columnwise_scale_inv.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
                "rowwise_scale_inv": rowwise_scale_inv,
                "columnwise_scale_inv": columnwise_scale_inv,
            },
        )

    def view(self, *shape) -> "MXFP4Tensor":
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape) -> "MXFP4Tensor":
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> "MXFP4Tensor":
        rowwise_data = self._rowwise_data
        columnwise_data = self._columnwise_data
        rowwise_scale_inv = self._rowwise_scale_inv
        columnwise_scale_inv = self._columnwise_scale_inv

        if rowwise_data is not None and not rowwise_data.is_contiguous(memory_format=memory_format):
            pass
        elif columnwise_data is not None and not columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            pass
        elif rowwise_scale_inv is not None and not rowwise_scale_inv.is_contiguous():
            pass
        elif columnwise_scale_inv is not None and not columnwise_scale_inv.is_contiguous():
            pass
        else:
            return self

        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": (
                    rowwise_data.contiguous(memory_format=memory_format)
                    if rowwise_data is not None
                    else None
                ),
                "columnwise_data": (
                    columnwise_data.contiguous(memory_format=memory_format)
                    if columnwise_data is not None
                    else None
                ),
                "rowwise_scale_inv": (
                    rowwise_scale_inv.contiguous() if rowwise_scale_inv is not None else None
                ),
                "columnwise_scale_inv": (
                    columnwise_scale_inv.contiguous() if columnwise_scale_inv is not None else None
                ),
            },
        )

    def _view_or_reshape(self, shape, *, reshape: bool) -> "MXFP4Tensor":
        """Return an MXFP4Tensor with updated logical shape."""
        cur_shape = self.size()
        shape = _canonicalize_shape(cur_shape, shape)
        if tuple(shape) == tuple(cur_shape):
            return self

        if shape[-1] != cur_shape[-1]:
            if not reshape:
                warnings.warn(
                    "MXFP4Tensor does not support reshaping inner dimension. "
                    f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)}) "
                    "Falling back to dequantize().view(...) in higher precision.",
                    stacklevel=2,
                )
                return self.dequantize().view(*shape)
            raise RuntimeError(
                "MXFP4Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)})"
            )

        cur_rowwise_data = self._rowwise_data
        cur_columnwise_data = self._columnwise_data
        new_rowwise_data = None
        new_columnwise_data = None
        if cur_rowwise_data is not None:
            new_rowwise_data = (
                cur_rowwise_data.reshape(*shape) if reshape else cur_rowwise_data.view(*shape)
            )
        if cur_columnwise_data is not None:
            new_columnwise_data = (
                cur_columnwise_data.reshape(*shape) if reshape else cur_columnwise_data.view(*shape)
            )

        return MXFP4Tensor(
            shape,
            self.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=self._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=self._columnwise_scale_inv,
            fp4_dtype=self._fp4_dtype,
            quantizer=self._quantizer,
            block_size=self._block_size,
            device=self.device,
        )

    def matmul(self, B: "MXFP4Tensor", usage, usage_b, out_dtype):
        if not isinstance(B, MXFP4TensorStorage):
            raise TypeError(f"MXFP4Tensor.matmul expects MXFP4TensorStorage, got {type(B)}")
        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)

        return torch_npu.npu_quant_matmul(
            a,
            b,
            b_scale,
            pertoken_scale=a_scale,
            output_dtype=out_dtype,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )

    def matmul_add(self, main_grad, B: "MXFP4Tensor", usage, usage_b, out_dtype):
        if not isinstance(B, MXFP4TensorStorage):
            raise TypeError(f"MXFP4Tensor.matmul_add expects MXFP4TensorStorage, got {type(B)}")
        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)

        torch_npu.npu_add_quant_matmul_(
            main_grad,
            a,
            b,
            b_scale,
            x1_scale=a_scale,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, self._block_size],
        )

    def allgather_matmul(
        self,
        B,
        bias,
        world_size,
        group,
        usage,
        usage_b,
        out_dtype,
        return_gathered: bool = True,
    ):
        raise NotImplementedError("MXFP4 allgather_matmul is not implemented yet")

    def matmul_reduce_scatter(self, B, bias, world_size, group, usage, usage_b, out_dtype):
        raise NotImplementedError("MXFP4 matmul_reduce_scatter is not implemented yet")

    @classmethod
    def quant_reduce_scatter(cls, tensor, quantizer, world_size, group):
        raise NotImplementedError("MXFP4 quant_reduce_scatter is not implemented yet")


def _reshape_grad_to_input(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reshape view/reshape gradients back to the input tensor shape."""
    if isinstance(grad, QuantizedTensor) and hasattr(grad, "_view_or_reshape"):
        return grad._view_or_reshape(shape, reshape=True)
    return grad.reshape(shape)


class _ViewFunc(torch.autograd.Function):
    """Autograd-aware view for MXFP4Tensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP4Tensor,
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
    """Autograd-aware reshape for MXFP4Tensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP4Tensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        ctx.shape = torch.Size(tensor.origin_shape)
        if shape is None:
            return tensor.detach()
        return tensor._view_or_reshape(shape, reshape=True)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        return _reshape_grad_to_input(grad, ctx.shape), None


__all__ = ["MXFP4Quantizer", "MXFP4Tensor"]
