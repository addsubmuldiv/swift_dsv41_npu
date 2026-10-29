# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch_npu

try:
    import cann_ops_nn
except ImportError:
    cann_ops_nn = None

from transformer_engine.common.recipe import Recipe, W4A8BlockScaling

from ..constants import MXFP8_BLOCK_SCALING_SIZE
from ..quantized_tensor import QuantizedTensor, Quantizer
from ..utils import devices_match, view_as_n_dim
from ._quantization_helpers import _IdentityFunc
from .grouped_tensor import GroupedTensor
from .storage.mxfp8_tensor_storage import MXFP8TensorStorage
from .storage.w4a8_tensor_storage import W4A8WeightTensorStorage, _FromW4A8WeightFunc
from .utils import noop_requested


class W4A8WeightQuantizer(Quantizer):
    """Quantizer that builds hybrid MXFP4/MXFP8 weight storage."""

    def __init__(
        self,
        *,
        mxfp8_dtype: torch.dtype,
        mxfp4_dtype: Optional[torch.dtype] = None,
        rowwise: bool = True,
        columnwise: bool = True,
        block_size: int = MXFP8_BLOCK_SCALING_SIZE,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.mxfp8_dtype = mxfp8_dtype
        self.mxfp4_dtype = mxfp4_dtype
        self.block_size = block_size

    @property
    def fp4_dtype(self) -> torch.dtype:
        return torch_npu.float4_e2m1fn_x2

    def copy(self) -> "W4A8WeightQuantizer":
        quantizer = W4A8WeightQuantizer(
            mxfp8_dtype=self.mxfp8_dtype,
            mxfp4_dtype=self.mxfp4_dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            block_size=self.block_size,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        return quantizer

    def transpose(self, data, scale):
        data = data.transpose(-1, -2) if data.ndim > 2 else data.t()
        if scale is not None:
            scale = scale.transpose(-3, -2) if scale.ndim > 2 else scale.t()
        return data, scale

    def _quantize_weight(self, tensor: torch.Tensor):
        tensor_2d = view_as_n_dim(tensor)
        mxfp4_data, mxfp4_scale = None, None
        mxfp8_data, mxfp8_scale = None, None
        if self.rowwise_usage:
            mxfp4_data, mxfp4_scale = torch_npu.npu_dynamic_mx_quant(
                tensor_2d,
                axis=-1,
                dst_type=self.fp4_dtype,
            )
        if self.columnwise_usage:
            mxfp8_data, mxfp8_scale = torch_npu.npu_dynamic_mx_quant(
                tensor_2d,
                axis=-2,
                dst_type=self.mxfp8_dtype,
            )
        return mxfp4_data, mxfp4_scale, mxfp8_data, mxfp8_scale

    @staticmethod
    def _group_list_for_quant(group_list, device: torch.device):
        if isinstance(group_list, torch.Tensor):
            if group_list.device != device:
                group_list = group_list.to(device=device)
            return group_list.to(torch.int32)
        return torch.tensor(group_list, device=device, dtype=torch.int32)

    def _grouped_quantize_tensor(self, group_list, tensor: torch.Tensor):
        if cann_ops_nn is None:
            raise RuntimeError("cann_ops_nn not installed; grouped W4A8 unavailable")
        group_list = self._group_list_for_quant(group_list, tensor.device)
        mxfp8_fwd_data, mxfp8_fwd_scale = None, None
        mxfp8_bwd_data, mxfp8_bwd_scale = None, None

        mxfp4_data, mxfp4_scale = torch_npu.npu_dynamic_mx_quant(
            tensor,
            axis=-1,
            dst_type=torch_npu.float4_e2m1fn_x2,
        )
        mxfp8_data, mxfp8_scale1_mxfp8, mxfp8_scale2_mxfp8 = cann_ops_nn.mx_to_block_mx_quant(
            mxfp4_data,
            mxfp4_scale,
            dst_type=torch_npu.float8_e4m3fn,
            x_type=torch_npu.float4_e2m1fn_x2,
        )

        if self.rowwise_usage:
            mxfp8_fwd_data = mxfp8_data
            mxfp8_fwd_scale = mxfp8_scale1_mxfp8
        if self.columnwise_usage:
            mxfp8_bwd_data = mxfp8_data
            mxfp8_bwd_scale = mxfp8_scale2_mxfp8
        return mxfp8_fwd_data, mxfp8_fwd_scale, mxfp8_bwd_data, mxfp8_bwd_scale

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: "W4A8WeightTensor",
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> "W4A8WeightTensor":
        if src is None:
            return src
        if not isinstance(dst, W4A8WeightTensorStorage):
            raise ValueError("W4A8WeightQuantizer can only update W4A8WeightTensor")
        if noop_requested(noop_flag):
            return dst
        if not self.rowwise_usage and not self.columnwise_usage:
            return src
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()
        mxfp4_data, mxfp4_scale, mxfp8_data, mxfp8_scale = self._quantize_weight(src)
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        dst._mxfp4_rowwise_data = mxfp4_data
        dst._mxfp4_rowwise_scale_inv = mxfp4_scale
        dst._mxfp8_columnwise_data = mxfp8_data
        dst._mxfp8_columnwise_scale_inv = mxfp8_scale
        dst._mxfp4_dtype = self.fp4_dtype
        dst._mxfp8_dtype = self.mxfp8_dtype
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> "W4A8WeightTensor":
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor
        mxfp4_data, mxfp4_scale, mxfp8_data, mxfp8_scale = self._quantize_weight(tensor)
        return W4A8WeightTensor(
            tensor.shape,
            tensor.dtype,
            mxfp4_rowwise_data=mxfp4_data,
            mxfp4_rowwise_scale_inv=mxfp4_scale,
            mxfp8_columnwise_data=mxfp8_data,
            mxfp8_columnwise_scale_inv=mxfp8_scale,
            mxfp4_dtype=self.fp4_dtype,
            mxfp8_dtype=self.mxfp8_dtype,
            quantizer=self,
            block_size=self.block_size,
            device=tensor.device,
        )

    def update_grouped_quantized(self, src, dst: GroupedTensor, group_list, *, noop_flag=None):
        if src is None:
            return src
        if not isinstance(dst, GroupedTensor):
            raise ValueError("W4A8WeightQuantizer can only update GroupedTensor")
        if noop_requested(noop_flag):
            return dst
        if not self.rowwise_usage and not self.columnwise_usage:
            return src
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()
        mxfp4_data, mxfp4_scale, mxfp8_data, mxfp8_scale = self._grouped_quantize_tensor(
            group_list,
            src,
        )
        dst.logical_shape = tuple(src.shape)
        dst.fake_dtype = src.dtype
        dst.num_tensors = len(group_list)
        dst.quantizer = self
        dst.columnwise_data = mxfp8_data
        dst.columnwise_scale_inv = mxfp8_scale
        dst.rowwise_data = mxfp4_data
        dst.rowwise_scale_inv = mxfp4_scale
        dst.scale_inv = mxfp4_scale
        return dst

    def grouped_quantize_impl(self, group_list, tensor: torch.Tensor):
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor
        mxfp4_data, mxfp4_scale, mxfp8_data, mxfp8_scale = self._grouped_quantize_tensor(
            group_list,
            tensor,
        )
        return GroupedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            num_tensors=len(group_list),
            quantizer=self,
            columnwise_data=mxfp8_data,
            columnwise_scale_inv=mxfp8_scale,
            data=mxfp4_data,
            scale_inv=mxfp4_scale,
        )

    def dequantize_grouped_tensor(self, tensor, *, dtype=None, columnwise=False):
        """Dequantize grouped W4A8 weight buffers via the MXFP8 backward view.

        Mirrors ``W4A8WeightTensorStorage.dequantize``: the MXFP4 rowwise view
        uses an NPU-internal tensor format that pure-torch dequantization
        cannot consume, so the MXFP8 columnwise (dgrad) buffers are always the
        dequantize source regardless of ``columnwise``.
        """
        fake_dtype = tensor.fake_dtype
        if tensor.columnwise_data is None or tensor.columnwise_scale_inv is None:
            raise RuntimeError("W4A8 grouped dequantize requires the MXFP8 backward weight view")
        storage = MXFP8TensorStorage(
            rowwise_data=None,
            rowwise_scale_inv=None,
            columnwise_data=tensor.columnwise_data,
            columnwise_scale_inv=tensor.columnwise_scale_inv,
            fp8_dtype=self.mxfp8_dtype,
            quantizer=None,
            with_gemm_swizzled_scales=False,
            fake_dtype=fake_dtype,
        )
        return storage.dequantize(dtype=dtype if dtype is not None else fake_dtype)

    def is_quantizable(self, inp: torch.Tensor) -> bool:
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
    ) -> "W4A8WeightTensor":
        if device is None:
            device = torch.device("npu" if hasattr(torch, "npu") else "cpu")
        shape = tuple(shape)
        mxfp4_data = None
        mxfp4_scale = None
        mxfp8_data = None
        mxfp8_scale = None
        if self.rowwise_usage:
            mxfp4_data = torch.empty(
                shape,
                dtype=self.fp4_dtype,
                device=device,
                pin_memory=pin_memory,
            )
            mxfp4_scale = torch.empty(
                W4A8WeightTensorStorage.scale_shape_for_rowwise(shape, self.block_size),
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
        if self.columnwise_usage:
            mxfp8_data = torch.empty(
                shape,
                dtype=self.mxfp8_dtype,
                device=device,
                pin_memory=pin_memory,
            )
            mxfp8_scale = torch.empty(
                W4A8WeightTensorStorage.scale_shape_for_columnwise(shape, self.block_size),
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
        return W4A8WeightTensor(
            shape=shape,
            dtype=dtype,
            mxfp4_rowwise_data=mxfp4_data,
            mxfp4_rowwise_scale_inv=mxfp4_scale,
            mxfp8_columnwise_data=mxfp8_data,
            mxfp8_columnwise_scale_inv=mxfp8_scale,
            mxfp4_dtype=self.fp4_dtype,
            mxfp8_dtype=self.mxfp8_dtype,
            quantizer=self,
            block_size=self.block_size,
            requires_grad=requires_grad,
            device=device,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return W4A8BlockScaling


class W4A8WeightTensor(W4A8WeightTensorStorage, QuantizedTensor):
    """Tensor subclass for W4A8 hybrid weights."""

    def __new__(
        cls,
        *args,
        mxfp4_rowwise_data: Optional[torch.Tensor],
        mxfp4_rowwise_scale_inv: Optional[torch.Tensor],
        mxfp8_columnwise_data: Optional[torch.Tensor],
        mxfp8_columnwise_scale_inv: Optional[torch.Tensor],
        mxfp4_dtype: torch.dtype,
        mxfp8_dtype: torch.dtype,
        quantizer: Optional[Quantizer],
        block_size: int,
        **kwargs,
    ):
        return super().__new__(
            cls,
            mxfp4_rowwise_data,
            mxfp4_rowwise_scale_inv,
            mxfp8_columnwise_data,
            mxfp8_columnwise_scale_inv,
            mxfp4_dtype,
            mxfp8_dtype,
            quantizer,
            block_size,
            *args,
            **kwargs,
        )

    def _build_default_quantizer(self) -> Optional[Quantizer]:
        return W4A8WeightQuantizer(
            mxfp8_dtype=self._mxfp8_dtype,
            mxfp4_dtype=self._mxfp4_dtype,
            block_size=self._block_size,
        )

    @property
    def device(self):
        if self._mxfp4_rowwise_data is not None:
            return self._mxfp4_rowwise_data.device
        if self._mxfp8_columnwise_data is not None:
            return self._mxfp8_columnwise_data.device
        raise RuntimeError("W4A8WeightTensor has no data")

    @property
    def shape(self):
        return torch.Size(self.origin_shape)

    def size(self, *args, **kwargs):
        shape = torch.Size(self.origin_shape)
        if not args and not kwargs:
            return shape
        dim = kwargs.pop("dim", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"size() got an unexpected keyword argument '{unexpected}'")
        if dim is None:
            if len(args) != 1:
                raise TypeError("size expected at most 1 argument")
            dim = args[0]
        return shape[dim]

    def dim(self):
        return len(self.origin_shape)

    @property
    def is_cuda(self):
        if self._mxfp4_rowwise_data is not None:
            return self._mxfp4_rowwise_data.is_npu
        if self._mxfp8_columnwise_data is not None:
            return self._mxfp8_columnwise_data.is_npu
        raise RuntimeError("W4A8WeightTensor has no data")

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if torch.is_grad_enabled():
            result = _FromW4A8WeightFunc.apply(self, dtype)
        else:
            result = _FromW4A8WeightFunc.forward(None, self, dtype)
        if tuple(result.shape) != tuple(self.origin_shape):
            result = result.reshape(self.origin_shape)
        return result

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> "W4A8WeightTensor":
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        return super().quantize_(tensor, noop_flag=noop_flag)

    def detach(self) -> "W4A8WeightTensor":
        return W4A8WeightTensor.make_like(self)

    def clone(self) -> "W4A8WeightTensor":
        init_kwargs = {
            "mxfp4_rowwise_data": (
                self._mxfp4_rowwise_data.detach().clone()
                if self._mxfp4_rowwise_data is not None
                else None
            ),
            "mxfp4_rowwise_scale_inv": (
                self._mxfp4_rowwise_scale_inv.detach().clone()
                if self._mxfp4_rowwise_scale_inv is not None
                else None
            ),
            "mxfp8_columnwise_data": (
                self._mxfp8_columnwise_data.detach().clone()
                if self._mxfp8_columnwise_data is not None
                else None
            ),
            "mxfp8_columnwise_scale_inv": (
                self._mxfp8_columnwise_scale_inv.detach().clone()
                if self._mxfp8_columnwise_scale_inv is not None
                else None
            ),
        }
        return _IdentityFunc.apply(self, init_kwargs)


__all__ = ["W4A8WeightQuantizer", "W4A8WeightTensor"]
