# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Dense tensor wrapper and passthrough quantizer for QAT."""

from typing import Optional

import torch
from torch.utils._pytree import tree_map

from transformer_engine.common.recipe import Recipe

from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage, Quantizer


class _FromDummyTensorFunc(torch.autograd.Function):
    """Unwrap a dummy tensor while preserving its autograd edge."""

    @staticmethod
    def forward(ctx, tensor: "DummyTensor") -> torch.Tensor:
        return tensor._data

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad


class DummyTensor(QuantizedTensorStorage, QuantizedTensor):
    """Transparent tensor wrapper for dense values in the quantized-tensor flow."""

    _data: Optional[torch.Tensor]
    _quantizer: Optional[Quantizer]
    _grouped: bool

    @staticmethod
    def __new__(
        cls,
        shape,
        dtype: torch.dtype,
        *,
        data: torch.Tensor,
        quantizer: Optional[Quantizer] = None,
        grouped: bool = False,
        requires_grad: bool = False,
    ) -> "DummyTensor":
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            requires_grad=requires_grad,
            device=data.device,
        )
        instance._data = data.detach()
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._grouped = grouped
        return instance

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            tensor = args[0]
            data = tensor._data.detach() if func == torch.ops.aten.detach.default else tensor._data
            return cls(
                tensor.shape,
                tensor.dtype,
                data=data,
                quantizer=tensor._quantizer,
                grouped=tensor._grouped,
                requires_grad=(
                    False if func == torch.ops.aten.detach.default else tensor.requires_grad
                ),
            )

        if func == torch.ops.aten.copy_.default:
            dst, src = args[:2]
            src = src._data if isinstance(src, DummyTensor) else src
            data = dst._data if isinstance(dst, DummyTensor) else dst
            data.copy_(src, *args[2:], **kwargs)
            return dst

        def unwrap(value):
            return value.dequantize() if isinstance(value, DummyTensor) else value

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

    def _get_data(self) -> torch.Tensor:
        return self._data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        self._data = tensor.detach()

    data = property(_get_data, _set_data)

    def get_metadata(self):
        """Return the constructor metadata for the dense payload."""
        return {
            "data": self._data,
            "quantizer": self._quantizer,
            "grouped": self._grouped,
        }

    @classmethod
    def make_like(cls, tensor, *, shape=None, dtype=None, requires_grad=False):
        """Create a dense wrapper with matching payload shape and dtype."""
        shape = tensor.shape if shape is None else shape
        dtype = tensor.dtype if dtype is None else dtype
        metadata = tensor.get_metadata()
        metadata["data"] = metadata["data"].reshape(shape).to(dtype=dtype)
        return cls(shape, dtype, requires_grad=requires_grad, **metadata)

    def _update_data(
        self,
        tensor: torch.Tensor,
        *,
        grouped: bool,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> "DummyTensor":
        if self._grouped != grouped:
            raise ValueError("DummyTensor grouped layout does not match the quantizer call")
        if tuple(tensor.shape) != tuple(self.shape):
            raise ValueError(
                f"DummyTensor update shape must be {tuple(self.shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.device != self._data.device:
            tensor = tensor.to(device=self._data.device)
        if noop_flag is not None:
            tensor = torch.where(noop_flag == 0, tensor, self._data)
        self._data = tensor.detach()
        return self

    def get_data(self, usage):  # pylint: disable=unused-argument
        """Return the dense GEMM payload."""
        if self._grouped or self._data.dim() == 2:
            return self._data
        if self._data.dim() == 0:
            return self._data.view(1, 1)
        return self._data.view(-1, self._data.shape[-1])

    def can_reuse_quantized(self, quantizer: Quantizer) -> bool:
        """Dense input values must pass through the requested quantizer.

        Even fake-quantized payloads can be modified by dense in-place operations.
        Reuse of a cached workspace remains controlled by quantize_weight().
        """
        return False

    def matmul(self, B, usage=None, usage_b=None, out_dtype=None):
        """Support both ordinary Tensor.matmul and the quantized GEMM interface."""
        if usage is None:
            return torch.Tensor.matmul(self, B)
        from ..ops.gemm import general_gemm

        B = B.get_data(usage_b) if isinstance(B, DummyTensor) else B
        return general_gemm(self.get_data(usage), B, usage, usage_b, out_dtype)

    def matmul_add(self, main_grad, B, usage, usage_b, out_dtype):
        """Accumulate through the existing dense GEMM implementation."""
        from ..ops.gemm import general_gemm_add

        B = B.get_data(usage_b) if isinstance(B, DummyTensor) else B
        return general_gemm_add(main_grad, self.get_data(usage), B, usage, usage_b, out_dtype)

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
        """Use dense communication fusion, preserving the logical input shape."""
        from ..ops.fused.overlap import CommOverlapOps

        B = B.dequantize() if isinstance(B, QuantizedTensor) else B
        return CommOverlapOps.allgather_matmul(
            self.dequantize(), B, bias, world_size, group, usage, usage_b, out_dtype
        )

    def matmul_reduce_scatter(self, B, bias, world_size, group, usage, usage_b, out_dtype):
        """Use dense GEMM/reduce-scatter with the original shape restoration."""
        from ..ops.fused.overlap import CommOverlapOps

        B = B.dequantize() if isinstance(B, QuantizedTensor) else B
        return CommOverlapOps.matmul_reduce_scatter(
            self.dequantize(), B, bias, world_size, group, usage, usage_b, out_dtype
        )

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Return the single dense payload for either usage."""
        if not rowwise_data and not columnwise_data:
            raise ValueError("No data requested")
        if rowwise_data and columnwise_data:
            return (self._data,)
        return self._data

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ) -> None:
        """Accept usage updates since the dense payload supports both orientations."""

    def prepare_for_saving(self):
        """Move the dense payload into autograd's saved tensor list."""
        tensors = [self._data]
        self._data = None
        return tensors, self

    def restore_from_saved(self, tensors):
        """Restore the dense payload from autograd's saved tensor list."""
        self._data = tensors[0]
        return tensors[1:]

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy a dense payload from another dummy tensor."""
        if not isinstance(src, DummyTensor):
            raise TypeError("copy_from_storage expects DummyTensor")
        self._data.copy_(src._data)

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Return the wrapped dense tensor."""
        if torch.is_grad_enabled():
            tensor = _FromDummyTensorFunc.apply(self)
        else:
            tensor = self._data
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def __repr__(self) -> str:
        return f"DummyTensor(data={self._data})"


class DummyQuantizer(Quantizer):
    """Wrap a tensor without changing its dense values."""

    def __init__(self) -> None:
        super().__init__(rowwise=True, columnwise=True)

    def copy(self) -> "DummyQuantizer":
        """Create a shallow copy."""
        quantizer = DummyQuantizer()
        quantizer.rowwise_usage = self.rowwise_usage
        quantizer.columnwise_usage = self.columnwise_usage
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        return quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: DummyTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> DummyTensor:
        """Refresh an existing dummy tensor."""
        if not isinstance(dst, DummyTensor):
            raise ValueError("DummyQuantizer can only update DummyTensor")
        return dst._update_data(src, grouped=False, noop_flag=noop_flag)

    def quantize_impl(self, tensor: torch.Tensor) -> DummyTensor:
        """Wrap a tensor without changing its values."""
        if tensor is None:
            return tensor
        return DummyTensor(tensor.shape, tensor.dtype, data=tensor, quantizer=self)

    def update_grouped_quantized(
        self,
        src: torch.Tensor,
        dst: DummyTensor,
        group_list,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> DummyTensor:
        """Refresh an existing grouped dummy tensor."""
        if not isinstance(dst, DummyTensor):
            raise ValueError("DummyQuantizer can only update DummyTensor")
        return dst._update_data(src, grouped=True, noop_flag=noop_flag)

    def grouped_quantize_impl(  # pylint: disable=unused-argument
        self, group_list, tensor: torch.Tensor
    ) -> DummyTensor:
        """Wrap a grouped tensor without changing its values."""
        if tensor is None:
            return tensor
        return DummyTensor(
            tensor.shape,
            tensor.dtype,
            data=tensor,
            quantizer=self,
            grouped=True,
        )

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        """Passthrough quantization is compatible with any recipe."""
        return None


__all__ = ["DummyQuantizer", "DummyTensor"]
