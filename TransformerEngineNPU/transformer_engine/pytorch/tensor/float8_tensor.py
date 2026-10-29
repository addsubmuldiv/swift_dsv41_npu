# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Float8 tensor classes for TransformerEngine Ascend PyTorch.

This module provides Float8 tensor classes for NPU-optimized FP8 operations,
integrating MindSpeed's FP8 tensor implementation with TransformerEngine's interface.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import MutableSequence, Optional, Tuple

import torch
import torch_npu

from .grouped_tensor import GroupedTensor
from .utils import noop_requested, to_torch_dtype
from ..constants import TensorUsage
from ..distributed import gather_along_dim, get_hccl_comm_name
from ..quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    _dequantize_grouped_by_per_group_scale,
)
from ..utils import devices_match, view_as_n_dim
from ._quantization_helpers import (
    _IdentityFunc,
    _pack_fp8_tensor_for_pickle,
    _pack_tensor_for_pickle,
    _restore_fp8_tensor_from_pickle,
)
from .storage.float8_tensor_storage import Float8TensorStorage, _FromFloat8Func
from ...common.recipe import DelayedScaling, Float8CurrentScaling, Recipe

aten = torch.ops.aten


class Float8Quantizer(Quantizer):
    """Quantizer for Float8 tensors using delayed scaling.

    This quantizer implements delayed scaling where the scale factor
    is computed based on historical amax values.

    Parameters
    ----------
    scale : Optional[torch.Tensor], default = None
        Scale factor for quantization
    amax : Optional[torch.Tensor], default = None
        Maximum absolute value
    fp8_dtype : torch.dtype, default = torch.float8_e4m3fn
        FP8 dtype to use for quantization. Also accepts the
        ``transformer_engine_torch.DType`` compatibility enum (GPU TE
        convention), which is mapped to the matching ``torch.dtype``.
    """

    def __init__(
        self,
        scale: torch.Tensor,
        amax: torch.Tensor,
        fp8_dtype: torch.dtype,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.scale = scale
        self.amax = amax
        self.dtype = to_torch_dtype(fp8_dtype)

    def copy(self) -> Float8Quantizer:
        """Create shallow copy"""
        quantizer = Float8Quantizer(
            scale=self.scale,
            amax=self.amax,
            fp8_dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
        )
        quantizer.internal = self.internal

        return quantizer

    def _quantize_with_scale(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize FP32 input using the FP32 multiplier inside the NPU operator."""
        assert self.scale is not None
        scale = self.scale.clone().float().reshape((1,))
        input_tensor = view_as_n_dim(tensor)
        data = torch_npu.npu_quantize(
            input_tensor.float(),
            scale,
            zero_points=None,
            dtype=self.dtype,
            axis=-1,
            div_mode=False,
        )
        return data, 1 / scale

    def quantize_impl(self, tensor: torch.Tensor) -> Float8Tensor:
        """Quantize tensor to Float8.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Quantized Float8Tensor
        """
        if tensor is None:
            return tensor

        # Update amax for delayed scaling
        # This amax will be collected by FP8GlobalStateManager at autocast exit
        if self.amax is not None:
            amin, amax = tensor.aminmax()
            self.amax.copy_(torch.max(-amin, amax))

        data, scale = self._quantize_with_scale(tensor)
        return Float8Tensor(
            tensor.shape,
            tensor.dtype,
            data=data,
            fp8_scale_inv=scale,
            fp8_dtype=self.dtype,
            quantizer=self,
            device=tensor.device,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: "Float8Tensor",
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> "Float8Tensor":
        if not isinstance(dst, Float8Tensor):
            raise ValueError("Float8Quantizer can only update Float8Tensor")
        if noop_requested(noop_flag):
            return dst
        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Update amax for delayed scaling
        # This amax will be collected by FP8GlobalStateManager at autocast exit
        if self.amax is not None:
            amin, amax = src.aminmax()
            self.amax.copy_(torch.max(-amin, amax))

        data, scale = self._quantize_with_scale(src)
        # Update FP8 tensor
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        dst._data = data
        dst._scale_inv = scale
        dst._fp8_dtype = self.dtype
        return dst

    def grouped_quantize_impl(self, group_list, tensor):
        """Quantize tensor to Float8.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Quantized Float8Tensor
        """
        if tensor is None:
            return tensor

        # Update amax for delayed scaling
        # This amax will be collected by FP8GlobalStateManager at autocast exit
        if self.amax is not None:
            amin, amax = tensor.aminmax()
            self.amax.copy_(torch.max(-amin, amax))

        num_tensors = len(group_list)
        data, scale = self._quantize_with_scale(tensor)
        scale = scale.expand(num_tensors)
        data = data.view(tensor.shape)
        return GroupedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            num_tensors=num_tensors,
            quantizer=self,
            data=data,
            scale_inv=scale,
            columnwise_data=data,
            columnwise_scale_inv=scale,
        )

    def dequantize_grouped_tensor(self, tensor, *, dtype=None, columnwise=False):
        """Dequantize packed grouped buffers with one scalar scale per group."""
        return _dequantize_grouped_by_per_group_scale(
            tensor,
            dtype if dtype is not None else tensor.fake_dtype,
            columnwise=columnwise,
        )

    def update_grouped_quantized(self, src, dst: GroupedTensor, group_list, *, noop_flag=None):
        if not isinstance(dst, GroupedTensor):
            raise ValueError("Float8Quantizer can only update GroupedTensor")
        if noop_requested(noop_flag):
            return dst
        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Update amax for delayed scaling
        # This amax will be collected by FP8GlobalStateManager at autocast exit
        if self.amax is not None:
            amin, amax = src.aminmax()
            self.amax.copy_(torch.max(-amin, amax))

        num_tensors = len(group_list)
        data, scale = self._quantize_with_scale(src)
        scale = scale.expand(num_tensors)
        data = data.view(src.shape)
        dst.rowwise_data = data
        dst.rowwise_scale_inv = scale
        dst.columnwise_data = data
        dst.columnwise_scale_inv = scale
        dst.num_tensors = num_tensors
        dst.quantizer = self
        return dst

    def calibrate(self, tensor: torch.Tensor) -> None:
        """Calibrate quantizer with input tensor.

        Args:
            tensor: Input tensor for calibration
        """
        amin, amax = tensor.aminmax()
        self.amax.copy_(torch.max(-amin, amax))

    def make_empty(
        self,
        shape: MutableSequence[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "Float8Tensor":
        if device is None:
            device = torch.device("npu" if hasattr(torch, "npu") else "cpu")
        data = None
        if self.rowwise_usage:
            data = torch.empty(shape, dtype=self.dtype, device=device, pin_memory=pin_memory)
        data_transpose = None
        if self.columnwise_usage:
            transpose_shape = [shape[-1]] + list(shape[:-1])
            data_transpose = torch.empty(
                transpose_shape,
                dtype=self.dtype,
                device=device,
                pin_memory=pin_memory,
            )
        scale = torch.empty(1, dtype=torch.float32, device=device, pin_memory=pin_memory)
        return Float8Tensor(
            shape=shape,
            dtype=dtype,
            data=data,
            fp8_scale_inv=scale,
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=data_transpose,
            quantizer=self,
            device=device,
        )

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return DelayedScaling


class Float8CurrentScalingQuantizer(Quantizer):
    """Float8 quantizer with current scaling.

    This quantizer computes the scale factor based on the current
    tensor's maximum absolute value, without using history.

    """

    def __init__(
        self,
        scale: Optional[torch.Tensor] = None,
        amax: Optional[torch.Tensor] = None,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        rowwise: bool = True,
        columnwise: bool = True,
        force_pow_2_scales: bool = False,
        amax_epsilon: float = 0.0,
        amax_reduction_group=None,
        with_amax_reduction: bool = False,
        use_existing_amax: bool = False,
        margin: int = 0,
        fp8_max: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        if scale is None and device is not None:
            scale = torch.ones(1, device=device, dtype=torch.float32)
        if amax is None and device is not None:
            amax = torch.zeros(1, device=device, dtype=torch.float32)
        self.scale = scale
        self.amax = amax
        self.dtype = fp8_dtype
        self.use_existing_amax = use_existing_amax
        self.with_amax_reduction = with_amax_reduction
        self.amax_reduction_group = amax_reduction_group
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon
        self.fp8_max = fp8_max

    def __getstate__(self):
        """Exclude unpicklable process group from serialized state."""
        state = self.__dict__.copy()
        state["amax_reduction_group"] = None
        return state

    def copy(self) -> Float8CurrentScalingQuantizer:
        """Create shallow copy"""
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=self.dtype,
            device=0,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            with_amax_reduction=self.with_amax_reduction,
            amax_reduction_group=self.amax_reduction_group,
            use_existing_amax=self.use_existing_amax,
            force_pow_2_scales=self.force_pow_2_scales,
            amax_epsilon=self.amax_epsilon,
            scale=self.scale,
            amax=self.amax,
            fp8_max=self.fp8_max,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant

        return quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        if not isinstance(dst, Float8Tensor):
            raise ValueError("Float8CurrentScalingQuantizer can only update Float8Tensor")
        if noop_requested(noop_flag):
            return dst

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()
        # Launch cast kernel
        tensor_2d = view_as_n_dim(src)
        data, scale = torch_npu.npu_dynamic_quant(
            tensor_2d,
            dst_type=self.dtype,
            quant_mode="pertensor",
            dst_type_max=self.fp8_max if self.dtype == torch_npu.hifloat8 else 0,
        )
        # Update FP8 tensor
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        dst._data = data
        dst._scale_inv = scale
        dst._fp8_dtype = self.dtype
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to Float8.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Quantized Float8Tensor
        """
        if tensor is None:
            return tensor
        tensor_2d = view_as_n_dim(tensor)
        data, scale = torch_npu.npu_dynamic_quant(
            tensor_2d,
            dst_type=self.dtype,
            quant_mode="pertensor",
            dst_type_max=self.fp8_max if self.dtype == torch_npu.hifloat8 else 0,
        )
        return Float8Tensor(
            tensor.shape,
            tensor.dtype,
            data=data,
            fp8_scale_inv=scale,
            fp8_dtype=self.dtype,
            quantizer=self,
            device=tensor.device,
        )

    def grouped_quantize_impl(self, group_list, tensor):
        """Grouped quantize tensor to Float8.
        Args:
            group_list: List of groups to quantize
        """
        if tensor is None:
            return tensor
        num_tensors = len(group_list)
        data, scale = torch_npu.npu_dynamic_quant(
            tensor.view(num_tensors, -1),
            dst_type=self.dtype,
            dst_type_max=self.fp8_max if self.dtype == torch_npu.hifloat8 else 0,
        )
        if self.columnwise_use_group_quant:
            scale = scale.expand(num_tensors)
        data = data.view(tensor.shape)
        return GroupedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            num_tensors=num_tensors,
            quantizer=self,
            data=data,
            scale_inv=scale,
            columnwise_data=data,
            columnwise_scale_inv=scale,
        )

    def dequantize_grouped_tensor(self, tensor, *, dtype=None, columnwise=False):
        """Dequantize packed grouped buffers with one scalar scale per group."""
        return _dequantize_grouped_by_per_group_scale(
            tensor,
            dtype if dtype is not None else tensor.fake_dtype,
            columnwise=columnwise,
        )

    def update_grouped_quantized(self, src, dst: "GroupedTensor", group_list, *, noop_flag=None):
        if src is None:
            return src
        if noop_requested(noop_flag):
            return dst
        num_tensors = len(group_list)

        data, scale = torch_npu.npu_dynamic_quant(
            src.view(num_tensors, -1),
            dst_type=self.dtype,
            dst_type_max=self.fp8_max if self.dtype == torch_npu.hifloat8 else 0,
        )
        if self.columnwise_use_group_quant:
            scale = scale.expand(num_tensors)
        data = data.view(src.shape)
        dst.rowwise_data = data
        dst.rowwise_scale_inv = scale
        dst.columnwise_data = data
        dst.columnwise_scale_inv = scale
        dst.num_tensors = num_tensors
        dst.quantizer = self
        return dst

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "Float8Tensor":
        if device is None:
            device = torch.device("npu" if hasattr(torch, "npu") else "cpu")
        data = torch.empty(shape, dtype=self.dtype, device=device, pin_memory=pin_memory)
        scale = torch.empty(1, dtype=torch.float32, device=device, pin_memory=pin_memory)
        return Float8Tensor(
            shape=shape,
            dtype=dtype,
            data=data,
            fp8_scale_inv=scale,
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=None,
            quantizer=self,
            device=device,
        )

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return Float8CurrentScaling


class Float8Tensor(Float8TensorStorage, QuantizedTensor):
    """Float8 tensor class for FP8 quantized tensors.

    This class wraps a quantized FP8 tensor with its scale factors
    and provides methods for quantization and dequantization.
    """

    def __repr__(self, *, tensor_contents=None):
        return (
            "Float8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize()}"
            ")"
        )

    @property
    def shape(self):
        """Return the logical shape of the tensor."""
        if self._data is not None:
            return self._data.shape
        if self._transpose is not None:
            transpose_shape = self._transpose.shape
            return torch.Size(tuple(transpose_shape[1:]) + (transpose_shape[0],))
        return torch.Tensor.size(self)

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Construct plain PyTorch tensor from Float8Tensor."""
        if dtype is None:
            dtype = self.dtype
        tensor = self.contiguous()
        if torch.is_grad_enabled():
            result = _FromFloat8Func.apply(tensor, dtype)
        else:
            result = _FromFloat8Func.forward(None, tensor, dtype)
        if tuple(result.shape) != tuple(self.origin_shape):
            result = result.reshape(self.origin_shape)
        return result

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8Tensor:
        """Update FP8 data in-place from a dense tensor."""
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        return super().quantize_(tensor, noop_flag=noop_flag)

    def detach(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return Float8Tensor.make_like(self)

    def clone(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        assert self._data is not None
        data = self._data.detach().clone()
        data_transpose = None
        if self._transpose is not None:
            data_transpose = self._transpose.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "data": data,
                "data_transpose": data_transpose,
            },
        )

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: Optional[torch.Tensor],
        fp8_dtype: torch.dtype,
        fp8_scale_inv: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.Size,
        data_transpose: Optional[torch.Tensor] = None,
    ) -> Float8Tensor:
        """Build Float8Tensor during unpickling."""
        data = _restore_fp8_tensor_from_pickle(data, fp8_dtype)
        data_transpose = _restore_fp8_tensor_from_pickle(data_transpose, fp8_dtype)
        device = None
        if data is not None:
            device = data.device
        elif data_transpose is not None:
            device = data_transpose.device
        return Float8Tensor(
            shape=shape,
            dtype=dtype,
            data=data,
            fp8_scale_inv=fp8_scale_inv,
            fp8_dtype=fp8_dtype,
            data_transpose=data_transpose,
            quantizer=None,
            device=device,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Serialize FP8 payload as raw uint8 bytes to avoid float8 storage pickle gaps."""
        return (
            _make_float8_tensor_in_reduce_ex,
            (
                _pack_fp8_tensor_for_pickle(self._data),
                self._fp8_dtype,
                _pack_tensor_for_pickle(self._scale_inv),
                self.dtype,
                torch.Size(self.origin_shape),
                _pack_fp8_tensor_for_pickle(self._transpose),
            ),
        )

    def _get_data(self) -> Float8Tensor:
        """Get tensor data property."""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data, copying FP8 buffers or requantizing dense input."""
        new_device = (
            tensor.device
            if isinstance(tensor, Float8Tensor) or getattr(tensor, "is_npu", False)
            else self.device
        )
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        if isinstance(tensor, Float8Tensor):
            if (
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    Float8Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(Float8Tensor, type(self)).data.__set__(self, dummy_tensor)

            self.origin_shape = tuple(tensor.origin_shape)
            self.dtype = tensor.dtype
            self._data = tensor._data
            self._quantizer = tensor._quantizer.copy() if tensor._quantizer is not None else None
            self._fp8_dtype = tensor._fp8_dtype
            self._scale_inv = tensor._scale_inv
            self._transpose = tensor._transpose
            self._transpose_invalid = tensor._transpose_invalid
            return

        if self._quantizer is None:
            raise AssertionError("Can't quantize without a quantizer")
        self._quantizer.internal = False
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    data = property(_get_data, _set_data)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == aten.copy_.default:
            dst, src = args[0], args[1]
            if isinstance(dst, Float8Tensor) and isinstance(src, Float8Tensor):
                dst.copy_from_storage(src)
                dst.origin_shape = src.origin_shape
                dst.dtype = src.dtype
                return dst

        if func == aten.view.default:
            tensor = args[0]
            return tensor.view(*args[1])

        if func in (aten.slice.Tensor, aten.select.int):
            tensor = args[0]
            if tensor._data is None:
                return super().__torch_dispatch__(func, types, args, kwargs)
            logical_data = tensor._data.view(tensor.origin_shape)
            data_slice = func(logical_data, *args[1:], **kwargs)
            return Float8Tensor(
                shape=data_slice.shape,
                dtype=tensor.dtype,
                data=data_slice,
                fp8_scale_inv=tensor._scale_inv,
                fp8_dtype=tensor._fp8_dtype,
                data_transpose=None,
                quantizer=tensor._quantizer,
                device=tensor.device,
            )

        if func == aten.split.Tensor:
            tensor = args[0]
            if tensor._data is None:
                return super().__torch_dispatch__(func, types, args, kwargs)
            logical_data = tensor._data.view(tensor.origin_shape)
            data_splits = func(logical_data, *args[1:], **kwargs)
            return [
                Float8Tensor(
                    shape=split_tensor.shape,
                    dtype=tensor.dtype,
                    data=split_tensor,
                    fp8_scale_inv=tensor._scale_inv,
                    fp8_dtype=tensor._fp8_dtype,
                    data_transpose=None,
                    quantizer=tensor._quantizer,
                    device=tensor.device,
                )
                for split_tensor in data_splits
            ]

        if func == aten.new_zeros.default:
            tensor = args[0]
            shape = torch.Size(args[1])
            out = tensor._quantizer.make_empty(
                shape=shape,
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=False,
            )
            if out._data is not None:
                out._data.zero_()
            if out._transpose is not None:
                out._transpose.zero_()
            if out._scale_inv is not None:
                out._scale_inv.copy_(tensor._scale_inv.detach())
            return out

        if func == aten.as_strided.default:
            tensor = args[0]
            if tensor._data is None:
                return super().__torch_dispatch__(func, types, args, kwargs)
            size = args[1]
            stride = args[2]
            storage_offset = kwargs.get("storage_offset", args[3] if len(args) > 3 else 0)
            logical_data = tensor._data.view(tensor.origin_shape)
            out_data = func(logical_data, size, stride, storage_offset)
            out_transpose = None
            if tensor._transpose is not None and not tensor._transpose_invalid:
                transpose_size = [size[-1]] + list(size[:-1])
                transpose_stride = [stride[-1]] + list(stride[:-1])
                out_transpose = func(
                    tensor._transpose,
                    transpose_size,
                    transpose_stride,
                    storage_offset,
                )
            return Float8Tensor(
                shape=out_data.shape,
                dtype=tensor.dtype,
                data=out_data,
                fp8_scale_inv=tensor._scale_inv,
                fp8_dtype=tensor._fp8_dtype,
                data_transpose=out_transpose,
                quantizer=tensor._quantizer,
                device=tensor.device,
            )

        return super().__torch_dispatch__(func, types, args, kwargs)

    def view(self, *shape) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8Tensor:
        """Returns tensor with data in provided memory format

        Returns ``self`` if data is already in correct memory format.

        """  # Check if tensor already has correct memory format
        if self._data is not None and not self._data.is_contiguous(  # noqa: SIM114
            memory_format=memory_format
        ):
            pass
        elif self._transpose is not None and not self._transpose.is_contiguous(
            memory_format=memory_format
        ):
            pass
        else:
            # Tensor has correct memory format, so return immediately
            return self

        # Construct tensor with correct data format
        data, data_transpose = None, None
        if self._data is not None:
            data = self._data.contiguous(memory_format=memory_format)
        if self._transpose is not None and not self._transpose_invalid:
            data_transpose = self._transpose.contiguous(memory_format=memory_format)
        return _IdentityFunc.apply(
            self,
            {"data": data, "data_transpose": data_transpose},
        )

    def _reset_caches(self) -> None:
        """Mark the transpose cache as invalid after an in-place update."""
        self._transpose_invalid = True

    def remove_caches(self) -> None:
        """Drop transpose cache explicitly."""
        self._transpose_invalid = True
        self._transpose = None

    def _view_or_reshape(self, shape, *, reshape: bool) -> Float8Tensor:
        """Return a Float8Tensor with updated logical shape.

        We currently only reshape the rowwise FP8 data directly. The transpose cache
        can be reused only when its cached layout still matches the new logical shape.
        """
        cur_shape = self.size()
        shape = _canonicalize_shape(cur_shape, shape)
        if tuple(shape) == tuple(cur_shape):
            return self

        op = "reshape" if reshape else "view"
        if self._data is None:
            raise RuntimeError(f"Float8Tensor.{op} requires rowwise FP8 data to be present")
        out_data = self._data.reshape(*shape) if reshape else self._data.view(*shape)
        out_shape = out_data.size()

        out_transpose = None if self._transpose_invalid else self._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if out_transpose_shape[0] != out_shape[-1] or out_transpose_shape[1:] != out_shape[:-1]:
                out_transpose = None
            else:
                transpose_shape = [out_shape[-1]] + list(out_shape[:-1])
                out_transpose = (
                    out_transpose.reshape(*transpose_shape)
                    if reshape
                    else out_transpose.view(*transpose_shape)
                )

        return Float8Tensor(
            out_shape,
            self.dtype,
            data=out_data,
            fp8_scale_inv=self._scale_inv,
            fp8_dtype=self._fp8_dtype,
            data_transpose=out_transpose,
            quantizer=self._quantizer,
            device=self.device,
        )

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA/NPU-backed device."""
        if self._data is not None:
            return self._data.is_npu
        if self._transpose is not None:
            return self._transpose.is_npu
        raise RuntimeError("Both data and transpose are None")

    @property
    def is_cpu(self):
        """Return whether the tensor is on CPU."""
        if self._data is not None:
            return self._data.is_cpu
        if self._transpose is not None:
            return self._transpose.is_cpu
        raise RuntimeError("Both data and transpose are None")

    def matmul(self, B: Float8Tensor, usage: TensorUsage, usage_b, out_dtype):
        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)
        out = torch_npu.npu_quant_matmul(
            a,
            b,
            b_scale,
            pertoken_scale=a_scale,
            output_dtype=out_dtype,
            **self.mm_kwargs,
        )
        if len(out.shape) == 2:
            return out
        return out.reshape(*self.origin_shape[:-1], *out.shape[1:])

    def allgather_matmul(
        self,
        B: Float8Tensor,
        bias,
        world_size,
        group,
        usage,
        usage_b,
        out_dtype,
        return_gathered: bool = True,
    ):
        _, x1_scale = gather_along_dim(self._scale_inv)
        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)
        # x1 scale 因为是单标量tensor 与之前allgather输出 tensor相同 可以省一次allgather
        hcomm_name = get_hccl_comm_name(group)
        output, gather_out, _ = torch_npu.npu_all_gather_quant_mm(
            a,
            b,
            hcomm_name,
            world_size,
            bias=bias,
            x1_scale=a_scale,
            x2_scale=b_scale,
            **self.mm_kwargs,
            y_dtype=out_dtype,
        )
        gather_out = Float8Tensor(
            self.origin_shape,
            self.dtype,
            data=gather_out,
            fp8_scale_inv=x1_scale,
            fp8_dtype=self._fp8_dtype,
            quantizer=self._quantizer,
        )
        return output.view(-1, self.origin_shape[1], output.shape[1]), gather_out

    def matmul_reduce_scatter(
        self, B: Float8Tensor, bias, world_size, group, usage, usage_b, out_dtype
    ):
        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)

        hcomm_name = get_hccl_comm_name(group)
        output, _ = torch_npu.npu_quant_mm_reduce_scatter(
            a,
            b,
            hcomm_name,
            world_size,
            bias=bias,
            reduce_op="sum",
            x1_scale=a_scale,
            x2_scale=b_scale,
            **self.mm_kwargs,
            y_dtype=out_dtype,
        )
        return output.view(-1, self.origin_shape[1], output.shape[1])


def _make_float8_tensor_in_reduce_ex(
    data: Optional[torch.Tensor],
    fp8_dtype: torch.dtype,
    fp8_scale_inv: torch.Tensor,
    dtype: torch.dtype,
    shape: torch.Size,
    data_transpose: Optional[torch.Tensor] = None,
) -> Float8Tensor:
    """Reconstruct a ``Float8Tensor`` from its pickle payload."""
    return Float8Tensor._make_in_reduce_ex(
        data,
        fp8_dtype,
        fp8_scale_inv,
        dtype,
        shape,
        data_transpose,
    )


__all__ = [
    "Float8Quantizer",
    "Float8CurrentScalingQuantizer",
    "Float8Tensor",
]


def _reshape_grad_to_input(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reshape view/reshape gradients back to the input tensor shape."""
    if isinstance(grad, QuantizedTensor) and hasattr(grad, "_view_or_reshape"):
        return grad._view_or_reshape(shape, reshape=True)
    return grad.reshape(shape)


class _ViewFunc(torch.autograd.Function):
    """Autograd-aware view for Float8Tensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
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
    """Autograd-aware reshape for Float8Tensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
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
