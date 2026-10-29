# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""HiF8 quantizer backed by the independent delayed-policy state machine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

from transformer_engine.common.recipe import HIF8DelayedScaling, Recipe

from .float8_tensor import Float8Quantizer, Float8Tensor
from .grouped_tensor import GroupedTensor
from .utils import noop_requested
from ..utils import devices_match, view_as_n_dim

if TYPE_CHECKING:
    from ..quantization.state.hif8_delayed_state import HIF8DelayedScalingRecipeState


class HIF8Quantizer(Float8Quantizer):
    """Float8-compatible quantizer with region-based HiF8 cadence."""

    def __init__(
        self,
        state: "HIF8DelayedScalingRecipeState",
        index: int,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        self.state = state
        self.index = index
        super().__init__(
            state.scale[index].reshape((1,)),
            state.pending_amax[index].reshape((1,)),
            state.dtype,
            rowwise=rowwise,
            columnwise=columnwise,
        )

    @property
    def amax(self) -> torch.Tensor:
        """Return the live pending-amax slice after communication-buffer growth."""
        return self.state.pending_amax[self.index].reshape((1,))

    @amax.setter
    def amax(self, value: torch.Tensor) -> None:
        # Float8Quantizer initializes this attribute. HiF8 reads the live
        # state view through the property getter, so the base-class assignment
        # intentionally does not retain the old slice.
        del value

    def copy(self) -> "HIF8Quantizer":
        """Create a shallow copy sharing policy state and cadence."""
        quantizer = HIF8Quantizer(
            self.state,
            self.index,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        return quantizer

    def _quantize_data(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale_inv = self.state.observe_and_get_scale_inv(self.index, tensor)
        data = torch_npu.npu_quantize(
            view_as_n_dim(tensor),
            scale_inv,
            zero_points=None,
            dtype=self.dtype,
            axis=-1,
        )
        return data, scale_inv

    def quantize_impl(self, tensor: torch.Tensor) -> Float8Tensor:
        if tensor is None:
            return tensor
        data, scale_inv = self._quantize_data(tensor)
        return Float8Tensor(
            tensor.shape,
            tensor.dtype,
            data=data,
            fp8_scale_inv=scale_inv,
            fp8_dtype=self.dtype,
            quantizer=self,
            device=tensor.device,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: Float8Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8Tensor:
        if not isinstance(dst, Float8Tensor):
            raise ValueError("HIF8Quantizer can only update Float8Tensor")
        if noop_requested(noop_flag):
            return dst
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        data, scale_inv = self._quantize_data(src)
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        dst._data = data
        dst._scale_inv = scale_inv
        dst._fp8_dtype = self.dtype
        return dst

    def grouped_quantize_impl(self, group_list, tensor):
        if tensor is None:
            return tensor
        num_tensors = len(group_list)
        data, scale_inv = self._quantize_data(tensor)
        expanded_scale_inv = scale_inv.expand(num_tensors)
        data = data.view(tensor.shape)
        return GroupedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            num_tensors=num_tensors,
            quantizer=self,
            data=data,
            scale_inv=expanded_scale_inv,
            columnwise_data=data,
            columnwise_scale_inv=expanded_scale_inv,
        )

    def update_grouped_quantized(self, src, dst: GroupedTensor, group_list, *, noop_flag=None):
        if not isinstance(dst, GroupedTensor):
            raise ValueError("HIF8Quantizer can only update GroupedTensor")
        if noop_requested(noop_flag):
            return dst
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        num_tensors = len(group_list)
        data, scale_inv = self._quantize_data(src)
        expanded_scale_inv = scale_inv.expand(num_tensors)
        data = data.view(src.shape)
        dst.rowwise_data = data
        dst.rowwise_scale_inv = expanded_scale_inv
        dst.columnwise_data = data
        dst.columnwise_scale_inv = expanded_scale_inv
        dst.num_tensors = num_tensors
        dst.quantizer = self
        return dst

    def calibrate(self, tensor: torch.Tensor) -> None:
        self.state.observe_for_calibration(self.index, tensor)

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return HIF8DelayedScaling
