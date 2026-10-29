# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for RMSNorm for NPU."""

from __future__ import annotations
from collections.abc import Iterable
import math
import warnings
from typing import Optional

import torch
import torch_npu

from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Quantizer
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
)
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize, maybe_autocast_dtype

__all__ = ["RMSNorm"]


class RMSNorm(BasicOperation):
    r"""Root Mean Square Layer Normalization

    Applies Root Mean Square Layer Normalization over a mini-batch of
    inputs as described in the paper
    `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`__ .

    .. math::
        y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma

    :math:`\gamma` is a learnable affine transform parameter that
    matches the inner-most dimensions of the input tensor.

    Parameters
    ----------
    normalized_shape : int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator for numerical stability
    device : torch.device, default = default CUDA device
        Tensor device
    dtype : torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = False
        If ``True``, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma)

    sm_margin : int, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.
        For more fine-grained control, provide a dict with the SM
        margin at each compute stage ("forward", "backward",
        "inference").

    """

    def __init__(
        self,
        normalized_shape: Iterable[int] | int,
        *,
        eps: float = 1e-5,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        sm_margin: int = 0,
    ) -> None:
        super().__init__()
        self.eps: float = eps
        self.zero_centered_gamma: bool = zero_centered_gamma

        # Parameter shape
        if not isinstance(normalized_shape, Iterable):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        # Parameter device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True

        # Initialize parameters if needed
        weight = torch.empty(
            normalized_shape,
            device=device,
            dtype=canonicalize_dtype(dtype),
        )
        weight = torch.nn.Parameter(weight)
        self.weight: torch.nn.Parameter
        self.register_parameter("weight", weight)
        if not defer_param_init:
            self.reset_parameters()

        # Note: SM margin is CUDA-specific and not supported on NPU
        # Keep the parameter for API compatibility but ignore it
        if sm_margin != 0:
            warnings.warn(
                "sm_margin parameter is CUDA-specific and has no effect on NPU. "
                "It is kept for API compatibility only.",
                UserWarning,
                stacklevel=2,
            )
        self._sm_margins: dict[str, int] = {"forward": 0, "backward": 0, "inference": 0}

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Parameter device
        weight = self.weight
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Initialize param buffers
        if not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)

        # Initialize values
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(weight)
        else:
            torch.nn.init.ones_(weight)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()
        if self.weight.device.type == "meta":
            self.reset_parameters()

    # pylint: disable=arguments-differ  # TE operation uses a fixed forward signature.
    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        # Check tensor dims
        weight = self.weight
        weight_dims = tuple(weight.size())
        input_dims = tuple(input_.size())
        if len(input_dims) < len(weight_dims) or input_dims[-len(weight_dims) :] != weight_dims:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) are not compatible"
            )

        # Check input tensors
        inner_dim = math.prod(weight_dims)
        dtype = maybe_autocast_dtype(default_dtype=weight.dtype)
        x = maybe_dequantize(input_.contiguous(), dtype).view((-1, inner_dim))
        w = maybe_dequantize(self.weight, dtype).view((inner_dim,))

        # Handle zero_centered_gamma
        if self.zero_centered_gamma:
            w = w.float() + 1.0

        # Compute RMSNorm using torch_npu
        # Note: torch_npu.npu_rms_norm returns (y, rstd)
        # where rstd is the reciprocal of standard deviation
        y, rstd = torch_npu.npu_rms_norm(x, w, epsilon=self.eps)

        # Save state for backward pass
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x, rstd)
            ctx.save_for_backward(x, rstd)
            ctx.dtype = dtype

        # Reshape output tensor
        out = y.view(input_dims)
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        # Saved tensors from forward pass
        x, rstd = ctx.saved_tensors

        # Tensor dims
        weight_dims = self.weight.size()
        inner_dim = math.prod(weight_dims)
        # Check input tensors
        dtype = ctx.dtype
        dy = maybe_dequantize(grad_output.contiguous(), dtype).view(x.size())
        w = maybe_dequantize(self.weight, dtype).view((inner_dim,))

        # Handle zero_centered_gamma: must match forward's w = 1 + weight
        if self.zero_centered_gamma:
            w = w.float() + 1.0

        dx, dw = torch_npu.npu_rms_norm_backward(dy, x, w, rstd)

        # Clear saved tensors if possible

        clear_tensor_data(x)
        clear_tensor_data(rstd)

        # Reshape results
        grad_input = dx.view(grad_output.size())
        grad_weight = dw.to(self.weight.dtype).view(weight_dims)
        return grad_input, (grad_weight,)

    def op_onnx_forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Every operand in this function has a defined ONNX translation."""
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        variance = input_.pow(2).mean(-1, keepdim=True)
        normalized = input_ * torch.rsqrt(variance + self.eps)
        return normalized * weight
