# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for Layer Normalization."""

from __future__ import annotations
from collections.abc import Iterable
import math
import os
from typing import Optional

import torch

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

__all__ = ["LayerNorm"]


class LayerNorm(BasicOperation):
    r"""Layer Normalization

    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform
    parameters that match the inner-most dimensions of the input tensor.

    Parameters
    ----------
    normalized_shape : int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator of layer normalization for
        numerical stability
    device : torch.device, default = default CUDA device
        Tensor device
    dtype : torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = False
        If ``True``, the :math:`\gamma` parameter is initialized to zero
        and the calculation changes to

            .. math::
                y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma) + \beta

    sm_margin : int or dict, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.
        For more fine-grained control, provide a dict with the SM
        margin at each compute stage ("forward", "backward",
        "inference"). NOTE: SM margin is CUDA-specific and has no effect
        on NPU; kept for API compatibility with NVTE.

    """

    def __init__(
        self,
        normalized_shape: Iterable[int] | int,
        *,
        eps: float = 1e-5,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        sm_margin: int | dict[str, int] = 0,
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
        dtype = canonicalize_dtype(dtype)
        weight = torch.empty(
            normalized_shape,
            device=device,
            dtype=dtype,
        )
        bias = torch.empty(
            normalized_shape,
            device=device,
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        self.weight: torch.nn.Parameter
        self.bias: torch.nn.Parameter
        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)
        if not defer_param_init:
            self.reset_parameters()

        # Number of SMs to exclude when launching CUDA kernels (CUDA-specific;
        # kept for API compatibility with NVTE, has no effect on NPU).
        self._sm_margins: dict[str, int]
        if isinstance(sm_margin, dict):

            def getenv(name: str) -> int:
                return int(os.getenv(name, "0"))

            self._sm_margins = {
                "forward": sm_margin.get("forward", getenv("NVTE_FWD_LAYERNORM_SM_MARGIN")),
                "backward": sm_margin.get("backward", getenv("NVTE_BWD_LAYERNORM_SM_MARGIN")),
                "inference": sm_margin.get("inference", getenv("NVTE_INF_LAYERNORM_SM_MARGIN")),
            }
        else:

            def getenv(name: str) -> int:
                return int(os.getenv(name, str(sm_margin)))

            self._sm_margins = {
                "forward": getenv("NVTE_FWD_LAYERNORM_SM_MARGIN"),
                "backward": getenv("NVTE_BWD_LAYERNORM_SM_MARGIN"),
                "inference": getenv("NVTE_INF_LAYERNORM_SM_MARGIN"),
            }

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Parameter device
        weight = self.weight
        bias = self.bias
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Initialize param buffers
        if not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)
        if not devices_match(bias.device, device):
            bias = torch.empty_like(bias, device=device)

        # Initialize values
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(weight)
        else:
            torch.nn.init.ones_(weight)
        torch.nn.init.zeros_(bias)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.weight = weight
        self.bias = bias

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()
        if self.weight.device.type == "meta" or self.bias.device.type == "meta":
            self.reset_parameters()

    # pylint: disable=arguments-differ  # override intentionally simplifies base signature (matches NVTE)
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
        b = maybe_dequantize(self.bias, dtype).view((inner_dim,))

        # zero_centered_gamma: the effective weight is (1 + gamma).
        # aclnnLayerNorm requires matching dtypes, so promote the computation to
        # fp32 (matches the (1+gamma) math at higher precision) and cast the output
        # back to the compute dtype.
        if self.zero_centered_gamma:
            x = x.float()
            w = w.float() + 1.0
            b = b.float()

        # Compute LayerNorm via torch_npu
        # Note: torch.native_layer_norm returns (y, means, rstdevs) where rstdevs is
        # the reciprocal standard deviation (1/sqrt(var + eps)), not the raw variance.
        # On bf16 input the kernel rounds the saved statistics to bf16, while
        # NVTE's fused kernel keeps them in fp32. Since dgamma/dbeta multiply
        # every element by these stats, a bf16 rstd injects a ~0.4% correlated
        # error into the parameter grads (matched_ratio ~0.986 on 4096-token
        # rows, REPORT atk §3#16). Promote the whole computation to fp32 so the
        # saved stats and the backward math are full-precision on every path
        # (the zc path already did this); the bf16 forward output is unchanged
        # (only cast timing differs, one rounding either way).
        if not self.zero_centered_gamma and x.dtype != torch.float32:
            x = x.float()
            w = w.float()
            b = b.float() if b is not None else None
        y, means, rstdevs = torch.native_layer_norm(x, (inner_dim,), w, b, self.eps)

        # Save state for backward pass
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x, means, rstdevs)
            ctx.save_for_backward(x, means, rstdevs)
            ctx.dtype = dtype

        # Reshape output tensor (cast back to the compute dtype for the fp32 path)
        out = y.to(dtype).view(input_dims)
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        # Saved tensors from forward pass
        x, means, rstdevs = ctx.saved_tensors

        # Tensor dims
        weight_dims = self.weight.size()
        inner_dim = math.prod(weight_dims)

        # Check input tensors
        dtype = ctx.dtype
        dy = maybe_dequantize(grad_output.contiguous(), dtype).view(x.size())
        w = maybe_dequantize(self.weight, dtype).view((inner_dim,))
        b = maybe_dequantize(self.bias, dtype).view((inner_dim,))

        # Match the forward's precision regime: the forward promotes the whole
        # computation to fp32 (zero_centered_gamma's (1 + gamma), and the bf16
        # non-zc path for full-precision saved statistics — a bf16 rstd injects
        # ~0.4% into every dgamma/dbeta product), so the saved x/means/rstdevs
        # are fp32 and the backward math runs at fp32 too.
        if x.dtype == torch.float32 and dtype != torch.float32:
            dy = dy.float()
            w = w.float()
            if b is not None:
                b = b.float()
        if self.zero_centered_gamma:
            w = w.float() + 1.0

        # Use torch's native_layer_norm_backward (the exact primitive that
        # F.layer_norm's autograd dispatches to, i.e. aclnnLayerNormBackward) so
        # training is bit-identical to the torch reference. The mask is always all-
        # True because the fuser needs grad_input/grad_weight/grad_bias.
        dx, dw, db = torch.ops.aten.native_layer_norm_backward(
            dy, x, (inner_dim,), means, rstdevs, w, b, [True, True, True]
        )

        # Clear saved tensors if possible
        clear_tensor_data(x)
        clear_tensor_data(means)
        clear_tensor_data(rstdevs)

        # Reshape results (cast back to the compute dtype for the fp32 zero_centered
        # path; a no-op for the plain bf16 path)
        grad_input = dx.to(dtype).view(grad_output.size())
        grad_weight = dw.to(self.weight.dtype).view(weight_dims)
        grad_bias = db.to(self.bias.dtype).view(weight_dims)
        return grad_input, (grad_weight, grad_bias)

    def op_onnx_forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Every operand in this function has a defined ONNX translation."""
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        return torch.nn.functional.layer_norm(
            input_, input_.shape[-1:], weight, self.bias, self.eps
        )
