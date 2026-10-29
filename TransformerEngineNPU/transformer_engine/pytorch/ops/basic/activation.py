# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fusible activation operations for NPU.

This module provides concrete FusibleOperation classes for all 12 activation
functions supported by TransformerEngine. Each class inherits from
_ActivationOperation and implements the abstract methods
_activation_forward_impl and _activation_backward_impl by delegating to
the pure-PyTorch implementations in act_impl.py.

These classes enable the TE operation fusion pipeline (Sequential + fuser)
to work on NPU, and provide the public API symbols:
    te.pytorch.ops.GELU, te.pytorch.ops.SiLU, te.pytorch.ops.ReLU,
    te.pytorch.ops.QGELU, te.pytorch.ops.SReLU, te.pytorch.ops.GLU,
    te.pytorch.ops.GEGLU, te.pytorch.ops.SwiGLU, te.pytorch.ops.ReGLU,
    te.pytorch.ops.QGEGLU, te.pytorch.ops.SReGLU, te.pytorch.ops.ClampedSwiGLU
"""

# Activation hooks intentionally specialize the generic BasicOperation signatures.
# pylint: disable=arguments-differ

from __future__ import annotations

import torch
from typing import Any, Iterable, Optional

from .npu_activation import (
    gelu_fwd,
    gelu_bwd,
    silu_fwd,
    silu_bwd,
    relu_fwd,
    relu_bwd,
    qgelu_fwd,
    qgelu_bwd,
    srelu_fwd,
    srelu_bwd,
    glu_fwd,
    glu_bwd,
    geglu_fwd,
    geglu_bwd,
    reglu_fwd,
    reglu_bwd,
    sreglu_fwd,
    sreglu_bwd,
    qgeglu_fwd,
    qgeglu_bwd,
    clamped_swiglu_fwd,
    clamped_swiglu_bwd,
)
from ..op import BasicOperation, OperationContext
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor.float8_tensor import Float8CurrentScalingQuantizer, Quantizer
from ...utils import clear_tensor_data
from .._common import maybe_dequantize
import abc


__all__ = [
    "GELU",
    "GEGLU",
    "GLU",
    "QGELU",
    "QGEGLU",
    "ReLU",
    "ReGLU",
    "SReLU",
    "SReGLU",
    "ScaledSReLU",
    "SiLU",
]


class _ActivationOperation(BasicOperation, metaclass=abc.ABCMeta):
    r"""Apply activation function

    Activation functions are either element-wise unary functions or
    variants of the gated linear unit (GLU). Recall that GLU is
    computed by splitting the input tensor into chunks :math:`a` and
    :math:`b` along the last dimension and computing

    .. math::
       \text{GLU}(a,b) = \sigma(a) * b

    .. warning::

       Transformer Engine gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    Parameters
    ----------
    cache_quantized_input : bool, default = False
        Quantize input tensor when caching for use in the backward
        pass. This will typically reduce memory usage but require
        extra compute and increase numerical error. This feature is
        highly experimental.

    """

    def __init__(self, *, cache_quantized_input: bool = False):
        super().__init__()
        self.cache_quantized_input: bool = cache_quantized_input

    @abc.abstractmethod
    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        """Forward implementation"""

    @abc.abstractmethod
    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        """Backward implementation"""

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        # Compute dtype
        dtype: torch.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("npu")
        else:
            dtype = input_.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")

        # Check input tensor
        x = maybe_dequantize(input_.contiguous(), dtype)

        # Launch kernel
        y = self._activation_forward_impl(x)

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(torch.float8_e4m3fn, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)

        # Save state for backward pass
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x)
            ctx.save_for_backward(x)
            ctx.dtype = dtype
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        # Saved tensors from forward pass
        (x,) = ctx.saved_tensors

        # Check input tensor
        x = maybe_dequantize(x.contiguous(), ctx.dtype)

        # Check grad output tensor
        dy = maybe_dequantize(grad_output.contiguous(), x.dtype)

        # Launch kernel
        dx = self._activation_backward_impl(dy, x)

        # Clear input tensor if possible
        clear_tensor_data(x)

        return dx, ()


# =============================================================================
# Element-wise activation FusibleOperation classes
# =============================================================================


class GELU(_ActivationOperation):
    r"""Apply the Gaussian Error Linear Unit (GELU) activation function.

    .. math::
       \text{GELU}(x) = \frac{x}{2} \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}
       \left(x + 0.044715 x^3\right)\right)\right)
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return gelu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return gelu_bwd(x, dy)


class SiLU(_ActivationOperation):
    r"""Apply the Sigmoid Linear Unit (SiLU) activation function.

    .. math::
       \text{SiLU}(x) = x \cdot \sigma(x)
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return silu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return silu_bwd(x, dy)


class ReLU(_ActivationOperation):
    r"""Apply the Rectified Linear Unit (ReLU) activation function.

    .. math::
       \text{ReLU}(x) = \max(x, 0)
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return relu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return relu_bwd(x, dy)


class QGELU(_ActivationOperation):
    r"""Apply the Quick GELU activation function (tanh approximation).

    .. math::
       \text{QGELU}(x) = x \cdot \sigma(1.702 x)
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return qgelu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return qgelu_bwd(x, dy)


class SReLU(_ActivationOperation):
    r"""Apply the Squared ReLU activation function.

    .. math::
       \text{SReLU}(x) = \max(x, 0)^2
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return srelu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return srelu_bwd(x, dy)


class ScaledSReLU(BasicOperation):
    r"""Squared ReLU with per-row post-scaling.

    This is the scaled variant of :class:`SReLU` used by the Megatron fused
    grouped-MLP MoE path (``megatron/core/transformer/moe/experts.py``). If the
    SReLU output has shape ``(d_1, ..., d_n)``, it is multiplied with an extra
    input tensor (the per-row scale) of shape ``(d_1, ..., d_{n-1}``).

    Mirrors the NVIDIA Transformer Engine semantics: the scale is a plain
    ``maybe_dequantize``'d tensor, so the op runs in BF16/FP16/FP32 and also
    accepts FP8-quantized inputs when available. It does not require FP8.

    Parameters
    ----------
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.
    """

    num_extra_inputs: int = 1

    def __init__(self, *, activation_recompute_in_mlp: bool = False) -> None:
        super().__init__()
        self.activation_recompute_in_mlp: bool = activation_recompute_in_mlp

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        extra_input = basic_op_extra_inputs[0][0]

        # Compute dtype
        dtype: torch.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("npu")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        x = maybe_dequantize(input_.contiguous(), dtype)
        scales = maybe_dequantize(extra_input, dtype)
        y = srelu_fwd(x) * scales.unsqueeze(-1)

        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(x, scales)

        return y, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        del basic_op_grad_extra_outputs

        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        ctx = basic_op_ctxs[0]
        x, scales = ctx.saved_tensors
        x = maybe_dequantize(x.contiguous(), ctx.dtype)
        scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output.contiguous(), ctx.dtype)

        grad_input = None
        if ctx.input_requires_grad:
            grad_srelu_out = grad_output * scales.unsqueeze(-1)
            grad_input = srelu_bwd(x, grad_srelu_out)

        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            srelu_out = srelu_fwd(x)
            grad_extra_input = torch.linalg.vecdot(srelu_out, grad_output)

        clear_tensor_data(ctx.saved_tensors[0])

        return grad_input, [()], [(grad_extra_input,)]


class GLU(_ActivationOperation):
    r"""Apply the Gated Linear Unit (GLU) activation function.

    Input is split into :math:`a` and :math:`b` along the last dimension.

    .. math::
       \text{GLU}(a, b) = \sigma(a) \cdot b
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return glu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return glu_bwd(x, dy)


# =============================================================================
# Gated (GLU) activation FusibleOperation classes
# =============================================================================


class GEGLU(_ActivationOperation):
    r"""Apply the GELU Gated Linear Unit (GEGLU) activation function.

    Input is split into :math:`a` and :math:`b` along the last dimension.

    .. math::
       \text{GEGLU}(a, b) = \text{GELU}(a) \cdot b
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return geglu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return geglu_bwd(x, dy)


class ReGLU(_ActivationOperation):
    r"""Apply the ReLU Gated Linear Unit (ReGLU) activation function.

    Input is split into :math:`a` and :math:`b` along the last dimension.

    .. math::
       \text{ReGLU}(a, b) = \max(a, 0) \cdot b
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return reglu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return reglu_bwd(x, dy)


class QGEGLU(_ActivationOperation):
    r"""Apply the Quick GELU Gated Linear Unit (QGEGLU) activation function.

    Input is split into :math:`a` and :math:`b` along the last dimension.

    .. math::
       \text{QGEGLU}(a, b) = \text{QGELU}(a) \cdot b
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return qgeglu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return qgeglu_bwd(x, dy)


class SReGLU(_ActivationOperation):
    r"""Apply the Squared ReLU Gated Linear Unit (SReGLU) activation function.

    Input is split into :math:`a` and :math:`b` along the last dimension.

    .. math::
       \text{SReGLU}(a, b) = \max(a, 0)^2 \cdot b
    """

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return sreglu_fwd(x)

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return sreglu_bwd(x, dy)


class ClampedSwiGLU(_ActivationOperation):
    r"""Apply the NVTE-compatible Clamped SwiGLU activation function.

    Input is split into :math:`a` and :math:`b` along the last dimension.
    The activation input is upper-clamped, while the linear input is
    symmetrically clamped and shifted by ``glu_linear_offset``.

    .. math::
       \text{ClampedSwiGLU}(a, b) =
       \min(a, l)\sigma(\alpha\min(a, l))
       (\text{clamp}(b, -l, l) + o)

    Parameters
    ----------
    limit : float, default = 7.0
        Clamping threshold.
    alpha : float, default = 1.702
        Scaling factor for the sigmoid function.
    glu_linear_offset : float, default = 1.0
        Offset added to the clamped linear component.
    glu_interleave_size : int, optional
        Experimental block-interleaved GLU layout. This layout is not yet
        supported by the NPU activation backend.
    """

    def __init__(
        self,
        *,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        if glu_interleave_size is not None:
            raise NotImplementedError(
                "ClampedSwiGLU glu_interleave_size is not supported on the NPU backend"
            )
        super().__init__(cache_quantized_input=cache_quantized_input)
        self.limit = limit
        self.alpha = alpha
        self.glu_linear_offset = glu_linear_offset
        self.glu_interleave_size = glu_interleave_size

    def _activation_forward_impl(self, x, next_op_input_quantizer=None):
        return clamped_swiglu_fwd(
            x,
            limit=self.limit,
            alpha=self.alpha,
            glu_linear_offset=self.glu_linear_offset,
        )

    def _activation_backward_impl(self, dy, x, prev_op_grad_output_quantizer=None):
        return clamped_swiglu_bwd(
            x,
            dy,
            limit=self.limit,
            alpha=self.alpha,
            glu_linear_offset=self.glu_linear_offset,
        )
