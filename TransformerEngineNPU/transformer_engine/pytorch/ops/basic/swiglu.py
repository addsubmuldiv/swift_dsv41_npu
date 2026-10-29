# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for SwiGLU and variants."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from .activation import ClampedSwiGLU
from .npu_activation import (
    swiglu_bwd,
    swiglu_fwd,
)

from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Float8CurrentScalingQuantizer, Quantizer
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

__all__ = ["SwiGLU", "ScaledSwiGLU", "ScaledClampedQGeGLU"]


class SwiGLU(BasicOperation):
    r"""Swish gated linear unit

    The input tensor is split into chunks :math:``a`` and :math:``b``
    along the last dimension and the following is computed:

    .. math::

       \text{SwiGLU}(a,b) = \text{SiLU}(a) * b

    where

    .. math::

       \text{SiLU}(x) = x \sigma(x) = \frac{x}{1+\exp(-x)}

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:``a`` and
       :math:``b``. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    The Sigmoid Linear Unit (SiLU) gating function is also known as
    the swish function. See
    `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__.

    Parameters
    ----------
    cache_quantized_input : bool, default = False
        Quantize input tensor when caching for use in the backward
        pass. This will typically reduce memory usage but require
        extra compute and increase numerical error. This feature is
        highly experimental.
    glu_interleave_size : int, optional
        When set, the GLU activations will use a block interleaved
        format. Instead of interpreting the input tensor as a
        concatenation of gates and linear units (e.g.
        :math:``[a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4]``
        in the above notation), it will be interpreted
        as alternating blocks of gates and linear units (e.g.
        :math:``[a_1, a_2, b_1, b_2, a_3, a_4, b_3, b_4]``
        when the interleave size is 2). This data format is highly
        experiental and is primarily intended to support some advanced
        fused kernels.

    """

    def __init__(
        self,
        *,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    def op_forward(  # pylint: disable=arguments-differ  # same signature pattern as every basic op (bias.py etc.): the base class declares **kwargs but no override accepts it; the dispatcher (op.py fuser_forward) passes both quantizers by keyword
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
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
        input_ = maybe_dequantize(input_.contiguous(), dtype)

        # Remove interleaving if needed
        swiglu_in = input_
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Launch kernel
        out = swiglu_fwd(swiglu_in)

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(
                torch.float8_e4m3fn,
                input_.device,
            )
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            input_ = input_quantizer(input_)

        # Save state for backward pass
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.save_for_backward(input_)
            ctx.dtype = dtype
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        # Saved tensors from forward pass
        (input_,) = ctx.saved_tensors

        # Make sure tensors have correct dtypes
        x = maybe_dequantize(input_.contiguous(), ctx.dtype)
        dy = maybe_dequantize(grad_output.contiguous(), ctx.dtype)

        # Remove interleaving if needed
        swiglu_in = x
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Launch kernel
        grad_swiglu_in = swiglu_bwd(swiglu_in, dy)

        # Apply interleaving if needed
        dx = grad_swiglu_in
        if self.glu_interleave_size is not None:
            shape = dx.size()
            dx = dx.reshape(
                -1,
                2,
                shape[-1] // (2 * self.glu_interleave_size),
                self.glu_interleave_size,
            )
            dx = dx.transpose(1, 2).contiguous()
            dx = dx.view(shape)

        # Clear input tensor if possible
        clear_tensor_data(input_)

        return dx, ()


class _ScaledGLU(BasicOperation):
    """SwiGLU-family activation with per-row scales (fused grouped MLP middle op).

    Mirrors the NVIDIA Transformer Engine ``_ScaledGLU`` used by the Megatron
    fused grouped-MLP MoE path. The GLU output (shape ``(d_1, ..., d_n)``) is
    multiplied with an extra input tensor (the per-row scale) of shape
    ``(d_1, ..., d_{n-1}``). The scale is a plain ``maybe_dequantize``'d tensor,
    so the op runs in BF16/FP16/FP32 and also accepts FP8-quantized inputs when
    available. It does not require FP8.
    """

    num_extra_inputs: int = 1

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
    ) -> None:
        super().__init__()
        if glu_interleave_size is not None:
            raise NotImplementedError("glu_interleave_size is not supported on the NPU backend")
        self.glu_interleave_size: Optional[int] = glu_interleave_size
        self.activation_recompute_in_mlp: bool = activation_recompute_in_mlp

    def _glu_forward(self, swiglu_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _glu_backward(
        self,
        grad_swiglu_out: torch.Tensor,
        swiglu_in: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

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
        swiglu_out = self._glu_forward(x)
        out = swiglu_out * scales.unsqueeze(-1)

        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(x, scales)

        return out, [()]

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
            grad_swiglu_out = grad_output * scales.unsqueeze(-1)
            grad_input = self._glu_backward(grad_swiglu_out, x)

        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            swiglu_out = self._glu_forward(x)
            grad_extra_input = torch.linalg.vecdot(swiglu_out, grad_output)

        clear_tensor_data(ctx.saved_tensors[0])

        return grad_input, [()], [(grad_extra_input,)]


class ScaledSwiGLU(_ScaledGLU):
    r"""SwiGLU with post-scaling (matches cuDNN grouped GEMM ``act_func="swiglu"``).

    If the GLU output has shape ``(d_1, ..., d_n)``, it is multiplied
    with an extra input tensor of shape ``(d_1, ..., d_{n-1}``).

    Parameters
    ----------
    glu_interleave_size : int, optional
        Experimental block-interleaved GLU layout. Not supported on the NPU
        backend (raises ``NotImplementedError``).
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.
    """

    def _glu_forward(self, swiglu_in: torch.Tensor) -> torch.Tensor:
        return swiglu_fwd(swiglu_in)

    def _glu_backward(
        self,
        grad_swiglu_out: torch.Tensor,
        swiglu_in: torch.Tensor,
    ) -> torch.Tensor:
        return swiglu_bwd(swiglu_in, grad_swiglu_out)


class ScaledClampedQGeGLU(_ScaledGLU):
    r"""Clamped QGeGLU with post-scaling
    (matches cuDNN grouped GEMM ``act_func="geglu"``).

    Same layout and scaling contract as :class:`ScaledSwiGLU`, but the GLU
    uses :class:`ClampedSwiGLU` numerics (default ``limit`` / ``alpha`` match
    cuDNN).

    Parameters
    ----------
    glu_interleave_size : int, optional
        Experimental block-interleaved GLU layout. Not supported on the NPU
        backend (raises ``NotImplementedError``).
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.
    limit : float, default ``7.0``
        Clamp limit (see :class:`ClampedSwiGLU`).
    alpha : float, default ``1.702``
        Sigmoid scale (see :class:`ClampedSwiGLU`).
    glu_linear_offset : float, default ``1.0``
        Offset added to the linear component after clamping
        (see :class:`ClampedSwiGLU`).
    """

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
    ) -> None:
        super().__init__(
            glu_interleave_size,
            activation_recompute_in_mlp=activation_recompute_in_mlp,
        )
        self._clamped: ClampedSwiGLU = ClampedSwiGLU(
            limit=limit,
            alpha=alpha,
            glu_linear_offset=glu_linear_offset,
        )

    def _glu_forward(self, swiglu_in: torch.Tensor) -> torch.Tensor:
        return self._clamped._activation_forward_impl(swiglu_in)

    def _glu_backward(
        self,
        grad_swiglu_out: torch.Tensor,
        swiglu_in: torch.Tensor,
    ) -> torch.Tensor:
        return self._clamped._activation_backward_impl(grad_swiglu_out, swiglu_in)
