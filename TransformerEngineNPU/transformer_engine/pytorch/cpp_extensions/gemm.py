# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
"""GEMM entry points of the ``cpp_extensions`` CUDA-compatibility surface.

Upstream CUDA TE re-exports its C++ GEMM binding from here, and downstream
projects import it directly: Megatron-LM, for example, builds its
``te_general_gemm`` helper from ``cpp_extensions.general_gemm``. On NPU that
binding does not exist, so the import fails and the callers silently degrade --
the Megatron-FSDP fused-wgrad path would then fall back to
``torch.matmul(..., out=)``, which cannot write an fp32 ``main_grad`` from bf16
operands.

``general_gemm`` covers the dense subset used by Megatron on top of the NPU
usage-based GEMM primitives.
"""

from typing import Iterable, Optional

import torch

from ..constants import TensorUsage
from ..ops.gemm import general_gemm as _usage_general_gemm
from ..ops.gemm import general_gemm_add as _usage_general_gemm_add

__all__ = ["general_gemm"]

_GEMM_LAYOUTS = ("TN", "NN", "NT")


def _validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Whether a GEMM scaling factor is consistent with its usage (mirrors CUDA TE)."""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def _layout_usages(layout: str):
    """Translate a CUDA TE ``layout`` string into NPU usage flags.

    CUDA TE computes ``op(B) @ op(A)`` with ``layout = (op(A), op(B))`` while the
    NPU usage-based GEMM computes ``op(X) @ op(Y)``: swap the operands and take the
    transpose flags in reverse order.
    """
    if layout not in _GEMM_LAYOUTS:
        raise ValueError(f"GEMM layout {layout} not supported.")
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    usage_x = TensorUsage.LHS_TRANS if transb else TensorUsage.LHS
    usage_y = TensorUsage.RHS_TRANS if transa else TensorUsage.RHS
    return usage_x, usage_y


def _is_dense_float(tensor) -> bool:
    return isinstance(tensor, torch.Tensor) and tensor.is_floating_point()


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params=None,
    gelu: bool = False,
    gelu_in: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub=None,
    ub_type=None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """Dense NPU counterpart of CUDA TE's ``general_gemm``.

    Mirrors the CUDA contract for the subset Megatron uses: ``result = op(B) @ op(A)``
    with ``layout = (op(A), op(B))`` among ``TN``/``NN``/``NT``; ``alpha`` scales the
    product and ``beta`` the incoming ``out`` (the latter must be zero unless
    ``accumulate`` is set); and the return value is the CUDA 4-tuple
    ``(out, bias_grad, gelu_input, extra_output)``.

    The CUDA-only extras -- fused GELU, userbuffers, extra outputs, split
    accumulators, ``out`` combined with ``bias``, and the fused bias gradient -- are
    rejected loudly instead of being silently ignored. ``grad`` is the upstream hint
    that the second operand carries a weight gradient, which the dense path does not
    need.
    """
    if out is not None and not out.is_contiguous():
        raise ValueError("Output tensor is not contiguous.")
    if accumulate and out is None:
        raise ValueError("general_gemm requires out= when accumulate is True.")
    if out is not None and bias is not None:
        raise NotImplementedError("general_gemm does not support out= together with bias")
    if gelu or gelu_in is not None:
        raise NotImplementedError("general_gemm does not support the fused gelu epilogue")
    if ub is not None or ub_type is not None or extra_output is not None or bulk_overlap:
        raise NotImplementedError("general_gemm does not support userbuffers or extra outputs")
    if use_split_accumulator:
        raise NotImplementedError("general_gemm does not support split accumulators")
    if grad and bias is not None:
        raise NotImplementedError("general_gemm does not produce the fused bias gradient")

    alpha = _validate_gemm_scale(alpha, True)
    beta = _validate_gemm_scale(beta, accumulate)
    usage_x, usage_y = _layout_usages(layout)

    if out is not None and alpha == 1.0:
        # Fused accumulation fast path (Megatron's default): the primitive
        # accumulates in fp32 into the fp32 `out`.
        if beta == 0.0:
            out.zero_()
        _usage_general_gemm_add(
            out,
            B,
            A,
            usage_x,
            usage_y,
            out_dtype if out_dtype is not None else out.dtype,
            quantization_params,
        )
        return out, None, None, None

    res_dtype = out_dtype if out_dtype is not None else B.dtype
    # The dense primitive ignores out_dtype, so cast explicitly to keep the math in
    # the requested precision. Quantized operands carry their own precision.
    if _is_dense_float(A) and _is_dense_float(B):
        A, B = A.to(res_dtype), B.to(res_dtype)
    if bias is not None:
        # addmm requires the bias and the product to share a dtype, and the bias is
        # accumulated in the requested precision. Without this cast the fp32 MoE
        # router scoring (out_dtype=fp32 with a bf16 bias) would degrade to bf16.
        bias = bias.to(res_dtype)
    result = _usage_general_gemm(
        B, A, usage_x, usage_y, res_dtype, bias=bias, quantization_params=quantization_params
    )
    if alpha != 1.0:
        result = result * alpha
    if out is not None:
        out.mul_(beta).add_(result.to(out.dtype))
        return out, None, None, None
    return result, None, None, None
