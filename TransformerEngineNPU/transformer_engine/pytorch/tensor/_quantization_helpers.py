# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


def _pack_fp8_tensor_for_pickle(
    tensor: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Pack an FP8 tensor as CPU uint8 bytes for pickling."""
    if tensor is None:
        return None
    return tensor.detach().contiguous().view(torch.uint8).cpu()


def _pack_tensor_for_pickle(
    tensor: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Copy tensor metadata to CPU for pickling."""
    if tensor is None:
        return None
    return tensor.detach().clone().cpu()


def _restore_fp8_tensor_from_pickle(
    tensor: Optional[torch.Tensor],
    fp8_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Restore an FP8 tensor from a uint8 pickle payload."""
    if tensor is not None and tensor.dtype == torch.uint8:
        return tensor.view(dtype=fp8_dtype)
    return tensor


class _IdentityFunc(torch.autograd.Function):
    """Identity function

    If constructor keyword-arguments are provided, then construct a
    new Float8Tensor using the provided tensor's attributes.

    """

    @staticmethod
    def forward(
        ctx, tensor: "QuantizedTensor", init_kwargs: Optional[Dict[str, Any]] = None
    ) -> "QuantizedTensor":
        # pylint: disable=missing-function-docstring

        # Return input tensor if constructor kwargs are not provided
        if init_kwargs is None:
            return tensor.detach()

        # Construct new tensor if constructor kwargs are provided
        ctx.input_dtype = tensor.dtype
        kwargs = tensor.get_metadata()
        for key, val in init_kwargs.items():
            kwargs[key] = val
        return type(tensor)(tensor.origin_shape, tensor.dtype, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        grad_input = grad_output
        if grad_input.dtype == ctx.input_dtype:
            grad_input = grad_input.detach()
        else:
            grad_input = grad_input.to(ctx.input_dtype)
        return grad_input, None
