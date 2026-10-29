# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations

from typing import Optional

import torch

from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage
from ..torch_version import torch_version
from ..utils import canonicalize_dtype


def is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Check if tensor is a quantized tensor."""

    return isinstance(tensor, (QuantizedTensor, QuantizedTensorStorage))


def maybe_dequantize(
    tensor: torch.Tensor | QuantizedTensorStorage, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize tensor to given dtype or just convert if not a quantized tensor"""
    if is_quantized_tensor(tensor):
        return tensor.dequantize(dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def maybe_autocast_dtype(
    *,
    device_type: str = "cuda",
    default_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """Get autocast dtype if enabled"""

    if torch_version() >= (2, 4, 3):
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    else:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return canonicalize_dtype(default_dtype)


def get_main_grad_from_param(
    weight_param: torch.nn.Parameter,
    *,
    op_label: str = "",
) -> torch.Tensor:
    """Refresh ``main_grad`` from FSDP if needed and return it."""
    if hasattr(weight_param, "__fsdp_param__"):
        weight_param.main_grad = weight_param.get_main_grad()
    if not hasattr(weight_param, "main_grad") or weight_param.main_grad is None:
        prefix = f"{op_label} " if op_label else ""
        raise RuntimeError(
            f"{prefix}operation is configured with accumulate_into_main_grad=True, "
            "but weight parameter does not have a valid main_grad attribute"
        )
    return weight_param.main_grad


def get_accumulate_flag_in_param(weight_param: torch.nn.Parameter) -> bool:
    """Return whether wgrad GEMM should accumulate into ``main_grad``."""
    return not getattr(weight_param, "overwrite_main_grad", False)


def get_dummy_wgrads_for_params(
    weight_params: list[torch.nn.Parameter],
) -> list[Optional[torch.Tensor]]:
    """Build dummy ``.grad`` placeholders for params using ``main_grad``."""
    from ..module.base import setup_dummy_wgrad  # pylint: disable=import-outside-toplevel

    out: list[Optional[torch.Tensor]] = []
    for weight_param in weight_params:
        if hasattr(weight_param, "grad_added_to_main_grad"):
            main_grad = get_main_grad_from_param(weight_param)
            out.append(setup_dummy_wgrad(weight_param, main_grad))
        else:
            out.append(None)
    return out
