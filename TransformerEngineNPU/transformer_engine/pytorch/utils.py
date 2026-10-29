# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Utility functions for TransformerEngine Ascend PyTorch API

This module provides utility functions for NPU-optimized operations,
integrating MindSpeed's utilities with TransformerEngine's interface.
"""

import contextlib
from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch_npu

from .constants import NPUVersion, QuantDtype


@lru_cache(maxsize=None)
def triton_available() -> bool:
    """Return whether Triton and its language module can be imported."""
    with contextlib.suppress(ImportError):
        import triton  # noqa: F401
        import triton.language  # noqa: F401

        return True
    return False


@lru_cache(maxsize=None)
def get_cached_ones_tensor(
    num_elements: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a cached ``torch.ones`` tensor.
    Tensors are cached by ``(num_elements, dtype, device)`` and kept alive
    by the cache, ensuring stable data pointers across CUDA graph replays.
    """
    return torch.ones(num_elements, dtype=dtype, device=device)


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype if needed."""
    if tensor is None:
        return None
    if tensor.dtype == dtype:
        return tensor
    with torch.enable_grad():
        return tensor.to(dtype=dtype)


@lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().npu()


def clear_tensor_data(*tensors: Tuple[Optional[torch.Tensor], ...]) -> None:
    """
    Trick to deallocate tensor memory when delete operation does not
    release the tensor due to PyTorch override.

    Must be used carefully.
    """
    for t in tensors:
        if t is not None:
            # Workaround for double buffering in cpu offload
            if hasattr(t, "_do_not_clear"):
                continue
            if hasattr(t, "get_data_tensors"):
                if any(hasattr(tensor, "_do_not_clear") for tensor in t.get_data_tensors()):
                    continue

            if hasattr(t, "clear"):
                t.clear()
            else:
                t.data = _empty_tensor()
            del t


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def mark_grouped_tensor(*tensors: List[Any]):
    """
    Needed for paged stashing in Megatron-LM. This attribute allows
    Megatron-LM to detect which tensors are dynamic (varying shapes)
    and remove the padding before doing the `save_for_backward` to
    save memory.
    Note: Only columnwise data is saved for backward.
    """
    for tensor in tensors:
        if tensor is None:
            continue
        if hasattr(tensor, "columnwise_data"):
            assert tensor.columnwise_data is not None, (
                "Columnwise data is not set for grouped tensor"
            )
            assert tensor.columnwise_scale_inv is not None, (
                "Columnwise scale inverse is not set for grouped tensor"
            )
            setattr(tensor.columnwise_data, "grouped_tensor_scale_inv", False)
            setattr(tensor.columnwise_scale_inv, "grouped_tensor_scale_inv", True)
        else:
            setattr(tensor, "grouped_tensor_scale_inv", False)


def init_method_constant(value: float = 0.0):
    """Create an initialization method that initializes with a constant value."""

    def init_fn(tensor: torch.Tensor) -> None:
        with torch.no_grad():
            tensor.fill_(value)

    return init_fn


def init_method_normal(sigma: float) -> Callable:
    """Init method to set all tensor elements to a normal distribution."""

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def get_default_init_method() -> Callable:
    """Weight initialization method if not provided by user"""
    return init_method_normal(0.023)


def requires_grad(*tensors: Optional[torch.Tensor]) -> bool:
    """Check if any of the tensors requires gradient."""
    return any(tensor is not None and tensor.requires_grad for tensor in tensors)


def needs_quantized_gemm(tensor: torch.Tensor) -> bool:
    """Check if tensor needs quantized GEMM."""
    # Simplified implementation
    return tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]


def assert_dim_for_fp8_exec(input_tensor: torch.Tensor, weight_tensor: torch.Tensor) -> None:
    """Assert dimensions for FP8 execution."""
    # Check that input and weight have compatible dimensions for FP8 GEMM
    assert len(input_tensor.shape) >= 2, "Input tensor must have at least 2 dimensions"
    assert len(weight_tensor.shape) == 2, "Weight tensor must have 2 dimensions"
    assert input_tensor.shape[-1] == weight_tensor.shape[1], (
        f"Input feature dimension {input_tensor.shape[-1]} must match "
        f"weight dimension {weight_tensor.shape[1]}"
    )


def assert_dim_for_all_gather(
    tensor: torch.Tensor, with_all_gather: bool, quantizer: Optional[object]
) -> None:
    """Assert dimensions for all-gather operation."""
    if with_all_gather:
        assert len(tensor.shape) >= 2, "Tensor must have at least 2 dimensions for all-gather"
        if quantizer is not None:
            # Additional checks for quantized tensors
            pass


def profiler_range_push(label: str) -> None:
    """Push profiler range (no-op for now, can be extended for NPU profiling)."""
    pass


def profiler_range_pop(label: str) -> None:
    """Pop profiler range (no-op for now, can be extended for NPU profiling)."""
    pass


def get_profiler_range_context(label: str):
    """Get profiler range context manager."""

    class ProfilerRangeContext:
        def __enter__(self):
            profiler_range_push(label)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            profiler_range_pop(label)

    return ProfilerRangeContext()


def assert_viewless_tensor(tensor: torch.Tensor, extra_msg: Optional[str] = None) -> torch.Tensor:
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set).
    """
    if isinstance(tensor, list):
        return [assert_viewless_tensor(t) for t in tensor]
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        f"likely accumulate over iterations). {extra_msg}"
    )
    return tensor


def safely_set_viewless_tensor_data(tensor: torch.Tensor, new_data_tensor: torch.Tensor) -> None:
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    extra_msg = (
        "FYI, tensor._base has shape "
        f"{'--' if tensor._base is None else tensor._base.shape},"
        f"and new_data_tensor has shape {new_data_tensor.shape}."
    )
    assert_viewless_tensor(tensor, extra_msg=extra_msg)
    tensor.data = new_data_tensor


class SplitAlongDim(torch.autograd.Function):
    """
    Split tensor along given dimension
    """

    @staticmethod
    def forward(
        ctx,
        mixed_x_layer: torch.Tensor,
        split_dim: int,
        split_size_or_sections: Union[int, List[int], Tuple[int]],
        squeeze=False,
    ) -> Tuple[torch.Tensor, ...]:
        ctx.split_dim = split_dim
        ctx.split_size_or_sections = split_size_or_sections
        out_list = torch.split(mixed_x_layer, split_size_or_sections, dim=split_dim)
        if squeeze:
            out_list = [x.squeeze(split_dim) for x in out_list]
        return out_list

    @staticmethod
    def backward(ctx, *grad_outputs):
        assert len(grad_outputs) > 0, "No gradients received for backprop!"

        if isinstance(ctx.split_size_or_sections, (list, tuple)):
            split_sizes = ctx.split_size_or_sections
            assert len(grad_outputs) == len(split_sizes), (
                "Unequal number of gradients vs split sections for backprop!"
            )
        if isinstance(ctx.split_size_or_sections, int):
            split_sizes = [ctx.split_size_or_sections] * len(grad_outputs)
        dims = len(grad_outputs[0].shape)
        split_dim = (ctx.split_dim + dims) % dims

        return torch.cat(grad_outputs, dim=split_dim), None, None, None


def get_cudnn_version() -> Tuple[int, int, int]:
    """Get cuDNN version.

    For Ascend NPUs, this returns a placeholder version since cuDNN is not available.

    Returns:
        Tuple of (major, minor, patch) version numbers
    """
    return 0, 0, 0


def get_device_compute_capability(
    device: Optional[Union[int, torch.device]] = None,
) -> Tuple[int, int]:
    """Get compute capability of device.

    For Ascend NPUs, this returns a placeholder compute capability.

    Args:
        device: Device to query (optional)

    Returns:
        Tuple of (major, minor) compute capability
    """
    return 9, 0


def canonicalize_device(device: Optional[torch.device | str]) -> torch.device:
    """Canonicalize PyTorch device

    If `None`, then returns the default NPU device.

    """
    if device is None:
        # Use default NPU device
        device = torch.get_default_device()
        if device.type != "npu":
            device = torch.device("npu", torch.npu.current_device())
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "npu" and device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    return device


def canonicalize_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    """Canonicalize PyTorch datatype

    If `None`, then returns the default PyTorch datatype.

    """
    if dtype is None:
        # Use default dtype
        dtype = torch.get_default_dtype()
    return dtype


def devices_match(device1: torch.device, device2: torch.device) -> bool:
    """Whether two devices are the same"""
    device1 = torch.device(device1)
    device2 = torch.device(device2)
    if device1.type != device2.type:
        return False
    if device1.type == "npu":
        index1 = device1.index
        index2 = device2.index
        if index1 == index2:
            return True
        if index1 is None:
            index1 = torch.npu.current_device()
        if index2 is None:
            index2 = torch.npu.current_device()
        return index1 == index2
    return device1 == device2


def round_up_to_nearest_multiple(value, multiple):
    """Round up `value` to the next mutiple of `multiple`"""
    if multiple == 0:
        raise ValueError("multiple cannot be zero.")
    return ((value + multiple - 1) // multiple) * multiple


def is_bf16_available() -> bool:
    """Check if BF16 is available.

    Ascend NPUs support BF16.

    Returns:
        True if BF16 is available
    """
    return True


# ============================================================================
# MindSpeed Utilities Integration
# ============================================================================


def view_as_n_dim(input_tensor: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """View tensor as n-dimensional.

    Args:
        input_tensor: Input tensor
        dim: Target number of dimensions (must be >= 2)

    Returns:
        Tensor reshaped to n dimensions
    """
    if dim < 2:
        raise AssertionError("dim should be greater than or equal to 2")
    if len(input_tensor.shape) == 0:
        return input_tensor.view([1] * dim)
    if len(input_tensor.shape) != dim:
        return input_tensor.view(-1, *input_tensor.shape[-dim + 1 :])
    return input_tensor


@lru_cache
def get_quant_dtype(quant_mode):
    if quant_mode == "hif8":
        return QuantDtype(torch_npu.hifloat8, torch_npu.hifloat8)
    elif quant_mode == "hybrid":
        return QuantDtype(torch.float8_e4m3fn, torch.float8_e5m2)
    return QuantDtype(torch.float8_e4m3fn, torch.float8_e4m3fn)


def check_npu_version(min_version=None, max_version=None):
    version = get_npu_version()
    if version == NPUVersion.NONE:
        return False
    if min_version is not None:
        if version < min_version:
            return False
    if max_version is not None:
        if version > max_version:
            return False
    return True


@lru_cache(maxsize=None)
def get_npu_version():
    try:
        device_name = torch_npu.npu.get_device_name()
    except Exception:
        return NPUVersion.NONE
    if "Ascend910_95" in device_name or "Ascend950" in device_name:
        return NPUVersion.A5
    elif "Ascend910_93" in device_name:
        return NPUVersion.A3
    elif "Ascend910B" in device_name:
        return NPUVersion.A2
    else:
        return NPUVersion.MAX_VERSION
