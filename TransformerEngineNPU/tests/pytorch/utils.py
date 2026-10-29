# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Shared pytest helpers for Ascend NPU tests."""

from __future__ import annotations

import socket
from typing import Optional
import torch


def dtype_tols(dtype: torch.dtype) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    Args:
        dtype: PyTorch datatype

    Returns:
        Dictionary containing rtol and atol
    """
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    raise ValueError(f"Unsupported dtype ({dtype})")


def quantization_tols(name: str) -> dict[str, float]:
    """Estimated numerical error for a quantization scheme

    NOTE: Not used in NPU tests since FP8 quantization is not supported.
    Retained for API compatibility when NPU FP8 support is added.

    Args:
        name: Quantization scheme name

    Returns:
        Dictionary containing rtol and atol
    """
    if name in (
        "fp8",
        "fp8_delayed_scaling",
        "fp8_current_scaling",
        "mxfp8",
        "mxfp8_block_scaling",
    ):
        # FP8 E4M3 has epsilon = 0.0625
        return dict(rtol=0.125, atol=0.0675)
    if name in ("nvfp4", "mxfp4", "mxfp4_block_scaling"):
        # FP4 E2M1 has epsilon = 0.25
        return dict(rtol=0.25, atol=0.125)
    raise ValueError(f"Unsupported quantization scheme ({name})")


@torch.no_grad
def assert_close(
    actual: Optional[torch.Tensor],
    expected: Optional[torch.Tensor],
    *,
    check_device: bool = False,
    check_dtype: bool = False,
    check_layout: bool = False,
    **kwargs,
) -> None:
    """Assert that two tensors are close

    This is a wrapper around torch.testing.assert_close. It changes the defaults
    for device and dtype checks (useful when the reference implementation is computed
    in high precision on CPU).

    Args:
        actual: Actual tensor
        expected: Expected tensor
        check_device: Whether to check device
        check_dtype: Whether to check dtype
        check_layout: Whether to check layout
        **kwargs: Additional arguments passed to torch.testing.assert_close
    """
    torch.testing.assert_close(
        actual,
        expected,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        **kwargs,
    )


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | tuple[int, ...],
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "npu",
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch operations
    in high precision on CPU. The test tensor is intended for use in
    TransformerEngine operations on NPU.

    Args:
        shape: Tensor shape
        min_val: Minimum random value
        max_val: Maximum random value
        ref_dtype: Reference tensor dtype
        ref_device: Reference tensor device
        test_dtype: Test tensor dtype
        test_device: Test tensor device
        requires_grad: Whether gradients are required

    Returns:
        Tuple of (reference tensor, test tensor)
    """
    # Random reference tensor
    ref = torch.empty(shape, dtype=ref_dtype, device=ref_device)
    ref.uniform_(min_val, max_val)

    # Construct test tensor from reference tensor
    test = ref.to(device=test_device, dtype=test_dtype)

    # Ensure reference and test tensors match
    # This is critical: after converting ref (float64) to test (fp16/bf16),
    # some values may lose precision. We sync ref back from test so that
    # both tensors hold exactly the same representable values.
    # This ensures comparison errors come only from TE vs PyTorch differences,
    # not from dtype conversion itself.
    ref.copy_(test.to(dtype=ref.dtype))

    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)

    return ref, test


def to_cpu(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert to FP64 CPU tensor

    Args:
        tensor: Input tensor

    Returns:
        FP64 CPU tensor or None
    """
    if tensor is None:
        return None
    return tensor.detach().to(dtype=torch.float64, device="cpu")


def npu_available() -> bool:
    """Return whether NPU device is available."""
    return hasattr(torch, "npu") and torch.npu.is_available()


def a5_npu_available() -> bool:
    """Return whether an Atlas A5 NPU device is available."""
    if not npu_available():
        return False

    from transformer_engine.pytorch.constants import NPUVersion
    from transformer_engine.pytorch.utils import check_npu_version

    return check_npu_version(NPUVersion.A5)


def npu_device_count() -> int:
    """Return the number of available NPU devices."""
    if not npu_available():
        return 0
    return torch.npu.device_count()


def get_free_port() -> str:
    """Return a free TCP port allocated by the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])
