# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Online numerical comparison for quantized GEMM outputs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch

from ..constants import USAGE_WITH_TRANS
from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage
from ..utils import view_as_n_dim


logger = logging.getLogger(__name__)


_ABSOLUTE_ERROR_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GemmComparisonConfig:
    """Process-wide online comparison settings for quantized GEMMs."""

    compare_with_cpu: bool = False
    compare_with_bf16: bool = False
    relative_error_tolerance: float = 1e-3

    @property
    def enabled(self) -> bool:
        """Return whether at least one reference path is enabled."""
        return self.compare_with_cpu or self.compare_with_bf16


_gemm_comparison_config = GemmComparisonConfig()


def configure_gemm_comparison(
    *,
    compare_with_cpu: bool = False,
    compare_with_bf16: bool = False,
    relative_error_tolerance: float = 1e-3,
) -> None:
    """Configure process-wide online comparison for quantized GEMMs.

    The configuration is intended to be set once during framework startup.
    Comparison is disabled by default and has no reference-compute overhead
    until one of the reference paths is enabled. A comparison fails only when
    the same element exceeds the fixed absolute tolerance (1e-6) and the
    configured relative tolerance.
    """
    if not math.isfinite(relative_error_tolerance) or relative_error_tolerance < 0:
        raise ValueError(
            "relative_error_tolerance must be a non-negative finite value, "
            f"but got {relative_error_tolerance!r}"
        )

    global _gemm_comparison_config
    _gemm_comparison_config = GemmComparisonConfig(
        compare_with_cpu=bool(compare_with_cpu),
        compare_with_bf16=bool(compare_with_bf16),
        relative_error_tolerance=float(relative_error_tolerance),
    )


def get_gemm_comparison_config() -> GemmComparisonConfig:
    """Return the active process-wide GEMM comparison configuration."""
    return _gemm_comparison_config


def _is_quantized_tensor(tensor) -> bool:
    """Return whether a tensor is a TE/TENPU quantized tensor."""
    return isinstance(tensor, (QuantizedTensor, QuantizedTensorStorage))


def _to_reference_tensor(tensor, usage, dtype: torch.dtype) -> torch.Tensor:
    """Convert the exact GEMM operand selected by ``usage`` to a reference tensor."""
    if _is_quantized_tensor(tensor):
        dequantize_for_usage = getattr(tensor, "dequantize_for_usage", None)
        if dequantize_for_usage is not None:
            tensor = dequantize_for_usage(usage, dtype=dtype)
            return view_as_n_dim(tensor).detach()
        tensor = tensor.dequantize(dtype=dtype)
    else:
        tensor = tensor.to(dtype=dtype)
    # Quantizers flatten all leading dimensions before GEMM. Dequantize may
    # restore the logical input shape, so reproduce that flattening before
    # applying the GEMM usage transpose.
    tensor = view_as_n_dim(tensor)
    if usage in USAGE_WITH_TRANS:
        tensor = tensor.t()
    return tensor.detach()


def _build_reference_output(
    a,
    b,
    usage_a,
    usage_b,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reproduce general_gemm with dequantized operands."""
    reference_a = _to_reference_tensor(a, usage_a, dtype).to(device=device)
    reference_b = _to_reference_tensor(b, usage_b, dtype).to(device=device)

    reference = torch.matmul(reference_a, reference_b)
    return reference


def _flat_index_to_coordinates(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    """Convert a flattened index to coordinates without device-side operations."""
    if not shape:
        return []
    coordinates = [0] * len(shape)
    remainder = flat_index
    for index in range(len(shape) - 1, -1, -1):
        dimension = shape[index]
        coordinates[index] = remainder % dimension
        remainder //= dimension
    return coordinates


def _comparison_failure_point(
    absolute_error: torch.Tensor,
    relative_error: torch.Tensor,
    failure_mask: torch.Tensor,
    shape: tuple[int, ...],
) -> tuple[list[int], float, float]:
    """Select the finite joint-failure point reported by the exception."""
    finite_mask = failure_mask & torch.isfinite(relative_error)
    if bool(torch.any(finite_mask).item()):
        score = torch.where(
            finite_mask,
            relative_error,
            torch.full_like(relative_error, float("-inf")),
        )
    else:
        score = torch.where(
            failure_mask,
            absolute_error,
            torch.full_like(absolute_error, float("-inf")),
        )
    flat_index = int(torch.argmax(score.reshape(-1)).item())
    return (
        _flat_index_to_coordinates(flat_index, shape),
        float(absolute_error.reshape(-1)[flat_index].item()),
        float(relative_error.reshape(-1)[flat_index].item()),
    )


def _check_reference(
    output: torch.Tensor,
    reference: torch.Tensor,
    reference_name: str,
    relative_error_tolerance: float,
) -> None:
    """Fail when one element exceeds both absolute and relative tolerances."""
    output_fp32 = output.detach().to(device=reference.device, dtype=torch.float32)
    reference_fp32 = reference.detach().to(dtype=torch.float32)
    if output_fp32.shape != reference_fp32.shape:
        raise ValueError(
            f"The quantized GEMM output shape {tuple(output_fp32.shape)} does not match "
            f"the {reference_name} reference shape {tuple(reference_fp32.shape)}"
        )
    absolute_error = torch.abs(reference_fp32 - output_fp32)
    # Relative error alone is unstable when the reference is zero or near zero.
    # Apply the absolute and relative thresholds to the same element so a tiny
    # absolute difference cannot fail comparison only because of its denominator.
    relative_error = absolute_error / torch.abs(reference_fp32)
    max_absolute_error = torch.max(absolute_error)
    # Risk: the full-tensor relative maximum is unstable at zero or near-zero
    # references and may be NaN for 0 / 0. Keep it aligned with MindSpeed
    # master for reporting, but use the joint per-element mask for failure.
    max_relative_error = torch.max(relative_error)
    failure_mask = (absolute_error > _ABSOLUTE_ERROR_TOLERANCE) & (
        relative_error > relative_error_tolerance
    )
    comparison_failed = bool(torch.any(failure_mask).item())

    logger.info("The error of quantized GEMM against %s:", reference_name)
    logger.info("[%s] Max Absolute Error: %s", output.device, max_absolute_error.item())
    logger.info("[%s] Max Relative Error: %s", output.device, max_relative_error.item())
    if comparison_failed:
        failure_index, failure_absolute, failure_relative = _comparison_failure_point(
            absolute_error,
            relative_error,
            failure_mask,
            tuple(reference_fp32.shape),
        )
        raise ValueError(
            f"The error of quantized GEMM against {reference_name} exceeds "
            f"absolute tolerance {_ABSOLUTE_ERROR_TOLERANCE} and relative "
            f"tolerance {relative_error_tolerance} at index "
            f"{failure_index}: absolute error {failure_absolute}, relative error "
            f"{failure_relative}"
        )


def compare_gemm_output(
    output: torch.Tensor,
    a,
    b,
    usage_a,
    usage_b,
) -> None:
    """Compare a quantized general_gemm result with configured references."""
    config = get_gemm_comparison_config()
    if (
        not config.enabled
        or output.numel() == 0
        or not (_is_quantized_tensor(a) or _is_quantized_tensor(b))
    ):
        return

    with torch.no_grad():
        if config.compare_with_cpu:
            cpu_reference = _build_reference_output(
                a,
                b,
                usage_a,
                usage_b,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            _check_reference(
                output,
                cpu_reference,
                "CPU FP32",
                config.relative_error_tolerance,
            )

        if config.compare_with_bf16:
            bf16_reference = _build_reference_output(
                a,
                b,
                usage_a,
                usage_b,
                dtype=torch.bfloat16,
                device=output.device,
            )
            _check_reference(
                output,
                bf16_reference,
                "BF16",
                config.relative_error_tolerance,
            )
