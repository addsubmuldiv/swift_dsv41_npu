# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, Optional

import torch


class FP8Format(NamedTuple):
    """FP8 format specification.

    Defines the properties of an FP8 format including range, exponent bits,
    mantissa bits, and PyTorch dtype.

    Attributes:
        max: Maximum representable value
        ebits: Number of exponent bits
        mbits: Number of mantissa bits
        dtype: PyTorch dtype for this format
    """

    max: float
    ebits: int
    mbits: int
    dtype: Optional[torch.dtype]

    @property
    def quant_dtype(self) -> torch.dtype:
        """Get quantization type.

        Returns:
            PyTorch dtype for quantization
        """
        if self.dtype is None:
            import torch_npu

            return torch_npu.hifloat8
        return self.dtype


class FormatEnum(Enum):
    """FP8 format enumeration.

    Defines standard FP8 formats with their specifications.
    """

    E4M3 = FP8Format(448, 4, 3, torch.float8_e4m3fn)
    E5M2 = FP8Format(57344, 5, 2, torch.float8_e5m2)
    HIF8 = FP8Format(57344, 5, 2, None)
    HIF8_STANDARD = FP8Format(32768, 5, 2, None)
    HIF8_224 = FP8Format(224, 5, 2, None)
    HIF8_15 = FP8Format(15, 5, 2, None)


class _FormatConfig(NamedTuple):
    """Format configuration for inputs, weights, and gradients.

    Attributes:
        fwd: Format for input/weight tensors
        bwd: Format for grad tensors
    """

    fwd: FormatEnum = FormatEnum.E4M3
    bwd: FormatEnum = FormatEnum.E4M3


class Format(Enum):
    """FP8 format configuration enumeration.

    Defines format configurations for different FP8 recipes.
    """

    E4M3 = _FormatConfig()
    E5M2 = _FormatConfig(fwd=FormatEnum.E5M2, bwd=FormatEnum.E5M2)
    HYBRID = _FormatConfig(bwd=FormatEnum.E5M2)
    HIF8 = _FormatConfig(fwd=FormatEnum.HIF8_15, bwd=FormatEnum.HIF8_224)
    HIF8_STANDARD = _FormatConfig(
        fwd=FormatEnum.HIF8_STANDARD,
        bwd=FormatEnum.HIF8_STANDARD,
    )

    @classmethod
    def from_str(cls, key: str) -> Optional["Format"]:
        """Get format from configuration key.

        Args:
            key: Configuration key string

        Returns:
            Format enum value or None
        """
        return getattr(cls, key.upper(), None)


_BACKWARD_OVERRIDES = (None, "high_precision", "dequantized")


def _default_backward_override() -> Optional[str]:
    """Return the backward override configured through the environment."""
    return os.getenv("NVTE_BACKWARD_OVERRIDE", None)


def _validate_backward_override(value: Optional[str]) -> None:
    assert value in _BACKWARD_OVERRIDES, (
        "NVTE_BACKWARD_OVERRIDE must be unset or one of: 'high_precision', 'dequantized'."
    )


class Recipe:
    """
    Base recipe class.
    """

    fp8_format: Format = Format.E4M3
    backward_override: Optional[str] = _default_backward_override()

    # Cached string representation. Lazily populated by ``_make_repr`` in
    # subclasses and invalidated by ``__setattr__`` whenever any attribute
    # changes. This makes repeated ``str(recipe)`` calls much cheaper.
    _cached_repr: Optional[str] = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_cached_repr" and self._cached_repr is not None:
            object.__setattr__(self, "_cached_repr", None)
        object.__setattr__(self, name, value)

    def _make_repr(self) -> str:
        """Build the string representation for this recipe.

        Subclasses must override this method. The result is cached by
        ``__repr__`` and reused until any attribute is mutated.
        """
        return ""

    def __repr__(self) -> str:
        if self._cached_repr is None:
            self._cached_repr = self._make_repr()
        return self._cached_repr

    def __post_init__(self) -> None:
        _validate_backward_override(self.backward_override)

    @classmethod
    def nvfp4(cls):
        """Whether the given recipe is NVFP4 1D block scaling."""
        return False

    @classmethod
    def mxfp4(cls):
        """Whether the given recipe is MXFP4 block scaling."""
        return False

    @classmethod
    def mxfp8(cls):
        """Whether the given recipe is MXFP8 block scaling."""
        return False

    @classmethod
    def mxfp8_32x32(cls):
        """Whether the given recipe uses 32x32 MXFP8 weight scaling."""
        return False

    @classmethod
    def w4a8(cls):
        """Whether the given recipe is W4A8 mixed block scaling."""
        return False

    @classmethod
    def w4a16(cls):
        """Whether the given recipe is W4A16 QAT block scaling."""
        return False

    @classmethod
    def delayed(cls):
        """Whether the given recipe is delayed scaling."""
        return False

    @classmethod
    def hif8_delayed(cls):
        """Whether the given recipe is HiF8 delayed scaling."""
        return False

    @classmethod
    def float8_current_scaling(cls):
        """Whether the given recipe is (per-tensor) current scaling."""
        return False

    @classmethod
    def float8_per_tensor_scaling(cls):
        """Whether the given recipe is per-tensor scaling."""
        return False

    @classmethod
    def float8_block_scaling(cls):
        """Whether the given recipe is float8 blockwise scaling."""
        return False

    @classmethod
    def custom(cls):
        """Whether the given recipe is custom."""
        return False


@dataclass(repr=False, kw_only=True)
class ScopedRecipe(Recipe):
    """Recipe with module-type-based quantization selection.

    ``all`` selects every module, ``moe-only`` selects GroupedLinear, and
    ``linear-only`` selects Linear roles (including fused linear modules).
    ``close`` disables all quantization for debugging.
    Scope is fixed when modules initialize their quantizers.
    """

    quantization_scope: str = "all"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantization_scope not in ("all", "moe-only", "linear-only", "close"):
            raise ValueError("quantization_scope must be one of: all, moe-only, linear-only, close")


@dataclass(frozen=True)
class MMParams:
    """Matrix multiplication options.

    Parameters
    ----------
    use_split_accumulator : bool, default = True
        Use FP8 fast accumulation on Hopper or Ada. For more details,
        see CUBLASLT_MATMUL_DESC_FAST_ACCUM option for cublasLtMatmul.
    """

    use_split_accumulator: bool = True

    def __repr__(self) -> str:
        return f"MMParams(use_split_accumulator={self.use_split_accumulator})"


@dataclass(frozen=True)
class QParams:
    """Quantization parameters.
    power_2_scale: use power of 2 scale parameter
    amax_epsilon: optional minimum value of abs max
    random_hadamard_transform: whether to use random hadamard transform
    stochastic_rounding: whether to use stochastic rounding
    """

    power_2_scale: bool = False
    amax_epsilon: float = 0.0
    random_hadamard_transform: bool = False
    stochastic_rounding: bool = False
    fp4_2d_quantization: bool = False

    def __repr__(self) -> str:
        return (
            f"Qparams(\npower_2_scale={self.power_2_scale},\n"
            f"amax_epsilon={self.amax_epsilon},\n"
            f"random_hadamard_transform={self.random_hadamard_transform},\n"
            f"stochastic_rounding={self.stochastic_rounding},\n"
            f"fp4_2d_quantization={self.fp4_2d_quantization}\n)"
        )
