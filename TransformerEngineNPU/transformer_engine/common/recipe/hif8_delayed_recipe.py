# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Recipe definition for the independent HiF8 delayed-scaling policy."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Optional, Tuple, Union

import torch

from .base import Format, Recipe, _default_backward_override, _validate_backward_override


class QuantizerRole(str, Enum):
    """Semantic role of a per-tensor quantizer."""

    INPUT = "input"
    WEIGHT = "weight"
    GRAD = "grad"


MarginFactory = Callable[[QuantizerRole], float]


@dataclass
class HIF8DelayedScaling(Recipe):
    """HiF8 policy with current-scaling warmup and sparse delayed updates.

    The first ``amax_collect_interval`` regions use current scaling and
    contribute one amax sample per region. Afterwards, quantization uses the
    delayed scale and samples amax once every ``scale_update_interval``
    regions. A repeated use of a quantizer inside one outer autocast region is
    treated as one step and contributes the maximum of all observed amaxes.

    ``margin_factory`` returns the positive divisor applied to the format max
    for each role. If it is omitted, the role-specific margin fields are
    interpreted as guard bits, i.e. the divisor is ``2 ** margin``.
    """

    fp8_format: Union[Format, str] = Format.HIF8_STANDARD
    hif8_input_margin: int = 11
    hif8_weight_margin: int = 12
    hif8_grad_margin: int = 11
    margin_factory: Optional[MarginFactory] = None
    amax_history_len: int = 128
    amax_compute_algo: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "max"
    amax_collect_interval: int = 5
    scale_update_interval: int = 10
    override_linear_precision: Tuple[bool, bool, bool] = (False, False, False)
    reduce_amax: bool = True
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    def __post_init__(self) -> None:
        _validate_backward_override(self.backward_override)
        assert self.backward_override is None, (
            "HIF8 delayed scaling only supports backward_override=None."
        )
        if isinstance(self.fp8_format, str):
            self.fp8_format = Format.from_str(self.fp8_format)
        if self.fp8_format == Format.HIF8:
            self.fp8_format = Format.HIF8_STANDARD
        if self.fp8_format != Format.HIF8_STANDARD:
            raise ValueError("HIF8DelayedScaling requires Format.HIF8_STANDARD")
        if self.amax_history_len < 1:
            raise ValueError("amax_history_len must be at least 1")
        if self.amax_collect_interval < 1:
            raise ValueError("amax_collect_interval must be at least 1")
        if self.scale_update_interval < 1:
            raise ValueError("scale_update_interval must be at least 1")
        if not callable(self.amax_compute_algo) and self.amax_compute_algo not in (
            "max",
            "mean",
            "most_recent",
        ):
            raise ValueError("amax_compute_algo must be callable, 'max', 'mean', or 'most_recent'")
        for name in (
            "hif8_input_margin",
            "hif8_weight_margin",
            "hif8_grad_margin",
        ):
            margin = getattr(self, name)
            if not isinstance(margin, int) or not 0 <= margin <= 15:
                raise ValueError(f"{name} must be an integer in [0, 15]")
        for role in QuantizerRole:
            self.margin_factor(role)

    @classmethod
    def hif8_delayed(cls) -> bool:
        return True

    @classmethod
    def float8_per_tensor_scaling(cls) -> bool:
        return True

    @property
    def fp8_max_fwd(self) -> float:
        return float(self.fp8_format.value.fwd.value.max)

    @property
    def fp8_max_bwd(self) -> float:
        return float(self.fp8_format.value.bwd.value.max)

    def margin_factor(self, role: QuantizerRole) -> float:
        """Return the role-specific divisor used as ``fp8_max / divisor``."""
        if not isinstance(role, QuantizerRole):
            role = QuantizerRole(role)
        if self.margin_factory is not None:
            factor = float(self.margin_factory(role))
        else:
            margin = {
                QuantizerRole.INPUT: self.hif8_input_margin,
                QuantizerRole.WEIGHT: self.hif8_weight_margin,
                QuantizerRole.GRAD: self.hif8_grad_margin,
            }[role]
            factor = float(2**margin)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"margin_factory must return a finite positive value for {role.value}")
        return factor

    def __repr__(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin(input={self.hif8_input_margin}, "
            f"weight={self.hif8_weight_margin}, grad={self.hif8_grad_margin}), "
            f"margin_factory={self.margin_factory!r}, "
            f"format={self.fp8_format.name}, "
            f"amax_history_len={self.amax_history_len}, "
            f"amax_compute_algo={self.amax_compute_algo}, "
            f"amax_collect_interval={self.amax_collect_interval}, "
            f"scale_update_interval={self.scale_update_interval}, "
            f"reduce_amax={self.reduce_amax}, "
            f"backward_override={self.backward_override}"
        )
