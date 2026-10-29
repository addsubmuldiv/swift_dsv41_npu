# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import torch

from transformer_engine.common.recipe import Format

from .base import Recipe, _default_backward_override, _validate_backward_override


@dataclass(repr=False)
class DelayedScaling(Recipe):
    """Delayed scaling FP8 recipe.

    This recipe uses amax history to compute scaling factors with a delay.
    The scaling factor is computed based on the maximum absolute value (amax)
    observed over a history of iterations.

    Parameters
    ----------
    margin : int, default = 0
        Margin for scaling factor computation
    fp8_format : Format, default = Format.HYBRID
        FP8 format to use: "E4M3", "E5M2", "HYBRID", or "HIF8"
    amax_history_len : int, default = 1024
        Length of amax history buffer
    amax_compute_algo : str, default = "max"
        Algorithm for computing amax: "max" or "most_recent"
    scaling_factor_compute_algo : callable, optional
        Optional callback to override default scale computation
    override_linear_precision : Tuple[bool, bool, bool], default = (False, False, False)
        Override precision for forward, input grad, and weight grad
    reduce_amax : bool, default = True
        Whether to reduce amax across the configured group at autocast exit
    fp8_dpa : bool, default = False
        Enable FP8 for dot product attention
    fp8_mha : bool, default = False
        Enable FP8 for multi-head attention
    """

    margin: int = 0
    fp8_format: Union[Format, str] = Format.HYBRID
    amax_history_len: int = 1024
    amax_compute_algo: Union[str, Callable[[torch.Tensor, torch.Tensor, int], None]] = "max"
    scaling_factor_compute_algo: Optional[Callable] = None
    override_linear_precision: Tuple[bool, bool, bool] = (False, False, False)
    reduce_amax: bool = True
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    _amax_history: dict = field(default_factory=dict)
    _scale: dict = field(default_factory=dict)
    _scale_inv: dict = field(default_factory=dict)

    # ---------- Compatibility parameters ----------
    params_dtype: torch.dtype = torch.bfloat16
    interval: int = 1
    device: torch.device = torch.device("npu")

    @classmethod
    def delayed(cls):
        return True

    def __post_init__(self) -> None:
        _validate_backward_override(self.backward_override)
        assert self.backward_override is None, (
            "Delayed scaling only supports backward_override=None."
        )

    @classmethod
    def float8_per_tensor_scaling(cls) -> bool:
        return True

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"amax_history_len={self.amax_history_len}, "
            f"amax_compute_algo={self.amax_compute_algo}, "
            f"scaling_factor_compute_algo={self.scaling_factor_compute_algo}, "
            f"reduce_amax={self.reduce_amax}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )
