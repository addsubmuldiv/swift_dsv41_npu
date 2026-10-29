# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Optional

from transformer_engine.common.recipe import Format, Recipe
from transformer_engine.common.recipe.base import _default_backward_override


@dataclass(repr=False)
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E8M0 (8 bits of exponent,
    0 bits of mantissa), equivalent to scaling by a power of 2.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), in this recipe the quantized tensor and its transpose
    are not numerically equivalent. Due to this, when Transformer Engine
    needs both the MXFP8 tensor and its transpose (e.g. to calculate both
    forward and backward pass), during the quantization both versions are
    computed from the high precision input to avoid double quantization
    errors.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward
                pass.
    """

    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    @classmethod
    def mxfp8(cls):
        return True

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )
