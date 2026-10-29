# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Optional

from transformer_engine.common.recipe.base import (
    Format,
    QParams,
    ScopedRecipe,
    _default_backward_override,
)


@dataclass(repr=False)
class W4A8BlockScaling(ScopedRecipe):
    """
    Use MXFP4 weights and MXFP8 activations/gradients.

    Forward GEMMs use a hybrid weight storage where the RHS weight is kept
    as MXFP4 for fprop and as MXFP8 for dgrad. Backward activation GEMMs
    continue to use MXFP8 on both sides.
    """

    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp4_quant_weight: QParams = QParams(power_2_scale=True)
    block_size: int = 32
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    @classmethod
    def w4a8(cls):
        return True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.block_size != 32:
            raise ValueError("W4A8BlockScaling currently supports block_size=32 only")

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"block_size={self.block_size}, "
            f"quantization_scope={self.quantization_scope}, "
            f"backward_override={self.backward_override}"
        )
