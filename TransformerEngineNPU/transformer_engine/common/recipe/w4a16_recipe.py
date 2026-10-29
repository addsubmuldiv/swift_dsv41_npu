# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Optional

from transformer_engine.common.recipe.base import (
    QParams,
    ScopedRecipe,
    _default_backward_override,
)


@dataclass(repr=False)
class W4A16BlockScaling(ScopedRecipe):
    """Use MXFP4 E2M1 fake-quantized weights and high-precision activations for QAT.

    ``quantization_scope`` selects all supported weights (``"all"``, default), only
    GroupedLinear weights (``"moe-only"``), or weights with a ``"linear"`` role
    (``"linear-only"``, including Linear, LayerNormLinear, and LayerNormMLP).
    ``"close"`` disables all quantization for debugging.
    Selection follows module type, not explicit expert identity.
    """

    fp4_quant_weight: QParams = QParams(power_2_scale=True)
    block_size: int = 32
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    @classmethod
    def w4a16(cls):
        return True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.block_size != 32:
            raise ValueError("W4A16BlockScaling currently supports block_size=32 only")

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"block_size={self.block_size}, "
            f"quantization_scope={self.quantization_scope}, "
            f"fp4_quant_weight={self.fp4_quant_weight}, "
            f"backward_override={self.backward_override}"
        )
