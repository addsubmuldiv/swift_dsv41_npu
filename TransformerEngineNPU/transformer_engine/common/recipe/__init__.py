# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""FP8 recipes for TransformerEngine Ascend

This module provides FP8 quantization recipes for NPU-optimized operations,
including DelayedScaling, CurrentScaling, MXFP8, and BlockScaling recipes.
"""

from .base import Recipe, ScopedRecipe, Format, FormatEnum
from .block_recipe import Float8BlockScaling
from .current_recipe import Float8CurrentScaling
from .custom_recipe import CustomRecipe
from .delayed_recipe import DelayedScaling
from .hif8_delayed_recipe import HIF8DelayedScaling, QuantizerRole
from .mxfp4_recipe import MXFP4BlockScaling
from .mxfp8_32x32_recipe import MXFP832x32BlockScaling
from .mxfp8_recipe import MXFP8BlockScaling
from .w4a16_recipe import W4A16BlockScaling
from .w4a8_recipe import W4A8BlockScaling


__all__ = [
    "Format",
    "Recipe",
    "ScopedRecipe",
    "CustomRecipe",
    "FormatEnum",
    "DelayedScaling",
    "HIF8DelayedScaling",
    "QuantizerRole",
    "Float8CurrentScaling",
    "MXFP4BlockScaling",
    "MXFP832x32BlockScaling",
    "MXFP8BlockScaling",
    "W4A16BlockScaling",
    "W4A8BlockScaling",
    "Float8BlockScaling",
]


for _recipe_cls in (
    Recipe,
    ScopedRecipe,
    Format,
    FormatEnum,
    DelayedScaling,
    Float8CurrentScaling,
    CustomRecipe,
    MXFP4BlockScaling,
    MXFP832x32BlockScaling,
    MXFP8BlockScaling,
    W4A16BlockScaling,
    W4A8BlockScaling,
    Float8BlockScaling,
):
    _recipe_cls.__module__ = __name__

del _recipe_cls
