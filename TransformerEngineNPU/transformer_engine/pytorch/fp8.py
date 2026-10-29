# ruff: noqa: F401, E402
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""
DEPRECATED in favor of `transformer_engine.pytorch.quantization.py`.
"""

# pylint: disable=wrong-import-position,unused-import

import warnings

warnings.warn(
    "Using deprecated internal API from Transformer Engine. "
    "transformer_engine.pytorch.fp8 will be removed in a "
    "future release.",
    DeprecationWarning,
    stacklevel=2,
)


# There are some users indirectly importing these classes
# from fp8.py. This ensure backwards compatibility.
# https://github.com/Lightning-AI/lightning-thunder/pull/2635.
# noinspection PyUnusedImports
from transformer_engine.common.recipe import (
    CustomRecipe,
    DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,
    Format,
    HIF8DelayedScaling,
    MXFP4BlockScaling,
    MXFP832x32BlockScaling,
    MXFP8BlockScaling,
    QuantizerRole,
    Recipe,
    W4A16BlockScaling,
    W4A8BlockScaling,
)

from .quantization import (
    check_recipe_support,
    get_fp8_torch_dtype,
    CustomRecipeState,
    DelayedScalingRecipeState,
    Float8BlockScalingRecipeState,
    # NVFP4BlockScalingRecipeState,
    Float8CurrentScalingRecipeState,
    MXFP4BlockScalingRecipeState,
    MXFP832x32BlockScalingRecipeState,
    MXFP8BlockScalingRecipeState,
    RecipeState,
    W4A16BlockScalingRecipeState,
    W4A8BlockScalingRecipeState,
)
from .quantization import (
    get_default_recipe as get_default_fp8_recipe,
)

from .quantization.utils import get_fp8_te_dtype


# Importing each function instead of 'import *' allows us specify '__all__' in
# quantize.py and also makes any newer additions to quantize.py invisible via
# fp8.py so that we don't reinforce importing internal TE functions.
# noinspection PyUnusedImports
from .quantization.manager import (
    FP8GlobalStateManager,
    _amax_and_scale_update,
    _compute_amax_and_update_history,
    _compute_scaling_factor,
    _default_get_amax_and_update_history,
    _default_sf_compute,
    _update_amax_history,
    check_fp8_block_scaling_support,
    check_fp8_support,
    check_mxfp4_support,
    check_mxfp8_support,
    check_nvfp4_support,
    check_w4a16_support,
    fp8_autocast,
    fp8_model_init,
    # get_fp4_te_dtype,
    get_fp8_max,
    split_and_copy,
)
