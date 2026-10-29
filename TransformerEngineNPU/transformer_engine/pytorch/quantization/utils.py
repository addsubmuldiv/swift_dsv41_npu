# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import functools
from typing import Tuple, Union

import torch

try:
    import cann_ops_nn
except ImportError:
    cann_ops_nn = None

from transformer_engine.common.recipe import (
    CustomRecipe,
    DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,
    HIF8DelayedScaling,
    MXFP4BlockScaling,
    MXFP8BlockScaling,
    Recipe,
    W4A16BlockScaling,
    W4A8BlockScaling,
)
from ..constants import NPUVersion

from ..utils import check_npu_version


@functools.lru_cache(maxsize=None)
def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if check_npu_version(NPUVersion.A5):
        return True, ""
    return False, "FP8 requires Ascend 950 or newer (current device does not support FP8)"


@functools.lru_cache(maxsize=None)
def check_mxfp8_support() -> Tuple[bool, str]:
    """Return if mxfp8 support is available on the current device"""
    if check_npu_version(NPUVersion.A5):
        return True, ""
    return False, "MXFP8 requires Ascend 950 or newer (current device does not support FP8)"


@functools.lru_cache(maxsize=None)
def check_mxfp4_support() -> Tuple[bool, str]:
    """Return if mxfp4 support is available on the current device."""
    if check_npu_version(NPUVersion.A5):
        return True, ""
    return False, "MXFP8 requires Ascend 950 or newer (current device does not support FP8)"


@functools.lru_cache(maxsize=None)
def check_nvfp4_support() -> Tuple[bool, str]:
    """Return if nvfp4 support is available"""
    return False, "Device not support for NVFP4 execution."


@functools.lru_cache(maxsize=None)
def check_w4a8_support() -> Tuple[bool, str]:
    """Return if W4A8 support is available."""
    if cann_ops_nn is None:
        return (
            False,
            "cann_ops_nn is not installed; W4A8 requires the CANN ops extension",
        )
    if check_npu_version(NPUVersion.A5):
        return True, ""
    return (
        False,
        "W4A8 requires an MXFP4 dtype, which requires Ascend 950 or newer (current device does not support FP8)",
    )


@functools.lru_cache(maxsize=None)
def check_w4a16_support() -> Tuple[bool, str]:
    """Return if W4A16 QAT support is available."""
    if check_npu_version(NPUVersion.A5):
        return True, ""
    return (
        False,
        "W4A16 requires MXFP4 fake quantization, which requires Ascend 950 or newer",
    )


@functools.lru_cache(maxsize=None)
def check_fp8_block_scaling_support() -> Tuple[bool, str]:
    """Return if fp8 block scaling support is available"""
    if check_npu_version(NPUVersion.A5):
        return True, ""
    return (
        False,
        "FP8 block scaling requires Ascend 950 or newer (current device does not support FP8)",
    )


def check_recipe_support(recipe: Recipe) -> None:
    """Check if the given recipe is supported."""
    if isinstance(recipe, CustomRecipe):
        return
    recipe_supported = True
    unsupported_reason = ""
    if isinstance(recipe, (DelayedScaling, Float8CurrentScaling, HIF8DelayedScaling)):
        recipe_supported, unsupported_reason = check_fp8_support()
    if isinstance(recipe, Float8BlockScaling):
        recipe_supported, unsupported_reason = check_fp8_block_scaling_support()
    elif isinstance(recipe, MXFP4BlockScaling):
        recipe_supported, unsupported_reason = check_mxfp4_support()
    elif isinstance(recipe, W4A8BlockScaling):
        recipe_supported, unsupported_reason = check_w4a8_support()
    elif isinstance(recipe, W4A16BlockScaling):
        recipe_supported, unsupported_reason = check_w4a16_support()
    elif isinstance(recipe, MXFP8BlockScaling):
        recipe_supported, unsupported_reason = check_mxfp8_support()
    if not recipe_supported:
        raise RuntimeError(unsupported_reason)


def get_default_fp8_recipe():
    return MXFP8BlockScaling()


get_default_recipe = get_default_fp8_recipe


def is_fp8_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if FP8 support is available for the delayed
    scaling and per tensor current scaling recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for FP8 is available.

    """
    if return_reason:
        return check_fp8_support()
    return check_fp8_support()[0]


def is_hif8_delayed_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if HiF8 delayed scaling support is available.

    HiF8 delayed scaling shares the same hardware requirement as delayed
    scaling, so its availability is identical to ``is_fp8_available``.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for HiF8 delayed scaling is available.

    """
    if return_reason:
        return check_fp8_support()
    return check_fp8_support()[0]


def is_mxfp8_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the MXFP8 recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for MXFP8 is available.

    """
    if return_reason:
        return check_mxfp8_support()
    return check_mxfp8_support()[0]


def is_mxfp4_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the MXFP4 recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available.

    """
    if return_reason:
        return check_mxfp4_support()
    return check_mxfp4_support()[0]


def is_w4a8_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the W4A8 recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available.

    """
    if return_reason:
        return check_w4a8_support()
    return check_w4a8_support()[0]


def is_w4a16_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """Determine if support is available for the W4A16 QAT recipe."""
    if return_reason:
        return check_w4a16_support()
    return check_w4a16_support()[0]


def is_fp8_block_scaling_available(
    return_reason: bool = False,
) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the FP8 block scaling recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for FP8 block scaling is available.

    """
    if return_reason:
        return check_fp8_block_scaling_support()
    return check_fp8_block_scaling_support()[0]


def is_nvfp4_available(return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Determine if support is available for the NVFP4 recipe.

    Parameters
    ----------
    return_reason : bool, optional
        If ``False`` (default), return only a boolean indicating availability.
        If ``True``, return a tuple ``(is_available, reason)`` where ``reason`` provides
        a human-readable explanation when required support is not available. The reason
        will be an empty string if support for NVFP4 is available.

    """
    if return_reason:
        return check_nvfp4_support()
    return check_nvfp4_support()[0]


def get_fp8_torch_dtype(recipe: Recipe, fprop_tensor: bool = True) -> torch.dtype:
    """Get fp8 data type according to recipe and tensor"""
    if fprop_tensor:
        return recipe.fp8_format.value.fwd.value.quant_dtype
    return recipe.fp8_format.value.bwd.value.quant_dtype


get_fp8_te_dtype = get_fp8_torch_dtype


def get_fp8_max(recipe: Recipe, fprop_tensor: bool = True) -> torch.dtype:
    """Get fp8 data type according to recipe and tensor"""
    if fprop_tensor:
        return recipe.fp8_format.value.fwd.value.max
    return recipe.fp8_format.value.bwd.value.max
