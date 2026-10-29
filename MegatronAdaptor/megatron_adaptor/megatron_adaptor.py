# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from logging import getLogger

from torch_npu.contrib import transfer_to_npu  # noqa: F401

from megatron_adaptor.auto_execute import AutoExecuteFunction
from megatron_adaptor.features_manager.features_manager import FeaturesManager

LOG = getLogger(__name__)
_IS_FEATURES_PATCHED = False


@AutoExecuteFunction
def patch_features():
    global _IS_FEATURES_PATCHED
    if _IS_FEATURES_PATCHED:
        return
    _IS_FEATURES_PATCHED = True

    LOG.info("start to patch features in megatron adaptor.")
    from megatron_adaptor.utils.args_utils import get_mindspeed_args

    ms_args = get_mindspeed_args()

    FeaturesManager.apply_features_pre_patches(ms_args)
    FeaturesManager.apply_features_patches(ms_args)
