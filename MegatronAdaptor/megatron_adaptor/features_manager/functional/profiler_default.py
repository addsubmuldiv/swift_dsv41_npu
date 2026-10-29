# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature


class ProfilerDefaultFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('profiler-default', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from megatron_adaptor.patches.megatron.parallel_state import (
            get_nccl_options_add_group_info_wrapper,
        )

        patch_manager.register_patch(
            'megatron.core.parallel_state.get_nccl_options', get_nccl_options_add_group_info_wrapper
        )
