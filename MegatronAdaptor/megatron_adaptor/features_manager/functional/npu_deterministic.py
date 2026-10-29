# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature


class NPUDeterministicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("npu-deterministic")

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--npu-deterministic',
            action='store_true',
            default=False,
            help='enable deterministic computing for npu.',
        )

    def register_patches(self, patch_manager, args):
        if args.npu_deterministic:
            from megatron_adaptor.patches.patch_utils import is_megatron_training_available

            if not is_megatron_training_available():
                return

            from megatron_adaptor.patches.megatron.npu_deterministic import npu_deterministic_wrapper

            patch_manager.register_patch('megatron.training.initialize._set_random_seed', npu_deterministic_wrapper)
