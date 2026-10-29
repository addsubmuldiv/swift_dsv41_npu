# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature


class ProfileFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('profile', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            "--profile-level",
            type=str,
            default='level0',
            choices=['level0', 'level1', 'level2'],
            help="Profile level default level0.",
        )
        group.add_argument("--profile-with-cpu", action='store_true', default=False, help="Profile with cpu info.")
        group.add_argument("--profile-with-stack", action='store_true', default=False, help="Profile with stack info.")
        group.add_argument(
            "--profile-with-memory", action='store_true', default=False, help="Profile with memory info."
        )
        group.add_argument(
            "--profile-record-shapes", action='store_true', default=False, help="Profile record shape info."
        )
        group.add_argument("--profile-save-path", type=str, default='./profile_dir', help="Profile save path.")

    def register_patches(self, patch_manager, args):
        from megatron_adaptor.patches.patch_utils import is_megatron_training_available

        if not is_megatron_training_available():
            return
        if not getattr(args, 'profile', False):
            return
        # Megatron-native args arrive here as raw strings (unknown-args fallback);
        # compare numerically to avoid '10' <= '8' lexicographic misjudgment.
        profile_step_start = getattr(args, 'profile_step_start', None)
        profile_step_end = getattr(args, 'profile_step_end', None)
        # Both unset means the Megatron defaults (10/12) apply; nothing to validate.
        if profile_step_start is not None and profile_step_end is not None:
            profile_step_start = int(profile_step_start)
            profile_step_end = int(profile_step_end)
            if profile_step_end <= profile_step_start:
                raise ValueError(
                    f'--profile-step-end ({profile_step_end}) must be greater than '
                    f'--profile-step-start ({profile_step_start}), otherwise the profiling window is empty.'
                )
        from megatron_adaptor.patches.megatron.profile import train_wrapper, train_step_wrapper

        patch_manager.register_patch('megatron.training.training.train', train_wrapper)
        patch_manager.register_patch('megatron.training.training.train_step', train_step_wrapper)
