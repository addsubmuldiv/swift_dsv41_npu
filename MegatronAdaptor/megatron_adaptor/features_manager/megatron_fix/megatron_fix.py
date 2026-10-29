# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature


class MegatronFixFeature(MindSpeedFeature):
    """Fixes for upstream Megatron-LM bugs that cannot be patched directly."""

    def __init__(self):
        super().__init__('megatron-fix', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--fix-use-cpu-initialization',
            action='store_true',
            default=False,
            help='Fix De Morgan law violation in prepare_existing_model_chunks_for_distributed_training. '
            'When enabled, moves CPU params to NPU when --use-cpu-initialization is used. '
            'This is a workaround for a Megatron-LM bug. Remove this when upstream fixes it.',
        )
        group.add_argument(
            '--fix-gtp-sharp-init-ordering',
            action='store_true',
            default=False,
            help='Fix GTP_REMAT group initialization ordering in get_data_parallel_group. '
            'The GTP commit (c5ff22b7f) changed the default with_gtp_remat=True, but the SHARP '
            'barrier in initialize_model_parallel calls get_data_parallel_group BEFORE the '
            'GTP_REMAT group aliasing (when gtp_remat_size=1). This causes an assertion failure '
            'when --use-sharp is enabled. Remove this when upstream fixes the ordering. '
            '(Added: 2026-07-31, breaking commit: c5ff22b7f)',
        )

    def register_patches(self, patch_manager, args):
        if args.fix_use_cpu_initialization:
            from megatron_adaptor.patches.megatron.dist_training import (
                prepare_existing_model_chunks_for_distributed_training_wrapper,
            )

            patch_manager.register_patch(
                'megatron.training.models.dist_utils.prepare_existing_model_chunks_for_distributed_training',
                prepare_existing_model_chunks_for_distributed_training_wrapper,
            )

        if args.fix_gtp_sharp_init_ordering:
            from megatron_adaptor.patches.megatron.gtp_remat_fallback import (
                get_data_parallel_group_gtp_fallback_wrapper,
            )

            patch_manager.register_patch(
                'megatron.core.parallel_state.get_data_parallel_group',
                get_data_parallel_group_gtp_fallback_wrapper,
            )
