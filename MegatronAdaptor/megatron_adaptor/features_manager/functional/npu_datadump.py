# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature


class NPUDataDumpFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('npu-datadump')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--npu-datadump', action='store_true', default=False, help='enable npu data dump with mstt.')

    def register_patches(self, patch_manager, args):
        from megatron_adaptor.patches.patch_utils import is_megatron_training_available

        if not is_megatron_training_available():
            return

        from megatron_adaptor.patches.megatron.npu_datadump.npu_datadump import (
            dump_end_wrapper,
            dump_start_wrapper,
            init_debugger,
        )

        # Construct PrecisionDebugger at the register stage: pre-stage patches
        # (e.g. torch.compile -> dummy_compile) are already applied and the
        # megatron model is not built yet. See init_debugger for the timing
        # constraints.
        init_debugger()

        patch_manager.register_patch('megatron.training.training.train_step', dump_start_wrapper)
        patch_manager.register_patch('megatron.training.ft_integration.on_training_step_end', dump_end_wrapper)
