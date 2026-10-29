# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature
from megatron_adaptor.patches import PatchesManager


class TransformerEngineBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__("transformer-engine-basic", optimization_level=0)

    def register_patches(self, patch_manager: PatchesManager, args):
        from megatron_adaptor.patches.transformer_engine.transformer_engine import (
            get_npu_grad_accum_fusion_available,
            column_parallel_linear_init_wrapper,
            transformer_config_post_init_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.transformer.transformer_config.TransformerConfig.__post_init__",
            transformer_config_post_init_wrapper,
        )

        patch_manager.register_patch(
            "megatron.core.tensor_parallel.layers._grad_accum_fusion_available",
            get_npu_grad_accum_fusion_available,
        )
        patch_manager.register_patch(
            "megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__",
            column_parallel_linear_init_wrapper,
        )
