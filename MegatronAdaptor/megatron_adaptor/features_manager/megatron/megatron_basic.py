# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    print("FLA is not available")
    HAVE_FLA = False


class MegatronBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('megatron-basic', optimization_level=0)

    def pre_validate_args(self, args):
        """Turn --async-save off when the nvrx async backend is unusable.

        Upstream only fails at the first checkpoint, so fall back to synchronous saving
        here, before validate_args prints the argument snapshot it logs.
        """
        # Also called with a TransformerConfig, hence the attribute probe.
        if not getattr(args, 'async_save', False):
            return
        try:
            from megatron.core.dist_checkpointing.strategies.nvrx import has_nvrx_async_support
        except ImportError:
            # No nvrx branch on this version: async uses the in-tree implementation.
            return
        if has_nvrx_async_support() or getattr(args, 'async_strategy', 'nvrx') == 'mcore':
            return

        from megatron.training.utils import warn_rank_0

        # Not gated on args.use_dist_ckpt: validate_args derives it later.
        warn_rank_0(
            '--async-save requires nvidia-resiliency-ext, which is not available in this '
            'environment. Disabling --async-save; checkpoints will be saved synchronously.'
        )
        args.async_save = False

    def pre_register_patches(self, patch_manager, args):
        # Backfill nvrx.__version__ before megatron's import-time HAVE_NVRX probe reads it.
        from megatron_adaptor.patches.megatron.nvrx_compat import patch_nvrx_version_attr

        patch_nvrx_version_attr()

        # Eagerly backfill torch.cuda.get_device_capability before megatron.training
        # is imported.  MLM main #6538 (flash-linear-attention GDP kernels) probes
        # device capability at module import time in
        # megatron/core/ssm/ops/gdp/common.py.  On NPU, transfer_to_npu rewrites
        # torch.cuda.is_available() -> True but get_device_capability() -> None,
        # so `None[0]` raises TypeError during the megatron.training import chain.
        # The regular registration (register_non_mcore_basic_patches) only applies
        # after megatron.training has been imported, which is too late.
        # NOTE: the backfill helper must live in a megatron-free module — importing
        # patches.megatron.megatron_basic here would pull in megatron.core.jit
        # before the torch.compile -> dummy_compile stub is applied, capturing the
        # real torch.compile into jit_fuser and breaking JIT warmup.
        from megatron_adaptor.patches.megatron.device_probe_compat import patch_device_capability

        patch_device_capability()

    def register_patches(self, patch_manager, args):
        from megatron_adaptor.patches.patch_utils import is_megatron_training_available

        self.register_mcore_basic_patches(patch_manager, args)
        if not HAVE_FLA:
            self.register_fla_patches(patch_manager, args)
        if is_megatron_training_available():
            self.register_non_mcore_basic_patches(patch_manager, args)

    def register_mcore_basic_patches(self, pm, args):
        from megatron_adaptor.patches.megatron.transformer_config import (
            transformer_config_init_subclass,
            transformer_config_init_wrapper,
            transformer_config_post_init_wrapper,
        )

        # Argument injection + feature validation for mcore configs.
        # MLATransformerConfig is a dataclass subclass with its own generated
        # __init__, so it needs the argument-injection wrapper explicitly.
        pm.register_patch(
            'megatron.core.transformer.transformer_config.TransformerConfig.__init__',
            transformer_config_init_wrapper,
        )
        pm.register_patch(
            'megatron.core.transformer.transformer_config.TransformerConfig.__init_subclass__',
            classmethod(transformer_config_init_subclass),
        )
        pm.register_patch(
            'megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
            transformer_config_post_init_wrapper,
        )
        pm.register_patch(
            'megatron.core.transformer.transformer_config.MLATransformerConfig.__init__',
            transformer_config_init_wrapper,
        )

        from megatron_adaptor.patches.megatron.megatron_basic import (
            _set_cuda_rng_state,
            ScaledUpperTriangMaskedSoftmax,
            ScaledMaskedSoftmax,
            ScaledSoftmax,
            is_kernel_available,
            forward_fused_softmax,
            preload_tensors,
        )

        pm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)
        pm.register_patch(
            'megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax', ScaledUpperTriangMaskedSoftmax
        )
        pm.register_patch('megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax', ScaledMaskedSoftmax)
        pm.register_patch('megatron.core.fusions.fused_softmax.ScaledSoftmax', ScaledSoftmax)
        pm.register_patch(
            'megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available', is_kernel_available
        )
        pm.register_patch(
            'megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax', forward_fused_softmax
        )

        # Synchronous D2H copy and CPU-tensor snapshot during async checkpoint save,
        # so the staged data is not corrupted by subsequent training steps.
        pm.register_patch(
            'megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors',
            preload_tensors,
        )

    def register_fla_patches(self, pm, args):
        from megatron_adaptor.patches.megatron.gdn_packed_seq import (
            _npu_causal_conv1d,
            _npu_chunk_gated_delta_rule,
            _npu_l2norm,
        )

        pm.register_patch(
            'fla.modules.convolution.causal_conv1d',
            _npu_causal_conv1d,
            create_dummy=True,
        )
        pm.register_patch(
            'fla.ops.gated_delta_rule.chunk_gated_delta_rule',
            _npu_chunk_gated_delta_rule,
            create_dummy=True,
        )
        pm.register_patch(
            'fla.modules.l2norm.l2norm',
            _npu_l2norm,
            create_dummy=True,
        )

    def register_non_mcore_basic_patches(self, pm, args):
        from megatron_adaptor.patches.megatron.megatron_basic import (
            _compile_dependencies,
            get_device_wrapper,
            get_device_arch_version,
            _range_push_patch,
            _range_pop_patch,
        )

        # Capability stub lives in the megatron-free compat module (single
        # definition shared with the eager pre_register backfill).
        from megatron_adaptor.patches.megatron.device_probe_compat import get_device_capability
        from megatron_adaptor.patches.megatron.arguments_basic import (
            parse_args_wrapper,
            validate_args_wrapper,
        )

        # args parser patch
        pm.register_patch('megatron.training.arguments.parse_args', parse_args_wrapper)
        pm.register_patch('megatron.training.arguments.validate_args', validate_args_wrapper)
        pm.register_patch('megatron.training.yaml_arguments.validate_yaml', validate_args_wrapper)

        pm.register_patch('megatron.training.initialize._compile_dependencies', _compile_dependencies)
        pm.register_patch('megatron.training.dist_signal_handler.get_device', get_device_wrapper)
        pm.register_patch('megatron.training.utils.get_device_arch_version', get_device_arch_version)
        pm.register_patch('torch.cuda.get_device_capability', get_device_capability)
        pm.register_patch('torch.cuda.nvtx.range_push', _range_push_patch)
        pm.register_patch('torch.cuda.nvtx.range_pop', _range_pop_patch)
