# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import sys

from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature


class RequirementsBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('requirements-basic', optimization_level=0)

    def pre_register_patches(self, patch_manager, args):
        self.torch_adaptation(patch_manager, args)
        self.fused_weight_gradient_adaptation(patch_manager, args)

    def torch_adaptation(self, pm, args):
        from megatron_adaptor.patches.requirements.requirements import (
            type_wrapper,
            ensure_contiguous_wrapper,
            lcm,
            dummy_function,
            torch_all_reduce_double_dtype_bypass_wrapper,
            dummy_compile,
            version_wrapper,
        )

        pm.register_patch('torch.compile', dummy_compile)
        pm.register_patch('torch.jit.script', dummy_compile)
        pm.register_patch('torch.nn.parameter.Parameter.type', type_wrapper)
        pm.register_patch('torch.Tensor.type', type_wrapper)
        pm.register_patch('torch.Tensor.view', ensure_contiguous_wrapper)
        pm.register_patch('torch.distributed.all_reduce', torch_all_reduce_double_dtype_bypass_wrapper)
        pm.register_patch('torch._C._jit_set_nvfuser_enabled', dummy_function)
        # lmc is supported python >=3.9
        if sys.version_info < (3, 9):
            pm.register_patch('math.lcm', lcm, create_dummy=True)
        pm.register_patch('importlib.metadata.version', version_wrapper)

        # transformer_engine_torch.thd_get_partitioned_indices is deliberately
        # NOT patched: TE-NPU ships that function itself, and a local replacement
        # would shadow it (double-write).

    def fused_weight_gradient_adaptation(self, pm, args):
        def _npu_wgrad_gemm_accum_fp32(total_input, grad_output, main_grad):
            from megatron_adaptor.patches.transformer_engine.transformer_engine import (
                npu_wgrad_gemm_accum_fp32,
            )

            npu_wgrad_gemm_accum_fp32(total_input, grad_output, main_grad)

        def _npu_wgrad_gemm_accum_fp16(total_input, grad_output, main_grad):
            from megatron_adaptor.patches.transformer_engine.transformer_engine import (
                npu_wgrad_gemm_accum_fp16,
            )

            npu_wgrad_gemm_accum_fp16(total_input, grad_output, main_grad)

        pm.register_patch(
            "fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32",
            _npu_wgrad_gemm_accum_fp32,
            create_dummy=True,
        )
        pm.register_patch(
            "fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16",
            _npu_wgrad_gemm_accum_fp16,
            create_dummy=True,
        )
