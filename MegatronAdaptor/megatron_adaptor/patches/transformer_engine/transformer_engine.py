# Copyright 2025 Thinking Machines Lab
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps

HAVE_TE = False
try:
    import transformer_engine  # noqa

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    pass


def _normalize_stale_activation_func(config):
    """Rebind a stale activation_func so megatron's identity checks pass.

    msprobe (npu-datadump) wraps F.gelu/F.silu, but TransformerConfig's
    dataclass default was baked with the pre-wrap F.gelu (defaults are
    fixed at class creation and cannot be re-bound). Configs built from
    that stale default would fail the identity checks in __post_init__
    and MLP.forward, so rebind the instance attribute to the current
    (wrapped) equivalent, matched by function name.
    """
    if not getattr(config, 'bias_activation_fusion', False):
        return
    import torch.nn.functional as F

    from megatron.core.transformer.transformer_config import quick_gelu

    # Megatron's identity-check whitelist (F.relu covers the
    # use_te_activation_func check); in the dump scenario F.gelu/F.silu/F.relu
    # here are the msprobe wrappers, resolved at call time.
    valid_funcs = (F.gelu, F.silu, F.relu, quick_gelu)
    if config.activation_func in valid_funcs:
        return
    by_name = {getattr(func, '__name__', None): func for func in valid_funcs}
    replacement = by_name.get(getattr(config.activation_func, '__name__', None))
    if replacement is not None:
        config.activation_func = replacement


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        _normalize_stale_activation_func(self)
        fn(self)

    return wrapper


def get_npu_grad_accum_fusion_available():
    if not HAVE_TE:
        return False
    try:
        from torch.npu import is_available

        return is_available()
    except (ImportError, AttributeError):
        return False


def npu_wgrad_gemm_accum_fp32(total_input, grad_output, main_grad):
    if not HAVE_TE:
        raise RuntimeError("TransformerEngineNPU is not available. Cannot use fused GEMM accumulation.")
    from transformer_engine.pytorch.ops.gemm import wgrad_gemm_accum_fp32

    wgrad_gemm_accum_fp32(total_input, grad_output, main_grad)


def npu_wgrad_gemm_accum_fp16(total_input, grad_output, main_grad):
    if not HAVE_TE:
        raise RuntimeError("TransformerEngineNPU is not available. Cannot use fused GEMM accumulation.")
    from transformer_engine.pytorch.ops.gemm import wgrad_gemm_accum_fp16

    wgrad_gemm_accum_fp16(total_input, grad_output, main_grad)


def column_parallel_linear_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except RuntimeError as e:
            if "fused_weight_gradient_mlp_cuda" in str(e):
                raise RuntimeError(
                    "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                    "to True but the NPU fused GEMM accumulation operator is not available. "
                    "Please check your NPU hardware and CANN installation, or turn off "
                    "gradient accumulation fusion with --no-gradient-accumulation-fusion."
                )
            raise

    return wrapper
