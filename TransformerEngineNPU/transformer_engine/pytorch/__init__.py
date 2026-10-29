# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine bindings for PyTorch (Ascend NPUs)"""

# ruff: noqa: F401
# pylint: disable=wrong-import-position
import signal

import torch

from . import distributed, tensor
from .module import (
    Fp8Padding,
    Fp8Unpadding,
    TransformerEngineBaseModule,
    GroupedLinear,
    LayerNorm,
    LayerNormMLP,
    LayerNormLinear,
    Linear,
    RMSNorm,
)
from .quantization import (
    autocast,
    fp8_autocast,
    fp8_model_init,
    get_default_recipe,
    is_fp8_available,
    quantized_model_init,
    reset_fp8_amax_history,
    reset_hif8_amax_history,
)
from .cpu_offload import (
    get_cpu_offload_context,
    mark_not_offload,
    start_offload,
    is_cpu_offload_enabled,
)
from .permutation import (
    moe_permute,
    moe_unpermute,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
    moe_permute_with_probs,
    moe_permute_and_pad_with_probs,
)
from .attention import DotProductAttention
from .ops.gemm_comparison import (
    GemmComparisonConfig,
    configure_gemm_comparison,
    get_gemm_comparison_config,
)
from .cross_entropy import parallel_cross_entropy
from .tensor.float8_blockwise_tensor import (
    _make_float8_blockwise_tensor_in_reduce_ex,
)
from .tensor.float8_tensor import _make_float8_tensor_in_reduce_ex
from .tensor.mxfp8_tensor import _make_mxfp8_tensor_in_reduce_ex
from .tensor import Float8Tensor  # noqa: E402  (public top-level export, matches NVTE)

from .optimizers import (
    MultiTensorApply,
    multi_tensor_applier,
    multi_tensor_scale,
    multi_tensor_scale_tensor,
    multi_tensor_l2norm,
    multi_tensor_unscale_l2norm,
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)

try:
    torch._dynamo.config.error_on_nested_jit_trace = False
except AttributeError:
    pass

# Allow QuantizedTensor checkpoints to load with ``weights_only=True``.
try:
    from torch.serialization import add_safe_globals

    add_safe_globals(
        [
            # Storage mixins used when pickling internal tensors.
            tensor.QuantizedTensorStorage,
            tensor.Float8TensorStorage,
            tensor.MXFP8TensorStorage,
            tensor.Float8BlockwiseQTensorStorage,
            # Quantizer types embedded in tensor metadata.
            tensor.Quantizer,
            tensor.Float8Quantizer,
            tensor.Float8CurrentScalingQuantizer,
            tensor.MXFP8Quantizer,
            tensor.MXFP832x32Quantizer,
            tensor.Float8BlockQuantizer,
            # Module-level reconstructors returned by ``__reduce_ex__``.
            _make_float8_tensor_in_reduce_ex,
            _make_mxfp8_tensor_in_reduce_ex,
            _make_float8_blockwise_tensor_in_reduce_ex,
            # Megatron stores args.exit_signal in checkpoints.
            signal.Signals,
            (signal.SIGTERM, "signal.SIGTERM"),
        ]
    )
except (ImportError, AttributeError):
    import warnings as _warnings

    _warnings.warn(
        "transformer_engine: torch.serialization.add_safe_globals is unavailable; "
        "weights-only loading of QuantizedTensor checkpoints will not work",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = [
    "parallel_cross_entropy",
    "tensor",
    "Float8Tensor",
    "distributed",
    "DotProductAttention",
    "GemmComparisonConfig",
    "Linear",
    "GroupedLinear",
    "LayerNorm",
    "LayerNormLinear",
    "LayerNormMLP",
    "RMSNorm",
    "TransformerEngineBaseModule",
    "fp8_autocast",
    "fp8_model_init",
    "autocast",
    "quantized_model_init",
    "is_fp8_available",
    "get_default_recipe",
    "configure_gemm_comparison",
    "get_gemm_comparison_config",
    "reset_fp8_amax_history",
    "reset_hif8_amax_history",
    "get_cpu_offload_context",
    "mark_not_offload",
    "start_offload",
    "is_cpu_offload_enabled",
    "Fp8Padding",
    "Fp8Unpadding",
    "moe_permute",
    "moe_unpermute",
    "moe_sort_chunks_by_index",
    "moe_sort_chunks_by_index_with_probs",
    "moe_permute_with_probs",
    "moe_permute_and_pad_with_probs",
    "MultiTensorApply",
    "multi_tensor_applier",
    "multi_tensor_scale",
    "multi_tensor_scale_tensor",
    "multi_tensor_l2norm",
    "multi_tensor_unscale_l2norm",
    "multi_tensor_compute_scale_and_scale_inv",
    "multi_tensor_compute_scale_inv_e8m0",
]
