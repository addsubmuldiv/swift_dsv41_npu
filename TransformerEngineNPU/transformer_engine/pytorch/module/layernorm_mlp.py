# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
import dataclasses
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch_npu

from ..constants import FP8BwdTensorIdx, FP8FwdTensorIdx, TensorUsage, dist_group_type
from ..ops.basic.npu_activation import (
    ACTIVATION_FWD,
    ACTIVATION_BWD,
    GLU_VARIANTS,
)
from ..distributed import (
    _fsdp_gather_tensors,
    _fsdp_scatter_tensors,
    gather_along_dim,
    get_distributed_world_size,
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
    reduce_scatter_along_dim,
)
from ..jit import no_torch_dynamo
from ..ops.fused.overlap import CommOverlapOps
from ..ops.gemm import general_gemm, general_gemm_add
from ..quantization import FP8GlobalStateManager, QuantizerRole
from ..quantized_tensor import (
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_func_ctx,
    restore_from_saved,
)
from ..tensor.utils import clear_columnwise_cache
from ..utils import (
    cast_if_needed,
    divide,
    get_default_init_method,
    init_method_constant,
    requires_grad,
)
from ._common import WeightGradStore
from .base import (
    TransformerEngineBaseModule,
    _check_fp8_reduce_and_update,
    quantize_weight,
    setup_dummy_wgrad,
)


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


@dataclass(slots=True)
class _LayerNormMLPArgs:
    is_first_microbatch: bool
    fp8: bool
    debug: bool
    fp8_calibration: bool
    fp8_meta: Any
    wgrad_store: WeightGradStore
    eps: float
    fc1_input_quantizer: Optional[Quantizer]
    fc1_weight_quantizer: Optional[Quantizer]
    fc1_output_quantizer: Optional[Quantizer]
    fc1_grad_input_quantizer: Optional[Quantizer]
    fc1_grad_weight_quantizer: Optional[Quantizer]
    fc1_grad_output_quantizer: Optional[Quantizer]
    fc2_input_quantizer: Optional[Quantizer]
    fc2_weight_quantizer: Optional[Quantizer]
    fc2_output_quantizer: Optional[Quantizer]
    fc2_grad_input_quantizer: Optional[Quantizer]
    fc2_grad_weight_quantizer: Optional[Quantizer]
    fc2_grad_output_quantizer: Optional[Quantizer]
    fuse_wgrad_accumulation: bool
    tp_group: Optional[dist_group_type]
    tp_size: int
    sequence_parallel: bool
    activation_dtype: torch.dtype
    tensor_parallel: bool
    set_parallel_mode: bool
    is_grad_enabled: bool
    return_layernorm_output: bool
    return_layernorm_output_gathered: bool
    zero_centered_gamma: bool
    normalization: str
    activation: str
    activation_params: Optional[dict]
    overlap_ag_fprop: bool
    overlap_rs_fprop: bool
    overlap_ag_dgrad: bool
    overlap_rs_dgrad: bool
    skip_fp8_weight_update: Optional[bool]
    fp8_output: bool
    fc1_weight_workspace: Optional[QuantizedTensorStorage]
    fc2_weight_workspace: Optional[QuantizedTensorStorage]
    cache_weight: bool
    fsdp_group: Optional[Any]
    is_fsdp2: bool
    checkpoint: bool
    recompute_for_bwd: bool

    @classmethod
    def from_args(cls, other: "_LayerNormMLPArgs", **overrides) -> "_LayerNormMLPArgs":
        kwargs = {f.name: getattr(other, f.name) for f in dataclasses.fields(cls)}
        kwargs.update(overrides)
        return cls(**kwargs)


def _apply_norm(
    inputmat,
    ln_weight,
    ln_bias,
    eps,
    normalization,
    zero_centered_gamma,
    activation_dtype=None,
):
    """Normalization forward via the fused kernels (ops.LayerNorm/ops.RMSNorm recipe).

    LayerNorm uses torch.native_layer_norm (aclnnLayerNorm) and keeps the
    kernel's mean/rstd for the backward pass, so forward and backward share
    the exact same statistics. RMSNorm uses torch_npu.npu_rms_norm, whose
    returned rstd feeds the fused backward. zero_centered_gamma promotes the
    whole LayerNorm computation to fp32 (aclnnLayerNorm requires matching
    x/w/b dtypes and (1+gamma) benefits from the higher precision); the output
    is cast back to the input dtype.

    The bf16 non-zc path also runs at fp32: on bf16 input the kernel rounds
    the saved statistics to bf16 (NVTE's fused kernel keeps fp32), and a bf16
    rstd injects a ~0.4% correlated error into every dgamma/dbeta product
    (REPORT atk 3#16). The bf16 output is unchanged (one rounding either way).
    """
    if normalization == "LayerNorm":
        gamma = ln_weight
        bias = ln_bias
        if activation_dtype is not None:
            gamma = cast_if_needed(gamma, activation_dtype)
            if bias is not None:
                bias = cast_if_needed(bias, activation_dtype)
        if zero_centered_gamma:
            ln_out, mu, rsigma = torch.native_layer_norm(
                inputmat.float(),
                (inputmat.shape[-1],),
                gamma.float() + 1.0,
                bias.float() if bias is not None else None,
                eps,
            )
            ln_out = ln_out.to(inputmat.dtype)
        elif inputmat.dtype != torch.float32:
            ln_out, mu, rsigma = torch.native_layer_norm(
                inputmat.float(),
                (inputmat.shape[-1],),
                gamma.float(),
                bias.float() if bias is not None else None,
                eps,
            )
            ln_out = ln_out.to(inputmat.dtype)
        else:
            ln_out, mu, rsigma = torch.native_layer_norm(
                inputmat,
                (inputmat.shape[-1],),
                gamma,
                bias,
                eps,
            )
    elif normalization == "RMSNorm":
        gamma = ln_weight
        if activation_dtype is not None:
            gamma = cast_if_needed(gamma, activation_dtype)
        if zero_centered_gamma:
            gamma = gamma.float() + 1.0
        ln_out, rrsigma = torch_npu.npu_rms_norm(inputmat, gamma, epsilon=eps)
        mu = None
        rsigma = rrsigma
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return ln_out, mu, rsigma


def _norm_bwd(
    dgrad,
    inputmat,
    ln_weight,
    ln_bias,
    mu,
    rsigma,
    zero_centered_gamma,
    normalization,
):
    """Normalization backward via the fused kernels (ops.LayerNorm/ops.RMSNorm recipe).

    LayerNorm uses torch.ops.aten.native_layer_norm_backward
    (aclnnLayerNormBackward, the primitive F.layer_norm's autograd dispatches
    to, fp32 accumulation) with the forward kernel's mu/rsigma. RMSNorm uses
    torch_npu.npu_rms_norm_backward. ``ln_weight``/``ln_bias`` are saved in
    the parameter dtype, so they are cast to the activation dtype first.

    The LayerNorm forward promotes bf16 computations to fp32 (full-precision
    saved statistics, matching NVTE), so the saved inputmat/mu/rsigma are
    fp32 and the backward math runs at fp32 too, cast back at the end.
    """
    if normalization == "LayerNorm":
        inner_dim = inputmat.shape[-1]
        dtype = dgrad.dtype
        if mu.dtype == torch.float32 and dtype != torch.float32:
            # Forward's fp32-stats regime: run the backward math at fp32 too
            # (the saved bf16 inputmat is promoted losslessly here; the saved
            # mu/rsigma are already fp32).
            dy = dgrad.float()
            x = inputmat.float()
            weight = ln_weight.float()
            bias = ln_bias.float() if ln_bias is not None else None
        else:
            x = inputmat
            weight = ln_weight.to(inputmat.dtype)
            bias = ln_bias.to(inputmat.dtype) if ln_bias is not None else None
            dy = dgrad
        if zero_centered_gamma:
            weight = weight.float() + 1.0
        # aten rejects bias=None with output_mask[2]=True; request db only when
        # the bias exists.
        dx, dweight, dbias = torch.ops.aten.native_layer_norm_backward(
            dy, x, (inner_dim,), mu, rsigma, weight, bias, [True, True, bias is not None]
        )
        dx = dx.to(dtype)
        dgamma = dweight.to(ln_weight.dtype)
        dbeta = dbias.to(ln_bias.dtype) if dbias is not None else None
        return dx, dgamma, dbeta
    if normalization == "RMSNorm":
        dtype = inputmat.dtype
        weight = ln_weight.to(dtype)
        if zero_centered_gamma:
            weight = weight.float() + 1.0
        dx, dweight = torch_npu.npu_rms_norm_backward(dgrad, inputmat, weight, rsigma)
        dgamma = dweight.to(ln_weight.dtype)
        return dx, dgamma, None
    raise ValueError(f"Unknown normalization: {normalization}")


def _layer_norm_mlp_setup_ctx(
    ctx,
    args,
    inp,
    tensors_to_save,
    tensor_objects,
    ctx_attrs,
    ln_weight,
    ln_bias,
    fc1_weight,
    fc2_weight,
):
    ctx.save_for_backward(*tensors_to_save)
    ctx.tensor_objects = tensor_objects
    ctx.args = args
    ctx.inp_shape = inp.shape
    ctx.fp8 = args.fp8
    ctx.debug = args.debug
    ctx.sequence_parallel = args.sequence_parallel
    ctx.requires_dgrad = requires_grad(inp, ln_weight, ln_bias)
    ctx.fc1_weight_requires_grad = fc1_weight.requires_grad
    ctx.fc2_weight_requires_grad = fc2_weight.requires_grad
    ctx.requires_wgrad = ctx.fc1_weight_requires_grad or ctx.fc2_weight_requires_grad
    ctx.fsdp_group = args.fsdp_group
    ctx.is_fsdp2 = args.is_fsdp2

    if args.fp8 or args.debug:
        ctx.is_weight_param_quantized = ctx_attrs["is_weight_param_quantized"]
        ctx.fsdp_shapes = ctx_attrs["fsdp_shapes"]
        if args.fp8 and ctx_attrs.get("reduce_and_update"):
            ctx.reduce_and_update_bwd_fp8_tensors = _check_fp8_reduce_and_update()
    else:
        ctx.fsdp_shapes = ctx_attrs["fsdp_shapes"]

    if not args.fp8 and args.fuse_wgrad_accumulation:
        ctx.fc1_weight_python_object_ref = ctx_attrs.get("fc1_weight_python_object_ref")
        ctx.fc2_weight_python_object_ref = ctx_attrs.get("fc2_weight_python_object_ref")
        ctx.fc1_main_grad_func = ctx_attrs.get("fc1_main_grad_func")
        ctx.fc2_main_grad_func = ctx_attrs.get("fc2_main_grad_func")
        ctx.fc1_weight_overwrites_main_grad = ctx_attrs.get(
            "fc1_weight_overwrites_main_grad", False
        )
        ctx.fc2_weight_overwrites_main_grad = ctx_attrs.get(
            "fc2_weight_overwrites_main_grad", False
        )


def _save_backward_state(
    ctx,
    args,
    inp,
    inputmat,
    ln_weight,
    ln_bias,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
    fc1_weight_final,
    fc2_weight_final,
    fc1_input,
    fc1_out,
    fc2_input,
    act_out,
    mu,
    rsigma,
):
    ctx_attrs = {}

    if args.fp8 or args.debug:
        is_weight_param_quantized = isinstance(fc1_weight, QuantizedTensorStorage)
        fsdp2_skip_columnwise = args.is_fsdp2 and not args.recompute_for_bwd

        fc1_wt_save = (
            None
            if fsdp2_skip_columnwise and fc1_weight_final is not fc1_weight
            else fc1_weight_final
        )
        fc2_wt_save = (
            None
            if fsdp2_skip_columnwise and fc2_weight_final is not fc2_weight
            else fc2_weight_final
        )
        fsdp_shapes = _fsdp_scatter_tensors(
            args.fsdp_group,
            mu,
            rsigma,
            fc1_weight_final if not is_weight_param_quantized else None,
            fc2_weight_final if not is_weight_param_quantized else None,
            fc1_input,
            fc1_out,
            fc2_input,
            act_out,
        )
        ctx_attrs["reduce_and_update"] = requires_grad(
            inp,
            ln_weight,
            ln_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        )
        ctx_attrs["is_weight_param_quantized"] = is_weight_param_quantized
    else:
        fc1_wt_save = fc2_wt_save = None
        fsdp_shapes = _fsdp_scatter_tensors(
            args.fsdp_group, mu, rsigma, fc1_input, fc1_out, act_out
        )
    if not args.fp8 and args.fuse_wgrad_accumulation:
        ctx_attrs["fc1_weight_python_object_ref"] = (
            weakref.ref(fc1_weight) if fc1_weight.requires_grad else None
        )
        ctx_attrs["fc2_weight_python_object_ref"] = (
            weakref.ref(fc2_weight) if fc2_weight.requires_grad else None
        )
        if fc1_weight.requires_grad:
            ctx_attrs["fc1_main_grad_func"] = (
                fc1_weight.get_main_grad
                if hasattr(fc1_weight, "__fsdp_param__")
                else lambda: fc1_weight.main_grad
            )
        else:
            ctx_attrs["fc1_main_grad_func"] = None
        if fc2_weight.requires_grad:
            ctx_attrs["fc2_main_grad_func"] = (
                fc2_weight.get_main_grad
                if hasattr(fc2_weight, "__fsdp_param__")
                else lambda: fc2_weight.main_grad
            )
        else:
            ctx_attrs["fc2_main_grad_func"] = None
        ctx_attrs["fc1_weight_overwrites_main_grad"] = getattr(
            fc1_weight, "overwrite_main_grad", False
        )
        ctx_attrs["fc2_weight_overwrites_main_grad"] = getattr(
            fc2_weight, "overwrite_main_grad", False
        )
    ctx_attrs["fsdp_shapes"] = fsdp_shapes

    tensors_to_save, tensor_objects = prepare_for_saving(
        inputmat,
        ln_weight,
        ln_bias,
        fc1_wt_save,
        fc1_weight,
        fc1_bias,
        fc2_wt_save,
        fc2_weight,
        fc2_bias,
        fc1_input,
        fc1_out,
        fc2_input,
        act_out,
        mu,
        rsigma,
    )

    _layer_norm_mlp_setup_ctx(
        ctx,
        args,
        inp,
        tensors_to_save,
        tensor_objects,
        ctx_attrs,
        ln_weight,
        ln_bias,
        fc1_weight,
        fc2_weight,
    )

    restored_tensors = None
    if args.recompute_for_bwd:
        restored_tensors = restore_from_saved(tensor_objects, tensors_to_save)

    return fc1_wt_save, fc2_wt_save, restored_tensors


def _layer_norm_mlp_forward_impl(
    ctx,
    inp,
    ln_weight,
    ln_bias,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
    args: _LayerNormMLPArgs,
):
    inputmat = inp.reshape(-1, inp.shape[-1])
    inputmat = cast_if_needed(inputmat, args.activation_dtype)

    save_for_checkpoint = args.checkpoint and args.is_grad_enabled and not args.recompute_for_bwd
    is_recomputation = args.checkpoint and args.is_grad_enabled and args.recompute_for_bwd

    if save_for_checkpoint:
        ctx.autocast_state = FP8GlobalStateManager.get_autocast_state()
        if args.fp8 and (
            args.fp8_meta["recipe"].hif8_delayed() or args.fp8_meta["recipe"].delayed()
        ):
            FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(args.fp8_meta)

        tensors_to_save, tensor_objects = prepare_for_saving(
            inp,
            ln_weight,
            ln_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.args = args
        ctx.checkpoint = args.checkpoint

    ln_out, mu, rsigma = _apply_norm(
        inputmat,
        ln_weight,
        ln_bias,
        args.eps,
        args.normalization,
        args.zero_centered_gamma,
        args.activation_dtype,
    )

    ln_out_return = None
    if args.return_layernorm_output and not is_recomputation:
        ln_out_return = ln_out.reshape(inp.shape)
        if (
            args.return_layernorm_output_gathered
            and args.tensor_parallel
            and args.sequence_parallel
        ):
            ln_out_return, _ = gather_along_dim(ln_out_return, args.tp_group)

    ln_out_total = ln_out
    if (
        args.tensor_parallel
        and args.sequence_parallel
        and args.set_parallel_mode
        and not args.overlap_ag_fprop
    ):
        ln_out_total, _ = gather_along_dim(ln_out_total, args.tp_group)

    new_fc1_weight_workspace = None
    new_fc2_weight_workspace = None
    update_ws = args.is_first_microbatch is None or args.is_first_microbatch

    # ===== FC1 =====
    fc1_input_columnwise = False
    fc2_input_columnwise = False
    if args.fp8 or args.debug:
        backwards_needs_saved_activations = args.is_grad_enabled and (
            not args.checkpoint or is_recomputation
        )
        backwards_needs_fc1_input = backwards_needs_saved_activations and fc1_weight.requires_grad
        backward_needs_fc1_weight = args.is_grad_enabled and requires_grad(inp, ln_weight, ln_bias)
        backward_needs_fc2_input = backwards_needs_saved_activations and fc2_weight.requires_grad
        backward_needs_fc2_weight = args.is_grad_enabled

        activation_recompute_forward = (
            is_fp8_activation_recompute_enabled() and not in_fp8_activation_recompute_phase()
        )
        fsdp2_skip_columnwise = args.is_fsdp2 and not is_recomputation
        fc1_input_columnwise = backwards_needs_fc1_input
        fc2_input_columnwise = backward_needs_fc2_input
        fc1_weight_columnwise = backward_needs_fc1_weight and not fsdp2_skip_columnwise
        fc2_weight_columnwise = backward_needs_fc2_weight and not fsdp2_skip_columnwise
        if activation_recompute_forward:
            fc1_weight_columnwise = fc1_weight_columnwise or not fsdp2_skip_columnwise
            fc2_weight_columnwise = fc2_weight_columnwise or not fsdp2_skip_columnwise

        args.fc1_input_quantizer.set_usage(rowwise=True, columnwise=fc1_input_columnwise)
        args.fc2_input_quantizer.set_usage(rowwise=True, columnwise=fc2_input_columnwise)
        args.fc1_weight_quantizer.set_usage(rowwise=True, columnwise=fc1_weight_columnwise)
        args.fc2_weight_quantizer.set_usage(rowwise=True, columnwise=fc2_weight_columnwise)

        if fsdp2_skip_columnwise:
            args.fc1_weight_quantizer.set_usage(columnwise=False)
            args.fc2_weight_quantizer.set_usage(columnwise=False)

        fc1_input = args.fc1_input_quantizer(ln_out_total)
        fc1_weight_final, new_fc1_weight_workspace = quantize_weight(
            tensor=fc1_weight,
            quantizer=args.fc1_weight_quantizer,
            workspace=args.fc1_weight_workspace,
            update_workspace=update_ws,
            skip_update_flag=args.skip_fp8_weight_update,
            fsdp_group=args.fsdp_group,
            workspace_dtype=args.activation_dtype,
            cache=args.cache_weight,
        )
        fc1_weight_final.update_usage(rowwise_usage=True)
    else:
        fc1_input = cast_if_needed(ln_out_total, args.activation_dtype)
        fc1_weight_final = cast_if_needed(fc1_weight, args.activation_dtype)
        if args.fp8_calibration:
            if args.fc1_input_quantizer is not None:
                args.fc1_input_quantizer.calibrate(ln_out_total)
            if args.fc1_weight_quantizer is not None:
                args.fc1_weight_quantizer.calibrate(fc1_weight)

    fc1_mm_kwargs = {
        "usage_a": TensorUsage.LHS,
        "usage_b": TensorUsage.RHS_TRANS,
        "out_dtype": args.activation_dtype,
    }
    if args.overlap_ag_fprop:
        fc1_out, fc1_input = CommOverlapOps.allgather_matmul(
            fc1_input,
            fc1_weight_final,
            fc1_bias,
            get_distributed_world_size(args.tp_group),
            args.tp_group,
            **fc1_mm_kwargs,
        )
    else:
        fc1_out = general_gemm(
            fc1_input,
            fc1_weight_final,
            bias=fc1_bias,
            quantization_params=args.fc1_output_quantizer if args.debug else None,
            **fc1_mm_kwargs,
        )
    if isinstance(fc1_input, QuantizedTensorStorage) and fc1_input_columnwise:
        fc1_input.update_usage(rowwise_usage=False, columnwise_usage=True)

    # ===== Activation =====
    activation_func = ACTIVATION_FWD[args.activation]
    act_params = args.activation_params or {}
    if args.activation == "clamped_swiglu":
        act_out = activation_func(fc1_out, **act_params)
    else:
        act_out = activation_func(fc1_out)
    if (
        not args.fp8
        and not args.debug
        and args.fp8_calibration
        and args.fc2_input_quantizer is not None
    ):
        args.fc2_input_quantizer.calibrate(act_out)

    # ===== FC2 setup (quantize/cast) =====
    if args.fp8 or args.debug:
        fc2_input = args.fc2_input_quantizer.quantize(act_out)
        fc2_weight_final, new_fc2_weight_workspace = quantize_weight(
            tensor=fc2_weight,
            quantizer=args.fc2_weight_quantizer,
            workspace=args.fc2_weight_workspace,
            update_workspace=update_ws,
            skip_update_flag=args.skip_fp8_weight_update,
            fsdp_group=args.fsdp_group,
            workspace_dtype=args.activation_dtype,
            cache=args.cache_weight,
        )
        fc2_weight_final.update_usage(rowwise_usage=True)
    else:
        fc2_input = cast_if_needed(act_out, args.activation_dtype)
        fc2_weight_final = cast_if_needed(fc2_weight, args.activation_dtype)
        if args.fp8_calibration and args.fc2_weight_quantizer is not None:
            args.fc2_weight_quantizer.calibrate(fc2_weight)

    # ===== Recomp check (after FC1 + activation + FC2 setup) =====
    if is_recomputation:
        _, _, backward_tensors = _save_backward_state(
            ctx,
            args,
            inp,
            inputmat,
            ln_weight,
            ln_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc1_weight_final,
            fc2_weight_final,
            fc1_input,
            fc1_out,
            fc2_input,
            act_out,
            mu,
            rsigma,
        )
        return ctx, *backward_tensors

    # ===== FC2 compute =====
    fc2_mm_kwargs = {
        "usage_a": TensorUsage.LHS,
        "usage_b": TensorUsage.RHS_TRANS,
        "out_dtype": args.activation_dtype,
    }
    if args.overlap_rs_fprop:
        fc2_out = CommOverlapOps.matmul_reduce_scatter(
            fc2_input,
            fc2_weight_final,
            fc2_bias,
            get_distributed_world_size(args.tp_group),
            args.tp_group,
            **fc2_mm_kwargs,
        )
    else:
        fc2_out = general_gemm(
            fc2_input,
            fc2_weight_final,
            bias=fc2_bias,
            quantization_params=args.fc2_output_quantizer if args.debug else None,
            **fc2_mm_kwargs,
        )
    if isinstance(fc2_input, QuantizedTensorStorage) and fc2_input_columnwise:
        fc2_input.update_usage(rowwise_usage=False, columnwise_usage=True)

    # ===== TP reduction =====
    tp_world_size = get_distributed_world_size(args.tp_group)
    if (
        args.tensor_parallel
        and args.set_parallel_mode
        and not args.overlap_rs_fprop
        and tp_world_size > 1
    ):
        if args.sequence_parallel:
            if args.fp8:
                recipe = FP8GlobalStateManager.get_fp8_recipe()
                fc2_out, _ = CommOverlapOps.reduce_scatter(
                    fc2_out,
                    args.fc2_output_quantizer,
                    tp_world_size,
                    args.tp_group,
                    use_quant=args.fp8_output and recipe.mxfp8(),
                )
            else:
                fc2_out, _ = reduce_scatter_along_dim(fc2_out, args.tp_group)
        else:
            torch.distributed.all_reduce(fc2_out, group=args.tp_group)

    # ===== Backward state =====
    if args.is_grad_enabled and not save_for_checkpoint:
        _save_backward_state(
            ctx,
            args,
            inp,
            inputmat,
            ln_weight,
            ln_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc1_weight_final,
            fc2_weight_final,
            fc1_input,
            fc1_out,
            fc2_input,
            act_out,
            mu,
            rsigma,
        )

    out = fc2_out.reshape(*inp.shape[:-1], fc2_out.shape[-1])
    ln_out_for_return = ln_out_return if args.return_layernorm_output else None
    return out, ln_out_for_return, new_fc1_weight_workspace, new_fc2_weight_workspace


def _layer_norm_mlp_recompute(ctx):
    tensors = restore_from_func_ctx(ctx)
    if getattr(ctx, "checkpoint", False):
        (
            inp_r,
            ln_weight_r,
            ln_bias_r,
            fc1_weight_r,
            fc1_bias_r,
            fc2_weight_r,
            fc2_bias_r,
        ) = tensors
        args = _LayerNormMLPArgs.from_args(
            ctx.args,
            fc1_weight_workspace=None,
            fc2_weight_workspace=None,
            recompute_for_bwd=True,
        )
        final_autocast_state = FP8GlobalStateManager.get_autocast_state()

        FP8GlobalStateManager.set_autocast_state(ctx.autocast_state)
        if args.fp8 and (
            args.fp8_meta["recipe"].hif8_delayed() or args.fp8_meta["recipe"].delayed()
        ):
            FP8GlobalStateManager.get_old_fp8_meta_tensors_for_recompute(args.fp8_meta)
        try:
            return _layer_norm_mlp_forward_impl(
                ctx,
                inp_r,
                ln_weight_r,
                ln_bias_r,
                fc1_weight_r,
                fc1_bias_r,
                fc2_weight_r,
                fc2_bias_r,
                args,
            )
        finally:
            FP8GlobalStateManager.set_autocast_state(final_autocast_state)
            if args.fp8 and (
                args.fp8_meta["recipe"].hif8_delayed() or args.fp8_meta["recipe"].delayed()
            ):
                FP8GlobalStateManager.restore_fp8_meta_tensors(args.fp8_meta)
    return ctx, *tensors


def _layer_norm_mlp_backward(
    ctx,
    grad_output: torch.Tensor,
    *rest,
):
    recompute_result = _layer_norm_mlp_recompute(ctx)

    (
        ctx,
        inputmat,
        ln_weight,
        ln_bias,
        fc1_wt_save,
        fc1_weight,
        fc1_bias,
        fc2_wt_save,
        fc2_weight,
        fc2_bias,
        fc1_input,
        fc1_out,
        fc2_input,
        act_out,
        mu,
        rsigma,
    ) = recompute_result
    is_fsdp2 = getattr(ctx, "is_fsdp2", False)
    is_weight_param_quantized = getattr(ctx, "is_weight_param_quantized", False)

    args = ctx.args

    # A saved-tensor hook may return detached weights without the Python
    # attributes installed by MCore DDP.  Recover the original Parameters and
    # main_grad buffers for non-FP8 fused accumulation while keeping the saved
    # weights as the GEMM inputs.  Keep the FP8 path unchanged.
    fc1_weight_python_object = None
    fc2_weight_python_object = None
    fc1_weight_main_grad = None
    fc2_weight_main_grad = None
    if not ctx.fp8 and args.fuse_wgrad_accumulation:
        fc1_weight_ref = ctx.fc1_weight_python_object_ref
        fc2_weight_ref = ctx.fc2_weight_python_object_ref
        ctx.fc1_weight_python_object_ref = None
        ctx.fc2_weight_python_object_ref = None
        fc1_weight_python_object = fc1_weight_ref() if fc1_weight_ref is not None else None
        fc2_weight_python_object = fc2_weight_ref() if fc2_weight_ref is not None else None
        if ctx.fc1_weight_requires_grad:
            assert fc1_weight_python_object is not None, (
                "fc1 weight was removed while fuse_wgrad_accumulation=True"
            )
            fc1_weight_main_grad = ctx.fc1_main_grad_func()
            fc1_weight_python_object.main_grad = fc1_weight_main_grad
        if ctx.fc2_weight_requires_grad:
            assert fc2_weight_python_object is not None, (
                "fc2 weight was removed while fuse_wgrad_accumulation=True"
            )
            fc2_weight_main_grad = ctx.fc2_main_grad_func()
            fc2_weight_python_object.main_grad = fc2_weight_main_grad

    fsdp_group = getattr(ctx, "fsdp_group", None)
    if fsdp_group is not None:
        if ctx.fp8 or ctx.debug:
            _fsdp_gather_tensors(
                fsdp_group,
                ctx.fsdp_shapes,
                mu,
                rsigma,
                fc1_wt_save if not is_weight_param_quantized else None,
                fc2_wt_save if not is_weight_param_quantized else None,
                fc1_input,
                fc1_out,
                fc2_input,
                act_out,
            )
        else:
            _fsdp_gather_tensors(
                fsdp_group,
                ctx.fsdp_shapes,
                mu,
                rsigma,
                fc1_input,
                fc1_out,
                act_out,
            )

    grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    if args.return_layernorm_output and len(rest) > 0 and rest[0] is not None:
        grad_ln_out = rest[0].reshape(-1, rest[0].shape[-1])
    else:
        grad_ln_out = None

    tp_world_size = get_distributed_world_size(args.tp_group)

    if ctx.fp8 or ctx.debug:
        if not ctx.requires_dgrad:
            args.fc2_grad_output_quantizer.set_usage(rowwise=False)
        if not ctx.fc2_weight_requires_grad:
            args.fc2_grad_output_quantizer.set_usage(columnwise=False)

    if (
        args.tensor_parallel
        and args.set_parallel_mode
        and args.sequence_parallel
        and not args.overlap_ag_dgrad
    ):
        grad_output, _ = gather_along_dim(grad_output, args.tp_group)

    handle = None

    # ===== FC2 backward (dgrad + wgrad + bias_grad) =====
    if ctx.fp8 or ctx.debug:
        fc2_grad_output = args.fc2_grad_output_quantizer.quantize(grad_output)

        if fc2_wt_save is None and is_fsdp2:
            if isinstance(fc2_weight, QuantizedTensorStorage):
                fc2_wt_save = fc2_weight
            else:
                fc2_weight_quantizer = args.fc2_weight_quantizer
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=True)
                fc2_wt_save = fc2_weight_quantizer(fc2_weight)
        fc2_weight_final = fc2_wt_save
    else:
        fc2_grad_output = grad_output
        fc2_weight_final = fc2_weight

    if args.fp8:
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        can_accumulate_wgrad = (
            args.fuse_wgrad_accumulation
            and (not args.is_first_microbatch if args.is_first_microbatch is not None else True)
            and recipe.mxfp8()
        )
        accumulate_fc2_wgrad = ctx.fc2_weight_requires_grad and can_accumulate_wgrad
        accumulate_fc1_wgrad = ctx.fc1_weight_requires_grad and can_accumulate_wgrad
        fc2_wgrad_out_dtype = (
            fc2_weight.main_grad.dtype if accumulate_fc2_wgrad else args.activation_dtype
        )
        fc1_wgrad_out_dtype = (
            fc1_weight.main_grad.dtype if accumulate_fc1_wgrad else args.activation_dtype
        )
    else:
        accumulate_fc2_wgrad = (
            ctx.fc2_weight_requires_grad
            and args.fuse_wgrad_accumulation
            and fc2_weight_main_grad.dtype == torch.float32
        )
        accumulate_fc1_wgrad = (
            ctx.fc1_weight_requires_grad
            and args.fuse_wgrad_accumulation
            and fc1_weight_main_grad.dtype == torch.float32
        )
        fc2_wgrad_out_dtype = args.activation_dtype
        fc1_wgrad_out_dtype = args.activation_dtype

    if ctx.fp8 and args.overlap_ag_dgrad:
        fc2_dgrad, fc2_grad_output = CommOverlapOps.allgather_matmul(
            fc2_grad_output,
            fc2_weight_final,
            None,
            tp_world_size,
            args.tp_group,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS,
            out_dtype=args.activation_dtype,
        )
    elif ctx.fp8 and args.overlap_rs_dgrad:
        fc2_dgrad = CommOverlapOps.matmul_reduce_scatter(
            fc2_grad_output,
            fc2_weight_final,
            None,
            tp_world_size,
            args.tp_group,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS,
            out_dtype=args.activation_dtype,
        )
    else:
        fc2_dgrad = general_gemm(
            fc2_grad_output,
            fc2_weight_final,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS,
            out_dtype=args.activation_dtype,
            quantization_params=args.fc2_grad_input_quantizer if args.debug else None,
        )

    if (ctx.fp8 or ctx.debug) and is_fsdp2 and isinstance(fc2_weight_final, QuantizedTensorStorage):
        clear_columnwise_cache(fc2_weight_final)

    fc2_wgrad = None
    if ctx.fc2_weight_requires_grad:
        fc2_wgrad_gemm_kwargs = {
            "usage_a": TensorUsage.LHS_TRANS,
            "usage_b": TensorUsage.RHS,
            "out_dtype": fc2_wgrad_out_dtype,
        }

        if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
            fc2_bias_grad_delayed = None
            if fc2_bias is not None:
                fc2_bias_grad_delayed = grad_output.sum(dim=0)

            def fc2_wgrad_gemm(x, dy):
                if accumulate_fc2_wgrad:
                    main_grad = fc2_weight.main_grad if args.fp8 else fc2_weight_main_grad
                    weight = fc2_weight if args.fp8 else fc2_weight_python_object
                    general_gemm_add(
                        main_grad,
                        dy,
                        x,
                        quantization_params=(
                            args.fc2_grad_weight_quantizer if args.debug else None
                        ),
                        **fc2_wgrad_gemm_kwargs,
                    )
                    return setup_dummy_wgrad(weight, main_grad), fc2_bias_grad_delayed
                return (
                    general_gemm(
                        dy,
                        x,
                        quantization_params=(
                            args.fc2_grad_weight_quantizer if args.debug else None
                        ),
                        **fc2_wgrad_gemm_kwargs,
                    ),
                    fc2_bias_grad_delayed,
                )

            args.wgrad_store.put([fc2_input, fc2_grad_output], fc2_wgrad_gemm)
        else:
            if accumulate_fc2_wgrad:
                main_grad = fc2_weight.main_grad if args.fp8 else fc2_weight_main_grad
                weight = fc2_weight if args.fp8 else fc2_weight_python_object
                general_gemm_add(
                    main_grad,
                    fc2_grad_output,
                    fc2_input,
                    quantization_params=(args.fc2_grad_weight_quantizer if args.debug else None),
                    **fc2_wgrad_gemm_kwargs,
                )
                fc2_wgrad = setup_dummy_wgrad(weight, main_grad)
            else:
                fc2_wgrad = general_gemm(
                    fc2_grad_output,
                    fc2_input,
                    quantization_params=(args.fc2_grad_weight_quantizer if args.debug else None),
                    **fc2_wgrad_gemm_kwargs,
                )

    fc2_bias_grad = None
    if fc2_bias is not None:
        fc2_bias_grad = grad_output.sum(dim=0)

    # ===== Activation backward =====
    activation_func_bwd = ACTIVATION_BWD[args.activation]
    act_params = args.activation_params or {}
    if args.activation == "clamped_swiglu":
        dact = activation_func_bwd(fc1_out, fc2_dgrad, **act_params)
    else:
        dact = activation_func_bwd(fc1_out, fc2_dgrad)

    # ===== FC1 backward (dgrad + wgrad + bias_grad) =====
    if ctx.fp8 or ctx.debug:
        if not ctx.requires_dgrad:
            args.fc1_grad_output_quantizer.set_usage(rowwise=False)
        if not ctx.fc1_weight_requires_grad:
            args.fc1_grad_output_quantizer.set_usage(columnwise=False)

        fc1_grad_output = args.fc1_grad_output_quantizer.quantize(dact)

        if fc1_wt_save is None and is_fsdp2:
            if isinstance(fc1_weight, QuantizedTensorStorage):
                fc1_wt_save = fc1_weight
            else:
                fc1_weight_quantizer = args.fc1_weight_quantizer
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=True)
                fc1_wt_save = fc1_weight_quantizer(fc1_weight)

        fc1_weight_final = fc1_wt_save
    else:
        fc1_grad_output = dact
        fc1_weight_final = fc1_weight

    if ctx.fp8 and args.overlap_rs_dgrad:
        fc1_dgrad = CommOverlapOps.matmul_reduce_scatter(
            fc1_grad_output,
            fc1_weight_final,
            None,
            tp_world_size,
            args.tp_group,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS,
            out_dtype=args.activation_dtype,
        )
    else:
        fc1_dgrad = general_gemm(
            fc1_grad_output,
            fc1_weight_final,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS,
            out_dtype=args.activation_dtype,
            quantization_params=args.fc1_grad_input_quantizer if args.debug else None,
        )

    if (ctx.fp8 or ctx.debug) and is_fsdp2 and isinstance(fc1_weight_final, QuantizedTensorStorage):
        clear_columnwise_cache(fc1_weight_final)

    grad_ln_out_gathered = (
        grad_ln_out is not None
        and args.return_layernorm_output_gathered
        and args.sequence_parallel
        and args.set_parallel_mode
        and tp_world_size > 1
    )

    if grad_ln_out_gathered:
        if args.overlap_rs_dgrad:
            grad_ln_out, _ = reduce_scatter_along_dim(grad_ln_out, args.tp_group)
        else:
            fc1_dgrad = fc1_dgrad + grad_ln_out
            grad_ln_out = None

    if args.tensor_parallel and args.set_parallel_mode and tp_world_size > 1:
        if args.sequence_parallel and not args.overlap_rs_dgrad:
            fc1_dgrad, handle = reduce_scatter_along_dim(fc1_dgrad, args.tp_group, async_op=True)
        elif not args.sequence_parallel:
            handle = torch.distributed.all_reduce(fc1_dgrad, group=args.tp_group, async_op=True)

    fc1_wgrad = None
    if ctx.fc1_weight_requires_grad:
        fc1_wgrad_gemm_kwargs = {
            "usage_a": TensorUsage.LHS_TRANS,
            "usage_b": TensorUsage.RHS,
            "out_dtype": fc1_wgrad_out_dtype,
        }

        if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
            fc1_bias_grad_delayed = None
            if fc1_bias is not None:
                fc1_bias_grad_delayed = dact.sum(dim=0)

            def fc1_wgrad_gemm(x, dy):
                if accumulate_fc1_wgrad:
                    main_grad = fc1_weight.main_grad if args.fp8 else fc1_weight_main_grad
                    weight = fc1_weight if args.fp8 else fc1_weight_python_object
                    general_gemm_add(
                        main_grad,
                        dy,
                        x,
                        quantization_params=(
                            args.fc1_grad_weight_quantizer if args.debug else None
                        ),
                        **fc1_wgrad_gemm_kwargs,
                    )
                    return setup_dummy_wgrad(weight, main_grad), fc1_bias_grad_delayed
                return (
                    general_gemm(
                        dy,
                        x,
                        quantization_params=(
                            args.fc1_grad_weight_quantizer if args.debug else None
                        ),
                        **fc1_wgrad_gemm_kwargs,
                    ),
                    fc1_bias_grad_delayed,
                )

            args.wgrad_store.put([fc1_input, fc1_grad_output], fc1_wgrad_gemm)
        else:
            if accumulate_fc1_wgrad:
                main_grad = fc1_weight.main_grad if args.fp8 else fc1_weight_main_grad
                weight = fc1_weight if args.fp8 else fc1_weight_python_object
                general_gemm_add(
                    main_grad,
                    fc1_grad_output,
                    fc1_input,
                    quantization_params=(args.fc1_grad_weight_quantizer if args.debug else None),
                    **fc1_wgrad_gemm_kwargs,
                )
                fc1_wgrad = setup_dummy_wgrad(weight, main_grad)
            else:
                fc1_wgrad = general_gemm(
                    fc1_grad_output,
                    fc1_input,
                    quantization_params=(args.fc1_grad_weight_quantizer if args.debug else None),
                    **fc1_wgrad_gemm_kwargs,
                )

    fc1_bias_grad = None
    if fc1_bias is not None:
        fc1_bias_grad = dact.sum(dim=0)

    if handle is not None:
        handle.wait()
        handle = None

    if grad_ln_out is not None:
        fc1_dgrad = fc1_dgrad + grad_ln_out

    # ===== Norm backward =====
    dx, dgamma, dbeta = _norm_bwd(
        fc1_dgrad,
        inputmat,
        ln_weight,
        ln_bias,
        mu,
        rsigma,
        args.zero_centered_gamma,
        args.normalization,
    )

    if ctx.fp8 and getattr(ctx, "reduce_and_update_bwd_fp8_tensors", False):
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

    return (
        dx.reshape(ctx.inp_shape),
        dgamma,
        dbeta,
        fc1_wgrad,
        fc1_bias_grad,
        fc2_wgrad,
        fc2_bias_grad,
        None,
    )


class _LayerNormMLP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp,
        ln_weight,
        ln_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        args: _LayerNormMLPArgs,
    ):
        args.recompute_for_bwd = False
        return _layer_norm_mlp_forward_impl(
            ctx,
            inp,
            ln_weight,
            ln_bias,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            args,
        )

    @staticmethod
    def backward(ctx, grad_output, *rest):
        return _layer_norm_mlp_backward(ctx, grad_output, *rest)


class LayerNormMLP(TransformerEngineBaseModule):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        return_bias: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        activation: str = "gelu",
        activation_params: Optional[dict] = None,
        output_layer_init_method: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        set_parallel_mode: bool = False,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "npu",
        ub_overlap_ag: bool = False,
        name: Optional[str] = None,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        checkpoint: bool = False,
        rng_tracker_name: Optional[str] = None,
    ):
        super().__init__(name)

        # NVTE uses these dimensions to configure CUDA UserBuffers. The NPU
        # overlap backend does not require them, but they remain part of the API.
        del seq_length, micro_batch_size

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ("LayerNorm", "RMSNorm"), "Unsupported normalization type!"
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        self.zero_centered_gamma = zero_centered_gamma
        self.activation = activation
        self.activation_params = activation_params
        self.symmetric_ar_type = symmetric_ar_type
        self.checkpoint = checkpoint

        self.eps = eps

        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ub_bulk_wgrad)

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.tensor_parallel = self.tp_size > 1
        self.set_parallel_mode = set_parallel_mode
        self.gemm_bias_unfused_add = self.set_parallel_mode and self.apply_bias

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        self.size_per_partition = divide(self.ffn_hidden_size, self.tp_size)

        if activation in GLU_VARIANTS:
            fc1_output_features = 2 * self.size_per_partition
        else:
            fc1_output_features = self.size_per_partition

        self.overlap_ag_fprop = self.set_parallel_mode and self.sequence_parallel and ub_overlap_ag
        self.overlap_rs_dgrad = (
            self.set_parallel_mode and self.sequence_parallel and ub_overlap_rs_dgrad
        )
        self.overlap_rs_fprop = self.set_parallel_mode and self.sequence_parallel and ub_overlap_rs
        self.overlap_ag_dgrad = self.set_parallel_mode and self.sequence_parallel and ub_overlap_ag

        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        layer_norm_weight = torch.empty(hidden_size, device=device, dtype=params_dtype)
        self.register_parameter(
            "layer_norm_weight",
            torch.nn.Parameter(layer_norm_weight),
            init_fn=init_method_constant(float(not self.zero_centered_gamma)),
        )

        if normalization != "RMSNorm":
            layer_norm_bias = torch.zeros(hidden_size, device=device, dtype=params_dtype)
            self.register_parameter(
                "layer_norm_bias",
                torch.nn.Parameter(layer_norm_bias),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.layer_norm_bias = None

        fc1_weight = torch.empty(
            fc1_output_features, hidden_size, device=device, dtype=params_dtype
        )
        self.register_parameter(
            "fc1_weight",
            torch.nn.Parameter(fc1_weight),
            init_fn=init_method,
            get_rng_state_tracker=get_rng_state_tracker,
            fp8_meta_index=FP8FwdTensorIdx.GEMM1_WEIGHT,
        )

        if self.use_bias:
            fc1_bias = torch.empty(fc1_output_features, device=device, dtype=params_dtype)
            self.register_parameter(
                "fc1_bias",
                torch.nn.Parameter(fc1_bias),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.fc1_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        fc2_weight = torch.empty(
            hidden_size, self.size_per_partition, device=device, dtype=params_dtype
        )
        self.register_parameter(
            "fc2_weight",
            torch.nn.Parameter(fc2_weight),
            init_fn=output_layer_init_method,
            get_rng_state_tracker=get_rng_state_tracker,
            fp8_meta_index=FP8FwdTensorIdx.GEMM2_WEIGHT,
        )

        if self.use_bias:
            fc2_bias = torch.empty(hidden_size, device=device, dtype=params_dtype)
            self.register_parameter(
                "fc2_bias",
                torch.nn.Parameter(fc2_bias),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.fc2_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        if with_fp8_params:
            self.init_fp8_metadata(num_gemms=2)

        self.reset_parameters(defer_init=device == "meta")

        if self.wgrad_store.delay_wgrad_compute():
            # pylint: disable=redefined-argument-from-local
            for name, param in self.named_parameters():
                if name in ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"):
                    param.skip_backward_post_hook = True
            # pylint: enable=redefined-argument-from-local

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNormMLP.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.zero_centered_gamma:
            torch.nn.init.ones_(self.layer_norm_weight)
        else:
            torch.nn.init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            torch.nn.init.zeros_(self.layer_norm_bias)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        is_grad_enabled = torch.is_grad_enabled()

        inp = self.prepare_forward(inp, num_gemms=2)

        skip_fp8_weight_update = None

        if is_grad_enabled:
            fwd_fn = _LayerNormMLP.apply
            autograd_ctx = []
        else:
            fwd_fn = _LayerNormMLP.forward
            autograd_ctx = [None]

        debug = self.is_debug_iter()
        if self.fp8 and FP8GlobalStateManager.get_fp8_recipe().backward_override is not None:
            raise NotImplementedError(
                "NVTE_BACKWARD_OVERRIDE=high_precision/dequantized is not implemented in "
                "LayerNormMLP. Replace LayerNormMLP with LayerNormLinear + Linear to enable "
                "high_precision/dequantized backward."
            )
        quantizers = (
            self._get_quantizers(fp8_output, is_grad_enabled)
            if not debug
            else self._get_debug_quantizers(fp8_output, is_grad_enabled)
        )
        if debug and self.no_debug_features_active(quantizers):
            debug = False
            quantizers = self._get_quantizers(fp8_output, is_grad_enabled)
        (
            fc1_input_quantizer,
            fc1_weight_quantizer,
            fc1_output_quantizer,
            fc1_grad_input_quantizer,
            fc1_grad_weight_quantizer,
            fc1_grad_output_quantizer,
            fc2_input_quantizer,
            fc2_weight_quantizer,
            fc2_output_quantizer,
            fc2_grad_input_quantizer,
            fc2_grad_weight_quantizer,
            fc2_grad_output_quantizer,
        ) = quantizers
        fc1_weight, fc2_weight = self._get_weight_tensors()

        cache_name_fc1 = None if (is_first_microbatch is None or self.is_fsdp2) else "fc1_weight"
        cache_name_fc2 = None if (is_first_microbatch is None or self.is_fsdp2) else "fc2_weight"
        fc1_weight_workspace = (
            self._fp8_workspaces.get(cache_name_fc1) if cache_name_fc1 is not None else None
        )
        fc2_weight_workspace = (
            self._fp8_workspaces.get(cache_name_fc2) if cache_name_fc2 is not None else None
        )
        fc2_bias = self.fc2_bias if self.use_bias else None

        args = _LayerNormMLPArgs(
            is_first_microbatch=is_first_microbatch,
            fp8=self.fp8,
            debug=debug,
            fp8_calibration=self.fp8_calibration,
            fp8_meta=self.fp8_meta,
            wgrad_store=self.wgrad_store,
            eps=self.eps,
            fc1_input_quantizer=fc1_input_quantizer,
            fc1_weight_quantizer=fc1_weight_quantizer,
            fc1_output_quantizer=fc1_output_quantizer,
            fc1_grad_input_quantizer=fc1_grad_input_quantizer,
            fc1_grad_weight_quantizer=fc1_grad_weight_quantizer,
            fc1_grad_output_quantizer=fc1_grad_output_quantizer,
            fc2_input_quantizer=fc2_input_quantizer,
            fc2_weight_quantizer=fc2_weight_quantizer,
            fc2_output_quantizer=fc2_output_quantizer,
            fc2_grad_input_quantizer=fc2_grad_input_quantizer,
            fc2_grad_weight_quantizer=fc2_grad_weight_quantizer,
            fc2_grad_output_quantizer=fc2_grad_output_quantizer,
            fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
            sequence_parallel=self.sequence_parallel,
            activation_dtype=self.activation_dtype,
            tensor_parallel=self.tensor_parallel,
            set_parallel_mode=self.set_parallel_mode,
            is_grad_enabled=is_grad_enabled,
            return_layernorm_output=self.return_layernorm_output,
            return_layernorm_output_gathered=self.return_layernorm_output_gathered,
            zero_centered_gamma=self.zero_centered_gamma,
            normalization=self.normalization,
            activation=self.activation,
            activation_params=self.activation_params,
            overlap_ag_fprop=self.overlap_ag_fprop and not debug,
            overlap_rs_fprop=self.overlap_rs_fprop and not debug,
            overlap_ag_dgrad=self.overlap_ag_dgrad and not debug,
            overlap_rs_dgrad=self.overlap_rs_dgrad and not debug,
            skip_fp8_weight_update=skip_fp8_weight_update,
            fp8_output=fp8_output,
            fc1_weight_workspace=fc1_weight_workspace,
            fc2_weight_workspace=fc2_weight_workspace,
            cache_weight=cache_name_fc1 is not None,
            fsdp_group=self.fsdp_group,
            is_fsdp2=self.is_fsdp2,
            checkpoint=self.checkpoint,
            recompute_for_bwd=False,
        )

        try:
            out, ln_out, new_fc1_ws, new_fc2_ws = fwd_fn(
                *autograd_ctx,
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                fc1_weight,
                self.fc1_bias if self.use_bias else None,
                fc2_weight,
                fc2_bias if self.apply_bias and not self.gemm_bias_unfused_add else None,
                args,
            )

            if new_fc1_ws is not None and cache_name_fc1 is not None:
                if isinstance(new_fc1_ws, torch.Tensor):
                    new_fc1_ws = new_fc1_ws.detach()
                self._fp8_workspaces[cache_name_fc1] = new_fc1_ws
            if (
                new_fc2_ws is not None
                and cache_name_fc2 is not None
                and new_fc2_ws is not new_fc1_ws
            ):
                if isinstance(new_fc2_ws, torch.Tensor):
                    new_fc2_ws = new_fc2_ws.detach()
                self._fp8_workspaces[cache_name_fc2] = new_fc2_ws
        finally:
            self.end_forward()

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(fc2_bias, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return (
                    out,
                    cast_if_needed(fc2_bias, self.activation_dtype),
                    ln_out,
                )
            return out, cast_if_needed(fc2_bias, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.layer_norm_bias is not None:
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            set_tensor_model_parallel_attributes(
                tensor=self.fc1_weight,
                is_parallel=True,
                dim=0,
                stride=1,
            )
            set_tensor_model_parallel_attributes(
                tensor=self.fc2_weight,
                is_parallel=True,
                dim=1,
                stride=1,
            )

            if self.use_bias:
                set_tensor_model_parallel_attributes(self.fc1_bias, True, 0, 1)
                if self.set_parallel_mode:
                    setattr(self.fc2_bias, "sequence_parallel", self.sequence_parallel)

    def _get_quantizers(self, fp8_output, is_grad_enabled):
        if not self.fp8 and not self.fp8_calibration:
            return [None] * 12

        fc1_input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
        fc1_input_quantizer.internal = True
        if not (self.set_parallel_mode and self.sequence_parallel):
            fc1_input_quantizer.optimize_for_gemm = True

        fc2_input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM2_INPUT]
        fc2_input_quantizer.internal = True
        if not (self.set_parallel_mode and self.sequence_parallel):
            fc2_input_quantizer.optimize_for_gemm = True

        fc1_weight_quantizer, fc2_weight_quantizer = self._get_weight_quantizers()

        fc1_output_quantizer = None
        fc2_output_quantizer = None
        if fp8_output:
            fc2_output_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM2_OUTPUT]

        fc1_grad_input_quantizer = None
        fc1_grad_weight_quantizer = None
        fc1_grad_output_quantizer = None
        fc2_grad_input_quantizer = None
        fc2_grad_weight_quantizer = None
        fc2_grad_output_quantizer = None

        if is_grad_enabled:
            fc1_grad_output_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
            fc1_grad_output_quantizer.internal = True
            if not (self.set_parallel_mode and self.sequence_parallel):
                fc1_grad_output_quantizer.optimize_for_gemm = True

            fc2_grad_output_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT2]
            fc2_grad_output_quantizer.internal = True
            if not (self.set_parallel_mode and self.sequence_parallel):
                fc2_grad_output_quantizer.optimize_for_gemm = True

        return (
            fc1_input_quantizer,
            fc1_weight_quantizer,
            fc1_output_quantizer,
            fc1_grad_input_quantizer,
            fc1_grad_weight_quantizer,
            fc1_grad_output_quantizer,
            fc2_input_quantizer,
            fc2_weight_quantizer,
            fc2_output_quantizer,
            fc2_grad_input_quantizer,
            fc2_grad_weight_quantizer,
            fc2_grad_output_quantizer,
        )

    def _get_debug_quantizers(self, fp8_output, is_grad_enabled):
        """Return DebugQuantizer wrappers for FC1 and FC2 GEMM groups."""
        from ...debug.pytorch.debug_quantization import DebugQuantizer
        from ...debug.pytorch.debug_state import TEDebugState

        if not TEDebugState.debug_enabled:
            raise RuntimeError("TEDebugState.debug_enabled must be True to get debug quantizers")

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        base_quantizers = tuple(self._get_quantizers(fp8_output, is_grad_enabled))

        def make_debug_quantizers(prefix, offset):
            return tuple(
                DebugQuantizer(
                    f"{self.name}.{prefix}",
                    name,
                    quantizer,
                    self.tp_group,
                    self.tp_size,
                )
                for name, quantizer in zip(names, base_quantizers[offset : offset + len(names)])
            )

        return make_debug_quantizers("fc1", 0) + make_debug_quantizers("fc2", len(names))

    def _get_weight_quantizers(self):
        if not self.fp8 and not self.fp8_calibration:
            return [None, None]
        fc1_weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        fc1_weight_quantizer.internal = True
        fc2_weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM2_WEIGHT]
        fc2_weight_quantizer.internal = True
        return [fc1_weight_quantizer, fc2_weight_quantizer]

    def _get_weight_tensors(self):
        unfused_weights = [self.fc1_weight, self.fc2_weight]
        from ..quantized_tensor import QuantizedTensor

        if any(isinstance(w, QuantizedTensor) for w in unfused_weights):
            if not self.fp8:
                warnings.warn(
                    "You are using quantized weights without quantized compute. "
                    "Please make sure this is intentional."
                )
                unfused_weights = [
                    w.dequantize() if isinstance(w, QuantizedTensor) else w for w in unfused_weights
                ]
        return unfused_weights

    def need_backward_dw(self):
        return self.wgrad_store is not None and self.wgrad_store.delay_wgrad_compute()

    def backward_dw(self):
        if not self.need_backward_dw():
            return
        fc2_wgrad = fc2_bias_grad = fc1_wgrad = fc1_bias_grad = None
        if self.fc2_weight.requires_grad:
            (fc2_wgrad, fc2_bias_grad, *_), _ = self.wgrad_store.pop()
        if self.fc1_weight.requires_grad:
            (fc1_wgrad, fc1_bias_grad, *_), _ = self.wgrad_store.pop()
        if self.use_bias:
            if self.fc2_bias.grad is None and fc2_bias_grad is not None:
                self.fc2_bias.grad = fc2_bias_grad.to(self.fc2_bias.dtype)
            if self.fc1_bias.grad is None and fc1_bias_grad is not None:
                self.fc1_bias.grad = fc1_bias_grad.to(self.fc1_bias.dtype)
        if not self.fuse_wgrad_accumulation:
            if fc2_wgrad is not None:
                self.fc2_weight.grad = fc2_wgrad.to(self.fc2_weight.dtype)
            if fc1_wgrad is not None:
                self.fc1_weight.grad = fc1_wgrad.to(self.fc1_weight.dtype)
        del fc2_wgrad, fc2_bias_grad, fc1_wgrad, fc1_bias_grad

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        base_name = self.name or ""
        fc1_name = f"{base_name}.fc1" if base_name else "fc1"
        fc2_name = f"{base_name}.fc2" if base_name else "fc2"
        # The forward/backward slot layout must match FP8FwdTensorIdx /
        # FP8BwdTensorIdx exactly (see constants.py). LayerNormMLP uses two
        # GEMMs (fc1, fc2), so slots are laid out per GEMM as
        # [input, weight, output] (fwd) and [grad_output, grad_input,
        # grad_weight] (bwd). Some slots are never consumed by the module
        # (e.g. fc1's output, fc1's grad_input passthrough); they are still
        # labelled with their true semantic role so a role-aware qfactory
        # sees an accurate picture instead of a duplicated/mislabelled slot.
        if fwd:
            base = [
                # GEMM1_INPUT
                QuantizerRole(module_type="linear", tensor_type="input", name=fc1_name),
                # GEMM1_WEIGHT
                QuantizerRole(module_type="linear", tensor_type="weight", name=fc1_name),
                # GEMM1_OUTPUT (unused: fc1 output feeds activation, never quantized here)
                QuantizerRole(module_type="linear", tensor_type="output", name=fc1_name),
                # GEMM2_INPUT
                QuantizerRole(module_type="linear", tensor_type="input", name=fc2_name),
                # GEMM2_WEIGHT
                QuantizerRole(module_type="linear", tensor_type="weight", name=fc2_name),
                # GEMM2_OUTPUT (module-level output role; fall back to a
                # concrete role so a qfactory never receives None)
                self._output_quantizer_role
                or QuantizerRole(module_type="linear", tensor_type="output", name=fc2_name),
            ]
        else:
            base = [
                # GRAD_OUTPUT1 (fc1 grad_output)
                QuantizerRole(module_type="linear", tensor_type="grad_output", name=fc1_name),
                # GRAD_INPUT1 (module-level grad_input role; fall back to a
                # concrete role so a qfactory never receives None)
                self._grad_input_quantizer_role
                or QuantizerRole(module_type="linear", tensor_type="grad_input", name=fc1_name),
                # GRAD_WEIGHT1 (unused)
                QuantizerRole(module_type="linear", tensor_type="grad_weight", name=fc1_name),
                # GRAD_OUTPUT2 (fc2 grad_output)
                QuantizerRole(module_type="linear", tensor_type="grad_output", name=fc2_name),
            ]
        return [base[i % len(base)] for i in range(num_quantizers)]
