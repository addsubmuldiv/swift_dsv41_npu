# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import os
import warnings
import weakref
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch_npu

from ..constants import (
    FP8BwdTensorIdx,
    FP8FwdTensorIdx,
    ParallelMode,
    TensorUsage,
    dist_group_type,
)
from ..distributed import (
    DummyHandle,
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
from ..ops.gemm import (
    general_gemm,
    general_gemm_add,
)
from ._common import WeightGradStore, noop_cat
from ..quantization import FP8GlobalStateManager, QuantizerRole
from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
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
from .base import (
    TransformerEngineBaseModule,
    _check_fp8_reduce_and_update,
    quantize_weight,
    setup_dummy_wgrad,
)


@dataclass(slots=True)
class LayerNormLinearArgs:
    is_first_microbatch: Optional[bool]
    fp8: bool
    debug: bool
    eps: float
    weight_workspace: Optional[torch.Tensor]
    input_quantizer: Quantizer
    weight_quantizer: Quantizer
    output_quantizer: Quantizer
    grad_input_quantizer: Quantizer
    grad_weight_quantizer: Quantizer
    grad_output_quantizer: Quantizer
    fused_wgrad_accumulation: bool
    wgrad_store: WeightGradStore
    cpu_offloading: bool
    tp_group: torch.distributed.group
    tp_size: int
    sequence_parallel: bool
    activation_dtype: torch.dtype
    tensor_parallel: bool
    parallel_mode: Optional[str]
    is_grad_enabled: bool
    fp8_output: bool
    cache_weight: bool
    skip_fp8_weight_update: bool
    overlap_ag_fprop: bool
    overlap_rs_dgrad: bool
    overlap_rs_fprop: bool
    overlap_ag_dgrad: bool
    normalization: str
    zero_centered_gamma: bool
    return_layernorm_output: bool
    return_layernorm_output_gathered: bool
    fsdp_group: Optional[Any]
    is_fsdp2: bool
    backward_override: Optional[str] = None

    @property
    def ub_overlap_ag(self):
        return self.overlap_ag_dgrad


def _layernorm_linear_forward_impl(
    inp: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    weight: torch.Tensor,
    bias,
    args: LayerNormLinearArgs,
):
    out_features, in_features = weight.shape
    inp_shape = inp.shape
    assert inp_shape[-1] == in_features, "GEMM not possible"
    inp = inp.view((-1, in_features))
    inputmat = inp

    inputmat = cast_if_needed(inputmat, args.activation_dtype)
    ln_weight = cast_if_needed(ln_weight, args.activation_dtype)
    if ln_bias is not None:
        ln_bias = cast_if_needed(ln_bias, args.activation_dtype)

    tp_world_size = get_distributed_world_size(args.tp_group)
    with_input_all_gather = args.parallel_mode == ParallelMode.COLUMN and args.sequence_parallel
    backward_override = args.backward_override
    # Forward comm+GEMM overlap stays enabled under backward_override (forward
    # is unchanged FP8 compute); only backward overlaps are disabled, mirroring
    # upstream NVTE.
    overlap_ag_fprop = args.overlap_ag_fprop
    overlap_rs_fprop = args.overlap_rs_fprop

    if args.normalization == "LayerNorm":
        gamma = ln_weight
        if args.zero_centered_gamma:
            # aclnnLayerNorm requires matching x/w/b dtypes; promote the whole
            # (1 + gamma) computation to fp32 (ops.LayerNorm recipe) and cast
            # the output back.
            ln_out, mu, rsigma = torch.native_layer_norm(
                inputmat.float(),
                (inputmat.shape[-1],),
                gamma.float() + 1.0,
                ln_bias.float() if ln_bias is not None else None,
                args.eps,
            )
            ln_out = ln_out.to(inputmat.dtype)
        elif inputmat.dtype != torch.float32:
            # fp32 regime for the bf16 path too: the kernel would round the
            # saved statistics to bf16 (NVTE keeps fp32), and a bf16 rstd
            # injects a ~0.4% correlated error into dgamma/dbeta (REPORT atk
            # 3#16). The bf16 output is unchanged.
            ln_out, mu, rsigma = torch.native_layer_norm(
                inputmat.float(),
                (inputmat.shape[-1],),
                gamma.float(),
                ln_bias.float() if ln_bias is not None else None,
                args.eps,
            )
            ln_out = ln_out.to(inputmat.dtype)
        else:
            ln_out, mu, rsigma = torch.native_layer_norm(
                inputmat,
                (inputmat.shape[-1],),
                gamma,
                ln_bias,
                args.eps,
            )
    elif args.normalization == "RMSNorm":
        gamma = ln_weight
        if args.zero_centered_gamma:
            gamma = gamma.float() + 1.0
        ln_out, rrsigma = torch_npu.npu_rms_norm(inputmat, gamma, epsilon=args.eps)
        mu = None
        rsigma = rrsigma
    else:
        raise ValueError(f"Unsupported normalization type: {args.normalization}")

    ln_out_return = None
    if args.return_layernorm_output or args.return_layernorm_output_gathered:
        ln_out_return = ln_out

    # NVTE returns the gathered normalization output in high precision. If it is
    # requested, gather before quantization and reuse that tensor as the GEMM input
    # instead of performing a second, quantized all-gather in the overlap path.
    input_already_gathered = False
    if with_input_all_gather and args.return_layernorm_output_gathered:
        ln_out, _ = gather_along_dim(ln_out, args.tp_group)
        ln_out_return = ln_out
        input_already_gathered = True
        overlap_ag_fprop = False

    backward_needs_input = args.is_grad_enabled and weight.requires_grad
    new_weight_workspace = None

    is_quantized = args.fp8 or args.debug

    # region Input/weight quantization vs. raw passthrough
    if is_quantized:
        if args.input_quantizer is None:
            raise ValueError("Missing quantizer for input tensor")

        if with_input_all_gather and not overlap_ag_fprop and not input_already_gathered:
            ln_out, _ = gather_along_dim(ln_out, args.tp_group)

        args.input_quantizer.set_usage(
            rowwise=True,
            columnwise=backward_needs_input and backward_override is None,
        )
        mm_inp = args.input_quantizer(ln_out)

        weight_quantizer = args.weight_quantizer
        if weight_quantizer is not None and not weight_quantizer.can_reuse(weight):
            # FIX-0825: backward dgrad gemm unconditionally needs columnwise weight
            # data (e.g. MoE dispatcher output has inp.requires_grad=False but
            # weight wgrad/dgrad still run). Quantize columnwise whenever backward
            # may run for this module.
            columnwise_usage = (
                args.is_grad_enabled
                and (inp.requires_grad or weight.requires_grad)
                and not args.is_fsdp2
                and backward_override is None
            )
            if backward_override is not None:
                # Under a backward override the backward dgrad uses dequantized /
                # dense weights, so columnwise data is never needed. Short-circuit
                # here so the recompute fall-through below cannot resurrect it.
                columnwise_usage = False
            elif not columnwise_usage:
                columnwise_usage = (
                    is_fp8_activation_recompute_enabled()
                    and not in_fp8_activation_recompute_phase()
                )
            weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        elif isinstance(weight, QuantizedTensor):
            weight_quantizer = weight._quantizer

        update_ws = args.is_first_microbatch is None or args.is_first_microbatch
        weightmat, new_weight_workspace = quantize_weight(
            tensor=weight,
            quantizer=weight_quantizer,
            workspace=args.weight_workspace,
            update_workspace=update_ws,
            skip_update_flag=args.skip_fp8_weight_update,
            fsdp_group=args.fsdp_group,
            workspace_dtype=args.activation_dtype,
            cache=args.cache_weight,
        )
        weightmat.update_usage(rowwise_usage=True)
    else:
        mm_inp = ln_out
        weightmat = weight
        if with_input_all_gather and not overlap_ag_fprop and not input_already_gathered:
            mm_inp, _ = gather_along_dim(mm_inp, args.tp_group)
    # endregion

    mm_kwargs = {
        "usage_a": TensorUsage.LHS,
        "usage_b": TensorUsage.RHS_TRANS,
        "out_dtype": args.activation_dtype,
    }

    if overlap_ag_fprop:
        out, gather_out = CommOverlapOps.allgather_matmul(
            mm_inp,
            weightmat,
            bias,
            tp_world_size,
            args.tp_group,
            return_gathered=backward_override is None,
            **mm_kwargs,
        )
        if backward_override is None:
            # Standard FP8 backward consumes the gathered quantized input.
            mm_inp = gather_out
        # Under a backward override the operand is recomputed from the saved
        # LOCAL tensor (mirroring upstream NVTE), so neither an extra gathered
        # copy nor its collectives are needed.
    elif overlap_rs_fprop:
        out = CommOverlapOps.matmul_reduce_scatter(
            mm_inp,
            weightmat,
            bias,
            tp_world_size,
            args.tp_group,
            **mm_kwargs,
        )
    else:
        out = general_gemm(
            mm_inp,
            weightmat,
            bias=bias,
            quantization_params=args.output_quantizer if args.debug else None,
            **mm_kwargs,
        )

    if args.parallel_mode == ParallelMode.ROW and not overlap_rs_fprop and args.tp_size > 1:
        if args.sequence_parallel:
            out, _ = CommOverlapOps.reduce_scatter(
                out,
                args.output_quantizer,
                tp_world_size,
                args.tp_group,
                use_quant=args.fp8_output and FP8GlobalStateManager.get_fp8_recipe().mxfp8(),
            )
        else:
            torch.distributed.all_reduce(out, group=args.tp_group)

    out = out.view(-1, *inp_shape[1:-1], out_features)

    # region Prepare backward state
    tensors_to_save = None
    tensor_objects = None
    ctx_attrs = None
    if args.is_grad_enabled:
        is_weight_param_quantized = is_quantized and isinstance(weight, QuantizedTensorStorage)
        ln_out_to_save = mm_inp
        if backward_override == "high_precision":
            ln_out_to_save = ln_out
        # weightmat is None for unquantized compute; quantized paths save the quantized weight
        # (or None for fsdp2 re-quantization case)
        wt_save = weightmat if is_quantized else None
        if args.is_fsdp2 and wt_save is not weight:
            wt_save = None

        fsdp_shapes = _fsdp_scatter_tensors(
            args.fsdp_group,
            mu,
            rsigma,
            weightmat if is_quantized and not is_weight_param_quantized else None,
            ln_out_to_save if weight.requires_grad else None,
        )

        tensors_to_save, tensor_objects = prepare_for_saving(
            inputmat,
            ln_out_to_save,
            wt_save,
            weight,
            bias,
            ln_weight,
            mu,
            rsigma,
        )

        ctx_attrs = {
            "inp_shape": inp_shape,
            "normalization": args.normalization,
            "zero_centered_gamma": args.zero_centered_gamma,
            "return_layernorm_output": args.return_layernorm_output,
            "return_layernorm_output_gathered": args.return_layernorm_output_gathered,
            "with_input_all_gather": with_input_all_gather,
            "fsdp_shapes": fsdp_shapes,
        }
        if is_quantized:
            ctx_attrs["is_weight_param_quantized"] = is_weight_param_quantized
    # endregion Prepare backward state

    return (
        out,
        new_weight_workspace,
        tensors_to_save,
        tensor_objects,
        ctx_attrs,
        ln_out_return,
        with_input_all_gather,
        inp_shape,
        mm_inp,
    )


def _layernorm_linear_setup_ctx(
    ctx,
    tensors_to_save,
    tensor_objects,
    ctx_attrs,
    inp,
    ln_weight,
    ln_bias,
    weight,
    bias,
    args,
):
    ctx.save_for_backward(*tensors_to_save)
    ctx.tensor_objects = tensor_objects
    ctx.args = args
    ctx.use_bias = bias is not None
    # Whether the LayerNorm branch had a bias (ln_bias is not saved; the
    # backward computes dbeta from the dgrad column sum when this is set).
    ctx.ln_use_bias = ln_bias is not None
    ctx.inp_shape = ctx_attrs.get("inp_shape", inp.shape)
    ctx.fp8 = args.fp8
    ctx.backward_override = args.backward_override
    ctx.debug = args.debug
    ctx.sequence_parallel = args.sequence_parallel
    ctx.ub_overlap_ag = args.ub_overlap_ag

    ctx.requires_dgrad = inp.requires_grad
    ctx.requires_wgrad = weight.requires_grad

    ctx.normalization = ctx_attrs.get("normalization")
    ctx.zero_centered_gamma = ctx_attrs.get("zero_centered_gamma")
    ctx.return_layernorm_output = ctx_attrs.get("return_layernorm_output")
    ctx.return_layernorm_output_gathered = ctx_attrs.get("return_layernorm_output_gathered")
    ctx.with_input_all_gather = ctx_attrs.get("with_input_all_gather", False)

    ctx.fsdp_group = args.fsdp_group
    ctx.is_fsdp2 = args.is_fsdp2

    ctx.fsdp_shapes = ctx_attrs.get("fsdp_shapes")

    if args.fp8 or args.debug:
        ctx.is_weight_param_quantized = ctx_attrs.get("is_weight_param_quantized", False)
    if args.fp8:
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
    else:
        ctx.fp8_recipe = None

    if (
        args.fp8
        and args.backward_override is None
        and requires_grad(inp, ln_weight, ln_bias, weight, bias)
    ):
        ctx.reduce_and_update_bwd_fp8_tensors = _check_fp8_reduce_and_update()
    else:
        ctx.reduce_and_update_bwd_fp8_tensors = False

    ctx.fuse_wgrad_accumulation = args.fused_wgrad_accumulation and weight.requires_grad
    ctx.origin_weight_overwrites_main_grad = False
    if ctx.fuse_wgrad_accumulation:
        ctx.origin_weight_ref = weakref.ref(weight)
        ctx.origin_weight_overwrites_main_grad = getattr(weight, "overwrite_main_grad", False)
        if hasattr(weight, "__fsdp_param__"):
            ctx.main_grad_func = weight.get_main_grad
        else:
            ctx.main_grad_func = lambda: getattr(weight, "main_grad", None)
    if args.backward_override is not None:
        ctx.fp8 = False
        ctx.debug = False
        # Aligned with upstream NVTE: an override backward recomputes every
        # operand in plain precision, so none of the comm+GEMM fused ops
        # apply. Dgrad falls back to a plain GEMM followed by manual
        # collectives, and wgrad gathers the sharded input itself.
        args.overlap_ag_dgrad = False
        args.overlap_rs_dgrad = False
        ctx.ub_overlap_ag = False
        ctx.grad_input_quantizer = None
        ctx.grad_weight_quantizer = None
        ctx.grad_output_quantizer = None


def _layernorm_linear_backward(
    ctx,
    args: LayerNormLinearArgs,
    grad_output,
    ln_grad_out=None,
):
    is_fp8 = getattr(ctx, "fp8", False)
    is_quantized = is_fp8 or ctx.debug
    (
        inputmat,
        mm_inp,
        weightmat,
        weight,
        bias,
        ln_weight,
        mu,
        rsigma,
    ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

    ctx.tensor_objects = None

    # saved_tensors_hooks may restore the weight as a detached Tensor without
    # Parameter-only attributes used by MCore DDP.  Keep the restored tensor
    # for GEMM inputs, but use the original Parameter for gradient state and
    # dummy-wgrad bookkeeping.
    origin_weight = weight
    if ctx.fuse_wgrad_accumulation and ctx.requires_wgrad:
        origin_weight_ref = ctx.origin_weight_ref
        ctx.origin_weight_ref = None
        origin_weight = origin_weight_ref() if origin_weight_ref is not None else None
        assert origin_weight is not None, "weight was removed while fuse_wgrad_accumulation=True"

    fsdp_group = getattr(ctx, "fsdp_group", None)
    if fsdp_group is not None:
        _fsdp_gather_tensors(
            fsdp_group,
            ctx.fsdp_shapes,
            mu,
            rsigma,
            weightmat
            if is_quantized and not getattr(ctx, "is_weight_param_quantized", False)
            else None,
            mm_inp if ctx.requires_wgrad else None,
        )

    tp_world_size = get_distributed_world_size(args.tp_group)
    wgrad = None

    if args.grad_output_quantizer is not None and ctx.backward_override is None:
        # dgrad is always required for the normalization backward, even when
        # the original input does not require gradients. Reset both usages on
        # every invocation so a frozen/no-dgrad pass cannot poison the next one.
        args.grad_output_quantizer.set_usage(
            rowwise=True,
            columnwise=ctx.requires_wgrad,
        )

    mm_grad, grad_bias = TransformerEngineBaseModule.grad_output_preprocess(
        ctx,
        grad_output,
        args.parallel_mode == ParallelMode.ROW,
        args.grad_output_quantizer,
    )

    handle = DummyHandle

    is_fsdp2 = getattr(ctx, "is_fsdp2", False)
    if weightmat is None and is_fsdp2 and is_quantized:
        if isinstance(weight, QuantizedTensorStorage):
            weightmat = weight
        else:
            args.weight_quantizer.set_usage(rowwise=True, columnwise=True)
            weightmat = args.weight_quantizer(weight)

    dgrad_weight = weightmat if weightmat is not None else weight
    if ctx.backward_override == "dequantized":
        if isinstance(dgrad_weight, QuantizedTensorStorage):
            dgrad_weight = dgrad_weight.dequantize(dtype=args.activation_dtype)
        else:
            dgrad_weight = cast_if_needed(dgrad_weight, args.activation_dtype)
    elif ctx.backward_override == "high_precision":
        dgrad_weight = weight
        if isinstance(dgrad_weight, QuantizedTensorStorage):
            dgrad_weight = dgrad_weight.dequantize(dtype=args.activation_dtype)
        else:
            dgrad_weight = cast_if_needed(dgrad_weight, args.activation_dtype)

    dgrad_kwargs = {
        "usage_a": TensorUsage.LHS,
        "usage_b": TensorUsage.RHS,
        "out_dtype": args.activation_dtype,
    }
    if args.overlap_ag_dgrad:
        dgrad, mm_grad = CommOverlapOps.allgather_matmul(
            mm_grad,
            dgrad_weight,
            None,
            tp_world_size,
            args.tp_group,
            **dgrad_kwargs,
        )
    elif args.overlap_rs_dgrad:
        dgrad = CommOverlapOps.matmul_reduce_scatter(
            mm_grad,
            dgrad_weight,
            None,
            tp_world_size,
            args.tp_group,
            **dgrad_kwargs,
        )
    else:
        dgrad = general_gemm(
            mm_grad,
            dgrad_weight,
            quantization_params=args.grad_input_quantizer if args.debug else None,
            **dgrad_kwargs,
        )
        if tp_world_size > 1 and args.parallel_mode == ParallelMode.COLUMN:
            if args.sequence_parallel:
                dgrad, handle = reduce_scatter_along_dim(
                    dgrad,
                    args.tp_group,
                    async_op=True,
                )
            else:
                handle = torch.distributed.all_reduce(
                    dgrad,
                    group=args.tp_group,
                    async_op=True,
                )

    if is_fsdp2 and is_quantized and isinstance(weightmat, QuantizedTensorStorage):
        clear_columnwise_cache(weightmat)

    if ctx.requires_wgrad:
        main_grad = ctx.main_grad_func() if ctx.fuse_wgrad_accumulation else None
        if main_grad is not None:
            origin_weight.main_grad = main_grad
        use_fuse_wgrad_accumulation = (
            ctx.fuse_wgrad_accumulation
            and main_grad is not None
            and (
                (is_fp8 and ctx.fp8_recipe.mxfp8())
                or (not is_fp8 and main_grad.dtype == torch.float32)
            )
        )
        # MCore FSDP (and similar sharded wrappers) marks params with
        # overwrite_main_grad=True: their main_grad buffer is managed
        # externally and must be overwritten per micro-batch, never
        # accumulated. Regular DDP params accumulate across micro-batches.
        accumulate_wgrad_into_param_main_grad = (
            use_fuse_wgrad_accumulation
            and (args.is_first_microbatch is None or not args.is_first_microbatch)
            and not ctx.origin_weight_overwrites_main_grad
        )

        out_dtype = main_grad.dtype if use_fuse_wgrad_accumulation else args.activation_dtype

        if ctx.backward_override == "dequantized" and isinstance(mm_inp, QuantizedTensorStorage):
            mm_inp = mm_inp.dequantize(dtype=args.activation_dtype)
        elif ctx.backward_override == "high_precision":
            mm_inp = cast_if_needed(mm_inp, args.activation_dtype)

        if (
            ctx.backward_override is not None
            and args.parallel_mode == ParallelMode.COLUMN
            and args.sequence_parallel
            and tp_world_size > 1
        ):
            # Forward saved the local norm output (the fused AG fprop only
            # feeds the forward GEMM, mirroring upstream NVTE's
            # ln_out_needs_gather), so rebuild the full-sequence wgrad operand
            # here in plain precision.
            mm_inp, _ = gather_along_dim(mm_inp, args.tp_group)

        if not is_quantized:
            mm_inp = mm_inp.view(-1, mm_inp.shape[-1])
            mm_grad = mm_grad.view(-1, mm_grad.shape[-1])

        wgrad_kwargs = {
            "usage_a": TensorUsage.LHS_TRANS,
            "usage_b": TensorUsage.RHS,
            "out_dtype": out_dtype,
        }
        delay_wgrad_compute = (
            args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute()
        )

        def wgrad_gemm(x, dy):
            if use_fuse_wgrad_accumulation:
                if not accumulate_wgrad_into_param_main_grad:
                    main_grad.zero_()
                general_gemm_add(
                    main_grad,
                    dy,
                    x,
                    quantization_params=args.grad_weight_quantizer if args.debug else None,
                    **wgrad_kwargs,
                )
                # Delayed wgrad is consumed outside autograd, so it does not
                # need the dummy placeholder required by the immediate path.
                _wgrad = (
                    None if delay_wgrad_compute else setup_dummy_wgrad(origin_weight, main_grad)
                )
            else:
                _wgrad = general_gemm(
                    dy,
                    x,
                    quantization_params=args.grad_weight_quantizer if args.debug else None,
                    **wgrad_kwargs,
                )
            return _wgrad, None

        if delay_wgrad_compute:
            args.wgrad_store.put([mm_inp, mm_grad], wgrad_gemm)
        else:
            wgrad, _ = wgrad_gemm(mm_inp, mm_grad)

    if is_fp8 and ctx.reduce_and_update_bwd_fp8_tensors:
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

    if ctx.use_bias and grad_bias is None:
        grad_bias = torch.reshape(grad_output, (-1, grad_output.shape[-1])).sum(dim=0)

    if handle is not None:
        handle.wait()

    if dgrad is not None:
        dgrad = dgrad.view(ctx.inp_shape)

    if ctx.return_layernorm_output and ln_grad_out is not None:
        residual_grad = ln_grad_out
        if ctx.return_layernorm_output_gathered and ctx.with_input_all_gather:
            residual_grad, _ = reduce_scatter_along_dim(residual_grad, args.tp_group)
        residual_grad = (
            residual_grad.view_as(dgrad) if dgrad is not None else residual_grad.view(ctx.inp_shape)
        )
        dgrad = residual_grad if dgrad is None else dgrad + residual_grad

    dgamma = None
    dbeta = None
    if ctx.normalization == "LayerNorm":
        dgrad_2d = dgrad.view(-1, dgrad.shape[-1])
        inputmat_2d = inputmat.view(-1, ctx.inp_shape[-1])  # pylint: disable=no-member
        inner_dim = inputmat_2d.shape[-1]
        dtype = dgrad_2d.dtype
        if rsigma is not None and rsigma.dtype == torch.float32 and dtype != torch.float32:
            # Forward's fp32-stats regime (zc path and the bf16 full-precision
            # stats path): the saved mu/rsigma are fp32; run the backward math
            # at fp32 too (the saved bf16 inputmat is promoted losslessly
            # here) and cast the result back.
            dy = dgrad_2d.float()
            x = inputmat_2d.float()
            weight = ln_weight.float()
        else:
            dy = dgrad_2d
            x = inputmat_2d
            weight = ln_weight.to(inputmat_2d.dtype)
        if ctx.zero_centered_gamma:
            weight = weight.float() + 1.0
        # ln_bias is not saved; aten rejects bias=None with output_mask[2]=True,
        # so dbeta (a plain column sum of dgrad, fp32-accumulated by torch.sum)
        # is computed manually when the LayerNorm bias exists.
        dx, dweight, _ = torch.ops.aten.native_layer_norm_backward(
            dy, x, (inner_dim,), mu, rsigma, weight, None, [True, True, False]
        )
        dgrad = dx.to(dtype)
        dgamma = dweight.to(ln_weight.dtype)
        if ctx.ln_use_bias:
            dbeta = dgrad_2d.sum(dim=0)

    elif ctx.normalization == "RMSNorm":
        dgrad_2d = dgrad.view(-1, dgrad.shape[-1])
        inputmat_2d = inputmat.view(-1, ctx.inp_shape[-1])  # pylint: disable=no-member
        weight = ln_weight.to(inputmat_2d.dtype)
        if ctx.zero_centered_gamma:
            weight = weight.float() + 1.0

        if rsigma is not None:
            rrms = rsigma
        else:
            variance = inputmat_2d.pow(2).mean(dim=-1, keepdim=True)
            rrms = torch.rsqrt(variance + args.eps)

        dgrad, dweight = torch_npu.npu_rms_norm_backward(dgrad_2d, inputmat_2d, weight, rrms)
        dgamma = dweight.to(ln_weight.dtype)
        dbeta = None

    if not ctx.use_bias:
        grad_bias = None

    if is_fp8 and not isinstance(weight, QuantizedTensorStorage):
        _fsdp_scatter_tensors(fsdp_group, weightmat)

    return (
        dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
        dgamma,
        dbeta,
        wgrad,
        grad_bias,
    )


class _LayerNormLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        bias: torch.Tensor,
        args: LayerNormLinearArgs,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        (
            out,
            new_weight_workspace,
            tensors_to_save,
            tensor_objects,
            ctx_attrs,
            ln_out_return,
            with_input_all_gather,
            inp_shape,
            mm_inp,
        ) = _layernorm_linear_forward_impl(inp, ln_weight, ln_bias, weight, bias, args)

        if ctx is not None:
            _layernorm_linear_setup_ctx(
                ctx,
                tensors_to_save,
                tensor_objects,
                ctx_attrs,
                inp,
                ln_weight,
                ln_bias,
                weight,
                bias,
                args,
            )
        ln_out_for_return = None
        if args.return_layernorm_output:
            if args.return_layernorm_output_gathered:
                shape = list(inp_shape)
                shape[0] *= args.tp_size if with_input_all_gather else 1
                ln_out_for_return = ln_out_return.view(shape)
            else:
                ln_out_for_return = ln_out_return.view(inp_shape)
        return out, ln_out_for_return, new_weight_workspace

    @staticmethod
    def backward(ctx, grad_output, ln_grad_out, _grad_weight_workspace):
        args: LayerNormLinearArgs = ctx.args
        result = _layernorm_linear_backward(ctx, args, grad_output, ln_grad_out)
        return result + (None,)  # args


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int
) -> None:
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


class LayerNormLinear(TransformerEngineBaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "npu",
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = self.use_bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = (
            return_layernorm_output_gathered if return_layernorm_output else False
        )
        self.zero_centered_gamma = zero_centered_gamma
        self.symmetric_ar_type = symmetric_ar_type

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in [None, "column", "row"], (
            f"parallel_mode {parallel_mode} not supported"
        )

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        self.overlap_ag_fprop = (
            self.parallel_mode == "column" and self.sequence_parallel and ub_overlap_ag
        )
        self.overlap_rs_dgrad = (
            self.parallel_mode == "column" and self.sequence_parallel and ub_overlap_rs_dgrad
        )
        self.overlap_rs_fprop = (
            self.parallel_mode == "row" and self.sequence_parallel and ub_overlap_rs
        )
        self.overlap_ag_dgrad = (
            self.parallel_mode == "row" and self.sequence_parallel and ub_overlap_ag
        )

        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        assert parameters_split is None

        self.eps = eps
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(self.in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not self.zero_centered_gamma)),
        )
        if self.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(self.in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias, init_fn=init_method_constant(0.0)
            )
        else:
            self.layer_norm_bias = None

        weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )

        self.weight_names = ["weight"]
        self.bias_names = ["bias"]
        self.parameter_split_sizes = [out_features]

        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError(
                    "Splitting QuantizedTensor into multiple params is not supported"
                )

            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=FP8FwdTensorIdx.GEMM1_WEIGHT,
            )

        if self.use_bias:
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                self.register_parameter(
                    self.bias_names[i],
                    torch.nn.Parameter(bias_tensor[split_start:split_end]),
                    init_fn=init_method_constant(0.0),
                )
        else:
            for _name in self.bias_names:
                b = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, _name, b)

        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=device == "meta")

        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ub_bulk_wgrad)
        if self.wgrad_store.delay_wgrad_compute():
            for pname, param in self.named_parameters():
                if pname in self.weight_names or pname in self.bias_names:
                    param.skip_backward_post_hook = True

        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            if self.use_bias:
                for bias_name in self.bias_names:
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, bias_name),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, bias_name), True, 0, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
        fp8_grad: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        is_grad_enabled = torch.is_grad_enabled()

        inp = self.prepare_forward(inp, allow_non_contiguous=False)

        skip_fp8_weight_update = None

        weight_tensor = getattr(self, self.weight_names[0])
        bias_tensor = getattr(self, self.bias_names[0]) if self.use_bias else None

        debug = self.is_debug_iter()
        backward_override = (
            FP8GlobalStateManager.get_fp8_recipe().backward_override if self.fp8 else None
        )
        quantizers = (
            self._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)
            if not debug
            else self._get_debug_quantizers(fp8_output, fp8_grad, is_grad_enabled)
        )
        if debug and self.no_debug_features_active(quantizers):
            debug = False
            quantizers = self._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)
        (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
        ) = quantizers

        if is_grad_enabled:
            fwd_fn = _LayerNormLinear.apply
            autograd_ctx = []
        else:
            fwd_fn = _LayerNormLinear.forward
            autograd_ctx = [None]

        cache_name = None if (is_first_microbatch is None or self.is_fsdp2) else "weight"
        weight_workspace = self._fp8_workspaces.get(cache_name) if cache_name is not None else None

        non_tensor_args = LayerNormLinearArgs(
            is_first_microbatch=is_first_microbatch,
            fp8=self.fp8,
            debug=debug,
            eps=self.eps,
            weight_workspace=weight_workspace,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
            grad_input_quantizer=grad_input_quantizer,
            grad_weight_quantizer=grad_weight_quantizer,
            grad_output_quantizer=grad_output_quantizer,
            fused_wgrad_accumulation=self.fuse_wgrad_accumulation,
            wgrad_store=self.wgrad_store,
            cpu_offloading=False,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
            sequence_parallel=self.sequence_parallel,
            activation_dtype=self.activation_dtype,
            tensor_parallel=self.tp_size > 1,
            parallel_mode=self.parallel_mode,
            is_grad_enabled=is_grad_enabled,
            fp8_output=fp8_output,
            cache_weight=cache_name is not None,
            skip_fp8_weight_update=skip_fp8_weight_update,
            overlap_ag_fprop=self.overlap_ag_fprop and not debug,
            overlap_rs_dgrad=self.overlap_rs_dgrad and not debug,
            overlap_rs_fprop=self.overlap_rs_fprop and not debug,
            overlap_ag_dgrad=self.overlap_ag_dgrad and not debug,
            normalization=self.normalization,
            zero_centered_gamma=self.zero_centered_gamma,
            return_layernorm_output=self.return_layernorm_output,
            return_layernorm_output_gathered=self.return_layernorm_output_gathered,
            fsdp_group=self.fsdp_group,
            is_fsdp2=self.is_fsdp2,
            backward_override=backward_override,
        )

        try:
            out, ln_out, new_weight_workspace = fwd_fn(
                *autograd_ctx,
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                weight_tensor,
                bias_tensor if self.apply_bias and not self.gemm_bias_unfused_add else None,
                non_tensor_args,
            )

            if new_weight_workspace is not None and cache_name is not None:
                if isinstance(new_weight_workspace, torch.Tensor):
                    new_weight_workspace = new_weight_workspace.detach()
                self._fp8_workspaces[cache_name] = new_weight_workspace

        finally:
            self.end_forward()

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out

    def _get_quantizers(self, fp8_output, fp8_grad, is_grad_enabled):
        if not self.fp8:
            return [None] * 6
        grad_input_quantizer = None
        grad_weight_quantizer = None
        grad_output_quantizer = None
        output_quantizer = None
        input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
        input_quantizer.internal = True
        if not (self.parallel_mode == "column" and self.sequence_parallel):
            input_quantizer.optimize_for_gemm = True
        (weight_quantizer,) = self._get_weight_quantizers()
        if fp8_output:
            output_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_OUTPUT]
        if is_grad_enabled:
            grad_output_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
            grad_output_quantizer.internal = True
            if not (self.parallel_mode == "row" and self.sequence_parallel):
                grad_output_quantizer.optimize_for_gemm = True
            if fp8_grad:
                grad_input_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_INPUT1]
        return (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
        )

    def _get_weight_quantizers(self):
        if not self.fp8:
            return [None]
        weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        return [weight_quantizer]

    def _get_debug_quantizers(self, fp8_output, fp8_grad, is_grad_enabled):
        original_quantizers = self._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)
        from ...debug.pytorch.debug_quantization import DebugQuantizer
        from ...debug.pytorch.debug_state import TEDebugState

        assert TEDebugState.debug_enabled

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        return tuple(
            DebugQuantizer(self.name, name, q, self.tp_group, self.tp_size)
            for name, q in zip(names, original_quantizers)
        )

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        unfused_weights = [getattr(self, name) for name in self.weight_names]
        if any(isinstance(w, QuantizedTensor) for w in unfused_weights):
            if self.fp8:
                if len(unfused_weights) != 1:
                    raise RuntimeError(
                        "Splitting QuantizedTensor into multiple params is not supported"
                    )
            else:
                warnings.warn(
                    "You are using quantized weights without quantized compute. "
                    "Please make sure this is intentional."
                )
                unfused_weights = [w.dequantize() for w in unfused_weights]
        return unfused_weights

    def backward_dw(self):
        """Execute delayed wgrad and preserve non-fused fallback gradients."""
        if not self.need_backward_dw():
            return

        result, _ = self.wgrad_store.pop()
        wgrad, bgrad = result
        if wgrad is not None:
            weight_tensor = noop_cat(self._get_weight_tensors())
            weight_tensor.grad = wgrad.to(weight_tensor.dtype)
        if self.use_bias and bgrad is not None:
            bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
            if bias_tensor.grad is None:
                bias_tensor.grad = bgrad.to(bias_tensor.dtype)
        self._trigger_wgrad_accumulation_and_reduce_hooks()

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        name = self.name or ""
        if fwd:
            base = [
                QuantizerRole(module_type="linear", tensor_type="input", name=name),
                QuantizerRole(module_type="linear", tensor_type="weight", name=name),
                # Module-level output role; fall back to a concrete role so a
                # qfactory never receives None (its documented contract is to
                # always be handed a QuantizerRole).
                self._output_quantizer_role
                or QuantizerRole(module_type="linear", tensor_type="output", name=name),
            ]
        else:
            base = [
                QuantizerRole(module_type="linear", tensor_type="grad_output", name=name),
                # Module-level grad_input role; fall back so qfactory sees a
                # concrete role instead of None.
                self._grad_input_quantizer_role
                or QuantizerRole(module_type="linear", tensor_type="grad_input", name=name),
            ]
        return [base[i % len(base)] for i in range(num_quantizers)]


__all__ = ["LayerNormLinear"]
