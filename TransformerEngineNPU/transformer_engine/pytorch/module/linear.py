# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import warnings
import weakref
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch.nn

from ._common import WeightGradStore, noop_cat
from .base import (
    TransformerEngineBaseModule,
    _check_fp8_reduce_and_update,
    quantize_weight,
    setup_dummy_wgrad,
)
from ..constants import (
    FP8BwdTensorIdx,
    FP8FwdTensorIdx,
    GemmParallelModes,
    ParallelMode,
    TensorUsage,
    dist_group_type,
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
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from ..utils import (
    cast_if_needed,
    divide,
    init_method_constant,
    requires_grad,
)
from ..tensor.utils import clear_columnwise_cache
from ..tensor.float8_tensor import Float8Quantizer
from ...debug.pytorch.debug_state import TEDebugState


@dataclass(slots=True)
class LinearArgs:
    is_first_microbatch: bool
    fp8: bool
    debug: bool

    # --- Non-differentiable cached tensors ---
    weight_workspace: Optional[torch.Tensor]

    wgrad_store: WeightGradStore
    input_quantizer: Quantizer
    weight_quantizer: Quantizer
    output_quantizer: Quantizer
    grad_input_quantizer: Quantizer
    grad_weight_quantizer: Quantizer
    grad_output_quantizer: Quantizer
    fuse_wgrad_accumulation: bool
    cpu_offloading: bool
    tp_group: torch.distributed.group
    tp_size: int
    sequence_parallel: bool
    activation_dtype: torch.dtype
    tensor_parallel: bool
    parallel_mode: Literal["column", "row"]
    is_grad_enabled: bool
    fp8_output: bool  # fout use fp8
    cache_weight: bool
    skip_fp8_weight_update: bool
    save_original_input: bool
    overlap_ag_fprop: bool
    overlap_rs_dgrad: bool
    overlap_rs_fprop: bool
    overlap_ag_dgrad: bool
    fsdp_group: Optional[Any]
    is_fsdp2: bool
    backward_override: Optional[str] = None

    @property
    def ub_overlap_ag(self):
        return self.overlap_ag_dgrad


def _linear_forward_impl(
    weight: torch.Tensor,
    inp: torch.Tensor,
    bias,
    args: LinearArgs,
):
    tp_world_size = get_distributed_world_size(args.tp_group)
    backward_needs_input = args.is_grad_enabled and weight.requires_grad
    backward_override = args.backward_override
    save_original_input = args.save_original_input
    if backward_override == "high_precision":
        save_original_input = True
    elif backward_override == "dequantized":
        save_original_input = False
    inputmat, weightmat = inp, weight
    weight_quantizer = args.weight_quantizer

    out_use_quant = False
    new_weight_workspace = None
    own_quantized_input = False
    if (
        args.parallel_mode == ParallelMode.COLUMN
        and args.sequence_parallel
        and not args.overlap_ag_fprop
    ):
        inputmat, _ = gather_along_dim(inputmat, args.tp_group)
    original_inputmat = inputmat if save_original_input else None

    if args.fp8:
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        out_use_quant = args.fp8_output and recipe.mxfp8()
        if save_original_input and isinstance(args.input_quantizer, Float8Quantizer):
            warnings.warn(
                "DelayedScaling recipe is not supported with save_original_input, "
                "disabling save_original_input."
            )
            save_original_input = False

    # region forward Quantizer
    if args.fp8 or args.debug:
        if isinstance(inputmat, QuantizedTensorStorage):
            inputmat.update_usage(rowwise_usage=True)
        else:
            if args.input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            args.input_quantizer.set_usage(
                rowwise=True,
                columnwise=(
                    backward_needs_input and not save_original_input and backward_override is None
                ),
            )
            inputmat = args.input_quantizer(inputmat)
            own_quantized_input = not save_original_input

        if weight_quantizer is not None and (not weight_quantizer.can_reuse(weight) or args.debug):
            # FIX-0825: backward dgrad gemm needs columnwise weight data even
            # when inp.requires_grad is False.
            columnwise_usage = (
                args.is_grad_enabled
                and (inp.requires_grad or weight.requires_grad)
                and not args.is_fsdp2
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

        # Get quantized weight
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
        inputmat = cast_if_needed(inputmat, args.activation_dtype)  # Cast for AMP
        weightmat = cast_if_needed(weightmat, args.activation_dtype)  # Cast for AMP
    # endregion forward Quantizer

    # region forward mm
    if args.overlap_ag_fprop:
        out, gather_out = CommOverlapOps.allgather_matmul(
            inputmat,
            weightmat,
            bias,
            tp_world_size,
            args.tp_group,
            return_gathered=backward_override is None and not save_original_input,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS_TRANS,
            out_dtype=args.activation_dtype,
        )
        if gather_out is not None:
            inputmat = gather_out
    elif args.overlap_rs_fprop:
        out = CommOverlapOps.matmul_reduce_scatter(
            inputmat,
            weightmat,
            bias,
            tp_world_size,
            args.tp_group,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS_TRANS,
            out_dtype=args.activation_dtype,
        )
    elif (
        not args.fp8
        and not args.debug
        and bias is None
        and type(inputmat) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
        and type(weightmat) in (torch.Tensor, torch.nn.Parameter)
    ):
        # Same operands, transpose and matmul as dense general_gemm. Leave
        # communication, casting and backward-state preparation in place.
        out = torch.matmul(inputmat, weightmat.t())
    else:
        out = general_gemm(
            inputmat,
            weightmat,
            bias=bias,
            quantization_params=args.output_quantizer if args.debug else None,
            usage_a=TensorUsage.LHS,
            usage_b=TensorUsage.RHS_TRANS,
            out_dtype=args.activation_dtype,
        )
    # endregion forward mm

    # region Handle the out
    if args.parallel_mode == ParallelMode.ROW and not args.overlap_rs_fprop and args.tp_size > 1:
        if args.sequence_parallel:
            out, _ = CommOverlapOps.reduce_scatter(
                out,
                args.output_quantizer,
                tp_world_size,
                args.tp_group,
                out_use_quant,
            )
        else:
            torch.distributed.all_reduce(out, group=args.tp_group)

    if len(inp.shape) != 2:
        # GEMM flattens the input's leading dimensions. Restore every logical
        # intermediate dimension while allowing tensor-parallel collectives to
        # change the first dimension, matching NVTE's Linear output contract.
        out = out.view(-1, *inp.shape[1:-1], out.shape[-1])
    # endregion Handle the out

    # region Prepare backward state
    tensors_to_save = None
    tensor_objects = None
    ctx_attrs = None
    if args.is_grad_enabled:
        # Cached input tensor
        # Discard unneeded data in input tensor
        if (
            weight.requires_grad
            and own_quantized_input
            and isinstance(inputmat, QuantizedTensorStorage)
        ):
            if backward_override is not None:
                inputmat.update_usage(rowwise_usage=True, columnwise_usage=False)
            else:
                # Discard row-wise data since it is not needed in backward pass
                inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)

        saved_inputmat = inputmat
        if save_original_input:
            saved_inputmat = original_inputmat

        inp_shape = inp.shape

        # Scatter intermediate/activation tensors saved for the backward pass
        # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
        fsdp_shapes = _fsdp_scatter_tensors(
            args.fsdp_group,
            saved_inputmat,
            weightmat
            if (args.fp8 or args.debug) and not isinstance(weight, QuantizedTensorStorage)
            else None,
        )

        # FSDP2: Don't save FP8 workspace for non-quantized weights.
        # Backward will re-quantize from the FSDP2 all-gathered weight.
        wt_save = weightmat
        if args.is_fsdp2 and weightmat is not weight:
            wt_save = None
        tensors_to_save, tensor_objects = prepare_for_saving(
            saved_inputmat,
            wt_save,
            weight,
            bias,
        )

        owns_input = saved_inputmat is not inp

        ctx_attrs = {
            "weight_quantizer": weight_quantizer,
            "fsdp_shapes": fsdp_shapes,
            "owns_input": owns_input,
            "is_fsdp2": args.is_fsdp2,
            "inp_shape": inp_shape,
            "save_original_input": save_original_input,
        }
    # endregion Prepare backward state

    return out, new_weight_workspace, tensors_to_save, tensor_objects, ctx_attrs


def _linear_setup_ctx(
    ctx,
    tensors_to_save,
    tensor_objects,
    ctx_attrs,
    inp,
    weight,
    bias,
    args,
):
    ctx.save_for_backward(*tensors_to_save)
    ctx.tensor_objects = tensor_objects
    # Linear saves four logical entries. All-None metadata means each entry
    # maps directly to one saved tensor, including optional None values.
    ctx.plain_saved_tensors = (
        tensor_objects[0] is None
        and tensor_objects[1] is None
        and tensor_objects[2] is None
        and tensor_objects[3] is None
    )
    ctx.args = args
    ctx.use_bias = bias is not None
    ctx.inp_size = inp.size()
    ctx.inp_shape = ctx_attrs.get("inp_shape", inp.shape)
    ctx.debug = args.debug

    ctx.fp8 = args.fp8
    ctx.backward_override = args.backward_override
    ctx.sequence_parallel = args.sequence_parallel
    ctx.tp_group = args.tp_group
    ctx.fsdp_group = args.fsdp_group
    ctx.ub_overlap_ag = args.ub_overlap_ag

    ctx.requires_dgrad = inp.requires_grad
    ctx.requires_wgrad = weight.requires_grad
    ctx.save_original_input = ctx_attrs.get("save_original_input", args.save_original_input)

    ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if args.fp8 else None

    # The FP32 dtype gate on main_grad is applied in backward, once main_grad has
    # been materialized: FSDP params only expose it lazily via get_main_grad() and
    # must not be probed during forward.
    ctx.fuse_wgrad_accumulation = (
        args.fuse_wgrad_accumulation
        and (not args.fp8 or ctx.fp8_recipe.mxfp8() or args.backward_override is not None)
        # Dense wgrad uses the FP32 main_grad path, including the first
        # microbatch. Checkpoint replay may change is_first_microbatch;
        # that flag must not change the dtype of the GEMM result.
        and (not args.fp8 or args.is_first_microbatch is None or not args.is_first_microbatch)
    )
    ctx.accumulate_wgrad_into_param_main_grad = (
        args.is_first_microbatch is None or not args.is_first_microbatch
    ) and not getattr(weight, "overwrite_main_grad", False)

    # saved_tensors_hooks may restore the weight as a plain Tensor and drop
    # Python attributes attached to the original Parameter (for example,
    # main_grad and grad_added_to_main_grad). Keep a reference to the original
    # weight for fused wgrad accumulation and MCore DDP bookkeeping.
    # Registered whenever fused accumulation is requested; whether it is actually
    # used is decided in backward, once the main_grad dtype is known.
    if ctx.fuse_wgrad_accumulation and ctx.requires_wgrad:
        ctx.origin_weight_ref = weakref.ref(weight)
        if hasattr(weight, "__fsdp_param__"):
            ctx.main_grad_func = weight.get_main_grad
        else:
            ctx.main_grad_func = lambda: weight.main_grad

    # Forward-computed values that can't be derived here
    ctx.weight_quantizer = ctx_attrs.get("weight_quantizer")
    ctx.fsdp_shapes = ctx_attrs.get("fsdp_shapes")
    ctx.owns_input = ctx_attrs.get("owns_input")
    ctx.is_fsdp2 = ctx_attrs.get("is_fsdp2")

    if args.fp8 and args.backward_override is None and requires_grad(inp, weight, bias):
        ctx.reduce_and_update_bwd_fp8_tensors = _check_fp8_reduce_and_update()
    else:
        ctx.reduce_and_update_bwd_fp8_tensors = False
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


def _linear_backward(
    ctx,
    args: LinearArgs,
    grad_output,
):
    if ctx.plain_saved_tensors:
        # Access exactly once here, so swap/offload unpack hooks still run at
        # the original point. Never cache raw tensors outside save_for_backward.
        inputmat, weight_fp8, weight, bias = ctx.saved_tensors
    else:
        inputmat, weight_fp8, weight, bias = (  # pylint: disable=unbalanced-tuple-unpacking
            restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        )
    ctx.tensor_objects = None

    origin_weight = weight
    main_grad = None
    if ctx.fuse_wgrad_accumulation and ctx.requires_wgrad:
        origin_weight_ref = ctx.origin_weight_ref
        ctx.origin_weight_ref = None
        origin_weight = origin_weight_ref() if origin_weight_ref is not None else None
        assert origin_weight is not None, "weight was removed while fuse_wgrad_accumulation=True"
        main_grad = ctx.main_grad_func()
        origin_weight.main_grad = main_grad
        # FSDP params only expose main_grad in backward, so the FP32 dtype gate
        # lives here; a non-FP32 buffer falls back to the non-fused wgrad path.
        if main_grad.dtype != torch.float32:
            ctx.fuse_wgrad_accumulation = False
    # Gather intermediate/activation tensors if needed
    # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.distributed.FSDP already
    #       shards/unshards the base weights so we don't do it ourselves
    _fsdp_gather_tensors(
        ctx.fsdp_group,
        ctx.fsdp_shapes,
        inputmat,
        weight_fp8,
    )

    tp_world_size = get_distributed_world_size(ctx.tp_group)
    dgrad, wgrad = None, None
    backward_override = args.backward_override

    # region Prepare grad output
    if args.grad_output_quantizer is not None and backward_override is None:
        if not ctx.requires_dgrad:
            args.grad_output_quantizer.set_usage(rowwise=False)
        if not ctx.requires_wgrad:
            args.grad_output_quantizer.set_usage(columnwise=False)
    grad_output, grad_bias = TransformerEngineBaseModule.grad_output_preprocess(
        ctx,
        grad_output,
        args.parallel_mode == ParallelMode.ROW,
        args.grad_output_quantizer,
    )
    # endregion Prepare grad output

    # if args.backward_input_needs_gather:
    #     inputmat, work = allgather  # TODO @Muu 适配下这个路径 这里与dx可以做掩盖

    # FSDP2: Re-create workspace from all-gathered weight when
    # workspace was not saved.
    if weight_fp8 is None and ctx.is_fsdp2:
        if isinstance(weight, QuantizedTensorStorage):
            weight_fp8 = weight
        elif ctx.weight_quantizer is not None:
            ctx.weight_quantizer.set_usage(rowwise=True, columnwise=True)
            weight_fp8 = ctx.weight_quantizer(weight)

    handle = None
    if ctx.requires_dgrad:
        weight_for_dgrad = weight_fp8
        if backward_override == "dequantized":
            if isinstance(weight_for_dgrad, QuantizedTensorStorage):
                weight_for_dgrad = weight_for_dgrad.dequantize(dtype=args.activation_dtype)
            else:
                weight_for_dgrad = cast_if_needed(weight_for_dgrad, args.activation_dtype)
        elif backward_override == "high_precision":
            weight_for_dgrad = weight
            if isinstance(weight_for_dgrad, QuantizedTensorStorage):
                weight_for_dgrad = weight_for_dgrad.dequantize(dtype=args.activation_dtype)
            else:
                weight_for_dgrad = cast_if_needed(weight_for_dgrad, args.activation_dtype)

        if args.overlap_ag_dgrad:
            dgrad, grad_output = CommOverlapOps.allgather_matmul(
                grad_output,
                weight_for_dgrad,
                None,
                tp_world_size,
                ctx.tp_group,
                usage_a=TensorUsage.LHS,
                usage_b=TensorUsage.RHS,
                out_dtype=args.activation_dtype,
            )
        elif args.overlap_rs_dgrad:
            dgrad = CommOverlapOps.matmul_reduce_scatter(
                grad_output,
                weight_for_dgrad,
                None,
                tp_world_size,
                ctx.tp_group,
                usage_a=TensorUsage.LHS,
                usage_b=TensorUsage.RHS,
                out_dtype=args.activation_dtype,
            )
        else:
            if (
                not args.fp8
                and not args.debug
                and type(grad_output) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
                and type(weight_for_dgrad) in (torch.Tensor, torch.nn.Parameter)
            ):
                dgrad = torch.matmul(grad_output, weight_for_dgrad)
            else:
                dgrad = general_gemm(
                    grad_output,
                    weight_for_dgrad,
                    quantization_params=args.grad_input_quantizer if args.debug else None,
                    usage_a=TensorUsage.LHS,
                    usage_b=TensorUsage.RHS,
                    out_dtype=args.activation_dtype,
                )
            if tp_world_size > 1 and args.parallel_mode == ParallelMode.COLUMN:
                if args.sequence_parallel:
                    dgrad, handle = reduce_scatter_along_dim(
                        dgrad,
                        ctx.tp_group,
                        async_op=True,
                    )
                else:
                    handle = torch.distributed.all_reduce(dgrad, group=ctx.tp_group, async_op=True)
    # FSDP2 only handles deallocation of all-gathered weights it allocates.
    # Columnwise data is derived from rowwise after allgather; clear it here.
    if ctx.is_fsdp2 and isinstance(weight_fp8, QuantizedTensorStorage):
        clear_columnwise_cache(weight_fp8)
    if ctx.requires_wgrad:
        if ctx.fuse_wgrad_accumulation:
            out_dtype = main_grad.dtype
        else:
            out_dtype = args.activation_dtype

        if backward_override == "dequantized" and isinstance(inputmat, QuantizedTensorStorage):
            inputmat = inputmat.dequantize(dtype=args.activation_dtype)  # pylint: disable=no-member
        elif backward_override == "high_precision":
            inputmat = cast_if_needed(inputmat, args.activation_dtype)
        elif (
            ctx.save_original_input
            and ctx.fp8
            and not ctx.debug
            and not isinstance(inputmat, QuantizedTensorStorage)
        ):
            # Original unquantized input was saved (save_original_input=True).
            # Mirroring upstream NVTE: re-quantize it into a fresh columnwise
            # view here so the standard FP8 wgrad GEMM applies — the flag is a
            # memory optimization, not a precision switch.
            args.input_quantizer.set_usage(rowwise=False, columnwise=True)
            inputmat = args.input_quantizer(inputmat)

        if (
            (backward_override is not None or (ctx.save_original_input and args.overlap_ag_fprop))
            and args.parallel_mode == ParallelMode.COLUMN
            and args.sequence_parallel
            and tp_world_size > 1
        ):
            # Forward saved the local sequence shard (the fused AG fprop only
            # feeds the forward GEMM, mirroring upstream NVTE), so rebuild the
            # full-sequence wgrad operand here (plain after dequantize/cast or
            # quantized after re-quantize, depending on the path).
            inputmat, _ = gather_along_dim(inputmat, ctx.tp_group)

        # grad_output_preprocess (and any overlap all-gather) already returns 2D
        # dense gradients. Only the saved input may still need flattening.
        if not ctx.fp8 and not ctx.debug and inputmat.ndim != 2:
            inputmat = inputmat.view(-1, inputmat.shape[-1])  # pylint: disable=no-member

        delay_wgrad_compute = (
            args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute()
        )
        if (
            not ctx.fp8
            and not ctx.debug
            and not ctx.fuse_wgrad_accumulation
            and not delay_wgrad_compute
            and type(grad_output) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
            and type(inputmat) is torch.Tensor  # pylint: disable=unidiomatic-typecheck
        ):
            # Match the dense general_gemm path without constructing kwargs or a
            # deferred-call closure. Tensor subclasses retain the generic path.
            wgrad = torch.matmul(grad_output.t(), inputmat)
        else:
            wgrad_kwargs = {
                "usage_a": TensorUsage.LHS_TRANS,
                "usage_b": TensorUsage.RHS,
                "out_dtype": out_dtype,
            }

            def wgrad_gemm(
                x: QuantizedTensor,
                dy: QuantizedTensor,
            ):
                if ctx.fuse_wgrad_accumulation:
                    # Match LayerNormLinear's dense overwrite/accumulate
                    # semantics without changing the existing FP8 path.
                    if not ctx.fp8 and not ctx.accumulate_wgrad_into_param_main_grad:
                        main_grad.zero_()
                    general_gemm_add(
                        main_grad,
                        dy,
                        x,
                        quantization_params=args.grad_weight_quantizer if args.debug else None,
                        **wgrad_kwargs,
                    )
                    _wgrad = setup_dummy_wgrad(origin_weight)
                else:
                    _wgrad = general_gemm(
                        dy,
                        x,
                        quantization_params=args.grad_weight_quantizer if args.debug else None,
                        **wgrad_kwargs,
                    )
                # TODO @Muu 这里看下bgrad是否需要计算
                return _wgrad, None

            if delay_wgrad_compute:
                args.wgrad_store.put([inputmat, grad_output], wgrad_gemm)
            else:
                wgrad, _ = wgrad_gemm(inputmat, grad_output)
    # Update FP8 scaling factors if needed
    if ctx.reduce_and_update_bwd_fp8_tensors:
        FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
    if handle is not None:
        handle.wait()

    # Scatter fp8 weight buffers
    if (ctx.fp8 or ctx.debug) and not isinstance(weight, QuantizedTensorStorage):
        _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

    return (
        wgrad,
        dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
        grad_bias,
    )


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        bias,
        args: LinearArgs,
    ) -> Any:
        # Make sure input dimensions are compatible
        _, in_features = weight.shape
        assert inp.shape[-1] == in_features, "GEMM not possible"
        out, new_weight_workspace, tensors_to_save, tensor_objects, ctx_attrs = (
            _linear_forward_impl(
                weight,
                inp,
                bias,
                args,
            )
        )

        if ctx is not None:
            _linear_setup_ctx(
                ctx,
                tensors_to_save,
                tensor_objects,
                ctx_attrs,
                inp,
                weight,
                bias,
                args,
            )

        return out, new_weight_workspace

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _grad_weight_workspace):
        result = _linear_backward(
            ctx,
            ctx.args,
            grad_output,
        )
        return result + (None,)  # args


class Linear(TransformerEngineBaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        device: Union[torch.device, str] = "npu",
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        save_original_input: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        self.symmetric_ar_type = symmetric_ar_type
        self.save_original_input = save_original_input

        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ub_bulk_wgrad)

        if device == "meta":
            assert parameters_split is None, "Cannot split module parameters on 'meta' device."

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in GemmParallelModes, (
            f"parallel_mode {parallel_mode} not supported"
        )

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

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

        # Initialize params in FP8
        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        # Contiguous buffers for params
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

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for pname, split_size in parameters_split.items():
                self.weight_names.append(f"{pname.rstrip('_')}_weight")
                self.bias_names.append(f"{pname.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(pname, str) for pname in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            for pname in parameters_split:
                self.weight_names.append(f"{pname.rstrip('_')}_weight")
                self.bias_names.append(f"{pname.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        # Adjust parameter splits for tensor-parallel distribution
        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        # Construct weight parameters
        # Note: Register weights together so that they are adjacent to
        # each other in Linear.parameters(). This makes it more likely
        # that they will stay contiguous if the weights are
        # manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError(
                    "Splitting QuantizedTensor into multiple params is not supported"
                )

            # Construct weight parameter
            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=FP8FwdTensorIdx.GEMM1_WEIGHT,
            )

        # Construct bias parameters if needed
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
            for pname in self.bias_names:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, pname, bias)

        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        if self.wgrad_store.delay_wgrad_compute():
            for pname, param in self.named_parameters():
                if pname in self.weight_names or pname in self.bias_names:
                    param.skip_backward_post_hook = True

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
        fp8_grad: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        is_grad_enabled = torch.is_grad_enabled()

        inp = self.prepare_forward(inp)

        skip_fp8_weight_update = None  # TODO @Muu 图相关暂时不做处理

        try:
            weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()
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
                linear_fn = _Linear.apply
                autograd_ctx = []
            else:
                linear_fn = _Linear.forward
                autograd_ctx = [None]

            cache_name = None if (is_first_microbatch is None or self.is_fsdp2) else "weight"
            weight_workspace = (
                self._fp8_workspaces.get(cache_name) if cache_name is not None else None
            )

            linear_args = LinearArgs(
                is_first_microbatch=is_first_microbatch,
                fp8=self.fp8,
                debug=debug,
                weight_workspace=weight_workspace,
                wgrad_store=self.wgrad_store,
                input_quantizer=input_quantizer,
                weight_quantizer=weight_quantizer,
                output_quantizer=output_quantizer,
                grad_input_quantizer=grad_input_quantizer,
                grad_weight_quantizer=grad_weight_quantizer,
                grad_output_quantizer=grad_output_quantizer,
                fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                cpu_offloading=False,  # TODO @Muu 看下是否需要支持下 CPU OFFLOAD
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
                save_original_input=self.save_original_input,
                backward_override=backward_override,
                # Forward comm+GEMM overlap stays enabled under backward_override
                # (forward is unchanged FP8 compute); only backward overlaps are
                # disabled, mirroring upstream NVTE.
                overlap_ag_fprop=self.overlap_ag_fprop and not debug,
                # Backward dgrad AG/RS fused comm also supports plain tensors
                # (CommOverlapOps plain branch), so it stays enabled under the
                # override as well; see _linear_setup_ctx note.
                overlap_rs_dgrad=self.overlap_rs_dgrad and not debug,
                overlap_rs_fprop=self.overlap_rs_fprop and not debug,
                overlap_ag_dgrad=self.overlap_ag_dgrad and not debug,
                fsdp_group=self.fsdp_group,
                is_fsdp2=self.is_fsdp2,
            )

            bias = bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None

            out, new_weight_workspace = linear_fn(
                *autograd_ctx,
                weight_tensor,
                inp,
                bias,
                linear_args,
            )

            if new_weight_workspace is not None and cache_name is not None:
                if isinstance(new_weight_workspace, torch.Tensor):
                    new_weight_workspace = new_weight_workspace.detach()
                self._fp8_workspaces[cache_name] = new_weight_workspace

        finally:
            self.end_forward()
        if self.gemm_bias_unfused_add:
            out += cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for linear weights
            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for bias in self.bias_names:
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, bias),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, bias), True, 0, 1)

    def _get_quantizers(self, fp8_output, fp8_grad, is_grad_enabled):
        if not self.fp8:
            return (None, None, None, None, None, None)
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

    def _get_debug_quantizers(self, fp8_output, fp8_grad, is_grad_enabled):
        original_quantizers = self._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)
        assert TEDebugState.debug_enabled
        from ...debug.pytorch.debug_quantization import DebugQuantizer

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        return tuple(
            DebugQuantizer(self.name, name, q, self.tp_group, self.tp_size)
            for name, q in zip(names, original_quantizers)
        )

    def _get_weight_and_bias_tensors(
        self,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get concatenated weight and bias tensors
        unfused_weights = self._get_weight_tensors()
        weight_tensor = noop_cat(unfused_weights)
        if self.use_bias:
            bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
        else:
            bias_tensor = None
        return weight_tensor, bias_tensor

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        if len(self.weight_names) == 1:
            # Read the current registered parameter on every call: FSDP and
            # parameter reuse may replace it or rebind its storage.
            weight = getattr(self, self.weight_names[0])
            if type(weight) in (torch.Tensor, torch.nn.Parameter):
                return [weight]
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

    def _get_weight_quantizers(self):
        if not self.fp8:
            return [None]
        weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        return [weight_quantizer]

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


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int
) -> None:
    """Set attributes needed for TP"""
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)
