# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear API for TransformerEngine NPU PyTorch

This module implements grouped linear layers for Mixture of Experts (MoE) models,
integrating MindSpeed's NPU-optimized grouped matmul operations.

Reference: TransformerEngine/transformer_engine/pytorch/module/grouped_linear.py
"""

import warnings
import weakref
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from ..constants import GemmParallelModes, dist_group_type
from ..distributed import (
    get_distributed_world_size,
    set_tensor_model_parallel_attributes,
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
)
from ..ops.gemm import general_grouped_gemm, wgrad_gemm_accum_fp32
from ..quantization import FP8GlobalStateManager, QuantizerRole

from ..cpu_offload import is_cpu_offload_enabled, mark_not_offload, start_offload

from ..quantized_tensor import (
    QuantizedTensorStorage,
    prepare_for_saving,
    restore_from_saved,
)
from ..tensor.grouped_tensor import GroupedTensor
from ..utils import (
    cast_if_needed,
    divide,
    init_method_constant,
    requires_grad,
)
from ._common import WeightGradStore
from .base import (
    TransformerEngineBaseModule,
    _check_fp8_reduce_and_update,
    setup_dummy_wgrad,
    quantize_weight,
)
from .performance_grouped_linear_impl import (
    GroupedLinearArgs,
    GroupedLinearMetadata,
    _PerformanceGroupedLinear,
)


class _GroupedLinear(torch.autograd.Function):
    """GroupedLinear autograd function with FP8 support.

    This implements grouped matrix multiplication with support for:
    - NPU optimized grouped matmul
    - FP8 quantization (placeholder for NPU)
    - Gradient computation (dgrad and wgrad)
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        args: GroupedLinearArgs,
        *weights_and_biases,
    ) -> torch.Tensor:
        input_quantizer = args.input_quantizer
        weight_quantizer = args.weight_quantizer
        output_quantizer = args.output_quantizer
        num_gemms = len(args.m_splits)
        weights = list(weights_and_biases[:num_gemms])
        biases = list(weights_and_biases[num_gemms:])
        weight_requires_grad = weights[0].requires_grad
        backward_override = args.backward_override
        save_original_input = args.save_original_input or backward_override == "high_precision"

        # Configure quantizers
        if input_quantizer is not None:
            columnwise = (
                args.is_grad_enabled
                and weight_requires_grad
                and not save_original_input
                and backward_override is None
            )
            if columnwise:
                input_quantizer.columnwise_use_group_quant = True
            input_quantizer.set_usage(
                rowwise=True,
                columnwise=columnwise,
            )
        if weight_quantizer is not None:
            if backward_override is not None:
                columnwise_usage = False
            elif args.is_grad_enabled and inp.requires_grad:
                columnwise_usage = True
            else:
                columnwise_usage = (
                    is_fp8_activation_recompute_enabled()
                    and not in_fp8_activation_recompute_phase()
                )
            weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)
        if args.grad_output_quantizer is not None:
            args.grad_output_quantizer.columnwise_use_group_quant = True

        in_features = weights[0].size(-1)
        if inp.size(-1) != in_features:
            raise ValueError(
                f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
                f"weight tensor (shape={tuple(weights[0].size())})"
            )
        inp_view = inp.reshape(-1, in_features)

        cast_biases = biases
        if args.use_bias:
            bias_dtype = args.activation_dtype
            if args.fp8 and args.activation_dtype == torch.float32:
                # FP8 GEMM only supports BF16/FP16 bias. Dense FP32 GEMM must
                # preserve FP32 bias to match GroupedLinear's dtype contract.
                bias_dtype = torch.bfloat16
            cast_biases = [cast_if_needed(bias, bias_dtype) for bias in biases]
        new_workspace = None
        if args.fp8 or args.debug:
            inputmats = input_quantizer.grouped_quantize(inp_view, args.group_list)
            weight_stacked = torch.stack(weights, dim=0)
            update_ws = args.is_first_microbatch is None or args.is_first_microbatch
            weights_fp8, new_workspace = quantize_weight(
                tensor=weight_stacked,
                quantizer=weight_quantizer,
                workspace=args.weight_workspace,
                update_workspace=update_ws,
                skip_update_flag=args.skip_fp8_weight_update,
                workspace_dtype=args.activation_dtype,
                cache=args.cache_weight,
                group_list=args.group_list,
            )
        else:
            inputmats = cast_if_needed(inp_view, args.activation_dtype)
            weights_fp8 = [cast_if_needed(weight, args.activation_dtype) for weight in weights]
        out = general_grouped_gemm(
            weights_fp8,
            inputmats,
            args.group_split,
            use_bias=args.use_bias,
            biases=cast_biases,
            out_dtype=args.activation_dtype,
            quantization_params=output_quantizer if args.debug else None,
        )

        if args.fp8_calibration:
            input_quantizer.calibrate(inp_view)
            weight_quantizer.calibrate(
                weight_stacked if args.fp8 and not args.debug else torch.stack(weights, dim=0)
            )

        if args.cpu_offloading:
            start_offload(inputmats)
            if isinstance(weights_fp8, torch.Tensor):
                mark_weights_fp8 = [weights_fp8]
            else:
                mark_weights_fp8 = weights_fp8
            mark_not_offload(*mark_weights_fp8, *weights)

        if args.is_grad_enabled:
            ctx.args = args
            saved_inputmats = inputmats
            if backward_override == "high_precision" and weight_requires_grad:
                saved_inputmats = cast_if_needed(inp_view, args.activation_dtype)
            elif backward_override == "dequantized" and isinstance(inputmats, GroupedTensor):
                inputmats.update_usage(rowwise_usage=True, columnwise_usage=False)
            ctx.inputmats = None
            ctx.inputmats_objects = None
            # Register the activation through save_for_backward so the CPU
            # offload saved_tensors_hooks can intercept and offload it
            # (mirrors linear.py). GroupedTensor implements the same
            # prepare_for_saving/restore_from_saved protocol as quantized
            # tensors, so its payloads are decomposed and offloaded too;
            # keeping it on a plain ctx attribute would pin the payloads
            # (~one activation's worth of HBM) on NPU until backward.
            _in_tensors, ctx.inputmats_objects = prepare_for_saving(saved_inputmats)
            ctx.save_for_backward(*_in_tensors)
            ctx.weights_fp8 = weights_fp8
            ctx.biases = biases
            ctx.backward_override = backward_override
            if backward_override == "high_precision" and inp.requires_grad:
                ctx.saved_weights = weights

            ctx.num_gemms = num_gemms
            ctx.inp_shape = inp.shape
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weights[0].requires_grad

            # MCore FSDP: store references for lazy main_grad creation in backward
            if args.fuse_wgrad_accumulation and ctx.requires_wgrad:
                ctx.origin_weight_refs = [weakref.ref(w) for w in weights]
                ctx.origin_weights_overwrite_main_grad = getattr(
                    weights[0], "overwrite_main_grad", False
                )
                if hasattr(weights[0], "__fsdp_param__"):
                    ctx.main_grad_funcs = [weights[i].get_main_grad for i in range(num_gemms)]
                else:
                    ctx.main_grad_funcs = [
                        lambda j=i: weights[j].main_grad for i in range(num_gemms)
                    ]
            if args.fp8 and backward_override is None and requires_grad(inp, weights[0], biases[0]):
                ctx.reduce_and_update_bwd_fp8_tensors = _check_fp8_reduce_and_update()
            else:
                ctx.reduce_and_update_bwd_fp8_tensors = False
            if backward_override is not None:
                ctx.reduce_and_update_bwd_fp8_tensors = False

        return out.view(-1, *inp.shape[1:-1], out.shape[-1]), new_workspace

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, _grad_workspaces
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        args: GroupedLinearArgs = ctx.args
        N = ctx.num_gemms
        backward_override = args.backward_override
        main_grads = [None] * N

        origin_weights = [None] * N
        if args.fuse_wgrad_accumulation and ctx.requires_wgrad:
            origin_weight_refs = getattr(ctx, "origin_weight_refs", None)
            if origin_weight_refs is not None:
                ctx.origin_weight_refs = None
                origin_weights = [ref() if ref is not None else None for ref in origin_weight_refs]
                assert all(w is not None for w in origin_weights), (
                    "weight was removed while fuse_wgrad_accumulation=True"
                )
                main_grads = [main_grad_func() for main_grad_func in ctx.main_grad_funcs]
                for origin_weight, main_grad in zip(origin_weights, main_grads):
                    if main_grad is not None:
                        origin_weight.main_grad = main_grad

        grad_output_view = grad_output.view(-1, grad_output.shape[-1])

        if args.use_bias:
            grad_output_split = torch.split(grad_output_view, args.get_host_splits())
            grad_biases = [grad_output_split[i].sum(dim=0) for i in range(ctx.num_gemms)]
        else:
            grad_biases = [None] * N
        dgrad = None
        wgrad_list = [None] * N
        if (args.fp8 or args.debug) and backward_override is None:
            if not ctx.requires_dgrad:
                args.grad_output_quantizer.set_usage(rowwise=False)
            if not ctx.requires_wgrad:
                args.grad_output_quantizer.set_usage(columnwise=False)
            grad_output_mats: GroupedTensor = args.grad_output_quantizer.grouped_quantize(
                cast_if_needed(grad_output_view, args.activation_dtype)
                if args.debug
                else grad_output_view,
                args.group_list,
            )
        else:
            grad_output_mats = cast_if_needed(grad_output_view, args.activation_dtype)

        if ctx.requires_dgrad:
            weights_fp8: GroupedTensor = ctx.weights_fp8
            if backward_override == "dequantized":
                if isinstance(weights_fp8, GroupedTensor):
                    weight_data = weights_fp8.dequantize(dtype=args.activation_dtype)
                    if weight_data.ndim == 3:
                        weights_for_dgrad = [weight_data[i] for i in range(N)]
                    else:
                        out_features = grad_output_view.shape[-1]
                        weights_for_dgrad = [
                            weight_data[i * out_features : (i + 1) * out_features] for i in range(N)
                        ]
                else:
                    weights_for_dgrad = [
                        (
                            weight.dequantize(dtype=args.activation_dtype)
                            if isinstance(weight, QuantizedTensorStorage)
                            else cast_if_needed(weight, args.activation_dtype)
                        )
                        for weight in weights_fp8
                    ]
            elif backward_override == "high_precision":
                weights_for_dgrad = [
                    (
                        weight.dequantize(dtype=args.activation_dtype)
                        if isinstance(weight, QuantizedTensorStorage)
                        else cast_if_needed(weight, args.activation_dtype)
                    )
                    for weight in ctx.saved_weights
                ]
            else:
                weights_for_dgrad = weights_fp8
            dgrad = general_grouped_gemm(
                weights_for_dgrad,
                grad_output_mats,
                args.group_split,
                layout="NN",
                out_dtype=args.activation_dtype,
                quantization_params=args.grad_input_quantizer if args.debug else None,
            )

        def handle_custom_ddp_from_mcore(weight, _main_grad, _wgrad):
            if ctx.requires_wgrad:
                # Handle custom DDP from mcore.
                if args.fuse_wgrad_accumulation and hasattr(weight, "grad_added_to_main_grad"):
                    _wgrad = setup_dummy_wgrad(weight, _main_grad)
                elif args.fuse_wgrad_accumulation:
                    _wgrad = None
            else:
                _wgrad = None
            return _wgrad

        if ctx.requires_wgrad:
            # Input activation registered via save_for_backward; unpack the
            # (possibly CPU-offloaded and reloaded) tensor.
            (wgrad_inputmats,) = restore_from_saved(ctx.inputmats_objects, ctx.saved_tensors)
            if backward_override == "dequantized" and isinstance(wgrad_inputmats, GroupedTensor):
                wgrad_inputmats = wgrad_inputmats.dequantize(dtype=args.activation_dtype)  # pylint: disable=no-member
            elif backward_override == "high_precision":
                if isinstance(wgrad_inputmats, GroupedTensor):
                    wgrad_inputmats.rowwise_data = cast_if_needed(
                        wgrad_inputmats.rowwise_data, args.activation_dtype
                    )
                else:
                    wgrad_inputmats = cast_if_needed(wgrad_inputmats, args.activation_dtype)

            delay_wgrad_compute = (
                args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute()
            )

            def finalize_wgrad(_wgrad_list):
                # Set MCore DDP flag and install dummy wgrad where applicable.
                _wgrad_list = [
                    handle_custom_ddp_from_mcore(weight, main_grad, wg)
                    for weight, main_grad, wg in zip(origin_weights, main_grads, _wgrad_list)
                ]
                return _wgrad_list, None

            if args.fuse_wgrad_accumulation and not isinstance(wgrad_inputmats, GroupedTensor):
                # bf16 fused: per-expert addmm_ into main_grad.
                # torch.split requires host-side integer split sizes.  m_splits
                # may be accelerator-side GroupedLinearMetadata in the MoE path.
                m_splits = args.get_host_splits()

                def _run_addmm(x, dy):
                    x_splits = torch.split(x, m_splits)
                    dy_splits = torch.split(dy, m_splits)
                    first_microbatch = args.is_first_microbatch is True
                    overwrite = getattr(ctx, "origin_weights_overwrite_main_grad", False)
                    for i in range(N):
                        if main_grads[i] is None:
                            continue
                        # Match the dense Linear contract: the first microbatch
                        # or the layer-wide overwrite_main_grad flag must write
                        # main_grad instead of accumulating onto stale content.
                        if first_microbatch or overwrite:
                            main_grads[i].zero_()
                        wgrad_gemm_accum_fp32(x_splits[i], dy_splits[i], main_grads[i])
                    return finalize_wgrad([None] * N)

                if delay_wgrad_compute:
                    # Delayed: defer to backward_dw() via wgrad_store.
                    args.wgrad_store.put([wgrad_inputmats, grad_output_mats], _run_addmm)
                else:
                    # Eager: accumulate now.
                    wgrad_list, _ = _run_addmm(wgrad_inputmats, grad_output_mats)
            else:
                # Grouped GEMM wgrad (GroupedTensor input or non-fused).
                def grouped_gemm_wgrad(x, dy):
                    wgrad = general_grouped_gemm(
                        x,
                        dy,
                        args.group_split,
                        out=None,
                        layout="NT",
                        use_bias=args.use_bias if grad_biases[0] is None else None,
                        biases=ctx.biases,
                        group_type=2,
                        out_dtype=args.activation_dtype,
                        quantization_params=args.grad_weight_quantizer if args.debug else None,
                    )
                    _wgrad_list = [wgrad[i] for i in range(N)]
                    if args.fuse_wgrad_accumulation:
                        for i in range(N):
                            if main_grads[i] is not None:
                                if getattr(origin_weights[i], "overwrite_main_grad", False):
                                    main_grads[i].copy_(wgrad[i])
                                else:
                                    main_grads[i].add_(wgrad[i])
                        _wgrad_list = [None] * N
                    return finalize_wgrad(_wgrad_list)

                if delay_wgrad_compute:
                    # Delayed: defer to backward_dw() via wgrad_store.
                    args.wgrad_store.put([wgrad_inputmats, grad_output_mats], grouped_gemm_wgrad)
                else:
                    wgrad_list, _ = grouped_gemm_wgrad(wgrad_inputmats, grad_output_mats)

        if ctx.reduce_and_update_bwd_fp8_tensors:
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
        # Release consumed quantized activations. Plain ctx attributes are not
        # freed by autograd after backward runs; without this the FP8 copies
        # (inputmats/weights_fp8) stay alive as long as the node is referenced
        # by retained output tensors. Deferred wgrad paths already captured
        # what they need in wgrad_store before this point.
        ctx.inputmats = None
        ctx.weights_fp8 = None
        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            None,
            *wgrad_list,
            *grad_biases,
        )


class GroupedLinear(TransformerEngineBaseModule):
    """Grouped Linear layer with FP8 support for NPU.

    This layer implements grouped linear transformations, commonly used in
    Mixture of Experts (MoE) models. It supports both column-parallel and
    row-parallel modes for distributed training.

    Parameters
    ----------
    num_gemms : int
        Number of groups (experts)
    in_features : int
        Input feature dimension
    out_features : int
        Output feature dimension
    sequence_parallel : bool, default False
        Whether to use sequence parallelism
    fuse_wgrad_accumulation : bool, default False
        Whether to fuse weight gradient accumulation
    tp_group : ProcessGroup, optional
        Tensor parallel process group
    tp_size : int, default 1
        Tensor parallel world size
    get_rng_state_tracker : callable, optional
        RNG state tracker for initialization
    init_method : callable, optional
        Weight initialization method
    bias : bool, default True
        Whether to use bias
    return_bias : bool, default False
        Whether to return bias separately
    params_dtype : torch.dtype, optional
        Parameter data type
    parallel_mode : str, optional
        Parallelization mode: "column" or "row"
    device : torch.device, default "npu"
        Device for parameters
    """

    def __init__(
        self,
        num_gemms: int,
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
        device: Union[torch.device, str] = "npu",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        save_original_input: bool = False,
        single_grouped_weight: bool = False,
        single_grouped_bias: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)

        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        self.save_original_input = save_original_input
        self.single_grouped_weight = single_grouped_weight
        self.single_grouped_bias = single_grouped_bias

        assert not ub_overlap_rs and not ub_overlap_ag, (
            "GroupedLinear doesn't support Userbuffer overlap."
        )
        self.init_method = init_method
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        self.wgrad_store = WeightGradStore(delay_wgrad_compute)

        self._offsets = {
            "input": 0,
            "weight": 1,
            "output": 2,
            "grad_output": 0,
            "grad_input": 1,
        }
        self._num_fp8_tensors_per_gemm = {
            "fwd": 3,
            "bwd": 2,
        }

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        if self.tp_size > 1 and bias:
            raise ValueError(
                "GroupedLinear doesn't support bias when TP > 1. "
                "Because the TP communication is handled outside of this module."
            )

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in GemmParallelModes, (
            f"parallel_mode {parallel_mode} not supported"
        )

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        if isinstance(device, str):
            if device == "npu":
                device = torch.device(torch.npu.current_device())
            else:
                device = torch.device(device)
        self.device = device

        for i in range(self.num_gemms):
            # Construct weight parameter
            self.register_parameter(
                f"weight{i}",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        self.in_features,
                        device=device,
                        dtype=self.params_dtype,
                    ),
                ),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"],
            )

            # Construct bias parameters if needed
            if self.use_bias:
                self.register_parameter(
                    f"bias{i}",
                    torch.nn.Parameter(
                        torch.empty(
                            self.out_features,
                            device=device,
                            dtype=self.params_dtype,
                        ),
                    ),
                    init_fn=init_method_constant(0.0),
                )
            else:
                bias = torch.Tensor().to(dtype=self.params_dtype, device=device)
                setattr(self, f"bias{i}", bias)

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata(num_gemms=self.num_gemms)

        self.reset_parameters()

        if self.wgrad_store.delay_wgrad_compute():
            delayed_wgrad_param_names = {"weight", "bias"}
            for i in range(self.num_gemms):
                delayed_wgrad_param_names.update((f"weight{i}", f"bias{i}"))
            for pname, param in self.named_parameters():
                if pname in delayed_wgrad_param_names:
                    param.skip_backward_post_hook = True

    def make_grouped_weights(self, defer_init=False) -> None:
        """
        Convert parameters into a GroupedTensor and re-register them as parameters.
        """

        if defer_init:
            return

        weight_quantizers = self._get_weight_quantizers()
        recipe = (
            weight_quantizers[0]._get_compatible_recipe()
            if weight_quantizers and weight_quantizers[0] is not None
            else None
        )
        if recipe is not None and (
            recipe.hif8_delayed() or recipe.delayed() or recipe.float8_current_scaling()
        ):
            self.set_tensor_parallel_attributes(defer_init=defer_init)
            return

        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]

        # Create the weight storage.
        grouped_weights = torch.empty(
            self.num_gemms,
            self.out_features,
            self.in_features,
            dtype=weights[0].dtype,
            device=weights[0].device,
        )
        # Copy existing params into storage.
        with torch.no_grad():
            for i in range(self.num_gemms):
                grouped_weights[i].copy_(weights[i])  # pylint: disable=unsubscriptable-object

        # Re-register as a single grouped weight parameter.
        if not (
            isinstance(grouped_weights, torch.Tensor)
            and (weight_quantizers[0] is None or not weight_quantizers[0].internal)
        ):
            raise RuntimeError("Found internal quantizer with `single_grouped_weight=True`.")
        self.register_parameter(
            "weight",
            torch.nn.Parameter(grouped_weights),
            init_fn=self.init_method,
            get_rng_state_tracker=self.get_rng_state_tracker,
            fp8_meta_index=self._offsets["weight"],
        )
        for i in range(self.num_gemms):
            self.register_parameter(f"weight{i}", None)

        if self.use_bias and self.single_grouped_bias:
            self._make_grouped_biases()

        self.set_tensor_parallel_attributes(defer_init=defer_init)

    def _make_grouped_biases(self) -> None:
        """Pack per-GEMM biases into one ``GroupedTensor`` (``single_grouped_bias``)."""
        grouped_bias = getattr(self, "bias", None)
        if grouped_bias is not None and all(
            getattr(self, f"bias{i}", None) is None for i in range(self.num_gemms)
        ):
            return
        biases = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
        packed = torch.stack([b.detach().clone() for b in biases], dim=0).contiguous()
        grouped_bias = GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=self.num_gemms,
            tensor_shape=(self.out_features,),
            rowwise_data=packed,
            dtype=packed.dtype,
        )
        grouped_bias.requires_grad_(True)
        self.register_parameter("bias", torch.nn.Parameter(grouped_bias))
        for i in range(self.num_gemms):
            self.register_parameter(f"bias{i}", None)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)
        if self.single_grouped_weight:
            self.make_grouped_weights(defer_init=defer_init)
        elif self.single_grouped_bias:
            self._make_grouped_biases()

    def set_tensor_parallel_attributes(self, defer_init=False) -> None:
        """Set attributes needed for TP"""

        if defer_init:
            return
        # Set parallelism attributes for linear weights
        grouped_weight = getattr(self, "weight", None)
        if grouped_weight is not None:
            set_tensor_model_parallel_attributes(
                tensor=grouped_weight,
                is_parallel=True,
                dim=2 if self.parallel_mode == "row" else 1,
                stride=1,
            )
        else:
            for i in range(self.num_gemms):
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, f"weight{i}"),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

        # Set parallelism attributes for linear biases
        if self.use_bias:
            grouped_bias = getattr(self, "bias", None)
            if grouped_bias is not None:
                if self.parallel_mode == "row":
                    setattr(grouped_bias, "sequence_parallel", self.sequence_parallel)
                elif self.parallel_mode == "column":
                    set_tensor_model_parallel_attributes(grouped_bias, True, 0, 1)
            else:
                for i in range(self.num_gemms):
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, f"bias{i}"),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, f"bias{i}"), True, 0, 1)

    def _has_packed_grouped_weight(self) -> bool:
        """Return whether the module currently owns a packed 3D grouped weight."""
        grouped_weight = getattr(self, "weight", None)
        return (
            grouped_weight is not None
            and isinstance(grouped_weight, torch.Tensor)
            and grouped_weight.dim() == 3
            and grouped_weight.size(0) == self.num_gemms
        )

    def _num_quantizer_gemms(self) -> int:
        if self.primary_weights_in_fp8 and not self._has_packed_grouped_weight():
            return self.num_gemms
        return 1

    def _use_performance_grouped_linear(self) -> bool:
        """Performance path is valid only after grouped-weight packing succeeded."""
        if not self.single_grouped_weight:
            return False
        if not self._has_packed_grouped_weight():
            return False
        return True

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        grouped_weight = getattr(self, "weight", None)
        if self.single_grouped_weight:
            if self._has_packed_grouped_weight():
                return [grouped_weight]
            return [getattr(self, f"weight{i}") for i in range(self.num_gemms)]

        if grouped_weight is not None:
            weight_tensors = grouped_weight.quantized_tensors
            if weight_tensors is None:
                weight_tensors = grouped_weight.split_into_quantized_tensors()
        else:
            weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        if not self.fp8 and any(isinstance(w, QuantizedTensorStorage) for w in weight_tensors):
            warnings.warn(
                "You are using quantized weights without quantized compute. "
                "Please make sure this is intentional."
            )
            weight_tensors = [
                w.dequantize() if isinstance(w, QuantizedTensorStorage) else w
                for w in weight_tensors
            ]
        return weight_tensors

    def _get_bias_tensors(self, *, for_linear: bool = False) -> List[torch.Tensor]:
        """Bias tensors, optionally shaped for the selected linear autograd path."""
        grouped_bias = getattr(self, "bias", None)
        if self.single_grouped_bias:
            if grouped_bias is None:
                return [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
            if self._use_performance_grouped_linear():
                return [grouped_bias]
            if isinstance(grouped_bias, GroupedTensor):
                parts = grouped_bias.quantized_tensors
                if parts is None:
                    parts = grouped_bias.split_into_quantized_tensors()
                return [p.reshape(-1) for p in parts]
            assert isinstance(grouped_bias, torch.Tensor), "Expected grouped bias to be a tensor"
            assert grouped_bias.size(0) == self.num_gemms, "Grouped bias size mismatch"
            return [b.reshape(-1) for b in grouped_bias.unbind(dim=0)]

        if grouped_bias is not None:
            if not isinstance(grouped_bias, GroupedTensor):
                assert grouped_bias.size(0) == self.num_gemms, "Grouped bias size mismatch"
                bias_tensors = [b.reshape(-1) for b in grouped_bias.unbind(dim=0)]
            else:
                parts = grouped_bias.quantized_tensors
                if parts is None:
                    parts = grouped_bias.split_into_quantized_tensors()
                bias_tensors = [p.reshape(-1) for p in parts]
        else:
            bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

        if for_linear and self._use_performance_grouped_linear():
            if self.apply_bias:
                return [torch.stack(bias_tensors, dim=0).contiguous()]
            return bias_tensors[:1]

        return bias_tensors

    def forward(
        self,
        inp: torch.Tensor,
        m_splits: Union[List[int], torch.Tensor, GroupedLinearMetadata],
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass of GroupedLinear.

        Parameters
        ----------
        inp : torch.Tensor
            Input tensor
        m_splits : List[int], torch.Tensor, or GroupedLinearMetadata
            Split sizes for the input tensor, or precomputed accelerator-side
            metadata created by :meth:`make_grouped_linear_metadata`.
        is_first_microbatch : {True, False, None}, default None
            Flag for microbatch handling during gradient accumulation.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            Output tensor, or tuple of (output, bias) if return_bias=True
        """
        debug = self.is_debug_iter()

        assert not isinstance(inp, QuantizedTensorStorage), (
            "GroupedLinear doesn't support input tensor in FP8."
        )
        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        is_grad_enabled = torch.is_grad_enabled()

        inp = self.prepare_forward(inp, num_gemms=self.num_gemms)
        backward_override = (
            FP8GlobalStateManager.get_fp8_recipe().backward_override if self.fp8 else None
        )

        try:
            weight_tensors = self._get_weight_tensors()
            bias_tensors = self._get_bias_tensors(for_linear=True)
            if debug:
                debug_m_splits = m_splits
                if isinstance(m_splits, GroupedLinearMetadata):
                    debug_m_splits = m_splits.get_host_splits()
                elif isinstance(m_splits, torch.Tensor):
                    debug_m_splits = tuple(m_splits.tolist())
                quantizers = self._get_debug_quantizers(debug_m_splits)
            else:
                quantizers = self._get_quantizers()
            if debug and self.no_debug_features_active(
                [quantizer for quantizer_group in quantizers for quantizer in quantizer_group]
            ):
                debug = False
                quantizers = self._get_quantizers()

            (
                input_quantizers,
                weight_quantizers,
                output_quantizers,
                grad_input_quantizers,
                grad_weight_quantizers,
                grad_output_quantizers,
            ) = quantizers

            use_performance_grouped_linear = self._use_performance_grouped_linear()
            _linear = (
                _PerformanceGroupedLinear if use_performance_grouped_linear else _GroupedLinear
            )

            if is_grad_enabled:
                linear_fn = _linear.apply
                autograd_ctx = []
            else:
                linear_fn = _linear.forward
                autograd_ctx = [None]
            cache_weight = is_first_microbatch is not None
            cache_name = None if (is_first_microbatch is None or self.is_fsdp2) else "weight"
            weight_workspaces = self._get_weight_workspace(cache_weight, cache_name)

            non_tensor_args = GroupedLinearArgs(
                m_splits=m_splits,
                use_bias=self.apply_bias,
                is_first_microbatch=is_first_microbatch,
                fp8=self.fp8,
                fp8_calibration=self.fp8_calibration,
                wgrad_store=self.wgrad_store,
                input_quantizers=input_quantizers,
                weight_quantizers=weight_quantizers,
                output_quantizers=output_quantizers,
                grad_input_quantizers=grad_input_quantizers,
                grad_weight_quantizers=grad_weight_quantizers,
                grad_output_quantizers=grad_output_quantizers,
                fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                cpu_offloading=is_cpu_offload_enabled(),
                sequence_parallel=self.sequence_parallel,
                activation_dtype=self.activation_dtype,
                is_grad_enabled=is_grad_enabled,
                weight_workspaces=weight_workspaces,
                cache_weight=cache_weight,
                skip_fp8_weight_update=None,
                save_original_input=False,
                backward_override=backward_override,
                debug=debug,
            )

            if use_performance_grouped_linear:
                weight = (
                    weight_tensors[0]
                    if isinstance(weight_tensors, (list, tuple))
                    else weight_tensors
                )
                bias = bias_tensors[0] if isinstance(bias_tensors, (list, tuple)) else bias_tensors
                out, new_workspaces = linear_fn(*autograd_ctx, inp, non_tensor_args, weight, bias)
            else:
                out, new_workspaces = linear_fn(
                    *autograd_ctx, inp, non_tensor_args, *weight_tensors, *bias_tensors
                )

            if new_workspaces is not None:
                if isinstance(new_workspaces, torch.Tensor):
                    new_workspaces = new_workspaces.detach()
                if cache_name is not None:
                    self._fp8_workspaces[cache_name] = new_workspaces

        finally:
            self.end_forward()

        if self.return_bias:
            return_bias_tensors = self._get_bias_tensors()
            return out, [cast_if_needed(b, self.activation_dtype) for b in return_bias_tensors]
        return out

    def make_grouped_linear_metadata(self, m_splits: torch.Tensor) -> GroupedLinearMetadata:
        """Build reusable per-forward split metadata without a host sync."""
        if not isinstance(m_splits, torch.Tensor):
            raise TypeError(
                "GroupedLinear accelerator metadata requires a torch.Tensor, "
                f"but got {type(m_splits).__name__}."
            )
        if m_splits.dim() != 1:
            raise ValueError(
                "GroupedLinear split metadata must be one-dimensional, "
                f"but got shape {tuple(m_splits.shape)}."
            )
        if m_splits.numel() != self.num_gemms:
            raise ValueError(
                "GroupedLinear split metadata size must match num_gemms: "
                f"expected {self.num_gemms}, got {m_splits.numel()}."
            )
        group_split = m_splits
        if group_split.dtype != torch.int64:
            group_split = group_split.to(dtype=torch.int64)
        if group_split.device != self.device:
            group_split = group_split.to(device=self.device)
        group_split = group_split.contiguous()
        return GroupedLinearMetadata(
            group_split=group_split,
            group_list=torch.cumsum(group_split, dim=0),
        )

    def _get_weight_quantizers(self) -> List[Any]:
        """Get the weight quantizers of the module."""
        num_gemms = self._num_quantizer_gemms()
        if not self.fp8 and not self.fp8_calibration and not self.primary_weights_in_fp8:
            return [None] * num_gemms
        weight_quantizers = [
            self.quantizers["scaling_fwd"][
                self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
            ]
            for i in range(num_gemms)
        ]
        for i in range(num_gemms):
            if weight_quantizers[i] is not None:
                weight_quantizers[i].internal = not self.primary_weights_in_fp8
        return weight_quantizers

    def _get_quantizers(self) -> Tuple:
        num_gemms = self._num_quantizer_gemms()
        weight_quantizers = self._get_weight_quantizers()
        input_quantizers, output_quantizers = ([None] * num_gemms, [None] * num_gemms)
        grad_input_quantizers, grad_weight_quantizers, grad_output_quantizers = (
            [None] * num_gemms,
            [None] * num_gemms,
            [None] * num_gemms,
        )
        if self.fp8:
            input_quantizers = [
                self.quantizers["scaling_fwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ]
                for i in range(num_gemms)
            ]
            for i in range(num_gemms):
                if input_quantizers[i] is not None:
                    input_quantizers[i].internal = True
                    input_quantizers[i].optimize_for_gemm = True

            if torch.is_grad_enabled():
                grad_output_quantizers = [
                    self.quantizers["scaling_bwd"][
                        self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                    ]
                    for i in range(num_gemms)
                ]
                for i in range(num_gemms):
                    if grad_output_quantizers[i] is not None:
                        grad_output_quantizers[i].internal = True
                        grad_output_quantizers[i].optimize_for_gemm = True

        return (
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
        )

    def _get_debug_quantizers(self, m_splits: List[int]) -> Tuple:
        """Get debug quantizers for grouped GEMMs."""
        from ...debug.pytorch.debug_quantization import DebugQuantizer, GroupedDebugQuantizer
        from ...debug.pytorch.debug_state import TEDebugState

        if not TEDebugState.debug_enabled:
            raise RuntimeError("TEDebugState.debug_enabled must be True to get debug quantizers")

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        quantizers = self._get_quantizers()
        return tuple(
            [
                GroupedDebugQuantizer(
                    [
                        DebugQuantizer(
                            f"{self.name}.gemm_{gemm_idx}",
                            name,
                            group[0] if group else None,
                            self.tp_group,
                            self.tp_size,
                        )
                        for gemm_idx in range(len(m_splits))
                    ],
                    m_splits,
                )
            ]
            for name, group in zip(names, quantizers)
        )

    def _get_weight_workspace(self, cache_weight, cache_name):
        if not cache_weight:
            return [None]

        return [self._fp8_workspaces.get(cache_name)]

    def backward_dw(self):
        if not self.need_backward_dw():
            return
        (wgrad_list, _), _ = self.wgrad_store.pop()
        if not self.fuse_wgrad_accumulation:
            if self._has_packed_grouped_weight():
                self.weight.grad = wgrad_list.to(self.weight.dtype)
            else:
                for i, wgrad in enumerate(wgrad_list):
                    if wgrad is not None:
                        param = getattr(self, f"weight{i}")
                        param.grad = wgrad.to(param.dtype)
        del wgrad_list
        self._trigger_wgrad_accumulation_and_reduce_hooks()

    def set_recomputed_input_for_delayed_wgrad(self, tensor: torch.Tensor) -> None:
        """Replace the saved input used by the next delayed wgrad computation."""
        if not self.need_backward_dw():
            raise RuntimeError(
                "Cannot set a recomputed wgrad input when delay_wgrad_compute is disabled."
            )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Recomputed grouped-wgrad input must be a torch.Tensor, "
                f"but got {type(tensor).__name__}."
            )
        self.wgrad_store.put_recomputed_input(tensor.detach())

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        name = self.name or ""
        if fwd:
            base = [
                QuantizerRole(module_type="grouped_linear", tensor_type="input", name=name),
                QuantizerRole(module_type="grouped_linear", tensor_type="weight", name=name),
                # Module-level output role; fall back to a concrete role so a
                # qfactory never receives None (its documented contract is to
                # always be handed a QuantizerRole).
                self._output_quantizer_role
                or QuantizerRole(module_type="grouped_linear", tensor_type="output", name=name),
            ]
        else:
            base = [
                QuantizerRole(module_type="grouped_linear", tensor_type="grad_output", name=name),
                # Module-level grad_input role; fall back so qfactory sees a
                # concrete role instead of None.
                self._grad_input_quantizer_role
                or QuantizerRole(module_type="grouped_linear", tensor_type="grad_input", name=name),
            ]
        if self._num_quantizer_gemms() == 1:
            return base
        return [base[i % len(base)] for i in range(num_quantizers)]


__all__ = [
    "GroupedLinear",
]
