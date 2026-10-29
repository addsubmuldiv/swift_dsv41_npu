# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Performance GroupedLinear - optimized NPU Ascend path with single weight/bias tensor."""

import dataclasses
import weakref
from typing import List, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

from ..cpu_offload import mark_not_offload, start_offload
from ..distributed import (
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
)
from ..ops.gemm import general_grouped_gemm
from ..quantization import FP8GlobalStateManager
from ..quantized_tensor import (
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_func_ctx,
)
from ..utils import (
    cast_if_needed,
    requires_grad,
)
from ._common import WeightGradStore
from .base import (
    _check_fp8_reduce_and_update,
    quantize_weight,
    setup_dummy_wgrad,
)


@dataclasses.dataclass(frozen=True, slots=True)
class GroupedLinearMetadata:
    """Per-forward grouped GEMM split metadata kept on the accelerator.

    ``group_split`` contains the number of tokens assigned to each GEMM, while
    ``group_list`` contains the corresponding cumulative offsets.  Keeping the
    pair together lets FC1, FC2, and delayed wgrad reuse the same metadata
    without synchronizing through ``Tensor.tolist()`` or rebuilding it for each
    GroupedLinear invocation.
    """

    group_split: torch.Tensor
    group_list: torch.Tensor
    host_splits: Optional[Tuple[int, ...]] = None

    def __len__(self) -> int:
        return self.group_split.numel()

    def get_host_splits(self) -> Tuple[int, ...]:
        """Return split sizes on the host, converting lazily when required."""
        if self.host_splits is not None:
            return self.host_splits
        return tuple(self.group_split.tolist())


@dataclasses.dataclass(slots=True)
class GroupedLinearArgs:
    m_splits: Union[List[int], torch.Tensor, GroupedLinearMetadata]
    use_bias: bool
    is_first_microbatch: bool
    fp8: bool
    fp8_calibration: bool
    wgrad_store: WeightGradStore
    input_quantizers: List[Quantizer]
    weight_quantizers: List[Quantizer]
    output_quantizers: List[Quantizer]
    grad_input_quantizers: List[Quantizer]
    grad_weight_quantizers: List[Quantizer]
    grad_output_quantizers: List[Quantizer]
    fuse_wgrad_accumulation: bool
    cpu_offloading: bool
    sequence_parallel: bool
    activation_dtype: torch.dtype
    is_grad_enabled: bool
    weight_workspaces: List[torch.Tensor]
    cache_weight: bool
    skip_fp8_weight_update: bool
    save_original_input: bool
    debug: bool
    backward_override: Optional[str] = None
    group_list: torch.Tensor = dataclasses.field(init=False)
    group_split: torch.Tensor = dataclasses.field(init=False)

    @property
    def input_quantizer(self) -> Quantizer:
        return self.input_quantizers[0] if self.input_quantizers else None

    @property
    def weight_quantizer(self) -> Quantizer:
        return self.weight_quantizers[0] if self.weight_quantizers else None

    @property
    def output_quantizer(self) -> Quantizer:
        return self.output_quantizers[0] if self.output_quantizers else None

    @property
    def grad_input_quantizer(self) -> Quantizer:
        return self.grad_input_quantizers[0] if self.grad_input_quantizers else None

    @property
    def grad_output_quantizer(self) -> Quantizer:
        return self.grad_output_quantizers[0] if self.grad_output_quantizers else None

    @property
    def grad_weight_quantizer(self) -> Quantizer:
        return self.grad_weight_quantizers[0] if self.grad_weight_quantizers else None

    @property
    def weight_workspace(self):
        return self.weight_workspaces[0] if self.weight_workspaces else None

    def __post_init__(self):
        if isinstance(self.m_splits, GroupedLinearMetadata):
            self.group_split = self.m_splits.group_split
            self.group_list = self.m_splits.group_list
            return

        if isinstance(self.m_splits, torch.Tensor):
            group_split = self.m_splits
            if group_split.dtype != torch.int64:
                group_split = group_split.to(dtype=torch.int64)
            if group_split.device.type != "npu":
                group_split = group_split.to(device="npu")
            group_split = group_split.contiguous()
        else:
            group_split = torch.tensor(self.m_splits, device="npu", dtype=torch.int64)
        self.group_split = group_split
        self.group_list = torch.cumsum(group_split, dim=0)

    def get_host_splits(self) -> Tuple[int, ...]:
        """Return split sizes on the host only for code paths that require them."""
        if isinstance(self.m_splits, GroupedLinearMetadata):
            return self.m_splits.get_host_splits()
        if isinstance(self.m_splits, torch.Tensor):
            return tuple(self.m_splits.tolist())
        return tuple(self.m_splits)


class _PerformanceGroupedLinear(torch.autograd.Function):
    """PerformanceGroupedLinear autograd function.
    Optimized path with single quantizer, single weight/bias tensor (first dim = num_gemms).
    No save_original_input.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        args: GroupedLinearArgs,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, list]:
        input_quantizer = args.input_quantizer
        weight_quantizer = args.weight_quantizer
        output_quantizer = args.output_quantizer
        num_gemms = weight.size(0)
        in_features = weight.size(2)

        if inp.size(-1) != in_features:
            raise ValueError(
                f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
                f"weight tensor (shape={tuple(weight.size())})"
            )

        weight_requires_grad = weight.requires_grad
        inp_view = inp.reshape(-1, in_features)
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

        # Quantize input
        if args.fp8 or args.debug:
            inputmats = input_quantizer.grouped_quantize(
                cast_if_needed(inp_view, args.activation_dtype) if args.debug else inp_view,
                args.group_list,
            )
        else:
            inputmats = cast_if_needed(inp_view, args.activation_dtype)

        if args.cpu_offloading:
            start_offload(inputmats)

        # Quantize weight (single 3D tensor, first dim = num_gemms)
        new_workspace = None
        weight_data = weight.rowwise_data if isinstance(weight, GroupedTensor) else weight
        if args.fp8 or args.debug:
            if args.debug:
                weight_data = cast_if_needed(weight_data, args.activation_dtype)
            update_ws = args.is_first_microbatch is None or args.is_first_microbatch
            weight_fp8, new_workspace = quantize_weight(
                tensor=weight_data,
                quantizer=weight_quantizer,
                workspace=args.weight_workspace,
                update_workspace=update_ws,
                skip_update_flag=args.skip_fp8_weight_update,
                workspace_dtype=args.activation_dtype,
                cache=args.cache_weight,
                group_list=args.group_list,
            )
        else:
            weight_data = cast_if_needed(weight_data, args.activation_dtype)
            weight_fp8 = weight_data

        # Initialize bias
        bias_dtype = args.activation_dtype
        if args.fp8 and args.activation_dtype == torch.float32:
            # FP8 GEMM only supports BF16/FP16 bias. Dense FP32 GEMM must
            # preserve FP32 bias to match GroupedLinear's dtype contract.
            bias_dtype = torch.bfloat16
        if args.use_bias:
            bias = bias.rowwise_data if isinstance(bias, GroupedTensor) else bias
            bias = cast_if_needed(bias, bias_dtype)

        # Perform GEMM
        out = general_grouped_gemm(
            weight_fp8,
            inputmats,
            args.group_split,
            layout="TN",
            use_bias=args.use_bias,
            biases=bias,
            out_dtype=args.activation_dtype,
            quantization_params=output_quantizer if args.debug else None,
        )

        if args.fp8_calibration:
            input_quantizer.calibrate(inp_view)
            weight_quantizer.calibrate(weight)

        if args.cpu_offloading:
            mark_not_offload(weight_fp8, weight)

        if args.is_grad_enabled:
            if weight_requires_grad:
                if backward_override == "high_precision":
                    inputmats = cast_if_needed(inp_view, args.activation_dtype)
                elif backward_override == "dequantized" and isinstance(inputmats, GroupedTensor):
                    inputmats.update_usage(rowwise_usage=True, columnwise_usage=False)
                elif isinstance(inputmats, GroupedTensor):
                    inputmats.update_usage(rowwise_usage=False, columnwise_usage=True)
            else:
                # Backward restores exactly one input slot. A single None keeps
                # the saved-tensor count stable when the grouped weight is frozen.
                inputmats = None

            saved_weight = (
                weight if backward_override == "high_precision" and inp.requires_grad else None
            )
            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmats,
                weight_fp8,
                saved_weight,
                bias,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.args = args
            ctx.inp_shape = inp.shape
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.backward_override = backward_override
            if args.fuse_wgrad_accumulation and ctx.requires_wgrad:
                ctx.origin_weight_ref = weakref.ref(weight)
                ctx.origin_weights_overwrite_main_grad = getattr(
                    weight, "overwrite_main_grad", False
                )
                if hasattr(weight, "__fsdp_param__"):
                    ctx.main_grad_func = weight.get_main_grad
                else:
                    ctx.main_grad_func = lambda: weight.main_grad
            ctx.num_gemms = num_gemms
            if args.fp8 and requires_grad(inp, weight, bias):
                ctx.reduce_and_update_bwd_fp8_tensors = _check_fp8_reduce_and_update()
            else:
                ctx.reduce_and_update_bwd_fp8_tensors = False
            if backward_override is not None:
                ctx.reduce_and_update_bwd_fp8_tensors = False
                args.fp8 = False
                args.debug = False
                args.grad_input_quantizers = [None] * num_gemms
                args.grad_weight_quantizers = [None] * num_gemms
                args.grad_output_quantizers = [None] * num_gemms

        return out.view(-1, *inp.shape[1:-1], out.shape[-1]), new_workspace

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, _grad_workspaces
    ) -> Tuple[Optional[torch.Tensor], ...]:
        inputmats, weights_fp8, saved_weight, bias = restore_from_func_ctx(ctx)
        args: GroupedLinearArgs = ctx.args
        backward_override = args.backward_override
        origin_weight = None
        main_grad = None
        grad_out_quantizer = args.grad_output_quantizer

        if args.fuse_wgrad_accumulation and ctx.requires_wgrad:
            origin_weight_ref = ctx.origin_weight_ref
            ctx.origin_weight_ref = None
            origin_weight = origin_weight_ref() if origin_weight_ref is not None else None
            assert origin_weight is not None, (
                "weight was removed while fuse_wgrad_accumulation=True"
            )
            main_grad = ctx.main_grad_func()
            if main_grad is not None:
                origin_weight.main_grad = main_grad

        # Preprocess grad output
        grad_output_view = grad_output.view(-1, grad_output.shape[-1])
        grad_bias = None
        if args.use_bias:
            grad_bias = torch.stack(
                [
                    group.sum(dim=0)
                    for group in torch.split(grad_output_view, args.get_host_splits())
                ],
                dim=0,
            )

        if grad_out_quantizer is not None and backward_override is None:
            if not ctx.requires_dgrad:
                args.grad_output_quantizer.set_usage(rowwise=False)
            if not ctx.requires_wgrad:
                args.grad_output_quantizer.set_usage(columnwise=False)
            grad_output_quantized = grad_out_quantizer.grouped_quantize(
                cast_if_needed(grad_output_view, args.activation_dtype)
                if args.debug
                else grad_output_view,
                args.group_list,
            )
        else:
            grad_output_quantized = cast_if_needed(grad_output_view, args.activation_dtype)

        dgrad = None
        if ctx.requires_dgrad:
            if backward_override is None and isinstance(weights_fp8, GroupedTensor):
                # Under backward_override the forward deliberately produces
                # rowwise-only weight data; the override branches below never
                # use the columnwise orientation.
                weights_fp8.update_usage(columnwise_usage=True)
            out_features = grad_output_view.shape[-1]
            in_features = ctx.inp_shape[-1]
            if backward_override == "dequantized":
                weights_for_dgrad = weights_fp8.dequantize(dtype=args.activation_dtype)
            elif backward_override == "high_precision":
                weight_data = saved_weight
                if isinstance(weight_data, QuantizedTensorStorage):
                    weight_data = weight_data.dequantize(dtype=args.activation_dtype)
                else:
                    weight_data = cast_if_needed(weight_data, args.activation_dtype)
                weights_for_dgrad = weight_data.reshape(ctx.num_gemms, out_features, in_features)
            else:
                weights_for_dgrad = weights_fp8
                if isinstance(weights_for_dgrad, GroupedTensor):
                    weights_for_dgrad.update_usage(columnwise_usage=True)
            dgrad = general_grouped_gemm(
                weights_for_dgrad,
                grad_output_quantized,
                args.group_split,
                layout="NN",
                out_dtype=args.activation_dtype,
                quantization_params=args.grad_input_quantizer if args.debug else None,
            )

        wgrad = None
        if ctx.requires_wgrad:
            if backward_override == "dequantized" and isinstance(inputmats, GroupedTensor):
                inputmats = inputmats.dequantize(  # pylint: disable=no-member
                    dtype=args.activation_dtype
                )
            elif backward_override == "high_precision":
                if isinstance(inputmats, GroupedTensor):
                    inputmats.rowwise_data = cast_if_needed(
                        inputmats.rowwise_data, args.activation_dtype
                    )
                else:
                    inputmats = cast_if_needed(inputmats, args.activation_dtype)
            # Under backward_override the saved operands are guaranteed dense and
            # unquantized, so fused accumulation into main_grad is always safe.
            use_fuse_wgrad_accumulation = (
                args.fuse_wgrad_accumulation
                and main_grad is not None
                and not args.debug
                and main_grad.dtype == torch.float32
                and (
                    backward_override is not None
                    or (
                        not args.fp8
                        and getattr(inputmats, "quantizer", None) is None
                        and getattr(grad_output_quantized, "quantizer", None) is None
                    )
                )
            )
            if args.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    use_fuse_wgrad_accumulation
                    and not args.is_first_microbatch
                    and not getattr(ctx, "origin_weights_overwrite_main_grad", False)
                )
            else:
                accumulate_wgrad_into_param_main_grad = use_fuse_wgrad_accumulation and not getattr(
                    ctx, "origin_weights_overwrite_main_grad", False
                )

            delay_wgrad_compute = (
                args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute()
            )

            def wgrad_gemm(x, dy):
                recomputed_input = (
                    args.wgrad_store.pop_recomputed_input() if delay_wgrad_compute else None
                )
                if recomputed_input is not None:
                    if args.fp8 or args.debug:
                        raise RuntimeError(
                            "Replacing delayed grouped-wgrad input currently "
                            "supports only dense non-debug execution."
                        )
                    if tuple(recomputed_input.shape) != tuple(ctx.inp_shape):
                        raise RuntimeError(
                            "Recomputed grouped-wgrad input shape mismatch: "
                            f"expected {tuple(ctx.inp_shape)}, "
                            f"got {tuple(recomputed_input.shape)}."
                        )
                    if recomputed_input.device != x.device:
                        raise RuntimeError(
                            "Recomputed grouped-wgrad input device mismatch: "
                            f"expected {x.device}, "
                            f"got {recomputed_input.device}."
                        )
                    recomputed_input_view = recomputed_input.reshape(-1, recomputed_input.shape[-1])
                    x = cast_if_needed(recomputed_input_view, args.activation_dtype)
                if use_fuse_wgrad_accumulation:
                    if not accumulate_wgrad_into_param_main_grad:
                        main_grad.zero_()
                    general_grouped_gemm(
                        B=x,
                        A=dy,
                        group_split=args.group_split,
                        cumulative_group_list=args.group_list,
                        out=main_grad,
                        layout="NT",
                        group_type=2,
                        out_dtype=main_grad.dtype,
                    )
                    if delay_wgrad_compute:
                        _wgrad = None
                    elif hasattr(origin_weight, "grad_added_to_main_grad"):
                        _wgrad = setup_dummy_wgrad(origin_weight, main_grad)
                    else:
                        _wgrad = None
                else:
                    _wgrad = general_grouped_gemm(
                        B=x,
                        A=dy,
                        group_split=args.group_split,
                        layout="NT",
                        group_type=2,
                        out_dtype=args.activation_dtype,
                        quantization_params=(args.grad_weight_quantizer if args.debug else None),
                    )
                    if args.fuse_wgrad_accumulation:
                        if main_grad is None:
                            raise RuntimeError("Grouped wgrad accumulation requires main_grad.")
                        # Quantized GEMM does not implement the dense out/add
                        # contract. Accumulate its result explicitly, including
                        # delayed wgrad where backward_dw must not drop it.
                        overwrite = getattr(ctx, "origin_weights_overwrite_main_grad", False)
                        if args.is_first_microbatch is True or overwrite:
                            main_grad.copy_(_wgrad)
                        else:
                            main_grad.add_(_wgrad)
                        if hasattr(origin_weight, "grad_added_to_main_grad"):
                            origin_weight.grad_added_to_main_grad = True
                        if delay_wgrad_compute or not hasattr(
                            origin_weight, "grad_added_to_main_grad"
                        ):
                            _wgrad = None
                        else:
                            # Same helper as the fused branch above: main_grad is
                            # already fully accumulated here, so honor
                            # requires_dummy_wgrad and skip the placeholder when
                            # DDP does not run gradient reduction in overlap.
                            _wgrad = setup_dummy_wgrad(origin_weight, main_grad)
                return _wgrad, None

            if delay_wgrad_compute:
                args.wgrad_store.put([inputmats, grad_output_quantized], wgrad_gemm)
            else:
                wgrad, _ = wgrad_gemm(inputmats, grad_output_quantized)

        if ctx.reduce_and_update_bwd_fp8_tensors:
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            None,
            wgrad,
            grad_bias,
        )
