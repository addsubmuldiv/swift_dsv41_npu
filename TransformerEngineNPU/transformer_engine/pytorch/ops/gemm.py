# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
from typing import Optional

import torch
import torch_npu

from ..constants import NPUVersion, USAGE_WITH_TRANS
from ..quantization import FP8GlobalStateManager
from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage
from ..tensor.grouped_tensor import GroupedTensor
from ..utils import check_npu_version, get_quant_dtype, view_as_n_dim
from .gemm_comparison import compare_gemm_output


def _align_group_list_device(group_list, device):
    """Keep grouped GEMM split metadata on the same device as input tensors."""
    if isinstance(group_list, torch.Tensor) and group_list.device != device:
        return group_list.to(device=device)
    return group_list


class MXFP8MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, need_grad: bool = True):
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        qdtype = get_quant_dtype(recipe.fp8_format)
        dst_type = qdtype.fwd
        ctx.qdtype = qdtype
        x_2d = view_as_n_dim(x)
        ctx.output_dtype = x.dtype
        if need_grad:
            x_quant, x_scale, ctx.x, ctx.x_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                x_2d, dst_type=dst_type
            )
            w_quant, w_scale, ctx.w, ctx.w_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                weight, dst_type=dst_type
            )
        else:
            x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(x_2d, axis=-1, dst_type=dst_type)
            w_quant, w_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-1, dst_type=dst_type)
            ctx.save_for_backward(x, weight)
        output = torch_npu.npu_quant_matmul(
            x_quant,
            w_quant.t(),
            w_scale.transpose(0, 1),
            pertoken_scale=x_scale,
            output_dtype=x.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        if len(x.shape) != 2:
            output = output.reshape(*x.shape[:-1], *output.shape[1:])
        if weight.requires_grad:
            output.requires_grad = True
        return output

    @staticmethod
    def backward(ctx, grads: torch.Tensor):
        qdtype = ctx.qdtype
        grads_dx, grads_dx_scale, grads_dw, grads_dw_scale = (
            torch_npu.npu_dynamic_mx_quant_with_dual_axis(view_as_n_dim(grads), dst_type=qdtype.bwd)
        )

        if hasattr(ctx, "x"):
            x_quant, x_scale, w_quant, w_scale = ctx.x, ctx.x_scale, ctx.w, ctx.w_scale
        else:
            x, weight = ctx.saved_tensors
            w_quant, w_scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-2, dst_type=qdtype.fwd)
            x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(
                view_as_n_dim(x), axis=-2, dst_type=qdtype.fwd
            )

        dx = torch_npu.npu_quant_matmul(
            grads_dx,
            w_quant,
            w_scale,
            pertoken_scale=grads_dx_scale,
            output_dtype=ctx.output_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        dw = torch_npu.npu_quant_matmul(
            grads_dw.t(),
            x_quant,
            x_scale,
            pertoken_scale=grads_dw_scale.transpose(0, 1),
            output_dtype=ctx.output_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        return dx, dw, None, None, None


def _unwrap_quantized_storage_for_gemm(tensor, usage):
    if not isinstance(tensor, QuantizedTensorStorage) or isinstance(tensor, QuantizedTensor):
        return tensor

    from ...debug.pytorch.debug_quantization import (
        DebugQuantizedTensor,
        GroupedDebugQuantizedTensor,
    )

    unwrap_storage = isinstance(tensor, (DebugQuantizedTensor, GroupedDebugQuantizedTensor))

    return tensor.get_data(usage) if unwrap_storage else tensor


def _process_debug_gemm_output(tensor, debug_quantizer):
    if debug_quantizer is None:
        return tensor

    from ...debug.pytorch.debug_quantization import DebugQuantizer

    if isinstance(debug_quantizer, DebugQuantizer):
        return debug_quantizer.process_gemm_output(tensor)
    return tensor


def general_gemm(A, B, usage_a, usage_b, out_dtype, bias=None, quantization_params=None):
    A = _unwrap_quantized_storage_for_gemm(A, usage_a)
    B = _unwrap_quantized_storage_for_gemm(B, usage_b)
    if isinstance(A, QuantizedTensor):
        out = A.matmul(B, usage_a, usage_b, out_dtype)
        compare_gemm_output(out, A, B, usage_a, usage_b)
        out = out if bias is None else out + bias
        return _process_debug_gemm_output(out, quantization_params)

    if usage_a in USAGE_WITH_TRANS:
        A = A.t()
    if usage_b in USAGE_WITH_TRANS:
        B = B.t()

    if bias is not None and B.dim() == 2:
        # Fuse the bias into the GEMM: a separate add double-rounds, while
        # addmm rounds once and matches NVTE's epilogue behavior.
        # The forward inputmat is not flattened (2/3/4-D passed as-is), so
        # flatten here for addmm and reshape back; reshape is a zero-copy
        # view for already-flattened 2-D inputs.
        lead = A.shape[:-1]
        out = torch.addmm(bias, A.reshape(-1, A.shape[-1]), B).reshape(lead + (B.shape[-1],))
    else:
        out = torch.matmul(A, B)
        out = out if bias is None else out + bias
    return _process_debug_gemm_output(out, quantization_params)


def general_gemm_add(
    main_grad,
    A,
    B,
    usage_a,
    usage_b,
    out_dtype,
    quantization_params=None,
):
    A = _unwrap_quantized_storage_for_gemm(A, usage_a)
    B = _unwrap_quantized_storage_for_gemm(B, usage_b)
    # 检查total_input/grad_output的shape是否有维度为0
    if not all(A.shape) or not all(B.shape):
        return
    if isinstance(A, QuantizedTensor):
        A.matmul_add(main_grad, B, usage_a, usage_b, out_dtype)
    else:
        if usage_a in USAGE_WITH_TRANS:
            A = A.t()
        if usage_b in USAGE_WITH_TRANS:
            B = B.t()
        main_grad.addmm_(A, B)

    processed_out = _process_debug_gemm_output(main_grad, quantization_params)
    if processed_out is not main_grad:
        main_grad.copy_(processed_out)


def matmul_add(main_grad, inp, grad_out):
    # 检查total_input/grad_output的shape是否有维度为0
    # todo:a2/a3需要在累积值做转置后，在框架层做内存连续排布操作，待算子侧优化完毕后，a2/a3/a5逻辑可统一。
    if not all(inp.shape) or not all(grad_out.shape):
        return
    if check_npu_version(NPUVersion.A5):
        main_grad.addmm_(grad_out.t(), inp)
    else:
        main_grad.addmm_(grad_out.t().contiguous(), inp)


def wgrad_gemm_accum_fp32(total_input, grad_output, main_grad):
    matmul_add(main_grad, total_input, grad_output)


def wgrad_gemm_accum_fp16(total_input, grad_output, main_grad):
    matmul_add(main_grad, total_input, grad_output)


def general_grouped_gemm(
    B,
    A,
    group_split: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    layout="TN",
    use_bias: bool = False,
    biases=None,
    group_type: int = 0,
    group_list_type: int = 1,
    split_item: int = 3,
    out_dtype: torch.dtype = torch.bfloat16,
    quantization_params=None,
    cumulative_group_list: Optional[torch.Tensor] = None,
):
    usage_a, usage_b = f"L{layout[1]}", f"R{layout[0]}"
    A = _unwrap_quantized_storage_for_gemm(A, f"L{layout[1]}")
    B = _unwrap_quantized_storage_for_gemm(B, f"R{layout[0]}")

    grouped_debug = False
    if quantization_params is not None:
        from ...debug.pytorch.debug_quantization import GroupedDebugQuantizer

        grouped_debug = isinstance(quantization_params, GroupedDebugQuantizer)

    if grouped_debug:
        quantizers = quantization_params.quantizers

        if out is not None:
            for out_view, inputmat, weight, quantizer in zip(out.unbind(0), A, B, quantizers):
                general_gemm_add(out_view, inputmat, weight, usage_a, usage_b, out_dtype)
                processed_out = quantizer.process_gemm_output(out_view)
                if processed_out is not out_view:
                    out_view.copy_(processed_out)
            return None

        outputs = []
        for index, (inputmat, weight, quantizer) in enumerate(zip(A, B, quantizers)):
            bias = biases[index] if use_bias and biases is not None else None
            if not all(inputmat.shape) or not all(weight.shape):
                output = torch.zeros(
                    (
                        inputmat.shape[-1] if usage_a in USAGE_WITH_TRANS else inputmat.shape[-2],
                        weight.shape[-2] if usage_b in USAGE_WITH_TRANS else weight.shape[-1],
                    ),
                    dtype=out_dtype,
                    device=inputmat.device,
                )
                output = output if bias is None else output + bias
                outputs.append(quantizer.process_gemm_output(output))
                continue
            outputs.append(
                general_gemm(
                    inputmat,
                    weight,
                    usage_a,
                    usage_b,
                    out_dtype,
                    bias=bias,
                    quantization_params=quantizer,
                )
            )
        return torch.stack(outputs, dim=0) if layout == "NT" else torch.cat(outputs, dim=0)

    if isinstance(A, GroupedTensor) and isinstance(B, GroupedTensor):
        if out is not None:
            general_grouped_gemm_add_(
                out=out,
                B=B,
                A=A,
                group_split=group_split,
                layout=layout,
                group_list_type=group_list_type,
                cumulative_group_list=cumulative_group_list,
            )
            return None
        else:
            out = general_grouped_gemm_for_grouped_tensor(
                B,
                A,
                group_split,
                layout,
                use_bias,
                biases,
                group_type,
                group_list_type,
                split_item,
                out_dtype,
            )
            if quantization_params is not None:
                out = _process_debug_gemm_output(out, quantization_params)
            return out

    if isinstance(A, GroupedTensor) or isinstance(B, GroupedTensor):
        raise TypeError(
            "general_grouped_gemm expects both operands to be GroupedTensor "
            "(low-precision path) or both native tensors/lists (dense fallback); "
            f"got {type(A).__name__}(A) with {type(B).__name__}(B)"
        )

    if out is not None:
        general_grouped_gemm_add_(
            out=out,
            B=B,
            A=A,
            group_split=group_split,
            layout=layout,
            group_list_type=group_list_type,
            cumulative_group_list=cumulative_group_list,
        )
        return None

    if isinstance(B, torch.Tensor):
        B = [B]
    if group_type == 0 and len(B) == 1 and B[0].dim() == 2:
        # Packed weights are already 3D. Split-K (group_type == 2, used in
        # wgrad) keeps 2D weights.
        B = [B[0].unsqueeze(0)]
    if layout[0] == "T":
        B = [b.transpose(-1, -2) for b in B]
    if layout[1] == "T":
        if not A.is_contiguous():
            A = A.contiguous()
        A = A.t()

    if A.shape[-1] == 0 and B[0].shape[0] == 0:
        out = torch.zeros(
            len(group_split),
            *A.shape[:-1],
            *B[0].shape[1:],
            device=A.device,
            dtype=out_dtype,
        )
        if quantization_params is not None:
            out = _process_debug_gemm_output(out, quantization_params)
        return out

    if len(B) == 1 and B[0].dim() == 3 and A.shape[-1] == 0 and B[0].shape[-2] == 0:
        # aclnnGroupedMatmul rejects a zero contraction dim (K == 0) unless all
        # M/N are zero; mathematically the result is all zeros anyway.
        out = torch.zeros(
            len(group_split),
            A.shape[0],
            B[0].shape[-1],
            device=A.device,
            dtype=out_dtype,
        )
        if quantization_params is not None:
            out = _process_debug_gemm_output(out, quantization_params)
        return out

    out = torch_npu.npu_grouped_matmul(
        [A],
        B,
        group_list=group_split,
        output_dtype=out_dtype,
        group_type=group_type,
        group_list_type=group_list_type,
        split_item=split_item,
    )[0]

    if use_bias and biases is not None:
        out = out_add_biases(out, biases, group_split)
    if quantization_params is not None:
        out = _process_debug_gemm_output(out, quantization_params)
    return out


def general_grouped_gemm_for_grouped_tensor(
    B: GroupedTensor,
    A: GroupedTensor,
    group_split: torch.Tensor,
    layout="TN",
    use_bias: bool = False,
    biases=None,
    group_type: int = 0,
    group_list_type: int = 1,
    split_item: int = 3,
    out_dtype: torch.dtype = torch.bfloat16,
):
    # 暂时只支持单单单模式

    if not isinstance(A, GroupedTensor) or not isinstance(B, GroupedTensor):
        raise TypeError(
            "general_grouped_gemm_for_grouped_tensor expects GroupedTensor inputs, "
            f"but got {type(A)}(A) with {type(B)}(B)"
        )

    if (A.quantizer is None) != (B.quantizer is None):
        raise RuntimeError("Mixed dense and quantized GroupedTensor GEMM is not supported")

    a_data, a_scale = A.get_data(f"L{layout[1]}")
    b_data, b_scale = B.get_data(f"R{layout[0]}")

    is_low_precision = A.quantizer is not None and B.quantizer is not None
    if (
        (not is_low_precision or not A.quantizer._get_compatible_recipe().mxfp4())
        and a_data.shape[-1] == 0
        and b_data.shape[0] == 0
    ):
        return torch.zeros(
            len(group_split),
            *a_data.shape[:-1],
            *b_data.shape[1:],
            device=A.device,
            dtype=out_dtype,
        )

    group_split = _align_group_list_device(group_split, a_data.device)
    gmm_kwargs = {}
    if is_low_precision:
        gmm_kwargs["scale"] = [b_scale]
        gmm_kwargs["per_token_scale"] = [a_scale]
        recipe = A.quantizer._get_compatible_recipe()
        if recipe.mxfp8():
            gmm_kwargs["scale_dtype"] = torch_npu.float8_e8m0fnu
            gmm_kwargs["per_token_scale_dtype"] = torch_npu.float8_e8m0fnu
        elif recipe.mxfp4():
            gmm_kwargs["scale_dtype"] = torch_npu.float8_e8m0fnu
            gmm_kwargs["per_token_scale_dtype"] = torch_npu.float8_e8m0fnu
            if not (A.quantizer.with_rht and B.quantizer.with_rht):
                gmm_kwargs["x_dtype"] = torch_npu.float4_e2m1fn_x2
                gmm_kwargs["weight_dtype"] = torch_npu.float4_e2m1fn_x2
        if A.quantizer.dtype == torch_npu.hifloat8:
            gmm_kwargs["x_dtype"] = torch_npu.hifloat8
            gmm_kwargs["weight_dtype"] = torch_npu.hifloat8
        if (
            A.quantizer._get_compatible_recipe().mxfp8()
            and B.quantizer._get_compatible_recipe().w4a8()
        ):
            gmm_kwargs["weight_dtype"] = torch_npu.float8_e4m3fn
            gmm_kwargs["scale_dtype"] = torch_npu.float8_e8m0fnu
            gmm_kwargs["per_token_scale_dtype"] = torch_npu.float8_e8m0fnu
            gmm_kwargs["x_dtype"] = torch_npu.float8_e4m3fn

    out = torch_npu.npu_grouped_matmul(
        [a_data],
        [b_data],
        group_list=group_split,
        output_dtype=out_dtype,
        group_type=group_type,
        group_list_type=group_list_type,
        split_item=split_item,
        **gmm_kwargs,
    )[0]

    if use_bias and biases is not None:
        out = out_add_biases(out, biases, group_split)
    return out


def general_grouped_gemm_add_(
    out: torch.Tensor,
    B,
    A,
    group_split: torch.Tensor,
    layout="TN",
    group_list_type: int = 1,
    cumulative_group_list: Optional[torch.Tensor] = None,
):
    a_is_grouped = isinstance(A, GroupedTensor)
    b_is_grouped = isinstance(B, GroupedTensor)
    if a_is_grouped != b_is_grouped:
        raise TypeError(
            "general_grouped_gemm_add_ expects both operands to be GroupedTensor "
            "or both native tensors"
        )

    if a_is_grouped:
        if (A.quantizer is None) != (B.quantizer is None):
            raise RuntimeError("Mixed dense and quantized GroupedTensor GEMM is not supported")
        a_data, a_scale = A.get_data(f"L{layout[1]}")
        b_data, b_scale = B.get_data(f"R{layout[0]}")
        is_low_precision = A.quantizer is not None and B.quantizer is not None
    else:
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            raise TypeError("Dense general_grouped_gemm_add_ operands must be tensors")
        a_data, a_scale = A, None
        b_data, b_scale = B, None
        is_low_precision = False

    group_split = _align_group_list_device(group_split, a_data.device)
    g_size = len(group_split)

    if not is_low_precision:
        # general_grouped_gemm is declared as (B, A, ...). The wgrad caller
        # passes (saved_input, grad_output), and this helper receives the same
        # roles: formal B is saved_input and formal A is grad_output.
        #
        # The A5 primitive computes x1.T @ x2, while TE stores weight/main_grad
        # as [G, out_features, in_features]. Therefore it must receive
        # (grad_output, saved_input), yielding [G, N, K] from [M, N] and [M, K].
        grad_output = A.rowwise_data if a_is_grouped else A
        saved_input = B.rowwise_data if b_is_grouped else B
        out_data = out.rowwise_data if isinstance(out, GroupedTensor) else out
        if grad_output.shape[0] == 0 and saved_input.shape[0] == 0:
            return
        expected_shape = (g_size, grad_output.shape[-1], saved_input.shape[-1])
        if tuple(out.shape) != expected_shape:
            raise RuntimeError(
                "GroupedLinear fused wgrad expects main_grad with shape "
                f"{expected_shape}, but got {tuple(out.shape)}"
            )
        group_list = cumulative_group_list
        if group_list is None:
            group_list = torch.cumsum(group_split, dim=0)
        else:
            group_list = _align_group_list_device(group_list, grad_output.device)
        torch_npu.npu_grouped_matmul_add_(
            out_data,
            grad_output,
            saved_input,
            group_list,
        )
    elif A.quantizer._get_compatible_recipe().mxfp8():
        if a_data.shape[-1] == 0 and b_data.shape[0] == 0:
            return
        torch_npu.npu_add_quant_gmm_(
            out.view(g_size, b_data.shape[-1], -1),
            b_data,
            a_data,
            a_scale,
            x1_scale=b_scale,
            group_list_type=group_list_type,
            group_list=group_split,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )
    else:
        raise RuntimeError(
            "Currently, general_grouped_gemm_add_ only supports BF16 and MXFP8 formats"
        )


def out_add_biases(out, biases, group_split):
    if isinstance(group_split, torch.Tensor):
        group_split = group_split.tolist()
    out = list(torch.split(out, group_split))
    for i, bias in enumerate(biases):
        out[i] = out[i] + bias
    return torch.cat(out, dim=0)
