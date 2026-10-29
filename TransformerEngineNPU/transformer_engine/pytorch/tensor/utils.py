# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions for using fp8 tensors as weights"""

import math
from typing import Optional

import torch
import torch_npu

from ..quantized_tensor import QuantizedTensor


_PARTIAL_AMAX_CHUNK_BYTES = 32 * 1024 * 1024


def to_torch_dtype(dtype) -> torch.dtype:
    """Normalize a dtype argument to ``torch.dtype``.

    TENPU represents datatypes as plain ``torch.dtype`` throughout, while GPU
    TE uses the ``transformer_engine_torch.DType`` pybind enum. Accept either:
    ``torch.dtype`` is returned unchanged, and any object exposing a
    ``torch_dtype`` property (the ``transformer_engine_torch`` compatibility
    enum) is mapped through it.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == torch_npu.hifloat8:
        return dtype
    torch_dtype = getattr(dtype, "torch_dtype", None)
    if torch_dtype is not None:
        return torch_dtype
    raise TypeError(f"Unsupported dtype {dtype!r}: expected torch.dtype or a DType-like enum.")


def _mxfp8_compute_partial_amax(
    inp: torch.Tensor,
    amax_rowwise: torch.Tensor,
    amax_colwise: torch.Tensor,
    rows: int,
    cols: int,
    start_offset: int,
) -> None:
    """Compute rowwise and columnwise MXFP8 amax values for an input shard."""
    if not inp.is_contiguous() or inp.ndim != 1:
        raise ValueError("input must be a contiguous 1D tensor")
    if inp.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input must be float32 or bfloat16, got {inp.dtype}")
    if rows <= 0 or cols <= 0 or rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(f"rows and cols must be positive multiples of 32, got {rows} and {cols}")
    if start_offset < 0 or start_offset + inp.numel() > rows * cols:
        raise ValueError("input shard must be within the flattened model weight")
    if not amax_rowwise.is_contiguous() or not amax_colwise.is_contiguous():
        raise ValueError("amax tensors must be contiguous")
    if amax_rowwise.dtype != torch.float32 or amax_colwise.dtype != torch.float32:
        raise ValueError("amax tensors must be float32")
    expected_rowwise_shape = (rows, cols // 32)
    expected_colwise_shape = (rows // 32, cols)
    if tuple(amax_rowwise.shape) != expected_rowwise_shape:
        raise ValueError(
            f"Unexpected rowwise amax shape: expected {expected_rowwise_shape}, "
            f"got {tuple(amax_rowwise.shape)}"
        )
    if tuple(amax_colwise.shape) != expected_colwise_shape:
        raise ValueError(
            f"Unexpected columnwise amax shape: expected {expected_colwise_shape}, "
            f"got {tuple(amax_colwise.shape)}"
        )
    if inp.numel() == 0:
        return

    end_offset = start_offset + inp.numel()
    max_chunk_numel = max(1, _PARTIAL_AMAX_CHUNK_BYTES // inp.element_size())

    rowwise_block_start = start_offset // 32
    rowwise_block_end = (end_offset + 31) // 32
    rowwise_blocks_per_chunk = max(1, max_chunk_numel // 32)
    for block_start in range(rowwise_block_start, rowwise_block_end, rowwise_blocks_per_chunk):
        block_end = min(block_start + rowwise_blocks_per_chunk, rowwise_block_end)
        chunk_start = block_start * 32
        chunk_end = block_end * 32
        copy_start = max(start_offset, chunk_start)
        copy_end = min(end_offset, chunk_end)
        rowwise_values = torch.zeros(
            (block_end - block_start) * 32,
            dtype=inp.dtype,
            device=inp.device,
        )
        rowwise_values[copy_start - chunk_start : copy_end - chunk_start].copy_(
            inp[copy_start - start_offset : copy_end - start_offset]
        )
        rowwise_values.abs_()
        amax_rowwise.reshape(-1)[block_start:block_end].copy_(
            rowwise_values.reshape(-1, 32).amax(dim=1).float()
        )
        del rowwise_values

    columnwise_block_len = 32 * cols
    columnwise_block_start = start_offset // columnwise_block_len
    columnwise_block_end = (end_offset + columnwise_block_len - 1) // columnwise_block_len
    columnwise_blocks_per_chunk = max(1, max_chunk_numel // columnwise_block_len)
    for block_start in range(
        columnwise_block_start,
        columnwise_block_end,
        columnwise_blocks_per_chunk,
    ):
        block_end = min(block_start + columnwise_blocks_per_chunk, columnwise_block_end)
        chunk_start = block_start * columnwise_block_len
        chunk_end = block_end * columnwise_block_len
        copy_start = max(start_offset, chunk_start)
        copy_end = min(end_offset, chunk_end)
        columnwise_values = torch.zeros(
            (block_end - block_start) * columnwise_block_len,
            dtype=inp.dtype,
            device=inp.device,
        )
        columnwise_values[copy_start - chunk_start : copy_end - chunk_start].copy_(
            inp[copy_start - start_offset : copy_end - start_offset]
        )
        columnwise_values.abs_()
        amax_colwise[block_start:block_end].copy_(
            columnwise_values.reshape(block_end - block_start, 32, cols).amax(dim=1).float()
        )
        del columnwise_values


def _fp8_block_scaling_compute_partial_amax(
    inp: torch.Tensor,
    amax: torch.Tensor,
    h: int,
    w: int,
    start_offset: int,
    block_len: int,
) -> None:
    """Compute 2D block-scaled FP8 amax values for a contiguous input shard."""
    if block_len != 128:
        raise ValueError(f"Only block_len=128 is supported, got {block_len}")
    if not inp.is_contiguous():
        raise ValueError("input must be contiguous")
    if inp.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input must be float32 or bfloat16, got {inp.dtype}")
    if amax.ndim != 2 or amax.dtype != torch.float32:
        raise ValueError("amax must be a 2D float32 tensor")
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be positive, got {h} and {w}")
    if start_offset < 0 or start_offset + inp.numel() > h * w:
        raise ValueError("input shard must be within the flattened model weight")

    block_rows = (h + block_len - 1) // block_len
    block_cols = (w + block_len - 1) // block_len
    if amax.shape[0] < block_rows or amax.shape[1] < block_cols:
        raise ValueError(
            f"amax shape {tuple(amax.shape)} is smaller than {(block_rows, block_cols)}"
        )
    if inp.numel() == 0:
        return

    tile_row_size = block_len * w
    tile_row_start = start_offset // tile_row_size
    if (
        h % block_len == 0
        and w % block_len == 0
        and start_offset % tile_row_size == 0
        and inp.numel() % tile_row_size == 0
    ):
        tile_rows = inp.numel() // tile_row_size
        local_amax = inp.view(tile_rows, block_len, block_cols, block_len).abs()
        amax[tile_row_start : tile_row_start + tile_rows, :block_cols].copy_(
            local_amax.amax(dim=3).amax(dim=1)
        )
        return

    end_offset = start_offset + inp.numel()
    tile_row_end = (end_offset + tile_row_size - 1) // tile_row_size
    padded_width = block_cols * block_len
    max_chunk_numel = max(1, _PARTIAL_AMAX_CHUNK_BYTES // inp.element_size())
    tile_rows_per_chunk = max(1, max_chunk_numel // (block_len * padded_width))
    for chunk_tile_start in range(tile_row_start, tile_row_end, tile_rows_per_chunk):
        chunk_tile_end = min(chunk_tile_start + tile_rows_per_chunk, tile_row_end)
        chunk_start = chunk_tile_start * tile_row_size
        chunk_end = chunk_tile_end * tile_row_size
        copy_start = max(start_offset, chunk_start)
        copy_end = min(end_offset, chunk_end)
        chunk_tile_rows = chunk_tile_end - chunk_tile_start
        values = torch.zeros(
            (chunk_tile_rows * block_len, w),
            dtype=inp.dtype,
            device=inp.device,
        )
        values.reshape(-1)[copy_start - chunk_start : copy_end - chunk_start].copy_(
            inp[copy_start - start_offset : copy_end - start_offset]
        )
        values.abs_()
        if w % block_len == 0:
            local_amax = values.reshape(chunk_tile_rows, block_len, block_cols, block_len)
        else:
            padded_values = torch.zeros(
                (chunk_tile_rows * block_len, padded_width),
                dtype=inp.dtype,
                device=inp.device,
            )
            padded_values[:, :w].copy_(values)
            local_amax = padded_values.reshape(
                chunk_tile_rows,
                block_len,
                block_cols,
                block_len,
            )
        amax[chunk_tile_start:chunk_tile_end, :block_cols].copy_(
            local_amax.amax(dim=3).amax(dim=1).float()
        )
        del local_amax, values
        if w % block_len != 0:
            del padded_values


def replace_raw_data(tensor: QuantizedTensor, new_raw_data: torch.Tensor) -> None:
    r"""Change a quantized tensor's raw data buffer while preserving values."""
    # pylint: disable=cyclic-import
    from .float8_blockwise_tensor import Float8BlockwiseQTensor
    from .float8_tensor import Float8Tensor
    from .mxfp8_tensor import MXFP8Tensor

    if isinstance(tensor, Float8Tensor):
        raw_data_attr = "_data"
    elif isinstance(tensor, Float8BlockwiseQTensor):
        raw_data_attr = "_rowwise_data"
    elif isinstance(tensor, MXFP8Tensor):
        raise NotImplementedError("replace_raw_data for MXFP8Tensor is not supported yet")
    else:
        raise ValueError(f"replace_raw_data for {type(tensor)} is not supported yet")

    old_raw_data = getattr(tensor, raw_data_attr)
    target_raw_data = new_raw_data
    if old_raw_data.dtype != new_raw_data.dtype:
        if (
            new_raw_data.dtype == torch.uint8
            and old_raw_data.element_size() == new_raw_data.element_size()
            and old_raw_data.size() == new_raw_data.size()
        ):
            target_raw_data = new_raw_data.view(dtype=old_raw_data.dtype)
        else:
            raise ValueError(
                "The data types of raw data don't match: "
                f"old dtype={old_raw_data.dtype}, new dtype={new_raw_data.dtype}"
            )
    target_raw_data.detach().copy_(old_raw_data)
    setattr(tensor, raw_data_attr, target_raw_data)


def quantize_master_weights(
    model_weights,
    master_weights,
    start_offsets,
    group,
    fsdp_shard_model_weights=None,
    manual_post_all_gather_processing=False,
):
    r"""Helper function to cast master weights to FP8 primary weights."""
    from .float8_blockwise_tensor import Float8BlockQuantizer
    from .float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
    from .mxfp8_tensor import MXFP8Quantizer

    delayed_scaling_params = []
    current_scaling_params = []
    blockwise_scaling_params = []
    mxfp8_scaling_params = []

    if fsdp_shard_model_weights is None:
        use_fsdp_shard_model_weights = False
        fsdp_shard_model_weights = [None] * len(model_weights)
    else:
        use_fsdp_shard_model_weights = True

    for model_weight, master_weight, start_offset, fsdp_shard_model_weight in zip(
        model_weights,
        master_weights,
        start_offsets,
        fsdp_shard_model_weights,
    ):
        if hasattr(model_weight, "clear_high_precision_init_val"):
            model_weight.clear_high_precision_init_val()

        if master_weight is not None:
            master_weight = master_weight.to(model_weight.dtype)

        quantizer = model_weight._get_quantizer()
        params = (model_weight, master_weight, start_offset, fsdp_shard_model_weight)

        if isinstance(quantizer, Float8Quantizer):
            delayed_scaling_params.append(params)
        elif isinstance(quantizer, Float8CurrentScalingQuantizer):
            current_scaling_params.append(params)
        elif isinstance(quantizer, Float8BlockQuantizer):
            blockwise_scaling_params.append(params)
        elif isinstance(quantizer, MXFP8Quantizer):
            mxfp8_scaling_params.append(params)
        else:
            raise ValueError(f"quantize_master_weights for {type(quantizer)} is not supported yet")

    extra_args = [
        group,
        use_fsdp_shard_model_weights,
        manual_post_all_gather_processing,
    ]
    if delayed_scaling_params:
        _cast_master_weights_to_fp8_delayed_scaling(delayed_scaling_params, *extra_args)
    if current_scaling_params:
        _cast_master_weights_to_fp8_current_scaling(current_scaling_params, *extra_args)
    if blockwise_scaling_params:
        _cast_master_weights_to_fp8_blockwise_scaling(blockwise_scaling_params, *extra_args)
    if mxfp8_scaling_params:
        _cast_master_weights_to_fp8_mxfp8_scaling(mxfp8_scaling_params, *extra_args)


def cast_master_weights_to_fp8(
    model_weights,
    master_weights,
    start_offsets,
    group,
    fsdp_shard_model_weights=None,
    manual_post_all_gather_processing=False,
):
    r"""Compatibility wrapper for casting master weights to FP8 primary weights."""
    quantize_master_weights(
        model_weights,
        master_weights,
        start_offsets,
        group,
        fsdp_shard_model_weights,
        manual_post_all_gather_processing,
    )


def _cast_master_weights_to_fp8_delayed_scaling(
    params,
    group,
    use_fsdp_shard_model_weights=False,
    manual_post_all_gather_processing=False,
):
    from ..optimizers.multi_tensor_apply import multi_tensor_applier
    from ..optimizers.multi_tensor_scale import multi_tensor_scale
    from ..triton import fp8_tensorwise_scaling as triton_ops

    amaxes, scales, scale_invs = [], [], []
    cast_args = []

    for model_weight, master_weight, start_offset, shard_model_weight_raw in params:
        if not manual_post_all_gather_processing:
            model_weight._reset_caches()

        quantizer = model_weight._get_quantizer()
        amaxes.append(quantizer.amax.view(1))
        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_weight._scale_inv.view(1))

        if master_weight is None:
            continue

        end_offset = _validate_master_weight_shard(model_weight, master_weight, start_offset)
        if not use_fsdp_shard_model_weights:
            shard_model_weight_raw = model_weight._data.view(-1)[start_offset:end_offset]
        amin, local_amax = master_weight.aminmax()
        quantizer.amax.copy_(torch.maximum(-amin, local_amax))
        cast_args.append((master_weight, shard_model_weight_raw, quantizer))

    for master_weight, shard_model_weight_raw, quantizer in cast_args:
        if quantizer.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ) and master_weight.dtype in (torch.float32, torch.bfloat16):
            triton_ops.fp8_tensorwise_partial_cast(
                master_weight.reshape(-1),
                shard_model_weight_raw.reshape(-1),
                quantizer.scale,
                quantizer.dtype,
            )
        else:
            quantized_fragment = quantizer.quantize(master_weight.view(1, -1))
            _copy_quantized_fragment(quantized_fragment._data, shard_model_weight_raw)

    if amaxes:
        dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=amaxes[0].device)
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [amaxes, packed_amax_views], 1.0
        )
        torch.distributed.all_reduce(
            packed_amaxes,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
        )
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [packed_amax_views, amaxes], 1.0
        )

        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [scales, packed_scale_views], 1.0
        )
        torch.reciprocal(packed_scales, out=packed_scales)
        multi_tensor_applier(
            multi_tensor_scale,
            dummy_overflow_buf,
            [packed_scale_views, scale_invs],
            1.0,
        )


def _cast_master_weights_to_fp8_current_scaling(
    params,
    group,
    use_fsdp_shard_model_weights=False,
    manual_post_all_gather_processing=False,
):
    from ..optimizers.multi_tensor_apply import multi_tensor_applier
    from ..optimizers.multi_tensor_compute_scale import multi_tensor_compute_scale_and_scale_inv
    from ..triton import fp8_tensorwise_scaling as triton_ops

    first_quantizer = params[0][0]._get_quantizer()
    device = params[0][0].device
    fp8_dtype = first_quantizer.dtype
    force_pow_2_scales = first_quantizer.force_pow_2_scales
    amax_epsilon = first_quantizer.amax_epsilon
    fp8_max = first_quantizer.fp8_max or torch.finfo(fp8_dtype).max

    # Create a dummy overflow buffer required by multi_tensor_applier.
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)

    # Collect local amax values into a contiguous buffer for one HCCL all-reduce.
    packed_amaxes = torch.zeros(len(params), dtype=torch.float32, device=device)
    amaxes = [packed_amaxes[i : i + 1] for i in range(len(params))]
    scales, scale_invs = [], []

    # Step 1: Compute local amax values and collect scale metadata.
    for (model_weight, master_weight, _, _), amax in zip(params, amaxes):
        quantizer = model_weight._get_quantizer()
        if quantizer.dtype != fp8_dtype:
            raise ValueError(
                "All model weights must have the same fp8 dtype, "
                f"expected {fp8_dtype} but got {quantizer.dtype}"
            )
        if quantizer.force_pow_2_scales != force_pow_2_scales:
            raise ValueError(
                "All model weights must have the same force_pow_2_scales, "
                f"expected {force_pow_2_scales} but got {quantizer.force_pow_2_scales}"
            )
        if quantizer.amax_epsilon != amax_epsilon:
            raise ValueError(
                "All model weights must have the same amax_epsilon, "
                f"expected {amax_epsilon} but got {quantizer.amax_epsilon}"
            )

        scales.append(torch.empty(1, dtype=torch.float32, device=device))
        scale_invs.append(model_weight._scale_inv.view(1))
        if master_weight is not None:
            amin, local_amax = master_weight.aminmax()
            amax.copy_(torch.maximum(-amin, local_amax))

    # Step 2: Reduce local amax values across the data-parallel group.
    torch.distributed.all_reduce(
        packed_amaxes,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    # Step 3: Compute global scales and scale inverses.
    multi_tensor_applier(
        multi_tensor_compute_scale_and_scale_inv,
        dummy_overflow_buf,
        [amaxes, scales, scale_invs],
        fp8_max,
        force_pow_2_scales,
        amax_epsilon,
    )

    from .float8_tensor import Float8Quantizer

    # Step 4: Cast master-weight shards and write their FP8 payloads in place.
    for (
        model_weight,
        master_weight,
        start_offset,
        model_weight_fragment,
    ), scale in zip(params, scales):
        if not manual_post_all_gather_processing:
            model_weight._reset_caches()
        # The local rank does not own this master-weight shard.
        if master_weight is None:
            continue

        end_offset = _validate_master_weight_shard(model_weight, master_weight, start_offset)
        if not use_fsdp_shard_model_weights:
            model_weight_fragment = model_weight._data.view(-1)[start_offset:end_offset]
        if fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and master_weight.dtype in (
            torch.float32,
            torch.bfloat16,
        ):
            triton_ops.fp8_tensorwise_partial_cast(
                master_weight.reshape(-1),
                model_weight_fragment.reshape(-1),
                scale,
                fp8_dtype,
            )
        else:
            quantizer = Float8Quantizer(scale=scale, amax=None, fp8_dtype=fp8_dtype)
            quantized_fragment = quantizer.quantize(master_weight)
            # NPU quantization returns a new payload, so copy it into the shard buffer.
            _copy_quantized_fragment(quantized_fragment._data, model_weight_fragment)


def _cast_master_weights_to_fp8_blockwise_scaling(
    params,
    group,
    use_fsdp_shard_model_weights=False,
    manual_post_all_gather_processing=False,
):
    from ..optimizers.multi_tensor_apply import multi_tensor_applier
    from ..optimizers.multi_tensor_compute_scale import multi_tensor_compute_scale_and_scale_inv
    from ..triton import fp8_block_scaling as triton_ops

    # Parameter attributes
    device = params[0][0].device
    block_len = params[0][0]._get_quantizer().block_len
    fp8_dtype = params[0][0]._get_quantizer().dtype
    force_pow_2_scales = params[0][0]._get_quantizer().force_pow_2_scales
    amax_epsilon = params[0][0]._get_quantizer().amax_epsilon
    if params[0][0]._get_quantizer().block_scaling_dim != 2:
        raise NotImplementedError(
            "Casting master weights only supports 2D blockwise scaling on NPU"
        )

    # Create a dummy overflow buffer, it's needed by multi_tensor_applier.
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)

    # Get the total number of amax elements in all the model weights.
    cu_amax_sizes = [0]
    for model_weight, _, _, _ in params:
        scale_shape = model_weight._get_quantizer().get_scale_shape(
            model_weight.shape, columnwise=False
        )
        num_amaxes = scale_shape[0] * scale_shape[1]
        cu_amax_sizes.append(cu_amax_sizes[-1] + num_amaxes)

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # HCCL kernels at once.
    packed_amaxes = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If a block of a master weight is empty, the
    #         corresponding amax will be set to 0.
    # ---------------------------------------------------------------------------------------------
    amaxes, scales, scale_invs = [], [], []

    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):
        # Make sure all the model weights have the same numerical options.
        quantizer = model_weight._get_quantizer()
        if block_len != quantizer.block_len:
            raise ValueError(
                "All model weights must have the same block_len, "
                f"expected {block_len} but got {quantizer.block_len}"
            )
        if fp8_dtype != quantizer.dtype:
            raise ValueError(
                "All model weights must have the same fp8 dtype, "
                f"expected {fp8_dtype} but got {quantizer.dtype}"
            )
        if force_pow_2_scales != quantizer.force_pow_2_scales:
            raise ValueError(
                "All model weights must have the same force_pow_2_scales, "
                f"expected {force_pow_2_scales} but got {quantizer.force_pow_2_scales}"
            )
        if amax_epsilon != quantizer.amax_epsilon:
            raise ValueError(
                "All model weights must have the same amax_epsilon, "
                f"expected {amax_epsilon} but got {quantizer.amax_epsilon}"
            )
        if quantizer.block_scaling_dim != 2:
            raise NotImplementedError(
                "Casting master weights only supports 2D blockwise scaling on NPU"
            )

        scale_shape = quantizer.get_scale_shape(model_weight.shape, columnwise=False)
        amax = packed_amaxes[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        scale = torch.empty(scale_shape, dtype=torch.float32, device=device)
        scale_inv = getattr(model_weight, "_rowwise_scale_inv", None)
        if scale_inv is None:
            raise ValueError("model_weight._rowwise_scale_inv must not be None")
        if len(scale_shape) != 2:
            raise ValueError(f"scale_shape must be 2D, got {len(scale_shape)}D shape {scale_shape}")
        if len(scale_inv.shape) != 2:
            raise ValueError(
                f"scale_inv must be 2D, got {len(scale_inv.shape)}D shape {scale_inv.shape}"
            )
        if scale_inv.shape[0] != scale_shape[0]:
            raise ValueError(
                f"scale_inv dim 0 mismatch: scale_inv.shape={scale_inv.shape}, "
                f"scale_shape={scale_shape}"
            )
        if scale_inv.shape[1] != scale_shape[1]:
            raise ValueError(
                f"scale_inv dim 1 mismatch: scale_inv.shape={scale_inv.shape}, "
                f"scale_shape={scale_shape}"
            )

        amaxes.append(amax)
        scales.append(scale)
        scale_invs.append(scale_inv)

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            _validate_master_weight_shard(model_weight, master_weight, start_offset)
            if len(model_weight.shape) != 2:
                raise ValueError(
                    "model_weight must be 2D for blockwise scaling, "
                    f"got {len(model_weight.shape)}D shape {model_weight.shape}"
                )
            h, w = model_weight.shape
            _fp8_block_scaling_compute_partial_amax(
                master_weight, amax, h, w, start_offset, block_len
            )

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(
        packed_amaxes,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    multi_tensor_applier(
        multi_tensor_compute_scale_and_scale_inv,
        dummy_overflow_buf,
        [amaxes, scales, scale_invs],
        torch.finfo(fp8_dtype).max,
        force_pow_2_scales,
        amax_epsilon,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (
        model_weight,
        master_weight,
        start_offset,
        model_weight_fragment,
    ), scale in zip(params, scales):
        if not manual_post_all_gather_processing:
            model_weight.update_usage(rowwise_usage=True, columnwise_usage=False)
        if master_weight is None:
            continue

        end_offset = _validate_master_weight_shard(model_weight, master_weight, start_offset)
        if not use_fsdp_shard_model_weights:
            model_weight_fragment = model_weight._rowwise_data.reshape(-1)[start_offset:end_offset]
        if len(model_weight.shape) != 2:
            raise ValueError(
                "model_weight must be 2D for blockwise scaling partial cast, "
                f"got {len(model_weight.shape)}D shape {model_weight.shape}"
            )
        h, w = model_weight.shape
        triton_ops.fp8_block_scaling_partial_cast(
            master_weight,
            model_weight_fragment,
            scale,
            h,
            w,
            start_offset,
            block_len,
            fp8_dtype,
        )


def _cast_master_weights_to_fp8_mxfp8_scaling(
    params,
    group,
    use_fsdp_shard_model_weights=False,
    manual_post_all_gather_processing=False,  # pylint: disable=unused-argument
):
    r"""Helper function to cast master weights to FP8 primary weights for MXFP8 scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    from ..constants import MXFP8_BLOCK_SCALING_SIZE
    from ..optimizers.multi_tensor_apply import multi_tensor_applier
    from ..optimizers.multi_tensor_compute_scale import multi_tensor_compute_scale_inv_e8m0
    from ..triton import mxfp8_scaling as triton_ops

    # Parameter attributes
    device = params[0][0].device
    fp8_dtype = params[0][0]._get_quantizer().dtype
    if fp8_dtype != torch.float8_e4m3fn:
        raise ValueError(f"MXFP8 only supports float8_e4m3fn, got {fp8_dtype}")
    block_len = MXFP8_BLOCK_SCALING_SIZE
    rowwise_shapes, colwise_shapes = [], []
    cu_rowwise_amax_sizes, cu_colwise_amax_sizes = [0], [0]

    # Get the total number of amax elements in all the model weights.
    for model_weight, _, _, _ in params:
        if len(model_weight.shape) != 2:
            raise ValueError(
                "model_weight must be 2D for MXFP8 scaling, "
                f"got {len(model_weight.shape)}D shape {model_weight.shape}"
            )
        quantizer = model_weight._get_quantizer()
        if quantizer.dtype != fp8_dtype:
            raise ValueError(
                "All model weights must have the same fp8 dtype, "
                f"expected {fp8_dtype} but got {quantizer.dtype}"
            )

        h, w = model_weight.shape
        rowwise_shape = (h, math.ceil(w / block_len))
        colwise_shape = (math.ceil(h / block_len), w)
        expected_rowwise_shape = (h, math.ceil(rowwise_shape[1] / 2), 2)
        expected_colwise_shape = (math.ceil(colwise_shape[0] / 2), w, 2)
        if tuple(model_weight._rowwise_scale_inv.shape) != expected_rowwise_shape:
            raise ValueError(
                "Unexpected rowwise MXFP8 scale shape: "
                f"expected {expected_rowwise_shape}, "
                f"got {tuple(model_weight._rowwise_scale_inv.shape)}"
            )
        if tuple(model_weight._columnwise_scale_inv.shape) != expected_colwise_shape:
            raise ValueError(
                "Unexpected columnwise MXFP8 scale shape: "
                f"expected {expected_colwise_shape}, "
                f"got {tuple(model_weight._columnwise_scale_inv.shape)}"
            )
        if model_weight._rowwise_data is None or model_weight._columnwise_data is None:
            raise ValueError("MXFP8 rowwise and columnwise data buffers must not be None")

        rowwise_shapes.append(rowwise_shape)
        colwise_shapes.append(colwise_shape)
        cu_rowwise_amax_sizes.append(cu_rowwise_amax_sizes[-1] + math.prod(rowwise_shape))
        cu_colwise_amax_sizes.append(cu_colwise_amax_sizes[-1] + math.prod(colwise_shape))

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # HCCL kernels at once.
    packed_amaxes = torch.zeros(
        cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[-1],
        dtype=torch.float32,
        device=device,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If a block of a master weight is empty, the
    #         corresponding amax will be set to 0.
    # ---------------------------------------------------------------------------------------------
    amaxes_rowwise, scale_invs_rowwise = [], []
    amaxes_colwise, scale_invs_colwise = [], []
    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):
        rowwise_start = cu_rowwise_amax_sizes[i]
        rowwise_end = cu_rowwise_amax_sizes[i + 1]
        colwise_start = cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[i]
        colwise_end = cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[i + 1]
        amax_rowwise = packed_amaxes[rowwise_start:rowwise_end].reshape(rowwise_shapes[i])
        amax_colwise = packed_amaxes[colwise_start:colwise_end].reshape(colwise_shapes[i])
        scale_inv_rowwise = torch.empty_like(amax_rowwise, dtype=torch.uint8)
        scale_inv_colwise = torch.empty_like(amax_colwise, dtype=torch.uint8)
        amaxes_rowwise.append(amax_rowwise)
        amaxes_colwise.append(amax_colwise)
        scale_invs_rowwise.append(scale_inv_rowwise)
        scale_invs_colwise.append(scale_inv_colwise)

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            _validate_master_weight_shard(model_weight, master_weight, start_offset)
            h, w = model_weight.shape
            _mxfp8_compute_partial_amax(
                master_weight.reshape(-1),
                amax_rowwise,
                amax_colwise,
                h,
                w,
                start_offset,
            )

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(
        packed_amaxes,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    multi_tensor_applier(
        multi_tensor_compute_scale_inv_e8m0,
        None,
        [
            amaxes_rowwise + amaxes_colwise,
            scale_invs_rowwise + scale_invs_colwise,
        ],
    )

    # Pack logical 2D E8M0 scales into the NPU storage layout.
    for (model_weight, _, _, _), rowwise_scale, colwise_scale in zip(
        params,
        scale_invs_rowwise,
        scale_invs_colwise,
    ):
        model_weight._rowwise_scale_inv.zero_()
        model_weight._rowwise_scale_inv[:, :, 0].copy_(rowwise_scale[:, 0::2])
        odd_rowwise_scales = rowwise_scale[:, 1::2]
        model_weight._rowwise_scale_inv[:, : odd_rowwise_scales.shape[1], 1].copy_(
            odd_rowwise_scales
        )
        model_weight._columnwise_scale_inv.zero_()
        model_weight._columnwise_scale_inv[:, :, 0].copy_(colwise_scale[0::2])
        odd_colwise_scales = colwise_scale[1::2]
        model_weight._columnwise_scale_inv[: odd_colwise_scales.shape[0], :, 1].copy_(
            odd_colwise_scales
        )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (
        model_weight,
        master_weight,
        start_offset,
        model_weight_fragment,
    ), scale_inv_rowwise, scale_inv_colwise in zip(
        params,
        scale_invs_rowwise,
        scale_invs_colwise,
    ):
        if master_weight is None:
            continue

        end_offset = _validate_master_weight_shard(model_weight, master_weight, start_offset)
        if use_fsdp_shard_model_weights:
            rowwise_fragment, columnwise_fragment = model_weight_fragment
        else:
            rowwise_fragment = model_weight._rowwise_data.reshape(-1)[start_offset:end_offset]
            columnwise_fragment = model_weight._columnwise_data.reshape(-1)[start_offset:end_offset]

        h, w = model_weight.shape
        triton_ops.mxfp8_scaling_partial_cast(
            master_weight.reshape(-1),
            rowwise_fragment.reshape(-1),
            columnwise_fragment.reshape(-1),
            scale_inv_rowwise,
            scale_inv_colwise,
            h,
            w,
            start_offset,
        )


def _validate_master_weight_shard(model_weight, master_weight, start_offset: int) -> int:
    if start_offset is None:
        raise ValueError("start_offset must not be None when master_weight is provided")
    if start_offset < 0:
        raise ValueError(f"start_offset must be non-negative, got {start_offset}")
    end_offset = start_offset + master_weight.numel()
    if end_offset > model_weight.numel():
        raise ValueError(
            f"end_offset ({end_offset}) exceeds model_weight numel ({model_weight.numel()}), "
            f"start_offset={start_offset}, master_weight numel={master_weight.numel()}"
        )
    return end_offset


def _copy_quantized_fragment(source: torch.Tensor, destination: torch.Tensor) -> None:
    source = source.reshape(-1)
    destination = destination.reshape(-1)
    if source.numel() != destination.numel():
        raise ValueError(
            "Quantized source and destination sizes don't match: "
            f"source numel={source.numel()}, destination numel={destination.numel()}"
        )
    if source.dtype != destination.dtype:
        if source.element_size() != destination.element_size():
            raise ValueError(
                "Quantized source and destination data types don't match: "
                f"source dtype={source.dtype}, destination dtype={destination.dtype}"
            )
        source = source.view(destination.dtype)
    destination.copy_(source)


def post_all_gather_processing(model_weights) -> None:
    """Post-process FP8 primary weights after all-gather."""
    from .float8_blockwise_tensor import Float8BlockwiseQTensor
    from .float8_tensor import Float8Tensor
    from .mxfp8_tensor import MXFP8Tensor
    from .storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage

    if not isinstance(model_weights, list):
        model_weights = [model_weights]

    for model_weight in model_weights:
        if isinstance(model_weight, (Float8BlockwiseQTensor, Float8BlockwiseQTensorStorage)):
            model_weight._create_columnwise()
        elif isinstance(model_weight, Float8Tensor):
            model_weight._reset_caches()
        elif isinstance(model_weight, MXFP8Tensor):
            pass
        elif isinstance(model_weight, QuantizedTensor):
            raise ValueError(
                f"post_all_gather_processing for {type(model_weight)} is not supported"
            )


def clear_columnwise_cache(tensor) -> None:
    """Clear the columnwise cache of a quantized tensor.

    Use-case: FSDP2, where TE allocates all-gathered columnwise data
    (by deriving it from all-gathered rowwise data) in FSDP2 hooks.
    FSDP2 cannot deallocate it when done.
    """
    if hasattr(tensor, "_columnwise_data"):
        tensor._columnwise_data = None
    if hasattr(tensor, "_columnwise_scale_inv"):
        tensor._columnwise_scale_inv = None
    if hasattr(tensor, "_mxfp8_columnwise_data"):
        tensor._mxfp8_columnwise_data = None
    if hasattr(tensor, "_mxfp8_columnwise_scale_inv"):
        tensor._mxfp8_columnwise_scale_inv = None
    if hasattr(tensor, "_transpose"):
        tensor._transpose = None
    if hasattr(tensor, "_transpose_invalid"):
        tensor._transpose_invalid = True


def noop_requested(noop_flag: Optional[torch.Tensor]) -> bool:
    """Return whether an optional FP8 update skip flag is set."""
    if noop_flag is None:
        return False
    return bool(noop_flag.detach().flatten()[0].item())
