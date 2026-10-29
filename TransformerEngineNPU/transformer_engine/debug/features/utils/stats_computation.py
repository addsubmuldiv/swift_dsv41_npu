# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Mathematical functions used to compute tensor statistics."""

from typing import NamedTuple

import torch
import torch.nn.functional as F


class BlockwiseDynamicRangeStat(NamedTuple):
    """Blockwise dynamic-range statistic configuration."""

    block_size: int
    dims: int
    max_over_orientations: bool

    def __str__(self):
        suffix = "_max_over_orientations" if self.max_over_orientations else ""
        return f"max_blockwise_dynamic_range_block_size_{self.block_size}_dims_{self.dims}{suffix}"


@torch.compile
def _compute_dynamic_range_top(tensor):
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.numel() == 0:
        return torch.inf
    amax = tensor_abs.max().float()
    if not amax.all():
        amax = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amax)


@torch.compile
def _compute_dynamic_range_bottom(tensor):
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.any():
        amin = tensor_abs.min().float()
    else:
        amin = torch.tensor(1, device=tensor.device).to(torch.float)
    return torch.log2(amin)


def compute_max_blockwise_dynamic_range(tensor, stat_config):
    """Compute the maximum dynamic range within one- or two-dimensional blocks."""

    block_size = stat_config.block_size
    dims = stat_config.dims
    max_over_orientations = stat_config.max_over_orientations

    def _compute_for_one_orientation(tensor):
        total_numel = tensor.numel()
        assert dims in [1, 2], f"dims must be 1 or 2, got {dims}"

        # torch.compile friendly code - standard ** power does not work with jit
        total_block_size = block_size * block_size if dims == 2 else block_size
        assert total_numel % total_block_size == 0, (
            f"Tensor numel ({total_numel}) is not divisible by block_size ({block_size})."
        )

        tensor = tensor.abs().float()
        if dims == 1:
            tensor = tensor.reshape(-1, block_size)
            per_block_amax = tensor.amax(dim=1)
            per_block_amin = tensor.masked_fill(tensor == 0, float("inf")).amin(dim=1)
        else:
            # We want to have tensor of shape [nr_blocks, block_size, block_size],
            # where each block is a block_size x block_size tile of the original tensor.
            dim_y = tensor.shape[-1] // block_size
            tensor = (
                tensor.reshape(-1, block_size, dim_y, block_size)
                .permute(0, 2, 1, 3)
                .reshape(-1, block_size, block_size)
            )
            per_block_amax = tensor.amax(dim=(1, 2))
            per_block_amin = tensor.masked_fill(tensor == 0, float("inf")).amin(dim=(1, 2))

        # Identify blocks that contain any non-zero element
        nonzero_blocks = per_block_amax != 0
        dynamic_range_per_block = torch.where(
            nonzero_blocks,
            torch.log2(per_block_amax) - torch.log2(per_block_amin),
            torch.zeros_like(per_block_amax, dtype=torch.float32),
        )
        return dynamic_range_per_block.max()

    # Flatten to 2D
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])
    if max_over_orientations:
        return max(
            _compute_for_one_orientation(tensor_2d),  # Rowwise orientation
            _compute_for_one_orientation(tensor_2d.transpose(-2, -1)),  # Columnwise orientation
        )
    return _compute_for_one_orientation(tensor_2d)


@torch.compile
def compute_variance(variances, numels, sums):
    """Combine variance helper values with the Welford formula."""
    mean = torch.sum(sums) / torch.sum(numels)
    means = sums / numels
    return torch.sum(numels * (variances - torch.pow(means - mean, 2))) / torch.sum(numels)


@torch.compile
def compute_std(variances, numels, sums):
    return torch.sqrt(compute_variance(variances, numels, sums))


def compute_fp8_delayed_scaling_overflows_num(tensor, quantized_tensor):
    """Compute delayed-scaling FP8 overflow count."""
    dtype = quantized_tensor._fp8_dtype
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"Unsupported FP8 dtype {dtype}")
    fp8_max = torch.finfo(dtype).max
    fp8_min = -fp8_max
    scale_inv = quantized_tensor._scale_inv
    overflows = (tensor > fp8_max * scale_inv) | (tensor < fp8_min * scale_inv)
    return overflows.sum()


FP8_NEGATIVE_ZERO = 128  # represnts -0.0 in fp8


def _nonzero_min(tensor):
    nonzero = tensor[tensor != 0]
    if nonzero.numel() == 0:
        return tensor.new_zeros(())
    return nonzero.min()


def _scale_inv(quantized_tensor, columnwise=False):
    # Standard QuantizedTensor private fields (Float8Tensor, MXFP8Tensor, etc.)
    if (
        columnwise
        and hasattr(quantized_tensor, "_columnwise_scale_inv")
        and quantized_tensor._columnwise_scale_inv is not None
    ):
        scale_inv = quantized_tensor._columnwise_scale_inv
    elif hasattr(quantized_tensor, "_scale_inv") and quantized_tensor._scale_inv is not None:
        scale_inv = quantized_tensor._scale_inv
    elif (
        hasattr(quantized_tensor, "_rowwise_scale_inv")
        and quantized_tensor._rowwise_scale_inv is not None
    ):
        scale_inv = quantized_tensor._rowwise_scale_inv
    # GroupedTensor public fields (no underscore prefix)
    elif (
        columnwise
        and hasattr(quantized_tensor, "columnwise_scale_inv")
        and quantized_tensor.columnwise_scale_inv is not None
    ):
        scale_inv = quantized_tensor.columnwise_scale_inv
    elif (
        hasattr(quantized_tensor, "rowwise_scale_inv")
        and quantized_tensor.rowwise_scale_inv is not None
    ):
        scale_inv = quantized_tensor.rowwise_scale_inv
    else:
        raise ValueError("Quantized tensor does not contain requested scale_inv")
    if scale_inv is None:
        raise ValueError("Quantized tensor does not contain requested scale_inv")

    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

    if isinstance(quantized_tensor, GroupedTensor) and isinstance(
        quantized_tensor.quantizer, MXFP8Quantizer
    ):
        return MXFP8TensorStorage._decode_scale_inv(scale_inv)
    return scale_inv.to(torch.float32)


def count_nonzero_fp8(fp8_data: torch.Tensor) -> torch.Tensor:
    """Count the number of non-zero elements in the fp8 data."""
    fp8_data = fp8_data.view(dtype=torch.uint8)
    zero_vals = torch.tensor([0, FP8_NEGATIVE_ZERO], device=fp8_data.device, dtype=torch.uint8)
    return fp8_data.numel() - torch.isin(fp8_data, zero_vals).sum()


def _dequantize(quantized_tensor, dtype, columnwise=False):
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

    if isinstance(quantized_tensor, GroupedTensor):
        return quantized_tensor.dequantize(dtype=dtype, columnwise=columnwise)
    return quantized_tensor.dequantize()


def _get(buffers, stat_name):
    return buffers[:, stats_to_num[stat_name]]


stats_to_num = {
    "min": 0,
    "max": 1,
    "sum": 2,
    "mean": 3,
    "numel": 4,
    "l1_norm": 5,
    "l2_norm_square": 6,
    "l2_norm": 7,
    "variance": 8,
    "cur_amax": 9,
    "dynamic_range_top": 10,
    "dynamic_range_bottom": 11,
    "std": 12,
    "dynamic_range": 13,
    "fp8_delayed_scaling_overflows_num": 14,
    "fp8_delayed_scaling_overflows%": 15,
    "overflows_num": 16,
    "overflows%": 17,
}

DEPENDENCIES = {
    "min": {"min"},
    "max": {"max"},
    "sum": {"sum"},
    "mean": {"sum", "numel"},
    "numel": {"numel"},
    "l1_norm": {"l1_norm"},
    "l2_norm_square": {"l2_norm_square", "numel"},
    "l2_norm": {"l2_norm_square"},
    "variance": {"variance", "numel", "sum"},
    "cur_amax": {"cur_amax"},
    "dynamic_range_top": {"dynamic_range_top"},
    "dynamic_range_bottom": {"dynamic_range_bottom"},
    "std": {"variance", "numel", "sum"},
    "dynamic_range": {"dynamic_range_top", "dynamic_range_bottom"},
    "fp8_delayed_scaling_overflows_num": {"fp8_delayed_scaling_overflows_num"},
    "fp8_delayed_scaling_overflows%": {"fp8_delayed_scaling_overflows_num", "numel"},
    "overflows_num": {"overflows_num"},
    "overflows%": {"overflows_num", "numel"},
}

STATS = {
    "min": (lambda x, _: torch.min(x), lambda buffers: min(_get(buffers, "min"))),
    "max": (lambda x, _: torch.max(x), lambda buffers: max(_get(buffers, "max"))),
    "sum": (lambda x, _: torch.sum(x), lambda buffers: sum(_get(buffers, "sum"))),
    "mean": (
        lambda x, _: torch.mean(x),
        lambda buffers: sum(_get(buffers, "sum")) / sum(_get(buffers, "numel")),
    ),
    "numel": (
        lambda x, _: x.numel() if hasattr(x, "numel") else x.get_data_tensors()[0].numel(),
        lambda buffers: sum(_get(buffers, "numel")),
    ),
    "l1_norm": (lambda x, _: torch.norm(x, p=1), lambda buffers: sum(_get(buffers, "l1_norm"))),
    "l2_norm_square": (
        lambda x, _: torch.sum(x**2),
        lambda buffers: sum(_get(buffers, "l2_norm_square")),
    ),
    "l2_norm": (
        lambda x, _: torch.norm(x, p=2),
        lambda buffers: torch.sqrt(sum(_get(buffers, "l2_norm_square"))),
    ),
    "variance": (
        lambda x, _: torch.var(x),
        lambda buffers: compute_variance(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "cur_amax": (lambda x, _: x.abs().max(), lambda buffers: max(_get(buffers, "cur_amax"))),
    "dynamic_range_top": (
        lambda x, _: _compute_dynamic_range_top(x),
        lambda buffers: max(_get(buffers, "dynamic_range_top")),
    ),
    "dynamic_range_bottom": (
        lambda x, _: _compute_dynamic_range_bottom(x),
        lambda buffers: min(_get(buffers, "dynamic_range_bottom")),
    ),
    "std": (
        lambda x, _: torch.std(x),
        lambda buffers: compute_std(
            _get(buffers, "variance"), _get(buffers, "numel"), _get(buffers, "sum")
        ),
    ),
    "dynamic_range": (
        lambda x, _: _compute_dynamic_range_top(x) - _compute_dynamic_range_bottom(x),
        lambda buffers: max(_get(buffers, "dynamic_range_top"))
        - min(_get(buffers, "dynamic_range_bottom")),
    ),
    "fp8_delayed_scaling_overflows_num": (
        lambda x, aux: compute_fp8_delayed_scaling_overflows_num(x, aux["fp8_delayed_scaling"]),
        lambda buffers: sum(_get(buffers, "fp8_delayed_scaling_overflows_num")),
    ),
    "fp8_delayed_scaling_overflows%": (
        lambda x, aux: compute_fp8_delayed_scaling_overflows_num(x, aux["fp8_delayed_scaling"])
        / x.numel()
        * 100,
        lambda buffers: 100
        * sum(_get(buffers, "fp8_delayed_scaling_overflows_num"))
        / sum(_get(buffers, "numel")),
    ),
    "overflows_num": (
        lambda x, aux: compute_fp8_delayed_scaling_overflows_num(x, aux[""]),
        lambda buffers: sum(_get(buffers, "overflows_num")),
    ),
    "overflows%": (
        lambda x, aux: compute_fp8_delayed_scaling_overflows_num(x, aux[""]) / x.numel() * 100,
        lambda buffers: 100 * sum(_get(buffers, "overflows_num")) / sum(_get(buffers, "numel")),
    ),
}


def add_underflows_stats(recipe_name, columnwise=False):
    suffix = "_columnwise" if columnwise else ""
    prefix = f"{recipe_name}_" if recipe_name else ""
    stat_num = f"{prefix}underflows_num{suffix}"
    stat_pct = f"{prefix}underflows%{suffix}"
    stats_to_num[stat_num] = len(stats_to_num)
    stats_to_num[stat_pct] = len(stats_to_num)
    STATS[stat_num] = (
        lambda x, aux, _recipe=recipe_name, _columnwise=columnwise: x.count_nonzero()
        - count_nonzero_fp8(
            aux[_recipe].get_data_tensors(
                rowwise_data=not _columnwise,
                columnwise_data=_columnwise,
            )
        ),
        lambda buffers, _stat=stat_num: sum(_get(buffers, _stat)),
    )
    STATS[stat_pct] = (
        lambda x, aux, _recipe=recipe_name, _columnwise=columnwise: (
            x.count_nonzero()
            - count_nonzero_fp8(
                aux[_recipe].get_data_tensors(
                    rowwise_data=not _columnwise,
                    columnwise_data=_columnwise,
                )
            )
        )
        / x.numel()
        * 100,
        lambda buffers, _stat=stat_num: 100
        * sum(_get(buffers, _stat))
        / sum(_get(buffers, "numel")),
    )
    DEPENDENCIES[stat_num] = {stat_num}
    DEPENDENCIES[stat_pct] = {stat_num, "numel"}


def add_scale_inv_stats(recipe_name, columnwise=False):
    suffix = "_columnwise" if columnwise else ""
    prefix = f"{recipe_name}_" if recipe_name else ""
    stat_min = f"{prefix}scale_inv_min{suffix}"
    stat_max = f"{prefix}scale_inv_max{suffix}"
    stats_to_num[stat_min] = len(stats_to_num)
    stats_to_num[stat_max] = len(stats_to_num)
    STATS[stat_min] = (
        lambda _, aux, _recipe=recipe_name, _columnwise=columnwise: _nonzero_min(
            _scale_inv(aux[_recipe], _columnwise)
        ),
        lambda buffers, _stat=stat_min: min(_get(buffers, _stat)),
    )
    STATS[stat_max] = (
        lambda _, aux, _recipe=recipe_name, _columnwise=columnwise: _scale_inv(
            aux[_recipe], _columnwise
        ).max(),
        lambda buffers, _stat=stat_max: max(_get(buffers, _stat)),
    )
    DEPENDENCIES[stat_min] = {stat_min}
    DEPENDENCIES[stat_max] = {stat_max}


def add_mse_stats(recipe_name, columnwise=False):
    suffix = "_columnwise" if columnwise else ""
    prefix = f"{recipe_name}_" if recipe_name else ""
    stat_mse = f"{prefix}mse{suffix}"
    stat_error = f"{prefix}total_square_error{suffix}"
    stats_to_num[stat_mse] = len(stats_to_num)
    stats_to_num[stat_error] = len(stats_to_num)
    STATS[stat_mse] = (
        lambda x, aux, _recipe=recipe_name, _columnwise=columnwise: F.mse_loss(
            x, _dequantize(aux[_recipe], x.dtype, _columnwise), reduction="mean"
        ),
        lambda buffers, _stat=stat_error: sum(_get(buffers, _stat)) / sum(_get(buffers, "numel")),
    )
    STATS[stat_error] = (
        lambda x, aux, _recipe=recipe_name, _columnwise=columnwise: F.mse_loss(
            x, _dequantize(aux[_recipe], x.dtype, _columnwise), reduction="sum"
        ),
        lambda buffers, _stat=stat_error: sum(_get(buffers, _stat)),
    )
    DEPENDENCIES[stat_error] = {stat_error}
    DEPENDENCIES[stat_mse] = {stat_mse, stat_error, "numel"}


def add_max_blockwise_dynamic_range_stats(block_size, dims, max_over_orientations=False):
    stat = BlockwiseDynamicRangeStat(block_size, dims, max_over_orientations)
    if stat in stats_to_num:
        return stat
    stats_to_num[stat] = len(stats_to_num)
    DEPENDENCIES[stat] = {stat}
    STATS[stat] = (
        lambda x, _, _stat=stat: compute_max_blockwise_dynamic_range(x, _stat),
        lambda buffers, _stat=stat: max(_get(buffers, _stat)),
    )
    return stat


for _columnwise in (True, False):
    for _recipe_name in (
        "",
        "fp8_delayed_scaling",
        "mxfp8",
        "fp8_current_scaling",
        "fp8_block_scaling",
    ):
        add_underflows_stats(_recipe_name, _columnwise)
        add_scale_inv_stats(_recipe_name, _columnwise)
        add_mse_stats(_recipe_name, _columnwise)
