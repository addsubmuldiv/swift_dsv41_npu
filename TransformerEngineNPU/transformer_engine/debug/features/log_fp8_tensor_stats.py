# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Log statistics for FP8 quantized tensors."""

from contextlib import contextmanager
import warnings

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method
import torch

from transformer_engine.debug.features.utils import get_reduction_params, next_enabled_iter
from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


ALL_RECIPE_NAMES = [
    "fp8_delayed_scaling",
    "fp8_current_scaling",
    "mxfp8",
    "fp8_block_scaling",
]


def _get_recipe_name(quantizer):
    if quantizer is None:
        return ""
    if isinstance(quantizer, Float8Quantizer):
        return "fp8_delayed_scaling"
    if isinstance(quantizer, Float8CurrentScalingQuantizer):
        return "fp8_current_scaling"
    if isinstance(quantizer, MXFP8Quantizer):
        return "mxfp8"
    if isinstance(quantizer, Float8BlockQuantizer):
        return "fp8_block_scaling"
    raise ValueError(f"Unsupported quantizer type: {type(quantizer)}")


def _get_new_quantizer(recipe_name, fp8_dtype, device):
    if recipe_name == "fp8_block_scaling":
        return Float8BlockQuantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    if recipe_name == "fp8_current_scaling":
        return Float8CurrentScalingQuantizer(
            fp8_dtype=fp8_dtype,
            device=device,
            rowwise=True,
            columnwise=True,
        )
    if recipe_name == "mxfp8":
        return MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    if recipe_name == "fp8_delayed_scaling":
        raise ValueError("Cannot recreate quantizer for fp8_delayed_scaling")
    raise ValueError(f"Unsupported recipe name: {recipe_name}")


@Registry.register_feature(namespace="transformer_engine")
class LogFp8TensorStats(BaseLogTensorStats):
    """Collect statistics for the current FP8 recipe or what-if FP8 recipes."""

    @staticmethod
    def get_recipe_from_stat(stat, default_recipe=""):
        columnwise_stat = stat.endswith("_columnwise")
        for recipe_name in ALL_RECIPE_NAMES:
            if recipe_name in stat:
                return recipe_name, columnwise_stat
        return default_recipe, columnwise_stat

    def check_if_stat_is_supported(self, stat, current_recipe, high_precision_tensor_provided):
        columnwise = stat.endswith("_columnwise")
        if columnwise:
            stat = stat[: -len("_columnwise")]
        recipe_from_stat, _ = self.get_recipe_from_stat(stat, default_recipe=current_recipe)
        stat_without_recipe = stat.replace(recipe_from_stat + "_", "")

        need_high_precision_tensor_stats = ["underflows%", "overflows%", "mse"]
        if (
            stat_without_recipe in need_high_precision_tensor_stats
            and not high_precision_tensor_provided
        ):
            raise ValueError(
                f"Stat {stat} requires a high precision tensor to be provided. "
                "This feature is not supported for weight tensors when using fp8 model "
                "parameters."
            )

        if current_recipe == "" and recipe_from_stat == "":
            raise ValueError(
                f"Stat {stat} does not contain a recipe name and the current recipe is not set."
            )

        if recipe_from_stat != "" and recipe_from_stat not in ALL_RECIPE_NAMES:
            raise ValueError(f"Stat {stat} contains an unsupported recipe name: {recipe_from_stat}")

        if recipe_from_stat in ["fp8_delayed_scaling", "fp8_current_scaling"] and columnwise:
            raise ValueError(
                f"Stat {stat} is not supported. Columnwise tensor statistics are not supported for"
                " fp8_delayed_scaling and fp8_current_scaling."
            )

        if recipe_from_stat == "fp8_delayed_scaling" and stat_without_recipe == "overflows%":
            return True

        if recipe_from_stat == "fp8_block_scaling":
            from transformer_engine.pytorch.quantization import is_fp8_block_scaling_available

            fp8_block_available, reason_for_no_fp8_block = is_fp8_block_scaling_available(
                return_reason=True
            )
            if not fp8_block_available:
                raise ValueError(
                    f"Stat {stat} needs FP8 block scaling support. {reason_for_no_fp8_block}"
                )

        if recipe_from_stat == "mxfp8":
            from transformer_engine.pytorch.quantization import is_mxfp8_available

            mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
            if not mxfp8_available:
                raise ValueError(f"Stat {stat} needs MXFP8 support. {reason_for_no_mxfp8}")

        supported_stats = ["underflows%", "scale_inv_min", "scale_inv_max", "mse"]
        if stat_without_recipe not in supported_stats:
            raise ValueError(
                f"Stat {stat} contains an unsupported stat name: {stat_without_recipe}"
            )
        return True

    @contextmanager
    def update_aux_dict(
        self,
        recipe_name,
        quantized_tensor,
        quantizer,
        original_tensor,
        recipes_in_stats,
    ):
        """Prepare one quantized tensor per recipe and restore tensor usage afterward."""
        aux_dict = {"": quantized_tensor, recipe_name: quantized_tensor}
        old_rowwise_usage = quantizer.rowwise_usage
        old_columnwise_usage = quantizer.columnwise_usage

        for stat_recipe, columnwise in recipes_in_stats:
            if stat_recipe != recipe_name:
                if original_tensor is None:
                    raise ValueError(f"Recipe {stat_recipe} requires a high precision tensor")
                aux_dict[stat_recipe] = _get_new_quantizer(
                    stat_recipe,
                    quantizer.dtype,
                    original_tensor.device,
                )(original_tensor)
            elif hasattr(quantized_tensor, "update_usage"):
                if columnwise:
                    quantized_tensor.update_usage(columnwise_usage=True)
                else:
                    quantized_tensor.update_usage(rowwise_usage=True)

        try:
            yield aux_dict
        finally:
            if hasattr(quantized_tensor, "update_usage"):
                quantized_tensor.update_usage(
                    rowwise_usage=old_rowwise_usage,
                    columnwise_usage=old_columnwise_usage,
                )

    @api_method
    def inspect_tensor_enabled(self, config, layer_name, tensor_name, iteration):
        run_current, next_iter = next_enabled_iter(
            config.get("start_step", None),
            config.get("end_step", None),
            config.get("start_end_list", None),
            config.get("freq", 1),
            iteration,
        )
        STATS_BUFFERS.layers_to_next_iter[layer_name] = next_iter
        return run_current, next_iter

    @api_method
    def inspect_tensor(
        self,
        config,
        layer_name,
        tensor_name,
        iteration,
        tp_group,
        tensor,
        rowwise_quantized_tensor=None,
        columnwise_quantized_tensor=None,
        quantizer=None,
        tp_size=1,
    ):
        assert rowwise_quantized_tensor is columnwise_quantized_tensor

        if quantizer is None:
            warnings.warn(
                f"[LogFp8TensorStats] Skipping stats collection for layer '{layer_name}', "
                f"tensor '{tensor_name}': layer runs in high precision (no quantizer).",
                stacklevel=2,
            )
            return
        quantized_tensor = rowwise_quantized_tensor
        if quantized_tensor is None:
            warnings.warn(
                f"[LogFp8TensorStats] Skipping {layer_name}.{tensor_name}: no quantized tensor.",
                stacklevel=2,
            )
            return
        if isinstance(quantized_tensor, torch.Tensor) and not isinstance(
            quantized_tensor, (QuantizedTensor, GroupedTensor)
        ):
            warnings.warn(
                f"[LogFp8TensorStats] Skipping stats collection for layer '{layer_name}', "
                f"tensor '{tensor_name}': incompatible precision "
                f"(expected QuantizedTensor, got {type(quantized_tensor).__name__}).",
                stacklevel=2,
            )
            return

        recipe_name = _get_recipe_name(quantizer)
        stats = config["stats"]
        for stat in stats:
            self.check_if_stat_is_supported(stat, recipe_name, tensor is not None)

        start_step = config.get("start_step", None)
        end_step = config.get("end_step", None)
        start_end_list = config.get("start_end_list", None)
        if start_end_list is not None:
            start_end_list = tuple(tuple(int(x) for x in interval) for interval in start_end_list)

        options = (
            start_step,
            end_step,
            start_end_list,
            "fp8",
        )

        skip_reduction, reduction_group, reduce_within_microbatch = get_reduction_params(
            tensor_name, tp_group, tp_size
        )
        device = tensor.device if tensor is not None else quantized_tensor.device
        STATS_BUFFERS.try_add_buffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats=stats,
            options=options,
            reduction_group=reduction_group,
            reduce_within_microbatch=reduce_within_microbatch,
            device=device,
        )

        recipes_in_stats = [
            self.get_recipe_from_stat(stat, default_recipe=recipe_name) for stat in stats
        ]

        with self.update_aux_dict(
            recipe_name,
            quantized_tensor,
            quantizer,
            tensor,
            recipes_in_stats,
        ) as aux_dict:
            STATS_BUFFERS.feed(
                layer_name,
                tensor_name,
                options,
                tensor,
                iteration,
                skip_reduction,
                aux_dict=aux_dict,
            )
        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=inspect_tensor: {tensor_name}",
            layer_name,
            extra_cachable_args=(tensor_name,),
        )
