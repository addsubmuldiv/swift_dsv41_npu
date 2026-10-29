# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Buffers used by tensor-statistics debug features."""

import logging
from collections import defaultdict

import torch

from nvdlfw_inspect.logging import FileLogger, MetricLogger
from nvdlfw_inspect.utils import gather_along_first_dim

from transformer_engine.debug.features.utils.stats_computation import (
    DEPENDENCIES,
    STATS,
    stats_to_num,
)
from transformer_engine.debug.pytorch.debug_state import TEDebugState


class _Buffer:
    """Store and combine statistics for one tensor in one layer."""

    def __init__(
        self,
        layer_name,
        tensor_name,
        stats,
        reduction_group,
        reduce_within_microbatch,
        *,
        device,
    ):
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.stats_to_log = stats
        self.reduction_group = reduction_group
        self.reduce_within_microbatch = reduce_within_microbatch
        self.stats_to_compute = set()
        for stat in stats:
            self.stats_to_compute.update(DEPENDENCIES[stat])
        self._buffer = torch.zeros(len(STATS), dtype=torch.float32, device=device)
        self._new_buffer = self._buffer.clone()
        self._tmp_buffer = self._buffer.clone()
        self.modified = torch.zeros(1, dtype=torch.bool, device=device)
        self.iteration = None
        self.skip_reduction = False

    def _reset_before_next_step(self):
        """Reset the state after the logging"""
        self.modified[0] = False

    def feed(self, tensor, iteration, aux_dict=None):
        """Compute and accumulate registered helper statistics."""
        self.iteration = iteration
        if self.modified[0] and not self.reduce_within_microbatch:
            return

        if tensor is not None:
            if (
                tensor.numel() == 0
                if hasattr(tensor, "numel")
                else all((t is None or t.numel() == 0) for t in tensor.get_data_tensors())
            ):
                return

        for stat_name in self.stats_to_compute:
            fn, _ = STATS[stat_name]
            self._tmp_buffer[stats_to_num[stat_name]] = fn(tensor, aux_dict)

        buffers = torch.cat((self._buffer.unsqueeze(0), self._tmp_buffer.unsqueeze(0)), dim=0)

        for stat_name in self.stats_to_compute:
            fn, combinator = STATS[stat_name]
            if self.modified[0]:
                self._new_buffer[stats_to_num[stat_name]] = combinator(buffers)
            else:
                fn = STATS[stat_name][0]
                self._new_buffer[stats_to_num[stat_name]] = fn(tensor, aux_dict)
        self._buffer.copy_(self._new_buffer)
        self.modified[0] = True

    def _gather_helper_stats(self) -> torch.Tensor:
        """
        If tensor stats should be accumulated among many nodes,
        this method gathers all stats from the nodes where the stat was modified.
        """
        if self.skip_reduction:
            return self._buffer.unsqueeze(0)
        mask = gather_along_first_dim(self.modified, process_group=self.reduction_group)[0]
        gathered_buffer, _ = gather_along_first_dim(
            self._buffer.unsqueeze(0), process_group=self.reduction_group
        )
        return gathered_buffer[mask.to(torch.bool)]

    def log(self):
        """Log combined statistics and reset the modified marker."""
        gathered_helper_stats = self._gather_helper_stats()
        if not self.modified[0]:
            return {}

        output = {}
        for stat_name in self.stats_to_log:
            combiner = STATS[stat_name][1]
            stat_value = combiner(gathered_helper_stats)

            stat_name_str = str(stat_name)
            metric_name = f"{self.layer_name}_{self.tensor_name}_{stat_name_str}"

            # nvdlfw_inspect's default FileLogger formats scalar values with 4 decimals.
            # Keep custom/TensorBoard loggers untouched, but use scientific notation for
            # all values in the default statistics log.
            for logger in MetricLogger.enabled_loggers:
                uses_default_file_format = (
                    isinstance(logger, FileLogger)
                    and logger.__class__.log_scalar is FileLogger.log_scalar
                    and hasattr(logger, "stats_logger")
                )
                if uses_default_file_format:
                    scalar_value = stat_value
                    if isinstance(stat_value, torch.Tensor):
                        scalar_value = stat_value.detach().float().cpu().item()
                    logger.stats_logger.log(
                        logging.INFO,
                        "{} \t\t\t\t iteration={:06d} \t\t\t\t value={:.4e}".format(
                            metric_name,
                            self.iteration,
                            scalar_value,
                        ),
                    )
                else:
                    logger.log_scalar(metric_name, stat_value, self.iteration)
            output[(self.layer_name, self.tensor_name, stat_name, self.iteration)] = stat_value
        self._reset_before_next_step()
        return output


class StatsBuffers:
    """Manage statistics buffers for all layer tensors."""

    def __init__(self):
        self.buffers = {}
        self.reduction_group_to_buffer = defaultdict(list)
        self.at_least_one_layer_fed = False
        self.layers_to_next_iter = {}

    def _if_run_reduction(self):
        if self.at_least_one_layer_fed:
            return True

        iteration = TEDebugState.get_iteration()
        layers_to_remove = []
        for layer_name, next_iter in self.layers_to_next_iter.items():
            if next_iter is None:
                layers_to_remove.append(layer_name)
            elif iteration >= next_iter:
                return True
        for layer_name in layers_to_remove:
            self.layers_to_next_iter.pop(layer_name, None)
        return False

    def reset(self):
        self.buffers = {}
        self.reduction_group_to_buffer = defaultdict(list)
        self.at_least_one_layer_fed = False
        self.layers_to_next_iter = {}

    def try_add_buffer(
        self,
        layer_name,
        tensor_name,
        stats,
        options,
        reduction_group,
        reduce_within_microbatch,
        *,
        device,
    ):
        if (layer_name, tensor_name, options) in self.buffers:
            return
        buffer = _Buffer(
            layer_name,
            tensor_name,
            stats,
            reduction_group,
            reduce_within_microbatch,
            device=device,
        )
        self.buffers[(layer_name, tensor_name, options)] = buffer
        self.reduction_group_to_buffer[reduction_group].append(buffer)

    def feed(
        self,
        layer_name,
        tensor_name,
        options,
        tensor,
        iteration,
        skip_reduction,
        aux_dict=None,
    ):
        self.at_least_one_layer_fed = True
        buffer = self.buffers[(layer_name, tensor_name, options)]
        buffer.feed(tensor, iteration, aux_dict)
        buffer.skip_reduction = skip_reduction

    def log_stats(self):
        if not self._if_run_reduction():
            return {}

        output = {}
        for reduction_group, buffers in self.reduction_group_to_buffer.items():
            changed_buffers = [
                buffer
                for buffer in buffers
                if gather_along_first_dim(
                    buffer.modified.unsqueeze(0), process_group=reduction_group
                )[0].any()
            ]
            for buffer in changed_buffers:
                output.update(buffer.log())
        self.at_least_one_layer_fed = False
        return output


STATS_BUFFERS = StatsBuffers()
