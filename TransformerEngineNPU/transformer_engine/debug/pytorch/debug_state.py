# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Manage TransformerEngine debug state."""

import sys


class TEDebugState:
    """Global state shared by debug-enabled TE modules."""

    layer_count = 1
    layers_initialized = {}
    weight_tensor_tp_group_reduce = True
    debug_enabled = None

    @classmethod
    def initialize(cls):
        """Check whether nvdlfw_inspect has an active debug manager."""
        if "nvdlfw_inspect" not in sys.modules:
            return

        import nvdlfw_inspect.api as debug_api

        if cls.debug_enabled is False and debug_api.DEBUG_MANAGER is not None:
            raise RuntimeError(
                "[nv_dlfw_inspect] nvdlfw_inspect should be initialized before "
                "the first TE module is initialized"
            )
        cls.debug_enabled = debug_api.DEBUG_MANAGER is not None

    @classmethod
    def _reset(cls):
        """Reset debug state."""
        # pylint: disable=cyclic-import
        from ..features.utils.stats_buffer import STATS_BUFFERS

        STATS_BUFFERS.reset()
        cls.debug_enabled = None
        cls.layers_initialized.clear()

    @classmethod
    def get_layer_count(cls):
        """Return the next default layer index."""
        layer_count = cls.layer_count
        cls.layer_count += 1
        return layer_count

    @classmethod
    def set_weight_tensor_tp_group_reduce(cls, enabled):
        """Set weight tensor reduction mode for stats features."""
        cls.weight_tensor_tp_group_reduce = enabled

    @classmethod
    def get_iteration(cls):
        """Return the current nvdlfw_inspect iteration."""
        import nvdlfw_inspect.api as debug_api

        return debug_api.DEBUG_MANAGER._trainer_iteration_count


def set_weight_tensor_tp_group_reduce(enabled):
    """Set weight tensor reduction mode for stats features."""
    TEDebugState.set_weight_tensor_tp_group_reduce(enabled)
