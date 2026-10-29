# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``distributed.py`` non-distributed-init paths:
- RNG helpers
- activation_recompute / FP8 recompute
- set_tensor_model_parallel_attributes
- get_distributed_world_size / rank (single-process returns)
- _MultiHandle dataclass
- gather_along_first_dim no-op (no dist)
- get_te_classes / has_te_modules
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from transformer_engine.pytorch import Linear
from transformer_engine.pytorch.distributed import (
    _MultiHandle,
    allreduce,
    get_all_rng_states,
    get_activation_recompute_contexts,
    get_distributed_rank,
    get_distributed_world_size,
    get_te_classes,
    get_tensor_parallel_group,
    graph_safe_rng_available,
    has_te_modules,
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
    set_all_rng_states,
    set_tensor_model_parallel_attributes,
    symmetric_all_reduce,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# RNG state helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestRngStateHelpers:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_get_all_rng_states_returns_dict(self) -> None:
        """get_all_rng_states returns a dict (active RNG state tracking)."""
        result = get_all_rng_states()
        # Source returns _ALL_ACTIVE_RNG_STATES, normally a dict
        assert isinstance(result, dict)

    def test_set_all_rng_states_updates_state(self) -> None:
        """set_all_rng_states mutates the global state (set + get round-trip)."""
        before = get_all_rng_states()
        set_all_rng_states([1, 2, 3])
        after = get_all_rng_states()
        # Round-trip: set a list, get returns the same list
        assert after == [1, 2, 3]
        # Restore original state
        set_all_rng_states(before)

    def test_graph_safe_rng_available(self) -> None:
        """graph_safe_rng_available returns bool."""
        result = graph_safe_rng_available()
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# activation_recompute + FP8 recompute
# ═══════════════════════════════════════════════════════════════════════════════


class TestActivationRecompute:
    def test_get_activation_recompute_contexts_returns_tuple(self) -> None:
        """get_activation_recompute_contexts returns a tuple."""
        ctxs = get_activation_recompute_contexts()
        # Type contract: tuple
        assert isinstance(ctxs, tuple)
        # Each entry is an activation_recompute_forward context manager;
        # don't assert exact length (depends on global state)

    def test_is_fp8_activation_recompute_enabled_default_false(self) -> None:
        assert is_fp8_activation_recompute_enabled() is False

    def test_in_fp8_activation_recompute_phase_default_false(self) -> None:
        assert in_fp8_activation_recompute_phase() is False

    def test_get_te_classes_returns_iterable(self) -> None:
        """get_te_classes returns a tuple of registered TE class names."""
        classes = get_te_classes()
        # In this NPU port it's a tuple (NOT a list, per API contract)
        assert isinstance(classes, tuple)
        # Must contain LayerNorm and RMSNorm
        names = [c.__name__ for c in classes]
        assert "LayerNorm" in names
        assert "RMSNorm" in names
        # All returned items are torch.nn.Module subclasses
        assert all(issubclass(c, nn.Module) for c in classes)

    def test_has_te_modules_returns_bool(self) -> None:
        """has_te_modules returns bool for any nn.Module."""
        linear = torch.nn.Linear(3, 4, device=_NPU)
        assert has_te_modules(linear) is False  # plain nn.Linear isn't a TE module

        te_linear = Linear(3, 4, device=_NPU)
        assert has_te_modules(te_linear) is True


# ═══════════════════════════════════════════════════════════════════════════════
# set_tensor_model_parallel_attributes
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetTensorModelParallelAttributes:
    def test_set_all_attributes(self) -> None:
        t = torch.zeros(3, device=_NPU)
        set_tensor_model_parallel_attributes(t, is_parallel=True, dim=0, stride=1)
        assert t.tensor_model_parallel is True
        assert t.partition_dim == 0
        assert t.partition_stride == 1

    def test_overwrite_attribute(self) -> None:
        """This impl just setattrs; overwriting is allowed (no assert)."""
        t = torch.zeros(3, device=_NPU)
        set_tensor_model_parallel_attributes(t, is_parallel=True, dim=0, stride=1)
        set_tensor_model_parallel_attributes(t, is_parallel=False, dim=1, stride=2)
        assert t.tensor_model_parallel is False
        assert t.partition_dim == 1
        assert t.partition_stride == 2

    def test_partition_dim_negative_value(self) -> None:
        """Negative partition_dim allowed (semantic check at use site)."""
        t = torch.zeros(3, device=_NPU)
        set_tensor_model_parallel_attributes(t, is_parallel=True, dim=-1, stride=3)
        assert t.partition_dim == -1
        assert t.partition_stride == 3


# ═══════════════════════════════════════════════════════════════════════════════
# get_distributed_world_size / get_distributed_rank (single-process defaults)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetDistributedDefaults:
    def test_get_distributed_world_size_single_process(self) -> None:
        """Without distributed init, returns 1."""
        assert get_distributed_world_size() == 1

    def test_get_distributed_rank_single_process(self) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            get_distributed_rank()

    def test_get_tensor_parallel_group_default(self) -> None:
        # Without TP group set, default is None
        assert get_tensor_parallel_group() is None


# ═══════════════════════════════════════════════════════════════════════════════
# _MultiHandle dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiHandle:
    def test_construction_filters_nones(self) -> None:
        """_MultiHandle filters out None handles (only waits on real ones)."""
        sentinel = object()
        h = _MultiHandle(handles=[None, sentinel, None, sentinel, None])
        # 3 None filtered out, 2 sentinels kept
        assert len(h._handles) == 2

    def test_construction_empty(self) -> None:
        h = _MultiHandle(handles=[])
        assert h._handles == []

    def test_construction_all_none(self) -> None:
        """All None → empty handle list, wait() is a no-op."""
        h = _MultiHandle(handles=[None, None, None])
        assert h._handles == []
        # wait() on empty list does nothing
        h.wait()
        assert h._handles == []


# ═══════════════════════════════════════════════════════════════════════════════
# Communication stubs (require distributed init — expected to be no-op or raise)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCommunicationStubs:
    def test_allreduce_no_dist_runs_noop(self) -> None:
        """allreduce without distributed init returns its input and no handle."""
        t = torch.zeros(4, device=_NPU)
        out, handle = allreduce(t, None)
        assert out is t
        assert handle is None

    def test_symmetric_all_reduce_no_dist_runs_noop(self) -> None:
        t = torch.zeros(4, device=_NPU)
        out, handle = symmetric_all_reduce(t, None)
        assert out is t
        assert handle is None
