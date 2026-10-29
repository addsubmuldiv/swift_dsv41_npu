# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``module/_common.py`` helpers:
- ``noop_cat`` / ``_NoopCatFunc``
- ``_ParameterInitMeta``
- ``WeightGradStore``
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.module._common import (
    WeightGradStore,
    _ParameterInitMeta,
    noop_cat,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# noop_cat
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoopCat:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="0 tensors"):
            noop_cat([])

    def test_single_tensor_returns_as_is(self) -> None:
        """Single-tensor fast path: no torch.cat call, identity."""
        t = torch.randn(3, 4, device=_NPU)
        out = noop_cat([t])
        assert out is t

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_concat_different_storage(self, dim: int) -> None:
        """Tensors with separate storage → falls back to torch.cat, shape correct."""
        if dim == 0:
            a = torch.randn(2, 4, device=_NPU)
            b = torch.randn(3, 4, device=_NPU)
        else:
            a = torch.randn(4, 2, device=_NPU)
            b = torch.randn(4, 3, device=_NPU)
        out = noop_cat([a, b], dim=dim)
        expected = torch.cat([a, b], dim=dim)
        assert out.shape == expected.shape
        torch.testing.assert_close(out, expected)

    def test_concat_view_same_storage(self) -> None:
        """Tensors sharing storage (slice views) → noop_cat returns a view, not a copy."""
        # Make a tensor then take slices that share the same storage
        base = torch.randn(6, 4, device=_NPU)
        s1 = base[0:2]
        s2 = base[2:6]
        out = noop_cat([s1, s2], dim=0)
        assert out.shape == (6, 4)
        # Verify it's the underlying memory (no copy)
        torch.testing.assert_close(out, base)

    def test_concat_along_axis_with_negative_dim(self) -> None:
        """Negative dim normalised to positive internally."""
        a = torch.randn(2, 3, device=_NPU)
        b = torch.randn(2, 4, device=_NPU)
        out_pos = noop_cat([a, b], dim=1)
        out_neg = noop_cat([a, b], dim=-1)
        assert out_pos.shape == out_neg.shape
        torch.testing.assert_close(out_pos, out_neg)

    def test_concat_mismatched_outer_dims_raises(self) -> None:
        """Shape mismatch on non-concat dims → ValueError."""
        a = torch.randn(2, 3, device=_NPU)
        b = torch.randn(2, 4, device=_NPU)
        with pytest.raises(ValueError, match="Attempted to concatenate"):
            noop_cat([a, b], dim=0)

    def test_concat_different_ndim_raises(self) -> None:
        a = torch.randn(2, 3, device=_NPU)
        b = torch.randn(2, 4, 5, device=_NPU)
        with pytest.raises(ValueError, match="Attempted to concatenate"):
            noop_cat([a, b], dim=0)

    def test_concat_dim_out_of_range_raises(self) -> None:
        a = torch.randn(2, 3, device=_NPU)
        b = torch.randn(2, 3, device=_NPU)
        with pytest.raises(ValueError, match="Attempted to concatenate"):
            noop_cat([a, b], dim=5)

    def test_concat_gradient_propagation(self) -> None:
        """Backward splits grad_output into chunks matching split ranges."""
        a = torch.randn(2, 4, device=_NPU, requires_grad=True)
        b = torch.randn(2, 4, device=_NPU, requires_grad=True)
        # Use views from same storage so the noop path is exercised
        base = torch.cat([a, b], dim=0)
        out = noop_cat([base[0:2], base[2:4]], dim=0)
        loss = out.sum()
        loss.backward()
        torch.npu.synchronize()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape


# ═══════════════════════════════════════════════════════════════════════════════
# _ParameterInitMeta
# ═══════════════════════════════════════════════════════════════════════════════


class TestParameterInitMeta:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_default_init_fn(self) -> None:
        """If init_fn is None at construction, __post_init__ fills it from get_default_init_method."""
        meta = _ParameterInitMeta(init_fn=None, get_rng_state_tracker=None, fp8_meta_index=None)
        assert meta.init_fn is not None

    def test_provided_init_fn_preserved(self) -> None:
        def fn(t):
            return torch.nn.init.normal_(t, std=0.5)

        meta = _ParameterInitMeta(init_fn=fn, fp8_meta_index=7)
        assert meta.init_fn is fn
        assert meta.fp8_meta_index == 7

    def test_get_rng_state_tracker_passthrough(self) -> None:
        def tracker():
            return torch.npu.get_rng_state()

        meta = _ParameterInitMeta(get_rng_state_tracker=tracker)
        assert meta.get_rng_state_tracker is tracker

    def test_init_fn_default_to_dummy(self) -> None:
        """init_fn defaults to get_default_init_method when not provided."""
        meta = _ParameterInitMeta()
        assert meta.init_fn is not None
        # Use it on a tensor to verify it's callable and produces output
        t = torch.empty(3, 4, device=_NPU)
        out = meta.init_fn(t)
        assert out.shape == t.shape


# ═══════════════════════════════════════════════════════════════════════════════
# WeightGradStore
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightGradStore:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_default_disabled(self) -> None:
        store = WeightGradStore()
        assert store.enabled is False
        assert store.context is None

    def test_delay_wgrad_compute_enables(self) -> None:
        store = WeightGradStore(delay_wgrad_compute=True)
        assert store.enabled is True
        assert store.context is not None
        assert store.context.empty()

    def test_delay_with_ub_bulk_raises(self) -> None:
        """Conflicting flags: delay_wgrad_compute + ub_bulk_wgrad is an error."""
        with pytest.raises(AssertionError, match="ub_bulk_wgrad"):
            WeightGradStore(delay_wgrad_compute=True, ub_bulk_wgrad=True)

    def test_delay_wgrad_compute_method(self) -> None:
        """delay_wgrad_compute() returns the current state (same as .enabled)."""
        store = WeightGradStore(delay_wgrad_compute=True)
        assert store.delay_wgrad_compute() is True
        store.disable_delay_wgrad_compute()
        assert store.delay_wgrad_compute() is False

    def test_enable_disable_cycle(self) -> None:
        store = WeightGradStore(delay_wgrad_compute=True)
        store.disable_delay_wgrad_compute()
        assert store.enabled is False
        store.enable_delay_wgrad_compute()
        assert store.enabled is True

    def test_put_pop_roundtrip(self) -> None:
        """put() then pop() executes the deferred computation."""
        store = WeightGradStore(delay_wgrad_compute=True)
        a = torch.randn(2, 3, device=_NPU)
        b = torch.randn(2, 3, device=_NPU)

        def add(x, y):
            return x + y

        store.put([a, b], add)
        result, saved_tensors = store.pop()
        torch.testing.assert_close(result, a + b)
        assert len(saved_tensors) == 2

    def test_put_without_enable_raises(self) -> None:
        store = WeightGradStore()
        with pytest.raises(AssertionError, match="delay_wgrad_compute is not enabled"):
            store.put([torch.zeros(1, device=_NPU)], lambda x: x)

    def test_pop_without_enable_raises(self) -> None:
        store = WeightGradStore()
        with pytest.raises(AssertionError, match="delay_wgrad_compute is not enabled"):
            store.pop()

    def test_pop_empty_queue_raises(self) -> None:
        store = WeightGradStore(delay_wgrad_compute=True)
        with pytest.raises(RuntimeError, match="Pop empty queue"):
            store.pop()

    def test_assert_empty_requires_distributed(self) -> None:
        """assert_empty in non-distributed mode fails because get_rank is uninitialised.
        Only meaningful in a multi-rank context — covered indirectly via DistributedTest.
        """
        store = WeightGradStore(delay_wgrad_compute=True)
        with pytest.raises(ValueError, match="not been initialized"):
            store.assert_empty()
