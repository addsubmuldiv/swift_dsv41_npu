# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Tests for ``torch_all_reduce_double_dtype_bypass_wrapper``.

HCCL lacks fp64 collectives, so double ``all_reduce`` calls run on a float32
copy and are copied back (``Tensor.copy_`` promotes to float64 in place).
A recording fake stands in for the collective, keeping tests single-process.
"""

import pytest
import torch

from megatron_adaptor.patches.requirements.requirements import torch_all_reduce_double_dtype_bypass_wrapper


class _FakeHandle:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


class _ReduceRecorder:
    """Minimal ``all_reduce`` stand-in: adds 1.0 to the tensor in place."""

    def __init__(self, handle=None):
        self.handle = handle
        self.calls = []

    def __call__(self, tensor, *args, **kwargs):
        self.calls.append((tensor, args, kwargs))
        tensor.add_(1.0)
        return self.handle


class TestDoubleAllReduceWrapper:
    @pytest.mark.parametrize("dtype", [torch.double, torch.float], ids=["double", "non-double"])
    def test_wrapper_contract(self, dtype):
        """In-place reduction, handle forwarding, and unchanged args per path."""
        handle = _FakeHandle()
        recorder = _ReduceRecorder(handle=handle)
        wrapper = torch_all_reduce_double_dtype_bypass_wrapper(recorder)
        original = torch.zeros(2, dtype=dtype)

        result = wrapper(original, 123, group="g", async_op=True)

        passed_tensor, args, kwargs = recorder.calls[0]
        assert result is handle
        assert original.dtype == dtype
        assert torch.equal(original, torch.ones(2, dtype=dtype))
        assert args == (123,)
        assert kwargs == {"group": "g", "async_op": True}
        if dtype == torch.double:
            # reduced on a float32 copy, result copied back after completion
            assert passed_tensor.dtype == torch.float
            assert passed_tensor.data_ptr() != original.data_ptr()
            assert handle.wait_calls == 1
        else:
            # pass-through: no copy, no wait, caller owns the handle
            assert passed_tensor is original
            assert handle.wait_calls == 0

    def test_sync_call_returns_none_and_tensor_by_keyword(self):
        """Sync path (no handle) and ``tensor=`` keyword invocation."""
        recorder = _ReduceRecorder()
        wrapper = torch_all_reduce_double_dtype_bypass_wrapper(recorder)
        original = torch.zeros(2, dtype=torch.double)

        result = wrapper(tensor=original)

        assert result is None
        assert torch.equal(original, torch.ones(2, dtype=torch.double))
