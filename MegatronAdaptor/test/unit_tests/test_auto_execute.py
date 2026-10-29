# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import pytest

from megatron_adaptor import auto_execute as auto_execute_module
from megatron_adaptor.auto_execute import AutoExecuteFunction, NoExecuteFunction


def _make_recorder(calls):
    @AutoExecuteFunction
    def record(value):
        calls.append(value)
        return value

    return record


class TestNoExecuteFunction:
    def test_suppresses_and_restores_auto_execution(self):
        calls = []
        record = _make_recorder(calls)

        with NoExecuteFunction():
            assert record("suppressed") is None

        assert record("executed") == "executed"
        assert calls == ["executed"]

    def test_nested_inner_exit_keeps_outer_suppression(self):
        calls = []
        record = _make_recorder(calls)

        with NoExecuteFunction():
            with NoExecuteFunction():
                assert record("inner") is None
            # inner exit must keep outer suppression active
            assert record("outer") is None

        assert record("after") == "after"
        assert calls == ["after"]

    def test_exit_restores_after_exception(self):
        calls = []
        record = _make_recorder(calls)

        with pytest.raises(RuntimeError, match="boom"):
            with NoExecuteFunction():
                assert record("suppressed") is None
                raise RuntimeError("boom")

        assert record("after") == "after"
        assert calls == ["after"]

    def test_exit_restores_user_preset_value(self):
        """__exit__ restores the ``__enter__`` snapshot, so a preset-off
        user keeps it off after the ``with`` block.
        """
        calls = []
        record = _make_recorder(calls)
        auto_execute_module._STATE["auto_execute"] = False  # user preset

        with NoExecuteFunction():
            assert record("suppressed") is None

        assert record("still_preset_off") is None
        assert not calls

        auto_execute_module._STATE["auto_execute"] = True  # restore for other tests
