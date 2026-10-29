# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Offline tests for the NPUDataDumpFeature patch wrappers.

Fake msprobe modules are injected into sys.modules before importing the patch
module, so these tests run without msprobe installed.

Run:
    pytest test/unit_tests/patches/megatron/test_npu_datadump.py -v -s
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# 离线环境无 msprobe 依赖：在导入补丁模块前注入 fake 模块，保证单测可离线运行。
sys.modules['msprobe'] = MagicMock(name='msprobe')
sys.modules['msprobe.pytorch'] = MagicMock(name='msprobe.pytorch')

import megatron_adaptor  # noqa: F401,E402  确保补丁入口已执行
from megatron_adaptor.patches.megatron.npu_datadump import npu_datadump as npd_mod  # noqa: E402


class TestDumpStartWrapper:
    """dump_start_wrapper 的 start 时机与模型参数传递。"""

    def test_start_called_with_model_before_fn(self):
        calls = []
        fake_debugger = MagicMock()
        fake_debugger.start.side_effect = lambda **kwargs: calls.append('start')

        def train_step(*args, **kwargs):
            calls.append('fn')
            return 'step_ret'

        wrapper = npd_mod.dump_start_wrapper(train_step)
        with patch.object(npd_mod, 'MSTT_DEBUGGER', fake_debugger):
            ret = wrapper('forward_step_func', 'data_iterator', 'model_obj', 'optimizer')
        assert ret == 'step_ret'
        # start 必须先于原始 train_step 执行，且模型取第 3 个位置参数（args[2]）。
        assert calls == ['start', 'fn']
        fake_debugger.start.assert_called_once_with(model='model_obj')

    def test_wrapper_keeps_original_metadata(self):
        def train_step(*args, **kwargs):
            return None

        wrapper = npd_mod.dump_start_wrapper(train_step)
        assert wrapper.__name__ == 'train_step'
        assert wrapper.__wrapped__ is train_step


class TestDumpEndWrapper:
    """dump_end_wrapper 的 stop/step 时机（须在原始钩子之后）。"""

    def test_stop_and_step_called_after_fn(self):
        calls = []
        fake_debugger = MagicMock()
        fake_debugger.stop.side_effect = lambda: calls.append('stop')
        fake_debugger.step.side_effect = lambda: calls.append('step')

        def on_training_step_end(*args, **kwargs):
            calls.append('fn')

        wrapper = npd_mod.dump_end_wrapper(on_training_step_end)
        with patch.object(npd_mod, 'MSTT_DEBUGGER', fake_debugger):
            ret = wrapper()
        assert ret is None
        assert calls == ['fn', 'stop', 'step']


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short", "-s"])
