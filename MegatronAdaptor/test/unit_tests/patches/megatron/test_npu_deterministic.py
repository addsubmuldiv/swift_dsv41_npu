# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Tests for the ``npu_deterministic`` patch module.

The patch decorates ``megatron.training.initialize._set_random_seed``:
after the original seed initialization completes, ``extend_seed_all`` applies
the NPU deterministic configuration (HCCL_DETERMINISTIC / CLOSE_MATMUL_K_SHIFT /
PYTHONHASHSEED env vars, PyTorch deterministic algorithms, torch_npu RNG seeds).

This test verifies offline (with torch/torch_npu calls mocked, so no global
deterministic mode is actually turned on and no NPU device is required):
  1. ``extend_seed_all`` writes exactly the expected env vars and seed calls.
  2. The wrapper runs the original seed function FIRST and ``extend_seed_all``
     AFTER it, passing the un-offset seed through.

Run:
    pytest test/unit_tests/patches/megatron/test_npu_deterministic.py -v -s
"""

import os
from unittest import mock

import megatron_adaptor  # noqa: F401
from megatron_adaptor.patches.megatron import npu_deterministic as npu_det


class TestExtendSeedAll:
    def test_env_vars_and_seed_calls(self):
        saved = {key: os.environ.get(key) for key in ('HCCL_DETERMINISTIC', 'CLOSE_MATMUL_K_SHIFT', 'PYTHONHASHSEED')}
        try:
            with (
                mock.patch.object(npu_det.torch, 'use_deterministic_algorithms') as mock_det,
                mock.patch.object(npu_det.torch_npu.npu, 'manual_seed_all') as mock_seed_all,
                mock.patch.object(npu_det.torch_npu.npu, 'manual_seed') as mock_seed,
            ):
                npu_det.extend_seed_all(seed=1234)

            assert os.environ['HCCL_DETERMINISTIC'] == 'True'
            assert os.environ['CLOSE_MATMUL_K_SHIFT'] == '1'
            assert os.environ['PYTHONHASHSEED'] == '1234'
            mock_det.assert_called_once_with(True)
            mock_seed_all.assert_called_once_with(1234)
            mock_seed.assert_called_once_with(1234)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestNpuDeterministicWrapper:
    def test_original_seed_init_runs_before_extend(self):
        calls = []

        def fake_set_random_seed(seed, *args, **kwargs):
            calls.append(('original', seed, args, kwargs))

        with mock.patch.object(npu_det, 'extend_seed_all', side_effect=lambda seed: calls.append(('extend', seed))):
            wrapped = npu_det.npu_deterministic_wrapper(fake_set_random_seed)
            wrapped(1234, False, False, False, use_cudagraphable_rng=True)

        assert len(calls) == 2
        assert calls[0] == (('original', 1234, (False, False, False), {'use_cudagraphable_rng': True}))
        assert calls[1] == ('extend', 1234)

    def test_wrapper_keeps_original_metadata(self):
        def _set_random_seed(seed_):
            pass

        wrapped = npu_det.npu_deterministic_wrapper(_set_random_seed)
        assert wrapped.__name__ == '_set_random_seed'


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short", "-s"])
