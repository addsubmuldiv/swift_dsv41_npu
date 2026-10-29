# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression test for the ``profiler-default`` feature.

This feature wraps ``megatron.core.parallel_state.get_nccl_options`` so that
every parallel group's pg_options carries ``hccl_config={"group_name": ...}``.
The HCCL communicator picks that name up when it is created, which lets msprof
traces identify which parallel domain (dp/tp/pp/...) each communication group
belongs to.

Run:
    pytest test/unit_tests/patches/megatron/test_profiler_default.py -v -s
"""

import os
from datetime import timedelta

import pytest
import torch
import torch_npu  # noqa: F401

import megatron_adaptor  # noqa: F401  importing applies the MA patches
from megatron.core import parallel_state as ps


class TestProfilerDefault:
    def test_created_groups_carry_group_name(self):
        if not torch.npu.is_available() or torch.npu.device_count() < 1:
            pytest.skip("requires at least 1 NPU")

        # 1. single-rank hccl world
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29611")
        torch.npu.set_device(0)
        torch.distributed.init_process_group(backend="hccl", rank=0, world_size=1, timeout=timedelta(minutes=10))
        try:
            # 2. let megatron build its parallel groups. Internally it calls
            #    get_nccl_options(...) per group; the profiler-default wrapper
            #    injects hccl_config={"group_name": <pg_name>} there.
            ps.initialize_model_parallel()

            # 3. read the name back from created groups. Megatron hands out
            #    torch's outer ProcessGroup wrapper; hccl_config lives on the
            #    inner HCCL backend object -- the same way the torch_npu
            #    profiler reads it.
            checks = [
                ("tp", ps.get_tensor_model_parallel_group()),
                ("pp", ps.get_pipeline_model_parallel_group()),
                ("dp", ps.get_data_parallel_group()),
            ]
            for expect_name, group in checks:
                hccl_pg = group._get_backend(torch.device("npu"))
                hccl_config = hccl_pg.options.hccl_config
                assert hccl_config.get("group_name") == expect_name, (
                    f"{expect_name} group lost its group_name during group creation: hccl_config={hccl_config}"
                )
        finally:
            # 4. cleanup for other tests in the same process
            ps.destroy_model_parallel()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
