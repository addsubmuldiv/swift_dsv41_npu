# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Regression test for the ``preload_tensors`` patch.

During async checkpoint save, ``preload_tensors`` first "stages" a copy of the
tensors to be saved, then a background worker writes them to disk. The upstream
implementation returns the original object for tensors already on CPU (no copy),
so when a subsequent training step updates that tensor in place, it also mutates
the data the background worker is writing -- corrupting the checkpoint.

The patched ``preload_tensors`` ``clone()``s CPU tensors, so the staged data is
an independent snapshot and later mutations of the source tensor do not affect it.

This test verifies exactly that: after staging, mutating the source tensor must
leave the staged copy unchanged.

Run:
    pytest test/unit_tests/patches/megatron/test_preload_tensors.py -v -s
"""

import torch

# Importing applies the NPU patches: if the preload_tensors patch is registered,
# it replaces FileSystemWriterAsync.preload_tensors (upstream implementation before the patch).
import megatron_adaptor  # noqa: F401
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
)


class TestPreloadTensors:
    def test_cpu_tensor_is_snapshotted(self):
        # Source tensor: values [0, 1, 2, ..., 7]
        src = torch.arange(8, dtype=torch.float32)

        # Build the write_buckets argument for preload_tensors.
        write_buckets = [("shard.distcp", "key0", ([], [("weight", src)]))]

        # Stage. non_blocking=False keeps the pure-CPU path (no device sync).
        staged = FileSystemWriterAsync.preload_tensors(write_buckets, non_blocking=False)

        # Extract the staged tensor from the same structure.
        staged_tensor = staged[0][2][1][0][1]

        # Simulate "after staging, while the background is still writing, the next
        # training/optimizer step updates the source tensor in place".
        src.mul_(-1.0)  # src becomes [0, -1, -2, ..., -7]

        # Patched -> staged is a clone, unaffected by the src mutation, still [0..7], assertion passes.
        # Unpatched -> staged is src itself, becomes [0,-1,..,-7], assertion fails (reproduces the bug).
        assert torch.equal(staged_tensor, torch.arange(8, dtype=torch.float32)), (
            "The staged CPU tensor was affected by the in-place mutation of the source tensor; "
            "this means preload_tensors did not snapshot (clone) the CPU tensor, "
            "so async save would write corrupted data."
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short", "-s"])
