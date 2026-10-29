# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""DistributedTest-based coverage for ``distributed.py`` TP/SP communication.

Conventions:
- Collectives are called with an explicit process group (``dist.group.WORLD``),
  never ``None`` — ``group=None`` silently short-circuits to a no-op in several
  wrappers, which would make the test vacuous.
- Inputs are rank-distinct (scaled by ``rank + 1``) and outputs are checked
  numerically against the expected collective result, not just by shape.
- Only ``NotImplementedError`` (a genuinely unsupported collective) is skipped;
  ``RuntimeError`` from a broken collective must fail the test.
"""

from __future__ import annotations

import os
import pytest
import torch
import torch.distributed as dist
from distributed_testing import DistributedTest
from transformer_engine.pytorch.distributed import (
    _all_gather_dense_along_dim,
    _all_gather_raw_along_dim,
    gather_split_1d_tensor,
    get_distributed_rank,
    get_distributed_world_size,
    get_hccl_comm_name,
    set_tensor_model_parallel_attributes,
    split_tensor_into_1d_equal_chunks,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


def _rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


class TestAllGather(DistributedTest):
    world_size = 2
    backend = "hccl"
    reuse_dist_env = False

    def test_all_gather_dense_along_dim(self) -> None:
        rank = _rank()
        torch.npu.set_device(rank)
        t = torch.tensor([[float(rank + 1)]], device="npu")
        try:
            out, _ = _all_gather_dense_along_dim(t, dist.group.WORLD, dim=0)
        except NotImplementedError as e:
            pytest.skip(f"all_gather_dense not implemented: {e}")
        assert out.shape == (2, 1)
        torch.testing.assert_close(
            out.cpu().reshape(-1),
            torch.tensor([1.0, 2.0], dtype=out.dtype),
        )

    def test_all_gather_raw_along_dim(self) -> None:
        rank = _rank()
        torch.npu.set_device(rank)
        t = torch.tensor([[float(rank + 1)]], device="npu")
        try:
            out, _ = _all_gather_raw_along_dim(t, dist.group.WORLD, dim=0)
        except NotImplementedError as e:
            pytest.skip(f"all_gather_raw not implemented: {e}")
        assert out.shape == (2, 1)
        torch.testing.assert_close(
            out.cpu().reshape(-1),
            torch.tensor([1.0, 2.0], dtype=out.dtype),
        )


class TestGetDistributed(DistributedTest):
    world_size = 2
    backend = "hccl"
    reuse_dist_env = False

    def test_get_distributed_world_size_with_dist(self) -> None:
        assert get_distributed_world_size() == 2

    def test_get_distributed_rank_with_dist(self) -> None:
        assert get_distributed_rank() == dist.get_rank()


class TestHcclCommName(DistributedTest):
    world_size = 2
    backend = "hccl"

    def test_get_hccl_comm_name_runs(self) -> None:
        rank = _rank()
        torch.npu.set_device(rank)
        try:
            name = get_hccl_comm_name(dist.group.WORLD)
        except NotImplementedError as e:
            pytest.skip(f"get_hccl_comm_name not supported: {e}")
        assert isinstance(name, str)
        assert len(name) > 0


class TestSplitAndGatherChunks(DistributedTest):
    world_size = 2
    backend = "hccl"
    reuse_dist_env = False

    def test_split_tensor_into_chunks(self) -> None:
        rank = _rank()
        torch.npu.set_device(rank)
        t = torch.arange(8, dtype=torch.float32, device="npu").reshape(4, 2)
        try:
            chunks = split_tensor_into_1d_equal_chunks(t, dist.group.WORLD)
        except NotImplementedError as e:
            pytest.skip(f"split_tensor not implemented: {e}")
        assert isinstance(chunks, torch.Tensor)
        assert chunks.numel() == t.numel() // 2

    def test_gather_split_1d_tensor(self) -> None:
        rank = _rank()
        torch.npu.set_device(rank)
        t = torch.arange(8, dtype=torch.float32, device="npu").reshape(4, 2)
        try:
            chunk = split_tensor_into_1d_equal_chunks(t, dist.group.WORLD)
            gathered = gather_split_1d_tensor(chunk, dist.group.WORLD)
        except NotImplementedError as e:
            pytest.skip(f"gather_split_1d not implemented: {e}")
        assert gathered.shape == (8,)
        torch.testing.assert_close(gathered.cpu(), t.cpu().reshape(-1))


class TestSetTensorModelParallelAttrsDist(DistributedTest):
    world_size = 2
    backend = "hccl"

    def test_set_tp_attrs_dist(self) -> None:
        rank = _rank()
        torch.npu.set_device(rank)
        t = torch.zeros(3, device="npu")
        set_tensor_model_parallel_attributes(t, is_parallel=True, dim=0, stride=1)
        assert t.tensor_model_parallel is True
        assert t.partition_dim == 0
        assert t.partition_stride == 1
