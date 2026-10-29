# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""B-class UT: ops/basic/ comm wrappers via DistributedTest."""

from __future__ import annotations

import os
import pytest
import torch
import torch.distributed as dist
from distributed_testing import DistributedTest
from transformer_engine.pytorch.ops.basic.all_gather import AllGather
from transformer_engine.pytorch.ops.basic.all_reduce import AllReduce
from transformer_engine.pytorch.ops.basic.reduce_scatter import ReduceScatter
from transformer_engine.pytorch.ops.op import OperationContext
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


def _rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


class TestAllGather(DistributedTest):
    world_size = 2
    backend = "hccl"
    reuse_dist_env = False

    def test_construct_and_shape(self) -> None:
        rank = _rank()
        world_size = _world_size()
        torch.npu.set_device(rank)
        op = AllGather(process_group=dist.group.WORLD)
        x = torch.ones(2, 4, device="npu") * (rank + 1.0)
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        assert out.shape[0] == x.shape[0] * world_size
        assert out.shape[1:] == x.shape[1:]
        expected = torch.cat(
            [torch.ones_like(x) * (r + 1.0) for r in range(world_size)],
            dim=0,
        )
        torch.testing.assert_close(out.cpu(), expected.cpu())


class TestAllReduce(DistributedTest):
    world_size = 2
    backend = "hccl"
    reuse_dist_env = False

    def test_construct_and_forward(self) -> None:
        rank = _rank()
        world_size = _world_size()
        torch.npu.set_device(rank)
        op = AllReduce(process_group=dist.group.WORLD)
        x = torch.ones(2, 4, device="npu") * (rank + 1.0)
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        assert out.shape == x.shape
        expected_value = sum(range(1, world_size + 1))
        torch.testing.assert_close(out.cpu(), torch.full_like(x.cpu(), expected_value))


class TestReduceScatter(DistributedTest):
    world_size = 2
    backend = "hccl"
    reuse_dist_env = False

    def test_construct_and_shape(self) -> None:
        rank = _rank()
        world_size = _world_size()
        torch.npu.set_device(rank)
        op = ReduceScatter(process_group=dist.group.WORLD)
        x = torch.ones(4, 8, device="npu") * (rank + 1.0)
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        assert out.shape[1:] == x.shape[1:]
        assert out.shape[0] == x.shape[0] // world_size
        expected_value = sum(range(1, world_size + 1))
        torch.testing.assert_close(out.cpu(), torch.full_like(out.cpu(), expected_value))

    def test_backward_rank_distinct_grad_out(self) -> None:
        rank = _rank()
        world_size = _world_size()
        torch.npu.set_device(rank)
        op = ReduceScatter(process_group=dist.group.WORLD)
        x = torch.ones(4, 8, device="npu")
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        # Rank-distinct grad_out: an all-ones grad_out would make the all-gather
        # below trivially true even for a broken (e.g. no-op) backward.
        grad_out = torch.ones_like(out) * (rank + 1.0)
        grad_in, extras = op.op_backward(ctx, grad_out)
        assert grad_in.shape == x.shape
        assert not extras
        # ReduceScatter backward all-gathers grad_out along dim 0, so grad_in is the
        # rank-ordered concatenation of every rank's grad_out chunk:
        # grad_in[r*chunk : (r+1)*chunk] == r + 1.0
        chunk = x.shape[0] // world_size
        expected = torch.cat(
            [torch.full((chunk, x.shape[1]), float(r + 1)) for r in range(world_size)],
            dim=0,
        )
        torch.testing.assert_close(grad_in.cpu(), expected)
