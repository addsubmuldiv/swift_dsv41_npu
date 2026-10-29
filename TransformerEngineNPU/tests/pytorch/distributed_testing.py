# Copyright (c) Microsoft Corporation.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed test framework for Ascend NPU tests."""

from __future__ import annotations

import inspect
import os
import time
from multiprocessing.pool import RUN

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from _pytest.outcomes import Skipped
from _pytest.fixtures import FixtureLookupError

from utils import npu_available, npu_device_count, get_free_port


def _resolve_world_size(world_size_attr, fixture_kwargs):
    """Return the concrete world_size from a class attribute or fixture kwargs."""
    if world_size_attr is not None:
        return world_size_attr
    # Dynamic: extract from fixture kwargs
    if "world_size" in fixture_kwargs:
        return fixture_kwargs["world_size"]
    if "sharding_dims" in fixture_kwargs:
        return fixture_kwargs["sharding_dims"][0]
    return 2


class DistributedTest:
    """Base class for distributed tests.

    Usage::

        class TestExample(DistributedTest):
            world_size = 4

            def test_something(self):
                rank = int(os.environ["LOCAL_RANK"])
                ...

    Features:

    * ``@pytest.mark.parametrize`` on test methods.
    * ``@pytest.mark.world_size(N)`` to override *world_size* per test.
    * ``class_tmpdir`` — a per-class temporary directory shared across ranks.

    Attributes:
        backend: Distributed backend (default ``"hccl"``).
        timeout: Seconds before a worker invocation is considered hung.
        world_size: Default world size.  Set to ``None`` when the world
            size should be resolved dynamically from fixture kwargs
            (e.g. a ``sharding_dims`` parametrization).
        reuse_dist_env: If ``True``, the process pool is reused across
            tests within the same class.
    """

    is_dist_test: bool = True
    world_size: int | None = 2
    backend: str = "hccl"
    timeout: int = 600
    reuse_dist_env: bool = True
    _pool_cache: dict[int, mp.Pool] = {}

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def class_tmpdir(cls, tmpdir_factory):
        """Per-class temporary directory shared across all ranks."""
        return tmpdir_factory.mktemp(cls.__name__)

    def run(self, **fixture_kwargs):
        """Execute the intercepted test method with resolved fixtures.

        Subclasses may override this when they need custom pre/post
        logic around the test call.
        """
        getattr(self, self._current_test_name)(**fixture_kwargs)

    def __call__(self, request=None):
        self._current_test_name = request.function.__name__  # pylint: disable=attribute-defined-outside-init
        test_func = getattr(self, self._current_test_name)
        self._fixture_kwargs = self._get_fixture_kwargs(request, test_func)  # pylint: disable=attribute-defined-outside-init

        if not npu_available():
            pytest.skip("only supported in accelerator environments.")

        world_size = _resolve_world_size(self.world_size, self._fixture_kwargs)

        # Allow per-test world_size override via pytest mark
        for mark in getattr(request.function, "pytestmark", []):
            if mark.name == "world_size":
                world_size = mark.args[0]
                break

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            self._launch_procs(procs)
            time.sleep(0.5)

    def _get_fixture_kwargs(self, request, func):
        if not request:
            return {}
        fixture_kwargs = {}
        params = inspect.getfullargspec(func).args
        params.remove("self")
        for p in params:
            try:
                fixture_kwargs[p] = request.getfixturevalue(p)
            except FixtureLookupError:
                pass  # test methods can have kwargs that are not fixtures
        return fixture_kwargs

    def _launch_procs(self, num_procs):
        if npu_device_count() < num_procs:
            pytest.skip(f"Not enough NPUs: {num_procs} required, {npu_device_count()} available")

        mp.set_start_method("forkserver", force=True)
        if self.reuse_dist_env:
            # Close other cached pools to avoid HCCL port conflicts
            # (all pools' workers try to bind to the same HCCL default port)
            for cached_procs, cached_pool in list(self._pool_cache.items()):
                if cached_procs != num_procs and cached_pool._state == RUN:
                    self._close_pool(cached_pool, cached_procs, force=True)
                    del self._pool_cache[cached_procs]
            if num_procs not in self._pool_cache:
                self._pool_cache[num_procs] = mp.Pool(processes=num_procs)
            pool = self._pool_cache[num_procs]
        else:
            pool = mp.Pool(processes=num_procs)
        master_port = get_free_port()

        args = [(local_rank, num_procs, master_port) for local_rank in range(num_procs)]
        skip_msgs_async = pool.starmap_async(self._dist_run, args)

        try:
            skip_msgs = skip_msgs_async.get(self.timeout)
        except mp.TimeoutError:
            pytest.exit("Distributed test timed out, exiting", returncode=1)

        self._close_pool(pool, num_procs)

        non_empty = [m for m in skip_msgs if m]
        if non_empty:
            assert len(set(non_empty)) == 1, (
                f"Multiple different skip messages received: {non_empty}"
            )
            pytest.skip(non_empty[0])

    def _dist_run(self, local_rank, num_procs, master_port):
        """Initialize the distributed environment and execute the user function."""
        if dist.is_initialized():
            local_rank = dist.get_rank()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(num_procs)
        os.environ["LOCAL_SIZE"] = str(num_procs)

        torch.npu.set_device(local_rank % torch.npu.device_count())
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                world_size=num_procs,
                rank=local_rank,
                init_method=f"tcp://127.0.0.1:{master_port}",
            )

        skip_msg = ""
        try:
            self.run(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                skip_msg = getattr(e, "msg", "")
            else:
                raise
        return skip_msg

    def _dist_destroy(self):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _close_pool(self, pool, num_procs, force=False):
        if force or not self.reuse_dist_env:
            pool.starmap(self._dist_destroy, [() for _ in range(num_procs)])
            pool.close()
            pool.join()
