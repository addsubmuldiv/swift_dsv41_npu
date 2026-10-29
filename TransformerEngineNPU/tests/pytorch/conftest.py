# Copyright (c) Microsoft Corporation.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

"""pytest hooks for the distributed test framework."""

from multiprocessing.pool import RUN
import sys

import pytest


def _reset_transformer_engine_debug_state():
    """Keep nvdlfw_inspect debug state from leaking across tests."""
    debug_api = sys.modules.get("nvdlfw_inspect.api")
    if debug_api is None:
        return

    if getattr(debug_api, "DEBUG_MANAGER", None) is not None:
        end_debug = getattr(debug_api, "end_debug", None)
        if callable(end_debug):
            end_debug()

    debug_state = sys.modules.get("transformer_engine.debug.pytorch.debug_state")
    if debug_state is not None:
        debug_state.TEDebugState._reset()


def pytest_configure(config):
    config.option.durations = 0
    config.option.durations_min = 1
    config.option.verbose = True


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    """Intercept DistributedTest execution and launch via the framework."""
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # prevent double execution


def pytest_runtest_teardown(item, nextitem):
    """Clean up cached process pools when leaving a DistributedTest class."""
    if (
        item.cls is not None
        and getattr(item.cls, "is_dist_test", False)
        and getattr(item.cls, "reuse_dist_env", True)
        and (not nextitem or item.cls != nextitem.cls)
    ):
        dist_test_class = item.cls()
        if hasattr(dist_test_class, "_pool_cache"):
            for num_procs, pool in dist_test_class._pool_cache.items():
                if pool._state == RUN:
                    dist_test_class._close_pool(pool, num_procs, force=True)
            dist_test_class._pool_cache.clear()
    _reset_transformer_engine_debug_state()
