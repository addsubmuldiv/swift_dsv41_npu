# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Pytest fixtures for Transformer Engine debug tests."""

from pathlib import Path

import pytest


_DEBUG_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DEBUG_TEST_DIR.parents[2]


def pytest_addoption(parser):
    parser.addoption(
        "--feature_dirs",
        nargs="+",
        action="store",
        default=[str(_REPO_ROOT / "transformer_engine" / "debug" / "features")],
        help="List of debug feature directories.",
    )
    parser.addoption(
        "--configs_dir",
        action="store",
        default=str(_DEBUG_TEST_DIR / "test_configs"),
        type=str,
        help="Path to debug test configs.",
    )


@pytest.fixture
def feature_dirs(request):
    return request.config.getoption("--feature_dirs")


@pytest.fixture
def configs_dir(request):
    return request.config.getoption("--configs_dir")
