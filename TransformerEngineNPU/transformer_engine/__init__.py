# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine NPU - A lightweight framework for Transformer models on Ascend NPUs"""

# common/pytorch 为 eager re-export,对齐 NVTE release_v2.17,不使用 __all__
# pylint: disable=unused-import
# ruff: noqa: F401

import transformer_engine.common

from . import pytorch

__version__ = "2.17.1"
__license__ = "See LICENSE for license information"
