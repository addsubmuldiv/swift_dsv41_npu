# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations.

This operation-based API is experimental and subject to change.

"""

# Re-exports are intentional; each name is part of this package's public API.
# ruff: noqa: F401, F403
from .basic import (
    AddExtraInput,
    AllGather,
    AllReduce,
    Bias,
    ConstantScale,
    Identity,
    L2Normalization,
    MakeExtraOutput,
    Quantize,
    ReduceScatter,
    Reshape,
    RMSNorm,
)
from .fuser import (
    register_backward_fusion,
    register_forward_backward_fusion,
    register_forward_fusion,
)
from .op import BasicOperation, FusedOperation, FusibleOperation
from .sequential import Sequential
from . import fused
from .basic import *
