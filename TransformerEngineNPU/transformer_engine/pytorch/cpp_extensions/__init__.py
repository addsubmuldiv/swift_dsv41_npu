# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for c++ extensions.

Upstream mirror also star-imports ``transformer_engine_torch`` (the C++ extension
module) and ``fused_attn``; neither exists on NPU -- the C++ extension is CUDA-only
and attention does not route through here -- so only the Python GEMM wrappers are
re-exported. Call sites guarding on the missing symbols keep their current
behaviour.
"""

# Re-exports are intentional; each name is part of this package's public API.
# ruff: noqa: F401, F403
from .gemm import *
