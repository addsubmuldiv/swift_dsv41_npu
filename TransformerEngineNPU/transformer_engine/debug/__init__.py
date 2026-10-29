# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

# See LICENSE for license information.

"""Top-level package for numerical debugging."""

# ruff: noqa: F401, E402
try:
    from . import pytorch
    from .pytorch.debug_state import set_weight_tensor_tp_group_reduce
except ImportError:
    pass
