# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data (legacy compatibility shim).

Mirrors NVTE's deprecated ``transformer_engine.pytorch.float8_tensor`` module so
Megatron-Core's ``dist_checkpointing`` (which imports
``from transformer_engine.pytorch.float8_tensor import Float8Tensor``) resolves on
TENPU. The canonical path is ``transformer_engine.pytorch.tensor.float8_tensor``
or the top-level ``from transformer_engine.pytorch import Float8Tensor``.
"""

import warnings

from .tensor.float8_tensor import Float8Tensor

warnings.warn(
    "transformer_engine.pytorch.float8_tensor is deprecated and will be removed"
    " in a future release. Float8Tensor should be imported directly through "
    "`from transformer_engine.pytorch import Float8Tensor`",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Float8Tensor"]
