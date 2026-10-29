# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Basic operations used by the PyTorch operation fuser."""

from .add_extra_input import AddExtraInput
from .all_gather import AllGather
from .all_reduce import AllReduce
from .bias import Bias
from .constant_scale import ConstantScale
from .identity import Identity
from .l2normalization import L2Normalization
from .layernorm import LayerNorm
from .make_extra_output import MakeExtraOutput
from .quantize import Quantize
from .reduce_scatter import ReduceScatter
from .reshape import Reshape
from .rmsnorm import RMSNorm

from .activation import (
    _ActivationOperation,
    GELU,
    SiLU,
    ReLU,
    QGELU,
    SReLU,
    GLU,
    GEGLU,
    ReGLU,
    QGEGLU,
    SReGLU,
    ClampedSwiGLU,
    ScaledSReLU,
)
from .swiglu import SwiGLU, ScaledSwiGLU, ScaledClampedQGeGLU

__all__ = [
    "AddExtraInput",
    "AllGather",
    "AllReduce",
    "Bias",
    "ConstantScale",
    "Identity",
    "L2Normalization",
    "LayerNorm",
    "MakeExtraOutput",
    "Quantize",
    "Reshape",
    "ReduceScatter",
    "_ActivationOperation",
    "GELU",
    "GEGLU",
    "GLU",
    "QGELU",
    "QGEGLU",
    "ReLU",
    "ReGLU",
    "SReLU",
    "SReGLU",
    "ScaledSReLU",
    "SiLU",
    "SwiGLU",
    "ScaledSwiGLU",
    "ScaledClampedQGeGLU",
    "ClampedSwiGLU",
    "RMSNorm",
]
