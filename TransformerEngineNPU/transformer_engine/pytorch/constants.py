# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Constants for TransformerEngine Ascend PyTorch API"""

from enum import Enum, IntEnum, unique

import torch
import torch_npu

# FP8 Dtype compatibility check
if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e5m2"):
    torch.float8_e4m3fn = torch.bfloat16
    torch.float8_e5m2 = torch.bfloat16


@unique
class NPUVersion(IntEnum):
    NONE = 0
    A2 = 2
    A3 = 3
    A5 = 5
    MAX_VERSION = 999


class Fp8Recipe(str, Enum):
    """FP8 recipe enumeration.

    Defines available FP8 quantization recipes.
    """

    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp4 = "mxfp4"
    mxfp8 = "mxfp8"
    blockwise = "blockwise"


@unique
class TensorUsage(str, Enum):
    LHS = "LN"
    LHS_TRANS = "LT"
    RHS = "RN"
    RHS_TRANS = "RT"


USAGE_WITH_TRANS = (TensorUsage.LHS_TRANS, TensorUsage.RHS_TRANS)
USAGE_WITHOUT_TRANS = (TensorUsage.LHS, TensorUsage.RHS)


@unique
class ParallelMode(str, Enum):
    COLUMN = "column"
    ROW = "row"


class QuantDtype:
    def __init__(self, fwd: torch.dtype, bwd: torch.dtype):
        self.fwd = fwd
        self.bwd = bwd
        if self.fwd == torch_npu.hifloat8:
            self.mm_kwargs = {"x1_dtype": self.fwd, "x2_dtype": self.fwd}
            self.gmm_kwargs = {"x_dtype": self.fwd, "weight_dtype": self.fwd}
        else:
            self.mm_kwargs = {}
            self.gmm_kwargs = {}


# FP8 tensor indices
class FP8FwdTensorIdx(IntEnum):
    """FP8 tensor indices for forward pass."""

    GEMM1_INPUT = 0
    GEMM1_WEIGHT = 1
    GEMM1_OUTPUT = 2
    GEMM2_INPUT = 3
    GEMM2_WEIGHT = 4
    GEMM2_OUTPUT = 5
    GEMM3_INPUT = 6
    GEMM3_WEIGHT = 7
    GEMM3_OUTPUT = 8


class FP8BwdTensorIdx(IntEnum):
    """FP8 tensor indices for backward pass."""

    GRAD_OUTPUT1 = 0
    GRAD_INPUT1 = 1
    GRAD_WEIGHT1 = 2
    GRAD_OUTPUT2 = 3
    GRAD_INPUT2 = 4
    GRAD_WEIGHT2 = 5
    GRAD_OUTPUT3 = 6
    GRAD_INPUT3 = 7
    GRAD_WEIGHT3 = 8


# Gemm parallel modes
GemmParallelModes = ("row", "column", None)

# Attention types
AttnTypes = ("self", "cross")

# Attention bias types
AttnBiasTypes = ("pre_scale_bias", "post_scale_bias", "no_bias", "alibi")

# Distributed group type

MXFP4_BLOCK_SCALING_SIZE = 32
MXFP8_BLOCK_SCALING_SIZE = 32
dist_group_type = torch.distributed.ProcessGroup


__all__ = [
    "FP8FwdTensorIdx",
    "FP8BwdTensorIdx",
    "GemmParallelModes",
    "AttnTypes",
    "AttnBiasTypes",
    "dist_group_type",
]
