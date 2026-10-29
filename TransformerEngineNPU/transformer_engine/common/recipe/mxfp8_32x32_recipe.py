# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass

from .mxfp8_recipe import MXFP8BlockScaling


@dataclass(repr=False)
class MXFP832x32BlockScaling(MXFP8BlockScaling):
    """MXFP8 recipe with one shared FP8 weight and 32x32 scaling blocks.

    Activations and gradients keep the standard directional MXFP8 layout.
    Weights are quantized with ``npu_dynamic_block_mx_quant`` and share a
    single FP8 payload between forward and data-gradient GEMMs. The operator
    returns an independent scale layout for each GEMM orientation.
    """

    @classmethod
    def mxfp8_32x32(cls):
        return True
