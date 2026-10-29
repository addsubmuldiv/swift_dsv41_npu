# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass, replace
from typing import Optional

from transformer_engine.common.recipe.base import (
    MMParams,
    QParams,
    Recipe,
    _default_backward_override,
)


@dataclass(repr=False)
class MXFP4BlockScaling(Recipe):
    """
    Use MXFP4 block scaling for W4A4 GEMMs.

    The recipe follows the 4-bit block-scaling recipe shape used by NVFP4,
    while the NPU tensor/operator skeleton is aligned with the existing MXFP8
    implementation. Activations, weights, and gradients all use MXFP4 views.

    Notes
    -----
    The ``with_rht`` flag controls the Random Hadamard Transform for the
    **activations and gradients** only. Weights always have RHT disabled
    (``fp4_quant_fwd_weight.random_hadamard_transform = False``) because
    weight tensors are typically already well-conditioned and applying RHT
    on the weight side hurts GEMM numerics relative to the reference MXFP4
    kernels.

    ``with_rht`` is consumed exactly once in ``__post_init__`` to propagate
    into the ``fp4_quant_*`` ``QParams``. Mutating ``with_rht`` after
    construction has no effect; construct a new recipe instance to change
    the RHT setting.
    """

    margin: int = 0
    block_size: int = 32
    with_rht: bool = True

    fp4_quant_fwd_inp: QParams = QParams(power_2_scale=True)
    fp4_quant_fwd_weight: QParams = QParams(power_2_scale=True)
    fp4_quant_bwd_grad: QParams = QParams(power_2_scale=True)

    fp4_gemm_fprop: MMParams = MMParams(use_split_accumulator=True)
    fp4_gemm_dgrad: MMParams = MMParams(use_split_accumulator=True)
    fp4_gemm_wgrad: MMParams = MMParams(use_split_accumulator=True)

    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    @classmethod
    def mxfp4(cls):
        return True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.block_size != 32:
            raise ValueError("MXFP4BlockScaling currently supports block_size=32 only")
        self.fp4_quant_fwd_inp = replace(
            self.fp4_quant_fwd_inp,
            random_hadamard_transform=self.with_rht,
        )
        self.fp4_quant_fwd_weight = replace(
            self.fp4_quant_fwd_weight,
            random_hadamard_transform=False,
        )
        self.fp4_quant_bwd_grad = replace(
            self.fp4_quant_bwd_grad,
            random_hadamard_transform=self.with_rht,
        )

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"margin={self.margin}, "
            f"block_size={self.block_size}, "
            f"with_rht={self.with_rht}, "
            f"fp4_quant_fwd_inp={self.fp4_quant_fwd_inp}, "
            f"fp4_quant_fwd_weight={self.fp4_quant_fwd_weight}, "
            f"fp4_quant_bwd_grad={self.fp4_quant_bwd_grad}, "
            f"fp4_gemm_fprop={self.fp4_gemm_fprop}, "
            f"fp4_gemm_dgrad={self.fp4_gemm_dgrad}, "
            f"fp4_gemm_wgrad={self.fp4_gemm_wgrad}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}, "
            f"backward_override={self.backward_override}"
        )
