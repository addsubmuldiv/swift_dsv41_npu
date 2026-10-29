# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Fused optimizers and multi-tensor kernels for Ascend NPU."""
from .fused_adam import FusedAdam
from .fused_sgd import FusedSGD
from .multi_tensor_apply import MultiTensorApply, multi_tensor_applier
from .multi_tensor_scale import multi_tensor_scale, multi_tensor_scale_tensor
from .multi_tensor_l2norm import multi_tensor_l2norm, multi_tensor_unscale_l2norm
from .multi_tensor_compute_scale import (
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)
