# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tensor classes for TransformerEngine Ascend PyTorch"""

from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
)
from .dummy_tensor import DummyQuantizer, DummyTensor
from .float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor
from .float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer, Float8Tensor
from .hif8_tensor import HIF8Quantizer
from .mxfp4_tensor import MXFP4Quantizer, MXFP4Tensor
from .mxfp8_tensor import MXFP832x32Quantizer, MXFP8Quantizer, MXFP8Tensor
from .w4a16_quantizer import FakeMXFP4Quantizer
from .w4a8_tensor import W4A8WeightQuantizer, W4A8WeightTensor
from .storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from .storage.float8_tensor_storage import Float8TensorStorage
from .storage.mxfp4_tensor_storage import MXFP4TensorStorage
from .storage.mxfp8_tensor_storage import MXFP8TensorStorage
from .storage.w4a8_tensor_storage import W4A8WeightTensorStorage
from .utils import (
    cast_master_weights_to_fp8,
    post_all_gather_processing,
    replace_raw_data,
)

__all__ = [
    "QuantizedTensorStorage",
    "QuantizedTensor",
    "Quantizer",
    "DummyQuantizer",
    "DummyTensor",
    "Float8Quantizer",
    "Float8CurrentScalingQuantizer",
    "HIF8Quantizer",
    "MXFP4Quantizer",
    "MXFP8Quantizer",
    "FakeMXFP4Quantizer",
    "MXFP832x32Quantizer",
    "W4A8WeightQuantizer",
    "Float8BlockQuantizer",
    "Float8TensorStorage",
    "MXFP4TensorStorage",
    "MXFP8TensorStorage",
    "W4A8WeightTensorStorage",
    "Float8BlockwiseQTensorStorage",
    "Float8Tensor",
    "MXFP4Tensor",
    "MXFP8Tensor",
    "W4A8WeightTensor",
    "Float8BlockwiseQTensor",
    "cast_master_weights_to_fp8",
    "post_all_gather_processing",
    "replace_raw_data",
]
