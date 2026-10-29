# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Behavioral coverage for packed GroupedTensor storage on NPU."""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.constants import TensorUsage
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from utils import assert_close, dtype_tols, npu_available, to_cpu


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


def _npu_tensor(values: torch.Tensor) -> torch.Tensor:
    return values.to("npu")


class TestGroupedTensorPackedStorage:
    def test_packed_storage_splits_and_transposes_without_losing_values(self) -> None:
        raw = _npu_tensor(torch.arange(12, dtype=torch.float32).reshape(2, 2, 3))
        grouped = GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=2,
            tensor_shape=(2, 3),
            rowwise_data=raw,
        )

        split = grouped.split_into_quantized_tensors()
        assert len(split) == 2
        for index, member in enumerate(split):
            assert_close(to_cpu(member), to_cpu(raw[index]), **dtype_tols(torch.float32))

        transposed, scale = grouped.get_data(TensorUsage.RHS_TRANS)
        assert scale is None
        assert_close(to_cpu(transposed), to_cpu(raw.transpose(-1, -2)), **dtype_tols(torch.float32))
        assert GroupedTensor.as_dense_grouped_tensor(grouped, num_tensors=2) is grouped

    def test_copy_and_optimizer_update_change_packed_storage(self) -> None:
        grouped = GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=2,
            tensor_shape=(2, 3),
            rowwise_data=_npu_tensor(torch.zeros(2, 2, 3, dtype=torch.float32)),
        )
        replacement = _npu_tensor(torch.arange(12, dtype=torch.float32).reshape(2, 2, 3))

        grouped.copy_(replacement)
        grouped.add_(1.5)
        expected = replacement + 1.5
        restored = torch.stack(grouped.split_into_quantized_tensors())
        assert_close(to_cpu(restored), to_cpu(expected), **dtype_tols(torch.float32))

        zeros = torch.zeros_like(grouped)
        assert zeros.num_tensors == grouped.num_tensors
        assert_close(
            to_cpu(zeros.rowwise_data),
            to_cpu(torch.zeros_like(replacement)),
            **dtype_tols(torch.float32),
        )
