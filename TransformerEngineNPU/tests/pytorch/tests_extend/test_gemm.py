# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Numerical coverage for dense GEMM helper contracts."""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.constants import TensorUsage
from transformer_engine.pytorch.cpp_extensions import general_gemm as cpp_general_gemm
from transformer_engine.pytorch.ops.gemm import (
    general_gemm,
    general_gemm_add,
    general_grouped_gemm,
    matmul_add,
    out_add_biases,
)
from utils import assert_close, dtype_tols, npu_available, to_cpu

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


@pytest.mark.parametrize(
    ("usage_a", "usage_b", "a_shape", "b_shape"),
    [
        (TensorUsage.LHS, TensorUsage.RHS, (2, 3), (3, 4)),
        (TensorUsage.LHS_TRANS, TensorUsage.RHS, (3, 2), (3, 4)),
        (TensorUsage.LHS, TensorUsage.RHS_TRANS, (2, 3), (4, 3)),
        (TensorUsage.LHS_TRANS, TensorUsage.RHS_TRANS, (3, 2), (4, 3)),
    ],
)
def test_general_gemm_matches_requested_transposes(usage_a, usage_b, a_shape, b_shape):
    a = torch.arange(1, 1 + a_shape[0] * a_shape[1], dtype=torch.float32).reshape(a_shape).to("npu")
    b = torch.arange(1, 1 + b_shape[0] * b_shape[1], dtype=torch.float32).reshape(b_shape).to("npu")
    bias = torch.arange(4, dtype=torch.float32).to("npu")

    actual = general_gemm(a, b, usage_a, usage_b, torch.float32, bias=bias)
    expected_a = a.t() if usage_a is TensorUsage.LHS_TRANS else a
    expected_b = b.t() if usage_b is TensorUsage.RHS_TRANS else b
    expected = torch.matmul(expected_a, expected_b) + bias
    assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(torch.float32))


def test_gemm_gradient_accumulators_match_torch_matmul():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3).to("npu")
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4).to("npu")
    main_grad = torch.ones(2, 4, dtype=torch.float32, device="npu")

    general_gemm_add(main_grad, a, b, TensorUsage.LHS, TensorUsage.RHS, torch.float32)
    expected_main_grad = torch.ones_like(main_grad) + torch.matmul(a, b)
    assert_close(to_cpu(main_grad), to_cpu(expected_main_grad), **dtype_tols(torch.float32))

    inp = torch.arange(6, dtype=torch.float32).reshape(2, 3).to("npu")
    grad_out = torch.arange(8, dtype=torch.float32).reshape(2, 4).to("npu")
    weight_grad = torch.zeros(4, 3, dtype=torch.float32, device="npu")
    matmul_add(weight_grad, inp, grad_out)
    assert_close(
        to_cpu(weight_grad), to_cpu(torch.matmul(grad_out.t(), inp)), **dtype_tols(torch.float32)
    )


def test_empty_gemm_inputs_leave_accumulators_unchanged():
    main_grad = torch.ones(2, 4, dtype=torch.float32, device="npu")
    empty = torch.empty(0, 3, dtype=torch.float32, device="npu")
    rhs = torch.empty(3, 4, dtype=torch.float32, device="npu")

    general_gemm_add(main_grad, empty, rhs, TensorUsage.LHS, TensorUsage.RHS, torch.float32)
    assert_close(
        to_cpu(main_grad), torch.ones(2, 4, dtype=torch.float64), **dtype_tols(torch.float32)
    )


def test_grouped_gemm_empty_path_and_bias_partitioning():
    group_split = torch.tensor([1, 2], dtype=torch.int64, device="npu")
    empty_input = torch.empty(3, 0, dtype=torch.float32, device="npu")
    empty_weight = torch.empty(4, 0, dtype=torch.float32, device="npu")

    empty_output = general_grouped_gemm(
        empty_weight,
        empty_input,
        group_split,
        layout="TN",
        out_dtype=torch.float32,
    )
    assert empty_output.shape == (2, 3, 4)
    assert torch.count_nonzero(empty_output).item() == 0

    output = torch.arange(12, dtype=torch.float32, device="npu").reshape(3, 4)
    biases = [torch.ones(4, device="npu"), torch.full((4,), 2.0, device="npu")]
    actual = out_add_biases(output, biases, group_split)
    expected = output.clone()
    expected[0] += 1.0
    expected[1:] += 2.0
    assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(torch.float32))


def test_cpp_general_gemm_keeps_fp32_output_with_bf16_bias():
    """Lower-precision bias must not downgrade an fp32 GEMM (MoE router scoring)."""
    weight = torch.randn(4, 3, dtype=torch.bfloat16, device="npu")
    inp = torch.randn(2, 3, dtype=torch.bfloat16, device="npu")
    bias = torch.randn(4, dtype=torch.bfloat16, device="npu")

    actual = cpp_general_gemm(weight, inp, out_dtype=torch.float32, layout="TN", bias=bias)[0]

    assert actual.dtype == torch.float32
    expected = inp.float() @ weight.float().t() + bias.float()
    assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(torch.float32))
