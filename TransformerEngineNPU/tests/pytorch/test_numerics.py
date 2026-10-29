# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Numerical accuracy tests migrated from Transformer Engine."""

from dataclasses import dataclass
from typing import List

import pytest
import torch
from torch import nn
from torch.nn import Parameter

from transformer_engine.pytorch import RMSNorm

from utils import dtype_tols, npu_available


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

seed = 1234


@dataclass(frozen=True)
class ModelConfig:
    """Minimal model configuration used by the RMSNorm accuracy test."""

    max_seqlen_q: int
    hidden_size: int


model_configs = {
    "126m": ModelConfig(max_seqlen_q=2048, hidden_size=12 * 64),
}

param_types = [torch.float32, torch.float16, torch.bfloat16]
batch_sizes = [1, 2]
all_boolean = [True, False]


def reset_rng_states() -> None:
    """Reset CPU and NPU random number generators."""
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def assert_allclose(
    l1: List[torch.Tensor],
    l2: List[torch.Tensor],
    atol: float = None,
    rtol: float = None,
) -> None:
    """Ensure two lists of tensors are numerically close."""
    assert len(l1) == len(l2), "Unequal number of outputs."
    for t1, t2 in zip(l1, l2):
        tolerances = dtype_tols(t2.dtype)
        if rtol is not None:
            tolerances["rtol"] = rtol
        if atol is not None:
            tolerances["atol"] = atol
        torch.testing.assert_close(t1, t2, **tolerances)


# Adapted from https://github.com/bzhangGo/rmsnorm/blob/c6691f20ec0af4128c8159c903071f7575404295/rmsnorm_torch.py
class TorchRMSNorm(nn.Module):
    """Plain PyTorch RMSNorm reference implementation."""

    def __init__(self, in_features, zero_centered_gamma, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.in_features = in_features
        self.zero_centered_gamma = zero_centered_gamma

        initial_value = torch.zeros(in_features) if zero_centered_gamma else torch.ones(in_features)
        self.weight = nn.Parameter(initial_value)
        self.register_parameter("weight", self.weight)

    def forward(self, x):
        norm_x2 = torch.sum(x.float() ** 2, dim=-1, keepdim=True)
        d_x = self.in_features

        rms_x2 = norm_x2 / d_x + self.eps
        r_rms_x = rms_x2 ** (-1.0 / 2)
        x_normed = x * r_rms_x

        w = self.weight.float()
        if self.zero_centered_gamma:
            w = 1 + w
        return (w * x_normed).to(x.dtype)


def _test_granular_accuracy(block, bs, dtype, config):
    reset_rng_states()
    inp_hidden_states = torch.randn(
        (config.max_seqlen_q, bs, config.hidden_size),
        dtype=dtype,
        device="npu",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    out = block(inp_hidden_states)
    out.sum().backward()

    torch.npu.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for parameter in block.parameters():
        if parameter.requires_grad:
            outputs.append(parameter.grad)
    return outputs


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("model", ["126m"])
@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-7])
@pytest.mark.parametrize("zero_centered_gamma", all_boolean)
def test_rmsnorm_accuracy(dtype, bs, model, eps, zero_centered_gamma):
    """Compare TENPU RMSNorm forward and backward numerics with PyTorch."""
    config = model_configs[model]

    te_rmsnorm = RMSNorm(
        config.hidden_size,
        eps=eps,
        params_dtype=dtype,
        zero_centered_gamma=zero_centered_gamma,
        device="npu",
    ).eval()

    torch_rmsnorm = (
        TorchRMSNorm(config.hidden_size, eps=eps, zero_centered_gamma=zero_centered_gamma)
        .to(dtype=dtype, device="npu")
        .eval()
    )

    # Share params
    with torch.no_grad():
        torch_rmsnorm.weight = Parameter(te_rmsnorm.weight.clone())

    te_outputs = _test_granular_accuracy(te_rmsnorm, bs, dtype, config)
    torch_outputs = _test_granular_accuracy(torch_rmsnorm, bs, dtype, config)

    atol = {
        torch.float32: 1e-7,
        torch.half: 2e-3,
        torch.bfloat16: 2e-2,
    }

    # Check output.
    assert_allclose([te_outputs[0]], [torch_outputs[0]], atol[dtype])

    atol[torch.float32] = 2e-3
    rtol = {
        torch.float32: 1.3e-6,
        torch.half: 1e-3,
        torch.bfloat16: 1.6e-2,
    }
    # Check gradients
    for te_output, torch_output in zip(te_outputs[1:], torch_outputs[1:]):
        assert_allclose([te_output], [torch_output], atol[dtype], rtol[dtype])
