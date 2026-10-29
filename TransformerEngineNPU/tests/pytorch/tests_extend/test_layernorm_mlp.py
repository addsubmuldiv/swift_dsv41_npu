# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``module/layernorm_mlp.py`` non-FP8 helpers:
- ``set_tensor_model_parallel_attributes``
- ``_apply_norm`` LayerNorm and RMSNorm paths
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.module.layernorm_mlp import (
    LayerNormMLP,
    _apply_norm,
    set_tensor_model_parallel_attributes,
)
from utils import assert_close, dtype_tols, npu_available, to_cpu

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# set_tensor_model_parallel_attributes
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetTensorModelParallelAttributes:
    def test_set_all_attributes(self) -> None:
        t = torch.zeros(3, device=_NPU)
        set_tensor_model_parallel_attributes(t, is_parallel=True, dim=0, stride=1)
        assert t.tensor_model_parallel is True
        assert t.partition_dim == 0
        assert t.partition_stride == 1

    def test_already_set_raises(self) -> None:
        t = torch.zeros(3, device=_NPU)
        t.tensor_model_parallel = True
        with pytest.raises(AssertionError):
            set_tensor_model_parallel_attributes(t, is_parallel=False, dim=0, stride=1)


# ═══════════════════════════════════════════════════════════════════════════════
# _LayerNormMLPNonTensorArgs NamedTuple
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# _apply_norm
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyNorm:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_layernorm_path_returns_three(self) -> None:
        x = torch.randn(2, 8, device=_NPU)
        gamma = torch.ones(8, device=_NPU)
        beta = torch.zeros(8, device=_NPU)
        ln_out, mu, rsigma = _apply_norm(
            x, gamma, beta, eps=1e-5, normalization="LayerNorm", zero_centered_gamma=False
        )
        assert ln_out.shape == x.shape
        assert mu is not None
        assert rsigma is not None

    def test_rmsnorm_path(self) -> None:
        x = torch.randn(2, 8, device=_NPU)
        gamma = torch.ones(8, device=_NPU)
        ln_out, mu, rsigma = _apply_norm(
            x, gamma, None, eps=1e-5, normalization="RMSNorm", zero_centered_gamma=False
        )
        assert ln_out.shape == x.shape
        assert mu is None
        assert rsigma is not None

    def test_layernorm_zero_centered_gamma(self) -> None:
        x = torch.randn(2, 8, device=_NPU)
        gamma = torch.zeros(8, device=_NPU)
        ln_out, _, _ = _apply_norm(
            x, gamma, None, eps=1e-5, normalization="LayerNorm", zero_centered_gamma=True
        )
        assert ln_out.shape == x.shape

    def test_unknown_normalization_raises(self) -> None:
        x = torch.randn(2, 8, device=_NPU)
        gamma = torch.ones(8, device=_NPU)
        with pytest.raises(ValueError, match="Unknown normalization"):
            _apply_norm(
                x, gamma, None, eps=1e-5, normalization="BatchNorm", zero_centered_gamma=False
            )


class TestLayerNormMLPForward:
    @staticmethod
    def setup_method() -> None:
        _seed()

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_layernorm_mlp_forward_and_backward_matches_torch(self, normalization: str) -> None:
        module = LayerNormMLP(
            8,
            16,
            normalization=normalization,
            device=_NPU,
            params_dtype=torch.float32,
        )
        inp = torch.randn(2, 3, 8, device="cpu").to(_NPU).requires_grad_(True)

        actual = module(inp)
        if normalization == "LayerNorm":
            normalized = torch.nn.functional.layer_norm(
                inp,
                (8,),
                module.layer_norm_weight,
                module.layer_norm_bias,
                module.eps,
            )
        else:
            normalized = inp * torch.rsqrt(inp.square().mean(dim=-1, keepdim=True) + module.eps)
            normalized = normalized * module.layer_norm_weight
        expected = torch.nn.functional.linear(
            torch.nn.functional.gelu(
                torch.nn.functional.linear(normalized, module.fc1_weight, module.fc1_bias)
            ),
            module.fc2_weight,
            module.fc2_bias,
        )
        assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(torch.float32))

        actual.square().mean().backward()
        torch.npu.synchronize()
        assert inp.grad is not None
        assert module.layer_norm_weight.grad is not None
        assert module.fc1_weight.grad is not None
        assert module.fc2_weight.grad is not None

    def test_return_layernorm_output_matches_intermediate(self) -> None:
        module = LayerNormMLP(
            8,
            16,
            return_layernorm_output=True,
            device=_NPU,
            params_dtype=torch.float32,
        )
        inp = torch.randn(2, 3, 8, device="cpu").to(_NPU)

        output, layernorm_output = module(inp)
        expected_layernorm = torch.nn.functional.layer_norm(
            inp,
            (8,),
            module.layer_norm_weight,
            module.layer_norm_bias,
            module.eps,
        )
        expected_output = torch.nn.functional.linear(
            torch.nn.functional.gelu(
                torch.nn.functional.linear(
                    expected_layernorm,
                    module.fc1_weight,
                    module.fc1_bias,
                )
            ),
            module.fc2_weight,
            module.fc2_bias,
        )
        assert_close(
            to_cpu(layernorm_output), to_cpu(expected_layernorm), **dtype_tols(torch.float32)
        )
        assert_close(to_cpu(output), to_cpu(expected_output), **dtype_tols(torch.float32))
