# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Supplementary LayerNorm coverage.  ``te.LayerNorm`` subclasses the fuser
``ops.LayerNorm`` (matching NVTE) with deprecated-arg handling and
sequence-parallel Megatron integration.  The norm op does not expose
``normalized_shape`` as an attribute (dimensions come from ``weight.shape``).
Currently only exercised indirectly through ``LayerNormLinear`` / ``LayerNormMLP``
tests.  Upstream: ``test_numerics.py`` TestLayerNorm, ``test_sanity.py``
LayerNormLinear tests.
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.module.layernorm import LayerNorm
from utils import assert_close, dtype_tols, make_reference_and_test_tensors, npu_available, to_cpu

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# =============================================================================
# Init
# =============================================================================


class TestLayerNormInit:
    @staticmethod
    def setup_method() -> None:
        _seed()

    @pytest.mark.parametrize("hidden_size", [32, 64])
    def test_init_basic(self, hidden_size: int) -> None:
        ln = LayerNorm(hidden_size, device=_NPU)
        # The fuser norm op does not expose `normalized_shape` (same as NVTE);
        # dimensions come from the parameter shapes.
        assert ln.weight.shape == (hidden_size,)
        assert ln.bias.shape == (hidden_size,)
        assert ln.eps == 1e-5
        assert ln.sequence_parallel is None

    def test_init_tuple_shape(self) -> None:
        ln = LayerNorm((4, 8), device=_NPU)
        assert ln.weight.shape == (4, 8)

    def test_init_custom_eps(self) -> None:
        ln = LayerNorm(16, eps=1e-3, device=_NPU)
        assert ln.eps == 1e-3

    def test_init_sequence_parallel_flag(self) -> None:
        ln = LayerNorm(16, sequence_parallel=True, device=_NPU)
        assert ln.sequence_parallel is True

    def test_init_zero_centered_gamma(self) -> None:
        # Real implementation supports zero_centered_gamma: weight init 0, math uses (1 + gamma).
        ln = LayerNorm(32, zero_centered_gamma=True, device=_NPU)
        assert ln.zero_centered_gamma is True
        assert torch.allclose(ln.weight.data, torch.zeros_like(ln.weight))
        assert torch.allclose(ln.bias.data, torch.zeros_like(ln.bias))

    def test_init_deprecated_hidden_size(self) -> None:
        with pytest.warns(DeprecationWarning, match="renamed"):
            ln = LayerNorm(hidden_size=32, device=_NPU)
        assert ln.weight.shape == (32,)

    def test_init_deprecated_params_dtype(self) -> None:
        ln = LayerNorm(params_dtype=torch.float32, hidden_size=32, device=_NPU)
        assert ln.weight.dtype == torch.float32

    def test_init_both_hidden_size_and_normalized_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Both.*hidden_size"):
            LayerNorm(normalized_shape=64, hidden_size=64, device=_NPU)

    def test_init_no_args_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Neither"):
            LayerNorm()

    def test_init_params_dtype_and_dtype_conflict(self) -> None:
        with pytest.raises(RuntimeError, match="Both.*dtype"):
            LayerNorm(params_dtype=torch.float32, dtype=torch.float16, hidden_size=32)


# =============================================================================
# Forward
# =============================================================================


class TestLayerNormForward:
    @staticmethod
    def setup_method() -> None:
        _seed()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("hidden_size", [32, 64])
    def test_forward_numerics(self, dtype: torch.dtype, hidden_size: int) -> None:
        ref, test = make_reference_and_test_tensors(
            (2, 8, hidden_size),
            test_dtype=dtype,
            test_device=_NPU,
        )
        ln = LayerNorm(hidden_size, dtype=dtype, device=_NPU)
        out = ln(test)
        expected = torch.nn.functional.layer_norm(
            test.float(),
            (hidden_size,),
            ln.weight.float(),
            ln.bias.float(),
            ln.eps,
        ).to(dtype)
        assert out.shape == test.shape
        assert out.dtype == dtype
        assert_close(to_cpu(out), to_cpu(expected), **dtype_tols(dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_zero_centered(self, dtype: torch.dtype) -> None:
        hidden_size = 32
        ref, test = make_reference_and_test_tensors(
            (2, 8, hidden_size),
            test_dtype=dtype,
            test_device=_NPU,
        )
        ln = LayerNorm(hidden_size, dtype=dtype, device=_NPU, zero_centered_gamma=True)
        out = ln(test)
        expected = torch.nn.functional.layer_norm(
            test.float(),
            (hidden_size,),
            ln.weight.float() + 1,
            ln.bias.float(),
            ln.eps,
        ).to(dtype)
        assert out.shape == test.shape
        assert_close(to_cpu(out), to_cpu(expected), **dtype_tols(dtype))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_backward(self, dtype: torch.dtype) -> None:
        ln = LayerNorm(32, dtype=dtype, device=_NPU)
        inp = torch.randn(3, 8, 32, dtype=dtype, device="cpu").to(device=_NPU)
        inp.requires_grad_(True)
        out = ln(inp)
        loss = out.sum()
        loss.backward()
        torch.npu.synchronize()
        assert inp.grad is not None
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None

    @pytest.mark.parametrize("shape", [(2, 16), (1, 1, 32), (3, 4, 5, 8)])
    def test_forward_various_shapes(self, shape: tuple[int, ...]) -> None:
        hidden_size = shape[-1]
        ln = LayerNorm(hidden_size, device=_NPU)
        inp = torch.randn(*shape, device="cpu").to(device=_NPU)
        out = ln(inp)
        assert out.shape == inp.shape
        assert torch.isfinite(out).all()


# =============================================================================
# Utility methods
# =============================================================================


class TestLayerNormUtils:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_fast_setattr(self) -> None:
        ln = LayerNorm(16, device=_NPU)
        ln.fast_setattr("_custom_flag", 42)
        assert ln._custom_flag == 42

    def test_reset_layer_norm_parameters_deprecated(self) -> None:
        ln = LayerNorm(16, device=_NPU)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            ln.reset_layer_norm_parameters()
        assert torch.allclose(ln.weight.data, torch.ones_like(ln.weight))
        assert ln.bias.data.sum().item() == 0.0

    def test_reset_parameters_defer_init_warns(self) -> None:
        ln = LayerNorm(16, device=_NPU)
        with pytest.warns(DeprecationWarning, match="defer_init"):
            ln.reset_parameters(defer_init=True)
        assert torch.allclose(ln.weight.data, torch.ones_like(ln.weight))

    def test_reset_parameters_defer_init_false(self) -> None:
        ln = LayerNorm(16, device=_NPU)
        with pytest.warns(DeprecationWarning, match="defer_init"):
            ln.reset_parameters(defer_init=False)
        assert torch.allclose(ln.weight.data, torch.ones_like(ln.weight))

    def test_reset_parameters_sets_sp_flag(self) -> None:
        ln = LayerNorm(16, sequence_parallel=True, device=_NPU)
        ln.reset_parameters()
        assert ln.weight.sequence_parallel is True
        assert ln.bias.sequence_parallel is True
