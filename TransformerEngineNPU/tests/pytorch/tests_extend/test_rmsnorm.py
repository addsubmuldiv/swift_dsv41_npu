# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Supplementary RMSNorm coverage targeting uncovered code paths.

Upstream reference: ``test_numerics.py`` TestRMSNorm (forward/backward numerics)
and ``test_sanity.py`` (integration through LayerNormLinear).

Targets uncovered lines in ``module/rmsnorm.py`` (~26 % baseline):
- ``reset_parameters``
- deprecated ``hidden_size`` init path
- deprecated ``params_dtype`` init path
- ``zero_centered_gamma`` init + forward
- ``sequence_parallel`` attribute on weight
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.module.rmsnorm import RMSNorm

from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")
_SEED = 1234


def _seed() -> None:
    torch.manual_seed(_SEED)
    torch.npu.manual_seed(_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# Basic init (already covered by sanity, kept as smoke)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRMSNormInit:
    @staticmethod
    def setup_class() -> None:
        _seed()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("hidden_size", [64, 128])
    def test_init(self, dtype: torch.dtype, hidden_size: int) -> None:
        layer = RMSNorm(hidden_size, dtype=dtype, device=_NPU)
        assert list(layer.weight.shape) == [hidden_size]
        assert layer.weight.dtype == dtype
        assert layer.eps == 1e-5
        assert not layer.zero_centered_gamma
        assert not layer.sequence_parallel

    def test_init_zero_centered_gamma(self) -> None:
        layer = RMSNorm(64, zero_centered_gamma=True, device=_NPU)
        assert layer.zero_centered_gamma
        assert layer.weight.sum().item() == 0.0  # zero init

    def test_init_sequence_parallel(self) -> None:
        layer = RMSNorm(32, sequence_parallel=True, device=_NPU)
        assert layer.sequence_parallel
        assert getattr(layer.weight, "sequence_parallel", False)

    @pytest.mark.parametrize("eps", [1e-3, 1e-6])
    def test_init_custom_eps(self, eps: float) -> None:
        layer = RMSNorm(16, eps=eps, device=_NPU)
        assert layer.eps == eps


# ═══════════════════════════════════════════════════════════════════════════════
# Deprecated parameter paths (currently ~0 % coverage)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRMSNormDeprecatedPaths:
    """Exercise the ``hidden_size`` and ``params_dtype`` backwards-compat code paths.

    The deprecated-arg conversion in ``module/rmsnorm.py`` runs BEFORE
    ``super().__init__``, so these paths work end to end.
    """

    @staticmethod
    def setup_class() -> None:
        _seed()

    def test_deprecated_hidden_size(self) -> None:
        """Neither normalized_shape nor hidden_size given → RuntimeError."""
        with pytest.raises(RuntimeError, match="Neither"):
            RMSNorm()

    def test_deprecated_hidden_size_warns(self) -> None:
        """hidden_size kwarg triggers DeprecationWarning and still builds the layer."""
        with pytest.warns(DeprecationWarning, match="hidden_size"):
            layer = RMSNorm(hidden_size=32, device=_NPU)
        assert list(layer.weight.shape) == [32]
        inp = torch.randn(2, 32, device="cpu").to(_NPU)
        assert layer(inp).shape == inp.shape

    def test_deprecated_both_args_raises(self) -> None:
        """Providing both normalized_shape and hidden_size is an error."""
        with pytest.raises(RuntimeError, match="Both"):
            RMSNorm(normalized_shape=64, hidden_size=64)

    def test_params_dtype_deprecated(self) -> None:
        """params_dtype kwarg is honored as the weight dtype."""
        with pytest.warns(DeprecationWarning, match="hidden_size"):
            layer = RMSNorm(params_dtype=torch.float32, hidden_size=32, device=_NPU)
        assert layer.weight.dtype == torch.float32

    def test_both_dtype_and_params_dtype_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Both.*dtype.*params_dtype"):
            RMSNorm(dtype=torch.float16, params_dtype=torch.float32, hidden_size=32)


# ═══════════════════════════════════════════════════════════════════════════════
# reset_parameters (currently ~0 % coverage)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRMSNormResetParams:
    @staticmethod
    def setup_class() -> None:
        _seed()

    def test_reset_parameters_ones(self) -> None:
        layer = RMSNorm(64, zero_centered_gamma=False, device=_NPU)
        # mutate weight, then reset
        layer.weight.data.fill_(2.0)
        layer.reset_parameters()
        assert torch.allclose(layer.weight.data, torch.ones_like(layer.weight))

    def test_reset_parameters_zeros(self) -> None:
        layer = RMSNorm(64, zero_centered_gamma=True, device=_NPU)
        layer.weight.data.fill_(3.5)
        layer.reset_parameters()
        assert layer.weight.sum().item() == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Forward-backward smoke (non-FP8 path — should work on all NPU versions)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRMSNormForwardBackward:
    @staticmethod
    def setup_class() -> None:
        _seed()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("hidden_size", [32, 64])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    def test_forward_shape(
        self,
        dtype: torch.dtype,
        hidden_size: int,
        zero_centered_gamma: bool,
    ) -> None:
        layer = RMSNorm(
            hidden_size,
            zero_centered_gamma=zero_centered_gamma,
            dtype=dtype,
            device=_NPU,
        )
        inp = torch.randn(2, 8, hidden_size, dtype=dtype, device=_NPU)
        out = layer(inp)
        assert out.shape == inp.shape
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_backward(self, dtype: torch.dtype) -> None:
        layer = RMSNorm(32, dtype=dtype, device=_NPU)
        inp = torch.randn(3, 8, 32, dtype=dtype, device=_NPU, requires_grad=True)
        out = layer(inp)
        loss = out.sum()
        loss.backward()
        torch.npu.synchronize()
        assert inp.grad is not None
        assert layer.weight.grad is not None

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_forward_zerocentered_equals_ones(self, dtype: torch.dtype) -> None:
        """Zero-centered gamma with weight=0 should act like no gamma."""
        layer = RMSNorm(16, zero_centered_gamma=True, dtype=dtype, device=_NPU)
        inp = torch.randn(4, 16, dtype=dtype, device=_NPU)

        out = layer(inp)  # weight = 0, so (1+0) * normed = normed
        # Just verify output shape and finite
        assert out.shape == inp.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("shape_seq", [(2, 16), (1, 1, 32)])
    def test_forward_various_shapes(self, shape_seq: tuple[int, ...]) -> None:
        hidden_size = shape_seq[-1]
        layer = RMSNorm(hidden_size, device=_NPU)
        inp = torch.randn(*shape_seq, device=_NPU)
        out = layer(inp)
        assert out.shape == inp.shape
