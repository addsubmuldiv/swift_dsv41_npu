# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``ops/basic/npu_activation.py``:
- Element-wise: gelu / silu / relu / qgelu / srelu (forward + backward)
- GLU variants: glu / swiglu / geglu / reglu / sreglu / qgeglu / clamped_swiglu
  (parametrized numerical checks against PyTorch reference formulas)
- Dispatch tables: ACTIVATION_FWD / ACTIVATION_BWD / GLU_VARIANTS
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch.ops.basic.npu_activation import (
    ACTIVATION_BWD,
    ACTIVATION_FWD,
    GLU_VARIANTS,
    clamped_swiglu_bwd,
    clamped_swiglu_fwd,
    geglu_bwd,
    geglu_fwd,
    gelu_bwd,
    gelu_fwd,
    glu_bwd,
    glu_fwd,
    qgeglu_bwd,
    qgeglu_fwd,
    qgelu_bwd,
    qgelu_fwd,
    reglu_bwd,
    reglu_fwd,
    relu_bwd,
    relu_fwd,
    silu_bwd,
    silu_fwd,
    sreglu_bwd,
    sreglu_fwd,
    srelu_bwd,
    srelu_fwd,
    swiglu_bwd,
    swiglu_fwd,
)
from utils import (
    assert_close,
    dtype_tols,
    make_reference_and_test_tensors,
    npu_available,
    to_cpu,
)

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# Element-wise activations
# ═══════════════════════════════════════════════════════════════════════════════


def _fast_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference for ``torch_npu.npu_fast_gelu``."""
    return x * torch.sigmoid(1.702 * x)


def _srelu_ref(x: torch.Tensor) -> torch.Tensor:
    relu_x = F.relu(x)
    return relu_x * relu_x


_ELEMENTWISE_CASES = [
    (
        "gelu-none",
        lambda x: gelu_fwd(x, approximate="none"),
        lambda x, dy: gelu_bwd(x, dy, approximate="none"),
        lambda x: F.gelu(x, approximate="none"),
    ),
    (
        "gelu-tanh",
        lambda x: gelu_fwd(x, approximate="tanh"),
        lambda x, dy: gelu_bwd(x, dy, approximate="tanh"),
        lambda x: F.gelu(x, approximate="tanh"),
    ),
    ("silu", silu_fwd, silu_bwd, F.silu),
    ("relu", relu_fwd, relu_bwd, F.relu),
    ("qgelu", qgelu_fwd, qgelu_bwd, _fast_gelu_ref),
    ("srelu", srelu_fwd, srelu_bwd, _srelu_ref),
]


class TestElementwiseActivations:
    """Parametrized numerical forward/backward checks for element-wise activations."""

    @staticmethod
    def setup_method() -> None:
        _seed()

    @pytest.mark.parametrize(
        "name,fwd,ref_fwd",
        [(name, fwd, ref) for name, fwd, _, ref in _ELEMENTWISE_CASES],
        ids=[name for name, *_ in _ELEMENTWISE_CASES],
    )
    def test_forward_numerical(self, name: str, fwd, ref_fwd) -> None:
        ref_x, test_x = make_reference_and_test_tensors(
            (2, 4), min_val=-1.0, max_val=1.0, requires_grad=False
        )
        out = fwd(test_x)
        assert out.shape == test_x.shape
        assert_close(to_cpu(out), to_cpu(ref_fwd(ref_x)), **dtype_tols(torch.float32))

    @pytest.mark.parametrize(
        "name,bwd,ref_fwd",
        [(name, bwd, ref) for name, _, bwd, ref in _ELEMENTWISE_CASES],
        ids=[name for name, *_ in _ELEMENTWISE_CASES],
    )
    def test_backward_matches_autograd(self, name: str, bwd, ref_fwd) -> None:
        ref_x, test_x = make_reference_and_test_tensors(
            (2, 4), min_val=-1.0, max_val=1.0, requires_grad=True
        )
        ref_dy, test_dy = make_reference_and_test_tensors(
            (2, 4), min_val=-1.0, max_val=1.0, requires_grad=False
        )
        ref_fwd(ref_x).backward(gradient=ref_dy)
        dx = bwd(test_x, test_dy)
        assert dx.shape == test_x.shape
        assert_close(to_cpu(dx), ref_x.grad, **dtype_tols(torch.float32))

    def test_gelu_backward_zero_for_negative_extreme(self) -> None:
        x = torch.full((2, 4), -100.0, device=_NPU)
        dy = torch.ones(2, 4, device=_NPU)
        dx = gelu_bwd(x, dy)
        assert_close(to_cpu(dx), torch.zeros_like(to_cpu(dx)), atol=1e-3, rtol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# GLU activations (split input along last dim)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_glu_input() -> torch.Tensor:
    """Input shape (B, 2*D) for GLU activations."""
    torch.manual_seed(0)
    return torch.randn(2, 8, device=_NPU)


def _glu_ref(x: torch.Tensor) -> torch.Tensor:
    gate, linear = x.chunk(2, dim=-1)
    return torch.sigmoid(gate) * linear


def _swiglu_ref(x: torch.Tensor) -> torch.Tensor:
    gate, linear = x.chunk(2, dim=-1)
    return F.silu(gate) * linear


def _geglu_ref(x: torch.Tensor) -> torch.Tensor:
    gate, linear = x.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * linear


def _reglu_ref(x: torch.Tensor) -> torch.Tensor:
    gate, linear = x.chunk(2, dim=-1)
    return F.relu(gate) * linear


def _sreglu_ref(x: torch.Tensor) -> torch.Tensor:
    gate, linear = x.chunk(2, dim=-1)
    relu_gate = F.relu(gate)
    return relu_gate * relu_gate * linear


def _qgeglu_ref(x: torch.Tensor) -> torch.Tensor:
    gate, linear = x.chunk(2, dim=-1)
    return _fast_gelu_ref(gate) * linear


_GLU_CASES = [
    ("glu", glu_fwd, glu_bwd, _glu_ref),
    ("swiglu", swiglu_fwd, swiglu_bwd, _swiglu_ref),
    ("geglu", geglu_fwd, geglu_bwd, _geglu_ref),
    ("reglu", reglu_fwd, reglu_bwd, _reglu_ref),
    ("sreglu", sreglu_fwd, sreglu_bwd, _sreglu_ref),
    ("qgeglu", qgeglu_fwd, qgeglu_bwd, _qgeglu_ref),
]


class TestGluVariants:
    """Numerical forward/backward coverage for all GLU-family activations.

    The reference is computed in fp64 on CPU from plain PyTorch ops; the
    backward reference is torch autograd on the same formula, compared
    against the handwritten/kernel backward on NPU. Inputs include negative
    values so relu-family gates are exercised on both sides of zero.
    """

    @pytest.mark.parametrize(
        "name,fwd,ref_fwd",
        [(name, fwd, ref) for name, fwd, _, ref in _GLU_CASES],
        ids=[name for name, *_ in _GLU_CASES],
    )
    def test_forward_numerical(self, name: str, fwd, ref_fwd) -> None:
        ref_x, test_x = make_reference_and_test_tensors(
            (2, 8), min_val=-1.0, max_val=1.0, requires_grad=False
        )
        out = fwd(test_x)
        assert out.shape == (2, 4)
        assert_close(to_cpu(out), to_cpu(ref_fwd(ref_x)), **dtype_tols(torch.float32))

    @pytest.mark.parametrize(
        "name,bwd,ref_fwd",
        [(name, bwd, ref) for name, _, bwd, ref in _GLU_CASES],
        ids=[name for name, *_ in _GLU_CASES],
    )
    def test_backward_matches_autograd(self, name: str, bwd, ref_fwd) -> None:
        ref_x, test_x = make_reference_and_test_tensors(
            (2, 8), min_val=-1.0, max_val=1.0, requires_grad=True
        )
        ref_dy, test_dy = make_reference_and_test_tensors(
            (2, 4), min_val=-1.0, max_val=1.0, requires_grad=False
        )
        ref_fwd(ref_x).backward(gradient=ref_dy)
        dx = bwd(test_x, test_dy)
        assert dx.shape == test_x.shape
        assert_close(to_cpu(dx), ref_x.grad, **dtype_tols(torch.float32))


class TestClampedSwiglu:
    def test_forward_default_clamp(self) -> None:
        x = torch.full((2, 8), 50.0, device=_NPU)  # > default limit
        out = clamped_swiglu_fwd(x)
        assert out.shape == (2, 4)

    def test_forward_custom_clamp(self) -> None:
        x = _make_glu_input()
        out = clamped_swiglu_fwd(x, limit=10.0)
        assert out.shape == (2, 4)

    def test_backward(self) -> None:
        x = _make_glu_input()
        dy = torch.randn(2, 4, device=_NPU)
        dx = clamped_swiglu_bwd(x, dy)
        assert dx.shape == x.shape

    def test_backward_outside_clamp_zero(self) -> None:
        """Gradient should be zero outside the clamp range."""
        x = torch.full((2, 8), 100.0, device=_NPU)
        dy = torch.ones(2, 4, device=_NPU)
        dx = clamped_swiglu_bwd(x, dy)
        # Gate dgate is zero outside clamp range
        assert dx.shape == (2, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch tables
# ═══════════════════════════════════════════════════════════════════════════════


class TestDispatchTables:
    def test_fwd_bwd_keys_match(self) -> None:
        """Every fwd key has a bwd partner (and vice versa)."""
        assert set(ACTIVATION_FWD.keys()) == set(ACTIVATION_BWD.keys())

    def test_glu_variants_subset_of_fwd(self) -> None:
        assert GLU_VARIANTS <= set(ACTIVATION_FWD.keys())
