# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``module/layernorm_linear.py``:
- ``set_tensor_model_parallel_attributes`` (module-level function)
- ``LayerNormLinear.__init__`` and key attributes
- ``LayerNormLinear.reset_parameters`` (defer_init + normal paths)
- ``LayerNormLinear.ub_overlap_ag`` property
- Non-FP8 forward smoke test (FP8 path xfail on A2)
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.module.layernorm_linear import (
    LayerNormLinear,
    set_tensor_model_parallel_attributes,
)
from utils import npu_available

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
        """Sets the three expected attributes on the tensor."""
        t = torch.zeros(3, device=_NPU)
        set_tensor_model_parallel_attributes(t, is_parallel=True, dim=0, stride=1)
        assert t.tensor_model_parallel is True
        assert t.partition_dim == 0
        assert t.partition_stride == 1

    def test_is_parallel_false(self) -> None:
        t = torch.zeros(3, device=_NPU)
        set_tensor_model_parallel_attributes(t, is_parallel=False, dim=1, stride=2)
        assert t.tensor_model_parallel is False
        assert t.partition_dim == 1
        assert t.partition_stride == 2

    def test_already_set_raises(self) -> None:
        """If tensor already has a parallel attribute, raises AssertionError."""
        t = torch.zeros(3, device=_NPU)
        t.tensor_model_parallel = True
        with pytest.raises(AssertionError):
            set_tensor_model_parallel_attributes(t, is_parallel=False, dim=0, stride=1)


# ═══════════════════════════════════════════════════════════════════════════════
# LayerNormLinear.__init__
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayerNormLinearInit:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_default_layer_norm(self) -> None:
        m = LayerNormLinear(64, 32, device=_NPU)
        assert m.in_features == 64
        assert m.out_features == 32
        assert m.normalization == "LayerNorm"
        assert m.use_bias is True
        assert m.return_bias is False
        assert m.zero_centered_gamma is False
        assert m.eps == 1e-5
        # Parameters
        assert m.weight.shape == (32, 64)  # out_features × in_features
        assert m.bias.shape == (32,)
        assert m.layer_norm_weight.shape == (64,)
        assert m.layer_norm_bias.shape == (64,)

    def test_rms_norm(self) -> None:
        m = LayerNormLinear(64, 32, normalization="RMSNorm", device=_NPU)
        assert m.normalization == "RMSNorm"
        # RMSNorm has no bias
        assert m.layer_norm_weight.shape == (64,)
        assert not hasattr(m, "layer_norm_bias") or m.layer_norm_bias is None

    def test_no_bias(self) -> None:
        m = LayerNormLinear(64, 32, bias=False, device=_NPU)
        assert m.use_bias is False
        # bias=False → bias parameter is registered but has 0 elements
        assert m.bias.numel() == 0

    def test_custom_eps(self) -> None:
        m = LayerNormLinear(64, 32, eps=1e-3, device=_NPU)
        assert m.eps == 1e-3

    def test_parallel_mode_column_divides_out(self) -> None:
        m = LayerNormLinear(64, 32, parallel_mode="column", tp_size=2, device=_NPU)
        # tp_size=2 halves out_features
        assert m.out_features == 16
        # Weight shape reflects partitioned output
        assert m.weight.shape == (16, 64)

    def test_parallel_mode_row_divides_in(self) -> None:
        m = LayerNormLinear(64, 32, parallel_mode="row", tp_size=2, device=_NPU)
        # tp_size=2 halves in_features
        assert m.in_features == 32
        assert m.weight.shape == (32, 32)

    def test_unsupported_parallel_mode_raises(self) -> None:
        with pytest.raises(AssertionError, match="parallel_mode"):
            LayerNormLinear(64, 32, parallel_mode="invalid", device=_NPU)

    def test_unsupported_normalization_raises(self) -> None:
        with pytest.raises(AssertionError, match="Unsupported normalization"):
            LayerNormLinear(64, 32, normalization="BatchNorm", device=_NPU)

    def test_weight_names_attribute(self) -> None:
        m = LayerNormLinear(64, 32, device=_NPU)
        assert "weight" in m.weight_names
        assert m.weight_names[0] == "weight"

    def test_bias_names_attribute(self) -> None:
        m = LayerNormLinear(64, 32, device=_NPU)
        assert "bias" in m.bias_names
        assert m.bias_names[0] == "bias"

    def test_overlap_flags_default(self) -> None:
        m = LayerNormLinear(64, 32, device=_NPU)
        assert m.overlap_ag_fprop is False
        assert m.overlap_rs_dgrad is False
        assert m.overlap_rs_fprop is False
        assert m.overlap_ag_dgrad is False


# ═══════════════════════════════════════════════════════════════════════════════
# reset_parameters
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayerNormLinearResetParameters:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_defer_init_does_not_modify(self) -> None:
        """defer_init=True → reset_parameters early-returns before touching weights.

        Default reset_parameters was called in __init__ so sequence_parallel IS
        set; the defer_init path leaves subsequent state alone (no double-set
        crash).
        """
        m = LayerNormLinear(64, 32, sequence_parallel=True, device=_NPU)
        # Set a sentinel; defer_init should not overwrite it
        m.layer_norm_weight.weight_marker = "sentinel"  # type: ignore[attr-defined]
        m.reset_parameters(defer_init=True)
        assert m.layer_norm_weight.weight_marker == "sentinel"  # type: ignore[attr-defined]

    def test_normal_reset_runs_without_error(self) -> None:
        """Smoke: defer_init=False on a fresh module completes reset without raising."""
        m = LayerNormLinear(64, 32, sequence_parallel=True, device=_NPU)
        m.reset_parameters(defer_init=False)
        assert m.layer_norm_weight.shape == (64,)
        assert torch.isfinite(m.layer_norm_weight).all().item()

    def test_reset_propagates_sequence_parallel_attr(self) -> None:
        """Decision point 8: sequence_parallel attr is set on layer_norm params when sp=True.
        Requires tp_size>1 for self.sequence_parallel to be True (decision point 7).
        """
        m = LayerNormLinear(
            64, 32, parallel_mode="column", tp_size=2, sequence_parallel=True, device=_NPU
        )
        # __init__ ran reset_parameters; sp attr should be set on layer_norm params
        assert m.sequence_parallel is True
        assert m.layer_norm_weight.sequence_parallel is True
        if m.normalization == "LayerNorm":
            assert m.layer_norm_bias.sequence_parallel is True
        # Re-reset; attr should remain True (idempotent)
        m.reset_parameters(defer_init=False)
        assert m.layer_norm_weight.sequence_parallel is True


# ═══════════════════════════════════════════════════════════════════════════════
# ub_overlap_ag property
# ═══════════════════════════════════════════════════════════════════════════════


class TestOverlapFlags:
    """The overlap_* attributes are computed from the four ub_* constructor flags
    combined with parallel_mode and sequence_parallel.
    """

    def test_default_all_false(self) -> None:
        m = LayerNormLinear(64, 32, device=_NPU)
        assert m.overlap_ag_fprop is False
        assert m.overlap_rs_dgrad is False
        assert m.overlap_rs_fprop is False
        assert m.overlap_ag_dgrad is False

    def test_column_parallel_with_sp_ag(self) -> None:
        """column + sp + ub_overlap_ag=True → overlap_ag_fprop True (rs_dgrad needs rs_dgrad flag)."""
        m = LayerNormLinear(
            64,
            32,
            parallel_mode="column",
            tp_size=2,
            sequence_parallel=True,
            ub_overlap_ag=True,
            device=_NPU,
        )
        # column + sp + ub_overlap_ag → overlap_ag_fprop True
        assert m.overlap_ag_fprop is True
        # overlap_rs_dgrad requires ub_overlap_rs_dgrad=True, not just ub_overlap_ag
        assert m.overlap_rs_dgrad is False

    def test_row_parallel_with_sp_ag(self) -> None:
        """row + sp + ub_overlap_ag=True → overlap_ag_dgrad True."""
        m = LayerNormLinear(
            64,
            32,
            parallel_mode="row",
            tp_size=2,
            sequence_parallel=True,
            ub_overlap_ag=True,
            device=_NPU,
        )
        # row + sp + ub_overlap_ag → overlap_ag_dgrad True (not overlap_ag_fprop)
        assert m.overlap_ag_dgrad is True
        assert m.overlap_ag_fprop is False

    def test_column_parallel_with_sp_rs_dgrad(self) -> None:
        """column + sp + ub_overlap_rs_dgrad=True → overlap_rs_dgrad True."""
        m = LayerNormLinear(
            64,
            32,
            parallel_mode="column",
            tp_size=2,
            sequence_parallel=True,
            ub_overlap_rs_dgrad=True,
            device=_NPU,
        )
        assert m.overlap_rs_dgrad is True
        # ub_overlap_rs_dgrad doesn't trigger overlap_ag_fprop
        assert m.overlap_ag_fprop is False

    def test_sequence_parallel_propagates_to_attr(self) -> None:
        """Decision point 7: tp_size>1 + sequence_parallel=True → self.sequence_parallel = True."""
        m = LayerNormLinear(
            64, 32, parallel_mode="column", tp_size=2, sequence_parallel=True, device=_NPU
        )
        assert m.sequence_parallel is True
        m2 = LayerNormLinear(64, 32, tp_size=1, sequence_parallel=True, device=_NPU)
        # tp_size=1 → sequence_parallel stays False
        assert m2.sequence_parallel is False


# ═══════════════════════════════════════════════════════════════════════════════
# _LayerNormLinearNonTensorArgs.ub_overlap_ag
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Forward smoke (non-FP8 path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayerNormLinearForward:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_forward_no_fp8_runs(self) -> None:
        """Default forward (fp8=False) calls basic_linear + LayerNorm + Linear."""
        m = LayerNormLinear(64, 32, device=_NPU)
        x = torch.randn(2, 8, 64, device=_NPU)
        out = m(x)
        torch.npu.synchronize()
        assert out.shape == (2, 8, 32)

    def test_forward_returns_layernorm_output(self) -> None:
        """return_layernorm_output=True → tuple (out, ln_out)."""
        m = LayerNormLinear(64, 32, return_layernorm_output=True, device=_NPU)
        x = torch.randn(2, 8, 64, device=_NPU)
        result = m(x)
        assert isinstance(result, tuple)
        out, ln_out = result
        assert out.shape == (2, 8, 32)
        assert ln_out.shape == (2, 8, 64)

    def test_forward_rms_norm(self) -> None:
        """Normalization='RMSNorm' → forward completes."""
        m = LayerNormLinear(64, 32, normalization="RMSNorm", device=_NPU)
        x = torch.randn(2, 8, 64, device=_NPU)
        out = m(x)
        torch.npu.synchronize()
        assert out.shape == (2, 8, 32)
