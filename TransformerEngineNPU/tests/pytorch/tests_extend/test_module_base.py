# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``module/base.py`` non-FP8 helpers:
- ``UserBufferQuantizationMode`` enum
- ``setup_dummy_wgrad`` / ``get_dummy_wgrad`` (no FP8 path)
- ``initialize_ub`` / ``destroy_ub`` no-op stubs
- ``fill_userbuffers_buffer_for_all_gather`` / ``get_ub`` no-op stubs
- ``TransformerEngineBaseModule`` via ``Linear`` and ``LayerNormLinear``
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.module import base
from transformer_engine.pytorch.module.base import (
    UserBufferQuantizationMode,
    _is_weight_workspace_valid,
    destroy_ub,
    fill_userbuffers_buffer_for_all_gather,
    get_dummy_wgrad,
    get_ub,
    initialize_ub,
    setup_dummy_wgrad,
)
from transformer_engine.pytorch import Linear, LayerNormLinear
from transformer_engine.pytorch.tensor import Float8TensorStorage
from utils import assert_close, dtype_tols, npu_available, to_cpu

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# UserBufferQuantizationMode enum
# ═══════════════════════════════════════════════════════════════════════════════


class TestUserBufferQuantizationMode:
    def test_none_value(self) -> None:
        assert UserBufferQuantizationMode.NONE.value == "none"

    def test_fp8_value(self) -> None:
        assert UserBufferQuantizationMode.FP8.value == "fp8"

    def test_membership(self) -> None:
        assert UserBufferQuantizationMode("none") is UserBufferQuantizationMode.NONE
        assert UserBufferQuantizationMode("fp8") is UserBufferQuantizationMode.FP8

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            UserBufferQuantizationMode("invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# setup_dummy_wgrad / get_dummy_wgrad (no FP8 path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDummyWgrad:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_get_dummy_wgrad_shape(self) -> None:
        """get_dummy_wgrad returns tensor of requested shape on NPU."""
        w = get_dummy_wgrad([3, 4], torch.float32, zero=False)
        assert w.shape == (3, 4)
        assert w.dtype == torch.float32
        assert w.device.type == "npu"
        assert w.requires_grad is False

    def test_get_dummy_wgrad_zero_filled(self) -> None:
        """zero=True → tensor is all zeros."""
        w = get_dummy_wgrad([2, 3], torch.float32, zero=True)
        torch.testing.assert_close(w.cpu(), torch.zeros(2, 3))

    def test_get_dummy_wgrad_not_zero_uninitialized(self) -> None:
        """zero=False → tensor not pre-filled with zeros (memory values)."""
        # Just verify it doesn't raise; contents are uninitialized.
        w = get_dummy_wgrad([2, 2], torch.float32, zero=False)
        assert w.shape == (2, 2)

    def test_get_dummy_wgrad_cache_same_key(self) -> None:
        """Same (shape, dtype, zero) key → same tensor returned."""
        w1 = get_dummy_wgrad([2, 2], torch.float32, zero=True)
        w2 = get_dummy_wgrad([2, 2], torch.float32, zero=True)
        assert w1.data_ptr() == w2.data_ptr()  # cache hit

    def test_get_dummy_wgrad_different_zero_different_cache(self) -> None:
        """zero=True vs zero=False → different cache entries."""
        w_zero = get_dummy_wgrad([3, 3], torch.float16, zero=True)
        w_nonzero = get_dummy_wgrad([3, 3], torch.float16, zero=False)
        assert w_zero.data_ptr() != w_nonzero.data_ptr()

    def test_setup_dummy_wgrad_without_marker_returns_none(self) -> None:
        """Plain tensor without grad_added_to_main_grad → returns None."""
        w = torch.zeros(3, 4, device=_NPU)
        result = setup_dummy_wgrad(w)
        assert result is None

    def test_setup_dummy_wgrad_with_marker_returns_tensor(self) -> None:
        """setup_dummy_wgrad returns a dummy wgrad and actively flips the marker."""
        w = torch.zeros(3, 4, device=_NPU)
        # Marker attribute present but NOT yet set — the function itself must set it
        # (base.py: weight.grad_added_to_main_grad = True).
        w.grad_added_to_main_grad = False
        w.main_grad = torch.zeros(3, 4, device=_NPU)
        result = setup_dummy_wgrad(w)
        assert result is not None
        assert result.shape == (3, 4)
        assert w.grad_added_to_main_grad is True

    def test_setup_dummy_wgrad_skips_placeholder_without_overlap(self) -> None:
        """Fused accumulation needs no persistent dummy when DDP overlap is disabled."""
        w = torch.zeros(3, 4, device=_NPU)
        w.grad_added_to_main_grad = False
        w.main_grad = torch.zeros(3, 4, device=_NPU)
        w.requires_dummy_wgrad = False

        cache_entries = len(base._dummy_wgrads)
        result = setup_dummy_wgrad(w)

        assert result is None
        assert w.grad_added_to_main_grad is True
        assert len(base._dummy_wgrads) == cache_entries

    def test_setup_dummy_wgrad_zero_out_wgrad(self) -> None:
        """zero_out_wgrad=True → dummy wgrad is zero-filled (base.py zero=... branch)."""
        w = torch.zeros(2, 5, device=_NPU)
        w.grad_added_to_main_grad = False
        w.main_grad = torch.zeros(2, 5, device=_NPU)
        w.zero_out_wgrad = True
        result = setup_dummy_wgrad(w)
        assert result is not None
        torch.testing.assert_close(result.cpu(), torch.zeros(2, 5))


# ═══════════════════════════════════════════════════════════════════════════════
# UserBuffer stubs: initialize_ub / fill_userbuffers_buffer_for_all_gather /
# get_ub are literal ``pass`` stubs in this NPU port (base.py); destroy_ub
# clears the module-level _ub_communicators global.
# ═══════════════════════════════════════════════════════════════════════════════


class TestUserBufferStubs:
    def test_ub_stubs_exist_and_accept_calls(self) -> None:
        # smoke: these are literal-pass stubs; only verify they exist, accept the
        # documented arguments, and produce no side effects.
        initialize_ub(shape=[3, 4], tp_size=1)
        initialize_ub(
            shape=[2, 2],
            tp_size=1,
            use_fp8=True,
            quantization_modes=[UserBufferQuantizationMode.NONE],
        )
        t = torch.zeros(3, 4, device=_NPU)
        fill_userbuffers_buffer_for_all_gather(
            t,
            ub_comm=None,
            ub_obj=None,
            tp_size=1,
            tp_rank=0,
            ub_algo=None,
        )
        assert get_ub(shape=[3, 4], dtype=torch.float32, ub_comm=None, ub_algo=None) is None

    def test_destroy_ub_clears_communicators(self) -> None:
        """destroy_ub must reset the module-level _ub_communicators global."""
        destroy_ub()
        assert base._ub_communicators is None


# ═══════════════════════════════════════════════════════════════════════════════
# _is_weight_workspace_valid - workspace validation helper
# ═══════════════════════════════════════════════════════════════════════════════


class _MockQuantizer:
    """Minimal quantizer stand-in carrying only the usage flags the helper reads."""

    def __init__(self, *, columnwise_usage: bool = False, rowwise_usage: bool = False) -> None:
        self.columnwise_usage = columnwise_usage
        self.rowwise_usage = rowwise_usage


def _make_float8_storage(data) -> Float8TensorStorage:
    """Bare Float8TensorStorage instance with only the ``_data`` attribute set, so
    isinstance checks pass without FP8 hardware. ``object.__new__`` is used because
    the class's own ``__new__`` requires data/fp8_scale_inv/fp8_dtype kwargs.
    """
    workspace = object.__new__(Float8TensorStorage)
    workspace._data = data
    return workspace


class TestIsWeightWorkspaceValid:
    """Test _is_weight_workspace_valid for different storage types."""

    def test_none_workspace_returns_true(self) -> None:
        """None workspace should return True (no validation needed)."""
        assert _is_weight_workspace_valid(None, None) is True

    def test_float8_workspace_columnwise_missing_data_invalid(self) -> None:
        """Float8TensorStorage: columnwise_usage=True and _data is None → False."""
        workspace = _make_float8_storage(None)
        quantizer = _MockQuantizer(columnwise_usage=True)
        assert _is_weight_workspace_valid(workspace, quantizer) is False

    def test_float8_workspace_columnwise_with_data_valid(self) -> None:
        """Float8TensorStorage: columnwise_usage=True and _data present → True."""
        workspace = _make_float8_storage(torch.zeros(4, 4))
        quantizer = _MockQuantizer(columnwise_usage=True)
        assert _is_weight_workspace_valid(workspace, quantizer) is True

    def test_float8_workspace_rowwise_only_ignores_columnwise_data(self) -> None:
        """Float8TensorStorage: rowwise-only usage does not require _data → True."""
        workspace = _make_float8_storage(None)
        quantizer = _MockQuantizer(rowwise_usage=True)
        assert _is_weight_workspace_valid(workspace, quantizer) is True


# ═══════════════════════════════════════════════════════════════════════════════
# TransformerEngineBaseModule via Linear - state_dict serialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleStateDict:
    """Test module state_dict save/load triggers base.py get_extra_state / set_extra_state."""

    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_linear_state_dict_roundtrip(self) -> None:
        """Linear state_dict save/load should work without error."""
        module = Linear(in_features=64, out_features=256)
        inp = torch.randn(2, 8, 64, device=_NPU)
        # Forward pass to ensure weights are initialized
        _ = module(inp)
        # Save state
        sd = module.state_dict()
        assert len(sd) > 0
        # Load state into a fresh instance (load target != save source)
        module2 = Linear(in_features=64, out_features=256)
        module2.load_state_dict(sd)
        # Verify still works
        out = module2(inp)
        # Linear transforms [B,S,in] -> [B,S,out]
        assert out.shape == (2, 8, 256)

    def test_layernorm_linear_state_dict_roundtrip(self) -> None:
        """LayerNormLinear state_dict save/load should work without error."""
        module = LayerNormLinear(in_features=64, out_features=256)
        inp = torch.randn(2, 8, 64, device=_NPU)
        # Forward pass
        _ = module(inp)
        # Save and load into a fresh instance (load target != save source)
        sd = module.state_dict()
        assert len(sd) > 0
        module2 = LayerNormLinear(in_features=64, out_features=256)
        module2.load_state_dict(sd)
        # Verify still works
        out = module2(inp)
        # LayerNormLinear transforms [B,S,in] -> [B,S,out]
        assert out.shape == (2, 8, 256)

    def test_state_dict_preserves_weights(self) -> None:
        """Loading a state dict must restore the saved weights into a perturbed module."""
        module = Linear(in_features=32, out_features=128)
        inp = torch.randn(1, 4, 32, device=_NPU)
        # Run forward to initialize
        _ = module(inp)
        # Capture weights and save state. NOTE: clone every entry — state_dict()
        # tensors share storage with the live parameters, so the in-place
        # perturbation below would otherwise corrupt the snapshot too.
        weight_before = module.weight.clone()
        sd = {k: v.clone() for k, v in module.state_dict().items()}
        # Perturb weights so that a no-op load would fail the comparison below
        module.weight.data.fill_(1.0)
        assert not torch.allclose(module.weight.cpu(), weight_before.cpu())
        # Load must restore the saved values
        module.load_state_dict(sd)
        torch.testing.assert_close(module.weight.cpu(), weight_before.cpu())


# ═══════════════════════════════════════════════════════════════════════════════
# TransformerEngineBaseModule - forward paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleForwardPaths:
    """Test forward passes exercise base.py caching and preparation paths."""

    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_linear_matches_torch_reference(self) -> None:
        """Linear output must match a torch.nn.functional.linear reference computed
        from the same weights (not a re-seeded self-comparison).
        """
        module = Linear(in_features=64, out_features=256)
        inp = torch.randn(2, 8, 64, device=_NPU, dtype=torch.float32)
        out = module(inp)
        expected = torch.nn.functional.linear(inp, module.weight, module.bias)
        assert_close(to_cpu(out), to_cpu(expected), **dtype_tols(torch.float32))

    def test_linear_backward_triggers_grad_path(self) -> None:
        """Backward pass should exercise grad_output_preprocess in base.py."""
        module = Linear(in_features=64, out_features=256)
        inp = torch.randn(2, 8, 64, device=_NPU, requires_grad=True)
        out = module(inp)
        loss = out.sum()
        loss.backward()
        # Gradients should be computed
        assert module.weight.grad is not None

    def test_layernorm_linear_forward_backward(self) -> None:
        """LayerNormLinear forward+backward exercises base.py prepare_forward + grad_preprocess."""
        module = LayerNormLinear(in_features=64, out_features=256)
        inp = torch.randn(2, 8, 64, device=_NPU, requires_grad=True)
        out = module(inp)
        loss = out.sum()
        loss.backward()
        # Verify gradients exist
        assert module.weight.grad is not None

    def test_layernorm_linear_multiple_microbatches(self) -> None:
        """Multiple microbatches exercise is_first_microbatch caching logic."""
        module = LayerNormLinear(in_features=64, out_features=128)
        batch1 = torch.randn(2, 4, 64, device=_NPU)
        batch2 = torch.randn(2, 4, 64, device=_NPU)
        # First microbatch
        out1 = module(batch1, is_first_microbatch=True)
        # Second microbatch
        out2 = module(batch2, is_first_microbatch=False)
        # Both should compute without error
        assert out1.shape == (2, 4, 128)
        assert out2.shape == (2, 4, 128)
