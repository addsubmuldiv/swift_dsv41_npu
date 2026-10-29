# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``module/utils.py`` helpers:
- cast_if_needed / clear_tensor_data
- ensure_divisibility / divide
- init_method_constant / init_method_normal / get_default_init_method
- requires_grad / needs_quantized_gemm
- assert_dim_for_fp8_exec / assert_dim_for_all_gather
- profiler_range_push/pop/get_profiler_range_context
- assert_viewless_tensor / safely_set_viewless_tensor_data
- SplitAlongDim
- get_cudnn_version / get_device_compute_capability
- canonicalize_device / canonicalize_dtype / devices_match
- round_up_to_nearest_multiple / is_bf16_available
- view_as_n_dim / get_quant_dtype / check_npu_version / get_npu_version
"""

from __future__ import annotations

import pytest
import torch
import torch_npu

from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.utils import (
    SplitAlongDim,
    assert_dim_for_all_gather,
    assert_dim_for_fp8_exec,
    assert_viewless_tensor,
    canonicalize_device,
    canonicalize_dtype,
    cast_if_needed,
    check_npu_version,
    clear_tensor_data,
    devices_match,
    divide,
    get_cached_ones_tensor,
    get_cudnn_version,
    get_default_init_method,
    get_device_compute_capability,
    get_npu_version,
    get_profiler_range_context,
    get_quant_dtype,
    init_method_constant,
    init_method_normal,
    is_bf16_available,
    mark_grouped_tensor,
    needs_quantized_gemm,
    profiler_range_pop,
    profiler_range_push,
    requires_grad,
    round_up_to_nearest_multiple,
    safely_set_viewless_tensor_data,
    view_as_n_dim,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# cast_if_needed
# ═══════════════════════════════════════════════════════════════════════════════


class TestCastIfNeeded:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_none_returns_none(self) -> None:
        assert cast_if_needed(None, torch.float32) is None

    def test_same_dtype_returns_same(self) -> None:
        t = torch.randn(3, device=_NPU)
        out = cast_if_needed(t, torch.float32)
        assert out is t

    def test_different_dtype_casts(self) -> None:
        t = torch.randn(3, device=_NPU)
        out = cast_if_needed(t, torch.float16)
        assert out.dtype == torch.float16
        assert out.device.type == "npu"


class TestTensorMetadataHelpers:
    def test_cached_ones_reuses_tensor_and_preserves_value(self) -> None:
        first = get_cached_ones_tensor(6, torch.float32, _NPU)
        second = get_cached_ones_tensor(6, torch.float32, _NPU)
        assert first is second
        torch.testing.assert_close(first, torch.ones(6, device=_NPU))

    def test_mark_grouped_tensor_marks_every_tensor(self) -> None:
        first = torch.zeros(2, device=_NPU)
        second = torch.zeros(3, device=_NPU)
        mark_grouped_tensor(first, None, second)
        assert first.grouped_tensor_scale_inv is False
        assert second.grouped_tensor_scale_inv is False

    @pytest.mark.parametrize(
        ("quant_mode", "expected_fwd", "expected_bwd"),
        [
            ("hif8", torch_npu.hifloat8, torch_npu.hifloat8),
            ("hybrid", torch.float8_e4m3fn, torch.float8_e5m2),
            ("default", torch.float8_e4m3fn, torch.float8_e4m3fn),
        ],
    )
    def test_get_quant_dtype_returns_dtype_pair(
        self,
        quant_mode: str,
        expected_fwd: torch.dtype,
        expected_bwd: torch.dtype,
    ) -> None:
        dtype_pair = get_quant_dtype(quant_mode)
        assert dtype_pair.fwd == expected_fwd
        assert dtype_pair.bwd == expected_bwd


# ═══════════════════════════════════════════════════════════════════════════════
# divide / ensure_divisibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestDivide:
    def test_divisible(self) -> None:
        assert divide(10, 2) == 5

    def test_not_divisible_raises(self) -> None:
        with pytest.raises(ValueError, match="not divisible"):
            divide(7, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# init methods
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitMethods:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_init_method_constant(self) -> None:
        init_fn = init_method_constant(0.5)
        t = torch.empty(3, 4, device=_NPU)
        init_fn(t)
        assert torch.allclose(t, torch.full_like(t, 0.5))

    def test_init_method_normal(self) -> None:
        init_fn = init_method_normal(0.5)
        t = torch.empty(100, 100, device=_NPU)
        init_fn(t)
        # ~95% of values within 2σ
        torch.npu.synchronize()
        z = (t - t.mean()) / t.std()
        assert (z.abs() < 2.5).float().mean().item() > 0.95

    def test_get_default_init_method_returns_normal(self) -> None:
        fn = get_default_init_method()
        assert fn is not None
        t = torch.empty(10, 10, device=_NPU)
        fn(t)
        assert t.std().item() > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# requires_grad / needs_quantized_gemm
# ═══════════════════════════════════════════════════════════════════════════════


class TestRequiresGrad:
    def test_any_true(self) -> None:
        a = torch.zeros(1, device=_NPU)
        b = torch.zeros(1, device=_NPU, requires_grad=True)
        assert requires_grad(a, b) is True

    def test_all_false(self) -> None:
        a = torch.zeros(1, device=_NPU)
        assert requires_grad(a) is False

    def test_with_none(self) -> None:
        assert requires_grad(None, torch.zeros(1, device=_NPU)) is False


class TestNeedsQuantizedGemm:
    def test_fp8_e4m3_returns_true(self) -> None:
        t = torch.empty(1, dtype=torch.float8_e4m3fn, device=_NPU)
        assert needs_quantized_gemm(t) is True

    def test_fp32_returns_false(self) -> None:
        t = torch.empty(1, dtype=torch.float32, device=_NPU)
        assert needs_quantized_gemm(t) is False


# ═══════════════════════════════════════════════════════════════════════════════
# assert_dim_for_fp8_exec
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssertDimForFp8Exec:
    def test_valid_dims(self) -> None:
        inp = torch.randn(2, 8, device=_NPU)
        w = torch.randn(4, 8, device=_NPU)  # (out, in) so shape[1]=8 = inp last dim
        assert_dim_for_fp8_exec(inp, w)  # no raise

    def test_input_1d_raises(self) -> None:
        inp = torch.randn(8, device=_NPU)
        w = torch.randn(8, 4, device=_NPU)
        with pytest.raises(AssertionError, match="at least 2 dimensions"):
            assert_dim_for_fp8_exec(inp, w)

    def test_weight_not_2d_raises(self) -> None:
        inp = torch.randn(2, 8, device=_NPU)
        w = torch.randn(2, 4, 8, device=_NPU)
        with pytest.raises(AssertionError, match="must have 2 dimensions"):
            assert_dim_for_fp8_exec(inp, w)

    def test_feature_dim_mismatch_raises(self) -> None:
        inp = torch.randn(2, 8, device=_NPU)
        w = torch.randn(16, 4, device=_NPU)  # in_features=16 != 8
        with pytest.raises(AssertionError, match="must match"):
            assert_dim_for_fp8_exec(inp, w)


# ═══════════════════════════════════════════════════════════════════════════════
# assert_dim_for_all_gather
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssertDimForAllGather:
    def test_valid_2d(self) -> None:
        t = torch.randn(2, 8, device=_NPU)
        assert_dim_for_all_gather(t, with_all_gather=True, quantizer=None)

    def test_with_all_gather_and_1d_raises(self) -> None:
        t = torch.randn(8, device=_NPU)
        with pytest.raises(AssertionError, match="at least 2 dimensions"):
            assert_dim_for_all_gather(t, with_all_gather=True, quantizer=None)

    def test_without_all_gather_skips_check(self) -> None:
        t = torch.randn(8, device=_NPU)  # 1d but no all_gather
        assert_dim_for_all_gather(t, with_all_gather=False, quantizer=None)


# ═══════════════════════════════════════════════════════════════════════════════
# profiler_range
# ═══════════════════════════════════════════════════════════════════════════════


class TestProfilerRange:
    def test_push_pop_no_op(self) -> None:
        profiler_range_push("test")  # no-op
        profiler_range_pop("test")  # no-op

    def test_context_manager(self) -> None:
        with get_profiler_range_context("ctx") as cm:
            assert cm is not None


# ═══════════════════════════════════════════════════════════════════════════════
# assert_viewless_tensor / safely_set_viewless_tensor_data
# ═══════════════════════════════════════════════════════════════════════════════


class TestViewlessTensor:
    def test_viewless_returns_unchanged(self) -> None:
        t = torch.zeros(3, 4, device=_NPU)
        out = assert_viewless_tensor(t)
        assert out is t

    def test_view_raises(self) -> None:
        base = torch.zeros(3, 4, device=_NPU)
        view = base[0]
        with pytest.raises(AssertionError, match="memory leak"):
            assert_viewless_tensor(view)

    def test_safely_set_data(self) -> None:
        t = torch.zeros(3, 4, device=_NPU)
        new_data = torch.ones(3, 4, device=_NPU)
        safely_set_viewless_tensor_data(t, new_data)
        torch.testing.assert_close(t, new_data)


# ═══════════════════════════════════════════════════════════════════════════════
# SplitAlongDim
# ═══════════════════════════════════════════════════════════════════════════════


class TestSplitAlongDim:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_forward_split_int(self) -> None:
        x = torch.randn(6, 4, device=_NPU, requires_grad=True)
        outs = SplitAlongDim.apply(x, 0, 3)
        assert len(outs) == 2
        assert outs[0].shape == (3, 4)
        torch.testing.assert_close(outs[0], x[:3])
        torch.testing.assert_close(outs[1], x[3:])

    def test_forward_split_list(self) -> None:
        x = torch.randn(6, 4, device=_NPU)
        outs = SplitAlongDim.apply(x, 0, [2, 4])
        assert len(outs) == 2
        assert outs[0].shape == (2, 4)
        assert outs[1].shape == (4, 4)

    def test_backward_concat(self) -> None:
        x = torch.randn(6, 4, device=_NPU, requires_grad=True)
        outs = SplitAlongDim.apply(x, 0, 3)
        loss = outs[0].sum() + outs[1].sum()
        loss.backward()
        torch.npu.synchronize()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ═══════════════════════════════════════════════════════════════════════════════
# Device helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeviceHelpers:
    def test_get_cudnn_version_placeholder(self) -> None:
        # NPU-port placeholder: cuDNN does not exist on Ascend and the function
        # unconditionally returns a stub tuple — assert the contract (type/shape),
        # not the hardcoded placeholder values.
        v = get_cudnn_version()
        assert isinstance(v, tuple)
        assert len(v) == 3
        assert all(isinstance(x, int) for x in v)

    def test_get_device_compute_capability_placeholder(self) -> None:
        # NPU-port placeholder: "compute capability" is a CUDA concept; the function
        # unconditionally returns a stub tuple — assert the contract only.
        v = get_device_compute_capability()
        assert isinstance(v, tuple)
        assert len(v) == 2
        assert all(isinstance(x, int) for x in v)

    def test_is_bf16_available(self) -> None:
        assert is_bf16_available() is True

    def test_canonicalize_device_none(self) -> None:
        d = canonicalize_device(None)
        assert d.type == "npu"

    def test_canonicalize_device_str(self) -> None:
        d = canonicalize_device("npu:0")
        assert d.type == "npu"
        assert d.index == 0

    def test_canonicalize_dtype_none(self) -> None:
        d = canonicalize_dtype(None)
        assert d == torch.get_default_dtype()

    def test_canonicalize_dtype_explicit(self) -> None:
        assert canonicalize_dtype(torch.float16) == torch.float16

    def test_devices_match_same_npu(self) -> None:
        assert devices_match(torch.device("npu:0"), torch.device("npu:0")) is True

    def test_devices_different_index(self) -> None:
        # First check available devices
        if torch.npu.device_count() < 2:
            pytest.skip("Need ≥2 NPU devices")
        assert devices_match(torch.device("npu:0"), torch.device("npu:1")) is False

    def test_round_up_to_nearest(self) -> None:
        assert round_up_to_nearest_multiple(7, 5) == 10
        assert round_up_to_nearest_multiple(10, 5) == 10
        assert round_up_to_nearest_multiple(0, 5) == 0

    def test_round_up_zero_multiple_raises(self) -> None:
        with pytest.raises(ValueError, match="multiple cannot be zero"):
            round_up_to_nearest_multiple(7, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# view_as_n_dim
# ═══════════════════════════════════════════════════════════════════════════════


class TestViewAsNDim:
    def test_already_n_dim(self) -> None:
        t = torch.randn(2, 3, device=_NPU)
        out = view_as_n_dim(t, dim=2)
        assert out.shape == (2, 3)

    def test_lower_dim_to_n(self) -> None:
        """2D tensor viewed as 3D: view(-1, *shape[-dim+1:]) = view(1, 2, 3) when dim=3."""
        t = torch.randn(2, 3, device=_NPU)
        out = view_as_n_dim(t, dim=3)
        assert out.shape == (1, 2, 3)

    def test_higher_dim_to_n(self) -> None:
        t = torch.randn(2, 3, 4, device=_NPU)
        out = view_as_n_dim(t, dim=2)
        assert out.shape == (6, 4)

    def test_zero_dim_to_n(self) -> None:
        t = torch.tensor(3.14, device=_NPU)
        out = view_as_n_dim(t, dim=2)
        assert out.shape == (1, 1)

    def test_dim_lt_2_raises(self) -> None:
        t = torch.randn(2, 3, device=_NPU)
        with pytest.raises(AssertionError, match="greater than or equal to 2"):
            view_as_n_dim(t, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# NPU version detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestNpuVersion:
    def test_get_npu_version_cross_validates_with_check(self) -> None:
        """get_npu_version() must return an NPUVersion that check_npu_version accepts."""
        v = get_npu_version()
        assert isinstance(v, NPUVersion)
        # Dynamic cross-validation: the detected version must pass its own min-version check
        assert check_npu_version(v) is True
        # No-arg form agrees with whether any version was detected at all
        assert check_npu_version() is (v is not NPUVersion.NONE)

    def test_check_npu_version_no_args_returns_true(self) -> None:
        assert check_npu_version() is True

    @pytest.mark.parametrize("min_version_name", ["A2", "A3", "A5"])
    def test_check_npu_version_min_version_matches_detected(self, min_version_name: str) -> None:
        """Expected value is derived from the dynamically detected version, never
        hardcoded to the machine this test was written on.
        """
        min_version = getattr(NPUVersion, min_version_name)
        detected = get_npu_version()
        expected = bool(detected is not NPUVersion.NONE and detected >= min_version)
        assert check_npu_version(min_version) is expected


# ═══════════════════════════════════════════════════════════════════════════════
# clear_tensor_data
# ═══════════════════════════════════════════════════════════════════════════════


class TestClearTensorData:
    def test_clear_none_skips(self) -> None:
        # Should not raise on None inputs
        clear_tensor_data(None, None)

    def test_clear_with_do_not_clear_flag(self) -> None:
        t = torch.zeros(3, device=_NPU)
        t._do_not_clear = True
        # Tensor with this flag should be skipped (not cleared)
        clear_tensor_data(t)
        assert t.numel() == 3  # unchanged
        del t._do_not_clear
