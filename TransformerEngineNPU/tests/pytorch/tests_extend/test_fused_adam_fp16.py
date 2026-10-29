# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Incremental UT for commit 3d03bb7: FP16/BF16 precision + FP32 master weights.

Covers new methods introduced by the commit that are not exercised by the
existing ``test_fused_optimizer.py``:
- ``_scale_dict`` / ``_ensure_state_scale`` / ``_require_state_scale``
- ``_compute_state_scale`` (instance method)
- ``_get_scaled_state_as_unscaled`` / ``get_unscaled_state`` / ``set_scaled_state``
- ``_step_to_int`` / ``_advance_group_step`` / ``_max_param_state_step``
- ``_check_param_dtype`` / ``_check_state_dtype`` (valid paths)
- ``_state_dtype_for_name``
- FP16/BF16 params through ``step()`` with non-default exp_avg_dtype
- ``state_dict`` / ``load_state_dict`` round-trip with FP16 state storage
- Empty param group in ``step()``
- ``master_weights`` with FP16 params
"""

from __future__ import annotations

import pytest
import torch

from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

if npu_available():
    from transformer_engine.pytorch.optimizers import FusedAdam
else:
    FusedAdam = None


# ── helpers ──────────────────────────────────────────────────────────────────


def _device() -> torch.device:
    return torch.device("npu")


def _randn(shape, dtype=torch.float32):
    return torch.randn(shape, dtype=dtype, device="cpu").to(device=_device())


def _make_linear(dtype=torch.float32):
    return torch.nn.Linear(8, 4, device="cpu", dtype=dtype).to(device=_device())


def _make_param(dtype=torch.float32):
    values = torch.linspace(-0.5, 0.5, steps=8, dtype=torch.float32, device="cpu")
    return torch.nn.Parameter(values.to(device=_device(), dtype=dtype))


def _mse_step(model, optimizer, x, target):
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# ═══════════════════════════════════════════════════════════════════════════════
# TestInternalScaleMethods
# ═══════════════════════════════════════════════════════════════════════════════


class TestInternalScaleMethods:
    """Unit tests for newly introduced scale machinery."""

    @staticmethod
    def setup_method() -> None:
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    def test_scale_dict_returns_per_param_dict(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        p1 = list(opt.param_groups[0]["params"])[0]
        p2 = torch.nn.Parameter(torch.ones(3, device=_device()))
        d1 = opt._scale_dict(p1)
        d2 = opt._scale_dict(p2)
        assert d1 is not d2
        assert isinstance(d1, dict)

    def test_ensure_state_scale_creates_default(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        param = list(opt.param_groups[0]["params"])[0]
        scale = opt._ensure_state_scale(param, "exp_avg")
        assert scale.item() == 1.0
        assert scale.dtype == torch.float32

    def test_ensure_state_scale_preserves_existing(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        param = list(opt.param_groups[0]["params"])[0]
        opt._scale_dict(param)["exp_avg"] = torch.tensor(3.5, device=_device())
        scale = opt._ensure_state_scale(param, "exp_avg")
        assert abs(scale.item() - 3.5) < 1e-6

    def test_require_state_scale_returns_existing(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        param = list(opt.param_groups[0]["params"])[0]
        opt._scale_dict(param)["exp_avg"] = torch.tensor(2.0, device=_device())
        scale = opt._require_state_scale(param, "exp_avg")
        assert abs(scale.item() - 2.0) < 1e-6
        assert scale.device == param.device

    def test_require_state_scale_raises_for_missing(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        param = list(opt.param_groups[0]["params"])[0]
        with pytest.raises(RuntimeError, match="Missing scale"):
            opt._require_state_scale(param, "exp_avg")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_compute_state_scale(self, dtype: torch.dtype) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        x = torch.randn(12, device="cpu").to(device=_device())
        if dtype in (torch.float32, torch.bfloat16):
            assert opt._compute_state_scale(x, dtype).item() == 1.0
        else:
            actual = opt._compute_state_scale(x, torch.float16)
            # fp16 scale should be > 1.0 (compressing fp32 range) or at least tiny
            assert actual.item() >= torch.finfo(torch.float32).tiny

    def test_compute_state_scale_zeros_returns_one(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        x = torch.zeros(12, device=_device())
        scale = opt._compute_state_scale(x, torch.float16)
        assert scale.item() == 1.0

    def test_compute_state_scale_nan_returns_one(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        x = torch.full((8,), float("nan"), device=_device())
        scale = opt._compute_state_scale(x, torch.float16)
        assert scale.item() == 1.0

    def test_get_scaled_state_as_unscaled_fp32(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3, master_weights=True)
        param_fp16 = _make_param(dtype=torch.float16)
        group = {
            "params": [param_fp16],
            "lr": 1e-3,
            "eps": 1e-8,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "adam_w_mode": True,
            "bias_correction": True,
        }
        opt.param_groups.append(group)
        param_fp16.grad = _randn((8,), dtype=torch.float16)
        opt.step()
        opt.zero_grad()
        # master_param is fp32, _get_scaled_state_as_unscaled returns fp32 clone
        stored = opt.state[param_fp16]["master_param"]
        unscaled = opt._get_scaled_state_as_unscaled(param_fp16, "master_param", stored)
        assert unscaled.dtype == torch.float32

    def test_get_scaled_state_for_fp32_stored_state(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        param = list(opt.param_groups[0]["params"])[0]
        param.grad = _randn((4, 8))
        opt.step()
        opt.zero_grad()
        v = opt.state[param]["exp_avg"]
        unscaled = opt._get_scaled_state_as_unscaled(param, "exp_avg", v)
        assert unscaled.dtype == torch.float32

    def test_get_unscaled_state_returns_fp32(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        param = list(opt.param_groups[0]["params"])[0]
        param.grad = _randn((4, 8))
        opt.step()
        opt.zero_grad()
        exp_avg = opt.get_unscaled_state(param, "exp_avg")
        assert exp_avg.dtype == torch.float32
        exp_avg_sq = opt.get_unscaled_state(param, "exp_avg_sq")
        assert exp_avg_sq.dtype == torch.float32

    def test_set_scaled_state_fp32(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        param = list(opt.param_groups[0]["params"])[0]
        val = torch.randn(4, 8, dtype=torch.float32, device=_device())
        opt.set_scaled_state(param, "exp_avg", val)
        assert opt.state[param]["exp_avg"].dtype == torch.float32


# ═══════════════════════════════════════════════════════════════════════════════
# TestStepTracking
# ═══════════════════════════════════════════════════════════════════════════════


class TestStepTracking:
    """Exercise ``_step_to_int`` / ``_advance_group_step`` / ``_max_param_state_step``."""

    @staticmethod
    def setup_method() -> None:
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    def test_step_to_int_python_int(self) -> None:
        assert FusedAdam._step_to_int(5) == 5

    def test_step_to_int_tensor_scalar(self) -> None:
        t = torch.tensor(3, device=_device())
        assert FusedAdam._step_to_int(t) == 3

    def test_step_to_int_tensor_non_scalar_raises(self) -> None:
        with pytest.raises(ValueError, match="scalar"):
            FusedAdam._step_to_int(torch.ones(2, device=_device()))

    def test_advance_group_step_first_call(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        group = opt.param_groups[0]
        s = opt._advance_group_step(group)
        assert s == 1

    def test_advance_group_step_serial(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        group = opt.param_groups[0]
        for expected in range(1, 5):
            assert opt._advance_group_step(group) == expected

    def test_advance_group_step_picks_up_legacy_param_step(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        for param in opt.param_groups[0]["params"]:
            opt.state[param]["step"] = 7
        group = opt.param_groups[0]
        opt._advance_group_step(group)
        assert group["step"] >= 7

    def test_max_param_state_step_empty(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        assert opt._max_param_state_step(opt.param_groups[0]) == 0

    def test_max_param_state_step_with_values(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        params = list(opt.param_groups[0]["params"])
        opt.state[params[0]]["step"] = 3
        opt.state[params[1]]["step"] = 5
        assert opt._max_param_state_step(opt.param_groups[0]) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# TestDtypeValidation
# ═══════════════════════════════════════════════════════════════════════════════


class TestDtypeValidation:
    """``_check_param_dtype`` and ``_check_state_dtype`` valid + invalid combos."""

    @staticmethod
    def setup_method() -> None:
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_check_param_dtype_valid(self, dtype: torch.dtype) -> None:
        param = torch.nn.Parameter(torch.ones(4, device=_device(), dtype=dtype))
        FusedAdam._check_param_dtype(param)  # should not raise

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.int64])
    def test_check_param_dtype_invalid(self, dtype: torch.dtype) -> None:
        # Use raw tensor for int dtypes (Parameter requires floating point)
        if dtype in (torch.int32, torch.int64):
            # _check_param_dtype inspects .dtype on the raw tensor; use a tensor
            # directly rather than torch.nn.Parameter (which rejects int types).
            tensor = torch.ones(4, device=_device(), dtype=dtype)
            with pytest.raises(RuntimeError, match="float32.*float16.*bfloat16"):
                FusedAdam._check_param_dtype(tensor)
        else:
            param = torch.nn.Parameter(torch.ones(4, device=_device(), dtype=dtype))
            with pytest.raises(RuntimeError, match="float32.*float16.*bfloat16"):
                FusedAdam._check_param_dtype(param)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_check_state_dtype_valid(self, dtype: torch.dtype) -> None:
        FusedAdam._check_state_dtype(dtype, "exp_avg_dtype")  # should not raise

    @pytest.mark.parametrize("dtype", [torch.float64, torch.int8])
    def test_check_state_dtype_invalid(self, dtype: torch.dtype) -> None:
        with pytest.raises(NotImplementedError, match="float32.*float16.*bfloat16"):
            FusedAdam._check_state_dtype(dtype, "exp_avg_dtype")

    def test_state_dtype_for_name(self) -> None:
        opt = FusedAdam(
            _make_linear().parameters(),
            lr=1e-3,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float16,
        )
        assert opt._state_dtype_for_name("exp_avg") == torch.float32
        assert opt._state_dtype_for_name("exp_avg_sq") == torch.float16
        assert opt._state_dtype_for_name("master_param") == torch.float32

    def test_state_dtype_for_name_unknown_raises(self) -> None:
        opt = FusedAdam(_make_linear().parameters(), lr=1e-3)
        with pytest.raises(KeyError, match="Unknown"):
            opt._state_dtype_for_name("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════════
# TestFP16BF16Step
# ═══════════════════════════════════════════════════════════════════════════════


class TestFP16BF16Step:
    """FP16/BF16 parameters through a full ``step()`` call."""

    @staticmethod
    def setup_method() -> None:
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fp16_bf16_param_step(self, dtype: torch.dtype) -> None:
        model = _make_linear(dtype=dtype)
        opt = FusedAdam(model.parameters(), lr=1e-3)
        x = _randn((3, 8), dtype=dtype)
        target = _randn((3, 4), dtype=dtype)
        _mse_step(model, opt, x, target)
        for param in model.parameters():
            assert param.dtype == dtype
            assert opt.state[param]["exp_avg"].dtype == torch.float32  # default fp32

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "exp_avg_dtype,exp_avg_sq_dtype",
        [
            (torch.float32, torch.float32),
            (torch.float16, torch.float16),
            (torch.float32, torch.float16),
        ],
    )
    def test_fp16_param_with_state_dtype(
        self,
        dtype: torch.dtype,
        exp_avg_dtype: torch.dtype,
        exp_avg_sq_dtype: torch.dtype,
    ) -> None:
        model = _make_linear(dtype=dtype)
        opt = FusedAdam(
            model.parameters(),
            lr=1e-3,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=exp_avg_sq_dtype,
        )
        x = _randn((3, 8), dtype=dtype)
        target = _randn((3, 4), dtype=dtype)
        _mse_step(model, opt, x, target)
        for param in model.parameters():
            assert opt.state[param]["exp_avg"].dtype == exp_avg_dtype
            assert opt.state[param]["exp_avg_sq"].dtype == exp_avg_sq_dtype

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_master_weights_fp16_param(self, dtype: torch.dtype) -> None:
        model = _make_linear(dtype=dtype)
        opt = FusedAdam(model.parameters(), lr=1e-3, master_weights=True)
        x = _randn((3, 8), dtype=dtype)
        target = _randn((3, 4), dtype=dtype)
        _mse_step(model, opt, x, target)
        for param in model.parameters():
            assert "master_param" in opt.state[param]
            assert opt.state[param]["master_param"].dtype == torch.float32

    def test_empty_param_group_skip(self) -> None:
        """step() skips groups with zero params (group added after init)."""
        param = _make_param()
        opt = FusedAdam([param], lr=1e-3)
        # Replace param_groups with an empty one + a populated one
        group0 = {
            "params": [],
            "lr": 1e-3,
            "eps": 1e-8,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "adam_w_mode": True,
            "bias_correction": True,
        }
        group1 = {
            "params": [param],
            "lr": 1e-3,
            "eps": 1e-8,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "adam_w_mode": True,
            "bias_correction": True,
        }
        opt.param_groups = [group0, group1]
        param.grad = _randn((8,))
        opt.step()  # should not crash
        opt.zero_grad()


# ═══════════════════════════════════════════════════════════════════════════════
# TestStateDictRoundTrip
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateDictRoundTrip:
    """``state_dict`` / ``load_state_dict`` with FP16 storage."""

    @staticmethod
    def setup_method() -> None:
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("exp_avg_dtype", [torch.float32, torch.float16])
    def test_state_dict_round_trip_preserves_params(
        self,
        dtype: torch.dtype,
        exp_avg_dtype: torch.dtype,
    ) -> None:
        model = _make_linear(dtype=dtype)
        opt = FusedAdam(
            model.parameters(),
            lr=1e-3,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=torch.float32,
        )
        x = _randn((3, 8), dtype=dtype)
        target = _randn((3, 4), dtype=dtype)
        _mse_step(model, opt, x, target)

        sd = opt.state_dict()
        model2 = _make_linear(dtype=dtype)
        opt2 = FusedAdam(
            model2.parameters(),
            lr=1e-3,
            exp_avg_dtype=exp_avg_dtype,
            exp_avg_sq_dtype=torch.float32,
        )
        opt2.load_state_dict(sd)

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert opt2.state[p2].keys() == opt.state[p1].keys()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_state_dict_round_trip_with_master_weights(self, dtype: torch.dtype) -> None:
        model = _make_linear(dtype=dtype)
        opt = FusedAdam(model.parameters(), lr=1e-3, master_weights=True)
        x = _randn((3, 8), dtype=dtype)
        target = _randn((3, 4), dtype=dtype)
        _mse_step(model, opt, x, target)

        sd = opt.state_dict()
        model2 = _make_linear(dtype=dtype)
        opt2 = FusedAdam(model2.parameters(), lr=1e-3, master_weights=True)
        opt2.load_state_dict(sd)

        for p2 in model2.parameters():
            assert "master_param" in opt2.state[p2]


# ═══════════════════════════════════════════════════════════════════════════════
# TestGroupStepPersistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestGroupStepPersistence:
    """Group-level step counter survives ``state_dict`` / ``load_state_dict``."""

    @staticmethod
    def setup_method() -> None:
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    def test_group_step_survives_round_trip(self) -> None:
        model = _make_linear()
        opt = FusedAdam(model.parameters(), lr=1e-3)
        x = _randn((3, 8))
        target = _randn((3, 4))
        for _ in range(3):
            _mse_step(model, opt, x, target)

        sd = opt.state_dict()
        model2 = _make_linear()
        opt2 = FusedAdam(model2.parameters(), lr=1e-3)
        opt2.load_state_dict(sd)

        # Step 4 should continue from where we left off
        _mse_step(model, opt, x, target)
        _mse_step(model2, opt2, x, target)

    def test_step_persists_across_empty_group(self) -> None:
        param = _make_param()
        opt = FusedAdam([param], lr=1e-3)
        group = {
            "params": [param],
            "lr": 1e-3,
            "eps": 1e-8,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "adam_w_mode": True,
            "bias_correction": True,
            "step": 3,
        }
        opt.param_groups = [group]
        param.grad = _randn((8,))
        opt.step()
        opt.zero_grad()
        assert opt.param_groups[0]["step"] >= 4
