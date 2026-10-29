# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``multi_tensor_compute_scale.py``:
- ``multi_tensor_compute_scale_and_scale_inv``
- ``multi_tensor_compute_scale_inv_e8m0`` (MXFP8)
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.optimizers.multi_tensor_compute_scale import (
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")
_EPS = 0.0
_MAX_FP8 = 448.0  # float8_e4m3fn max


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# multi_tensor_compute_scale_and_scale_inv
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiTensorComputeScaleAndScaleInv:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_invalid_tensor_list_count_raises(self) -> None:
        """Outer tensor_lists must have exactly 3 inner lists (amax, scale, scale_inv)."""
        with pytest.raises(AssertionError, match="exactly 3 elements"):
            multi_tensor_compute_scale_and_scale_inv(
                chunk_size=0,
                noop_flag=torch.zeros(1, dtype=torch.int32, device=_NPU),
                tensor_lists=[
                    [torch.tensor([1.0], device=_NPU)],
                    [torch.zeros(1, device=_NPU)],
                ],  # only 2
                max_fp8=_MAX_FP8,
                force_pow_2_scales=False,
                epsilon=_EPS,
            )

    @pytest.mark.parametrize(
        ("amax_value", "force_pow_2_scales", "expected_scale", "expected_scale_inv"),
        [
            (4.0, False, 112.0, 1.0 / 112.0),
            (4.0, True, 64.0, 1.0 / 64.0),
            (float("inf"), True, 0.0, float("inf")),
        ],
    )
    def test_amax_to_scale_inv(
        self,
        amax_value: float,
        force_pow_2_scales: bool,
        expected_scale: float,
        expected_scale_inv: float,
    ) -> None:
        """amax=4.0, max_fp8=448.0 → scale = 448/4 = 112, scale_inv = 1/112."""
        amax = torch.tensor([amax_value], device=_NPU)
        scale = torch.zeros(1, device=_NPU)
        scale_inv = torch.zeros(1, device=_NPU)

        multi_tensor_compute_scale_and_scale_inv(
            chunk_size=0,
            noop_flag=torch.zeros(1, dtype=torch.int32, device=_NPU),
            tensor_lists=[[amax], [scale], [scale_inv]],
            max_fp8=_MAX_FP8,
            force_pow_2_scales=force_pow_2_scales,
            epsilon=_EPS,
        )

        assert abs(scale.item() - expected_scale) < 1e-3
        if expected_scale_inv == float("inf"):
            assert torch.isinf(scale_inv).all()
        else:
            assert abs(scale_inv.item() - expected_scale_inv) < 1e-5

    def test_zero_amax_finite_scale(self) -> None:
        """amax=0 → final where(amax==0, 1.0) overwrites scale to 1.0; not inf."""
        amax = torch.tensor([0.0], device=_NPU)
        scale = torch.zeros(1, device=_NPU)
        scale_inv = torch.zeros(1, device=_NPU)

        multi_tensor_compute_scale_and_scale_inv(
            chunk_size=0,
            noop_flag=torch.zeros(1, dtype=torch.int32, device=_NPU),
            tensor_lists=[[amax], [scale], [scale_inv]],
            max_fp8=_MAX_FP8,
            force_pow_2_scales=False,
            epsilon=_EPS,
        )

        # Final where(amax==0, 1.0) sets scale to 1.0
        assert abs(scale.item() - 1.0) < 1e-3
        assert abs(scale_inv.item() - 1.0) < 1e-3

    def test_multiple_tensors_in_list(self) -> None:
        """Loop iterates over all tensor triples."""
        amax1 = torch.tensor([4.0], device=_NPU)
        amax2 = torch.tensor([2.0], device=_NPU)
        scale1 = torch.zeros(1, device=_NPU)
        scale2 = torch.zeros(1, device=_NPU)
        scale_inv1 = torch.zeros(1, device=_NPU)
        scale_inv2 = torch.zeros(1, device=_NPU)

        multi_tensor_compute_scale_and_scale_inv(
            chunk_size=0,
            noop_flag=torch.zeros(1, dtype=torch.int32, device=_NPU),
            tensor_lists=[[amax1, amax2], [scale1, scale2], [scale_inv1, scale_inv2]],
            max_fp8=_MAX_FP8,
            force_pow_2_scales=False,
            epsilon=_EPS,
        )

        assert abs(scale1.item() - 112.0) < 1e-3
        assert abs(scale2.item() - 224.0) < 1e-3

    def test_in_place_modification(self) -> None:
        """The function writes to existing scale/scale_inv tensors (in-place)."""
        amax = torch.tensor([4.0], device=_NPU)
        scale = torch.full((1,), 99.0, device=_NPU)  # sentinel
        scale_inv = torch.full((1,), 99.0, device=_NPU)

        multi_tensor_compute_scale_and_scale_inv(
            chunk_size=0,
            noop_flag=torch.zeros(1, dtype=torch.int32, device=_NPU),
            tensor_lists=[[amax], [scale], [scale_inv]],
            max_fp8=_MAX_FP8,
            force_pow_2_scales=False,
            epsilon=_EPS,
        )

        # Both overwritten
        assert scale.item() != 99.0
        assert scale_inv.item() != 99.0


# ═══════════════════════════════════════════════════════════════════════════════
# multi_tensor_compute_scale_inv_e8m0
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiTensorComputeScaleInvE8M0:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_dummy_not_none_raises(self) -> None:
        """E8M0 doesn't support noop_flag — non-None dummy must raise."""
        with pytest.raises(RuntimeError, match="No-op flag is not supported"):
            multi_tensor_compute_scale_inv_e8m0(
                chunk_size=0,
                dummy=torch.tensor([1]),
                tensor_lists=[[], []],
            )

    def test_dummy_none_runs(self) -> None:
        """dummy=None → runs through amax list and writes E8M0 exponent."""
        amax = torch.tensor([4.0, 16.0, 1.0], dtype=torch.float32, device=_NPU)
        scale_inv = torch.zeros(3, dtype=torch.uint8, device=_NPU)

        multi_tensor_compute_scale_inv_e8m0(
            chunk_size=0,
            dummy=None,
            tensor_lists=[[amax], [scale_inv]],
        )

        # Outputs are now non-zero E8M0 exponents
        assert scale_inv.dtype == torch.uint8
        # Each element should be in [1, 254] (E8M0 reserved encoding range)
        assert (scale_inv >= 1).all().item()
        assert (scale_inv <= 254).all().item()
