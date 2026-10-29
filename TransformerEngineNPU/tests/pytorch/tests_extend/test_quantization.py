# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""A5 coverage for quantization contexts and GEMM."""

import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch.constants import NPUVersion, TensorUsage
from transformer_engine.pytorch.ops.gemm import general_gemm
from transformer_engine.pytorch.quantization import (
    autocast,
    is_fp8_available,
    is_mxfp8_available,
    is_w4a8_available,
)
from transformer_engine.pytorch.utils import check_npu_version
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")
_A5 = check_npu_version(NPUVersion.A5)


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestAutocast:
    def test_autocast_disabled_noop(self):
        inp = torch.randn(2, 4, device="cpu").to("npu")
        module = torch.nn.Linear(4, 4, device="npu")
        expected = module(inp)
        with autocast(enabled=False, recipe=recipe.DelayedScaling()):
            actual = module(inp)
        torch.testing.assert_close(actual.cpu(), expected.cpu())

    def test_autocast_enabled(self):
        inp = torch.randn(2, 4, device="cpu").to("npu")
        module = torch.nn.Linear(4, 4, device="npu")
        expected = module(inp)
        with autocast(enabled=True, recipe=recipe.DelayedScaling()):
            actual = module(inp)
        torch.testing.assert_close(actual.cpu(), expected.cpu())


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestIsFP8Available:
    @pytest.mark.parametrize(
        "availability_fn",
        [is_fp8_available, is_mxfp8_available, is_w4a8_available],
    )
    def test_reports_boolean_capability(self, availability_fn):
        available, reason = availability_fn(return_reason=True)
        assert isinstance(available, bool)
        assert reason is None or isinstance(reason, str)


@pytest.mark.skipif(not _A5, reason="ATLAS A5 REQUIRED")
class TestGemm:
    def test_general_gemm_float32(self):
        lhs = torch.randn(2, 4, device="cpu").to("npu")
        rhs = torch.randn(4, 8, device="cpu").to("npu")
        out = general_gemm(
            lhs,
            rhs,
            TensorUsage.LHS,
            TensorUsage.RHS,
            torch.float32,
        )
        torch.testing.assert_close(out.cpu(), (lhs @ rhs).cpu())
