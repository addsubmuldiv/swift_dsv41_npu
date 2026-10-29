# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8BlockScaling,
    HIF8DelayedScaling,
    MXFP8BlockScaling,
)
from transformer_engine.pytorch import Linear, quantized_model_init
from transformer_engine.pytorch.quantization import (
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_hif8_delayed_available as is_hif8_available,
    is_mxfp8_available,
)

from utils import npu_available


# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
    return_reason=True
)


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


# FP8 per tensor delayed scaling
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFP8Recipe:
    @classmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)

    @pytest.mark.parametrize(
        "model_init_recipe",
        [
            pytest.param(
                MXFP8BlockScaling(),
                marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
            ),
            pytest.param(
                Float8BlockScaling(),
                marks=pytest.mark.skipif(
                    not fp8_block_scaling_available,
                    reason=reason_for_no_fp8_block_scaling,
                ),
            ),
        ],
    )
    def test_check_for_weight_tensor_and_recipe_correspondence(self, model_init_recipe):
        with quantized_model_init(enabled=True, recipe=model_init_recipe):
            linear = Linear(32, 32, params_dtype=torch.bfloat16, device="npu")

        x = torch.randn(32, 32, dtype=torch.bfloat16, device="cpu").to(device="npu")
        with te.autocast(enabled=True, recipe=DelayedScaling()):
            with pytest.raises(RuntimeError) as excinfo:
                _ = linear(x)
            assert "Recipe mismatch for " in str(excinfo.value)


class TestBackwardOverrideRecipeValidation:
    """backward_override must be validated on every recipe that declares it."""

    @pytest.mark.parametrize(
        "recipe_cls",
        [DelayedScaling, HIF8DelayedScaling],
        ids=["delayed_scaling", "hif8_delayed"],
    )
    def test_delayed_style_recipes_reject_backward_override(self, recipe_cls):
        # One mode per recipe suffices: the assertion is identical for both.
        with pytest.raises(AssertionError, match="backward_override"):
            recipe_cls(backward_override="high_precision")

    @pytest.mark.parametrize(
        "recipe_cls",
        [
            DelayedScaling,
            Float8BlockScaling,
            MXFP8BlockScaling,
            pytest.param(
                HIF8DelayedScaling,
                marks=pytest.mark.skipif(
                    not is_hif8_available(),
                    reason="HiF8 requires NPU support",
                ),
            ),
        ],
        ids=["delayed_scaling", "float8_block", "mxfp8_block", "hif8_delayed"],
    )
    def test_invalid_backward_override_value_is_rejected(self, recipe_cls):
        with pytest.raises(AssertionError, match="NVTE_BACKWARD_OVERRIDE"):
            recipe_cls(backward_override="invalid_mode")

    def test_default_backward_override_is_none(self):
        # Single representative suffices – default is a dataclass field, not hardware path.
        assert MXFP8BlockScaling().backward_override is None
