# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for CustomRecipe."""

from __future__ import annotations

import pytest
import torch

from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch import Linear, autocast
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
)
from transformer_engine.pytorch.quantization.quantizer_role import (
    DelayedScalingRequest,
    QuantizerRole,
)
from transformer_engine.pytorch.quantization.state import CustomRecipeState, RecipeState
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.utils import init_method_normal

from utils import npu_available


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_fp8_available, _reason_for_no_fp8 = is_fp8_available(return_reason=True)
_fp8_block_available, _reason_for_no_fp8_block = is_fp8_block_scaling_available(return_reason=True)
_mxfp8_available, _reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)


def _current_scaling_factory(role: QuantizerRole) -> Float8CurrentScalingQuantizer:
    return Float8CurrentScalingQuantizer(device="npu")


def _mxfp8_factory(role: QuantizerRole) -> MXFP8Quantizer:
    return MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)


_BLOCK_SCALING_DIMS = {
    "input": (1, 2),
    "output": (1, 2),
    "weight": 2,
    "grad_output": 1,
    "grad_input": 1,
}


def _block_scaling_factory(role: QuantizerRole) -> Float8BlockQuantizer:
    return Float8BlockQuantizer(
        fp8_dtype=torch.float8_e4m3fn,
        block_scaling_dim=_BLOCK_SCALING_DIMS.get(role.tensor_type, (1, 2)),
    )


def _delayed_scaling_factory(role: QuantizerRole) -> DelayedScalingRequest:
    return DelayedScalingRequest()


@pytest.fixture(autouse=True)
def _reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


def _make_state(recipe, mode, tensor_types):
    roles = [QuantizerRole(tensor_type=tensor_type) for tensor_type in tensor_types]
    state = RecipeState.create(
        recipe,
        mode=mode,
        num_quantizers=len(roles),
        roles=roles,
    )
    return state, state.make_quantizers()


class TestCustomRecipeState:
    def test_factory_receives_roles_and_builds_mixed_quantizers(self):
        seen_roles = []

        def qfactory(role):
            seen_roles.append(role)
            if role.tensor_type == "weight":
                return DelayedScalingRequest()
            return Float8CurrentScalingQuantizer(device="npu")

        state, quantizers = _make_state(
            CustomRecipe(qfactory=qfactory),
            "forward",
            ("input", "weight", "output"),
        )

        assert [role.tensor_type for role in seen_roles] == ["input", "weight", "output"]
        assert isinstance(quantizers[0], Float8CurrentScalingQuantizer)
        assert isinstance(quantizers[1], Float8Quantizer)
        assert isinstance(quantizers[2], Float8CurrentScalingQuantizer)
        assert state._has_delayed_scaling

    @pytest.mark.parametrize(
        ("qfactory", "error"),
        [
            (lambda role: None, "returned None"),
            (
                lambda role: DelayedScalingRequest(margin=0 if role.tensor_type == "input" else 1),
                "identical parameters",
            ),
        ],
    )
    def test_rejects_invalid_factory_results(self, qfactory, error):
        recipe = CustomRecipe(qfactory=qfactory)
        state = CustomRecipeState(
            recipe,
            mode="forward",
            num_quantizers=2,
            roles=[
                QuantizerRole(tensor_type="input"),
                QuantizerRole(tensor_type="weight"),
            ],
        )
        with pytest.raises(ValueError, match=error):
            state.make_quantizers()

    def test_rejects_role_count_mismatch(self):
        recipe = CustomRecipe(qfactory=_current_scaling_factory)
        with pytest.raises(ValueError, match="roles to match num_quantizers"):
            CustomRecipeState(
                recipe,
                mode="forward",
                num_quantizers=2,
                roles=[QuantizerRole()],
            )

    def test_inherit_preserves_delayed_scaling_buffers(self):
        recipe = CustomRecipe(qfactory=_delayed_scaling_factory)
        old_state, _ = _make_state(recipe, "forward", ("input", "weight"))
        old_scale = old_state.scale
        old_history = old_state.amax_history

        new_state = CustomRecipeState(
            recipe,
            mode="forward",
            num_quantizers=2,
            roles=[
                QuantizerRole(tensor_type="input"),
                QuantizerRole(tensor_type="weight"),
            ],
        )
        assert new_state.inherit_state_from(old_state)
        new_state.make_quantizers()

        assert new_state.scale.data_ptr() == old_scale.data_ptr()
        assert new_state.amax_history.data_ptr() == old_history.data_ptr()


_HIDDEN = 32
_NUM_TOKENS = 16


def _make_linear(dtype):
    return Linear(
        _HIDDEN,
        _HIDDEN,
        init_method=init_method_normal(0.023),
        params_dtype=dtype,
        device="npu",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestCustomRecipeLinear:
    @pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8)
    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(_current_scaling_factory, id="current"),
            pytest.param(_delayed_scaling_factory, id="delayed"),
            pytest.param(
                _block_scaling_factory,
                id="block",
                marks=pytest.mark.skipif(
                    not _fp8_block_available,
                    reason=_reason_for_no_fp8_block,
                ),
            ),
            pytest.param(
                _mxfp8_factory,
                id="mxfp8",
                marks=pytest.mark.skipif(
                    not _mxfp8_available,
                    reason=_reason_for_no_mxfp8,
                ),
            ),
        ],
    )
    def test_forward_backward_smoke(self, dtype, factory):
        torch.manual_seed(1234)
        torch.npu.manual_seed(1234)
        block = _make_linear(dtype)
        inp = torch.randn(
            (_NUM_TOKENS, _HIDDEN),
            dtype=dtype,
            device="npu",
            requires_grad=True,
        )

        with autocast(enabled=True, recipe=CustomRecipe(qfactory=factory)):
            out = block(inp)
        out.sum().backward()
        torch.npu.synchronize()

        assert out.shape == inp.shape
        assert block.weight.grad is not None
        assert inp.grad is not None
        assert torch.isfinite(out).all()
        assert torch.isfinite(block.weight.grad).all()
        assert torch.isfinite(inp.grad).all()
