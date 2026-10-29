# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Behavioral coverage for recipe-state factory and quantizer layouts."""

from __future__ import annotations

import pytest
import torch

from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,
    MXFP4BlockScaling,
    MXFP8BlockScaling,
    W4A8BlockScaling,
    W4A16BlockScaling,
)
from transformer_engine.pytorch import Linear, autocast, quantized_model_init
from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.quantization.state.base import RecipeState
from transformer_engine.pytorch.quantization.quantizer_role import QuantizerRole
from transformer_engine.pytorch.tensor.dummy_tensor import DummyQuantizer, DummyTensor
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
)
from transformer_engine.pytorch.tensor.mxfp4_tensor import MXFP4Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.w4a8_tensor import W4A8WeightQuantizer
from transformer_engine.pytorch.utils import check_npu_version
from utils import npu_available

pytestmark = [
    pytest.mark.skipif(not npu_available(), reason="NPU device is required"),
    pytest.mark.skipif(not check_npu_version(NPUVersion.A5), reason="Atlas A5 is required"),
]


@pytest.mark.parametrize(
    ("recipe", "mode", "num_quantizers", "quantizer_type"),
    [
        (DelayedScaling(), "forward", 2, Float8Quantizer),
        (Float8CurrentScaling(), "backward", 2, Float8CurrentScalingQuantizer),
        (Float8BlockScaling(), "forward", 3, Float8BlockQuantizer),
        (Float8BlockScaling(), "backward", 2, Float8BlockQuantizer),
        (MXFP8BlockScaling(), "forward", 2, MXFP8Quantizer),
        (MXFP4BlockScaling(), "forward", 3, MXFP4Quantizer),
        (MXFP4BlockScaling(), "backward", 2, MXFP4Quantizer),
        (W4A8BlockScaling(), "forward", 3, None),
        (W4A8BlockScaling(), "backward", 3, MXFP8Quantizer),
    ],
)
def test_recipe_state_factory_builds_expected_quantizer_layout(
    recipe, mode, num_quantizers, quantizer_type
):
    state = RecipeState.create(
        recipe,
        mode=mode,
        num_quantizers=num_quantizers,
        device=torch.device("npu"),
    )
    quantizers = state.make_quantizers()

    assert state.recipe is recipe
    assert state.mode == mode
    assert len(quantizers) == num_quantizers
    if quantizer_type is not None:
        assert all(isinstance(quantizer, quantizer_type) for quantizer in quantizers)
    else:
        assert isinstance(quantizers[0], MXFP8Quantizer)
        assert isinstance(quantizers[1], W4A8WeightQuantizer)
        assert isinstance(quantizers[2], MXFP8Quantizer)


def test_float8_block_state_uses_role_specific_block_dimensions():
    recipe = Float8BlockScaling(
        x_block_scaling_dim=(1, 2),
        w_block_scaling_dim=2,
        grad_block_scaling_dim=1,
    )

    forward_quantizers = RecipeState.create(
        recipe,
        mode="forward",
        num_quantizers=3,
        device=torch.device("npu"),
    ).make_quantizers()
    backward_quantizers = RecipeState.create(
        recipe,
        mode="backward",
        num_quantizers=2,
        device=torch.device("npu"),
    ).make_quantizers()

    assert forward_quantizers[0].block_scaling_dim == (1, 2)
    assert forward_quantizers[1].block_scaling_dim == 2
    assert forward_quantizers[2].block_scaling_dim == (1, 2)
    assert backward_quantizers[0].block_scaling_dim == 1
    assert backward_quantizers[0].columnwise_group_quant_lhs_trans is True
    assert backward_quantizers[1].columnwise_group_quant_lhs_trans is False


def test_w4a8_forward_layout_preserves_hybrid_weight_quantizer():
    state = RecipeState.create(
        W4A8BlockScaling(),
        mode="forward",
        num_quantizers=3,
        device=torch.device("npu"),
    )
    input_quantizer, weight_quantizer, output_quantizer = state.make_quantizers()

    assert isinstance(input_quantizer, MXFP8Quantizer)
    assert isinstance(weight_quantizer, W4A8WeightQuantizer)
    assert isinstance(output_quantizer, MXFP8Quantizer)
    assert weight_quantizer.block_size == state.recipe.block_size


@pytest.mark.parametrize("recipe_cls", [W4A16BlockScaling, W4A8BlockScaling])
@pytest.mark.parametrize("scope", ["all", "moe-only", "linear-only", "close"])
@pytest.mark.parametrize("mode", ["forward", "backward"])
def test_scoped_recipe_quantizer_layout(recipe_cls, scope, mode):
    tensor_types = (
        ("input", "weight", "output") if mode == "forward" else ("grad_output", "grad_input")
    )
    roles = [
        QuantizerRole(module_type=module_type, tensor_type=tensor_type)
        for module_type in ("linear", "grouped_linear")
        for tensor_type in tensor_types
    ]
    baseline = RecipeState.create(
        recipe_cls(), mode=mode, num_quantizers=len(roles), roles=roles
    ).make_quantizers()
    quantizers = RecipeState.create(
        recipe_cls(quantization_scope=scope), mode=mode, num_quantizers=len(roles), roles=roles
    ).make_quantizers()
    target = {"moe-only": "grouped_linear", "linear-only": "linear"}.get(scope)
    for role, quantizer, original in zip(roles, quantizers, baseline):
        expected = (
            type(original)
            if scope != "close" and (target is None or role.module_type == target)
            else DummyQuantizer
        )
        # Exact match: reject subclass substitutions (e.g. DummyQuantizer).
        assert type(quantizer) is expected  # pylint: disable=unidiomatic-typecheck


@pytest.mark.parametrize("recipe_cls", [W4A16BlockScaling, W4A8BlockScaling])
def test_scoped_recipe_validation(recipe_cls):
    for mode, num_quantizers in (("forward", 3), ("backward", 2)):
        quantizers = RecipeState.create(
            recipe_cls(quantization_scope="close"), mode=mode, num_quantizers=num_quantizers
        ).make_quantizers()
        # Exact match: reject subclass substitutions (e.g. DummyQuantizer).
        assert all(type(quantizer) is DummyQuantizer for quantizer in quantizers)  # pylint: disable=unidiomatic-typecheck
    with pytest.raises(ValueError, match="quantization_scope"):
        recipe_cls(quantization_scope="invalid")
    with pytest.raises(ValueError, match="requires quantizer roles"):
        RecipeState.create(
            recipe_cls(quantization_scope="moe-only"), mode="forward", num_quantizers=3
        ).make_quantizers()


@pytest.mark.parametrize("scope", ["moe-only", "close"])
def test_w4a8_scoped_model_init_dense_forward(scope):
    recipe = W4A8BlockScaling(quantization_scope=scope)
    with quantized_model_init(enabled=True, recipe=recipe):
        module = Linear(64, 64, bias=False, params_dtype=torch.bfloat16, device="npu")
    assert isinstance(module.weight, DummyTensor)
    inp = torch.randn(32, 64, dtype=torch.bfloat16, device="npu")
    with torch.no_grad():
        expected = inp @ module.weight.dequantize().t()
        with autocast(enabled=True, recipe=recipe):
            output = module(inp)
    torch.testing.assert_close(output, expected)
