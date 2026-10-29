# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""HiF8 delayed-scaling tests that execute only on real NPU hardware."""

import pickle

import pytest
import torch
import torch_npu  # noqa: F401

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    DelayedScaling,
    HIF8DelayedScaling,
    QuantizerRole,
)
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    RecipeState,
    is_hif8_delayed_available,
    reset_hif8_amax_history,
)
from transformer_engine.pytorch.tensor import HIF8Quantizer


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or not is_hif8_delayed_available(),
    reason="HiF8 policy tests require a real NPU",
)

DEVICE = torch.device("npu")


@pytest.fixture(autouse=True)
def _reset_global_state():
    FP8GlobalStateManager.reset()
    yield
    FP8GlobalStateManager.reset()


def _make_hif8_state(recipe, roles, mode="forward"):
    state = RecipeState.create(
        recipe,
        mode=mode,
        num_quantizers=len(roles),
        device=DEVICE,
        roles=roles,
    )
    meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=mode == "forward")
    meta = {
        meta_key: state,
        "recipe": recipe,
        "fp8_group": None,
    }
    FP8GlobalStateManager.register_hif8_states(meta)
    return state


def _flush(mode="forward"):
    FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=mode == "forward")


def _npu_tensor(values):
    return torch.tensor(values, dtype=torch.float32, device=DEVICE)


def test_recipe_role_margins_are_independent_from_delayed_scaling():
    defaults = HIF8DelayedScaling()
    assert defaults.fp8_max_fwd == defaults.fp8_max_bwd == 32768.0
    assert defaults.margin_factor(QuantizerRole.INPUT) == 2.0**11
    assert defaults.margin_factor(QuantizerRole.WEIGHT) == 2.0**12
    assert defaults.margin_factor(QuantizerRole.GRAD) == 2.0**11

    recipe = HIF8DelayedScaling(
        margin_factory=lambda role: {
            QuantizerRole.INPUT: 2.0,
            QuantizerRole.WEIGHT: 4.0,
            QuantizerRole.GRAD: 8.0,
        }[role]
    )
    state = _make_hif8_state(
        recipe,
        [QuantizerRole.INPUT, QuantizerRole.WEIGHT, QuantizerRole.GRAD],
    )

    assert recipe.hif8_delayed()
    assert not recipe.delayed()
    assert DelayedScaling().delayed()
    assert not FP8GlobalStateManager.quantization_state.global_amax_buffer
    assert all(isinstance(q, HIF8Quantizer) for q in state.make_quantizers())
    torch.testing.assert_close(state.margin_factors, _npu_tensor([2.0, 4.0, 8.0]))
    torch.testing.assert_close(
        state.scale_numerator,
        _npu_tensor([32768.0 / 2.0, 32768.0 / 4.0, 32768.0 / 8.0]),
    )


def test_region_cadence_uses_current_then_sparse_delayed_updates():
    recipe = HIF8DelayedScaling(
        amax_history_len=8,
        amax_collect_interval=3,
        scale_update_interval=2,
        reduce_amax=False,
    )
    state = _make_hif8_state(recipe, [QuantizerRole.INPUT])
    quantizer = state.make_quantizers()[0]
    numerator = state.scale_numerator[0].clone()

    observed_scale_inv = []
    scales_after_flush = []
    for amax in (4.0, 2.0, 8.0, 32.0, 12.0):
        tensor = _npu_tensor([[amax, -amax / 2.0]])
        output = quantizer.quantize_impl(tensor)
        observed_scale_inv.append(output._scale_inv.clone())
        _flush()
        scales_after_flush.append(state.scale.clone())

    torch.testing.assert_close(observed_scale_inv[0], (4.0 / numerator).reshape((1,)))
    torch.testing.assert_close(observed_scale_inv[1], (2.0 / numerator).reshape((1,)))
    torch.testing.assert_close(observed_scale_inv[2], (8.0 / numerator).reshape((1,)))
    torch.testing.assert_close(scales_after_flush[0], torch.ones_like(state.scale))
    torch.testing.assert_close(scales_after_flush[1], torch.ones_like(state.scale))
    torch.testing.assert_close(scales_after_flush[2], (numerator / 8.0).reshape((1,)))

    # Region 4 is not sampled. Region 5 quantizes with the old delayed
    # scale, then installs the sample-derived scale for region 6.
    torch.testing.assert_close(
        observed_scale_inv[3],
        torch.reciprocal(scales_after_flush[2]),
    )
    torch.testing.assert_close(
        observed_scale_inv[4],
        torch.reciprocal(scales_after_flush[3]),
    )
    torch.testing.assert_close(scales_after_flush[3], scales_after_flush[2])
    torch.testing.assert_close(scales_after_flush[4], (numerator / 12.0).reshape((1,)))
    assert state.completed_steps.tolist() == [5]
    assert not state.region_active.any()


def test_reset_clears_history_and_restarts_current_warmup():
    recipe = HIF8DelayedScaling(
        amax_history_len=4,
        amax_collect_interval=2,
        scale_update_interval=2,
        reduce_amax=False,
    )
    state = _make_hif8_state(recipe, [QuantizerRole.INPUT])
    quantizer = state.make_quantizers()[0]
    for amax in (4.0, 8.0, 2.0):
        quantizer.quantize_impl(_npu_tensor([[amax, -amax]]))
        _flush()

    state.amax_history.copy_(_npu_tensor([[float("inf")], [float("nan")], [7.0], [9.0]]))
    scale_before = state.scale.clone()
    reset_hif8_amax_history()

    assert not state.amax_history.any()
    assert not state.pending_amax.any()
    assert state.completed_steps.tolist() == [0]
    assert not state.region_active.any()
    torch.testing.assert_close(state.scale, scale_before)

    output = quantizer.quantize_impl(_npu_tensor([[2.0, -1.0]]))
    torch.testing.assert_close(
        output._scale_inv,
        (2.0 / state.scale_numerator[0]).reshape((1,)),
    )
    _flush()
    assert state.completed_steps.tolist() == [1]
    torch.testing.assert_close(state.scale, scale_before)


def test_linear_lifecycle_and_checkpoint_use_real_npu_state():
    recipe = HIF8DelayedScaling(
        hif8_grad_margin=10,
        amax_history_len=4,
        amax_collect_interval=2,
        scale_update_interval=2,
        reduce_amax=False,
    )
    layer = te.Linear(
        16,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        device=DEVICE,
    )
    inp = torch.randn(16, 16, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

    with te.autocast(enabled=True, recipe=recipe):
        output = layer(inp)
        loss = output.float().sum()
    loss.backward()
    torch.npu.synchronize()

    fwd_state = layer.fp8_meta["scaling_fwd"]
    bwd_state = layer.fp8_meta["scaling_bwd"]
    assert fwd_state.completed_steps[0].item() == 1
    assert fwd_state.completed_steps[1].item() == 1
    assert bwd_state.completed_steps[0].item() == 1
    torch.testing.assert_close(
        fwd_state.margin_factors[:2],
        _npu_tensor([2.0**11, 2.0**12]),
    )
    torch.testing.assert_close(
        bwd_state.margin_factors,
        torch.full_like(bwd_state.margin_factors, 2.0**10),
    )

    serialized = layer.get_extra_state()
    restored = te.Linear(
        16,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        device=DEVICE,
    )
    restored.set_extra_state(serialized)
    restored_fwd = restored.fp8_meta["scaling_fwd"]
    restored_bwd = restored.fp8_meta["scaling_bwd"]
    torch.testing.assert_close(restored_fwd.scale, fwd_state.scale)
    torch.testing.assert_close(restored_fwd.scale_inv, fwd_state.scale_inv)
    torch.testing.assert_close(restored_fwd.amax_history, fwd_state.amax_history)
    torch.testing.assert_close(
        restored_fwd.history_write_index,
        fwd_state.history_write_index,
    )
    torch.testing.assert_close(restored_fwd.completed_steps, fwd_state.completed_steps)
    torch.testing.assert_close(restored_bwd.completed_steps, bwd_state.completed_steps)


def test_hif8_extra_state_survives_grouped_checkpoint_merge():
    recipe = HIF8DelayedScaling(
        amax_history_len=4,
        amax_collect_interval=2,
        scale_update_interval=2,
        reduce_amax=False,
    )
    layer = te.Linear(16, 16, bias=False, params_dtype=torch.bfloat16, device=DEVICE)
    inp = torch.randn(16, 16, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

    with te.autocast(enabled=True, recipe=recipe):
        loss = layer(inp).float().sum()
    loss.backward()
    torch.npu.synchronize()

    saved = pickle.loads(layer.get_extra_state().detach().cpu().numpy().tobytes())
    # Megatron's grouped-linear distributed checkpoint merge only forwards the
    # recipe and extra_fp8_variables, dropping every other key.
    merged = {
        "recipe": saved["recipe"],
        "extra_fp8_variables": dict(saved["extra_fp8_variables"]),
    }
    assert "hif8_state_fwd" in merged["extra_fp8_variables"]

    restored = te.Linear(16, 16, bias=False, params_dtype=torch.bfloat16, device=DEVICE)
    restored.set_extra_state(torch.frombuffer(bytearray(pickle.dumps(merged)), dtype=torch.uint8))

    for mode in ("scaling_fwd", "scaling_bwd"):
        source = layer.fp8_meta[mode]
        target = restored.fp8_meta[mode]
        torch.testing.assert_close(target.scale, source.scale)
        torch.testing.assert_close(target.scale_inv, source.scale_inv)
        torch.testing.assert_close(target.amax_history, source.amax_history)
        torch.testing.assert_close(target.history_write_index, source.history_write_index)
        torch.testing.assert_close(target.completed_steps, source.completed_steps)


def test_resume_with_new_amax_history_len_resizes_instead_of_resetting():
    recipe = HIF8DelayedScaling(
        amax_history_len=4,
        amax_collect_interval=4,
        scale_update_interval=4,
        reduce_amax=False,
    )
    layer = te.Linear(16, 16, bias=False, params_dtype=torch.bfloat16, device=DEVICE)
    inp = torch.randn(16, 16, dtype=torch.bfloat16, device=DEVICE, requires_grad=True)

    with te.autocast(enabled=True, recipe=recipe):
        loss = layer(inp).float().sum()
    loss.backward()
    torch.npu.synchronize()

    fwd_state = layer.fp8_meta["scaling_fwd"]
    saved_scale = fwd_state.scale.clone()
    saved_steps = fwd_state.completed_steps.clone()
    saved_bwd_scale = layer.fp8_meta["scaling_bwd"].scale.clone()

    restored = te.Linear(16, 16, bias=False, params_dtype=torch.bfloat16, device=DEVICE)
    restored.set_extra_state(layer.get_extra_state())

    resumed_recipe = HIF8DelayedScaling(
        amax_history_len=10,
        amax_collect_interval=4,
        scale_update_interval=4,
        reduce_amax=False,
    )
    with te.autocast(enabled=True, recipe=resumed_recipe):
        restored(inp)
    torch.npu.synchronize()

    resumed_fwd = restored.fp8_meta["scaling_fwd"]
    resumed_bwd = restored.fp8_meta["scaling_bwd"]
    assert resumed_fwd.recipe.amax_history_len == 10
    assert resumed_fwd.amax_history.shape[0] == 10
    assert resumed_bwd.amax_history.shape[0] == 10
    # The rebuilt state inherits the checkpoint scale and region cadence.
    torch.testing.assert_close(resumed_fwd.scale, saved_scale)
    # Input/weight regions each advance by the resumed forward. The output
    # quantizer never activates in a plain Linear forward, so its counter
    # keeps the saved value.
    torch.testing.assert_close(resumed_fwd.completed_steps[:2], saved_steps[:2] + 1)
    torch.testing.assert_close(resumed_fwd.completed_steps[2], saved_steps[2])
    torch.testing.assert_close(resumed_bwd.scale, saved_bwd_scale)
