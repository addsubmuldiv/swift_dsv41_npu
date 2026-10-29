# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Activation recomputation (checkpoint) coverage.

Guards ``transformer_engine.pytorch.distributed.checkpoint`` for the TE base
modules (Linear, LayerNormLinear, LayerNormMLP, GroupedLinear, and the
performance grouped-linear path), including:

- forward memory reduction from checkpointing
- numerical equivalence of outputs and gradients with/without checkpointing
  across all module types, in bf16, FP8 delayed scaling, HiF8 delayed scaling,
  and MXFP8 block scaling, on both the reentrant and non-reentrant checkpoint
  paths, run as multi-microbatch steps with ``is_first_microbatch``
  (weight-workspace reuse/update) semantics, which also guards the MXFP8
  GroupedLinear dgrad crash under reentrant recompute when the no-grad first
  forward caches a rowwise-only workspace (issue #40 / commit 85ec4505)
- the FP8 meta stash/restore cycle that keeps the recompute forward
  numerically identical to the first forward
- the ``activation_recompute_forward`` context manager state machine
- fallback to ``torch.utils.checkpoint`` for non-TE functions
"""

import torch
import pytest

from transformer_engine.common import recipe
from transformer_engine.pytorch import autocast
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    is_fp8_available,
    is_hif8_delayed_available,
    is_mxfp8_available,
)

torch.manual_seed(1234)
device = torch.device("npu" if torch.npu.is_available() else "cpu")

_fp8_available, _reason_for_no_fp8 = is_fp8_available(return_reason=True)
_mxfp8_available, _reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
_hif8_available, _reason_for_no_hif8 = is_hif8_delayed_available(return_reason=True)

_HIDDEN = 64
_FFN = 128
_SPLITS = [64, 64]

_GROUPED_TYPES = ("grouped_linear", "performance_grouped_linear")
_MODULE_TYPES = (
    "linear",
    "layernorm_linear",
    "layernorm_mlp",
    "grouped_linear",
    "performance_grouped_linear",
)
_LINEAR_TYPES = ("linear", "layernorm_linear", "layernorm_mlp")


@pytest.fixture(autouse=True)
def _reset_fp8_state():
    """Keep FP8 global state from leaking across tests."""
    yield
    FP8GlobalStateManager.reset()


def _build_module(module_type: str) -> torch.nn.Module:
    """Build one trainable TE module of the requested type."""
    if module_type == "linear":
        from transformer_engine.pytorch.module.linear import Linear

        return Linear(_HIDDEN, _FFN, params_dtype=torch.bfloat16).to(device)
    if module_type == "layernorm_linear":
        from transformer_engine.pytorch.module.layernorm_linear import LayerNormLinear

        return LayerNormLinear(_HIDDEN, _FFN, params_dtype=torch.bfloat16).to(device)
    if module_type == "layernorm_mlp":
        from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

        return LayerNormMLP(_HIDDEN, _FFN, params_dtype=torch.bfloat16).to(device)
    if module_type == "grouped_linear":
        from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

        return GroupedLinear(
            num_gemms=2,
            in_features=_HIDDEN,
            out_features=_FFN,
            params_dtype=torch.bfloat16,
        ).to(device)
    if module_type == "performance_grouped_linear":
        from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

        module = GroupedLinear(
            num_gemms=2,
            in_features=_HIDDEN,
            out_features=_FFN,
            params_dtype=torch.bfloat16,
            single_grouped_weight=True,
        ).to(device)
        assert module._use_performance_grouped_linear()
        return module
    raise ValueError(f"unknown module type: {module_type}")


def _make_input(module_type: str) -> torch.Tensor:
    if module_type in _GROUPED_TYPES:
        return torch.randn(sum(_SPLITS), _HIDDEN, dtype=torch.bfloat16, device=device)
    return torch.randn(128, 2, _HIDDEN, dtype=torch.bfloat16, device=device)


def _forward(
    model,
    module_type,
    inp,
    use_checkpoint,
    use_reentrant=True,
    is_first_microbatch=None,
):
    from transformer_engine.pytorch.distributed import checkpoint

    kwargs = {"is_first_microbatch": is_first_microbatch}
    if module_type in _GROUPED_TYPES:
        kwargs["m_splits"] = _SPLITS
    if use_checkpoint:
        return checkpoint(model, inp, use_reentrant=use_reentrant, **kwargs)
    return model(inp, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical equivalence across modules, FP8 recipes, and checkpoint modes
# ═══════════════════════════════════════════════════════════════════════════════


def _assert_recompute_matches_reference(
    module_type, fp8_recipe, use_reentrant, *, warmup_steps, rtol, atol
):
    """Assert checkpointed fwd/bwd equals the non-checkpointed reference.

    Shared by the recipe-parameterized equivalence test and the HiF8-specific
    test, which needs a longer warmup to cross the delayed-scaling cadence.
    Each training step runs as two accumulating microbatches with
    ``is_first_microbatch`` weight-workspace semantics (see ``_train_step``).
    """
    reference = _build_module(module_type)
    checkpointed = _build_module(module_type)
    checkpointed.load_state_dict(reference.state_dict())

    data = _make_input(module_type)
    ref_inp = data.detach().clone().requires_grad_(True)
    ckp_inp = data.detach().clone().requires_grad_(True)

    def _train_step(model, inp, use_checkpoint, collect=False):
        """One training step as two accumulating microbatches.

        The sequence deliberately starts with ``is_first_microbatch=False``:
        on a fresh module under the reentrant checkpoint, the first forward
        runs inside ``torch.no_grad()`` and caches the FP8 weight workspace,
        while the recompute forward replays ``is_first_microbatch=False`` and
        reuses that workspace without re-quantizing. Unless the first forward
        forces columnwise usage (commit 85ec4505), the cached MXFP8 workspace
        is rowwise-only and dgrad crashes with ``columnwise_data=None``
        (issue #40). The trailing ``is_first_microbatch=True`` microbatch then
        exercises the grad-enabled workspace update path. Gradients accumulate
        across microbatches like Megatron's grad accumulation.
        """
        model.zero_grad(set_to_none=False)
        inp.grad = None
        outs = []
        with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
            for is_first_microbatch in (False, True):
                out = _forward(
                    model,
                    module_type,
                    inp,
                    use_checkpoint,
                    use_reentrant,
                    is_first_microbatch,
                )
                out.sum().backward()
                if collect:
                    outs.append(out.detach().clone())
        if not collect:
            return None
        param_grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        return outs, param_grads, inp.grad.detach().clone()

    # Warm up both models identically so delayed-scaling amax histories converge.
    for _ in range(warmup_steps):
        _train_step(reference, ref_inp, use_checkpoint=False)
        _train_step(checkpointed, ckp_inp, use_checkpoint=True)

    ref_outs, ref_grads, ref_input_grad = _train_step(
        reference, ref_inp, use_checkpoint=False, collect=True
    )
    ckp_outs, ckp_grads, ckp_input_grad = _train_step(
        checkpointed, ckp_inp, use_checkpoint=True, collect=True
    )

    assert len(ckp_outs) == len(ref_outs)
    for microbatch, (ckp_out, ref_out) in enumerate(zip(ckp_outs, ref_outs)):
        torch.testing.assert_close(
            ckp_out,
            ref_out,
            rtol=rtol,
            atol=atol,
            msg=f"microbatch {microbatch} forward output mismatch",
        )
    torch.testing.assert_close(
        ckp_input_grad,
        ref_input_grad,
        rtol=rtol,
        atol=atol,
        msg="input gradient mismatch",
    )
    assert ckp_grads.keys() == ref_grads.keys()
    for name, ref_grad in ref_grads.items():
        torch.testing.assert_close(
            ckp_grads[name],
            ref_grad,
            rtol=rtol,
            atol=atol,
            msg=f"gradient mismatch for {name}",
        )


@pytest.mark.parametrize(
    "use_reentrant",
    [pytest.param(True, id="reentrant"), pytest.param(False, id="non_reentrant")],
)
@pytest.mark.parametrize(
    "fp8_recipe_cls",
    [
        pytest.param(None, id="no_fp8"),
        pytest.param(
            recipe.DelayedScaling,
            marks=pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8),
            id="delayed_scaling",
        ),
        pytest.param(
            recipe.MXFP8BlockScaling,
            marks=pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8),
            id="mxfp8",
        ),
    ],
)
@pytest.mark.parametrize("module_type", _MODULE_TYPES)
def test_activation_recompute_matches_reference(module_type, fp8_recipe_cls, use_reentrant):
    """Checkpointed fwd/bwd must match the non-checkpointed reference.

    This exercises the FP8 meta stash/restore path in
    TransformerEngineBaseModule (copy/get_old/restore of delayed-scaling
    amax and scale tensors) so both forward phases are numerically identical.
    """
    fp8_recipe = None if fp8_recipe_cls is None else fp8_recipe_cls()
    if fp8_recipe is None:
        rtol, atol = 0.0, 0.0
    else:
        rtol, atol = 0.125, 0.0675
    _assert_recompute_matches_reference(
        module_type, fp8_recipe, use_reentrant, warmup_steps=3, rtol=rtol, atol=atol
    )


@pytest.mark.skipif(not _hif8_available, reason=_reason_for_no_hif8)
@pytest.mark.parametrize(
    "use_reentrant",
    [pytest.param(True, id="reentrant"), pytest.param(False, id="non_reentrant")],
)
@pytest.mark.parametrize("module_type", _LINEAR_TYPES)
def test_activation_recompute_hif8_delayed(module_type, use_reentrant):
    """HiF8 delayed-scaling snapshot/restore keeps recompute numerically identical.

    HiF8 uses a distinct recompute stash/restore path (``snapshot_for_recompute``
    / ``load_recompute_snapshot``) from plain delayed scaling, so it needs its
    own guard. Its cadence must cross ``amax_collect_interval`` (default 5)
    before entering delayed scaling, hence the longer warmup.
    """
    fp8_recipe = recipe.HIF8DelayedScaling(reduce_amax=False)
    _assert_recompute_matches_reference(
        module_type,
        fp8_recipe,
        use_reentrant,
        warmup_steps=6,
        rtol=0.125,
        atol=0.0675,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Forward memory reduction
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device is required")
def test_checkpoint_reduces_forward_memory():
    """Checkpointing must reduce peak forward memory and preserve outputs."""
    from transformer_engine.pytorch.distributed import checkpoint
    from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

    def _build_layers():
        return [LayerNormMLP(128, 512, params_dtype=torch.bfloat16).to(device) for _ in range(2)]

    class _CheckpointedSequential(torch.nn.Module):
        """Sequential model that applies checkpoint to each module."""

        def __init__(self, modules):
            super().__init__()
            self.modules_list = torch.nn.ModuleList(modules)

        def forward(self, x):
            for module in self.modules_list:
                x = checkpoint(module, x)
            return x

    reference = torch.nn.Sequential(*_build_layers())
    checkpointed = _CheckpointedSequential(_build_layers())
    # Copy parameters layer-by-layer; state_dict keys differ due to the
    # ModuleList vs Sequential nesting prefix.
    with torch.no_grad():
        for ref_layer, ckp_layer in zip(reference, checkpointed.modules_list):
            ckp_layer.load_state_dict(ref_layer.state_dict())

    base = torch.randn(128, 1, 128, dtype=torch.bfloat16, device=device)
    ref_data = base.detach().clone().requires_grad_(True)
    ckp_data = base.detach().clone().requires_grad_(True)

    def _forward_peak_memory(model, tensor):
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.synchronize()
        start_mem = torch.npu.memory_allocated(device)
        out = model(tensor)
        torch.npu.synchronize()
        peak_mem = torch.npu.max_memory_allocated(device)
        return out, float(peak_mem - start_mem)

    # Warm up both models so allocator behavior is comparable.
    for _ in range(3):
        reference(ref_data).sum().backward()
        checkpointed(ckp_data).sum().backward()

    ref_out, ref_mem = _forward_peak_memory(reference, ref_data)
    ckp_out, ckp_mem = _forward_peak_memory(checkpointed, ckp_data)

    torch.testing.assert_close(ckp_out, ref_out, rtol=0.0, atol=0.0)
    assert ref_mem > ckp_mem, (
        f"checkpointing does not reduce forward memory, reference={ref_mem}, checkpointed={ckp_mem}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Recompute mechanism unit tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_recompute_context_manager_flags():
    """The recompute context manager must track phase globals and the
    is_first_fp8_module stack used to restore module order across phases.
    """
    from transformer_engine.pytorch.distributed import (
        activation_recompute_forward,
        in_fp8_activation_recompute_phase,
        is_fp8_activation_recompute_enabled,
    )

    activation_recompute_forward._is_first_fp8_module.clear()
    qstate = FP8GlobalStateManager.quantization_state
    original_is_first = qstate.is_first_fp8_module

    assert not is_fp8_activation_recompute_enabled()
    assert not in_fp8_activation_recompute_phase()

    try:
        with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
            assert is_fp8_activation_recompute_enabled()
            assert not in_fp8_activation_recompute_phase()

            # A later TE module flips the flag during the first forward phase;
            # the recompute phase must restore the stashed value on entry.
            qstate.is_first_fp8_module = not original_is_first
            with activation_recompute_forward(activation_recompute=True, recompute_phase=True):
                assert in_fp8_activation_recompute_phase()
                assert is_fp8_activation_recompute_enabled()
                assert qstate.is_first_fp8_module is original_is_first
        assert not is_fp8_activation_recompute_enabled()
        assert not in_fp8_activation_recompute_phase()
        assert not activation_recompute_forward._is_first_fp8_module
    finally:
        qstate.is_first_fp8_module = original_is_first
        activation_recompute_forward._is_first_fp8_module.clear()


@pytest.mark.skipif(not _fp8_available, reason=_reason_for_no_fp8)
def test_fp8_meta_stash_restore_cycle():
    """Direct guard on the delayed-scaling meta stash/restore cycle.

    Phase-1 forward stashes scale/amax; the recompute forward must switch
    back to the stashed values while preserving the updated ones, and
    restore them after the recompute phase ends.
    """
    from transformer_engine.pytorch.module.linear import Linear

    module = Linear(_HIDDEN, _FFN, params_dtype=torch.bfloat16).to(device)
    inp = torch.randn(128, 2, _HIDDEN, dtype=torch.bfloat16, device=device)
    with autocast(enabled=True, recipe=recipe.DelayedScaling()):
        module(inp).sum().backward()

    fp8_meta = module.fp8_meta
    scale = fp8_meta["scaling_fwd"].scale
    amax_history = fp8_meta["scaling_fwd"].amax_history
    stashed_scale = scale.clone()
    stashed_amax = amax_history.clone()

    # Phase-1 forward: stash the current scale/amax for the recompute phase.
    FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(fp8_meta)
    assert "global_fp8_buffer_pos_fwd_recompute" in fp8_meta

    # Simulate the amax/scale update at the end of the phase-1 forward.
    updated_scale = stashed_scale * 2.0
    updated_amax = stashed_amax + 1.0
    scale.copy_(updated_scale)
    amax_history.copy_(updated_amax)

    # Phase-2 (recompute) forward must switch back to the stashed values.
    FP8GlobalStateManager.get_old_fp8_meta_tensors_for_recompute(fp8_meta)
    torch.testing.assert_close(scale, stashed_scale)
    torch.testing.assert_close(amax_history, stashed_amax)

    # The updated values must be preserved for restoration after recompute.
    torch.testing.assert_close(fp8_meta["updated_scale_fwd"], updated_scale)
    torch.testing.assert_close(fp8_meta["updated_amax_history_fwd"], updated_amax)

    # After the recompute forward, the updated values must be restored.
    FP8GlobalStateManager.restore_fp8_meta_tensors(fp8_meta)
    torch.testing.assert_close(scale, updated_scale)
    torch.testing.assert_close(amax_history, updated_amax)


@pytest.mark.parametrize(
    "use_reentrant",
    [pytest.param(True, id="reentrant"), pytest.param(False, id="non_reentrant")],
)
def test_checkpoint_non_te_function_falls_back_to_torch(use_reentrant):
    """Non-TE functions must transparently fall back to torch.utils.checkpoint."""
    from transformer_engine.pytorch.distributed import checkpoint

    module = torch.nn.Linear(_HIDDEN, _FFN, dtype=torch.bfloat16, device=device)
    inp = torch.randn(128, _HIDDEN, dtype=torch.bfloat16, device=device)
    ckp_inp = inp.detach().clone().requires_grad_(True)
    ref_inp = inp.detach().clone().requires_grad_(True)

    ckp_out = checkpoint(module, ckp_inp, use_reentrant=use_reentrant)
    ref_out = module(ref_inp)
    torch.testing.assert_close(ckp_out, ref_out, rtol=0.0, atol=0.0)

    ckp_out.sum().backward()
    ref_out.sum().backward()
    torch.testing.assert_close(ckp_inp.grad, ref_inp.grad, rtol=0.0, atol=0.0)
