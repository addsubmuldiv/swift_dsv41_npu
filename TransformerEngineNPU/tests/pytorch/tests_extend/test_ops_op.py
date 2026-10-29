# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``ops/op.py``:
- ``OperationContext`` dataclass: field defaults + save_for_backward
- ``BasicOperation`` abstract class via minimal concrete subclass
- ``FusedOperation`` is_fused_op=True
- Error paths: NotImplementedError, RuntimeError guards
- Serialization: get_extra_state, set_extra_state
- FusedOperation delegation methods
"""

from __future__ import annotations

import pickle
import pytest
import torch

from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    FusedOperation,
    OperationContext,
)
from utils import assert_close, dtype_tols, npu_available, to_cpu

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# OperationContext
# ═══════════════════════════════════════════════════════════════════════════════


class TestOperationContext:
    def test_default_fields(self) -> None:
        ctx = OperationContext()
        assert ctx.saved_tensors is None
        assert ctx.to_save is None
        assert ctx._saved_tensors_range is None
        assert ctx.requires_grad is True

    def test_save_for_backward_records_to_save(self) -> None:
        ctx = OperationContext()
        a = torch.tensor([1.0, 2.0], device=_NPU)
        b = torch.tensor([3.0, 4.0], device=_NPU)
        ctx.save_for_backward(a, b)
        assert ctx.to_save is not None
        assert len(ctx.to_save) == 2
        assert ctx.to_save[0] is a
        assert ctx.to_save[1] is b
        # saved_tensors still None (the fuser promotes to_save → saved_tensors)
        assert ctx.saved_tensors is None

    def test_save_for_backward_with_none(self) -> None:
        ctx = OperationContext()
        ctx.save_for_backward(None, torch.tensor([1.0], device=_NPU), None)
        assert len(ctx.to_save) == 3
        assert ctx.to_save[0] is None
        assert ctx.to_save[1] is not None
        assert ctx.to_save[2] is None

    def test_save_for_backward_overwrites(self) -> None:
        ctx = OperationContext()
        ctx.save_for_backward(torch.tensor([1.0], device=_NPU))
        first = ctx.to_save
        ctx.save_for_backward(torch.tensor([2.0], device=_NPU))
        assert ctx.to_save is not first

    def test_requires_grad_field_assignable(self) -> None:
        ctx = OperationContext(requires_grad=False)
        assert ctx.requires_grad is False


# ═══════════════════════════════════════════════════════════════════════════════
# BasicOperation — minimal concrete subclass (does NOT override is_fused_op,
# so tests below exercise the base-class default implementation)
# ═══════════════════════════════════════════════════════════════════════════════


class _MinimalBasicOp(BasicOperation):
    def op_forward(
        self,
        ctx,
        input_,
        *,
        prev_op_grad_output_quantizer,
        next_op_input_quantizer,
        **kwargs,
    ):
        return input_

    def op_backward(self, ctx, grad_output):
        return grad_output, ()


class TestBasicOperationDefaults:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_module_subclass(self) -> None:
        op = _MinimalBasicOp()
        assert isinstance(op, torch.nn.Module)

    def test_get_input_quantizer_default_none(self) -> None:
        op = _MinimalBasicOp()
        assert op.get_input_quantizer() is None

    def test_get_grad_output_quantizer_default_none(self) -> None:
        op = _MinimalBasicOp()
        assert op.get_grad_output_quantizer() is None

    def test_pre_first_fuser_forward_runs(self) -> None:
        op = _MinimalBasicOp()
        op.pre_first_fuser_forward()  # no-op, should not raise

    def test_pre_fuser_forward_runs(self) -> None:
        op = _MinimalBasicOp()
        op.pre_fuser_forward(requires_grad=True)  # no-op, should not raise

    def test_basic_op_is_fused_op_base_default_false(self) -> None:
        """Base-class BasicOperation.is_fused_op default is False (op.py base impl)."""
        op = _MinimalBasicOp()
        assert op.is_fused_op is False


# ═══════════════════════════════════════════════════════════════════════════════
# FusedOperation
# ═══════════════════════════════════════════════════════════════════════════════


class _FusedOp(FusedOperation):  # pylint: disable=abstract-method
    pass


class TestFusedOperation:
    def test_fused_op_default_is_fused(self) -> None:
        op = _FusedOp([_MinimalBasicOp()])
        # FusedOperation.is_fused_op returns True (op.py FusedOperation impl)
        assert op.is_fused_op is True

    def test_fused_op_is_module(self) -> None:
        op = _FusedOp([_MinimalBasicOp()])
        assert isinstance(op, torch.nn.Module)

    def test_fused_op_stores_basic_ops(self) -> None:
        ops = [_MinimalBasicOp(), _MinimalBasicOp()]
        op = _FusedOp(ops)
        assert len(op.basic_ops) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# FusedOperation error paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestFusedOperationErrorPaths:
    """Test FusedOperation error handling."""

    def test_fused_op_single_op_still_valid(self) -> None:
        """FusedOperation with single basic op should work."""
        op = _FusedOp([_MinimalBasicOp()])
        assert op.is_fused_op is True
        assert len(op.basic_ops) == 1

    def test_fused_op_get_input_quantizer_delegates(self) -> None:
        """FusedOperation.get_input_quantizer should delegate to first op."""
        op1 = _MinimalBasicOp()
        # op1.get_input_quantizer returns None by default
        fused = _FusedOp([op1])
        # The fused op delegates to its first basic op
        result = fused.get_input_quantizer()
        assert result is None  # since op1 returns None

    def test_fused_op_get_grad_output_quantizer_delegates(self) -> None:
        """FusedOperation.get_grad_output_quantizer should delegate to last op."""
        op1 = _MinimalBasicOp()
        fused = _FusedOp([op1])
        result = fused.get_grad_output_quantizer()
        assert result is None

    def test_fused_op_pre_first_fuser_forward_fans_out(self) -> None:
        """FusedOperation.pre_first_fuser_forward should call all basic ops."""
        op1 = _MinimalBasicOp()
        op2 = _MinimalBasicOp()
        fused = _FusedOp([op1, op2])
        # Should not raise - just fans out to ops
        fused.pre_first_fuser_forward()

    def test_fused_op_pre_fuser_forward_fans_out(self) -> None:
        """FusedOperation.pre_fuser_forward should call all basic ops."""
        op1 = _MinimalBasicOp()
        op2 = _MinimalBasicOp()
        fused = _FusedOp([op1, op2])
        # Should not raise - just fans out to ops
        fused.pre_fuser_forward(requires_grad=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BasicOperation extra inputs/outputs error guards
# ═══════════════════════════════════════════════════════════════════════════════


class TestBasicOperationExtraIO:
    """Test error guards when basic op has extra inputs/outputs but doesn't override fuser_forward."""

    def test_fuser_forward_with_extra_inputs_raises(self) -> None:
        """fuser_forward should raise RuntimeError when op has extra inputs but doesn't override."""

        class _OpWithExtraInput(BasicOperation):
            @property
            def num_extra_inputs(self) -> int:
                return 1  # signals extra inputs exist

            def op_forward(
                self,
                ctx,
                input_,
                *,
                prev_op_grad_output_quantizer,
                next_op_input_quantizer,
                **kwargs,
            ):
                return input_

            def op_backward(self, ctx, grad_output):
                return grad_output, ()

        op = _OpWithExtraInput()
        ctx = OperationContext()
        x = torch.tensor([1.0], device=_NPU)
        # Message (after f-string fix in op.py) embeds the concrete class name and counts
        with pytest.raises(
            RuntimeError,
            match=r"_OpWithExtraInput operation has 1 extra tensor inputs and 0 extra tensor outputs",
        ):
            op.fuser_forward(
                basic_op_ctxs=[ctx],
                input_=x,
                basic_op_extra_inputs=[torch.tensor([2.0], device=_NPU)],
                prev_op_grad_output_quantizer=None,
                next_op_input_quantizer=None,
                basic_op_kwargs=[{}],
            )

    def test_fuser_backward_with_extra_outputs_raises(self) -> None:
        """fuser_backward should raise RuntimeError when op has extra outputs but doesn't override."""

        class _OpWithExtraOutput(BasicOperation):
            @property
            def num_extra_outputs(self) -> int:
                return 1  # signals extra outputs exist

            def op_forward(
                self,
                ctx,
                input_,
                *,
                prev_op_grad_output_quantizer,
                next_op_input_quantizer,
                **kwargs,
            ):
                return input_

            def op_backward(self, ctx, grad_output):
                return grad_output, ()  # but internally signals extra outputs

        op = _OpWithExtraOutput()
        ctx = OperationContext()
        grad = torch.tensor([1.0], device=_NPU)
        # Message (after f-string fix in op.py) embeds the concrete class name and counts
        with pytest.raises(
            RuntimeError,
            match=r"_OpWithExtraOutput operation has 0 extra tensor inputs and 1 extra tensor outputs",
        ):
            op.fuser_backward(
                basic_op_ctxs=[ctx],
                grad_output=grad,
                basic_op_grad_extra_outputs=[torch.tensor([2.0], device=_NPU)],
            )


# ═══════════════════════════════════════════════════════════════════════════════
# BasicOperation serialization: get_extra_state / set_extra_state
# ═══════════════════════════════════════════════════════════════════════════════


class TestBasicOperationSerialization:
    """Test get_extra_state and set_extra_state for checkpoint/restore.

    NOTE: a full non-empty FP8-state round-trip cannot be exercised through
    ``_MinimalBasicOp``: with ``num_quantizers() == 0`` (the base default),
    ``reset_recipe_state`` leaves ``_fp8_metas[mode]`` as None, so
    ``get_extra_state`` always returns the empty tensor on this op, and feeding
    a hand-built non-empty payload into ``set_extra_state`` would hit
    ``fp8_meta = None`` in op.py. We therefore cover the pickle.loads branch
    with a non-empty byte tensor whose payload unpickles to an empty dict.
    """

    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_get_extra_state_returns_empty_for_no_fp8(self) -> None:
        """BasicOperation without FP8 should return empty extra state."""
        op = _MinimalBasicOp()
        state = op.get_extra_state()
        # Without FP8, should return empty byte tensor
        assert isinstance(state, torch.Tensor)
        assert state.dtype == torch.uint8
        assert state.numel() == 0

    def test_set_extra_state_with_empty_state(self) -> None:
        """set_extra_state should handle empty/None state gracefully."""
        op = _MinimalBasicOp()
        # Empty byte tensor
        empty_state = torch.zeros(0, dtype=torch.uint8, device="cpu")
        op.set_extra_state(empty_state)
        # Should not raise

    def test_set_extra_state_with_none(self) -> None:
        """set_extra_state should handle None state."""
        op = _MinimalBasicOp()
        op.set_extra_state(None)
        # Should not raise

    def test_set_extra_state_empty_pickled_payload(self) -> None:
        """Non-empty byte tensor whose pickled payload is an empty dict must be a
        no-op: this exercises the pickle.loads branch in set_extra_state (op.py)
        and the ``len(state) == 0`` early return, without changing op state.
        """
        op = _MinimalBasicOp()
        payload = torch.frombuffer(bytearray(pickle.dumps({})), dtype=torch.uint8)
        assert payload.numel() > 0  # non-empty byte tensor, unlike the empty-state path
        op.set_extra_state(payload)
        # Empty payload must not have touched the op's quantization state
        assert op._fp8_metas is None
        assert op._quantizers is None


# ═══════════════════════════════════════════════════════════════════════════════
# BasicOperation forward/backward through fuser
# ═══════════════════════════════════════════════════════════════════════════════


class TestBasicOperationFuserForward:
    """Test BasicOperation.forward which wraps through OperationFuser."""

    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_basic_op_forward_through_fuser(self) -> None:
        """BasicOperation.forward should work through the fuser."""
        op = _MinimalBasicOp()
        x = torch.randn(2, 4, device=_NPU, requires_grad=True)
        out = op.forward(x)
        assert out.shape == x.shape
        assert out.requires_grad
        # Identity op: output values must equal the input
        assert_close(to_cpu(out), to_cpu(x), **dtype_tols(x.dtype))

    def test_basic_op_backward_through_fuser(self) -> None:
        """Backward through fuser: identity op must pass the gradient through as all-ones."""
        op = _MinimalBasicOp()
        x = torch.randn(2, 4, device=_NPU, requires_grad=True)
        out = op.forward(x)
        loss = out.sum()
        loss.backward()
        # op_forward is the identity, so d(sum(out))/dx is exactly ones
        assert_close(to_cpu(x.grad), to_cpu(torch.ones_like(x)), **dtype_tols(x.dtype))
