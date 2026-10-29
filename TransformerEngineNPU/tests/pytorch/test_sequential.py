# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for sequential.py."""

import pytest
import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_engine.pytorch.ops.sequential import Sequential
from transformer_engine.pytorch.ops.op import (
    OperationContext,
    BasicOperation,
)
from transformer_engine.pytorch.ops.fuser import OperationFuser


class SimpleLinearOp(BasicOperation):
    """A simple linear operation for testing: y = x * weight + bias"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def num_quantizers(self, mode: str) -> int:
        """No quantization for this simple op."""
        return 0

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: y = x @ weight.T + bias"""
        # Save for backward
        ctx.save_for_backward(input_)

        # Forward computation
        output = torch.nn.functional.linear(input_, self.weight, self.bias)
        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ):
        """Backward pass."""
        # Retrieve saved tensors
        input_ = ctx.saved_tensors[0]

        # Compute gradients
        grad_input = grad_output @ self.weight
        grad_weight = grad_output.T @ input_
        grad_bias = grad_output.sum(dim=0)

        return grad_input, [grad_weight, grad_bias]


class SimpleReLIOp(BasicOperation):
    """A simple ReLU operation for testing."""

    def num_quantizers(self, mode: str) -> int:
        """No quantization for this simple op."""
        return 0

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: y = ReLU(x)"""
        # Save for backward
        ctx.save_for_backward(input_)

        # Forward computation
        output = torch.nn.functional.relu(input_)
        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ):
        """Backward pass."""
        # Retrieve saved tensors
        input_ = ctx.saved_tensors[0]

        # Compute gradient (ReLU derivative)
        grad_input = grad_output * (input_ > 0).float()

        return grad_input, []


class ScaleOp(BasicOperation):
    """A simple scale operation for testing op_kwargs passthrough: y = x * scale.

    The scale factor is read from the ``scale`` forward kwarg when provided
    (forwarded via ``Sequential(op_kwargs=...)``), otherwise ``default_scale`` is used.
    """

    def __init__(self, default_scale: float = 1.0):
        super().__init__()
        self.default_scale = default_scale

    def num_quantizers(self, mode: str) -> int:
        """No quantization for this simple op."""
        return 0

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer=None,
        next_op_input_quantizer=None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: y = x * scale."""
        scale = kwargs.get("scale", self.default_scale)
        ctx.save_for_backward(input_, torch.as_tensor(scale))
        return input_ * scale

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ):
        """Backward pass."""
        input_, scale = ctx.saved_tensors
        grad_input = grad_output * scale
        return grad_input, []


class TestSequentialBasic:
    """Test basic Sequential container functionality."""

    def test_empty_sequential(self):
        """Test empty Sequential container."""
        seq = Sequential()
        assert len(seq) == 0
        assert not list(seq)

    def test_single_module(self):
        """Test Sequential with a single module."""
        linear = torch.nn.Linear(10, 5)
        seq = Sequential(linear)

        assert len(seq) == 1
        assert seq[0] is linear

    def test_multiple_modules(self):
        """Test Sequential with multiple modules."""
        linear1 = torch.nn.Linear(10, 5)
        relu = torch.nn.ReLU()
        linear2 = torch.nn.Linear(5, 3)

        seq = Sequential(linear1, relu, linear2)

        assert len(seq) == 3
        assert seq[0] is linear1
        assert seq[1] is relu
        assert seq[2] is linear2

    def test_sequential_with_dict(self):
        """Test Sequential initialized with dict."""
        linear1 = torch.nn.Linear(10, 5)
        relu = torch.nn.ReLU()

        seq = Sequential({'linear': linear1, 'relu': relu})

        assert len(seq) == 2
        # Access modules by name using getattr (torch.nn.Module interface)
        assert seq.linear is linear1
        assert seq.relu is relu


class TestSequentialOperations:
    """Test Sequential container operations."""

    def test_append(self):
        """Test append operation."""
        seq = Sequential()
        linear = torch.nn.Linear(10, 5)

        result = seq.append(linear)

        assert result is seq  # Should return self
        assert len(seq) == 1
        assert seq[0] is linear

    def test_extend(self):
        """Test extend operation."""
        seq = Sequential()
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        result = seq.extend([linear1, linear2])

        assert result is seq  # Should return self
        assert len(seq) == 2
        assert seq[0] is linear1
        assert seq[1] is linear2

    def test_insert(self):
        """Test insert operation."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)
        relu = torch.nn.ReLU()

        seq = Sequential(linear1, linear2)
        result = seq.insert(1, relu)

        assert result is seq  # Should return self
        assert len(seq) == 3
        assert seq[0] is linear1
        assert seq[1] is relu
        assert seq[2] is linear2

    def test_pop(self):
        """Test pop operation."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        seq = Sequential(linear1, linear2)
        popped = seq.pop(0)

        assert popped is linear1
        assert len(seq) == 1
        assert seq[0] is linear2

    def test_getitem_slice(self):
        """Test getitem with slice."""
        linear1 = torch.nn.Linear(10, 5)
        relu = torch.nn.ReLU()
        linear2 = torch.nn.Linear(5, 3)

        seq = Sequential(linear1, relu, linear2)
        sub_seq = seq[0:2]

        assert isinstance(sub_seq, Sequential)
        assert len(sub_seq) == 2
        assert sub_seq[0] is linear1
        assert sub_seq[1] is relu

    def test_setitem(self):
        """Test setitem operation."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)
        relu = torch.nn.ReLU()

        seq = Sequential(linear1, linear2)
        seq[1] = relu

        assert seq[1] is relu

    def test_delitem(self):
        """Test delitem operation."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        seq = Sequential(linear1, linear2)
        del seq[0]

        assert len(seq) == 1
        assert seq[0] is linear2

    def test_iadd(self):
        """Test in-place add operation."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        seq1 = Sequential(linear1)
        seq2 = Sequential(linear2)

        seq1 += seq2

        assert len(seq1) == 2
        assert seq1[0] is linear1
        assert seq1[1] is linear2

    def test_add(self):
        """Test add operation."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        seq1 = Sequential(linear1)
        seq2 = Sequential(linear2)

        result = seq1 + seq2

        assert len(result) == 2
        assert result[0] is linear1
        assert result[1] is linear2
        # Original sequences should be unchanged
        assert len(seq1) == 1
        assert len(seq2) == 1


class TestSequentialForward:
    """Test Sequential forward pass."""

    def test_forward_with_torch_modules(self):
        """Test forward pass with standard PyTorch modules."""
        linear = torch.nn.Linear(10, 5)
        relu = torch.nn.ReLU()

        seq = Sequential(linear, relu)

        input_tensor = torch.randn(3, 10)
        output = seq(input_tensor)

        # Check output shape
        assert output.shape == (3, 5)

        # Verify computation
        expected = relu(linear(input_tensor))
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_with_fusible_operations(self):
        """Test forward pass with FusibleOperation instances."""
        linear_op = SimpleLinearOp(in_features=10, out_features=5)
        relu_op = SimpleReLIOp()

        seq = Sequential(linear_op, relu_op)

        input_tensor = torch.randn(3, 10)
        output = seq(input_tensor)

        # Check output shape
        assert output.shape == (3, 5)

        # Verify computation
        expected_linear = torch.nn.functional.linear(input_tensor, linear_op.weight, linear_op.bias)
        expected = torch.nn.functional.relu(expected_linear)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_with_mixed_modules(self):
        """Test forward pass with mixed PyTorch and FusibleOperation modules."""
        linear_op = SimpleLinearOp(in_features=10, out_features=5)
        relu = torch.nn.ReLU()

        seq = Sequential(linear_op, relu)

        input_tensor = torch.randn(3, 10)
        output = seq(input_tensor)

        # Check output shape
        assert output.shape == (3, 5)

        # Verify computation
        expected_linear = torch.nn.functional.linear(input_tensor, linear_op.weight, linear_op.bias)
        expected = torch.nn.functional.relu(expected_linear)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_backward(self):
        """Test forward and backward pass."""
        linear_op = SimpleLinearOp(in_features=10, out_features=5)
        relu_op = SimpleReLIOp()

        seq = Sequential(linear_op, relu_op)

        input_tensor = torch.randn(3, 10, requires_grad=True)
        output = seq(input_tensor)

        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        # Check gradients
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape
        assert linear_op.weight.grad is not None
        assert linear_op.weight.grad.shape == linear_op.weight.shape
        assert linear_op.bias.grad is not None
        assert linear_op.bias.grad.shape == linear_op.bias.shape


class TestSequentialModuleGrouping:
    """Test Sequential module grouping for fusion."""

    def test_module_groups_creation(self):
        """Test that module groups are created correctly."""
        linear_op1 = SimpleLinearOp(in_features=10, out_features=5)
        linear_op2 = SimpleLinearOp(in_features=5, out_features=3)
        relu = torch.nn.ReLU()

        seq = Sequential(linear_op1, linear_op2, relu)

        # Trigger forward pass to create module groups
        input_tensor = torch.randn(3, 10)
        seq(input_tensor)

        # Check that module groups were created
        assert seq._module_groups is not None
        assert len(seq._module_groups) == 2  # One OperationFuser, one ReLU

        # First group should be OperationFuser (fused linear ops)
        assert isinstance(seq._module_groups[0], OperationFuser)
        # Second group should be ReLU
        assert seq._module_groups[1] is relu

    def test_module_groups_with_intervening_non_fusible(self):
        """Test module grouping with non-fusible operations in between."""
        linear_op1 = SimpleLinearOp(in_features=10, out_features=5)
        relu = torch.nn.ReLU()
        linear_op2 = SimpleLinearOp(in_features=5, out_features=3)

        seq = Sequential(linear_op1, relu, linear_op2)

        # Trigger forward pass
        input_tensor = torch.randn(3, 10)
        seq(input_tensor)

        # Check module groups
        assert seq._module_groups is not None
        assert len(seq._module_groups) == 3

        # Each fusible operation should be in its own OperationFuser
        assert isinstance(seq._module_groups[0], OperationFuser)
        assert seq._module_groups[1] is relu
        assert isinstance(seq._module_groups[2], OperationFuser)


class TestSequentialEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_index(self):
        """Test negative indexing."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        seq = Sequential(linear1, linear2)

        assert seq[-1] is linear2
        assert seq[-2] is linear1

    def test_index_out_of_range(self):
        """Test index out of range error."""
        linear = torch.nn.Linear(10, 5)
        seq = Sequential(linear)

        with pytest.raises(IndexError):
            _ = seq[5]

    def test_iteration(self):
        """Test iteration over modules."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        seq = Sequential(linear1, linear2)

        modules = list(seq)
        assert len(modules) == 2
        assert modules[0] is linear1
        assert modules[1] is linear2

    def test_len(self):
        """Test len operation."""
        linear1 = torch.nn.Linear(10, 5)
        linear2 = torch.nn.Linear(5, 3)

        seq = Sequential(linear1, linear2)
        assert len(seq) == 2

    def test_module_invalidation(self):
        """Test that module groups are invalidated on modification."""
        linear_op = SimpleLinearOp(in_features=10, out_features=5)
        seq = Sequential(linear_op)

        # Trigger forward pass
        input_tensor = torch.randn(3, 10)
        seq(input_tensor)

        # Module groups should be created
        assert seq._module_groups is not None

        # Modify the sequential
        seq.append(torch.nn.ReLU())

        # Module groups should be invalidated
        assert seq._module_groups is None


class TestSequentialOpKwargs:
    """Test forwarding per-op kwargs through Sequential.forward(op_kwargs=...)."""

    def test_op_kwargs_by_module_object(self):
        """Pass kwargs keyed by the module object itself."""
        scale1 = ScaleOp(default_scale=1.0)
        scale2 = ScaleOp(default_scale=1.0)
        seq = Sequential(scale1, scale2)

        input_tensor = torch.randn(3, 8)
        output = seq(input_tensor, op_kwargs={scale1: {"scale": 2.0}, scale2: {"scale": 3.0}})

        assert torch.allclose(output, input_tensor * 6.0)

    def test_op_kwargs_by_index(self):
        """Pass kwargs keyed by the module index."""
        scale1 = ScaleOp(default_scale=1.0)
        scale2 = ScaleOp(default_scale=1.0)
        seq = Sequential(scale1, scale2)

        input_tensor = torch.randn(3, 8)
        output = seq(input_tensor, op_kwargs={0: {"scale": 2.0}, 1: {"scale": 0.5}})

        assert torch.allclose(output, input_tensor)

    def test_op_kwargs_with_defaults(self):
        """Unmentioned ops fall back to their own default."""
        scale1 = ScaleOp(default_scale=2.0)
        scale2 = ScaleOp(default_scale=1.0)
        seq = Sequential(scale1, scale2)

        input_tensor = torch.randn(3, 8)
        output = seq(input_tensor, op_kwargs={scale2: {"scale": 5.0}})

        assert torch.allclose(output, input_tensor * 10.0)

    def test_op_kwargs_none_and_empty(self):
        """None/empty op_kwargs behave like not passing it at all."""
        scale = ScaleOp(default_scale=2.0)
        seq = Sequential(scale)

        input_tensor = torch.randn(3, 8)
        expected = seq(input_tensor)
        assert torch.allclose(seq(input_tensor, op_kwargs=None), expected)
        assert torch.allclose(seq(input_tensor, op_kwargs={}), expected)

    def test_op_kwargs_within_fused_group(self):
        """Only one op of a fused group receives its kwargs."""
        scale = ScaleOp(default_scale=1.0)
        relu = SimpleReLIOp()
        seq = Sequential(scale, relu)

        input_tensor = torch.randn(3, 8)
        output = seq(input_tensor, op_kwargs={scale: {"scale": 4.0}})

        expected = torch.nn.functional.relu(input_tensor * 4.0)
        assert torch.allclose(output, expected)

    def test_op_kwargs_non_basic_operation_raises(self):
        """kwargs may only target BasicOperation instances."""
        relu = torch.nn.ReLU()
        seq = Sequential(relu)

        input_tensor = torch.randn(3, 8)
        with pytest.raises(ValueError):
            seq(input_tensor, op_kwargs={relu: {"scale": 1.0}})

    def test_op_kwargs_unknown_key_raises(self):
        """Keys not present in the Sequential are rejected."""
        scale = ScaleOp()
        seq = Sequential(scale)

        input_tensor = torch.randn(3, 8)
        stranger = ScaleOp()
        with pytest.raises(ValueError):
            seq(input_tensor, op_kwargs={stranger: {"scale": 1.0}})

    def test_op_kwargs_forward_backward(self):
        """Gradients flow through an op that consumed forward kwargs."""
        scale = ScaleOp(default_scale=1.0)
        seq = Sequential(scale)

        input_tensor = torch.randn(3, 8, requires_grad=True)
        output = seq(input_tensor, op_kwargs={scale: {"scale": 3.0}})
        output.sum().backward()

        assert torch.allclose(input_tensor.grad, torch.full_like(input_tensor, 3.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
