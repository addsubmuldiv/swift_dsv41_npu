# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for op.py and fuser.py migration."""

import pytest
import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_engine.pytorch.ops.op import (
    OperationContext,
    BasicOperation,
    FusedOperation,
)
from transformer_engine.pytorch.ops.fuser import (
    OperationFuser,
    _is_graph_capturing,
    _split_tuple,
)


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

    def __init__(self):
        super().__init__()

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


class TestSimpleOperation:
    """Test simple operations with forward and backward."""

    def test_simple_linear_forward(self):
        """Test SimpleLinearOp forward pass."""
        # Create operation
        op = SimpleLinearOp(in_features=10, out_features=5)
        
        # Create input
        input_tensor = torch.randn(3, 10)  # batch_size=3, in_features=10
        
        # Forward pass using OperationFuser
        fuser = OperationFuser([op])
        output = fuser(input_tensor)
        
        # Check output shape
        assert output.shape == (3, 5), f"Expected shape (3, 5), got {output.shape}"
        
        # Verify computation
        expected = torch.nn.functional.linear(input_tensor, op.weight, op.bias)
        assert torch.allclose(output, expected, atol=1e-6), "Forward pass result mismatch"

    def test_simple_linear_backward(self):
        """Test SimpleLinearOp backward pass."""
        # Create operation
        op = SimpleLinearOp(in_features=10, out_features=5)
        
        # Create input with gradient tracking
        input_tensor = torch.randn(3, 10, requires_grad=True)
        
        # Forward pass
        fuser = OperationFuser([op])
        output = fuser(input_tensor)
        
        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        # Check gradients
        assert input_tensor.grad is not None, "Input gradient is None"
        assert input_tensor.grad.shape == input_tensor.shape, "Input gradient shape mismatch"
        assert op.weight.grad is not None, "Weight gradient is None"
        assert op.weight.grad.shape == op.weight.shape, "Weight gradient shape mismatch"
        assert op.bias.grad is not None, "Bias gradient is None"
        assert op.bias.grad.shape == op.bias.shape, "Bias gradient shape mismatch"

    def test_simple_relu_forward(self):
        """Test SimpleReLIOp forward pass."""
        # Create operation
        op = SimpleReLIOp()
        
        # Create input
        input_tensor = torch.randn(3, 10)
        
        # Forward pass
        fuser = OperationFuser([op])
        output = fuser(input_tensor)
        
        # Check output shape
        assert output.shape == input_tensor.shape, "Output shape mismatch"
        
        # Verify computation
        expected = torch.nn.functional.relu(input_tensor)
        assert torch.allclose(output, expected, atol=1e-6), "Forward pass result mismatch"

    def test_simple_relu_backward(self):
        """Test SimpleReLIOp backward pass."""
        # Create operation
        op = SimpleReLIOp()
        
        # Create input with gradient tracking
        input_tensor = torch.randn(3, 10, requires_grad=True)
        
        # Forward pass
        fuser = OperationFuser([op])
        output = fuser(input_tensor)
        
        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        # Check gradient
        assert input_tensor.grad is not None, "Input gradient is None"
        assert input_tensor.grad.shape == input_tensor.shape, "Input gradient shape mismatch"


class TestOperationFusion:
    """Test operation fusion with multiple operations."""

    def test_linear_relu_fusion_forward(self):
        """Test fusing Linear + ReLU operations."""
        # Create operations
        linear_op = SimpleLinearOp(in_features=10, out_features=5)
        relu_op = SimpleReLIOp()
        
        # Create input
        input_tensor = torch.randn(3, 10)
        
        # Forward pass with fused operations
        fuser = OperationFuser([linear_op, relu_op])
        output = fuser(input_tensor)
        
        # Check output shape
        assert output.shape == (3, 5), f"Expected shape (3, 5), got {output.shape}"
        
        # Verify computation
        expected_linear = torch.nn.functional.linear(input_tensor, linear_op.weight, linear_op.bias)
        expected = torch.nn.functional.relu(expected_linear)
        assert torch.allclose(output, expected, atol=1e-6), "Fused forward pass result mismatch"

    def test_linear_relu_fusion_backward(self):
        """Test backward pass with fused Linear + ReLU."""
        # Create operations
        linear_op = SimpleLinearOp(in_features=10, out_features=5)
        relu_op = SimpleReLIOp()
        
        # Create input with gradient tracking
        input_tensor = torch.randn(3, 10, requires_grad=True)
        
        # Forward pass
        fuser = OperationFuser([linear_op, relu_op])
        output = fuser(input_tensor)
        
        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        
        # Check gradients
        assert input_tensor.grad is not None, "Input gradient is None"
        assert input_tensor.grad.shape == input_tensor.shape, "Input gradient shape mismatch"
        assert linear_op.weight.grad is not None, "Weight gradient is None"
        assert linear_op.weight.grad.shape == linear_op.weight.shape, "Weight gradient shape mismatch"

    def test_fused_operation_class(self):
        """Test FusedOperation class."""
        # Create basic operations
        linear_op = SimpleLinearOp(in_features=10, out_features=5)
        relu_op = SimpleReLIOp()
        
        # Create fused operation
        fused_op = FusedOperation([linear_op, relu_op])
        
        # Check properties
        assert fused_op.is_fused_op is True, "is_fused_op should be True"
        assert len(fused_op.basic_ops) == 2, "Should have 2 basic operations"
        
        # Create input
        input_tensor = torch.randn(3, 10)
        
        # Forward pass
        output = fused_op(input_tensor)
        
        # Check output shape
        assert output.shape == (3, 5), f"Expected shape (3, 5), got {output.shape}"


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_graph_capturing(self):
        """Test _is_graph_capturing always returns False."""
        assert _is_graph_capturing() is False

    def test_split_tuple(self):
        """Test _split_tuple function."""
        t = (1, 2, 3, 4, 5)
        
        # Split at index 0
        left, right = _split_tuple(t, 0)
        assert left == ()
        assert right == (1, 2, 3, 4, 5)
        
        # Split at index 2
        left, right = _split_tuple(t, 2)
        assert left == (1, 2)
        assert right == (3, 4, 5)
        
        # Split at index 5 (end)
        left, right = _split_tuple(t, 5)
        assert left == (1, 2, 3, 4, 5)
        assert right == ()


class TestOperationContext:
    """Test OperationContext class."""

    def test_save_for_backward(self):
        """Test save_for_backward method."""
        ctx = OperationContext()
        tensor1 = torch.randn(3, 4)
        tensor2 = torch.randn(5, 6)
        
        ctx.save_for_backward(tensor1, tensor2)
        assert ctx.to_save is not None
        assert len(ctx.to_save) == 2
        assert ctx.to_save[0] is tensor1
        assert ctx.to_save[1] is tensor2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
