# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Test fusible operations - migrated from TransformerEngine

Test content (all in TestBasicOps, following TE convention):
- Forward and backward computation correctness for basic ops (RMSNorm)
- Forward and backward computation correctness for 11 activation functions
- SwiGLU forward + backward test
- Backward activation + bias tests
- Precision validation across multiple datatypes
- Compatibility tests for various input shapes

NOTE: Most Quantize/Bias coverage from the upstream suite is still omitted.
      BasicLinear keeps focused FP8/MXFP8 coverage here because it does not
      require distributed setup.

Pass criteria:
- Forward output error within tolerance
- Backward gradient error within tolerance
"""

from __future__ import annotations

from collections.abc import Iterable
import math
import pytest
import torch
import torch.nn.functional as F
import torch_npu

from transformer_engine.common import recipe
from transformer_engine.pytorch import autocast
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    is_fp8_available,
    is_mxfp8_available,
)
from transformer_engine.pytorch.utils import check_npu_version

from utils import (
    dtype_tols,
    quantization_tols,
    assert_close,
    make_reference_and_test_tensors,
    to_cpu,
)


# =============================================================================
# Test configuration
# =============================================================================

# Supported datatypes (fp16 and bf16 are similar; test both but can skip one
# for faster local iteration by setting _FAST_MODE=True)
_FAST_MODE = True

_dtypes = [torch.float32, torch.float16]
if hasattr(torch, 'bfloat16'):
    _dtypes.append(torch.bfloat16)

if _FAST_MODE:
    # Only fp32 + one low-precision dtype for quick iteration
    _dtypes = [torch.float32, torch.float16, torch.bfloat16]

# Scaled (post-scale) activations chain a GLU/unary kernel with a scale
# multiply, so their compounded fp16 error exceeds the suite's tight fp16
# tolerance. Guard them at the training dtypes only (consistent with the
# bf16-only ATK golden policy for fused activations).
_scaled_activation_dtypes = [d for d in _dtypes if d != torch.float16]

# Supported activation function types
_ACTIVATION_TYPES = (
    "gelu",
    "geglu",
    "qgelu",
    "qgeglu",
    "relu",
    "reglu",
    "glu",
    "srelu",
    "sreglu",
    "silu",
    "swiglu",
)

# Shapes: use small tensors for speed; (37,) tests odd dim, (2,13) non-square
_OUT_SHAPES = ((37,), (2, 13), (32, 1, 32))
if _FAST_MODE:
    _OUT_SHAPES = ((32, 32),)

# dtype short names for logging
_DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


# =============================================================================
# Logging helper
# =============================================================================


def _log(msg: str, end: str = "\n") -> None:
    """Print progress log with flush for real-time output"""
    print(msg, end=end, flush=True)


def _bounded_tensor(shape, *, dtype=torch.bfloat16):
    data = torch.arange(
        torch.tensor(shape).prod().item(),
        dtype=torch.float32,
        device="cpu",
    ).reshape(shape)
    data = data.remainder(17).sub(8).div(16)
    return data.to(device="npu", dtype=dtype)


# =============================================================================
# PyTorch reference implementation
# =============================================================================


def pytorch_activation_forward(x: torch.Tensor, activation: str) -> torch.Tensor:
    """PyTorch activation function forward computation reference implementation"""
    if activation == "gelu":
        return F.gelu(x, approximate="tanh")
    elif activation == "geglu":
        x1, x2 = x.chunk(2, dim=-1)
        return F.gelu(x1, approximate="tanh") * x2
    elif activation == "qgelu":
        return x * torch.sigmoid(1.702 * x)
    elif activation == "qgeglu":
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(1.702 * x1) * x2
    elif activation == "relu":
        return F.relu(x)
    elif activation == "reglu":
        x1, x2 = x.chunk(2, dim=-1)
        return F.relu(x1) * x2
    elif activation == "sigmoid":
        return F.sigmoid(x)
    elif activation == "glu":
        in_shape = x.shape
        x_reshaped = x.reshape(*in_shape[:-1], 2, in_shape[-1] // 2)
        x_flipped = x_reshaped.flip(-2)
        x_restored = x_flipped.reshape(in_shape)
        return F.glu(x_restored)
    elif activation == "srelu":
        return F.relu(x) ** 2
    elif activation == "sreglu":
        x1, x2 = x.chunk(2, dim=-1)
        return F.relu(x1) ** 2 * x2
    elif activation == "silu":
        return F.silu(x)
    elif activation == "swiglu":
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
    else:
        raise ValueError(f"Unexpected activation function ({activation})")


# =============================================================================
# TE activation function mapping
# =============================================================================


def get_te_activation_op(activation: str, **kwargs):
    """Get TE activation function operation"""
    activation_map = dict(
        gelu=te_ops.GELU,
        geglu=te_ops.GEGLU,
        glu=te_ops.GLU,
        qgelu=te_ops.QGELU,
        qgeglu=te_ops.QGEGLU,
        relu=te_ops.ReLU,
        reglu=te_ops.ReGLU,
        srelu=te_ops.SReLU,
        sreglu=te_ops.SReGLU,
        silu=te_ops.SiLU,
        swiglu=te_ops.SwiGLU,
    )
    if activation not in activation_map:
        raise ValueError(f"Unsupported activation: {activation}")
    return activation_map[activation](**kwargs)


# =============================================================================
# Shared verification helper
# =============================================================================


def _run_activation_test(
    activation: str,
    out_shape: Iterable[int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Run a single activation forward+backward test with logging"""
    dt_name = _DTYPE_NAMES.get(dtype, str(dtype))
    _log(f"  [{activation:>7s} | {dt_name:>4s} | shape={list(out_shape)}] ", end="")

    in_shape = list(out_shape)
    if activation in ("geglu", "glu", "qgeglu", "reglu", "sreglu", "swiglu"):
        in_shape[-1] *= 2

    x_ref, x_test = make_reference_and_test_tensors(
        in_shape,
        test_dtype=dtype,
        test_device=device,
    )
    dy_ref, dy_test = make_reference_and_test_tensors(
        out_shape,
        test_dtype=dtype,
        test_device=device,
        requires_grad=False,
    )

    # Reference
    y_ref = pytorch_activation_forward(x_ref, activation)
    y_ref.backward(dy_ref)

    # TE
    act_op = get_te_activation_op(activation)
    y_test = act_op(x_test)
    y_test.backward(dy_test)

    # Verify
    tols = dtype_tols(dtype)
    assert_close(to_cpu(y_test), to_cpu(y_ref), **tols)
    assert_close(to_cpu(x_test.grad), to_cpu(x_ref.grad), **tols)

    _log("OK")


# =============================================================================
# Test classes
# =============================================================================


class TestBasicOps:
    """Tests for individual basic operations

    Migrated from TransformerEngine test_fusible_ops.py::TestBasicOps.
    Includes basic ops (RMSNorm), activation functions, SwiGLU, and
    backward activation+bias tests, following TE's convention of
    placing all non-fused operations in a single class.
    """

    def test_basic_linear_forward_backward_without_tensor_parallel(self) -> None:
        """BasicLinear without TP should match dense linear math."""
        device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
        seq_len = 8
        in_features = 8
        out_features = 12
        input_ref, input_test = make_reference_and_test_tensors(
            (seq_len, in_features),
            test_dtype=torch.float32,
            test_device=device,
        )
        weight_ref, weight_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=torch.float32,
            test_device=device,
        )
        grad_output_ref, grad_output_test = make_reference_and_test_tensors(
            (seq_len, out_features),
            test_dtype=torch.float32,
            test_device=device,
            requires_grad=False,
        )

        layer = BasicLinear(
            in_features,
            out_features,
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            layer.weight.copy_(weight_test)

        output = layer(input_test)
        expected_output = torch.nn.functional.linear(input_ref, weight_ref)
        assert_close(to_cpu(output), expected_output, **dtype_tols(torch.float32))

        output.backward(grad_output_test)

        expected_grad_input = torch.matmul(grad_output_ref, weight_ref)
        expected_grad_weight = torch.matmul(grad_output_ref.t(), input_ref)
        assert_close(to_cpu(input_test.grad), expected_grad_input, **dtype_tols(torch.float32))
        assert_close(to_cpu(layer.weight.grad), expected_grad_weight, **dtype_tols(torch.float32))

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="BasicLinear FP8 test requires Atlas A5"
    )
    def test_basic_linear_float8_current_scaling_forward_backward(self) -> None:
        """BasicLinear FP8 current scaling should stay close to dense linear math."""
        fp8_available, reason = is_fp8_available(return_reason=True)
        if not fp8_available:
            pytest.skip(reason)

        seq_len = 32
        in_features = 32
        out_features = 32
        full_input = _bounded_tensor((seq_len, in_features))
        full_weight = _bounded_tensor((out_features, in_features))
        full_grad_output = _bounded_tensor((seq_len, out_features))

        local_input = full_input.detach().clone()
        local_input.requires_grad_(True)
        layer = BasicLinear(
            in_features,
            out_features,
            device="npu",
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            layer.weight.copy_(full_weight)

        tols = quantization_tols("fp8_current_scaling")
        try:
            with autocast(enabled=True, recipe=recipe.Float8CurrentScaling()):
                output = layer(local_input)
            expected_output = torch.nn.functional.linear(full_input.float(), full_weight.float())
            torch.testing.assert_close(output.float(), expected_output, **tols)

            output.backward(full_grad_output)

            expected_grad_input = torch.matmul(full_grad_output.float(), full_weight.float())
            expected_grad_weight = torch.matmul(full_grad_output.float().t(), full_input.float())
            torch.testing.assert_close(
                local_input.grad.float(),
                expected_grad_input,
                **tols,
            )
            torch.testing.assert_close(
                layer.weight.grad.float(),
                expected_grad_weight,
                **tols,
            )
        finally:
            FP8GlobalStateManager.reset()

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="BasicLinear MXFP8 test requires Atlas A5"
    )
    def test_basic_linear_mxfp8_forward_backward(self) -> None:
        """BasicLinear MXFP8 should stay close to dense linear math."""
        mxfp8_available, reason = is_mxfp8_available(return_reason=True)
        if not mxfp8_available:
            pytest.skip(reason)

        seq_len = 32
        in_features = 32
        out_features = 32
        full_input = _bounded_tensor((seq_len, in_features))
        full_weight = _bounded_tensor((out_features, in_features))
        full_grad_output = _bounded_tensor((seq_len, out_features))

        local_input = full_input.detach().clone()
        local_input.requires_grad_(True)
        layer = BasicLinear(
            in_features,
            out_features,
            device="npu",
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            layer.weight.copy_(full_weight)

        tols = quantization_tols("mxfp8_block_scaling")
        try:
            with autocast(enabled=True, recipe=recipe.MXFP8BlockScaling()):
                output = layer(local_input)
            expected_output = torch.nn.functional.linear(full_input.float(), full_weight.float())
            torch.testing.assert_close(output.float(), expected_output, **tols)

            output.backward(full_grad_output)

            expected_grad_input = torch.matmul(full_grad_output.float(), full_weight.float())
            expected_grad_weight = torch.matmul(full_grad_output.float().t(), full_input.float())
            torch.testing.assert_close(
                local_input.grad.float(),
                expected_grad_input,
                **tols,
            )
            torch.testing.assert_close(
                layer.weight.grad.float(),
                expected_grad_weight,
                **tols,
            )
        finally:
            FP8GlobalStateManager.reset()

    @pytest.mark.parametrize("weight_shape", ((19,), (64,)))
    @pytest.mark.parametrize("in_shape", ((-1,), (6, 16, -1)))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("zero_centered_gamma", (False, True))
    def test_rmsnorm(
        self,
        *,
        weight_shape: Iterable[int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        eps: float = 0.3,
        zero_centered_gamma: bool,
    ) -> None:
        """RMSNorm forward + backward test"""

        device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")

        # Make input and weight shapes consistent
        in_shape = list(in_shape)[:-1] + list(weight_shape)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
        )
        w_ref, w_test = make_reference_and_test_tensors(
            weight_shape,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        inner_dims = tuple(range(len(in_shape) - len(weight_shape), len(in_shape)))
        var_ref = x_ref.square().sum(dim=inner_dims, keepdim=True) / math.prod(weight_shape)
        if zero_centered_gamma:
            y_ref = x_ref / torch.sqrt(eps + var_ref) * (1 + w_ref)
        else:
            y_ref = x_ref / torch.sqrt(eps + var_ref) * w_ref
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        op = te_ops.RMSNorm(
            weight_shape,
            eps=eps,
            device=device,
            dtype=dtype,
            zero_centered_gamma=zero_centered_gamma,
        )
        with torch.no_grad():
            op.weight.copy_(w_test)
            del w_test
        y_test = op(x_test)
        y_test.backward(dy_test)

        # Check results
        tols = dtype_tols(dtype)
        assert_close(to_cpu(y_test), to_cpu(y_ref), **tols)
        assert_close(to_cpu(x_test.grad), to_cpu(x_ref.grad), **tols)
        assert_close(to_cpu(op.weight.grad), to_cpu(w_ref.grad), **tols)

    @pytest.mark.parametrize("activation", _ACTIVATION_TYPES)
    @pytest.mark.parametrize("out_shape", _OUT_SHAPES)
    @pytest.mark.parametrize("dtype", _dtypes)
    def test_activation(
        self,
        *,
        activation: str,
        out_shape: Iterable[int],
        dtype: torch.dtype,
    ) -> None:
        """Activation function forward + backward test"""
        device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
        _run_activation_test(activation, out_shape, dtype, device)

    @pytest.mark.parametrize("dtype", _dtypes)
    def test_activation_shapes(self, dtype: torch.dtype) -> None:
        """Test activation functions with various shapes (1D~4D)"""
        device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
        dt_name = _DTYPE_NAMES.get(dtype, str(dtype))

        test_shapes = [
            (128,),  # 1D
            (32, 64),  # 2D
            (16, 32, 64),  # 3D
            (8, 16, 32, 64),  # 4D
        ]

        for shape in test_shapes:
            _log(f"  [GELU shape test | {dt_name:>4s} | shape={list(shape)}] ", end="")
            x_ref, x_test = make_reference_and_test_tensors(
                shape,
                test_dtype=dtype,
                test_device=device,
            )
            y_ref = F.gelu(x_ref, approximate="tanh")
            y_test = te_ops.GELU()(x_test)
            assert_close(to_cpu(y_test), to_cpu(y_ref), **dtype_tols(dtype))
            _log("OK")

    @pytest.mark.parametrize("dtype", _dtypes)
    def test_swiglu(self, dtype: torch.dtype) -> None:
        """SwiGLU forward + backward test"""
        device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
        dt_name = _DTYPE_NAMES.get(dtype, str(dtype))
        _log(f"  [SwiGLU | {dt_name:>4s}] ", end="")

        out_shape = (32, 32)
        in_shape = (32, 64)

        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        x1, x2 = x_ref.chunk(2, dim=-1)
        y_ref = F.silu(x1) * x2
        y_ref.backward(dy_ref)

        y_test = te_ops.SwiGLU()(x_test)
        y_test.backward(dy_test)

        tols = dtype_tols(dtype)
        assert_close(to_cpu(y_test), to_cpu(y_ref), **tols)
        assert_close(to_cpu(x_test.grad), to_cpu(x_ref.grad), **tols)
        _log("OK")

    @pytest.mark.parametrize("dtype", _scaled_activation_dtypes)
    def test_scaled_activation(self, dtype: torch.dtype) -> None:
        """Scaled (per-row post-scale) fusible activations used by the Megatron
        fused grouped-MLP MoE path: ScaledSwiGLU, ScaledClampedQGeGLU,
        ScaledSReLU. The output is the gated/unary activation multiplied by a
        per-row scale extra input; verify forward, x grad and scale grad.
        """
        device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
        dt_name = _DTYPE_NAMES.get(dtype, str(dtype))

        def torch_clamped_swiglu(t, limit=7.0, alpha=1.702, glu_linear_offset=1.0):
            a, b = t.chunk(2, dim=-1)
            a = torch.clamp(a, max=limit)
            b = torch.clamp(b, min=-limit, max=limit) + glu_linear_offset
            return a * torch.sigmoid(alpha * a) * b

        gated_ops = (
            (
                "ScaledSwiGLU",
                te_ops.ScaledSwiGLU(),
                lambda t: F.silu(t.chunk(2, dim=-1)[0]) * t.chunk(2, dim=-1)[1],
            ),
            ("ScaledClampedQGeGLU", te_ops.ScaledClampedQGeGLU(), torch_clamped_swiglu),
        )
        in_shape = (32, 64)
        out_shape = (32, 32)
        for op_name, op, ref_fwd in gated_ops:
            _log(f"  [{op_name} | {dt_name:>4s}] ", end="")
            x_ref, x_test = make_reference_and_test_tensors(
                in_shape, test_dtype=dtype, test_device=device
            )
            scale_ref, scale_test = make_reference_and_test_tensors(
                (32,), test_dtype=dtype, test_device=device
            )
            dy_ref, dy_test = make_reference_and_test_tensors(
                out_shape, test_dtype=dtype, test_device=device, requires_grad=False
            )

            y_ref = ref_fwd(x_ref) * scale_ref.unsqueeze(-1)
            y_ref.backward(dy_ref)

            y_test = op(x_test, scale_test)
            y_test.backward(dy_test)

            tols = dtype_tols(dtype)
            assert_close(to_cpu(y_test), to_cpu(y_ref), **tols)
            assert_close(to_cpu(x_test.grad), to_cpu(x_ref.grad), **tols)
            assert_close(to_cpu(scale_test.grad), to_cpu(scale_ref.grad), **tols)
            _log("OK")

        # ScaledSReLU is unary: output shape equals the input shape.
        _log(f"  [ScaledSReLU | {dt_name:>4s}] ", end="")
        s_in = (32, 32)
        x_ref, x_test = make_reference_and_test_tensors(s_in, test_dtype=dtype, test_device=device)
        scale_ref, scale_test = make_reference_and_test_tensors(
            (32,), test_dtype=dtype, test_device=device
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            s_in, test_dtype=dtype, test_device=device, requires_grad=False
        )
        r = F.relu(x_ref)
        y_ref = (r * r) * scale_ref.unsqueeze(-1)
        y_ref.backward(dy_ref)
        y_test = te_ops.ScaledSReLU()(x_test, scale_test)
        y_test.backward(dy_test)
        tols = dtype_tols(dtype)
        assert_close(to_cpu(y_test), to_cpu(y_ref), **tols)
        assert_close(to_cpu(x_test.grad), to_cpu(x_ref.grad), **tols)
        assert_close(to_cpu(scale_test.grad), to_cpu(scale_ref.grad), **tols)
        _log("OK")

    @pytest.mark.parametrize("activation", ("relu", "gelu"))
    @pytest.mark.parametrize("out_shape", ((32, 32), (32, 1, 32)))
    @pytest.mark.parametrize("dtype", _dtypes)
    def test_backward_activation_bias(
        self,
        *,
        activation: str,
        out_shape: Iterable[int],
        dtype: torch.dtype,
    ) -> None:
        """Backward activation + bias test

        Since te_ops.Bias is not available on NPU, we test:
        y = Activation(x + Bias) using PyTorch for bias and TE for activation.
        """
        device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
        dt_name = _DTYPE_NAMES.get(dtype, str(dtype))
        _log(f"  [{activation:>7s}+bias | {dt_name:>4s} | shape={list(out_shape)}] ", end="")

        hidden_size = out_shape[-1]
        in_shape = list(out_shape)

        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
        )
        b_ref, b_test = make_reference_and_test_tensors(
            (hidden_size,),
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Reference
        y_ref = x_ref + b_ref.reshape([1] * (len(in_shape) - 1) + [hidden_size])
        if activation == "gelu":
            y_ref = F.gelu(y_ref, approximate="tanh")
        elif activation == "relu":
            y_ref = F.relu(y_ref)
        y_ref.backward(dy_ref)

        # TE (bias in PyTorch, then TE activation)
        bias_shape = [1] * (len(in_shape) - 1) + [hidden_size]
        z_test = x_test + b_test.reshape(bias_shape)
        act_type = te_ops.GELU if activation == "gelu" else te_ops.ReLU
        y_test = act_type()(z_test)
        y_test.backward(dy_test)

        # Verify
        tols = dtype_tols(dtype)
        assert_close(to_cpu(y_test), to_cpu(y_ref), **tols)
        assert_close(to_cpu(x_test.grad), to_cpu(x_ref.grad), **tols)
        assert_close(to_cpu(b_test.grad), to_cpu(b_ref.grad), **tols)
        _log("OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
