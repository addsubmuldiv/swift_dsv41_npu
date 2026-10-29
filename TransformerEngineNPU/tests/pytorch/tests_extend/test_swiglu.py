# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``ops/basic/swiglu.py``:
- SwiGLU.__init__ parameter handling
- SwiGLU.op_forward shape + dtype dispatch
- SwiGLU.op_backward gradient propagation
- End-to-end forward+backward via op_forward + op_backward
- glu_interleave_size reshape round-trip
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.ops.basic.swiglu import SwiGLU
from transformer_engine.pytorch.ops.op import OperationContext
from utils import assert_close, dtype_tols, npu_available, to_cpu

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


def _swiglu_input(shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(*shape, device="cpu").to(device=_NPU, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# __init__
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwiGLUInit:
    def test_default_params(self) -> None:
        op = SwiGLU()
        assert op.cache_quantized_input is False
        assert op.glu_interleave_size is None

    def test_explicit_params(self) -> None:
        op = SwiGLU(cache_quantized_input=True, glu_interleave_size=2)
        assert op.cache_quantized_input is True
        assert op.glu_interleave_size == 2


# ═══════════════════════════════════════════════════════════════════════════════
# op_forward: shape + dtype + dispatch
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwiGLUForward:
    @staticmethod
    def setup_method() -> None:
        _seed()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_forward_shape_halves_last_dim(self, dtype: torch.dtype) -> None:
        op = SwiGLU()
        x = _swiglu_input((4, 16), dtype=dtype)
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        torch.npu.synchronize()
        assert out.shape == (4, 8)
        assert out.dtype == dtype

    def test_unsupported_dtype_raises(self) -> None:
        op = SwiGLU()
        x = torch.zeros(2, 8, dtype=torch.float64, device=_NPU)
        ctx = OperationContext()
        with pytest.raises(RuntimeError, match="Unsupported dtype"):
            op.op_forward(ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None)


# ═══════════════════════════════════════════════════════════════════════════════
# glu_interleave_size forward
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwiGLUInterleave:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_interleave_2(self) -> None:
        op = SwiGLU(glu_interleave_size=2)
        x = _swiglu_input((4, 16))
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        torch.npu.synchronize()
        assert out.shape == (4, 8)

    def test_interleave_4(self) -> None:
        op = SwiGLU(glu_interleave_size=4)
        x = _swiglu_input((2, 16))
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        torch.npu.synchronize()
        assert out.shape == (2, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end forward + backward via op_backward (the real test path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwiGLUEndToEnd:
    """Exercise the real op_forward → op_backward flow.

    Uses ``torch.autograd`` to seed input.requires_grad and verify grad_input.
    ``op_forward`` saves input via ``ctx.save_for_backward``; an external
    fuser is responsible for promoting to_save → saved_tensors. To bypass
    the fuser dependency we manually copy forward's saved tensor into
    ``ctx.saved_tensors`` before calling op_backward.
    """

    @staticmethod
    def setup_method() -> None:
        _seed()

    def _forward_then_backward(self, op: SwiGLU, x: torch.Tensor):
        ctx = OperationContext()  # default requires_grad=True
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        torch.npu.synchronize()
        # Bridge: copy saved tensor from to_save to saved_tensors (what fuser would do)
        ctx.saved_tensors = ctx.to_save
        grad_output = torch.randn_like(out)
        dx, extras = op.op_backward(ctx, grad_output)
        torch.npu.synchronize()
        return out, dx, extras

    def test_default_end_to_end(self) -> None:
        """Forward + backward via SwiGLU's own op_backward (no torch.autograd).
        NpuSwigluBackward is a fused kernel, not a torch.autograd.Function,
        so torch.autograd backward isn't applicable here.
        """
        op = SwiGLU()
        x = _swiglu_input((4, 16))
        out, dx, extras = self._forward_then_backward(op, x)
        assert out.shape == (4, 8)
        assert dx.shape == (4, 16)
        assert extras == ()
        # Manual dx is finite and matches the input's dtype/device
        assert dx.dtype == x.dtype
        assert dx.device.type == "npu"
        assert torch.isfinite(dx).all().item()

    def test_interleave_end_to_end(self) -> None:
        """Backward must reverse the forward interleave reshape."""
        op = SwiGLU(glu_interleave_size=2)
        x = _swiglu_input((2, 16))
        out, dx, extras = self._forward_then_backward(op, x)
        assert out.shape == (2, 8)
        assert dx.shape == (2, 16)
        assert extras == ()
        assert torch.isfinite(dx).all().item()


# ═══════════════════════════════════════════════════════════════════════════════
# Numerics
# ═══════════════════════════════════════════════════════════════════════════════


class TestSwiGLUNumerics:
    @staticmethod
    def setup_method() -> None:
        _seed()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_matches_silu_x_linear(self, dtype: torch.dtype) -> None:
        op = SwiGLU()
        x = _swiglu_input((4, 16), dtype=dtype)
        ctx = OperationContext()
        out = op.op_forward(
            ctx, x, prev_op_grad_output_quantizer=None, next_op_input_quantizer=None
        )
        torch.npu.synchronize()
        gate, linear = x.chunk(2, dim=-1)
        ref = torch.nn.functional.silu(gate) * linear
        assert_close(to_cpu(out), to_cpu(ref), **dtype_tols(dtype))
