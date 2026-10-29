# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Tests for lightweight fusible basic ops."""

from __future__ import annotations

import torch

import transformer_engine.pytorch.ops as te_ops
import transformer_engine.pytorch.ops.basic.l2normalization as l2normalization_mod


def _make_tensor(shape: tuple[int, ...], *, requires_grad: bool = True) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float32, requires_grad=requires_grad)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.detach().to(dtype=torch.float64, device="cpu"),
        expected.detach().to(dtype=torch.float64, device="cpu"),
        rtol=0,
        atol=0,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.detach().to(dtype=torch.float64, device="cpu"),
        expected.detach().to(dtype=torch.float64, device="cpu"),
        rtol=1e-6,
        atol=1e-6,
    )


def test_identity() -> None:
    """Identity is an exact forward/backward pass-through."""

    x = _make_tensor((4, 8))
    dy = _make_tensor((4, 8), requires_grad=False)

    y = te_ops.Identity()(x)
    y.backward(dy)

    _assert_exact(y, x)
    _assert_exact(x.grad, dy)


def test_reshape() -> None:
    """Reshape restores the original shape in backward."""

    x_ref = _make_tensor((2, 3, 4))
    x_test = x_ref.detach().clone().requires_grad_()
    dy = _make_tensor((4, 6), requires_grad=False)

    y_ref = x_ref.reshape(4, 6)
    y_ref.backward(dy)
    y_test = te_ops.Reshape((4, 6))(x_test)
    y_test.backward(dy)

    _assert_exact(y_test, y_ref)
    _assert_exact(x_test.grad, x_ref.grad)


def test_bias() -> None:
    """Bias matches PyTorch broadcasting and bias-gradient reduction."""

    x_ref = _make_tensor((2, 3, 5))
    x_test = x_ref.detach().clone().requires_grad_()
    b_ref = _make_tensor((5,))
    dy = _make_tensor((2, 3, 5), requires_grad=False)

    y_ref = x_ref + b_ref.view(1, 1, -1)
    y_ref.backward(dy)

    op = te_ops.Bias(5, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        op.bias.copy_(b_ref)
    y_test = op(x_test)
    y_test.backward(dy)

    _assert_exact(y_test, y_ref)
    _assert_exact(x_test.grad, x_ref.grad)
    _assert_exact(op.bias.grad, b_ref.grad)


def test_constant_scale() -> None:
    """ConstantScale matches multiply-by-constant in forward and backward."""

    scale = -2.5
    x_ref = _make_tensor((3, 4))
    x_test = x_ref.detach().clone().requires_grad_()
    dy = _make_tensor((3, 4), requires_grad=False)

    y_ref = x_ref * scale
    y_ref.backward(dy)
    y_test = te_ops.ConstantScale(scale)(x_test)
    y_test.backward(dy)

    _assert_exact(y_test, y_ref)
    _assert_exact(x_test.grad, x_ref.grad)


def test_quantize_without_fp8_autocast_is_identity() -> None:
    """Quantize is a no-op outside an FP8 autocast context, like NVIDIA."""

    x = _make_tensor((4, 8))
    dy = _make_tensor((4, 8), requires_grad=False)

    y = te_ops.Quantize(forward=True, backward=True)(x)
    y.backward(dy)

    _assert_exact(y, x)
    _assert_exact(x.grad, dy)


def test_add_extra_input() -> None:
    """AddExtraInput joins two branches and sends gradients to both inputs."""

    x1 = _make_tensor((4, 8))
    x2 = _make_tensor((4, 8))
    dy = _make_tensor((4, 8), requires_grad=False)
    expected = x1.detach() + x2.detach()

    y = te_ops.AddExtraInput()(x1, x2)
    y.backward(dy)

    _assert_exact(y, expected)
    _assert_exact(x1.grad, dy)
    _assert_exact(x2.grad, dy)


def test_add_extra_input_in_place() -> None:
    """In-place AddExtraInput preserves NVIDIA's extra-input gradient contract."""

    x1 = _make_tensor((4, 8))
    x2 = _make_tensor((4, 8))
    dy = _make_tensor((4, 8), requires_grad=False)
    expected = x1.detach() + x2.detach()

    y = te_ops.AddExtraInput(in_place=True)(x1, x2)
    y.backward(dy)

    _assert_exact(y, expected)
    _assert_exact(x1.grad, dy)
    _assert_exact(x2.grad, dy)


def test_make_extra_output() -> None:
    """MakeExtraOutput splits one branch and sums both output gradients."""

    x = _make_tensor((4, 8))
    dy1 = _make_tensor((4, 8), requires_grad=False)
    dy2 = _make_tensor((4, 8), requires_grad=False)

    y1, y2 = te_ops.MakeExtraOutput()(x)
    (y1 * dy1 + y2 * dy2).sum().backward()

    _assert_exact(y1, x)
    _assert_exact(y2, x)
    _assert_exact(x.grad, dy1 + dy2)


def test_make_extra_output_in_place() -> None:
    """In-place MakeExtraOutput accumulates the main gradient into the extra-output grad."""

    x = _make_tensor((4, 8))
    dy1 = _make_tensor((4, 8), requires_grad=False)
    dy2 = _make_tensor((4, 8), requires_grad=False)

    y1, y2 = te_ops.MakeExtraOutput(in_place=True)(x)
    (y1 * dy1 + y2 * dy2).sum().backward()

    _assert_exact(y1, x)
    _assert_exact(y2, x)
    _assert_exact(x.grad, dy1 + dy2)


def test_l2_normalization() -> None:
    """L2Normalization matches PyTorch math and backward over the last dimension."""

    eps = 1e-6
    x_ref = _make_tensor((3, 4, 5))
    x_test = x_ref.detach().clone().requires_grad_()
    dy = _make_tensor((3, 4, 5), requires_grad=False)

    y_ref = x_ref * torch.rsqrt((x_ref * x_ref).sum(dim=-1, keepdim=True) + eps)
    y_ref.backward(dy)

    y_test = te_ops.L2Normalization(eps=eps)(x_test)
    y_test.backward(dy)

    _assert_close(y_test, y_ref)
    _assert_close(x_test.grad, x_ref.grad)


def test_l2_normalization_calls_cpu_offload_marker(monkeypatch) -> None:
    """L2Normalization calls the activation-offload hook when enabled."""

    eps = 1e-6
    x = _make_tensor((3, 5))
    marked_tensors = []

    monkeypatch.setattr(l2normalization_mod, "is_cpu_offload_enabled", lambda: True)
    monkeypatch.setattr(
        l2normalization_mod,
        "mark_activation_offload",
        lambda *tensors: marked_tensors.append(tensors),
    )

    y = te_ops.L2Normalization(eps=eps)(x)
    assert len(marked_tensors) == 1
    assert marked_tensors[0][0] is x
    _assert_close(
        marked_tensors[0][1],
        torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps),
    )
    y.sum().backward()


def test_basic_ops_compose_in_sequential() -> None:
    """A small Sequential pipeline exercises exported ops through the fuser."""

    x_ref = _make_tensor((2, 3))
    x_test = x_ref.detach().clone().requires_grad_()
    dy = _make_tensor((3, 2), requires_grad=False)

    op = te_ops.Sequential(
        te_ops.Identity(),
        te_ops.Bias(3, device="cpu", dtype=torch.float32),
        te_ops.ConstantScale(0.5),
        te_ops.Reshape((3, 2)),
    )
    with torch.no_grad():
        op[1].bias.copy_(torch.tensor([1.0, -2.0, 3.0]))

    y_ref = ((x_ref + op[1].bias.view(1, -1)) * 0.5).reshape(3, 2)
    y_ref.backward(dy)
    y_test = op(x_test)
    y_test.backward(dy)

    _assert_exact(y_test, y_ref)
    _assert_exact(x_test.grad, x_ref.grad)


def test_extra_tensor_channel_routing() -> None:
    """set_extra_*_channel keeps a residual connection inside a Sequential."""

    residual = te_ops.MakeExtraOutput()
    body = te_ops.Bias(8, device="cpu", dtype=torch.float32)
    add_residual = te_ops.AddExtraInput()
    residual.set_extra_output_channel(0, "residual")
    add_residual.set_extra_input_channel(0, "residual")

    model = te_ops.Sequential(residual, body, add_residual)
    x = _make_tensor((8,))
    y, residual_out = model(x)

    _assert_exact(y, 2 * x + body.bias)
    _assert_exact(residual_out, x)

    dy = _make_tensor((8,), requires_grad=False)
    y.backward(dy)
    _assert_exact(x.grad, 2 * dy)
    _assert_exact(body.bias.grad, dy)


def test_extra_output_channel_hidden_from_caller() -> None:
    """output_to_caller=False keeps a channel internal to the fuser."""

    producer = te_ops.MakeExtraOutput()
    consumer = te_ops.AddExtraInput()
    producer.set_extra_output_channel(0, "route", output_to_caller=False)
    consumer.set_extra_input_channel(0, "route")

    model = te_ops.Sequential(producer, consumer)
    x = _make_tensor((4,))
    y = model(x)

    assert isinstance(y, torch.Tensor)
    _assert_exact(y, 2 * x)

    dy = _make_tensor((4,), requires_grad=False)
    y.backward(dy)
    _assert_exact(x.grad, 2 * dy)


def test_joint_forward_backward_fusion() -> None:
    """register_forward_backward_fusion applies a single op to both passes."""

    class FusedResidual(te_ops.FusedOperation):
        """Fuse MakeExtraOutput + AddExtraInput into one op."""

        _enabled = True

        def __init__(self, producer, consumer) -> None:
            super().__init__((producer, consumer))

        def fuser_forward(self, basic_op_ctxs, input_, *, basic_op_extra_inputs, **unused):
            del basic_op_ctxs
            # The consumer slot is internal to this fusion, so the fuser
            # deliberately leaves it unset.
            assert basic_op_extra_inputs[1][0] is None
            return 2 * input_, [(input_,), ()]

        def fuser_backward(self, basic_op_ctxs, grad_output, *, basic_op_grad_extra_outputs):
            del basic_op_ctxs
            grad_route = basic_op_grad_extra_outputs[0][0]
            return (
                2 * grad_output
                + (torch.zeros_like(grad_output) if grad_route is None else grad_route),
                [(), ()],
                [(), (grad_output,)],
            )

    def fuse_residual(ops, **unused):
        if (
            FusedResidual._enabled
            and len(ops) == 2
            and isinstance(ops[0], te_ops.MakeExtraOutput)
            and isinstance(ops[1], te_ops.AddExtraInput)
        ):
            # We want to enable this fusion just for this test.
            # Hence disable it after fusing it once in the test.
            FusedResidual._enabled = False
            return [FusedResidual(*ops)]
        return ops

    residual = te_ops.MakeExtraOutput()
    consumer = te_ops.AddExtraInput()
    residual.set_extra_output_channel(0, "residual")
    consumer.set_extra_input_channel(0, "residual")
    model = te_ops.Sequential(residual, consumer)
    te_ops.register_forward_backward_fusion(fuse_residual, prepend=True)

    x = _make_tensor((4,))
    y, residual_out = model(x)

    # The joint fusion owns both passes.
    fuser = model._module_groups[0]
    assert len(fuser._forward_ops) == 1
    assert len(fuser._backward_ops) == 1
    assert fuser._backward_ops[0][0] is fuser._forward_ops[0][0]

    _assert_exact(y, 2 * x)
    _assert_exact(residual_out, x)

    dy = _make_tensor((4,), requires_grad=False)
    dresidual = _make_tensor((4,), requires_grad=False)
    (y * dy + residual_out * dresidual).sum().backward()
    _assert_exact(x.grad, 2 * dy + dresidual)


def test_internal_and_external_extra_inputs() -> None:
    """Unbound slots stay public while other slots use internal channels."""

    producer = te_ops.MakeExtraOutput()
    internal_consumer = te_ops.AddExtraInput()
    external_consumer = te_ops.AddExtraInput()
    producer.set_extra_output_channel(0, "route")
    internal_consumer.set_extra_input_channel(0, "route")
    model = te_ops.Sequential(producer, internal_consumer, external_consumer)

    x = _make_tensor((4,))
    extra = _make_tensor((4,))
    y, route = model(x, extra)

    _assert_exact(y, 2 * x + extra)
    _assert_exact(route, x)

    dy = _make_tensor((4,), requires_grad=False)
    y.backward(dy)
    _assert_exact(x.grad, 2 * dy)
    _assert_exact(extra.grad, dy)

    # Internal slots are unavailable before forward, so the no-grad path must
    # tolerate them without touching requires_grad flags.
    x_no_grad = x.detach()
    extra_no_grad = extra.detach()
    with torch.no_grad():
        y_no_grad, route_no_grad = model(x_no_grad, extra_no_grad)
        _assert_exact(y_no_grad, 2 * x_no_grad + extra_no_grad)
        _assert_exact(route_no_grad, x_no_grad)
