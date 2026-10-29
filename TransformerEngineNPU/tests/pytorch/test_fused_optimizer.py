# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Basic FusedAdam/FusedSGD tests for the NPU Python fallback optimizers."""

from __future__ import annotations

import copy
import inspect
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

from transformer_engine.pytorch.optimizers import FusedAdam, FusedSGD
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    Float8Tensor,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer


def _device():
    return torch.device("npu")


@pytest.fixture(name="optimizer_param", params=["tensor", "shard", "replicate"])
def optimizer_param_fixture(request):
    """Reuse existing checks with FSDP layouts; torchrun enables multiple ranks."""
    if request.param == "tensor":
        yield lambda param: param
        return
    owns_group = not dist.is_initialized()
    with tempfile.TemporaryDirectory() as temp_dir:
        if owns_group:
            backend = "hccl" if _device().type == "npu" else "gloo"
            if _device().type == "npu":
                torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            if "RANK" in os.environ:
                dist.init_process_group(backend)
            else:
                dist.init_process_group(
                    backend, init_method=Path(temp_dir, "store").as_uri(), rank=0, world_size=1
                )
        try:
            mesh = init_device_mesh(_device().type, (dist.get_world_size(),))
            placement = Shard(0) if request.param == "shard" else Replicate()

            def wrap(param):
                return torch.nn.Parameter(distribute_tensor(param.detach(), mesh, [placement]))

            yield wrap
        finally:
            if owns_group:
                dist.destroy_process_group()


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _randn(shape, dtype=torch.float32):
    return torch.randn(shape, dtype=dtype, device="cpu").to(device=_device())


def _make_linear():
    return torch.nn.Linear(8, 4, device="cpu", dtype=torch.float32).to(device=_device())


def _clone_model(model):
    return copy.deepcopy(model).to(device=_device())


def _assert_models_close(model, ref_model, rtol=1e-5, atol=1e-6):
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        torch.testing.assert_close(
            param.detach().to(device="cpu", dtype=torch.float32),
            ref_param.detach().to(device="cpu", dtype=torch.float32),
            rtol=rtol,
            atol=atol,
        )


def _max_param_diff(model, ref_model):
    max_diff = 0.0
    for param, ref_param in zip(model.parameters(), ref_model.parameters()):
        diff = (
            param.detach().to(device="cpu", dtype=torch.float32)
            - ref_param.detach().to(device="cpu", dtype=torch.float32)
        ).abs()
        max_diff = max(max_diff, float(diff.max().item()))
    return max_diff


def _assert_adam_reference_close(model, ref_model):
    """Check FusedAdam against torch.optim on NPU with practical kernel tolerance."""

    _assert_models_close(model, ref_model, rtol=1e-4, atol=1e-5)


_FP8_STATE_AVAILABLE = None


def _fp8_state_available():
    global _FP8_STATE_AVAILABLE

    if _FP8_STATE_AVAILABLE is not None:
        return _FP8_STATE_AVAILABLE
    if not FusedAdam._float8_state_available():
        _FP8_STATE_AVAILABLE = False
        return _FP8_STATE_AVAILABLE
    try:
        value = torch.zeros(8, dtype=torch.bfloat16, device=_device())
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=False,
            device=value.device,
        )
        fp8_value = quantizer(value)
        updated_value = quantizer.update_quantized(value + 1.0, fp8_value)
        param = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32, device=_device()))
        optimizer = FusedAdam(
            [param],
            lr=1e-3,
            exp_avg_dtype=torch.uint8,
            exp_avg_sq_dtype=torch.uint8,
        )
        param.grad = torch.ones_like(param)
        optimizer.step()
        state = optimizer.state[param]
        _FP8_STATE_AVAILABLE = all(
            (
                isinstance(fp8_value, Float8Tensor),
                isinstance(updated_value, Float8Tensor),
                fp8_value.dequantize(dtype=torch.float32).dtype == torch.float32,
                isinstance(state["exp_avg"], Float8Tensor),
                isinstance(state["exp_avg_sq"], Float8Tensor),
            )
        )
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        _FP8_STATE_AVAILABLE = False
    return _FP8_STATE_AVAILABLE


def _skip_if_fp8_state_unavailable():
    if not _fp8_state_available():
        pytest.skip("NPU Float8Tensor optimizer state is not available in this environment")


def _make_param(dtype=torch.float32, values=None):
    if values is None:
        values = torch.linspace(-0.5, 0.5, steps=8, dtype=torch.float32, device="cpu")
    values = values.to(device=_device(), dtype=dtype)
    requires_grad = torch.is_floating_point(values)
    return torch.nn.Parameter(values, requires_grad=requires_grad)


def _make_quantized_param(quantization):
    values = torch.linspace(-0.75, 0.75, steps=128 * 128, dtype=torch.float32).reshape(128, 128)
    values = values.to(device=_device(), dtype=torch.bfloat16)
    if quantization == "delayed":
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=_device()),
            amax=torch.zeros(1, dtype=torch.float32, device=_device()),
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=False,
        )
    elif quantization == "current":
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=False,
            device=_device(),
        )
    elif quantization == "blockwise":
        quantizer = Float8BlockQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=1,
        )
    else:
        quantizer = MXFP8Quantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=True,
        )
    return torch.nn.Parameter(quantizer(values))


def _reconstruct_remainder_master(param, remainder):
    high = param.detach().cpu().contiguous().view(torch.int16).to(torch.int32)
    low = remainder.detach().cpu().to(torch.int32)
    high -= (low < 0).to(torch.int32)
    bits = ((high & 0xFFFF) << 16) | (low & 0xFFFF)
    return bits.contiguous().view(torch.float32)


def _param_grad_like(param, offset=0):
    local_param = _local(param)
    grad = torch.linspace(
        0.1,
        0.8,
        steps=local_param.numel(),
        dtype=torch.float32,
        device="cpu",
    )
    grad = grad.reshape_as(local_param).add_(offset * 0.01)
    grad = grad.to(device=param.device, dtype=param.dtype)
    if isinstance(param, DTensor):
        grad = DTensor.from_local(
            grad,
            param.device_mesh,
            param.placements,
            shape=param.shape,
            stride=param.stride(),
        )
    return grad


def _adam_param_step(param, optimizer, step_idx=0):
    param.grad = _param_grad_like(param, step_idx)
    optimizer.step()
    optimizer.zero_grad()


def _first_optimizer_state(state_dict):
    return next(iter(state_dict["state"].values()))


def _manual_adam_update(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    *,
    lr,
    betas,
    eps,
    weight_decay,
    adam_w_mode,
    bias_correction,
    step,
):
    beta1, beta2 = betas
    param_fp32 = param.float().clone()
    grad_fp32 = grad.float().clone()
    if weight_decay != 0.0:
        if adam_w_mode:
            param_fp32.add_(param_fp32, alpha=-lr * weight_decay)
        else:
            grad_fp32 = grad_fp32.add(param_fp32, alpha=weight_decay)

    exp_avg = exp_avg.mul(beta1).add(grad_fp32, alpha=1.0 - beta1)
    exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad_fp32, grad_fp32, value=1.0 - beta2)
    denom = exp_avg_sq.sqrt()
    if bias_correction:
        step_size = lr / (1.0 - beta1**step)
        denom = denom / ((1.0 - beta2**step) ** 0.5)
    else:
        step_size = lr
    param_fp32.addcdiv_(exp_avg, denom.add(eps), value=-step_size)
    return param_fp32, exp_avg, exp_avg_sq


def _mse_step(model, optimizer, x, target):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    return loss


def _backward_only(model, x, target):
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    return loss


def _sparse_grad_like(param):
    indices = torch.tensor([[0]], dtype=torch.long, device=_device())
    values = torch.ones(1, dtype=param.dtype, device=_device())
    return torch.sparse_coo_tensor(indices, values, size=param.shape, device=_device())


def test_import_fused_sgd():
    assert inspect.isclass(FusedSGD)
    print("FusedSGD module:", FusedSGD.__module__)
    print("FusedSGD file:", inspect.getfile(FusedSGD))


def test_sgd_matches_torch_sgd():
    torch.manual_seed(1234)
    model = _make_linear()
    ref_model = _clone_model(model)

    kwargs = {
        "lr": 0.01,
        "momentum": 0.9,
        "dampening": 0.0,
        "weight_decay": 0.01,
        "nesterov": False,
    }
    optimizer = FusedSGD(model.parameters(), wd_after_momentum=False, **kwargs)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), **kwargs)

    x = _randn((3, 8))
    target = _randn((3, 4))
    for _ in range(3):
        _mse_step(model, optimizer, x, target)
        _mse_step(ref_model, ref_optimizer, x, target)

    _assert_models_close(model, ref_model)


def test_sgd_nesterov_matches_torch_sgd():
    torch.manual_seed(1234)
    model = _make_linear()
    ref_model = _clone_model(model)

    kwargs = {
        "lr": 0.01,
        "momentum": 0.9,
        "dampening": 0.0,
        "weight_decay": 0.01,
        "nesterov": True,
    }
    optimizer = FusedSGD(model.parameters(), wd_after_momentum=False, **kwargs)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), **kwargs)

    x = _randn((3, 8))
    target = _randn((3, 4))
    for _ in range(3):
        _mse_step(model, optimizer, x, target)
        _mse_step(ref_model, ref_optimizer, x, target)

    _assert_models_close(model, ref_model)


def test_sgd_multiple_param_groups():
    torch.manual_seed(1234)
    model = _make_linear()
    ref_model = _clone_model(model)

    kwargs = {
        "lr": 0.01,
        "momentum": 0.9,
        "dampening": 0.0,
        "nesterov": False,
    }
    optimizer = FusedSGD(
        [
            {"params": [model.weight], "weight_decay": 0.01},
            {"params": [model.bias], "weight_decay": 0.0},
        ],
        wd_after_momentum=False,
        **kwargs,
    )
    ref_optimizer = torch.optim.SGD(
        [
            {"params": [ref_model.weight], "weight_decay": 0.01},
            {"params": [ref_model.bias], "weight_decay": 0.0},
        ],
        **kwargs,
    )

    x = _randn((3, 8))
    target = _randn((3, 4))
    for _ in range(3):
        _mse_step(model, optimizer, x, target)
        _mse_step(ref_model, ref_optimizer, x, target)

    _assert_models_close(model, ref_model)


def test_sgd_state_dict_load_state_dict():
    torch.manual_seed(1234)
    model = _make_linear()
    kwargs = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.01}
    optimizer = FusedSGD(model.parameters(), **kwargs)

    x = _randn((3, 8))
    target = _randn((3, 4))
    for _ in range(2):
        _mse_step(model, optimizer, x, target)

    state_dict = optimizer.state_dict()
    state_count = len(state_dict["state"])
    optimizer2 = FusedSGD(model.parameters(), **kwargs)
    optimizer2.load_state_dict(state_dict)

    assert len(optimizer2.state_dict()["state"]) == state_count
    _mse_step(model, optimizer2, x, target)
    assert len(optimizer2.state_dict()["state"]) == state_count


def test_sgd_zero_grad():
    torch.manual_seed(1234)
    model = _make_linear()
    optimizer = FusedSGD(model.parameters(), lr=0.01, momentum=0.9)
    x = _randn((3, 8))
    target = _randn((3, 4))

    _backward_only(model, x, target)
    optimizer.zero_grad()
    assert all(param.grad is None for param in model.parameters())

    _backward_only(model, x, target)
    optimizer.zero_grad(set_to_none=True)
    assert all(param.grad is None for param in model.parameters())

    _backward_only(model, x, target)
    optimizer.zero_grad(set_to_none=False)
    for param in model.parameters():
        assert param.grad is not None
        torch.testing.assert_close(param.grad, torch.zeros_like(param.grad))


def test_sgd_skip_none_grad():
    torch.manual_seed(1234)
    model = _make_linear()
    optimizer = FusedSGD(model.parameters(), lr=0.01, momentum=0.9)
    x = _randn((3, 8))
    target = _randn((3, 4))

    _backward_only(model, x, target)
    before_weight = model.weight.detach().clone()
    before_bias = model.bias.detach().clone()
    model.weight.grad = None

    optimizer.step()

    torch.testing.assert_close(model.weight, before_weight)
    assert (model.bias.detach() - before_bias).abs().max().item() > 0.0


def test_sgd_sparse_grad_raises():
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    optimizer = FusedSGD([param], lr=0.01)
    param.grad = _sparse_grad_like(param)

    with pytest.raises(RuntimeError, match="sparse gradients"):
        optimizer.step()


def test_import_fused_adam():
    assert inspect.isclass(FusedAdam)
    print("FusedAdam module:", FusedAdam.__module__)
    print("FusedAdam file:", inspect.getfile(FusedAdam))


def test_fused_adam_signature():
    signature = inspect.signature(FusedAdam.__init__)
    expected_params = {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "bias_correction",
        "adam_w_mode",
        "master_weights",
        "master_weight_dtype",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
        "use_decoupled_grad",
        "store_param_remainders",
        "set_grad_none",
    }
    assert expected_params.issubset(signature.parameters)
    assert signature.parameters["master_weights"].default is False


def test_fused_adam_signature_matches_te():
    signature = inspect.signature(FusedAdam.__init__)
    expected_names = [
        "self",
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "bias_correction",
        "adam_w_mode",
        "capturable",
        "master_weights",
        "master_weight_dtype",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
        "use_decoupled_grad",
        "store_param_remainders",
        "set_grad_none",
    ]

    assert list(signature.parameters) == expected_names
    for name in expected_names[:7]:
        assert signature.parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in expected_names[7:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_fused_adam_defaults_match_te():
    signature = inspect.signature(FusedAdam.__init__)
    expected_defaults = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "bias_correction": True,
        "adam_w_mode": True,
        "capturable": False,
        "master_weights": False,
        "master_weight_dtype": torch.float32,
        "exp_avg_dtype": torch.float32,
        "exp_avg_sq_dtype": torch.float32,
        "use_decoupled_grad": False,
        "store_param_remainders": False,
        "set_grad_none": None,
    }

    for name, expected_default in expected_defaults.items():
        assert signature.parameters[name].default == expected_default


def test_capturable_explicitly_not_supported_on_npu():
    with pytest.raises(NotImplementedError, match="capturable=True"):
        FusedAdam(_make_linear().parameters(), lr=1e-3, capturable=True)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"master_weights": True, "master_weight_dtype": torch.float16}, "torch.float32"),
        ({"master_weights": True}, "only BF16"),
    ],
)
def test_store_param_remainders_rejects_unsupported_combinations(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        FusedAdam(
            _make_linear().parameters(),
            lr=1e-3,
            store_param_remainders=True,
            **kwargs,
        )


def test_store_param_remainders_without_master_weights_is_disabled():
    optimizer = FusedAdam(
        _make_linear().parameters(),
        lr=1e-3,
        store_param_remainders=True,
    )
    assert optimizer.store_param_remainders is False


def test_master_weight_dtype_fp16_explicit_behavior():
    model = _make_linear().to(dtype=torch.float16)
    optimizer = FusedAdam(
        model.parameters(),
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=torch.float16,
    )

    x = _randn((3, 8), dtype=torch.float16)
    target = _randn((3, 4), dtype=torch.float16)
    _mse_step(model, optimizer, x, target)

    for param in model.parameters():
        assert optimizer.state[param]["master_param"].dtype == torch.float16
        assert optimizer.get_unscaled_state(param, "master_param").dtype == torch.float32


@pytest.mark.parametrize("state_dtype_name", ["exp_avg_dtype", "exp_avg_sq_dtype"])
def test_unsupported_state_dtype_errors_include_argument_name(state_dtype_name):
    kwargs = {state_dtype_name: torch.int8}

    with pytest.raises(NotImplementedError, match=state_dtype_name):
        FusedAdam(_make_linear().parameters(), lr=1e-3, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_unsupported_parameter_dtype_errors_explicitly(dtype):
    values = torch.ones(4, dtype=dtype, device=_device())
    param = torch.nn.Parameter(values, requires_grad=torch.is_floating_point(values))
    optimizer = FusedAdam([param], lr=1e-3)
    if param.requires_grad:
        param.grad = torch.ones_like(param)

    with pytest.raises(
        RuntimeError,
        match="torch.float32, torch.float16, and torch.bfloat16",
    ):
        optimizer.step()


@pytest.mark.parametrize("numel", [1, 7])
@pytest.mark.parametrize("decoupled", [False, True])
def test_adamw_matches_manual_adamw_reference(optimizer_param, numel, decoupled):
    param = optimizer_param(_make_param(values=torch.linspace(-0.5, 0.5, numel)))
    ref_param = _local(param).detach().float().clone()
    ref_exp_avg = torch.zeros_like(ref_param)
    ref_exp_avg_sq = torch.zeros_like(ref_param)
    kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "bias_correction": True,
        "adam_w_mode": True,
    }
    optimizer = FusedAdam([param], use_decoupled_grad=decoupled, **kwargs)

    for step in range(1, 6):
        grad = _param_grad_like(param, step)
        if decoupled:
            param.decoupled_grad = grad
        else:
            param.grad = grad
        optimizer.step()
        ref_param, ref_exp_avg, ref_exp_avg_sq = _manual_adam_update(
            ref_param,
            _local(grad),
            ref_exp_avg,
            ref_exp_avg_sq,
            **kwargs,
            step=step,
        )
        optimizer.zero_grad()

    torch.testing.assert_close(_local(param).detach().float(), ref_param, rtol=1e-6, atol=1e-7)


def test_adam_l2_matches_manual_adam_reference():
    param = _make_param(dtype=torch.float32)
    ref_param = param.detach().float().clone()
    ref_exp_avg = torch.zeros_like(ref_param)
    ref_exp_avg_sq = torch.zeros_like(ref_param)
    kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "bias_correction": True,
        "adam_w_mode": False,
    }
    optimizer = FusedAdam([param], **kwargs)

    for step in range(1, 6):
        grad = _param_grad_like(param, step)
        param.grad = grad
        optimizer.step()
        ref_param, ref_exp_avg, ref_exp_avg_sq = _manual_adam_update(
            ref_param,
            grad,
            ref_exp_avg,
            ref_exp_avg_sq,
            **kwargs,
            step=step,
        )
        optimizer.zero_grad()

    torch.testing.assert_close(param.detach().float(), ref_param, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("adam_w_mode", [False, True])
def test_adam_bias_correction_false_explicitly_not_supported(adam_w_mode):
    with pytest.raises(NotImplementedError, match="bias_correction=True"):
        FusedAdam(
            [_make_param(dtype=torch.float32)],
            lr=1e-3,
            weight_decay=0.01,
            bias_correction=False,
            adam_w_mode=adam_w_mode,
        )


def test_adam_multi_params():
    torch.manual_seed(9876)
    options = {
        "lr": 5e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
    }
    tensors = [_randn(shape) for shape in ((17, 13), (29,), (8, 7), (1,))]
    ref_params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    ref_optimizer = torch.optim.Adam(ref_params, **options)
    optimizer = FusedAdam(params, **options)

    for _ in range(3):
        for ref_param, param in zip(ref_params, params):
            ref_param.grad = torch.rand_like(ref_param)
            param.grad = ref_param.grad
        ref_optimizer.step()
        optimizer.step()

    for param, ref_param in zip(params, ref_params):
        torch.testing.assert_close(param, ref_param, rtol=1e-4, atol=1e-5)


def test_adam_nondefault_options():
    torch.manual_seed(9876)
    options = {
        "lr": 0.01,
        "betas": (0.6, 0.9),
        "eps": 3e-6,
        "weight_decay": 0.0,
        "amsgrad": False,
    }
    ref_param = _make_param()
    param = torch.nn.Parameter(ref_param.detach().clone())
    ref_optimizer = torch.optim.Adam([ref_param], **options)
    optimizer = FusedAdam([param], **options)

    for _ in range(7):
        ref_param.grad = torch.rand_like(ref_param)
        param.grad = ref_param.grad
        ref_optimizer.step()
        optimizer.step()

    torch.testing.assert_close(param, ref_param, rtol=1e-4, atol=1e-5)


def test_adam_empty_param_group_advances_step():
    param = _make_param()
    optimizer = FusedAdam(
        [{"params": [param]}, {"params": []}],
        lr=1e-3,
    )

    for step in range(1, 4):
        param.grad = _param_grad_like(param, step)
        optimizer.step()

    populated_step, empty_step = (group["step"] for group in optimizer.param_groups)
    assert populated_step == 3
    assert empty_step == populated_step
    checkpoint = optimizer.state_dict()
    assert checkpoint["param_groups"][1]["step"] == 3
    assert isinstance(checkpoint["param_groups"][1]["step"], int)


def test_adam_empty_param_at_end_of_group():
    tensors = [
        torch.ones(4, dtype=torch.float32, device=_device()),
        torch.empty(0, dtype=torch.float32, device=_device()),
    ]
    ref_params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    params = [torch.nn.Parameter(tensor.clone()) for tensor in tensors]
    ref_optimizer = torch.optim.Adam(ref_params, lr=1e-3)
    optimizer = FusedAdam(params, lr=1e-3)

    for ref_param, param in zip(ref_params, params):
        ref_param.grad = torch.rand_like(ref_param)
        param.grad = ref_param.grad
    ref_optimizer.step()
    optimizer.step()

    for param, ref_param in zip(params, ref_params):
        torch.testing.assert_close(param, ref_param, rtol=1e-4, atol=1e-5)


def test_adam_multiple_param_groups():
    weight = _make_param(dtype=torch.float32)
    bias = _make_param(
        dtype=torch.float32,
        values=torch.linspace(0.25, 0.95, steps=8, dtype=torch.float32, device="cpu"),
    )
    ref_weight = weight.detach().float().clone()
    ref_bias = bias.detach().float().clone()
    weight_exp_avg = torch.zeros_like(ref_weight)
    weight_exp_avg_sq = torch.zeros_like(ref_weight)
    bias_exp_avg = torch.zeros_like(ref_bias)
    bias_exp_avg_sq = torch.zeros_like(ref_bias)
    common_kwargs = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "bias_correction": True,
        "adam_w_mode": True,
    }
    optimizer = FusedAdam(
        [
            {"params": [weight], "weight_decay": 0.01},
            {"params": [bias], "weight_decay": 0.0},
        ],
        **common_kwargs,
    )

    for step in range(1, 6):
        weight_grad = _param_grad_like(weight, step)
        bias_grad = _param_grad_like(bias, step + 10)
        weight.grad = weight_grad
        bias.grad = bias_grad
        optimizer.step()
        ref_weight, weight_exp_avg, weight_exp_avg_sq = _manual_adam_update(
            ref_weight,
            weight_grad,
            weight_exp_avg,
            weight_exp_avg_sq,
            weight_decay=0.01,
            **common_kwargs,
            step=step,
        )
        ref_bias, bias_exp_avg, bias_exp_avg_sq = _manual_adam_update(
            ref_bias,
            bias_grad,
            bias_exp_avg,
            bias_exp_avg_sq,
            weight_decay=0.0,
            **common_kwargs,
            step=step,
        )
        optimizer.zero_grad()

    assert optimizer.param_groups[0]["step"] == 5
    assert optimizer.param_groups[1]["step"] == 5
    torch.testing.assert_close(
        weight.detach().float(),
        ref_weight,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(bias.detach().float(), ref_bias, rtol=1e-6, atol=1e-7)


def test_adam_state_dict_load_state_dict():
    torch.manual_seed(1234)
    model = _make_linear()
    optimizer = FusedAdam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        bias_correction=True,
        adam_w_mode=True,
    )

    x = _randn((3, 8))
    target = _randn((3, 4))
    for _ in range(2):
        _mse_step(model, optimizer, x, target)

    state_dict = optimizer.state_dict()
    state_count = len(state_dict["state"])
    optimizer2 = FusedAdam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        bias_correction=True,
        adam_w_mode=True,
    )
    optimizer2.load_state_dict(state_dict)

    assert len(optimizer2.state_dict()["state"]) == state_count
    _mse_step(model, optimizer2, x, target)
    assert len(optimizer2.state_dict()["state"]) == state_count


def test_adam_group_step_saved_and_loaded():
    param = _make_param()
    optimizer = FusedAdam([param], lr=1e-3)
    _adam_param_step(param, optimizer, 0)
    _adam_param_step(param, optimizer, 1)

    state_dict = optimizer.state_dict()
    assert state_dict["param_groups"][0]["step"] == 2
    assert isinstance(state_dict["param_groups"][0]["step"], int)

    target_param = _make_param()
    target_optimizer = FusedAdam([target_param], lr=1e-3)
    target_optimizer.load_state_dict(state_dict)

    assert target_optimizer.param_groups[0]["step"] == 2
    assert isinstance(target_optimizer.param_groups[0]["step"], int)
    _adam_param_step(target_param, target_optimizer, 2)
    assert target_optimizer.param_groups[0]["step"] == 3


def test_adam_zero_grad():
    torch.manual_seed(1234)
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3)
    x = _randn((3, 8))
    target = _randn((3, 4))

    _backward_only(model, x, target)
    optimizer.zero_grad()
    assert all(param.grad is None for param in model.parameters())

    _backward_only(model, x, target)
    optimizer.zero_grad(set_to_none=True)
    assert all(param.grad is None for param in model.parameters())

    _backward_only(model, x, target)
    optimizer.zero_grad(set_to_none=False)
    for param in model.parameters():
        assert param.grad is not None
        torch.testing.assert_close(param.grad, torch.zeros_like(param.grad))


def test_adam_set_grad_none_constructor_warns():
    with pytest.warns(DeprecationWarning, match="set_grad_none"):
        FusedAdam(_make_linear().parameters(), lr=1e-3, set_grad_none=True)


def test_adam_set_grad_none_conflicting_zero_grad_raises():
    with pytest.warns(DeprecationWarning, match="set_grad_none"):
        optimizer = FusedAdam(_make_linear().parameters(), lr=1e-3, set_grad_none=True)

    with pytest.raises(ValueError, match="set_grad_none=True"):
        optimizer.zero_grad(set_to_none=False)


def test_adam_zero_grad_default_is_set_to_none_true():
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3)
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    optimizer.zero_grad()

    assert all(param.grad is None for param in model.parameters())


def test_adam_zero_grad_false_zeros_regular_grads_without_decoupled_grad():
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3, use_decoupled_grad=False)
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    optimizer.zero_grad(set_to_none=False)

    for param in model.parameters():
        assert param.grad is not None
        torch.testing.assert_close(param.grad, torch.zeros_like(param.grad))


def test_adam_zero_grad_decoupled_default_clears_only_decoupled_grad():
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3, use_decoupled_grad=True)
    grad_refs = []
    for param in model.parameters():
        param.grad = torch.ones_like(param)
        param.decoupled_grad = torch.full_like(param, 2.0)
        grad_refs.append(param.grad.detach().clone())

    optimizer.zero_grad()

    for param, grad_ref in zip(model.parameters(), grad_refs):
        assert param.decoupled_grad is None
        torch.testing.assert_close(param.grad, grad_ref)


def test_adam_zero_grad_decoupled_false_zeros_only_decoupled_grad():
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3, use_decoupled_grad=True)
    grad_refs = []
    for param in model.parameters():
        param.grad = torch.ones_like(param)
        param.decoupled_grad = torch.full_like(param, 2.0)
        grad_refs.append(param.grad.detach().clone())

    optimizer.zero_grad(set_to_none=False)

    for param, grad_ref in zip(model.parameters(), grad_refs):
        torch.testing.assert_close(param.decoupled_grad, torch.zeros_like(param))
        torch.testing.assert_close(param.grad, grad_ref)


def test_adam_zero_grad_decoupled_false_missing_decoupled_grad_is_noop():
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3, use_decoupled_grad=True)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
        if hasattr(param, "decoupled_grad"):
            delattr(param, "decoupled_grad")

    optimizer.zero_grad(set_to_none=False)

    for param in model.parameters():
        assert not hasattr(param, "decoupled_grad")


def test_adam_zero_grad_decoupled_false_after_set_to_none_is_noop():
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3, use_decoupled_grad=True)
    grad_refs = []
    for param in model.parameters():
        param.grad = torch.ones_like(param)
        param.decoupled_grad = torch.full_like(param, 2.0)
        grad_refs.append(param.grad.detach().clone())

    optimizer.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=False)

    for param, grad_ref in zip(model.parameters(), grad_refs):
        assert param.decoupled_grad is None
        torch.testing.assert_close(param.grad, grad_ref)


def test_adam_skip_none_grad():
    torch.manual_seed(1234)
    model = _make_linear()
    optimizer = FusedAdam(model.parameters(), lr=1e-3)
    x = _randn((3, 8))
    target = _randn((3, 4))

    _backward_only(model, x, target)
    before_weight = model.weight.detach().clone()
    before_bias = model.bias.detach().clone()
    model.weight.grad = None

    optimizer.step()

    torch.testing.assert_close(model.weight, before_weight)
    assert (model.bias.detach() - before_bias).abs().max().item() > 0.0
    assert optimizer.param_groups[0]["step"] == 1


def test_adam_none_grad_uses_group_step_when_grad_returns():
    param = _make_param()
    optimizer = FusedAdam([param], lr=1e-3, betas=(0.9, 0.999), bias_correction=True)

    before = param.detach().clone()
    optimizer.step()
    assert optimizer.param_groups[0]["step"] == 1
    assert isinstance(optimizer.param_groups[0]["step"], int)
    torch.testing.assert_close(param.detach(), before)

    grad = _param_grad_like(param, 0)
    param.grad = grad
    optimizer.step()
    assert optimizer.param_groups[0]["step"] == 2
    ref_param, _, _ = _manual_adam_update(
        before,
        grad,
        torch.zeros_like(before, dtype=torch.float32),
        torch.zeros_like(before, dtype=torch.float32),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
        bias_correction=True,
        step=2,
    )
    torch.testing.assert_close(param.detach().float(), ref_param, rtol=1e-6, atol=1e-7)

    param.grad = None
    optimizer.step()
    assert optimizer.param_groups[0]["step"] == 3
    before_return = param.detach().clone()

    param.grad = _param_grad_like(param, 2)
    optimizer.step()
    assert optimizer.param_groups[0]["step"] == 4
    assert (param.detach() - before_return).abs().max().item() > 0.0


def test_adam_sparse_grad_raises():
    param = torch.nn.Parameter(torch.ones(4, device=_device()))
    optimizer = FusedAdam([param], lr=1e-3)
    param.grad = _sparse_grad_like(param)

    with pytest.raises(RuntimeError, match="sparse gradients"):
        optimizer.step()


def test_adam_capturable_explicitly_not_supported_on_npu():
    with pytest.raises(NotImplementedError, match="capturable=True"):
        FusedAdam(_make_linear().parameters(), lr=1e-3, capturable=True)


@pytest.mark.parametrize(
    ("exp_avg_dtype", "exp_avg_sq_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.uint8, torch.uint8),
    ],
)
def test_adam_store_param_remainders_initialize_and_update(exp_avg_dtype, exp_avg_sq_dtype):
    if torch.uint8 in (exp_avg_dtype, exp_avg_sq_dtype):
        _skip_if_fp8_state_unavailable()
    param = _make_param(dtype=torch.bfloat16)
    initial_master = param.detach().float().cpu()
    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        master_weights=True,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
        store_param_remainders=True,
    )
    optimizer.initialize_state(param, store_param_remainders=True)
    state = optimizer.state[param]
    assert state["master_param"].dtype == torch.int16
    if exp_avg_dtype == torch.uint8:
        assert isinstance(state["exp_avg"], Float8Tensor)
    else:
        assert state["exp_avg"].dtype == exp_avg_dtype
    if exp_avg_sq_dtype == torch.uint8:
        assert isinstance(state["exp_avg_sq"], Float8Tensor)
    else:
        assert state["exp_avg_sq"].dtype == exp_avg_sq_dtype
    assert torch.count_nonzero(state["master_param"]).item() == 0

    grad = _param_grad_like(param)
    param.grad = grad
    optimizer.step()
    expected, expected_m, expected_v = _manual_adam_update(
        initial_master,
        grad.cpu(),
        torch.zeros_like(initial_master),
        torch.zeros_like(initial_master),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
        bias_correction=True,
        step=1,
    )
    torch.testing.assert_close(
        _reconstruct_remainder_master(param, state["master_param"]),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
    exp_avg_rtol = (
        1e-6 if exp_avg_dtype == torch.float32 else 2e-1 if exp_avg_dtype == torch.uint8 else 5e-3
    )
    exp_avg_atol = 5e-4 if exp_avg_dtype != torch.float32 else 1e-7
    exp_avg_sq_rtol = (
        1e-6
        if exp_avg_sq_dtype == torch.float32
        else 2e-1
        if exp_avg_sq_dtype == torch.uint8
        else 5e-3
    )
    exp_avg_sq_atol = 5e-4 if exp_avg_sq_dtype != torch.float32 else 1e-7
    actual_m = optimizer.get_unscaled_state(param, "exp_avg").cpu()
    actual_v = optimizer.get_unscaled_state(param, "exp_avg_sq").cpu()
    torch.testing.assert_close(actual_m, expected_m, rtol=exp_avg_rtol, atol=exp_avg_atol)
    torch.testing.assert_close(actual_v, expected_v, rtol=exp_avg_sq_rtol, atol=exp_avg_sq_atol)

    next_grad = _param_grad_like(param, 1)
    param.grad = next_grad
    optimizer.step()
    expected, expected_m, expected_v = _manual_adam_update(
        expected,
        next_grad.cpu(),
        actual_m,
        actual_v,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
        bias_correction=True,
        step=2,
    )
    torch.testing.assert_close(
        _reconstruct_remainder_master(param, state["master_param"]),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
    actual_m = optimizer.get_unscaled_state(param, "exp_avg").cpu()
    actual_v = optimizer.get_unscaled_state(param, "exp_avg_sq").cpu()
    torch.testing.assert_close(actual_m, expected_m, rtol=exp_avg_rtol, atol=exp_avg_atol)
    torch.testing.assert_close(actual_v, expected_v, rtol=exp_avg_sq_rtol, atol=exp_avg_sq_atol)


@pytest.mark.parametrize(
    ("exp_avg_dtype", "exp_avg_sq_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.uint8, torch.uint8),
    ],
)
def test_adam_store_param_remainders_checkpoint_resume(exp_avg_dtype, exp_avg_sq_dtype):
    if torch.uint8 in (exp_avg_dtype, exp_avg_sq_dtype):
        _skip_if_fp8_state_unavailable()
    param = _make_param(dtype=torch.bfloat16)
    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        master_weights=True,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
        store_param_remainders=True,
    )
    _adam_param_step(param, optimizer, 0)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint_state = _first_optimizer_state(checkpoint)
    assert checkpoint_state["master_param"].dtype == torch.int16
    checkpoint_state["master_param"] = checkpoint_state["master_param"].float().cpu()
    assert checkpoint["param_groups"][0]["step"] == 1

    target_param = torch.nn.Parameter(param.detach().clone())
    target_optimizer = FusedAdam(
        [target_param],
        lr=1e-3,
        master_weights=True,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
        store_param_remainders=True,
    )
    target_optimizer.load_state_dict(checkpoint)
    assert target_optimizer.param_groups[0]["step"] == 1
    assert torch.equal(
        target_optimizer.state[target_param]["master_param"],
        optimizer.state[param]["master_param"],
    )

    param.grad = _param_grad_like(param, 1)
    target_param.grad = param.grad.clone()
    optimizer.step()
    target_optimizer.step()
    assert torch.equal(target_param, param)
    assert torch.equal(
        target_optimizer.state[target_param]["master_param"],
        optimizer.state[param]["master_param"],
    )


def test_adam_store_param_remainders_rejects_fp32_master_checkpoint():
    param = _make_param(dtype=torch.bfloat16)
    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        master_weights=True,
        store_param_remainders=True,
    )

    with pytest.raises(RuntimeError, match="cannot be losslessly restored"):
        optimizer.set_scaled_state(param, "master_param", param.detach().float().cpu())


def test_adam_unsupported_master_weight_dtype_raises_on_master_path():
    model = _make_linear().to(dtype=torch.float16)

    with pytest.raises(NotImplementedError, match="master_weight_dtype"):
        FusedAdam(
            model.parameters(),
            lr=1e-3,
            master_weights=True,
            master_weight_dtype=torch.bfloat16,
        )


def test_adam_use_decoupled_grad():
    torch.manual_seed(1234)
    model = _make_linear()
    optimizer = FusedAdam(
        model.parameters(),
        lr=1e-3,
        adam_w_mode=True,
        use_decoupled_grad=True,
    )

    params = list(model.parameters())
    before_params = [param.detach().clone() for param in params]
    try:
        for param in params:
            param.grad = None
            param.decoupled_grad = torch.ones_like(param)

        optimizer.step()

        for before_param, param in zip(before_params, params):
            assert (param.detach() - before_param).abs().max().item() > 0.0
    finally:
        for param in params:
            if hasattr(param, "decoupled_grad"):
                delattr(param, "decoupled_grad")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_adam_fp16_bf16_params_step_with_fp32_default_states(dtype):
    param = _make_param(dtype=dtype)
    optimizer = FusedAdam([param], lr=1e-3)

    _adam_param_step(param, optimizer, 0)
    state = optimizer.state[param]

    assert param.dtype == dtype
    assert torch.isfinite(param.detach().float()).all()
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert "master_param" not in state


def test_adam_default_state_scales_are_reused():
    param = _make_param(dtype=torch.float32)
    optimizer = FusedAdam([param], lr=1e-3)

    _adam_param_step(param, optimizer, 0)
    scales = {name: optimizer._scales[param][name] for name in ("exp_avg", "exp_avg_sq")}
    _adam_param_step(param, optimizer, 1)

    for name, scale in scales.items():
        assert optimizer._scales[param][name] is scale


@pytest.mark.parametrize(
    ("exp_avg_dtype", "exp_avg_sq_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_adam_low_precision_state_checkpoint_semantics(
    exp_avg_dtype, exp_avg_sq_dtype, optimizer_param
):
    param = optimizer_param(_make_param(dtype=torch.float16))
    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
    )
    _adam_param_step(param, optimizer, 0)

    state = optimizer.state[param]
    assert state["exp_avg"].dtype == exp_avg_dtype
    assert state["exp_avg_sq"].dtype == exp_avg_sq_dtype
    assert optimizer.get_unscaled_state(param, "exp_avg").dtype == torch.float32
    assert optimizer.get_unscaled_state(param, "exp_avg_sq").dtype == torch.float32
    if exp_avg_dtype == torch.bfloat16:
        torch.testing.assert_close(
            optimizer._scales[param]["exp_avg"],
            torch.ones((), device=param.device, dtype=torch.float32),
        )
    if exp_avg_sq_dtype == torch.bfloat16:
        torch.testing.assert_close(
            optimizer._scales[param]["exp_avg_sq"],
            torch.ones((), device=param.device, dtype=torch.float32),
        )

    state_dict = optimizer.state_dict()
    checkpoint_state = _first_optimizer_state(state_dict)
    assert checkpoint_state["exp_avg"].dtype == torch.float32
    assert checkpoint_state["exp_avg_sq"].dtype == torch.float32

    if isinstance(param, DTensor):
        for value in checkpoint_state.values():
            assert isinstance(value, DTensor)
            assert value.device_mesh == param.device_mesh
            assert value.placements == param.placements
            assert value.shape == param.shape
            assert value.stride() == param.stride()

    target_param = optimizer_param(_make_param(dtype=torch.float16))
    target_optimizer = FusedAdam(
        [target_param],
        lr=1e-3,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
    )
    target_optimizer.load_state_dict(state_dict)
    target_state = target_optimizer.state[target_param]
    assert target_state["exp_avg"].dtype == exp_avg_dtype
    assert target_state["exp_avg_sq"].dtype == exp_avg_sq_dtype
    with torch.no_grad():
        _local(target_param).copy_(_local(param))
    _adam_param_step(param, optimizer, 1)
    _adam_param_step(target_param, target_optimizer, 1)
    torch.testing.assert_close(_local(target_param), _local(param))


@pytest.mark.parametrize("param_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("master_weight_dtype", [torch.float32, torch.float16])
def test_adam_master_weights_checkpoint_semantics(
    param_dtype, master_weight_dtype, optimizer_param
):
    param = optimizer_param(_make_param(dtype=param_dtype))
    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=master_weight_dtype,
    )
    _adam_param_step(param, optimizer, 0)

    state = optimizer.state[param]
    assert state["master_param"].dtype == master_weight_dtype
    assert optimizer.get_unscaled_state(param, "master_param").dtype == torch.float32
    torch.testing.assert_close(
        _local(param).detach(),
        optimizer.get_unscaled_state(param, "master_param").to(dtype=param_dtype),
        rtol=1e-2,
        atol=1e-2,
    )

    state_dict = optimizer.state_dict()
    checkpoint_state = _first_optimizer_state(state_dict)
    assert checkpoint_state["master_param"].dtype == torch.float32

    target_param = optimizer_param(_make_param(dtype=param_dtype))
    target_optimizer = FusedAdam(
        [target_param],
        lr=1e-3,
        master_weights=True,
        master_weight_dtype=master_weight_dtype,
    )
    target_optimizer.load_state_dict(state_dict)
    target_state = target_optimizer.state[target_param]
    assert target_state["master_param"].dtype == master_weight_dtype
    assert target_optimizer.get_unscaled_state(target_param, "master_param").dtype == torch.float32
    with torch.no_grad():
        _local(target_param).copy_(_local(param))
    _adam_param_step(param, optimizer, 1)
    _adam_param_step(target_param, target_optimizer, 1)
    torch.testing.assert_close(_local(target_param), _local(param))


@pytest.mark.parametrize("quantization", ["delayed", "current", "blockwise", "mxfp8"])
def test_adam_quantized_param_master_update_preserves_storage(quantization):
    try:
        param = _make_quantized_param(quantization)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"{quantization} model parameters are unavailable: {exc}")

    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        master_weights=True,
        use_decoupled_grad=True,
    )
    optimizer.initialize_state(param)
    initial_master = optimizer.state[param]["master_param"].detach().clone()
    storage_ptrs = {
        name: value.data_ptr()
        for name, value in param.get_metadata().items()
        if torch.is_tensor(value)
    }
    quantizer = param._get_quantizer()
    quantizer_metadata_ptrs = {
        name: value.data_ptr() for name, value in vars(quantizer).items() if torch.is_tensor(value)
    }
    grad = torch.full(param.shape, 0.25, dtype=torch.bfloat16, device=param.device)
    param.decoupled_grad = grad
    optimizer.step()

    expected, _, _ = _manual_adam_update(
        initial_master,
        grad,
        torch.zeros_like(initial_master),
        torch.zeros_like(initial_master),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
        bias_correction=True,
        step=1,
    )
    torch.testing.assert_close(
        optimizer.state[param]["master_param"],
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
    assert param._get_quantizer() is quantizer
    assert storage_ptrs == {
        name: value.data_ptr()
        for name, value in param.get_metadata().items()
        if torch.is_tensor(value)
    }
    assert quantizer_metadata_ptrs == {
        name: value.data_ptr() for name, value in vars(quantizer).items() if torch.is_tensor(value)
    }


@pytest.mark.parametrize("moment_dtype", [torch.float32, torch.bfloat16])
def test_adam_delayed_quantized_param_keeps_fp32_master_with_remainders(moment_dtype):
    try:
        param = _make_quantized_param("delayed")
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"delayed model parameters are unavailable: {exc}")

    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        master_weights=True,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
        use_decoupled_grad=True,
        store_param_remainders=True,
    )
    optimizer.initialize_state(param, store_param_remainders=False)
    assert optimizer.store_param_remainders is True
    assert optimizer.state[param]["master_param"].dtype == torch.float32
    assert optimizer.state[param]["exp_avg"].dtype == moment_dtype
    assert optimizer.state[param]["exp_avg_sq"].dtype == moment_dtype

    initial_master = optimizer.state[param]["master_param"].detach().clone()
    quantizer = param._get_quantizer()
    payload_ptr = param._data.data_ptr()
    scale_inv_ptr = param._scale_inv.data_ptr()
    quantizer.amax.zero_()
    grad = torch.full(param.shape, 0.25, dtype=torch.bfloat16, device=param.device)
    param.decoupled_grad = grad
    optimizer.step()

    expected, expected_m, expected_v = _manual_adam_update(
        initial_master,
        grad,
        torch.zeros_like(initial_master),
        torch.zeros_like(initial_master),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
        bias_correction=True,
        step=1,
    )
    torch.testing.assert_close(
        optimizer.state[param]["master_param"],
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
    moment_rtol = 5e-3 if moment_dtype == torch.bfloat16 else 1e-6
    moment_atol = 5e-4 if moment_dtype == torch.bfloat16 else 1e-7
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"].float(),
        expected_m,
        rtol=moment_rtol,
        atol=moment_atol,
    )
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg_sq"].float(),
        expected_v,
        rtol=moment_rtol,
        atol=moment_atol,
    )
    assert param._data.data_ptr() == payload_ptr
    assert param._scale_inv.data_ptr() == scale_inv_ptr
    assert optimizer._fp8_adam_metadata_cache

    expected_quantizer = Float8Quantizer(
        scale=quantizer.scale.detach().clone(),
        amax=torch.zeros_like(quantizer.amax),
        fp8_dtype=param._fp8_dtype,
        rowwise=True,
        columnwise=False,
    )
    expected_param = expected_quantizer(optimizer.state[param]["master_param"])
    torch.testing.assert_close(
        param._data.detach().contiguous().view(torch.uint8),
        expected_param._data.detach().contiguous().view(torch.uint8),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(quantizer.amax, expected_quantizer.amax, rtol=0.0, atol=0.0)
    torch.testing.assert_close(param._scale_inv, expected_param._scale_inv, rtol=0.0, atol=0.0)

    quantizer.amax.zero_()
    next_grad = torch.full(param.shape, -0.125, dtype=torch.bfloat16, device=param.device)
    param.decoupled_grad = next_grad
    optimizer.step()
    expected, expected_m, expected_v = _manual_adam_update(
        expected,
        next_grad,
        expected_m.to(moment_dtype).float(),
        expected_v.to(moment_dtype).float(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
        bias_correction=True,
        step=2,
    )
    torch.testing.assert_close(
        optimizer.state[param]["master_param"],
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg"].float(),
        expected_m,
        rtol=moment_rtol,
        atol=moment_atol,
    )
    torch.testing.assert_close(
        optimizer.state[param]["exp_avg_sq"].float(),
        expected_v,
        rtol=moment_rtol,
        atol=moment_atol,
    )

    checkpoint = copy.deepcopy(optimizer.state_dict())
    assert _first_optimizer_state(checkpoint)["master_param"].dtype == torch.float32

    target_param = _make_quantized_param("delayed")
    target_optimizer = FusedAdam(
        [target_param],
        lr=1e-3,
        master_weights=True,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
        use_decoupled_grad=True,
        store_param_remainders=True,
    )
    target_optimizer.load_state_dict(checkpoint)
    assert target_optimizer.state[target_param]["master_param"].dtype == torch.float32
    assert target_optimizer.state[target_param]["exp_avg"].dtype == moment_dtype
    assert target_optimizer.state[target_param]["exp_avg_sq"].dtype == moment_dtype
    torch.testing.assert_close(
        target_optimizer.state[target_param]["master_param"],
        optimizer.state[param]["master_param"],
        rtol=0.0,
        atol=0.0,
    )


def test_adam_delayed_quantized_params_share_triton_metadata():
    try:
        params = [_make_quantized_param("delayed") for _ in range(2)]
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"delayed model parameters are unavailable: {exc}")

    optimizer = FusedAdam(
        params,
        lr=1e-3,
        master_weights=True,
        use_decoupled_grad=True,
        store_param_remainders=True,
    )
    initial_masters = []
    for index, param in enumerate(params):
        optimizer.initialize_state(param, store_param_remainders=False)
        initial_masters.append(optimizer.state[param]["master_param"].detach().clone())
        param._get_quantizer().amax.zero_()
        param.decoupled_grad = torch.full(
            param.shape,
            0.25 + index * 0.125,
            dtype=torch.bfloat16,
            device=param.device,
        )

    optimizer.step()
    assert len(optimizer._fp8_adam_metadata_cache) == 1
    signature, metadata, _ = next(iter(optimizer._fp8_adam_metadata_cache.values()))
    assert len(signature) == len(params)
    metadata_ptr = metadata.data_ptr()
    for param, initial_master in zip(params, initial_masters):
        assert not torch.equal(optimizer.state[param]["master_param"], initial_master)
        param._get_quantizer().amax.zero_()

    optimizer.step()
    _, reused_metadata, _ = next(iter(optimizer._fp8_adam_metadata_cache.values()))
    assert reused_metadata.data_ptr() == metadata_ptr


def test_adam_quantized_param_requires_master_weights():
    try:
        param = _make_quantized_param("current")
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"current-scaling model parameters are unavailable: {exc}")
    optimizer = FusedAdam([param], lr=1e-3, use_decoupled_grad=True)
    param.decoupled_grad = torch.ones(param.shape, dtype=param.dtype, device=param.device)
    with pytest.raises(RuntimeError, match="require master_weights=True"):
        optimizer.step()


def test_adam_load_state_dict_rejects_remainder_without_remainder_mode():
    param = _make_param(dtype=torch.bfloat16)
    optimizer = FusedAdam([param], lr=1e-3, master_weights=True)
    _adam_param_step(param, optimizer, 0)
    state_dict = optimizer.state_dict()
    checkpoint_state = _first_optimizer_state(state_dict)
    checkpoint_state["master_param"] = torch.zeros_like(
        checkpoint_state["master_param"],
        dtype=torch.int16,
    )

    target_param = _make_param(dtype=torch.bfloat16)
    target_optimizer = FusedAdam([target_param], lr=1e-3, master_weights=True)
    with pytest.raises(RuntimeError, match="store_param_remainders"):
        target_optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize(
    ("exp_avg_dtype", "exp_avg_sq_dtype"),
    [
        (torch.uint8, torch.float32),
        (torch.float32, torch.uint8),
        (torch.uint8, torch.uint8),
    ],
)
def test_adam_fp8_state_checkpoint_semantics(exp_avg_dtype, exp_avg_sq_dtype):
    _skip_if_fp8_state_unavailable()
    param = _make_param(dtype=torch.float32)
    optimizer = FusedAdam(
        [param],
        lr=1e-3,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
    )
    _adam_param_step(param, optimizer, 0)

    state = optimizer.state[param]
    if exp_avg_dtype == torch.uint8:
        assert isinstance(state["exp_avg"], Float8Tensor)
        assert state["exp_avg"].origin_shape == param.shape
        assert state["exp_avg"]._data.shape == param.shape
        assert optimizer.get_unscaled_state(param, "exp_avg").dtype == torch.float32
    else:
        assert state["exp_avg"].dtype == torch.float32
    if exp_avg_sq_dtype == torch.uint8:
        assert isinstance(state["exp_avg_sq"], Float8Tensor)
        assert state["exp_avg_sq"].origin_shape == param.shape
        assert state["exp_avg_sq"]._data.shape == param.shape
        assert optimizer.fp8_exp_avg_sq_storage == "sqrt"
        assert optimizer.get_unscaled_state(param, "exp_avg_sq").dtype == torch.float32
    else:
        assert state["exp_avg_sq"].dtype == torch.float32

    state_dict = optimizer.state_dict()
    checkpoint_state = _first_optimizer_state(state_dict)
    assert checkpoint_state["exp_avg"].dtype == torch.float32
    assert checkpoint_state["exp_avg_sq"].dtype == torch.float32
    assert not isinstance(checkpoint_state["exp_avg"], Float8Tensor)
    assert not isinstance(checkpoint_state["exp_avg_sq"], Float8Tensor)

    target_param = _make_param(dtype=torch.float32)
    target_optimizer = FusedAdam(
        [target_param],
        lr=1e-3,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
    )
    target_optimizer.load_state_dict(state_dict)
    target_state = target_optimizer.state[target_param]
    if exp_avg_dtype == torch.uint8:
        assert isinstance(target_state["exp_avg"], Float8Tensor)
        assert target_state["exp_avg"]._data.shape == target_param.shape
    if exp_avg_sq_dtype == torch.uint8:
        assert isinstance(target_state["exp_avg_sq"], Float8Tensor)
        assert target_state["exp_avg_sq"]._data.shape == target_param.shape
    _adam_param_step(target_param, target_optimizer, 1)


def test_adam_load_state_dict_allows_plain_uint8_placeholder_then_fp32_restore():
    _skip_if_fp8_state_unavailable()
    param = _make_param(dtype=torch.float32)
    optimizer = FusedAdam([param], lr=1e-3)
    _adam_param_step(param, optimizer, 0)
    state_dict = optimizer.state_dict()
    checkpoint_state = _first_optimizer_state(state_dict)
    checkpoint_state["exp_avg"] = torch.zeros_like(
        checkpoint_state["exp_avg"],
        dtype=torch.uint8,
    )

    target_param = _make_param(dtype=torch.float32)
    target_optimizer = FusedAdam(
        [target_param],
        lr=1e-3,
        exp_avg_dtype=torch.uint8,
        exp_avg_sq_dtype=torch.float32,
    )
    target_optimizer.load_state_dict(state_dict)
    assert target_optimizer.state[target_param]["exp_avg"].dtype == torch.uint8

    target_optimizer.set_scaled_state(
        target_param,
        "exp_avg",
        torch.zeros_like(target_param, dtype=torch.float32),
    )
    assert isinstance(target_optimizer.state[target_param]["exp_avg"], Float8Tensor)


def test_adam_plain_uint8_fp8_state_without_restore_is_rejected_on_step():
    _skip_if_fp8_state_unavailable()
    param = _make_param(dtype=torch.float32)
    optimizer = FusedAdam([param], lr=1e-3, exp_avg_dtype=torch.uint8)
    optimizer.state[param]["exp_avg"] = torch.zeros_like(param, dtype=torch.uint8)
    optimizer.state[param]["exp_avg_sq"] = torch.zeros_like(param)
    param.grad = torch.ones_like(param)

    with pytest.raises(RuntimeError, match="legacy plain uint8"):
        optimizer.step()
