# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import math

import pytest
import torch
import torch.distributed as dist

from transformer_engine.common.recipe import W4A16BlockScaling
from transformer_engine.pytorch import GroupedLinear, Linear, autocast
from transformer_engine.pytorch.constants import TensorUsage
from transformer_engine.pytorch.module.base import quantize_weight
from transformer_engine.pytorch.ops.fused.overlap import CommOverlapOps
from transformer_engine.pytorch.ops.gemm import general_gemm, general_gemm_add
from transformer_engine.pytorch.quantization import (
    RecipeState,
    W4A16BlockScalingRecipeState,
    is_w4a16_available,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage
from transformer_engine.pytorch.tensor import (
    DummyQuantizer,
    DummyTensor,
    FakeMXFP4Quantizer,
    MXFP4Quantizer,
)
from transformer_engine.pytorch.utils import init_method_normal

from distributed_testing import DistributedTest
from utils import npu_available


_w4a16_available, _reason_for_no_w4a16 = is_w4a16_available(return_reason=True)
pytestmark = pytest.mark.skipif(
    not npu_available() or not _w4a16_available,
    reason=_reason_for_no_w4a16 or "NPU device is required",
)


def _uniform_npu(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
    values = torch.linspace(
        -1.0,
        1.0,
        steps=math.prod(shape),
        dtype=torch.float32,
        device="cpu",
    ).reshape(shape)
    return values.to(device=torch.device("npu"), dtype=dtype).contiguous()


def _dequantize_mxfp4_oracle(tensor, dtype: torch.dtype) -> torch.Tensor:
    packed = tensor._rowwise_data.contiguous().view(torch.uint8)
    codes = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1).flatten(-2)
    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=packed.device,
        dtype=torch.float32,
    )
    values = table[codes.long()].reshape(-1, tensor.shape[-1])
    num_blocks = tensor.shape[-1] // 32
    scales = torch.pow(2.0, tensor._rowwise_scale_inv.float() - 127.0)
    scales = scales.reshape(values.shape[0], -1)[:, :num_blocks]
    return (
        (values.reshape(values.shape[0], num_blocks, 32) * scales.unsqueeze(-1))
        .reshape(tensor.shape)
        .to(dtype)
    )


def test_dummy_quantizer() -> None:
    tensor = _uniform_npu((2, 2, 8), dtype=torch.bfloat16).requires_grad_()
    quantizer = DummyQuantizer()

    output = quantizer(tensor)
    grouped_output = quantizer.grouped_quantize(tensor, torch.tensor([1, 2], device="npu"))

    assert isinstance(output, DummyTensor)
    assert isinstance(output, QuantizedTensor)
    assert isinstance(grouped_output, DummyTensor)
    assert output.get_data(None).shape == (4, 8)
    assert grouped_output.get_data(None).shape == tensor.shape
    assert not quantizer.can_reuse(tensor)
    assert not quantizer.can_reuse(output)
    torch.testing.assert_close(output.dequantize(), tensor)
    torch.testing.assert_close(grouped_output.dequantize(), tensor)

    output.dequantize().square().sum().backward()
    torch.testing.assert_close(tensor.grad, 2 * tensor)


@pytest.mark.parametrize("grouped", [False, True])
def test_dummy_tensor_metadata_and_saving(grouped) -> None:
    tensor = _uniform_npu((2, 2, 8), dtype=torch.bfloat16)
    output = DummyTensor(tensor.shape, tensor.dtype, data=tensor, grouped=grouped)
    restored = DummyTensor(output.shape, output.dtype, **output.get_metadata())
    assert restored._data.data_ptr() == output._data.data_ptr()
    assert restored._grouped == grouped

    view = DummyTensor.make_like(output, shape=(4, 8))
    assert view.shape == (4, 8)
    assert view._grouped == grouped
    assert view._data.data_ptr() == output._data.data_ptr()
    converted = DummyTensor.make_like(output, dtype=torch.float32, requires_grad=True)
    assert converted.dtype == converted._data.dtype == torch.float32
    assert converted.requires_grad
    torch.testing.assert_close(converted.dequantize(), tensor.float())

    detached = output.detach()
    saved, storage = detached.prepare_for_saving()
    assert detached._data is None
    assert output._data is not None
    assert storage.restore_from_saved(saved) == []
    torch.testing.assert_close(detached.dequantize(), tensor)

    replacement = tensor.flip(0).contiguous()
    detached.copy_(replacement)
    torch.testing.assert_close(output.dequantize(), replacement)
    dense = torch.empty_like(tensor)
    dense.copy_(output)
    torch.testing.assert_close(dense, replacement)


@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
def test_dummy_tensor_dense_gemm(transpose_a, transpose_b, grad_dtype) -> None:
    a = _uniform_npu((32, 16) if transpose_a else (16, 32), dtype=torch.bfloat16)
    b = _uniform_npu((64, 32) if transpose_b else (32, 64), dtype=torch.bfloat16)
    usage_a = TensorUsage.LHS_TRANS if transpose_a else TensorUsage.LHS
    usage_b = TensorUsage.RHS_TRANS if transpose_b else TensorUsage.RHS
    a_mat = a.t() if transpose_a else a
    b_mat = b.t() if transpose_b else b
    expected = a_mat @ b_mat
    quantizer = DummyQuantizer()
    output = general_gemm(quantizer(a), quantizer(b), usage_a, usage_b, a.dtype)
    assert not isinstance(output, QuantizedTensor)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(quantizer(a_mat).matmul(quantizer(b_mat)), expected)

    main_grad = torch.ones_like(expected, dtype=grad_dtype)
    expected_grad = main_grad.clone().addmm_(a_mat.to(grad_dtype), b_mat.to(grad_dtype))
    general_gemm_add(main_grad, quantizer(a), quantizer(b), usage_a, usage_b, grad_dtype)
    torch.testing.assert_close(main_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wrapped_input", [False, True])
def test_fake_mxfp4_quantizer(dtype, wrapped_input) -> None:
    weight = _uniform_npu((64, 64), dtype=dtype).requires_grad_()
    quantizer = FakeMXFP4Quantizer()

    fake_quantized = quantizer(weight)
    expected = MXFP4Quantizer(
        rowwise=True,
        columnwise=False,
        block_size=32,
        with_rht=False,
    ).quantize_impl(weight.detach())
    assert expected._quantizer.can_reuse(expected)
    assert not expected._quantizer.can_reuse(fake_quantized)
    expected = _dequantize_mxfp4_oracle(expected, dtype)
    torch.npu.synchronize()

    assert isinstance(fake_quantized, DummyTensor)
    assert isinstance(fake_quantized, QuantizedTensorStorage)
    assert isinstance(fake_quantized, QuantizedTensor)
    assert fake_quantized.shape == weight.shape
    assert fake_quantized.dtype == weight.dtype
    torch.testing.assert_close(fake_quantized.dequantize(), expected, rtol=0, atol=0)

    fake_quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))

    new_weight = weight.detach().flip(0).contiguous()
    source = DummyQuantizer()(new_weight) if wrapped_input else new_weight
    fresh_weightmat, _ = quantize_weight(tensor=source, quantizer=quantizer)
    torch.testing.assert_close(
        fresh_weightmat.dequantize(), quantizer(new_weight).dequantize(), rtol=0, atol=0
    )
    assert isinstance(fresh_weightmat._quantizer, FakeMXFP4Quantizer)
    workspace = fake_quantized.detach()
    weightmat, new_workspace = quantize_weight(
        tensor=source,
        quantizer=quantizer,
        workspace=workspace,
        update_workspace=True,
        workspace_dtype=new_weight.dtype,
        cache=True,
    )
    expected = quantizer(new_weight)
    torch.npu.synchronize()

    assert weightmat is workspace
    assert new_workspace is None
    torch.testing.assert_close(weightmat.dequantize(), expected.dequantize(), rtol=0, atol=0)

    weightmat, _ = quantize_weight(
        tensor=weight.detach(),
        quantizer=quantizer,
        workspace=workspace,
        update_workspace=False,
    )
    assert weightmat is workspace
    torch.testing.assert_close(weightmat.dequantize(), expected.dequantize(), rtol=0, atol=0)


def test_w4a16_recipe_state() -> None:
    recipe = W4A16BlockScaling()
    fwd_state = RecipeState.create(recipe, mode="forward", num_quantizers=3)
    bwd_state = RecipeState.create(recipe, mode="backward", num_quantizers=2)
    fwd_quantizers = fwd_state.make_quantizers()
    bwd_quantizers = bwd_state.make_quantizers()

    assert isinstance(fwd_state, W4A16BlockScalingRecipeState)
    assert isinstance(fwd_quantizers[0], DummyQuantizer)
    assert isinstance(fwd_quantizers[1], FakeMXFP4Quantizer)
    assert isinstance(fwd_quantizers[2], DummyQuantizer)
    assert all(isinstance(quantizer, DummyQuantizer) for quantizer in bwd_quantizers)


@pytest.mark.parametrize(
    ("module_type", "input_shape"),
    [
        ("linear", (32, 64)),
        ("linear", (8, 1, 64)),
        ("grouped_linear", (5, 64)),
    ],
)
@pytest.mark.parametrize("scope", ["all", "moe-only", "linear-only", "close"])
def test_w4a16_recipe_module_forward_backward(module_type, input_shape, scope) -> None:
    if module_type == "linear":
        block = Linear(
            64,
            64,
            init_method=init_method_normal(0.023),
            params_dtype=torch.bfloat16,
            device="npu",
        )
        group_list = None
    else:
        block = GroupedLinear(
            num_gemms=2,
            in_features=64,
            out_features=64,
            bias=False,
            single_grouped_weight=True,
            params_dtype=torch.bfloat16,
            device="npu",
        )
        group_list = [2, 3]
    inp = _uniform_npu(input_shape, dtype=torch.bfloat16).requires_grad_()

    with autocast(enabled=True, recipe=W4A16BlockScaling(quantization_scope=scope)):
        out = block(inp) if group_list is None else block(inp, group_list)
    out.sum().backward()
    torch.npu.synchronize()

    assert out.shape == inp.shape
    assert out.dtype == inp.dtype
    assert inp.grad is not None
    assert block.weight.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(inp.grad).all()
    assert torch.isfinite(block.weight.grad).all()


class TestW4A16CommOverlap(DistributedTest):
    """Real HCCL tests; DistributedTest launches two NPU workers from pytest."""

    world_size = 2
    reuse_dist_env = False

    @pytest.mark.parametrize("shape", [(64, 256), (16, 4, 256)])
    @pytest.mark.parametrize("weight_kind", ["dummy", "fake_mxfp4"])
    def test_allgather_matmul(self, shape, weight_kind):
        group = dist.group.WORLD
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)
        # Different data on every rank exposes missing communication/rank-order bugs.
        generator = torch.Generator().manual_seed(1234 + rank)
        x = torch.randn(shape, generator=generator).to(device="npu", dtype=torch.bfloat16)
        weight = (torch.randn(256, 256, generator=generator) / 16).to(device="npu", dtype=x.dtype)
        bias = torch.linspace(-0.25, 0.25, 256).to(device="npu", dtype=x.dtype)
        weight_quantizer = FakeMXFP4Quantizer() if weight_kind == "fake_mxfp4" else DummyQuantizer()
        wrapped_weight = weight_quantizer(weight)

        local_x = x.reshape(-1, 256).contiguous()
        shards = [torch.empty_like(local_x) for _ in range(world)]
        dist.all_gather(shards, local_x, group=group)
        gathered_reference = torch.cat(shards, dim=0)
        # Compare against the actual fake-quantized values, not the original weight.
        expected = gathered_reference @ wrapped_weight.dequantize().t() + bias
        expected = expected.reshape(shape[0] * world, *shape[1:-1], 256)

        output, gathered = CommOverlapOps.allgather_matmul(
            DummyQuantizer()(x),
            wrapped_weight,
            bias,
            world,
            group,
            TensorUsage.LHS,
            TensorUsage.RHS_TRANS,
            x.dtype,
        )
        torch.npu.synchronize()
        assert output.shape == expected.shape
        assert output.dtype == x.dtype
        torch.testing.assert_close(gathered, gathered_reference, rtol=0, atol=0)
        # Fused bias/accumulation may round differently from separate BF16 ops.
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("shape", [(128, 256), (32, 4, 256)])
    @pytest.mark.parametrize("weight_kind", ["dummy", "fake_mxfp4"])
    def test_matmul_reduce_scatter(self, shape, weight_kind):
        group = dist.group.WORLD
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)
        generator = torch.Generator().manual_seed(5678 + rank)
        x = torch.randn(shape, generator=generator).to(device="npu", dtype=torch.bfloat16)
        weight = (torch.randn(256, 256, generator=generator) / 16).to(device="npu", dtype=x.dtype)
        weight_quantizer = FakeMXFP4Quantizer() if weight_kind == "fake_mxfp4" else DummyQuantizer()
        wrapped_weight = weight_quantizer(weight)

        # Every rank contributes a different partial product; SUM then scatter on M.
        partial = (x.reshape(-1, 256) @ wrapped_weight.dequantize()).contiguous()
        expected = torch.empty((partial.shape[0] // world, 256), device=x.device, dtype=x.dtype)
        dist.reduce_scatter_tensor(expected, partial, op=dist.ReduceOp.SUM, group=group)
        expected = expected.reshape(shape[0] // world, *shape[1:-1], 256)

        output = CommOverlapOps.matmul_reduce_scatter(
            DummyQuantizer()(x),
            wrapped_weight,
            None,
            world,
            group,
            TensorUsage.LHS,
            TensorUsage.RHS,
            x.dtype,
        )
        torch.npu.synchronize()
        assert output.shape == expected.shape
        assert output.dtype == x.dtype
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
