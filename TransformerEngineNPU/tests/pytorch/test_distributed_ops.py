import pytest
import torch
import torch.distributed as dist

from transformer_engine.pytorch import ops as te_ops
from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.distributed import (
    allreduce,
    gather_along_dim,
    gather_along_first_dim,
    reduce_scatter_along_dim,
    symmetric_all_reduce,
)
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from transformer_engine.pytorch.quantization import (
    is_fp8_available,
    is_mxfp8_available,
)
from transformer_engine.pytorch.tensor import (
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
)

from transformer_engine.pytorch.utils import check_npu_version
from distributed_testing import DistributedTest


def _world_group():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    if dist.get_world_size() < 2:
        raise RuntimeError("distributed op tests require WORLD_SIZE > 1")
    return dist.group.WORLD


def _ranked_tensor(shape, *, dtype=torch.float32, requires_grad=False):
    group = _world_group()
    rank = dist.get_rank(group)
    data = torch.arange(
        torch.tensor(shape).prod().item(),
        dtype=torch.float32,
        device="cpu",
    ).reshape(shape)
    data = data + rank * 1000
    return data.to(device="npu", dtype=dtype).requires_grad_(requires_grad)


def _assert_raw_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert raw storage equality without invoking unsupported NPU float8 isclose."""
    assert actual.dtype == expected.dtype
    if actual.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        actual = actual.detach().contiguous().view(torch.uint8).cpu()
        expected = expected.detach().contiguous().view(torch.uint8).cpu()
    _assert_close_exact(actual, expected)


def _assert_close_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _full_and_local_input(
    full_shape,
    *,
    shard_dim: int = 0,
    dtype: torch.dtype = torch.bfloat16,
):
    group = _world_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    full_cpu = (
        torch.arange(
            torch.tensor(full_shape).prod().item(),
            dtype=torch.float32,
            device="cpu",
        )
        .reshape(full_shape)
        .div(1000)
    )
    local_input = torch.chunk(full_cpu, world_size, dim=shard_dim)[rank].to(
        device="npu",
        dtype=dtype,
    )
    full_input = full_cpu.to(device="npu", dtype=dtype)
    return full_cpu, full_input, local_input


def _assert_float8_tensors_equal(actual, expected) -> None:
    _assert_raw_tensor_equal(actual._data, expected._data)
    _assert_close_exact(actual._scale_inv, expected._scale_inv)


def _assert_mxfp8_tensors_equal(actual, expected) -> None:
    _assert_raw_tensor_equal(actual._rowwise_data, expected._rowwise_data)
    _assert_close_exact(actual._rowwise_scale_inv, expected._rowwise_scale_inv)
    _assert_raw_tensor_equal(actual._columnwise_data, expected._columnwise_data)
    _assert_close_exact(actual._columnwise_scale_inv, expected._columnwise_scale_inv)


class TestDistributedOps(DistributedTest):
    """Distributed ops tests — framework handles env, set_device, and dist init."""

    world_size = 2
    reuse_dist_env = False

    def test_gather_along_dim_preserves_high_precision_path(self) -> None:
        """Ported from NVIDIA distributed all-gather coverage for the dense path."""
        group = _world_group()
        world_size = dist.get_world_size(group)
        x = _ranked_tensor((4, 8), dtype=torch.bfloat16)

        y, handle = gather_along_dim(x, group=group, dim=0)
        if handle is not None:
            handle.wait()

        expected = torch.cat(
            [
                (torch.arange(32, dtype=torch.float32, device="cpu").reshape(4, 8) + rank * 1000)
                for rank in range(world_size)
            ],
            dim=0,
        ).to(device="npu", dtype=torch.bfloat16)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)

    def test_basic_linear_column_sequence_parallel_forward_backward(self) -> None:
        """BasicLinear column TP + SP should match dense linear math."""
        group = _world_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        seq_len = 8
        in_features = 8
        out_features = 12
        assert seq_len % world_size == 0
        assert out_features % world_size == 0

        full_input = (
            torch.arange(seq_len * in_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, in_features)
            .to(device="npu")
        )
        full_weight = (
            torch.arange(out_features * in_features, dtype=torch.float32, device="cpu")
            .reshape(out_features, in_features)
            .to(device="npu")
        )
        full_grad_output = (
            torch.arange(seq_len * out_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, out_features)
            .to(device="npu")
        )

        local_input = torch.chunk(full_input, world_size, dim=0)[rank].detach().clone()
        local_input.requires_grad_(True)
        local_out_start = rank * (out_features // world_size)
        local_out_end = local_out_start + (out_features // world_size)

        layer = BasicLinear(
            in_features,
            out_features,
            device="npu",
            dtype=torch.float32,
            tensor_parallel_mode="column",
            tensor_parallel_group=group,
            sequence_parallel=True,
        )
        with torch.no_grad():
            layer.weight.copy_(full_weight[local_out_start:local_out_end])

        output = layer(local_input)
        expected_output = torch.nn.functional.linear(
            full_input,
            full_weight[local_out_start:local_out_end],
        )
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

        grad_output = full_grad_output[:, local_out_start:local_out_end]
        output.backward(grad_output)

        expected_full_grad_input = torch.matmul(full_grad_output, full_weight)
        expected_grad_input = torch.chunk(expected_full_grad_input, world_size, dim=0)[rank]
        expected_grad_weight = torch.matmul(grad_output.t(), full_input)
        torch.testing.assert_close(local_input.grad, expected_grad_input, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight.grad, expected_grad_weight, rtol=0, atol=0)

    def test_basic_linear_row_sequence_parallel_forward_backward(self) -> None:
        """BasicLinear row TP + SP should match dense linear math."""
        group = _world_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        seq_len = 8
        in_features = 8
        out_features = 12
        assert seq_len % world_size == 0
        assert in_features % world_size == 0

        full_input = (
            torch.arange(seq_len * in_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, in_features)
            .to(device="npu")
        )
        full_weight = (
            torch.arange(out_features * in_features, dtype=torch.float32, device="cpu")
            .reshape(out_features, in_features)
            .to(device="npu")
        )
        full_grad_output = (
            torch.arange(seq_len * out_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, out_features)
            .to(device="npu")
        )

        local_in_start = rank * (in_features // world_size)
        local_in_end = local_in_start + (in_features // world_size)
        local_input = full_input[:, local_in_start:local_in_end].detach().clone()
        local_input.requires_grad_(True)

        layer = BasicLinear(
            in_features,
            out_features,
            device="npu",
            dtype=torch.float32,
            tensor_parallel_mode="row",
            tensor_parallel_group=group,
            sequence_parallel=True,
        )
        with torch.no_grad():
            layer.weight.copy_(full_weight[:, local_in_start:local_in_end])

        output = layer(local_input)
        expected_full_output = torch.nn.functional.linear(full_input, full_weight)
        expected_output = torch.chunk(expected_full_output, world_size, dim=0)[rank]
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

        grad_output = torch.chunk(full_grad_output, world_size, dim=0)[rank]
        output.backward(grad_output)

        expected_grad_input = torch.matmul(
            full_grad_output,
            full_weight[:, local_in_start:local_in_end],
        )
        expected_grad_weight = torch.matmul(full_grad_output.t(), local_input.detach())
        torch.testing.assert_close(local_input.grad, expected_grad_input, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight.grad, expected_grad_weight, rtol=0, atol=0)

    def test_basic_linear_column_tensor_parallel_forward_backward(self) -> None:
        """BasicLinear column TP without SP should match dense linear math."""
        group = _world_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        seq_len = 8
        in_features = 8
        out_features = 12
        assert out_features % world_size == 0

        full_input = (
            torch.arange(seq_len * in_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, in_features)
            .to(device="npu")
        )
        full_weight = (
            torch.arange(out_features * in_features, dtype=torch.float32, device="cpu")
            .reshape(out_features, in_features)
            .to(device="npu")
        )
        full_grad_output = (
            torch.arange(seq_len * out_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, out_features)
            .to(device="npu")
        )

        local_out_start = rank * (out_features // world_size)
        local_out_end = local_out_start + (out_features // world_size)
        local_input = full_input.detach().clone()
        local_input.requires_grad_(True)
        layer = BasicLinear(
            in_features,
            out_features,
            device="npu",
            dtype=torch.float32,
            tensor_parallel_mode="column",
            tensor_parallel_group=group,
        )
        with torch.no_grad():
            layer.weight.copy_(full_weight[local_out_start:local_out_end])

        output = layer(local_input)
        expected_output = torch.nn.functional.linear(
            full_input,
            full_weight[local_out_start:local_out_end],
        )
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

        grad_output = full_grad_output[:, local_out_start:local_out_end]
        output.backward(grad_output)

        expected_grad_input = torch.matmul(full_grad_output, full_weight)
        expected_grad_weight = torch.matmul(grad_output.t(), full_input)
        torch.testing.assert_close(local_input.grad, expected_grad_input, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight.grad, expected_grad_weight, rtol=0, atol=0)

    def test_basic_linear_row_tensor_parallel_forward_backward(self) -> None:
        """BasicLinear row TP without SP should match dense linear math."""
        group = _world_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        seq_len = 8
        in_features = 8
        out_features = 12
        assert in_features % world_size == 0

        full_input = (
            torch.arange(seq_len * in_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, in_features)
            .to(device="npu")
        )
        full_weight = (
            torch.arange(out_features * in_features, dtype=torch.float32, device="cpu")
            .reshape(out_features, in_features)
            .to(device="npu")
        )
        full_grad_output = (
            torch.arange(seq_len * out_features, dtype=torch.float32, device="cpu")
            .reshape(seq_len, out_features)
            .to(device="npu")
        )

        local_in_start = rank * (in_features // world_size)
        local_in_end = local_in_start + (in_features // world_size)
        local_input = full_input[:, local_in_start:local_in_end].detach().clone()
        local_input.requires_grad_(True)
        layer = BasicLinear(
            in_features,
            out_features,
            device="npu",
            dtype=torch.float32,
            tensor_parallel_mode="row",
            tensor_parallel_group=group,
        )
        with torch.no_grad():
            layer.weight.copy_(full_weight[:, local_in_start:local_in_end])

        output = layer(local_input)
        expected_output = torch.nn.functional.linear(full_input, full_weight)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

        output.backward(full_grad_output)

        expected_grad_input = torch.matmul(
            full_grad_output,
            full_weight[:, local_in_start:local_in_end],
        )
        expected_grad_weight = torch.matmul(full_grad_output.t(), local_input.detach())
        torch.testing.assert_close(local_input.grad, expected_grad_input, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight.grad, expected_grad_weight, rtol=0, atol=0)

    def test_basic_communication_ops(self) -> None:
        """Adapted from NVIDIA TestFusibleOps communication-op tests."""
        group = _world_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        # AllGather forward concatenates rank-local tensors and backward reduce-scatters.
        x = _ranked_tensor((4, 8), dtype=torch.float32, requires_grad=True)
        y = te_ops.AllGather(process_group=group)(x)
        expected_y = torch.cat(
            [
                (
                    torch.arange(32, dtype=torch.float32, device="cpu").reshape(4, 8)
                    + src_rank * 1000
                )
                for src_rank in range(world_size)
            ],
            dim=0,
        ).to(device="npu")
        torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, world_size), rtol=0, atol=0)

        # ReduceScatter forward sums same-index chunks from every rank.
        rs_input = _ranked_tensor(
            (world_size * 4, 8),
            dtype=torch.float32,
            requires_grad=True,
        )
        rs_output = te_ops.ReduceScatter(process_group=group)(rs_input)
        per_rank_inputs = [
            torch.arange(
                world_size * 32,
                dtype=torch.float32,
                device="cpu",
            ).reshape(world_size * 4, 8)
            + src_rank * 1000
            for src_rank in range(world_size)
        ]
        expected_rs = sum(
            torch.chunk(per_rank_input, world_size, dim=0)[rank]
            for per_rank_input in per_rank_inputs
        ).to(device="npu")
        torch.testing.assert_close(rs_output, expected_rs, rtol=0, atol=0)
        rs_output.sum().backward()
        torch.testing.assert_close(rs_input.grad, torch.ones_like(rs_input), rtol=0, atol=0)

        # AllReduce forward sums tensors and backward is identity, matching NVIDIA op semantics.
        ar_input = _ranked_tensor((4, 8), dtype=torch.float32, requires_grad=True)
        ar_output = te_ops.AllReduce(process_group=group)(ar_input)
        expected_ar = sum(
            (torch.arange(32, dtype=torch.float32, device="cpu").reshape(4, 8) + src_rank * 1000)
            for src_rank in range(world_size)
        ).to(device="npu")
        torch.testing.assert_close(ar_output, expected_ar, rtol=0, atol=0)
        ar_output.sum().backward()
        torch.testing.assert_close(ar_input.grad, torch.ones_like(ar_input), rtol=0, atol=0)

    def test_allreduce_wrappers(self) -> None:
        """Directly cover distributed helpers used by communication ops."""
        group = _world_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        x = torch.full((4, 8), rank + 1, dtype=torch.float32, device="npu")
        y, handle = allreduce(x.clone(), group=group, async_op=True)
        if handle is not None:
            handle.wait()
        expected = torch.full_like(x, sum(range(1, world_size + 1)))
        torch.testing.assert_close(y, expected, rtol=0, atol=0)

        y, handle = symmetric_all_reduce(x.clone(), group=group)
        if handle is not None:
            handle.wait()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)

        rs_input = _ranked_tensor((world_size * 4, 8), dtype=torch.float32)
        rs_output, handle = reduce_scatter_along_dim(
            rs_input,
            group=group,
            async_op=True,
        )
        if handle is not None:
            handle.wait()
        per_rank_inputs = [
            torch.arange(
                world_size * 32,
                dtype=torch.float32,
                device="cpu",
            ).reshape(world_size * 4, 8)
            + src_rank * 1000
            for src_rank in range(world_size)
        ]
        expected_rs = sum(
            torch.chunk(per_rank_input, world_size, dim=0)[rank]
            for per_rank_input in per_rank_inputs
        ).to(device="npu")
        torch.testing.assert_close(rs_output, expected_rs, rtol=0, atol=0)

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="quantized distributed tests require Atlas A5"
    )
    def test_float8_gather_along_first_dim_matches_full_quantize(self) -> None:
        """NPU Float8 all-gather falls back when scale_inv is not globally reduced."""
        group = _world_group()
        fp8_available, reason = is_fp8_available(return_reason=True)
        if not fp8_available:
            if dist.get_rank(group) == 0:
                print(f"Skipping Float8 distributed all-gather test: {reason}")
            return

        world_size = dist.get_world_size(group)
        local_rows = 32
        hidden_size = 32
        full_rows = world_size * local_rows

        _, full_input, local_input = _full_and_local_input((full_rows, hidden_size))

        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            device=local_input.device,
        )
        with pytest.warns(UserWarning, match="Float8 all-gather on NPU falls back"):
            gathered, handle = gather_along_first_dim(
                local_input,
                group,
                async_op=False,
                quantizer=quantizer,
            )
        if handle is not None:
            handle.wait()
        reference = quantizer(full_input)

        _assert_float8_tensors_equal(gathered, reference)

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="quantized distributed tests require Atlas A5"
    )
    def test_float8_quantized_input_gather_matches_requantized_dequantized_shards(self) -> None:
        """Pre-quantized Float8 input must fallback from dequantized shard values."""
        group = _world_group()
        fp8_available, reason = is_fp8_available(return_reason=True)
        if not fp8_available:
            if dist.get_rank(group) == 0:
                print(f"Skipping Float8 distributed all-gather test: {reason}")
            return

        world_size = dist.get_world_size(group)
        full_cpu, _, local_input = _full_and_local_input((world_size * 32, 32))
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=torch.float8_e4m3fn,
            device=local_input.device,
        )
        local_quantized = quantizer(local_input)

        with pytest.warns(UserWarning, match="Float8 all-gather on NPU falls back"):
            gathered, handle = gather_along_first_dim(
                local_quantized,
                group,
                async_op=False,
            )
        if handle is not None:
            handle.wait()

        dequantized_shards = []
        for src_rank in range(world_size):
            shard = torch.chunk(full_cpu, world_size, dim=0)[src_rank].to(
                device="npu",
                dtype=torch.bfloat16,
            )
            dequantized_shards.append(quantizer(shard).dequantize(dtype=torch.bfloat16))
        reference_input = torch.cat(dequantized_shards, dim=0)
        reference = quantizer(reference_input)

        _assert_float8_tensors_equal(gathered, reference)

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="quantized distributed tests require Atlas A5"
    )
    def test_mxfp8_gather_along_first_dim_matches_full_quantize(self) -> None:
        """Adapted from NVIDIA quantized all-gather numerics coverage."""
        group = _world_group()
        mxfp8_available, reason = is_mxfp8_available(return_reason=True)
        if not mxfp8_available:
            if dist.get_rank(group) == 0:
                print(f"Skipping MXFP8 distributed all-gather test: {reason}")
            return

        world_size = dist.get_world_size(group)
        local_rows = 32
        hidden_size = 32
        full_rows = world_size * local_rows

        _, full_input, local_input = _full_and_local_input((full_rows, hidden_size))

        quantizer = MXFP8Quantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=True,
        )
        gathered, handle = gather_along_first_dim(
            local_input,
            group,
            async_op=False,
            quantizer=quantizer,
        )
        if handle is not None:
            handle.wait()
        reference = quantizer(full_input)

        _assert_mxfp8_tensors_equal(gathered, reference)

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="quantized distributed tests require Atlas A5"
    )
    def test_mxfp8_quantized_input_gather_matches_full_quantize(self) -> None:
        """Pre-quantized MXFP8 input exercises the QuantizedTensorStorage branch."""
        group = _world_group()
        mxfp8_available, reason = is_mxfp8_available(return_reason=True)
        if not mxfp8_available:
            if dist.get_rank(group) == 0:
                print(f"Skipping MXFP8 distributed all-gather test: {reason}")
            return

        world_size = dist.get_world_size(group)
        full_cpu, _, local_input = _full_and_local_input((world_size * 32, 32))
        quantizer = MXFP8Quantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=True,
        )
        local_quantized = quantizer(local_input)

        gathered, handle = gather_along_first_dim(
            local_quantized,
            group,
            async_op=False,
        )
        if handle is not None:
            handle.wait()
        dequantized_shards = []
        for src_rank in range(world_size):
            shard = torch.chunk(full_cpu, world_size, dim=0)[src_rank].to(
                device="npu",
                dtype=torch.bfloat16,
            )
            dequantized_shards.append(quantizer(shard).dequantize(dtype=torch.bfloat16))
        reference = quantizer(torch.cat(dequantized_shards, dim=0))

        _assert_mxfp8_tensors_equal(gathered, reference)

    @pytest.mark.skipif(
        not check_npu_version(NPUVersion.A5), reason="quantized distributed tests require Atlas A5"
    )
    def test_mxfp8_non_quantizable_local_gather_falls_back(self) -> None:
        """Local M not divisible by 32 should gather densely before MXFP8 quantize."""
        group = _world_group()
        mxfp8_available, reason = is_mxfp8_available(return_reason=True)
        if not mxfp8_available:
            if dist.get_rank(group) == 0:
                print(f"Skipping MXFP8 distributed all-gather test: {reason}")
            return

        world_size = dist.get_world_size(group)
        _, full_input, local_input = _full_and_local_input((world_size * 16, 32))
        quantizer = MXFP8Quantizer(
            fp8_dtype=torch.float8_e4m3fn,
            rowwise=True,
            columnwise=True,
        )
        assert not quantizer.is_quantizable(local_input)
        assert quantizer.is_quantizable(full_input)

        with pytest.warns(UserWarning, match="Cannot quantize input tensor for MXFP8"):
            gathered, handle = gather_along_first_dim(
                local_input,
                group,
                async_op=False,
                quantizer=quantizer,
            )
        if handle is not None:
            handle.wait()
        reference = quantizer(full_input)

        _assert_mxfp8_tensors_equal(gathered, reference)
