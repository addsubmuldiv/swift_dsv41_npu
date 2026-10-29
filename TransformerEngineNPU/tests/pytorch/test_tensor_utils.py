# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Distributed tests for casting master weights to FP8 primary weights."""

import io
import signal

import pytest
import torch
import torch.distributed as dist
from torch import nn

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,
    Format,
    MXFP8BlockScaling,
    Recipe,
)
from transformer_engine.pytorch.quantization import (
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.tensor import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
    Float8Tensor,
    MXFP8Quantizer,
    MXFP8Tensor,
)
from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
    Float8BlockwiseQTensorStorage,
)
from transformer_engine.pytorch.tensor.utils import (
    cast_master_weights_to_fp8,
    post_all_gather_processing,
)
from transformer_engine.pytorch.triton import fp8_tensorwise_scaling

from distributed_testing import DistributedTest

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
    return_reason=True
)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)

_quantization_list = []
if fp8_available:
    _quantization_list.extend(["fp8", "fp8_cs"])
if fp8_block_scaling_available:
    _quantization_list.append("fp8_block")
if mxfp8_available:
    _quantization_list.append("mxfp8")


def test_megatron_exit_signal_checkpoint_safe_load() -> None:
    """Allow safe loading of Megatron checkpoints containing args.exit_signal."""
    buffer = io.BytesIO()
    torch.save({"exit_signal": signal.SIGTERM}, buffer)

    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)

    assert checkpoint["exit_signal"] is signal.SIGTERM


@pytest.mark.skipif(
    not fp8_block_scaling_available,
    reason=reason_for_no_fp8_block_scaling,
)
def test_blockwise_post_all_gather_matches_quantizer_columnwise() -> None:
    """Recreated storage must match the normal NPU quantizer's columnwise representation."""
    torch.manual_seed(1234)
    dense = torch.randn(256, 512, dtype=torch.bfloat16, device="npu")
    quantizer = Float8BlockQuantizer(
        fp8_dtype=torch.float8_e4m3fn,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=2,
    )
    expected = quantizer(dense)
    actual = Float8BlockwiseQTensorStorage(
        rowwise_data=expected._rowwise_data.clone(),
        rowwise_scale_inv=expected._rowwise_scale_inv.clone(),
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=expected._fp8_dtype,
        quantizer=quantizer,
        is_2D_scaled=True,
    )

    post_all_gather_processing(actual)

    assert actual._columnwise_data is not None
    assert actual._columnwise_scale_inv is not None
    assert expected._columnwise_data is not None
    assert expected._columnwise_scale_inv is not None
    assert actual._columnwise_data.shape == expected._columnwise_data.shape
    assert actual._columnwise_scale_inv.shape == expected._columnwise_scale_inv.shape
    torch.testing.assert_close(
        actual._columnwise_data.view(torch.uint8),
        expected._columnwise_data.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual._columnwise_scale_inv,
        expected._columnwise_scale_inv,
        rtol=0,
        atol=0,
    )


def _get_quantization_recipe(quantization) -> Recipe:
    """Quantization recipe setup"""
    fp8_format = Format.HYBRID
    if quantization == "fp8":
        return DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")
    if quantization == "fp8_cs":
        return Float8CurrentScaling(fp8_format=fp8_format)
    if quantization == "fp8_block":
        return Float8BlockScaling(fp8_format=fp8_format)
    if quantization == "mxfp8":
        return MXFP8BlockScaling()
    raise ValueError(f"Unsupported quantization: {quantization}")


def _get_raw_data(quantized_tensor, colwise=False):
    """Get the underlying data of a quantized tensor, used in zero-1 optimizer"""
    if isinstance(quantized_tensor, Float8Tensor):
        assert not colwise, "Float8Tensor does not support get colwise data"
        raw_data = quantized_tensor._data
    elif isinstance(quantized_tensor, Float8BlockwiseQTensor):
        assert not colwise, "Float8BlockwiseQTensor does not support get colwise data"
        raw_data = quantized_tensor._rowwise_data
    elif isinstance(quantized_tensor, MXFP8Tensor):
        raw_data = quantized_tensor._columnwise_data if colwise else quantized_tensor._rowwise_data
    else:
        raise ValueError(f"Unsupported quantized tensor type: {type(quantized_tensor)}")
    assert raw_data is not None, f"{type(quantized_tensor)} raw data is None"
    return raw_data.view(torch.uint8) if raw_data.dtype != torch.uint8 else raw_data


def _assert_quantized_weights_equal(actual_weights, expected_weights) -> None:
    assert len(actual_weights) == len(expected_weights)
    for actual, expected in zip(actual_weights, expected_weights):
        assert actual.__class__ is expected.__class__
        assert actual.origin_shape == expected.origin_shape
        actual_metadata = actual.get_metadata()
        for name, expected_value in expected.get_metadata().items():
            if name == "quantizer":
                continue
            actual_value = actual_metadata[name]
            if expected_value is None:
                assert actual_value is None
            elif isinstance(expected_value, torch.Tensor):
                assert actual_value is not None
                assert actual_value.dtype == expected_value.dtype
                assert actual_value.shape == expected_value.shape
                assert torch.equal(
                    actual_value.detach().contiguous().view(torch.uint8).cpu(),
                    expected_value.detach().contiguous().view(torch.uint8).cpu(),
                )
            else:
                assert actual_value == expected_value


def _assert_tensorwise_partial_cast_matches_npu(master_weight, scale, fp8_dtype):
    import torch_npu

    expected = torch_npu.npu_quantize(
        master_weight.reshape(1, -1),
        scale.reciprocal(),
        zero_points=None,
        dtype=fp8_dtype,
        axis=-1,
    )
    output_storage = torch.full(
        (master_weight.numel() + 2,),
        0xA5,
        dtype=torch.uint8,
        device="npu",
    )
    actual = output_storage[1:-1].view(fp8_dtype)

    fp8_tensorwise_scaling.fp8_tensorwise_partial_cast(
        master_weight,
        actual,
        scale,
        fp8_dtype,
    )

    torch.testing.assert_close(
        actual.view(torch.uint8),
        expected.view(torch.uint8).reshape(-1),
        atol=0,
        rtol=0,
    )
    assert output_storage[0].item() == 0xA5
    assert output_storage[-1].item() == 0xA5


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_tensorwise_partial_cast(input_dtype, fp8_dtype):
    """Direct Triton writeback must match explicit-scale NPU quantization byte-for-byte."""
    torch.manual_seed(1234)

    scale = torch.tensor([3.1415927], dtype=torch.float32, device="npu")
    master_weight = torch.randn(65537, dtype=input_dtype, device="npu")
    fp8_max = torch.finfo(fp8_dtype).max
    master_weight[0] = fp8_max
    master_weight[-1] = -fp8_max

    _assert_tensorwise_partial_cast_matches_npu(master_weight, scale, fp8_dtype)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_tensorwise_partial_cast_rounding_boundaries(fp8_dtype):
    """Cover values where multiply and divide-by-reciprocal can round differently."""
    scale_value = 3.1415927
    fp8_values = torch.arange(256, dtype=torch.uint8).view(fp8_dtype).float()
    positive_finite = (
        torch.unique(fp8_values[torch.isfinite(fp8_values) & (fp8_values >= 0)]).sort().values
    )
    midpoints = (positive_finite[:-1] + positive_finite[1:]) / 2
    unscaled_midpoints = midpoints / scale_value
    lower = torch.nextafter(unscaled_midpoints, torch.full_like(unscaled_midpoints, -torch.inf))
    upper = torch.nextafter(unscaled_midpoints, torch.full_like(unscaled_midpoints, torch.inf))
    master_weight = torch.cat(
        (lower, unscaled_midpoints, upper, -lower, -unscaled_midpoints, -upper)
    ).npu()
    scale = torch.tensor([scale_value], dtype=torch.float32, device="npu")

    _assert_tensorwise_partial_cast_matches_npu(master_weight, scale, fp8_dtype)


def _run_blockwise_partial_cast_case(height, width, split, fp8_dtype, group):
    dense = torch.randn(height, width, dtype=torch.bfloat16, device="npu")
    quantizer = Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=False,
        block_scaling_dim=2,
    )
    actual = quantizer.make_empty(dense.shape, dtype=dense.dtype, device=dense.device)
    expected = quantizer.quantize(dense)

    rank = dist.get_rank(group)
    start = 0 if rank == 0 else split
    end = split if rank == 0 else dense.numel()
    master_weight = dense.reshape(-1)[start:end].float()
    cast_master_weights_to_fp8(
        [actual],
        [master_weight],
        [start],
        group,
    )

    torch.testing.assert_close(
        actual._rowwise_data.view(torch.uint8).reshape(-1)[start:end],
        expected._rowwise_data.view(torch.uint8).reshape(-1)[start:end],
        atol=0,
        rtol=0,
        msg=lambda msg: f"blockwise FP8 payload mismatch\n{msg}",
    )
    # The upstream shard path computes reciprocal(max_fp8 / amax), while the NPU operator
    # computes amax / max_fp8 directly. Allow the resulting one-FP32-ULP rounding difference.
    torch.testing.assert_close(
        actual._rowwise_scale_inv,
        expected._rowwise_scale_inv,
        atol=0,
        rtol=torch.finfo(torch.float32).eps,
        msg=lambda msg: f"blockwise scale_inv mismatch\n{msg}",
    )


def _run_mxfp8_partial_cast_case(height, width, split, group):
    dense = torch.randn(height, width, dtype=torch.bfloat16, device="npu")
    quantizer = MXFP8Quantizer(
        fp8_dtype=torch.float8_e4m3fn,
        rowwise=True,
        columnwise=True,
    )
    actual = quantizer.make_empty(dense.shape, dtype=dense.dtype, device=dense.device)
    expected = quantizer.quantize(dense)

    rank = dist.get_rank(group)
    start = 0 if rank == 0 else split
    end = split if rank == 0 else dense.numel()
    master_weight = dense.reshape(-1)[start:end].float()
    cast_master_weights_to_fp8(
        [actual],
        [master_weight],
        [start],
        group,
    )

    torch.testing.assert_close(
        actual._rowwise_scale_inv,
        expected._rowwise_scale_inv,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual._columnwise_scale_inv,
        expected._columnwise_scale_inv,
        atol=0,
        rtol=0,
    )
    for actual_data, expected_data in (
        (actual._rowwise_data, expected._rowwise_data),
        (actual._columnwise_data, expected._columnwise_data),
    ):
        torch.testing.assert_close(
            actual_data.view(torch.uint8).reshape(-1)[start:end],
            expected_data.view(torch.uint8).reshape(-1)[start:end],
            atol=0,
            rtol=0,
        )


class MiniOptimizer:
    """A mini replicated optimizer implementation, just used for this test"""

    def __init__(self, weights, lr, dp_group):
        self.world_size = dist.get_world_size(dp_group)

        self.weights = weights
        self.lr = lr
        self.dp_group = dp_group

        master_weights = []
        for weight in self.weights:
            master_weights.append(weight.detach().float())
        self.master_weights = master_weights

    def step(self):
        for weight, master_weight in zip(self.weights, self.master_weights):
            main_grad = weight.main_grad

            # Don't use all-reduce directly to explicitly control the reduce order.
            buffers = [torch.empty_like(main_grad) for _ in range(self.world_size)]
            dist.all_gather(buffers, main_grad, group=self.dp_group)
            for i in range(1, self.world_size):
                buffers[0] += buffers[i]
            main_grad.copy_(buffers[0])
            main_grad /= self.world_size

            master_weight -= main_grad * self.lr
            weight.data.copy_(master_weight)


class MiniZero_1:
    """A mini zero-1 optimizer implementation, just used for this test"""

    def __init__(self, weights, lr, dp_group, manual_post_all_gather_processing=False):
        self.rank = dist.get_rank(dp_group)
        self.world_size = dist.get_world_size(dp_group)

        self.weights = weights
        self.lr = lr
        self.dp_group = dp_group
        self.manual_post_all_gather_processing = manual_post_all_gather_processing

        # [self.offsets[i], self.offsets[i+1]) is the range of weights[i] in the global buffer
        self.offsets = [0]
        for weight in self.weights:
            self.offsets.append(self.offsets[-1] + weight.numel())
        # Padding to avoid global buffer cannot be divided by world size, so the offsets[-1] may
        # not be the end range of the last weight.
        if self.offsets[-1] % self.world_size != 0:
            self.offsets[-1] += self.world_size - self.offsets[-1] % self.world_size

        self.master_weights = []
        # The start offset of the master weight in the weight
        self.start_offsets = []
        # The overlapping area of the weight and this rank's local buffer
        self.overlapping_areas = []

        # The start and end of this rank's local buffer in the global buffer
        rank_start = self.offsets[-1] // self.world_size * self.rank
        rank_end = rank_start + self.offsets[-1] // self.world_size

        for weight, offset in zip(self.weights, self.offsets[:-1]):
            if offset >= rank_end or (offset + weight.numel()) <= rank_start:
                # This weight is not in this rank's local buffer
                master_weight = None
                start_offset = None
                overlapping_area = None
            else:
                overlapping_start = max(rank_start, offset)
                overlapping_end = min(rank_end, offset + weight.numel())
                length = overlapping_end - overlapping_start
                start_offset = overlapping_start - offset
                if isinstance(weight, QuantizedTensor):
                    # If weight is a FP8 tensor, we need to use the original high precision version
                    # to initialize the master weight.
                    high_precision_init_val = weight.get_high_precision_init_val().view(-1)
                    master_weight = high_precision_init_val.to(weight.device).float()[
                        start_offset : start_offset + length
                    ]
                else:
                    master_weight = (
                        weight.detach().view(-1).float()[start_offset : start_offset + length]
                    )
                overlapping_area = (overlapping_start, overlapping_end)
            self.master_weights.append(master_weight)
            self.start_offsets.append(start_offset)
            self.overlapping_areas.append(overlapping_area)

        # Create global buffer for grads reduce-scatter
        self.grad_buffer = torch.empty(
            [self.offsets[-1]], dtype=torch.float32, device=weights[0].device
        )
        self.grad_buffer_slice = self.grad_buffer[rank_start:rank_end]

        # Create global buffer for weights all-gather
        if isinstance(self.weights[0], QuantizedTensor):
            weight_buffer_dtype = torch.uint8
        else:
            weight_buffer_dtype = weights[0].dtype
        self.weight_buffer = torch.empty(
            [self.offsets[-1]], dtype=weight_buffer_dtype, device=weights[0].device
        )
        self.weight_buffer_slice = self.weight_buffer[rank_start:rank_end]

    def step(self):
        # -----------------------------------------------------------------------------------------
        # Step 1: Copy grads to the grad buffer
        # -----------------------------------------------------------------------------------------
        for weight, offset in zip(self.weights, self.offsets[:-1]):
            start = offset
            end = offset + weight.numel()
            self.grad_buffer[start:end].copy_(weight.main_grad.view(-1))

        # -----------------------------------------------------------------------------------------
        # Step 2: Grads reduce-scatter
        # -----------------------------------------------------------------------------------------
        # Don't use reduce_scatter directly to explicitly control the reduce order.
        buffers = [torch.empty_like(self.grad_buffer) for _ in range(self.world_size)]
        dist.all_gather(buffers, self.grad_buffer, group=self.dp_group)
        for i in range(1, self.world_size):
            buffers[0] += buffers[i]
        rank_start = self.offsets[-1] // self.world_size * self.rank
        rank_end = rank_start + self.offsets[-1] // self.world_size
        self.grad_buffer_slice.copy_(buffers[0][rank_start:rank_end])
        self.grad_buffer_slice /= self.world_size

        # -----------------------------------------------------------------------------------------
        # Step 3: Update master weights
        # -----------------------------------------------------------------------------------------
        for master_weight, overlapping_area in zip(self.master_weights, self.overlapping_areas):
            if master_weight is None:
                # This weight's master weight is in other rank.
                continue
            grad = self.grad_buffer[overlapping_area[0] : overlapping_area[1]]
            master_weight -= grad * self.lr

        # -----------------------------------------------------------------------------------------
        # Step 4: Cast master weights to BF16 or FP8, depending on the type of the weight
        # -----------------------------------------------------------------------------------------
        if isinstance(self.weights[0], QuantizedTensor):
            for weight in self.weights:
                assert isinstance(weight, QuantizedTensor)
            cast_master_weights_to_fp8(
                self.weights,
                self.master_weights,
                self.start_offsets,
                self.dp_group,
                manual_post_all_gather_processing=self.manual_post_all_gather_processing,
            )
            if isinstance(self.weights[0], MXFP8Tensor):
                for weight in self.weights:
                    assert weight._rowwise_data is not None
                    assert weight._rowwise_scale_inv is not None
                    assert weight._columnwise_data is not None
                    assert weight._columnwise_scale_inv is not None
        else:
            for weight, master_weight, start_offset in zip(
                self.weights, self.master_weights, self.start_offsets
            ):
                if master_weight is None:
                    continue
                start = start_offset
                end = start_offset + master_weight.numel()
                weight.data.view(-1)[start:end].copy_(master_weight)

        # -----------------------------------------------------------------------------------------
        # Step 5: Copy the updated weights (not all weights) to the weight buffer
        # -----------------------------------------------------------------------------------------
        colwise_list = [False]
        if isinstance(self.weights[0], MXFP8Tensor):
            colwise_list.append(True)

        for colwise in colwise_list:
            for i in range(len(self.weights)):
                master_weight = self.master_weights[i]
                if master_weight is None:
                    continue
                start_offset = self.start_offsets[i]
                if isinstance(self.weights[i], QuantizedTensor):
                    weight = _get_raw_data(self.weights[i], colwise)
                else:
                    weight = self.weights[i]
                weight_slice = weight.view(-1)[start_offset : start_offset + master_weight.numel()]
                overlapping_start, overlapping_end = self.overlapping_areas[i]
                self.weight_buffer[overlapping_start:overlapping_end].copy_(weight_slice)

            # -------------------------------------------------------------------------------------
            # Step 6: Weight all-gather (FP8 or BF16)
            # -------------------------------------------------------------------------------------
            dist.all_gather_into_tensor(
                self.weight_buffer, self.weight_buffer_slice, group=self.dp_group
            )

            # -------------------------------------------------------------------------------------
            # Step 7: Copy the gathered weights from weight buffer to the actual weights
            # -------------------------------------------------------------------------------------
            for weight, offset in zip(self.weights, self.offsets[:-1]):
                start = offset
                end = offset + weight.numel()
                if isinstance(weight, QuantizedTensor):
                    weight = _get_raw_data(weight, colwise)
                weight.view(-1).data.copy_(self.weight_buffer[start:end])

        if self.manual_post_all_gather_processing:
            post_all_gather_processing(
                [weight for weight in self.weights if isinstance(weight, QuantizedTensor)]
            )


def _make_models(quantization, linear_kwargs):
    """Create an FP8-primary-weight model and a BF16 twin with identical init weights."""
    with te.quantized_model_init(
        enabled=quantization is not None,
        recipe=_get_quantization_recipe(quantization),
        preserve_high_precision_init_val=True,
    ):
        model_fp8 = nn.Sequential(
            te.Linear(128, 256 + 32, **linear_kwargs),
            te.Linear(256 + 32, 256 * 3, **linear_kwargs),
            te.Linear(256 * 3, 128, **linear_kwargs),
        )

    model = nn.Sequential(
        te.Linear(128, 256 + 32, **linear_kwargs),
        te.Linear(256 + 32, 256 * 3, **linear_kwargs),
        te.Linear(256 * 3, 128, **linear_kwargs),
    )

    # Make sure the BF16 model and FP8 model have the same initial weights
    for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
        high_precision_init_val = w_fp8.get_high_precision_init_val()
        w.data.copy_(high_precision_init_val)

    return model_fp8, model


class TestCastMasterWeightsToFP8(DistributedTest):
    """Distributed tests for cast_master_weights_to_fp8 against a BF16 reference model."""

    world_size = 2

    @pytest.mark.skipif(
        not fp8_block_scaling_available,
        reason=reason_for_no_fp8_block_scaling,
    )
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_blockwise_partial_cast(self, fp8_dtype):
        """Triton blockwise partial cast must match full NPU quantization byte-for-byte."""
        torch.manual_seed(1234)
        dp_group = dist.group.WORLD

        _run_blockwise_partial_cast_case(64, 64, 1, fp8_dtype, dp_group)
        _run_blockwise_partial_cast_case(257, 385, 12345, fp8_dtype, dp_group)
        _run_blockwise_partial_cast_case(256, 256, 32768, fp8_dtype, dp_group)

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_mxfp8_partial_cast(self):
        """MXFP8 partial cast must match full dynamic quantization byte-for-byte."""
        torch.manual_seed(1234)
        dp_group = dist.group.WORLD

        _run_mxfp8_partial_cast_case(64, 64, 1, dp_group)
        _run_mxfp8_partial_cast_case(96, 288, 12345, dp_group)
        _run_mxfp8_partial_cast_case(768, 256, 131072, dp_group)

    def test_mini_optimizer(self):
        """Make sure the implementation of MiniZero_1 is correct"""
        dp_group = dist.group.WORLD
        rank = dist.get_rank(dp_group)
        world_size = dist.get_world_size(dp_group)

        torch.manual_seed(12345)

        weights = [
            torch.randn(256 * 256, dtype=torch.bfloat16, device="npu"),
            torch.randn(256 * 256 * 3, dtype=torch.bfloat16, device="npu"),
            torch.randn(256 * 256 * 2 - 1, dtype=torch.bfloat16, device="npu"),
        ]

        weights_1 = weights
        weights_2 = [weight.clone() for weight in weights]

        lr = 1.0
        optimizer_1 = MiniZero_1(weights_1, lr, dp_group)
        optimizer_2 = MiniOptimizer(weights_2, lr, dp_group)

        for _ in range(100):
            for w1, w2 in zip(weights_1, weights_2):
                main_grads = [
                    torch.randn_like(w1, dtype=torch.float32, device="npu")
                    for _ in range(world_size)
                ]
                # Choose based on rank to make sure the grads of different ranks are different.
                main_grad = main_grads[rank]
                w1.main_grad = main_grad
                w2.main_grad = main_grad

            optimizer_1.step()
            optimizer_2.step()

            for w1, w2 in zip(weights_1, weights_2):
                torch.testing.assert_close(w1, w2, atol=0, rtol=0)

    @pytest.mark.parametrize("manual_post_all_gather_processing", [False, True])
    @pytest.mark.parametrize("quantization", _quantization_list)
    def test_cast_master_weights_to_fp8(
        self,
        quantization,
        manual_post_all_gather_processing,
    ):
        """Train an FP8-primary-weight model with a mini zero-1 optimizer and compare its
        loss trajectory against a BF16 model trained under the same quantized autocast.
        """
        dp_group = dist.group.WORLD
        rank = dist.get_rank(dp_group)
        world_size = dist.get_world_size(dp_group)

        torch.manual_seed(12345)

        mock_groups = [dist.new_group(ranks=[i]) for i in range(world_size)]
        mock_group = mock_groups[rank]

        linear_kwargs = {
            "params_dtype": torch.bfloat16,
            "bias": False,
            "fuse_wgrad_accumulation": True,
        }
        model_fp8, model = _make_models(quantization, linear_kwargs)

        # Allocate main_grads for each weight
        for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
            w_fp8.main_grad = torch.zeros_like(w_fp8, dtype=torch.float32, device="npu")
            w.main_grad = torch.zeros_like(w, dtype=torch.float32, device="npu")

        optimizer_fp8 = MiniZero_1(
            list(model_fp8.parameters()),
            10.0,
            dp_group,
            manual_post_all_gather_processing,
        )
        optimizer = MiniZero_1(list(model.parameters()), 10.0, dp_group)

        for i in range(100):
            for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
                w_fp8.main_grad.zero_()
                w.main_grad.zero_()

            inputs = [
                torch.randn(32, 128, dtype=torch.bfloat16, device="npu") for _ in range(world_size)
            ]
            # Choose based on rank to make sure the inputs of different ranks are different.
            x = inputs[rank]

            with te.autocast(
                enabled=quantization is not None,
                recipe=_get_quantization_recipe(quantization),
                amax_reduction_group=mock_group,
            ):
                y_fp8 = model_fp8(x)

            with te.autocast(
                enabled=quantization is not None,
                recipe=_get_quantization_recipe(quantization),
                amax_reduction_group=mock_group,
            ):
                y = model(x)

            targets = [torch.randn_like(y) for _ in range(world_size)]
            # Choose based on rank to make sure the targets of different ranks are different.
            target = targets[rank]
            loss_fp8 = nn.MSELoss()(y_fp8, target)
            loss = nn.MSELoss()(y, target)

            loss_fp8.backward()
            loss.backward()

            optimizer_fp8.step()
            optimizer.step()

            assert torch.allclose(loss_fp8, loss, atol=0, rtol=0), (
                f"Loss mismatch at rank {rank}, step {i} for {quantization}"
            )

    @pytest.mark.parametrize("manual_post_all_gather_processing", [False, True])
    @pytest.mark.parametrize("quantization", _quantization_list)
    def test_cast_master_weights_checkpoint_resume(
        self,
        quantization,
        manual_post_all_gather_processing,
    ):
        """Resume sharded FP32-master updates from an FP8 primary-weight checkpoint."""
        dp_group = dist.group.WORLD
        rank = dist.get_rank(dp_group)
        linear_kwargs = {
            "params_dtype": torch.bfloat16,
            "bias": False,
            "fuse_wgrad_accumulation": True,
        }
        model_fp8, _ = _make_models(quantization, linear_kwargs)
        for weight in model_fp8.parameters():
            weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device="npu")

        optimizer_fp8 = MiniZero_1(
            list(model_fp8.parameters()),
            10.0,
            dp_group,
            manual_post_all_gather_processing,
        )
        for index, weight in enumerate(model_fp8.parameters()):
            weight.main_grad.fill_((rank + 1) * (index + 1))
        optimizer_fp8.step()

        buffer = io.BytesIO()
        torch.save(
            {
                "model": model_fp8.state_dict(),
                "master_weights": [
                    master_weight.detach().clone() if master_weight is not None else None
                    for master_weight in optimizer_fp8.master_weights
                ],
            },
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)

        restored_model, _ = _make_models(quantization, linear_kwargs)
        restored_model.load_state_dict(checkpoint["model"])
        for weight in restored_model.parameters():
            weight.main_grad = torch.zeros(weight.shape, dtype=torch.float32, device="npu")
        restored_optimizer = MiniZero_1(
            list(restored_model.parameters()),
            10.0,
            dp_group,
            manual_post_all_gather_processing,
        )
        for restored_master, saved_master in zip(
            restored_optimizer.master_weights,
            checkpoint["master_weights"],
        ):
            if saved_master is None:
                assert restored_master is None
            else:
                assert restored_master is not None
                restored_master.copy_(saved_master)

        _assert_quantized_weights_equal(
            list(restored_model.parameters()),
            list(model_fp8.parameters()),
        )
        for restored_master, expected_master in zip(
            restored_optimizer.master_weights,
            optimizer_fp8.master_weights,
        ):
            if expected_master is None:
                assert restored_master is None
            else:
                torch.testing.assert_close(restored_master, expected_master, atol=0, rtol=0)

        for index, (weight, restored_weight) in enumerate(
            zip(model_fp8.parameters(), restored_model.parameters())
        ):
            grad_value = (rank + 1) * (index + 1) / 4
            weight.main_grad.fill_(grad_value)
            restored_weight.main_grad.fill_(grad_value)
        optimizer_fp8.step()
        restored_optimizer.step()

        _assert_quantized_weights_equal(
            list(restored_model.parameters()),
            list(model_fp8.parameters()),
        )
        for restored_master, expected_master in zip(
            restored_optimizer.master_weights,
            optimizer_fp8.master_weights,
        ):
            if expected_master is None:
                assert restored_master is None
            else:
                torch.testing.assert_close(restored_master, expected_master, atol=0, rtol=0)
