# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""CPU offloading tests adapted for Ascend NPU.

These tests validate the CPU offloading functionality (OffloadableLayerState,
DefaultOffloadSynchronizer, ManualOffloadSynchronizer) on NPU, exercising
tensor offload/reload roundtrips, memory accounting, and integration with
TE-NPU modules (Linear, LayerNormLinear, LayerNormMLP, GroupedLinear).

Adapted from NVIDIA's test_cpu_offloading.py with the following NPU-specific
changes:
- Device: cuda → npu
- Streams: torch.cuda.Stream → torch.npu.Stream
- Events: torch.cuda.Event → torch.npu.Event
- Memory: torch.cuda.memory_allocated → torch.npu.memory_allocated
- Layer types: reduced to TE-NPU available modules
- Quantization: only MXFP8 (no DelayedScaling, Float8CurrentScaling, NVFP4)
- No CUDA graphs, no attention backend parametrization
"""

from __future__ import annotations

import contextlib
import gc
import random
from typing import Optional

import pytest
import torch


from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    Linear,
    GroupedLinear,
    LayerNormLinear,
    autocast,
)
from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.cpu_offload import (
    get_cpu_offload_context,
    OffloadableLayerState,
    DefaultOffloadSynchronizer,
    start_offload,
    mark_not_offload,
)
from transformer_engine.pytorch.ops import L2Normalization
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

from transformer_engine.pytorch.quantization.utils import check_npu_version
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_quantization_recipes = [
    None,
    pytest.param(
        recipe.MXFP8BlockScaling(),
        marks=pytest.mark.skipif(
            not check_npu_version(NPUVersion.A5),
            reason="FP8 (MXFP8BlockScaling) requires Ascend 950 or newer (Ascend 910B does not support FP8)",
        ),
    ),
]

_D = 256
_H = 4
_B = 128
_S = 512
_EPSILON = 0.1

gc.disable()


class _Utils:
    tensor1 = torch.randn((1024, 1024), device="npu", dtype=torch.bfloat16)

    @staticmethod
    def long_job(stream: Optional[torch.npu.Stream] = None):
        NUM_ITERS = 6000
        if stream is None:
            stream = torch.npu.current_stream()
        with torch.npu.stream(stream):
            for _ in range(NUM_ITERS):
                _Utils.tensor1.normal_()

    @staticmethod
    def measure_time(func):
        import time

        torch.npu.synchronize()
        start = time.time()
        func()
        torch.npu.synchronize()
        end = time.time()
        return (end - start) * 1000

    @staticmethod
    def get_npu_memory_mb():
        return torch.npu.memory_allocated() / (1024**2)

    @staticmethod
    def get_max_npu_memory_mb():
        return torch.npu.max_memory_allocated() / (1024**2)

    @staticmethod
    def get_cpu_memory_mb() -> float:  # pylint: disable=inconsistent-return-statements
        try:
            import psutil
            import os

            return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except ImportError:
            pytest.skip("psutil not installed")

    @staticmethod
    def get_layer_names():
        return [
            "linear",
            "layernorm_linear",
            "layernorm_mlp",
            "grouped_linear",
        ]

    @staticmethod
    def create_layer(layer_type: str):
        if layer_type == "linear":
            return Linear(_D, _D, params_dtype=torch.bfloat16, device="npu")
        elif layer_type == "layernorm_linear":
            return LayerNormLinear(_D, _D, params_dtype=torch.bfloat16, device="npu")
        elif layer_type == "layernorm_mlp":
            from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP

            return LayerNormMLP(_D, _D, params_dtype=torch.bfloat16, device="npu")
        elif layer_type == "grouped_linear":
            return GroupedLinear(_H, _D, _D, params_dtype=torch.bfloat16, device="npu")
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    @staticmethod
    def create_tensor(
        fp8_recipe: Optional[recipe.Recipe],
        requires_grad: bool = False,
    ) -> torch.Tensor:
        shape = (_B, _S, _D)
        tensor = torch.randn(shape, device="npu", dtype=torch.bfloat16)
        if fp8_recipe is None:
            return tensor.requires_grad_(requires_grad) if requires_grad else tensor
        if fp8_recipe.mxfp8():
            from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

            quantizer = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)
            return quantizer(tensor)
        raise ValueError(f"Unsupported recipe: {fp8_recipe}")

    @staticmethod
    def create_recipe_ctx(fp8_recipe: Optional[recipe.Recipe]):
        if fp8_recipe is None:
            return lambda: contextlib.nullcontext()  # pylint: disable=unnecessary-lambda
        return lambda: autocast(enabled=True, recipe=fp8_recipe)  # pylint: disable=unnecessary-lambda

    @staticmethod
    def get_tensor_size_mb(tensor):
        if tensor is None:
            return 0
        if isinstance(tensor, QuantizedTensorStorage):
            return sum(_Utils.get_tensor_size_mb(t) for t in tensor.get_data_tensors())
        return tensor.numel() * tensor.element_size() / (1024**2)

    @staticmethod
    def memory_leak_check():
        if _Utils.get_npu_memory_mb() > 1000:
            memory_num = _Utils.get_npu_memory_mb()
            gc.collect()
            gc.disable()
            raise RuntimeError(f"Memory leak: {memory_num} MB")


class TestOffloadableLayerState:
    @pytest.mark.parametrize("random_num_tensors", [True, False])
    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    def test_general(self, random_num_tensors, fp8_recipe):
        _Utils.memory_leak_check()
        NUM_ITERATIONS = 10

        stream = torch.npu.Stream()
        offload_layer_state = OffloadableLayerState(offload_stream=stream)

        for _ in range(NUM_ITERATIONS):
            original_tensors = []
            tensors_ids = []
            NUM_TENSORS = random.choice([1, 20]) if random_num_tensors else 1
            for _ in range(NUM_TENSORS):
                tensor = _Utils.create_tensor(fp8_recipe)
                original_tensors.append(tensor)
                tensor_id = offload_layer_state.push_tensor(tensor)
                assert tensor.device.type == "npu"
                tensors_ids.append(tensor_id)

            offload_layer_state.start_offload()
            offload_layer_state.release_activation_forward_gpu_memory()
            offload_layer_state.start_reload()

            for j in range(len(tensors_ids)):
                tensor_npu = offload_layer_state.pop_tensor(tensors_ids[j])
                assert tensor_npu.device.type == "npu"
                assert tensor_npu.shape == original_tensors[j].shape
                assert tensor_npu.dtype == original_tensors[j].dtype
                torch.testing.assert_close(tensor_npu, original_tensors[j])
            offload_layer_state.release_all_memory()
        torch.npu.synchronize()

    def test_offload_base_tensor(self):
        _Utils.memory_leak_check()
        stream = torch.npu.Stream()
        offload_layer_state = OffloadableLayerState(offload_stream=stream)
        init_npu_memory = _Utils.get_npu_memory_mb()
        x = _Utils.create_tensor(None)
        x_size = _Utils.get_tensor_size_mb(x)
        x_1 = x[::2]
        x_2 = x[1::2]

        start_offload(x_1, offload_base_tensor=True)
        start_offload(x_2, offload_base_tensor=True)
        x1_id = offload_layer_state.push_tensor(x_1)
        x2_id = offload_layer_state.push_tensor(x_2)
        del x_1, x_2
        offload_layer_state.start_offload()
        offload_layer_state.release_activation_forward_gpu_memory()

        assert offload_layer_state.get_offloaded_total_size_mb() == pytest.approx(x_size, 0.1)

        offload_layer_state.start_reload()
        x_1 = offload_layer_state.pop_tensor(x1_id)
        x_2 = offload_layer_state.pop_tensor(x2_id)
        assert x_1.device.type == "npu"
        assert x_2.device.type == "npu"

        assert torch.allclose(x_1, x[::2])
        assert torch.allclose(x_2, x[1::2])
        del x

        assert _Utils.get_npu_memory_mb() == pytest.approx(init_npu_memory + x_size, 0.1)


class TestDefaultOffloadSynchronizer:
    @pytest.mark.parametrize("random_num_tensors", [True, False])
    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    def test_general(self, random_num_tensors, fp8_recipe):
        _Utils.memory_leak_check()
        NUM_LAYERS = 10
        NUM_ITERATIONS = 10

        offload_synchronizer = DefaultOffloadSynchronizer(
            num_layers=NUM_LAYERS,
            num_offloaded_layers=NUM_LAYERS - 1,
        )

        for _ in range(NUM_ITERATIONS):
            original_tensors = []
            tensors_ids = []
            layer_ids = []

            for i in range(NUM_LAYERS):
                NUM_LAYER_TENSORS = random.randint(1, 10) if random_num_tensors else 1
                layer_tensors = []
                layer_tensors_ids = []
                layer_id = offload_synchronizer.fwd_step()
                for _ in range(NUM_LAYER_TENSORS):
                    tensor = _Utils.create_tensor(fp8_recipe)
                    layer_tensors.append(tensor)
                    tensor_id = offload_synchronizer.push_tensor(tensor)
                    assert tensor.device.type == "npu"
                    layer_tensors_ids.append(tensor_id)
                layer_ids.append(layer_id)
                tensors_ids.append(layer_tensors_ids)
                original_tensors.append(layer_tensors)
            for i in range(NUM_LAYERS - 1, -1, -1):
                offload_synchronizer.bwd_step(layer_ids[i])
                for j in range(len(tensors_ids[i])):
                    tensor_npu = offload_synchronizer.pop_tensor(tensors_ids[i][j])
                    assert tensor_npu.device.type == "npu"
                    assert tensor_npu.shape == original_tensors[i][j].shape
                    assert tensor_npu.dtype == original_tensors[i][j].dtype
                    torch.testing.assert_close(tensor_npu, original_tensors[i][j])
            offload_synchronizer.finish_part_of_bwd()
        torch.npu.synchronize()

    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    def test_memory(self, fp8_recipe):
        torch.npu.synchronize()
        _Utils.memory_leak_check()
        NUM_LAYERS = 10

        torch.npu.reset_peak_memory_stats()

        offload_synchronizer = DefaultOffloadSynchronizer(
            num_layers=NUM_LAYERS,
            num_offloaded_layers=NUM_LAYERS - 1,
        )

        init_npu_memory = _Utils.get_npu_memory_mb()
        tensor_ids = []

        torch.npu.synchronize()
        for _ in range(NUM_LAYERS):
            offload_synchronizer.fwd_step()
            tensor = _Utils.create_tensor(fp8_recipe)
            tensor_size = _Utils.get_tensor_size_mb(tensor)
            tensor_id = offload_synchronizer.push_tensor(tensor)
            assert tensor.device.type == "npu"
            tensor_ids.append(tensor_id)
            del tensor, tensor_id
        torch.npu.synchronize()

        if fp8_recipe is None:
            assert _Utils.get_max_npu_memory_mb() <= init_npu_memory + tensor_size * 4
        assert _Utils.get_npu_memory_mb() == pytest.approx(init_npu_memory + tensor_size, 0.1)

        for i in range(NUM_LAYERS - 1, -1, -1):
            offload_synchronizer.bwd_step(i)
            tensor_npu = offload_synchronizer.pop_tensor(tensor_ids[i])
            assert tensor_npu.device.type == "npu"
            del tensor_npu, tensor_ids[i]
        offload_synchronizer.finish_part_of_bwd()

        del tensor_ids
        torch.npu.synchronize()

        if fp8_recipe is None:
            assert _Utils.get_max_npu_memory_mb() <= init_npu_memory + tensor_size * 4
        assert _Utils.get_npu_memory_mb() == pytest.approx(init_npu_memory, 0.1)

    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    def test_multiple_tensor_offload(self, fp8_recipe):
        _Utils.memory_leak_check()
        init_cpu_memory = _Utils.get_cpu_memory_mb()
        init_npu_memory = _Utils.get_npu_memory_mb()
        offload_synchronizer = DefaultOffloadSynchronizer(
            num_layers=2,
            num_offloaded_layers=1,
        )
        x1 = _Utils.create_tensor(fp8_recipe)
        x_size = _Utils.get_tensor_size_mb(x1)
        offload_synchronizer.fwd_step()
        offload_synchronizer.push_tensor(x1)
        offload_synchronizer.push_tensor(x1)
        offload_synchronizer.push_tensor(x1)
        if fp8_recipe is not None:
            x1.dequantize()
        offload_synchronizer.fwd_step()
        assert _Utils.get_cpu_memory_mb() == pytest.approx(init_cpu_memory + 1 * x_size, 0.1)
        del x1
        offload_synchronizer.bwd_step(1)
        offload_synchronizer.bwd_step(0)
        offload_synchronizer.finish_part_of_bwd()

        assert _Utils.get_npu_memory_mb() == pytest.approx(init_npu_memory, 0.1)


class TestTELayers:
    @pytest.mark.parametrize("layer_type", _Utils.get_layer_names())
    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    def test_sanity(self, layer_type, fp8_recipe):
        _Utils.memory_leak_check()

        recipe_ctx = _Utils.create_recipe_ctx(fp8_recipe)
        OFFLOAD_LAYERS = 6
        NUM_LAYERS = 10
        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=OFFLOAD_LAYERS,
            model_layers=NUM_LAYERS,
        )
        layers = [_Utils.create_layer(layer_type) for _ in range(NUM_LAYERS)]
        inp = _Utils.create_tensor(None)
        m_splits = {"m_splits": [_B * _S // _H] * _H} if layer_type == "grouped_linear" else {}
        out = inp
        for i in range(NUM_LAYERS):
            with offload_ctx, recipe_ctx():
                out = layers[i](out, is_first_microbatch=False, **m_splits)
            out = sync_function(out)
        out.sum().backward()
        torch.npu.synchronize()
        del out, inp, layers

    @pytest.mark.parametrize("layer_type", _Utils.get_layer_names())
    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    def test_memory(self, layer_type, fp8_recipe):
        _Utils.memory_leak_check()

        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=1,
            model_layers=2,
            offload_activations=True,
            offload_weights=False,
        )
        recipe_ctx = _Utils.create_recipe_ctx(fp8_recipe)
        layer = _Utils.create_layer(layer_type)
        inp = _Utils.create_tensor(None)

        m_splits = {"m_splits": [_B * _S // _H] * _H} if layer_type == "grouped_linear" else {}

        with recipe_ctx():
            out = layer(inp, is_first_microbatch=True, **m_splits)
        out.sum().backward()

        del inp
        init_npu_memory = _Utils.get_npu_memory_mb()

        inp = _Utils.create_tensor(None)
        with recipe_ctx():
            out = layer(inp, is_first_microbatch=False, **m_splits)
        with recipe_ctx():
            out = out + 1
        del inp
        npu_memory_no_offload = _Utils.get_npu_memory_mb()

        out.sum().backward()
        inp = _Utils.create_tensor(None)
        with offload_ctx, recipe_ctx():
            out = layer(inp, is_first_microbatch=False, **m_splits)
        out = sync_function(out)
        with offload_ctx, recipe_ctx():
            out = out + 1
        out = sync_function(out)
        del inp
        assert _Utils.get_npu_memory_mb() == pytest.approx(init_npu_memory, 0.1)
        offloaded_memory_cpu = offload_ctx.offload_synchronizer.get_offloaded_total_size_mb()

        assert _Utils.get_npu_memory_mb() + offloaded_memory_cpu == pytest.approx(
            npu_memory_no_offload, 1
        )
        out.sum().backward()

    def test_l2_normalization_saved_activation_offloads_to_cpu(self):
        _Utils.memory_leak_check()

        offload_ctx, sync_function, manual_controller = get_cpu_offload_context(
            enabled=True,
            model_layers=1,
            offload_activations=True,
            manual_synchronization=True,
        )
        x_ref = torch.randn((512, 512), device="npu", dtype=torch.bfloat16, requires_grad=True)
        x_test = x_ref.detach().clone().requires_grad_()
        dy = torch.randn_like(x_ref)

        y_ref = L2Normalization()(x_ref)
        y_ref.backward(dy)

        with offload_ctx:
            y_test = L2Normalization()(x_test)
        y_test = sync_function(y_test)
        manual_controller.start_offload_layer(0)
        manual_controller.release_activation_forward_gpu_memory(0)

        assert offload_ctx.offload_synchronizer.get_offloaded_total_size_mb() > 0

        manual_controller.start_reload_layer(0)
        y_test.backward(dy)

        torch.testing.assert_close(y_test, y_ref)
        torch.testing.assert_close(x_test.grad, x_ref.grad)

    @pytest.mark.parametrize("layer_type", _Utils.get_layer_names())
    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    def test_manual_synchronization(self, fp8_recipe, layer_type):
        _Utils.memory_leak_check()

        offload_ctx, sync_function, manual_controller = get_cpu_offload_context(
            enabled=True,
            model_layers=6,
            offload_activations=True,
            manual_synchronization=True,
        )
        layer_1 = _Utils.create_layer(layer_type)
        layer_2 = _Utils.create_layer(layer_type)
        inp1 = _Utils.create_tensor(None)
        inp2 = _Utils.create_tensor(None)

        recipe_ctx = _Utils.create_recipe_ctx(fp8_recipe)

        m_splits = {"m_splits": [_B * _S // _H] * _H} if layer_type == "grouped_linear" else {}

        with offload_ctx, recipe_ctx():
            out_1 = layer_1(inp1, **m_splits)
        out_1 = sync_function(out_1)

        with offload_ctx, recipe_ctx():
            out_2 = layer_2(inp2, **m_splits)
        out_2 = sync_function(out_2)

        mark_not_offload(out_1, out_2)

        del inp1, inp2

        memory_before_offload = _Utils.get_npu_memory_mb()
        manual_controller.start_offload_layer(0)
        manual_controller.release_activation_forward_gpu_memory(0)
        manual_controller.start_offload_layer(1)
        manual_controller.release_activation_forward_gpu_memory(1)
        memory_after_offload = _Utils.get_npu_memory_mb()
        assert memory_after_offload - _EPSILON <= memory_before_offload

        manual_controller.start_reload_layer(0)
        manual_controller.start_reload_layer(1)

        memory_after_reload = _Utils.get_npu_memory_mb()
        assert memory_after_reload == pytest.approx(memory_before_offload, 1)

        out_1.sum().backward()
        out_2.sum().backward()

    @pytest.mark.parametrize("fp8_recipe", _quantization_recipes)
    @pytest.mark.parametrize("layer_type", _Utils.get_layer_names())
    def test_numerics(self, fp8_recipe, layer_type):
        recipe_ctx = _Utils.create_recipe_ctx(fp8_recipe)

        offload_ctx, sync_function = get_cpu_offload_context(
            enabled=True,
            num_layers=1,
            model_layers=2,
            offload_activations=True,
            offload_weights=False,
        )

        class Callable(torch.nn.Module):
            def __init__(self, offload_ctx=None, sync_function=None):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [_Utils.create_layer(layer_type) for _ in range(2)]
                )
                self.offload_ctx = offload_ctx
                self.sync_function = sync_function

            def forward(self, x):
                m_splits = (
                    {"m_splits": [_B * _S // _H] * _H} if layer_type == "grouped_linear" else {}
                )
                for layer in self.layers:
                    with self.offload_ctx, recipe_ctx():
                        x = layer(x, is_first_microbatch=False, **m_splits)
                    if self.sync_function is not None:
                        x = self.sync_function(x)
                return x

        callable_offload = Callable(offload_ctx=offload_ctx, sync_function=sync_function)
        callable_no_offload = Callable(offload_ctx=contextlib.nullcontext(), sync_function=None)

        for param_offload, param_no_offload in zip(
            callable_offload.parameters(), callable_no_offload.parameters()
        ):
            param_offload.data.copy_(param_no_offload.data)

        x = _Utils.create_tensor(None)

        for _ in range(4):
            out = callable_offload(x)
            out.sum().backward()
            out = callable_no_offload(x)
            out.sum().backward()

        callable_offload.zero_grad(set_to_none=True)
        out_offload = callable_offload(x)
        out_offload.sum().backward()

        offload_outs = [out_offload]
        for param in callable_offload.parameters():
            offload_outs.append(param.detach().clone())

        torch.npu.reset_peak_memory_stats()
        out_no_offload = callable_no_offload(x)
        out_no_offload.sum().backward()

        no_offload_outs = [out_no_offload]
        for param in callable_no_offload.parameters():
            no_offload_outs.append(param.detach().clone())

        for i in range(len(offload_outs)):
            assert torch.allclose(offload_outs[i], no_offload_outs[i]), f"Error in tensor {i}."

        torch.npu.synchronize()

    def test_example_from_doc(self):
        offload_stream = torch.npu.Stream()
        num_layers = 10
        layers = [_Utils.create_layer("linear") for _ in range(num_layers)]
        inp = [_Utils.create_tensor(None) for _ in range(num_layers)]
        out = [None] * num_layers
        cpu_offload_context, sync_function, manual_controller = get_cpu_offload_context(
            enabled=True,
            model_layers=num_layers,
            manual_synchronization=True,
            offload_stream=offload_stream,
        )

        for i in range(num_layers):
            with cpu_offload_context:
                out[i] = layers[i].forward(inp[i])
            out[i] = sync_function(out[i])
            manual_controller.start_offload_layer(i)

        offload_stream.synchronize()
        for i in range(num_layers):
            manual_controller.release_activation_forward_gpu_memory(i)

        for i in range(num_layers - 1, -1, -1):
            manual_controller.start_reload_layer(i)

        offload_stream.synchronize()
        for i in range(num_layers):
            out[i].sum().backward()
