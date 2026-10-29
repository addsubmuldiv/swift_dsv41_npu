# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 model sharding tests.

Usage:
  pytest tests/pytorch/test_fsdp.py -v --tb=short
  Use -k to filter: -k Linear, -k "not LayerNorm", etc.
"""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.distributed.tensor import DTensor

from transformer_engine.common import recipe as te_recipe
from transformer_engine.pytorch import (
    LayerNormLinear,
    Linear,
    autocast,
    fp8_model_init,
)
from transformer_engine.pytorch.constants import NPUVersion
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
from transformer_engine.pytorch.module.layernorm_mlp import LayerNormMLP
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.utils import check_npu_version

from utils import npu_available, npu_device_count
from distributed_testing import DistributedTest


# ═══════════════════════════════════════════════════════════════════════
# Helpers — device, dist, random
# ═══════════════════════════════════════════════════════════════════════

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


def _dist_print(msg):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(msg)


def _seed(seed=42):
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)


def _sync() -> None:
    torch.npu.synchronize()


def _randn_npu(shape, *, dtype, device, requires_grad=False):
    return (
        torch.randn(shape, dtype=dtype, device="cpu")
        .to(device=device)
        .requires_grad_(requires_grad)
    )


# ═══════════════════════════════════════════════════════════════════════
# FSDP2 availability
# ═══════════════════════════════════════════════════════════════════════


def _fsdp2_available() -> bool:
    try:
        __import__("torch.distributed._composable.fsdp")
        __import__("torch.distributed.device_mesh")
        return True
    except ImportError:
        return False


_fsdp2_skip = pytest.mark.skipif(not _fsdp2_available(), reason="FSDP2 not available")


# ═══════════════════════════════════════════════════════════════════════
# (recipe_name, fp8_init) parametrization
# ═══════════════════════════════════════════════════════════════════════


def _build_recipe_fp8_params():
    """Build (recipe_name, fp8_init) pairs. FP8 recipes only on A5 devices."""
    params = [pytest.param("None", False, id="None")]
    if check_npu_version(NPUVersion.A5):
        for name in ("DelayedScaling", "Float8CurrentScaling", "Float8BlockScaling"):
            params.append(pytest.param(name, False, id=name))
            params.append(pytest.param(name, True, id=f"{name}-fp8_init"))
        params.append(pytest.param("MXFP8BlockScaling", False, id="MXFP8BlockScaling"))
    return params


# ═══════════════════════════════════════════════════════════════════════
# Layer type mapping
# ═══════════════════════════════════════════════════════════════════════


def _get_te_layer_from_string(layer_name):
    te_layer_types = [Linear, LayerNormLinear, LayerNormMLP]
    te_layer_names = [layer.__name__ for layer in te_layer_types]
    te_layer_map = dict(zip([name.lower() for name in te_layer_names], te_layer_types))
    if layer_name.lower() not in te_layer_map:
        raise ValueError(
            f"'{layer_name}' is not a valid TE layer, please choose from {te_layer_names}."
        )
    return te_layer_map[layer_name.lower()]


# ═══════════════════════════════════════════════════════════════════════
# FSDP2 model builders
# ═══════════════════════════════════════════════════════════════════════


def _init_te_model(config):
    hidden_size = config.num_heads * config.head_dim
    args_list = [hidden_size, hidden_size]
    inp_shape = [config.seq_length, config.batch_size, hidden_size]
    out_shape = [config.seq_length, config.batch_size, hidden_size]

    if config.params_dtype == "float16":
        params_dtype = torch.float16
    elif config.params_dtype == "bfloat16":
        params_dtype = torch.bfloat16
    else:
        params_dtype = torch.float32

    kwargs = {"params_dtype": params_dtype, "device": "npu"}
    layer_type = _get_te_layer_from_string(config.layer_type)

    if layer_type is LayerNormLinear:
        args_list[1] *= 3  # QKV projection
        out_shape[-1] *= 3
    elif layer_type is LayerNormMLP:
        args_list[1] *= 4  # FFN hidden size

    model = layer_type(*args_list, **kwargs)
    return model, inp_shape, out_shape


# ═══════════════════════════════════════════════════════════════════════
# FSDP2 utilities
# ═══════════════════════════════════════════════════════════════════════


def _get_device_mesh(world_size, sharding_dims):
    from torch.distributed import DeviceMesh

    device_ids = list(range(world_size))
    _dist_print(f"sharding-dims:{sharding_dims}")
    assert sharding_dims[0] == world_size
    return DeviceMesh("npu", device_ids)


def _save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        if isinstance(param, QuantizedTensor):
            ignore_keys = [key for key in param.__dict__.keys() if key.startswith("_")]
        else:
            ignore_keys = []
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items() if k not in ignore_keys}
    return custom_attrs


def _restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)


def _shard_model_with_fsdp2(model, mesh):
    from torch.distributed._composable.fsdp import fully_shard

    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


# ═══════════════════════════════════════════════════════════════════════
# FP8 FSDP2 all-gather check
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _check_fp8_fsdp2_allgather(model):
    ref_allgathered = {}
    for name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            continue
        local_tensor = param._local_tensor
        device_mesh = param.device_mesh
        dist_group = (
            device_mesh.get_group(mesh_dim="shard")
            if device_mesh.ndim > 1
            else device_mesh.get_group()
        )
        hp_tensor = local_tensor.dequantize()
        gathered = [
            torch.zeros_like(hp_tensor) for _ in range(dist.get_world_size(group=dist_group))
        ]
        dist.all_gather(gathered, hp_tensor, group=dist_group)
        full_tensor = torch.cat(gathered, dim=0)
        ref_allgathered[name] = full_tensor

    for module in model.modules():
        if hasattr(module, "unshard"):
            module.unshard()

    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param.dequantize().float(),
            ref_allgathered[name].float(),
            atol=5e-2,
            rtol=5e-2,
        )

    for module in model.modules():
        if hasattr(module, "reshard"):
            module.reshard()


# ═══════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════


def _log_mem(tag, device):
    alloc = torch.npu.memory_allocated(device) / 1e6
    peak = torch.npu.max_memory_allocated(device) / 1e6
    _dist_print(f"Memory {tag}: alloc={alloc:.1f} MB, peak={peak:.1f} MB")


def _build_sharding_dims_params():
    return [[npu_device_count()]]  # 1D FSDP only


# ═══════════════════════════════════════════════════════════════════════
# FSDP2 test
# ═══════════════════════════════════════════════════════════════════════


class TestFSDP2(DistributedTest):
    """FSDP2 model sharding tests — runs each worker inline via ``DistributedTest``."""

    world_size = None  # resolved from sharding_dims fixture kwarg
    reuse_dist_env = False

    @_fsdp2_skip
    @pytest.mark.parametrize("sharding_dims", _build_sharding_dims_params())
    @pytest.mark.parametrize("layer_type", ["Linear", "LayerNormLinear", "LayerNormMLP"])
    @pytest.mark.parametrize("recipe_name,fp8_init", _build_recipe_fp8_params())
    def test_fsdp2(self, recipe_name, fp8_init, sharding_dims, layer_type):
        """FSDP2 model sharding — runs in each worker.

        Framework *DistributedTest._dist_run* already sets env vars,
        ``set_device`` and ``init_process_group``, so the body can
        jump straight into the training logic.
        """
        os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] = "auto"
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"npu:{rank}")

        torch.npu.synchronize()
        dist.barrier()
        FP8GlobalStateManager.reset()
        _seed(42)

        args = SimpleNamespace(
            recipe=recipe_name,
            fp8_init=fp8_init,
            sharding_dims=list(sharding_dims),
            layer_type=layer_type,
            seed=42,
            num_heads=32,
            head_dim=128,
            batch_size=16,
            seq_length=128,
            params_dtype="bfloat16",
            num_layers=4,
            iter=10,
        )

        _log_mem("before model init", device)

        recipe = getattr(te_recipe, args.recipe)() if args.recipe != "None" else None

        build_ctx = fp8_model_init if args.fp8_init else nullcontext
        build_kwargs = {"enabled": True, "recipe": recipe} if args.fp8_init else {}
        with build_ctx(**build_kwargs):
            model, inp_shape, out_shape = _init_te_model(args)
        model = model.to(device)

        _log_mem("after  model init", device)

        # Shard with FSDP2
        mesh = _get_device_mesh(world_size, args.sharding_dims)
        custom_attrs = _save_custom_attrs(model)
        model = _shard_model_with_fsdp2(model, mesh)
        _restore_custom_attrs(model, custom_attrs)
        assert model.is_fsdp2, "FSDP2 detection failed: is_fsdp2 should be True after fully_shard"

        _log_mem("after  FSDP2 shard", device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        torch.npu.reset_peak_memory_stats(device)

        for iteration in range(args.iter):
            optimizer.zero_grad()
            inp = _randn_npu(inp_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
            target = _randn_npu(out_shape, dtype=torch.bfloat16, device=device)

            with autocast(enabled=recipe is not None, recipe=recipe):
                output = model(inp)
                if isinstance(output, tuple):
                    output = output[0]
                loss = F.mse_loss(output, target)

            loss.backward()
            optimizer.step()
            _log_mem(f"iteration {iteration}, loss {loss.item():.6f}", device)

        if args.fp8_init:
            _check_fp8_fsdp2_allgather(model)

        _sync()
        gc.collect()
        torch.npu.empty_cache()
        if dist.is_initialized():
            dist.barrier()
