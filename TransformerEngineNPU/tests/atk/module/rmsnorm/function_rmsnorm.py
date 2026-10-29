# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK independent RMSNorm forward and backward comparison plugins.

Backward receives shared statistics generated once by the producer.
Both phases execute the installed TE implementation on NPU or GPU.
"""

import json
from typing import Dict

import torch

from atk.tasks.api_execute import register
from atk.case_generator.generator.data_types import DATATYPE_REGISTRY
from atk.case_generator.generator.data_types.data_base import BaseDataType

# ATK's API plugin loader resolves both BaseApi symbols from this namespace.
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi  # noqa: F401
from atk.tasks.api_execute.base_api import BaseApi


# Shared accuracy comparators (allclose / mare / mixed_tolerance) live in
# ../_comparators.py and are registered here under this module's prefix so
# the yaml standard.acc block can select any of the three standards.
import os
import sys

_comparators_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _comparators_dir not in sys.path:
    sys.path.insert(0, _comparators_dir)
from _comparators import register_module_comparators  # noqa: E402

register_module_comparators("rmsnorm")
from _nonfinite_contract import special_value_signatures  # noqa: E402


_module_dir = os.path.dirname(os.path.abspath(__file__))
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)
from rmsnorm_inputs import (  # noqa: E402
    BUNDLE_DTYPE,
    BUNDLE_FORMAT,
    prepare_input_bundle,
    validate_inputs,
    validate_inputs_against_case,
)


def _get_input_role():
    role = os.environ.get("TENPU_RMSNORM_INPUT_ROLE")
    if role not in ("producer", "receiver"):
        raise ValueError(
            "Set TENPU_RMSNORM_INPUT_ROLE=producer on NPU task, receiver on GPU server"
        )
    return role


@DATATYPE_REGISTRY.register("tenpu_rmsnorm_inputs")
class RMSNormInputBundle(BaseDataType):
    """ATK data plugin: only the producer prepares values; receiver awaits sync."""

    def gen_data(self):
        if _get_input_role() == "receiver":
            return None  # Serialized placeholder; ATK replaces it before device execution.
        return self.get_data(prepare_input_bundle(self.config.range_values))

    def get_data(self, data):
        # Called after ATK loads the synchronized file. Do not recompute shared statistics here.
        if not isinstance(data, dict) or data.get("format") != BUNDLE_FORMAT:
            raise ValueError("No complete RMSNorm input bundle received; use --syc_dataset")
        metadata = self.config.range_values
        if (
            data.get("phase") != metadata["phase"]
            or data.get("case_id") != metadata["source_case"]["id"]
        ):
            raise ValueError("RMSNorm input bundle belongs to another phase/case")
        validate_inputs(data["values"], backward=data["phase"] == "backward")
        validate_inputs_against_case(metadata["source_case"], data["values"][:5])
        return data


def _unpack_phase_inputs(input_data, phase):
    args = list(input_data.args)
    if len(args) == 1 and isinstance(args[0], dict):
        bundle = args[0]
        if bundle.get("format") != BUNDLE_FORMAT or bundle.get("phase") != phase:
            raise ValueError("Unexpected RMSNorm bundle format/phase")
        args = list(bundle["values"])
    validate_inputs(args, backward=phase == "backward")
    return args


def _validate_online_task(api):
    specs = api.task_result.case_config.inputs
    if len(specs) != 1 or getattr(specs[0], "dtype", None) != BUNDLE_DTYPE:
        return  # Direct tensor arguments do not require an input bundle.
    sync = api.task_result.syc_dataset_info
    if not sync or not sync.syc_dataset:
        raise ValueError("Online RMSNorm input bundles require --syc_dataset")
    if api.task_result.use_input_data_dir:
        raise ValueError(
            "Online bundles use ATK-managed inputs; put historical input_dir in case metadata"
        )
    if api.task_result.nodes.get_main_node().backend != "npu":
        raise ValueError("Online RMSNorm requires NPU as the main input-producing node")
    expected = "producer" if api.device == "npu" else "receiver"
    if _get_input_role() != expected:
        raise ValueError(f"RMSNorm {api.device} must use TENPU_RMSNORM_INPUT_ROLE={expected}")


def _encode_api_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _build_api_contract(module, output, eps, zero_centered_gamma) -> Dict:
    weight = module.weight
    return {
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "eps": float(eps),
        "zero_centered_gamma": bool(zero_centered_gamma),
        "named_parameters": [name for name, _ in module.named_parameters()],
        "state_dict_keys": list(module.state_dict().keys()),
    }


def _create_norm_module(args, device):
    import transformer_engine.pytorch as te

    x, _, gamma, eps, zero_centered = args[:5]
    module = te.RMSNorm(
        x.shape[-1],
        eps=float(eps),
        zero_centered_gamma=bool(zero_centered),
        params_dtype=x.dtype,
        device=device,
    )
    with torch.no_grad():
        module.weight.copy_(gamma)
    return module


def run_rmsnorm_forward(input_data, device):
    """Exercise the installed TE op_forward, including its saved statistics.

    requires_grad=True selects the training forward path, but no backward is
    run. Direct op invocation exposes the statistics without monkey-patching
    global torch/TE extension functions.
    """
    from transformer_engine.pytorch.ops.op import OperationContext

    args = _unpack_phase_inputs(input_data, "forward")
    module = _create_norm_module(args, device)
    ctx = OperationContext()
    ctx.requires_grad = True
    with torch.no_grad():
        output = module.op_forward(ctx, args[0].detach().clone(), None, None)
    _, rstdevs = ctx.to_save
    contract = _build_api_contract(module, output, args[3], args[4])
    if not torch.isfinite(args[0]).all():
        contract["special_value_signatures"] = special_value_signatures((output, rstdevs))
    return (
        output.detach(),
        rstdevs.detach().reshape(-1).clone(),
        _encode_api_contract(contract, output.device),
    )


def run_rmsnorm_backward(input_data, device):
    """Call the actual TE op_backward with a synthetic saved context.

    Both sides receive identical FP32 statistics from input.bin. Only statistic
    layout differs to match each release_v2.17 forward contract: TENPU
    saves input-dtype x and [rows, 1] statistics; NVTE saves input-dtype x and [rows]
    statistics. Each tensor is cloned because op_backward clears saved data.
    """
    from transformer_engine.pytorch.ops.op import OperationContext

    args = _unpack_phase_inputs(input_data, "backward")
    x, dy, _, eps, zero_centered, rstdevs = args
    module = _create_norm_module(args, device)
    saved_x = x.detach().reshape(-1, x.shape[-1]).contiguous().clone()
    rstdevs = rstdevs.detach().clone()
    if str(device).split(":", maxsplit=1)[0] == "npu":
        rstdevs = rstdevs.view(-1, 1)
    ctx = OperationContext()
    ctx.requires_grad = True
    ctx.dtype = x.dtype
    ctx.saved_tensors = (saved_x, rstdevs)
    with torch.no_grad():
        dx, (dw,) = module.op_backward(ctx, dy.detach().clone())
    contract = _build_api_contract(module, dx, eps, zero_centered)
    if not torch.isfinite(x).all():
        contract["special_value_signatures"] = special_value_signatures((dx, dw))
    return dx.detach(), dw.detach(), _encode_api_contract(contract, dx.device)


@register("function_rmsnorm_forward")
class FunctionRMSNormForward(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        _validate_online_task(self)

    def __call__(self, input_data, with_output=False):
        device = "cuda" if self.device in ("gpu", "cuda") else self.device
        if device not in ("npu", "cuda"):
            raise ValueError("Split RMSNorm compares TENPU/NPU with NVTE/GPU only")
        result = run_rmsnorm_forward(input_data, device)
        return result if with_output else None


@register("function_rmsnorm_backward")
class FunctionRMSNormBackward(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        _validate_online_task(self)

    def __call__(self, input_data, with_output=False):
        device = "cuda" if self.device in ("gpu", "cuda") else self.device
        if device not in ("npu", "cuda"):
            raise ValueError("Split RMSNorm compares TENPU/NPU with NVTE/GPU only")
        result = run_rmsnorm_backward(input_data, device)
        return result if with_output else None
