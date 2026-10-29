# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for the Megatron-facing fusible
activation ops in ``transformer_engine.pytorch.ops``.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of the ops. NVTE consistency
comes from the offline golden loop (``collect_nvte_golden.py`` + an ATK
``accuracy_load`` node), never from a local torch reference.

Guarded ops (all Megatron "Fusible ops" activation entries with references in
``megatron/core/extensions/transformer_engine.py`` and
``megatron/core/transformer/moe/experts.py``): GELU, SiLU, ReLU (unary) and
GEGLU, ReGLU, SwiGLU (gated), plus the per-row post-scaled fused-MoE variants
ScaledSwiGLU, ScaledClampedQGeGLU (gated) and ScaledSReLU (unary). The scaled
variants take an extra per-row scale input and do not require FP8 (they run in
BF16); only the FP8-quantized-input path remains out of scope.

These ops are stateless (no learnable parameters), so the outputs are the
forward result, the input gradient, and the serialized contract.
"""

import json
from typing import Dict, Optional, Tuple

import torch

from atk.tasks.api_execute import register

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

register_module_comparators("fused_activation")

RESULT_SCHEMA = [
    "forward_output",
    "input_grad",
    "api_contract_json_utf8",
]

# op_name -> class name in transformer_engine.pytorch.ops. Resolved lazily so
# importing this plugin does not require Transformer Engine at import time.
_OP_CLASS_NAMES = {
    "gelu": "GELU",
    "silu": "SiLU",
    "relu": "ReLU",
    "geglu": "GEGLU",
    "reglu": "ReGLU",
    "swiglu": "SwiGLU",
    "scaled_swiglu": "ScaledSwiGLU",
    "scaled_srelu": "ScaledSReLU",
    "scaled_clamped_qgeglu": "ScaledClampedQGeGLU",
}

# Scaled variants take a per-row scale extra input (x, scale) -> out.
_SCALED_OP_NAMES = {"scaled_swiglu", "scaled_srelu", "scaled_clamped_qgeglu"}


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _make_contract(
    op_name: str,
    x: torch.Tensor,
    output: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> Dict:
    contract = {
        "op": op_name,
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
    }
    if scale is not None:
        contract["scale_shape"] = list(scale.shape)
    return contract


def run_fused_activation_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward fusible-activation case using canonical ATK inputs.

    Input layout: ``x, dy, scale, op_name(str)``. ``dy`` matches the op output
    shape (gated ops halve the input's last dimension). ``scale`` is the per-row
    scale extra input for the scaled MoE variants (shape ``x.shape[:-1]``) and a
    placeholder ignored by non-scaled ops.
    """
    values = list(input_data.args)
    if len(values) != 4:
        raise ValueError(
            "fused_activation ATK case expects 4 positional inputs "
            "(x, dy, scale, op), got {}".format(len(values))
        )
    x, dy, scale, op_name = values
    op_name = str(op_name)
    if op_name not in _OP_CLASS_NAMES:
        raise ValueError(
            "Unsupported fused activation op '{}'; expected one of {}".format(
                op_name, sorted(_OP_CLASS_NAMES)
            )
        )

    from transformer_engine.pytorch import ops as te_ops

    x = x.detach().clone().requires_grad_(True)
    op = getattr(te_ops, _OP_CLASS_NAMES[op_name])()
    if op_name in _SCALED_OP_NAMES:
        scale = scale.detach().clone().requires_grad_(True)
        output = op(x, scale)
    else:
        scale = None
        output = op(x)
    loss = (output * dy).sum()
    loss.backward()

    contract = _make_contract(op_name, x, output, scale=scale)
    return (
        output,
        x.grad,
        _encode_contract(contract, output.device),
    )


@register("function_fused_activation")
class FunctionFusedActivation(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_fused_activation_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_fused_activation_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "fused_activation template has no CPU reference; only 'npu' "
                "execution is supported. NVTE consistency comes from the "
                "accuracy_load golden node, not from a local torch implementation. "
                "Got backend: {}".format(self.device)
            )
        return result if with_output else None
