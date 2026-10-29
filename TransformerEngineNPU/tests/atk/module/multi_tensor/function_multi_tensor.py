# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for TE multi-tensor ops.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of the
``transformer_engine.pytorch.optimizers`` multi-tensor helpers
(``multi_tensor_scale`` / ``multi_tensor_scale_tensor`` / ``multi_tensor_l2norm``).
NVTE consistency comes from the offline golden loop.

Scope decision: NVTE implements these as fused CUDA kernels (``tex.multi_tensor_*``),
while TENPU uses pure-torch per-tensor loops. ``multi_tensor_l2norm`` is the
precision-relevant one (cross-tensor L2 reduction); ``multi_tensor_scale`` /
``multi_tensor_scale_tensor`` are trivial copy+scale (fp32 exact) — the guard
verifies their API surface, in-place semantics and overflow flag. The fused-vs-
loop performance gap is a P3 (perf/memory) item, not covered here.
"""

import json
from typing import Dict, Tuple

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

register_module_comparators("multi_tensor")

RESULT_SCHEMA = [
    "primary",  # scale: out[0]; l2norm: total_norm (fp32) — tolerance compare
    "secondary",  # scale: out[1]; l2norm: per_tensor_norms (fp32/empty)
    "tertiary",  # scale: out[2]; l2norm: empty
    "overflow_flag",  # is_infinite / noop_flag (int32) — exact compare
    "api_contract_json_utf8",
]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def run_multi_tensor_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one multi-tensor op on 3 canonical fp32 tensors and return the results.

    Input layout: ``t0, t1, t2, op(str), scale(float), per_tensor(bool),
    chunk_size(int)``. ``op`` in {"scale", "scale_tensor", "l2norm"}. For scale
    ops the adapter builds the ``[[in...], [out...]]`` lists with zero outputs and
    compares the post-scale outputs; for l2norm it returns the total norm and the
    per-tensor norms (empty when ``per_tensor`` is False).
    """
    values = list(input_data.args)
    if len(values) != 7:
        raise ValueError(
            "multi_tensor ATK case expects 7 positional inputs "
            "(t0, t1, t2, op, scale, per_tensor, chunk_size), got {}".format(len(values))
        )
    t0, t1, t2, op, scale, per_tensor, chunk_size = values
    op = str(op)
    scale = float(scale)
    per_tensor = bool(per_tensor)
    chunk_size = int(chunk_size)

    from transformer_engine.pytorch.optimizers import (
        multi_tensor_l2norm,
        multi_tensor_scale,
        multi_tensor_scale_tensor,
    )

    device = t0.device
    empty = torch.zeros(0, dtype=torch.float32, device=device)
    overflow = torch.zeros(1, dtype=torch.int32, device=device)

    if op in ("scale", "scale_tensor"):
        in_list = [t.detach().clone() for t in (t0, t1, t2)]
        out_list = [torch.zeros_like(t) for t in in_list]
        if op == "scale":
            multi_tensor_scale(chunk_size, overflow, [in_list, out_list], scale)
        else:
            scale_tensor = torch.tensor([scale], dtype=torch.float32, device=device)
            multi_tensor_scale_tensor(chunk_size, overflow, [in_list, out_list], scale_tensor)
        primary, secondary, tertiary = out_list
    elif op == "l2norm":
        grad_list = [t.detach().clone() for t in (t0, t1, t2)]
        # NVTE's pybind binding accepts positional args only (no py::arg names),
        # so per_tensor must be passed positionally; TENPU's python impl accepts it too.
        total_norm, per_norms = multi_tensor_l2norm(chunk_size, overflow, [grad_list], per_tensor)
        primary = total_norm
        secondary = per_norms if per_norms is not None else empty
        tertiary = empty
    else:
        raise ValueError(
            "unsupported multi_tensor op: {} (want scale/scale_tensor/l2norm)".format(op)
        )

    contract = {
        "op": op,
        "chunk_size": chunk_size,
        "scale": scale,
        "per_tensor": per_tensor,
        "tensor_count": 3,
        "tensor_shape": list(t0.shape),
        "tensor_dtype": str(t0.dtype),
    }
    return (primary, secondary, tertiary, overflow, _encode_contract(contract, device))


# api_type must NOT contain the substrings "tensor" or "method": ATK's
# get_input_file_path / load_dataset match `"tensor" in api_type` to decide
# whether a tensor_input.bin / method_input.bin is expected. "function_multi_tensor"
# contains "tensor", which made ATK demand a tensor_input.bin that does not exist.
@register("function_mt")
class FunctionMultiTensor(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts and
    never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_multi_tensor_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_multi_tensor_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "multi_tensor template has no CPU reference; only 'npu' execution is "
                "supported. NVTE consistency comes from the accuracy_load golden node, "
                "not from a local torch implementation. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
