# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine ``moe_permute_with_probs``.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of
``transformer_engine.pytorch.permutation.moe_permute_with_probs``. NVTE consistency comes
from the offline golden loop (``collect_nvte_golden.py`` + an ATK
``accuracy_load`` node), never from a local torch reference.

The generator ships the routing map as a compact ``routing_spec`` string
(``';'``-joined rows, ``','``-joined expert targets, empty segment = dropped
token) because the map values *are* the semantics: random tensors would break
the token distribution and the group order. This adapter rebuilds the
``routing_map`` tensor from that spec, identically on the NPU side and the NVTE
golden side, before calling the API.
"""

import json
from typing import Dict, Tuple

import torch

from atk.tasks.api_execute import register

# ATK's API plugin loader resolves both BaseApi symbols from this namespace.
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi  # noqa: F401
from atk.tasks.api_execute.base_api import BaseApi

import os
import sys

_comparators_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _comparators_dir not in sys.path:
    sys.path.insert(0, _comparators_dir)
from _comparators import register_module_comparators  # noqa: E402
from _moe_convert import _convert_row_id_map_to_npu  # noqa: E402

register_module_comparators("moe_permute_with_probs")

RESULT_SCHEMA = [
    "forward_output",
    "permuted_probs",
    "inp_grad",
    "row_id_map",
    "api_contract_json_utf8",
]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _build_routing_map(
    routing_spec: str,
    num_experts: int,
    routing_dtype: str,
    device: str,
) -> torch.Tensor:
    """Rebuild the ``routing_map`` tensor from the generator's compact spec.

    mask rows: expert targets become 1s in a ``[num_tokens, num_experts]``
               bool tensor; an empty row (dropped token) stays all-zero.
               bool is preferred because CUDA NVTE's own permute tests use
               it and the Ascend NPU routing-map operator also
               accepts it.
    """
    rows = routing_spec.split(";")
    num_tokens = len(rows)
    if routing_dtype == "bool":
        dtype = torch.bool
    elif routing_dtype == "int32":
        # Legacy int32 mask attr: the NPU routing-map operator accepts int8
        # (not int32) — its support list is [DT_INT8, DT_BOOL] — while the NVTE
        # golden side keeps int32; the 0/1 values are identical so the outputs
        # stay comparable. Same conversion as the sibling adapters
        # (function_moe_permute.py / function_mp_pad_probs.py /
        # function_moe_unpermute.py); this one was missing it, which the
        # 2026-09-20 generalization surfaced as 80/150 device failures
        # (aclnnMoeTokenPermuteWithRoutingMap: "Tensor routingMap not
        # implemented for DT_INT32").
        if device == "npu":
            dtype = torch.int8
        else:
            dtype = torch.int32
    else:
        raise ValueError(
            f"Got routing_dtype {routing_dtype}, but tensor routingMap "
            "must be in list [DT_int32, DT_BOOL]."
        )
    routing_map = torch.zeros((num_tokens, num_experts), dtype=dtype, device=device)
    for row_index, row in enumerate(rows):
        if not row:
            continue
        for target in (int(value) for value in row.split(",")):
            if not 0 <= target < num_experts:
                raise ValueError(
                    "mask target {} out of range [0, {}) at row {}".format(
                        target, num_experts, row_index
                    )
                )
            routing_map[row_index, target] = 1
    return routing_map


def _call_moe_permute_with_probs(
    module,
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call ``moe_permute_with_probs`` on the shared API surface."""
    return module.moe_permute_with_probs(inp, probs, routing_map, num_out_tokens)


def _make_contract(
    inp,
    probs,
    permuted,
    permuted_probs,
    row_id_map,
    routing_map,
    num_experts,
    topk,
    routing_dtype,
    num_out_tokens,
    dropped,
) -> Dict:
    return {
        "output_shape": list(permuted.shape),
        "output_dtype": str(permuted.dtype),
        "input_shape": list(inp.shape),
        "probs_shape": list(probs.shape),
        "probs_dtype": str(probs.dtype),
        "permuted_probs_shape": list(permuted_probs.shape),
        "permuted_probs_dtype": str(permuted_probs.dtype),
        "routing_map_shape": list(routing_map.shape),
        # Record the backend-agnostic attr, not str(routing_map.dtype): the NPU
        # kernel only accepts int8/bool masks, so the adapter downcasts the
        # legacy int32 mask on the NPU side while the golden side keeps int32 —
        # str(routing_map.dtype) would differ by backend and break the exact
        # contract comparison (same convention as the moe_permute module).
        "routing_map_dtype": routing_dtype,
        "row_id_map_shape": list(row_id_map.shape),
        "row_id_map_dtype": str(row_id_map.dtype),
        "num_experts": num_experts,
        "topk": topk,
        "routing_dtype": routing_dtype,
        "num_out_tokens": num_out_tokens,
        "dropped": dropped,
    }


def run_moe_permute_with_probs_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward ``moe_permute_with_probs`` case using canonical ATK inputs.

    Input layout: ``inp, probs, dy, dy_probs, routing_spec(str),
    num_experts(int), topk(int), routing_dtype(str), num_out_tokens(int)``.
    Only the device implementation is provided; numerical consistency with NVTE
    is established by the offline golden loop, never by a local torch reference.
    """
    values = list(input_data.args)
    if len(values) != 9:
        raise ValueError(
            "moe_permute_with_probs ATK case expects 9 positional inputs, got {}".format(
                len(values)
            )
        )
    (
        inp,
        probs,
        dy,
        dy_probs,
        routing_spec,
        num_experts,
        topk,
        routing_dtype,
        num_out_tokens,
    ) = values
    routing_spec = str(routing_spec)
    num_experts = int(num_experts)
    topk = int(topk)
    routing_dtype = str(routing_dtype)
    num_out_tokens = int(num_out_tokens)

    import transformer_engine.pytorch as te

    inp = inp.detach().clone().requires_grad_(True)
    probs = probs.detach().clone().requires_grad_(True)
    routing_map = _build_routing_map(routing_spec, num_experts, routing_dtype, device)
    permuted, permuted_probs, row_id_map = _call_moe_permute_with_probs(
        te, inp, probs, routing_map, num_out_tokens
    )
    if device != "npu":
        row_id_map = _convert_row_id_map_to_npu(
            row_id_map, num_tokens=inp.size(0), drop_and_pad=False
        )
    loss = (permuted * dy).sum() + (permuted_probs * dy_probs).sum()
    loss.backward()

    dropped = sum(1 for row in routing_spec.split(";") if not row)
    contract = _make_contract(
        inp,
        probs,
        permuted,
        permuted_probs,
        row_id_map,
        routing_map,
        num_experts,
        topk,
        routing_dtype,
        num_out_tokens,
        dropped,
    )
    return (
        permuted,
        permuted_probs,
        inp.grad,
        row_id_map,
        _encode_contract(contract, permuted.device),
    )


@register("function_moe_permute_with_probs")
class FunctionMoePermuteWithProbs(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_moe_permute_with_probs_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_moe_permute_with_probs_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "moe_permute_with_probs template has no CPU reference; only 'npu' execution "
                "is supported. NVTE consistency comes from the accuracy_load golden "
                "node, not from a local torch implementation. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
