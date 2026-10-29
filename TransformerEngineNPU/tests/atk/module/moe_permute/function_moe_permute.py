# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine ``moe_permute``.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of
``transformer_engine.pytorch.permutation.moe_permute``. NVTE consistency comes
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

# ATK's API plugin loader resolves the BaseApi symbols from this namespace.
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
from _moe_convert import _convert_row_id_map_to_npu  # noqa: E402

register_module_comparators("moe_permute")

RESULT_SCHEMA = [
    "forward_output",
    "inp_grad",
    "row_id_map",
    "api_contract_json_utf8",
]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _build_routing_map(
    routing_spec: str,
    map_type: str,
    num_experts: int,
    topk: int,
    routing_dtype: str,
    device: str,
) -> torch.Tensor:
    """Rebuild the ``routing_map`` tensor from the generator's compact spec.

    mask rows: expert targets become 1s in a ``[num_tokens, num_experts]``
               bool tensor by default, or an integer tensor for the legacy
               ``routing_dtype="int32"`` attr — int8 on the NPU side (the
               dtype accepted by the Ascend routing-map operator), int32 on
               the NVTE golden side. An empty row (dropped token) stays
               all-zero; the 0/1 values are identical across backends, so
               permuted outputs and ``row_id_map`` remain comparable.
    index rows: the expert targets are copied verbatim into a
               ``[num_tokens, topk]`` tensor of the requested ``routing_dtype``;
               empty rows are invalid because index rows are fixed-length.
    """
    rows = routing_spec.split(";")
    num_tokens = len(rows)
    if map_type == "mask":
        if routing_dtype == "bool":
            dtype = torch.bool
        elif routing_dtype == "int32":
            # Legacy int32 mask attr: the NPU routing-map operator accepts
            # int8 (not int32), while the NVTE golden side keeps int32; the
            # 0/1 values are identical so the outputs stay comparable.
            if device == "npu":
                dtype = torch.int8
            else:
                dtype = torch.int32
        else:
            raise ValueError(
                f"Got routing_dtype {routing_dtype}, but tensor routingMap must be in list [DT_int32,DT_BOOL,]"
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
    if map_type == "index":
        if routing_dtype == "int32":
            dtype = torch.int32
        else:
            raise ValueError(
                f"Got routing_dtype {routing_dtype}, but tensor routingMap must be DT_int32"
            )
        routing_map = torch.empty((num_tokens, topk), dtype=dtype, device=device)
        for row_index, row in enumerate(rows):
            if not row:
                raise ValueError(
                    "index routing_map has fixed per-row length; empty row at {}".format(row_index)
                )
            targets = [int(value) for value in row.split(",")]
            if len(targets) != topk:
                raise ValueError(
                    "index row {} has {} targets, expected topk={}".format(
                        row_index, len(targets), topk
                    )
                )
            if any(not 0 <= value < num_experts for value in targets):
                raise ValueError(
                    "index target out of range [0, {}) at row {}".format(num_experts, row_index)
                )
            routing_map[row_index] = torch.tensor(targets, dtype=dtype, device=device)
        return routing_map
    raise ValueError("map_type should be one of 'mask' or 'index', got '{}'".format(map_type))


def _call_moe_permute(
    module,
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    max_token_num: int,
    map_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is a reserved interface in case the inputs are different
    """
    fn = module.moe_permute

    return fn(inp, routing_map, num_out_tokens, max_token_num, map_type)


def _make_contract(
    inp,
    permuted,
    row_id_map,
    routing_map,
    map_type,
    num_experts,
    topk,
    routing_dtype,
    num_out_tokens,
    max_token_num,
    dropped,
) -> Dict:
    return {
        "output_shape": list(permuted.shape),
        "output_dtype": str(permuted.dtype),
        "input_shape": list(inp.shape),
        "routing_map_shape": list(routing_map.shape),
        # Record the requested case attr, not the backend-normalized tensor
        # dtype: the int32 mask attr is built as int8 on NPU and int32 on the
        # CUDA golden side, so the contract must stay identical across backends.
        "routing_map_dtype": routing_dtype,
        "row_id_map_shape": list(row_id_map.shape),
        "row_id_map_dtype": str(row_id_map.dtype),
        "map_type": map_type,
        "num_experts": num_experts,
        "topk": topk,
        "routing_dtype": routing_dtype,
        "num_out_tokens": num_out_tokens,
        "max_token_num": max_token_num,
        "dropped": dropped,
    }


def run_moe_permute_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward ``moe_permute`` case using canonical ATK inputs.

    Input layout: ``inp, dy, map_type(str), routing_spec(str), num_experts(int),
    topk(int), routing_dtype(str), num_out_tokens(int), max_token_num(int)``.
    Only the device implementation is provided; numerical consistency with NVTE
    is established by the offline golden loop, never by a local torch reference.
    """
    values = list(input_data.args)
    if len(values) != 9:
        raise ValueError(
            "moe_permute ATK case expects 9 positional inputs, got {}".format(len(values))
        )
    (
        inp,
        dy,
        map_type,
        routing_spec,
        num_experts,
        topk,
        routing_dtype,
        num_out_tokens,
        max_token_num,
    ) = values
    map_type = str(map_type)
    routing_spec = str(routing_spec)
    num_experts = int(num_experts)
    topk = int(topk)
    routing_dtype = str(routing_dtype)
    num_out_tokens = int(num_out_tokens)
    max_token_num = int(max_token_num)

    import transformer_engine.pytorch as te

    inp = inp.detach().clone().requires_grad_(True)
    routing_map = _build_routing_map(
        routing_spec, map_type, num_experts, topk, routing_dtype, device
    )
    permuted, row_id_map = _call_moe_permute(
        te, inp, routing_map, num_out_tokens, max_token_num, map_type
    )
    if device != "npu":
        if map_type == "mask":
            row_id_map = _convert_row_id_map_to_npu(
                row_id_map, num_tokens=inp.size(0), drop_and_pad=False
            )
        else:
            # NVTE index-mode row_id_map is laid out topK-major
            # (row_id_map[k * num_tokens + token], see NVTE permutation.cu),
            # while the NPU sorted_indices is token-major ([num_tokens, topK]
            # flattened). Transpose the golden map before the exact comparison;
            # the layouts coincide at topk == 1, which is why only topk >= 2
            # index cases failed output_2.
            num_tokens = inp.size(0)
            row_id_map = row_id_map.view(topk, num_tokens).t().contiguous().view(-1)
    loss = (permuted * dy).sum()
    loss.backward()

    dropped = sum(1 for row in routing_spec.split(";") if not row)
    contract = _make_contract(
        inp,
        permuted,
        row_id_map,
        routing_map,
        map_type,
        num_experts,
        topk,
        routing_dtype,
        num_out_tokens,
        max_token_num,
        dropped,
    )
    return (
        permuted,
        inp.grad,
        row_id_map,
        _encode_contract(contract, permuted.device),
    )


@register("function_moe_permute")
class FunctionMoePermute(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_moe_permute_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_moe_permute_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "moe_permute template has no CPU reference; only 'npu' execution "
                "is supported. NVTE consistency comes from the accuracy_load golden "
                "node, not from a local torch implementation. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
