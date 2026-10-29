# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine ``moe_sort_chunks_by_index_with_probs``;
accuracy comparators are registered from ``../_comparators.py``.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of
``transformer_engine.pytorch.permutation.moe_sort_chunks_by_index_with_probs``.
NVTE consistency comes from the offline golden loop (``collect_nvte_golden.py``
+ an ATK ``accuracy_load`` node), never from a local torch reference.

The generator ships the split sizes and the chunk permutation as compact attr
strings (``split_spec`` / ``sorted_spec``) because those values *are* the
semantics: random tensors would produce invalid chunk permutations and
meaningless chunk boundaries. This adapter rebuilds the int32 tensors from
those specs, identically on the NPU side and the NVTE golden side, before
calling the API. The per-token ``probs`` tensor is permuted with the same
row_id_map as ``inp``; the backward seed tensors ``dy`` / ``dy_probs`` drive
``inp.grad`` and ``probs.grad``.
"""

import json
from typing import Dict, Tuple

import torch

from atk.tasks.api_execute import register

# ATK's API plugin loader resolves BaseApi symbols from this namespace.
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi  # noqa: F401
from atk.tasks.api_execute.base_api import BaseApi

import os
import sys

_comparators_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _comparators_dir not in sys.path:
    sys.path.insert(0, _comparators_dir)
from _comparators import register_module_comparators  # noqa: E402

register_module_comparators("moe_sort_chunks_by_index_with_probs")

RESULT_SCHEMA = [
    "forward_output",
    "permuted_probs",
    "input_grad",
    "probs_grad",
    "api_contract_json_utf8",
]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _parse_spec(split_spec: str, sorted_spec: str) -> Tuple[list, list]:
    """Parse the generator's compact attr specs into chunk sizes and order."""
    split_sizes = [int(value) for value in split_spec.split(",")]
    sorted_index = [int(value) for value in sorted_spec.split(",")]
    if len(split_sizes) != len(sorted_index):
        raise ValueError(
            "split_spec ({} chunks) and sorted_spec ({} chunks) must describe "
            "the same number of chunks".format(len(split_sizes), len(sorted_index))
        )
    return split_sizes, sorted_index


def _build_index_tensors(
    split_sizes: list,
    sorted_index: list,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the int32 split_sizes/sorted_index tensors from the specs."""
    return (
        torch.tensor(split_sizes, dtype=torch.int32, device=device),
        torch.tensor(sorted_index, dtype=torch.int32, device=device),
    )


def _call_moe_sort_chunks_by_index_with_probs(
    module,
    inp: torch.Tensor,
    probs: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Call ``moe_sort_chunks_by_index_with_probs`` on the shared API surface."""
    return module.moe_sort_chunks_by_index_with_probs(inp, probs, split_sizes, sorted_index)


def _make_contract(inp, probs, output, permuted_probs, split_sizes, sorted_index) -> Dict:
    return {
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "input_shape": list(inp.shape),
        "input_dtype": str(inp.dtype),
        "probs_shape": list(probs.shape),
        "probs_dtype": str(probs.dtype),
        "permuted_probs_shape": list(permuted_probs.shape),
        "permuted_probs_dtype": str(permuted_probs.dtype),
        "split_sizes_shape": list(split_sizes.shape),
        "split_sizes_dtype": str(split_sizes.dtype),
        "sorted_index_shape": list(sorted_index.shape),
        "sorted_index_dtype": str(sorted_index.dtype),
        "num_splits": int(split_sizes.size(0)),
        "sorted_index_values": sorted_index.tolist(),
    }


def run_moe_sort_chunks_by_index_with_probs_case(
    input_data, device: str
) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward case using canonical ATK inputs.

    Input layout: ``inp, probs, dy, dy_probs, split_spec(str), sorted_spec(str)``.
    ``split_spec`` is the ','-joined chunk sizes and ``sorted_spec`` the
    ','-joined chunk permutation (output chunk i = input chunk
    ``sorted_spec[i]``); ``probs`` is a per-token tensor permuted with the same
    row_id_map as ``inp``. Only the device implementation is provided;
    numerical consistency with NVTE is established by the offline golden loop,
    never by a local torch reference.
    """
    values = list(input_data.args)
    if len(values) != 6:
        raise ValueError(
            "moe_sort_chunks_by_index_with_probs ATK case expects 6 positional "
            "inputs, got {}".format(len(values))
        )
    inp, probs, dy, dy_probs, split_spec, sorted_spec = values
    split_spec = str(split_spec)
    sorted_spec = str(sorted_spec)

    import transformer_engine.pytorch as te

    inp = inp.detach().clone().requires_grad_(True)
    probs = probs.detach().clone().requires_grad_(True)
    split_sizes, sorted_index = _parse_spec(split_spec, sorted_spec)
    split_sizes, sorted_index = _build_index_tensors(split_sizes, sorted_index, device)
    output, permuted_probs = _call_moe_sort_chunks_by_index_with_probs(
        te, inp, probs, split_sizes, sorted_index
    )
    loss = (output * dy).sum() + (permuted_probs * dy_probs).sum()
    loss.backward()

    contract = _make_contract(inp, probs, output, permuted_probs, split_sizes, sorted_index)
    return (
        output,
        permuted_probs,
        inp.grad,
        probs.grad,
        _encode_contract(contract, output.device),
    )


@register("function_moe_sort_chunks_by_index_with_probs")
class FunctionMoeSortChunksByIndexWithProbs(BaseApi):
    """ATK module adapter with a device-only implementation.

    Only the ``npu`` backend is supported; the ``cpu`` backend is deliberately
    unsupported because this template ships no CPU reference (see the module
    README for the no-CPU-reference design).
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_moe_sort_chunks_by_index_with_probs_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_moe_sort_chunks_by_index_with_probs_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "moe_sort_chunks_by_index_with_probs template has no CPU "
                "reference; only 'npu' execution is supported. NVTE consistency "
                "comes from the accuracy_load golden node, not from a local "
                "torch implementation. Got backend: {}".format(self.device)
            )
        return result if with_output else None
