# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine ``moe_unpermute``.

``moe_unpermute`` reverses ``moe_permute`` (optionally merging the routed
copies with their probabilities and removing per-expert padding). The public
API dispatches over four branches:

1. **index**: ``map_type="index"`` without ``merging_probs``
   (``moe_permute`` -> ``moe_unpermute``).
2. **mask**: ``map_type="mask"`` without ``merging_probs``
   (``moe_permute`` -> ``moe_unpermute``).
3. **mask with probs**: ``map_type="mask"`` with ``merging_probs``
   (``moe_permute_with_probs`` -> ``moe_unpermute``, feeding the permuted
   tokens and the original ``[num_tokens, num_experts]`` routing
   probabilities into ``moe_unpermute`` as ``merging_probs``).
4. **mask with probs and pad**: ``map_type="mask"`` with ``merging_probs``
   and ``pad_offsets`` (``moe_permute_and_pad_with_probs`` ->
   ``moe_unpermute``, again with the original routing probabilities as
   ``merging_probs``). The permute side of this branch is routed through the
   CPU baseline ``mp_pad_probs_cpu`` whenever the routing map lives on CPU,
   mirroring ``function_mp_pad_probs.py``; on device tensors it calls the
   fused pad+permute API instead.

``merging_probs`` must be the original ``[num_tokens, num_experts]`` routing
probabilities on **both** backends: NVTE's ``unpermute_with_mask_map`` reads
``merging_probs.stride(1)`` (2-D layout, see its own permutation tests, which
pass the un-permuted ``probs`` tensor) and TENPU's
``_npu_moe_token_unpermute_with_routing_map`` documents/uses the same shape.
The 1-D ``permuted_probs`` returned by the permute APIs is only an
intermediate artifact and is never fed to ``moe_unpermute``.

The generator ships the routing map as a compact ``routing_spec`` string
(``';'``-joined rows, ``','``-joined expert targets, empty segment = dropped
token) because the map values *are* the semantics: random tensors would break
the token distribution and the group order. This adapter rebuilds the
``routing_map`` tensor from that spec, identically on the NPU side and the NVTE
golden side, before calling the APIs.

On the NPU side, ``moe_unpermute`` is device-only for every branch (the mask
path calls the Ascend ``_npu_moe_token_unpermute_with_routing_map`` kernel and
the index path calls ``npu_moe_token_unpermute``). The golden strategy
follows the permute side of each case:

- ``permute`` (mask/index) and ``with_probs`` modes follow the
  **no-CPU-reference** template: NVTE consistency comes from the offline
  golden loop (``collect_nvte_golden.py`` + an ATK ``accuracy_load`` node),
  never from a local torch reference.
- ``pad_with_probs`` mode follows the **CPU-reference** template: on the
  golden side the tensors are moved to CPU and both ``mp_pad_probs_cpu``
  (permute) and ``moe_unpermute_pad_probs_cpu`` (unpermute) from
  ``../_cpu_baselines.py`` are used, because ``inp.grad`` accumulates on the
  NPU/CPU path and cannot be converted to the CUDA-version layout.
"""

import inspect
import json
from typing import Dict, Optional, Tuple

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
from _cpu_baselines import mp_pad_probs_cpu, moe_unpermute_pad_probs_cpu  # noqa: E402
from _moe_convert import _convert_row_id_map_to_npu  # noqa: E402
from _moe_convert import _suppress_padded_permuted_to_npu  # noqa: E402

register_module_comparators("moe_unpermute")

RESULT_SCHEMA = [
    "unpermuted_output",
    "input_grad",
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
               bool (or int32) tensor; an empty row (dropped token) stays
               all-zero. bool is preferred because CUDA NVTE's own permute
               tests use it and the Ascend NPU routing-map operator also
               accepts it.
    index rows: the expert targets are copied verbatim into an int32
               ``[num_tokens, topk]`` tensor (the only supported
               ``routing_dtype`` for index maps); empty rows are invalid
               because index rows are fixed-length.
    """
    rows = routing_spec.split(";")
    num_tokens = len(rows)
    if map_type == "mask":
        if routing_dtype == "bool":
            dtype = torch.bool
        elif routing_dtype == "int32":
            # Legacy int32 mask attr: the NPU routing-map operator accepts int8
            # (not int32) — its support list is [DT_INT8, DT_BOOL] — while the
            # NVTE golden side keeps int32; the 0/1 values are identical so the
            # outputs stay comparable. Same conversion as function_moe_permute.py;
            # this adapter was missing it (the 2026-09-20 generalization found
            # the identical gap in function_mp_with_probs.py and
            # function_mp_pad_probs.py).
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
    if map_type == "index":
        if routing_dtype != "int32":
            raise ValueError(
                f"Got routing_dtype {routing_dtype}, but tensor routingMap must be DT_int32"
            )
        routing_map = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
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
            routing_map[row_index] = torch.tensor(targets, dtype=torch.int32, device=device)
        return routing_map
    raise ValueError("map_type should be one of 'mask' or 'index', got '{}'".format(map_type))


def _build_tokens_per_expert(
    tokens_per_expert_spec: str,
    num_experts: int,
    device: str,
) -> torch.Tensor:
    """Rebuild the ``[num_experts]`` tokens_per_expert tensor from the generator's spec.

    The generator derives per-expert routed token counts from the same
    owner+topk rule that builds ``routing_spec`` and ships them as a
    comma-joined string; the adapter parses them back into an int32 tensor on
    the target device. Values must be non-negative and match num_experts.
    """
    parts = tokens_per_expert_spec.split(",")
    if len(parts) != num_experts:
        raise ValueError(
            "tokens_per_expert spec has {} values, expected {} experts: {!r}".format(
                len(parts), num_experts, tokens_per_expert_spec
            )
        )
    counts = []
    for part in parts:
        value = int(part)
        if value < 0:
            raise ValueError("tokens_per_expert values must be non-negative, got {}".format(value))
        counts.append(value)
    return torch.tensor(counts, dtype=torch.int32, device=device)


def _call_moe_permute(
    module,
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    max_token_num: int,
    map_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Call ``moe_permute`` on the shared API surface.

    ``max_token_num`` is a TENPU workspace hint that upstream NVTE
    ``moe_permute`` may not declare; it is passed only when the imported
    implementation accepts it. The hint does not change output semantics, so
    the golden comparison stays valid across both backends.
    """
    fn = module.moe_permute
    try:
        accepts_max_token_num = "max_token_num" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        accepts_max_token_num = False
    if accepts_max_token_num:
        return fn(inp, routing_map, num_out_tokens, max_token_num, map_type)
    return fn(inp, routing_map, num_out_tokens, map_type=map_type)


def _call_moe_permute_with_probs(
    module,
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Call ``moe_permute_with_probs`` on the shared API surface."""
    return module.moe_permute_with_probs(inp, probs, routing_map, num_out_tokens)


def _call_mp_pad_probs(
    module,
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    align_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Call ``moe_permute_and_pad_with_probs`` on the shared API surface.

    The pad branch reuses the CPU baseline ``mp_pad_probs_cpu`` whenever the
    routing map lives on CPU (see ``function_mp_pad_probs.py``). The CPU
    baseline emits the row_id_map and padded tokens directly in the NPU-native
    layout, so no NVTE->NPU layout conversion is needed on that path. Device
    tensors (NPU under test, CUDA golden) call the fused pad+permute API.
    """
    if routing_map.is_cpu:
        return mp_pad_probs_cpu(inp, probs, routing_map, tokens_per_expert, align_size)
    return module.moe_permute_and_pad_with_probs(
        inp, probs, routing_map, tokens_per_expert, align_size
    )


def _call_moe_unpermute(
    module,
    permuted: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    restore_shape: torch.Size,
    map_type: str,
    pad_offsets: Optional[torch.Tensor],
    routing_map: Optional[torch.Tensor],
) -> torch.Tensor:
    """Call ``moe_unpermute`` on the shared API surface.

    ``pad_offsets`` and ``routing_map`` are TENPU-specific knobs; upstream
    NVTE ``moe_unpermute`` only declares ``pad_offsets`` (and not
    ``routing_map``). Each is passed only when the imported implementation
    accepts it, so the same execution path serves the NPU side and the CUDA
    golden side.
    """
    if permuted.is_cpu:
        fn = moe_unpermute_pad_probs_cpu
    else:
        fn = module.moe_unpermute
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = {
        "merging_probs": merging_probs,
        "restore_shape": restore_shape,
        "map_type": map_type,
    }
    if pad_offsets is not None and "pad_offsets" in params:
        kwargs["pad_offsets"] = pad_offsets
    if routing_map is not None and "routing_map" in params:
        kwargs["routing_map"] = routing_map
    return fn(permuted, row_id_map, **kwargs)


def _make_contract(
    inp,
    probs,
    unpermuted,
    permuted,
    row_id_map,
    routing_map,
    mode,
    map_type,
    num_experts,
    topk,
    routing_dtype,
    num_out_tokens,
    pad_offsets,
    dropped,
) -> Dict:
    return {
        "output_shape": list(unpermuted.shape),
        "output_dtype": str(unpermuted.dtype),
        "input_shape": list(inp.shape),
        "probs_shape": list(probs.shape) if probs is not None else None,
        "probs_dtype": str(probs.dtype) if probs is not None else None,
        "permuted_shape": list(permuted.shape),
        "permuted_dtype": str(permuted.dtype),
        "routing_map_shape": list(routing_map.shape),
        # Record the backend-agnostic attr, not str(routing_map.dtype): the NPU
        # kernel only accepts int8/bool masks, so the adapter downcasts the
        # legacy int32 mask on the NPU side while the golden side keeps int32 —
        # str(routing_map.dtype) would differ by backend and break the exact
        # contract comparison (same convention as the moe_permute module).
        "routing_map_dtype": routing_dtype,
        "row_id_map_shape": list(row_id_map.shape),
        "row_id_map_dtype": str(row_id_map.dtype),
        "mode": mode,
        "map_type": map_type,
        "num_experts": num_experts,
        "topk": topk,
        "routing_dtype": routing_dtype,
        "num_out_tokens": num_out_tokens,
        "pad_offsets_shape": (
            list(pad_offsets.shape) if isinstance(pad_offsets, torch.Tensor) else None
        ),
        "dropped": dropped,
    }


def run_moe_unpermute_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward ``moe_unpermute`` case using canonical ATK inputs.

    The input layout depends on the ``mode`` attr, which the generator places
    at the third positional input for ``permute`` and at the fourth positional
    input for ``with_probs`` / ``pad_with_probs`` (because those modes carry
    an extra ``probs`` tensor before ``dy``):

    - ``permute``: ``inp, dy, mode, map_type(str), routing_spec(str),
      num_experts(int), topk(int), routing_dtype(str), num_out_tokens(int),
      max_token_num(int)``.
    - ``with_probs``: ``inp, probs, dy, mode, routing_spec(str),
      num_experts(int), topk(int), routing_dtype(str), num_out_tokens(int)``.
    - ``pad_with_probs``: ``inp, probs, dy, mode, routing_spec(str),
      num_experts(int), topk(int), routing_dtype(str),
      tokens_per_expert_spec(str), align_size(int)``.

    Every mode runs the matching permute API first. The pad mode zeros unused
    fixed-capacity slots before ``moe_unpermute`` and explicitly retains the
    drop-and-pad mode, even when alignment adds no rows. It seeds backward with
    ``loss = (unpermuted * dy).sum()`` and returns the unpermuted tokens, the
    input gradient and the serialized API contract. The unpermute kernels (and,
    through the autograd chain, the permute kernels) are all exercised.
    """
    values = list(input_data.args)
    mode = None
    for mode_index in (2, 3):
        if len(values) > mode_index and str(values[mode_index]) in (
            "permute",
            "with_probs",
            "pad_with_probs",
        ):
            mode = str(values[mode_index])
            break
    if mode is None:
        raise ValueError(
            "moe_unpermute ATK case is missing a valid mode attr; got {} positional inputs".format(
                len(values)
            )
        )

    import transformer_engine.pytorch as te

    if mode == "permute":
        if len(values) != 10:
            raise ValueError(
                "moe_unpermute permute-mode ATK case expects 10 positional inputs, got {}".format(
                    len(values)
                )
            )
        (
            inp,
            dy,
            _mode,
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
        probs = None

        inp = inp.detach().clone().requires_grad_(True)
        routing_map = _build_routing_map(
            routing_spec, map_type, num_experts, topk, routing_dtype, device
        )
        permuted, row_id_map = _call_moe_permute(
            te, inp, routing_map, num_out_tokens, max_token_num, map_type
        )
        unpermuted = _call_moe_unpermute(
            te, permuted, row_id_map, None, inp.shape, map_type, None, None
        )
        pad_offsets = None
    elif mode == "with_probs":
        if len(values) != 9:
            raise ValueError(
                "moe_unpermute with_probs-mode ATK case expects 9 positional inputs, got {}".format(
                    len(values)
                )
            )
        (
            inp,
            probs,
            dy,
            _mode,
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
        map_type = "mask"

        inp = inp.detach().clone().requires_grad_(True)
        probs = probs.detach().clone().requires_grad_(True)
        routing_map = _build_routing_map(
            routing_spec, "mask", num_experts, topk, routing_dtype, device
        )
        permuted, permuted_probs, row_id_map = _call_moe_permute_with_probs(
            te, inp, probs, routing_map, num_out_tokens
        )
        unpermuted = _call_moe_unpermute(
            te, permuted, row_id_map, probs, inp.shape, "mask", None, routing_map
        )
        pad_offsets = None
    elif mode == "pad_with_probs":
        if len(values) != 10:
            raise ValueError(
                "moe_unpermute pad_with_probs-mode ATK case expects 10 "
                "positional inputs, got {}".format(len(values))
            )
        (
            inp,
            probs,
            dy,
            _mode,
            routing_spec,
            num_experts,
            topk,
            routing_dtype,
            tokens_per_expert_spec,
            align_size,
        ) = values
        # Force using cpu baseline
        if device != "npu":
            device = "cpu"
            inp = inp.cpu()
            probs = probs.cpu()
            dy = dy.cpu()
        routing_spec = str(routing_spec)
        num_experts = int(num_experts)
        topk = int(topk)
        routing_dtype = str(routing_dtype)
        tokens_per_expert_spec = str(tokens_per_expert_spec)
        align_size = int(align_size)
        map_type = "mask"

        inp = inp.detach().clone().requires_grad_(True)
        probs = probs.detach().clone().requires_grad_(True)
        routing_map = _build_routing_map(
            routing_spec, "mask", num_experts, topk, routing_dtype, device
        )
        tokens_per_expert = _build_tokens_per_expert(tokens_per_expert_spec, num_experts, device)
        # The pad branch reuses the CPU baseline for the permute side when the
        # routing map is a CPU tensor (mirroring function_mp_pad_probs.py);
        # device tensors run the fused pad+permute API.
        permuted, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert = (
            _call_mp_pad_probs(te, inp, probs, routing_map, tokens_per_expert, align_size)
        )
        # Both NPU and the CPU fallback use fixed-capacity expert blocks.
        # Unused slots may gather unrelated real tokens; exclude those slots
        # from the weighted merge and its gradient without mutating native
        # outputs. CUDA-labelled pad cases have already moved to CPU above.
        capacity = int(target_tokens_per_expert.sum()) // num_experts
        permuted = _suppress_padded_permuted_to_npu(
            permuted, capacity, align_size, tokens_per_expert, nvte_layout=False
        )
        # The preceding permute always used drop_and_pad=True. None means
        # alignment added no rows, not that its map has the non-padded layout.
        # Keep unpermute in the matching mode, including unequal expert loads
        # and the CPU baseline's all-zero offsets for last-expert-only padding.
        if pad_offsets is None:
            pad_offsets = torch.zeros(num_experts, dtype=torch.int64, device=permuted.device)
        unpermuted = _call_moe_unpermute(
            te,
            permuted,
            row_id_map,
            probs,
            inp.shape,
            "mask",
            pad_offsets,
            routing_map,
        )
    else:
        raise ValueError("unknown mode {}".format(mode))

    loss = (unpermuted * dy).sum()
    loss.backward()

    dropped = sum(1 for row in routing_spec.split(";") if not row)
    num_out_tokens = int(permuted.size(0))

    if device != "npu" and mode != "pad_with_probs" and map_type == "mask":
        row_id_map = _convert_row_id_map_to_npu(
            row_id_map, num_tokens=inp.size(0), drop_and_pad=False
        )

    contract = _make_contract(
        inp,
        probs,
        unpermuted,
        permuted,
        row_id_map,
        routing_map,
        mode,
        map_type,
        num_experts,
        topk,
        routing_dtype,
        num_out_tokens,
        pad_offsets,
        dropped,
    )
    return (
        unpermuted,
        inp.grad,
        _encode_contract(contract, unpermuted.device),
    )


@register("function_moe_unpermute")
class FunctionMoeUnpermute(BaseApi):
    """ATK module adapter that executes ``moe_unpermute`` only on the NPU backend.

    The unpermute kernels (and the permute APIs that feed them) run on Ascend
    NPU during the ATK task; the ``cpu_nvte`` accuracy-load node feeds the
    offline golden artifacts collected by ``collect_nvte_golden.py`` and never
    instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_moe_unpermute_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_moe_unpermute_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "moe_unpermute ATK execution supports only the 'npu' backend; "
                "the reference comes from the offline accuracy_load golden "
                "node (NVTE on CUDA, or the CPU baseline for pad_with_probs "
                "cases). Got backend: {}".format(self.device)
            )
        return result if with_output else None
