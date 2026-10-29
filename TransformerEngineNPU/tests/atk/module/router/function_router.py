# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for the TENPU MoE router APIs.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementations of
``transformer_engine.pytorch.router``. NVTE consistency comes from the offline
golden loop (``collect_nvte_golden.py`` + an ATK ``accuracy_load`` node).

Three case shapes share this module (dispatch on the first string attr):

- ``topk``:     ``fused_topk_with_score_function`` — logits dtype, attrs
                (topk:int, use_pre_softmax:bool, num_groups:int>0|0=None,
                group_topk:int>0|0=None, scaling via a fixed table).
- ``score``:    ``fused_compute_score_for_moe_aux_loss`` — fp32 scores.
- ``aux_loss``: ``fused_moe_aux_loss`` — scalar loss + grad_probs.

All three shapes emit exactly 4 outputs (schema slots): the primary output,
the secondary output (bool routing_map / scalar loss), the seeded gradient,
and the JSON contract as uint8.

Selection outputs are compared INDEX-FREE: when two implementations break an
exact top-k tie differently, the selected expert indices differ but the
multiset of selected scores is identical, so ``probs * routing_map`` sorted
row-wise matches bitwise where the raw outputs would be false positives.
The secondary ``routing_map`` slot carries that masked, row-sorted product
instead of the bool map itself (a sorted bool map would be a pure per-row
popcount pattern, carrying no information beyond the contract's popcount
fields; for the topk variant the product equals the already-sparse probs,
so it re-checks the primary output). The selected indices themselves are not
contractual — Megatron consumes probs/routing_map from a single
implementation. The ``score`` variant's ``scores`` are an elementwise
function of the logits and do not depend on the selection, so only its
``routing_map`` slot is canonicalized (to the masked, row-sorted scores).

grad_logits has no tie-invariant form (its values differ per expert), so
ties are removed at the source instead: fp32 selection inputs are
row-separated before the call (``_separate_rows``), enforcing a minimum
logit gap that maps to score gaps orders of magnitude above cross-device
rounding noise at every selection boundary. The transform uses only
bit-reproducible ops (correctly-rounded elementwise arithmetic, stable
sort, cummin, scatter) because the online mode runs this same adapter
independently on NPU and GPU — anything branching on computed scores would
round differently per device and fork the inputs. Rare residuals remain,
covered by the accuracy gate's abs-error backstop: bias-sum ties
(``s_i + b_i == s_j + b_j`` with ``s_i != s_j``, bias being independent of
logit separation), group-score sum collisions when
``topk // group_topk >= 2``, and sigmoid-saturation near-ties (harmless:
the score derivative there is ~0, so grad is ~0).

TENPU's router relies on autograd through small ops; the adapter seeds the
backward with the canonical ATK tensors so grad_logits / grad_probs enter the
golden comparison like any other output.
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
from _comparators import register_module_comparators  # noqa: E402  # noqa: E402

register_module_comparators("router")

RESULT_SCHEMA = [
    "primary_output",
    "secondary_output",
    "grad_output",
    "api_contract_json_utf8",
]

# scaling_factor table keyed by (score_function, bias_enabled): mirrors the
# Megatron router defaults (sigmoid+sampling uses 2.5, otherwise None).
_SCALING = {
    ("sigmoid", True): 2.5,
    ("sigmoid", False): None,
    ("softmax", False): None,
    ("softmax", True): None,
    ("sqrtsoftplus", False): None,
    ("sqrtsoftplus", True): 1.0,
}


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _popcount_invariants(routing_map: torch.Tensor) -> Tuple[int, int]:
    counts = routing_map.sum(dim=-1)
    return int(counts.min().item()), int(counts.max().item())


# Minimum per-row gap enforced between fp32 router logits before selection
# (see _separate_rows). 1e-4 keeps score-space gaps at hundreds of ULP for
# every supported score function in its active range (sqrtsoftplus' relative
# derivative bottoms out at 0.5, softmax's at ~1, sigmoid mid-range at ~0.5),
# far above the cross-device rounding noise that flips top-k order at 1-ULP
# gaps.
_LOGIT_SEPARATION = 1e-4


def _separate_rows(logits: torch.Tensor, h: float = _LOGIT_SEPARATION) -> torch.Tensor:
    """Enforce a minimum gap ``h`` between the values of each row.

    Near-tied selection scores (1-ULP gaps) legitimately break differently
    per backend — torch.topk and the NPU kernel pick different experts with
    equal justification — and grad_logits has no tie-invariant form. Pairwise
    logit separation removes the ambiguity at the source: every gap >= h maps
    to a score gap far above rounding noise, at every selection boundary
    (plain top-k, group ranking, masked experts) without needing to know
    where the boundary is.

    Online-mode constraint: NPU and GPU run this same adapter independently,
    so the transform must be BIT-REPRODUCIBLE across devices — only
    correctly-rounded elementwise ops, stable sort, cummin and scatter are
    allowed. stable=True is mandatory: exactly-equal logits must be assigned
    ladder slots in the same original order on both devices, or the scattered
    results fork.
    """
    ordered, order = torch.sort(logits, dim=-1, descending=True, stable=True)
    # Ladder v'[i] = min(v[i], v'[i-1] - h) = (cummin_j(v[j] + h*j))_i - h*i:
    # pushes each value down until it clears its larger neighbour by h. All
    # correctly-rounded fp32 ops -> identical bits on every device.
    steps = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype) * h
    ladder = torch.cummin(ordered + steps, dim=-1).values - steps
    return torch.empty_like(logits).scatter(-1, order, ladder)


def _prepare_logits(t0: torch.Tensor) -> torch.Tensor:
    """Detach/clone t0 and row-separate it when its selection can tie.

    fp32 selection cases get the separation; low-precision selection cases
    run dense (topk == E, forced by the hook) so no boundary exists to tie
    on, and fp64 is rejected by the API. aux_loss performs no selection and
    keeps the plain detach/clone path.
    """
    logits = t0.detach().clone()
    if logits.dtype == torch.float32:
        logits = _separate_rows(logits)
    return logits.requires_grad_(True)


def _canonical_rows(t: torch.Tensor) -> torch.Tensor:
    """Row-wise descending sort of ``t``, the sort indices discarded.

    Makes score outputs comparable without expert indices: an exact top-k tie
    lets torch.topk (NVTE golden) and the NPU kernel pick different experts,
    but the tied scores are equal, so the multiset of selected values — and
    therefore the sorted rows — is identical on both backends. Sorting is
    equality-preserving, so any value-level difference still lands in the
    accuracy gate; only expert positions are given up, which are not part of
    the API contract.
    """
    return torch.sort(t, dim=-1, descending=True).values


def _topk_contract(probs, routing_map, logits, topk) -> Dict:
    pc_min, pc_max = _popcount_invariants(routing_map)
    return {
        "probs_shape": list(probs.shape),
        "probs_dtype": str(probs.dtype),
        "logits_shape": list(logits.shape),
        "topk": int(topk),
        "map_row_popcount_min": pc_min,
        "map_row_popcount_max": pc_max,
    }


def _score_contract(routing_map, scores, logits, topk) -> Dict:
    pc_min, pc_max = _popcount_invariants(routing_map)
    return {
        "scores_shape": list(scores.shape),
        "scores_dtype": str(scores.dtype),
        "logits_shape": list(logits.shape),
        "topk": int(topk),
        "map_row_popcount_min": pc_min,
        "map_row_popcount_max": pc_max,
    }


def _aux_contract(loss, probs, tokens_per_expert, topk, coeff) -> Dict:
    return {
        "loss_dtype": str(loss.dtype),
        "probs_shape": list(probs.shape),
        "tokens_per_expert_shape": list(tokens_per_expert.shape),
        "tokens_per_expert_dtype": str(tokens_per_expert.dtype),
        "topk": int(topk),
        "coeff": float(coeff),
        "total_tokens": int(probs.shape[0]),
    }


def _synth_tokens_per_expert(seed: torch.Tensor, total: int, num_experts: int) -> torch.Tensor:
    """Deterministic per-expert token counts in [0, 2T/E].

    Real callers derive this from the routing map; deriving it from the
    canonical ATK seed keeps the case self-contained while producing the same
    value range. Detached: NVTE's backward returns no gradient for it.
    """
    base = seed.detach().reshape(-1)
    return (base.abs() * (2.0 * total / num_experts)).to(torch.int64).to(torch.float32)


def run_router_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one router case (forward + seeded backward) from canonical inputs.

    Input layout (9 positional): ``t0, t1, grad_seed0, grad_seed1, variant,
    score_function, a2, a3, a4``.

    Variant-specific roles:
      topk:     t0=logits, t1=expert_bias, a2=topk(int), a3=use_pre_softmax
                (bool), a4=groups<<8|group_topk packed int (0 = both None).
      score:    t0=logits, a2=topk(int).
      aux_loss: t0=probs, t1=tokens_per_expert seed, a2=topk(int), a3=coeff
                (float).
    """
    values = list(input_data.args)
    if len(values) != 9:
        raise ValueError("Router ATK case expects 9 positional inputs, got {}".format(len(values)))
    t0, t1, g0, g1, variant, score_function, a2, a3, a4 = values
    variant = str(variant)
    score_function = str(score_function)

    from transformer_engine.pytorch.router import (
        fused_compute_score_for_moe_aux_loss,
        fused_moe_aux_loss,
        fused_topk_with_score_function,
    )

    if variant == "topk":
        # generator attr order: variant, score_function, use_pre_softmax(a2),
        # unused(a3), packed groups<<8|group_topk(a4); topk itself comes from
        # a deterministic table keyed by the case scale (mirrors Megatron's
        # router defaults used with each configuration).
        use_pre_softmax = bool(a2)
        topk = int(a3)
        packed = int(a4)
        num_groups = (packed >> 8) if packed > 0 else None
        group_topk = (packed & 0xFF) if packed > 0 else None
        if (num_groups is None) != (group_topk is None):
            raise ValueError("groups and group_topk must be set together, got {}".format(packed))
        # bias enabled only where the score function supports it; the sampled
        # t1 is always valid fp32 so no shape branching is needed.
        bias_enabled = (
            score_function in ("sigmoid", "sqrtsoftplus") and float(t1.reshape(-1)[0].item()) > 0
        )
        logits = _prepare_logits(t0)
        probs, routing_map = fused_topk_with_score_function(
            logits,
            topk,
            use_pre_softmax,
            num_groups,
            group_topk,
            _SCALING.get((score_function, bias_enabled)),
            score_function,
            t1 if bias_enabled else None,
        )
        (probs * g0).sum().backward()
        # Index-free selection outputs: probs * routing_map zeroes the
        # unselected experts and sorts to the selected-score multiset, so an
        # exact tie breaking differently on the two backends still matches.
        # probs is already sparse, so the product re-checks the primary
        # output. grad_logits stays positional (see the module docstring for
        # the residual tie exposure).
        contract = _topk_contract(probs, routing_map, logits, topk)
        return (
            _canonical_rows(probs),
            _canonical_rows(probs * routing_map),
            logits.grad,
            _encode_contract(contract, probs.device),
        )

    if variant == "score":
        topk = int(a2)
        logits = _prepare_logits(t0)
        routing_map, scores = fused_compute_score_for_moe_aux_loss(logits, topk, score_function)
        (scores * g0).sum().backward()
        # scores are elementwise in the logits and tie-immune as-is; the
        # routing_map slot is canonicalized to the masked scores, same as the
        # topk branch.
        contract = _score_contract(routing_map, scores, logits, topk)
        return (
            scores,
            _canonical_rows(scores * routing_map),
            logits.grad,
            _encode_contract(contract, scores.device),
        )

    if variant == "aux_loss":
        topk = int(a2)
        coeff = float(a3)
        probs = t0.detach().clone().requires_grad_(True)
        total = int(probs.shape[0])
        num_experts = int(probs.shape[1])
        tokens_per_expert = _synth_tokens_per_expert(t1, total, num_experts)
        loss = fused_moe_aux_loss(probs, tokens_per_expert, total, num_experts, topk, coeff)
        (loss * g0.reshape(loss.shape)).sum().backward()
        contract = _aux_contract(loss, probs, tokens_per_expert, topk, coeff)
        return (
            probs,
            loss,
            probs.grad,
            _encode_contract(contract, loss.device),
        )

    raise ValueError("Unknown router variant: {}".format(variant))


@register("function_router")
class FunctionRouter(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_router_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_router_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "Router template has no CPU reference; only 'npu' execution is "
                "supported. NVTE consistency comes from the accuracy_load golden "
                "node, not from a local torch implementation. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
