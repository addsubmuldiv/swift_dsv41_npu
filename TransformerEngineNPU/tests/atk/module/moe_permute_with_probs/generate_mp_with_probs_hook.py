# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``moe_permute_with_probs``.

Single route (2026-09-20): the native ATK engine drives sampling from the yaml;
this subclass renders the semantic ``routing_spec`` and derives everything that
depends on it. The spec synthesis (``_owner_sequence`` / ``_spec_seed`` /
``_scramble`` / ``_routing_spec``) is moved verbatim from the curated
generator — it is the module's semantic contract: the routing map is rebuilt
from this string on both backends, so its content must stay bit-identical.

Adapter layout (``function_mp_with_probs.run_moe_permute_with_probs_case``):

  inp [T, H], probs [T, E], dy [nout, H], dy_probs [nout], routing_spec,
  num_experts, topk, routing_dtype, num_out_tokens

Coupling this hook owns (only enforced by construction — the adapter checks the
spec's dtype/target ranges, nothing else):

  1. ``routing_spec`` := T rows, one per token, each routing to ``topk``
     experts (owner + k mod E), deterministically scrambled so the same
     expert's tokens are scattered over the input.
  2. ``probs`` := [T, E] — it is gathered by the same routing map as inp, so its
     shape must match the map exactly (mask path only; no map_type here).
  3. ``num_out_tokens`` := tokens * topk (> 0); ``dy`` := [nout, H] and
     ``dy_probs`` := [nout] — the row count the operator produces, which is what
     the loss multiplies.
  4. every tensor mirrors inp's dtype (the adapter relies on broadcasting, so a
     mismatch would be silently accepted rather than rejected).

Determinism: identical yaml + identical ``atk case -s <seed>`` + identical ATK
version ⇒ identical cases.
"""

import random
from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# inp, probs, dy, dy_probs, routing_spec, num_experts, topk, routing_dtype,
# num_out_tokens.
_EXPECTED_TYPES = ("tensor",) * 4 + ("attr",) * 5

_KNOWN_ROUTING_DTYPES = ("bool", "int32")
_KNOWN_STYLES = ("uniform", "uneven", "skewed", "empty_expert")

_MAX_EXPERTS = 128
_MAX_TOPK = 8


def _owner_sequence(counts: List[int]) -> List[int]:
    """Expand per-expert token counts into an owner-per-token list (expert-major)."""
    sequence = []
    for expert, count in enumerate(counts):
        sequence.extend([expert] * count)
    return sequence


def _spec_seed(counts: List[int], topk: int, dropped: int) -> int:
    """Stable per-case seed, computed arithmetically (``hash()`` is salted)."""
    seed = 0x1D3
    for count in counts:
        seed = (seed * 31 + count) & 0xFFFFFFFF
    seed = (seed * 1009 + topk * 10007 + dropped * 100003) & 0xFFFFFFFF
    return seed


def _scramble(rows: List[str], seed: int) -> List[str]:
    """Deterministic shuffle of the per-token rows (regression guard)."""
    ordered = list(rows)
    rng = random.Random(seed)
    for _ in range(8):
        rng.shuffle(rows)
        if rows != ordered:
            return rows
        seed += 1
        rng = random.Random(seed)
    return rows


def _routing_spec(counts: List[int], topk: int) -> str:
    """Encode per-row routed-expert targets as a deterministic attr string."""
    num_experts = len(counts)
    rows = [
        ",".join(str((owner + k) % num_experts) for k in range(topk))
        for owner in _owner_sequence(counts)
    ]
    rows = _scramble(rows, _spec_seed(counts, topk, 0))
    return ";".join(rows)


@GENERATOR_REGISTRY.register("generate_mp_with_probs_hook")
class GenerateMoePermuteWithProbsHook(CaseGenerator):
    """Native sampling + spec rendering and dependent shapes."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_mp_with_probs_hook expects the declared layout (inp, "
                "probs, dy, dy_probs, routing_spec, num_experts, topk, "
                "routing_dtype, num_out_tokens) with types tensor x4 + attr x5, "
                "got {}".format(got)
            )
        super().__init__(config)
        self._inp_shape = None
        self._dtype = None
        self._style = None
        self._num_experts = None
        self._topk = None
        self._routing_dtype = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        # The attrs fire after the tensors; the spec needs all of them, so
        # everything is assembled in after_case_config.
        if index == 0:
            shape = list(input_case.shape)
            if len(shape) != 2 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_mp_with_probs_hook: inp must be 2-D [tokens, "
                    "hidden] with every dim >= 1, got {} — keep extra_numbers: 0 "
                    "and has_empty/has_scalar/has_infnan: false".format(input_case.shape)
                )
            self._inp_shape = shape
            self._dtype = input_case.dtype
        elif index == 4:
            style = str(input_case.range_values)
            if style not in _KNOWN_STYLES:
                raise ValueError(
                    "generate_mp_with_probs_hook: unknown routing style {!r} — "
                    "expected one of {} (check the yaml routing_spec pool)".format(
                        style, list(_KNOWN_STYLES)
                    )
                )
            self._style = style
        elif index == 5:
            self._num_experts = int(input_case.range_values)
        elif index == 6:
            self._topk = int(input_case.range_values)
        elif index == 7:
            routing_dtype = str(input_case.range_values)
            if routing_dtype not in _KNOWN_ROUTING_DTYPES:
                raise ValueError(
                    "generate_mp_with_probs_hook: unknown routing_dtype {!r} — "
                    "expected one of {} (check the yaml routing_dtype pool)".format(
                        routing_dtype, list(_KNOWN_ROUTING_DTYPES)
                    )
                )
            self._routing_dtype = routing_dtype
        return input_case

    @staticmethod
    def _split_evenly(total: int, groups: int) -> List[int]:
        base, remainder = divmod(total, groups)
        return [base + (1 if i < remainder else 0) for i in range(groups)]

    def _allocate(self, tokens: int, experts: int) -> List[int]:
        """Per-expert token counts summing to ``tokens`` (style-driven)."""
        style = self._style
        if style == "uniform":
            return self._split_evenly(tokens, experts)
        if style == "uneven":
            weights = [1.0 + 0.03 * (1 if i % 2 == 0 else -1) for i in range(experts)]
            scale = sum(weights)
            raw = [w / scale * tokens for w in weights]
            counts = [int(r) for r in raw]
            missing = tokens - sum(counts)
            order = sorted(range(experts), key=lambda i: (-(raw[i] % 1.0), i))
            for k in range(missing):
                counts[order[k % experts]] += 1
            return counts if sum(counts) == tokens else self._split_evenly(tokens, experts)
        if style == "skewed":
            head = max(0, min(tokens, int(round(tokens * 0.6))))
            return [head] + self._split_evenly(tokens - head, experts - 1)
        # empty_expert: two experts get nothing (underloaded experts).
        active = max(1, experts - 2)
        return self._split_evenly(tokens, active) + [0, 0]

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tokens, hidden = self._inp_shape
        dtype = self._dtype
        experts = max(2, min(int(self._num_experts), _MAX_EXPERTS))
        topk = max(1, min(int(self._topk), _MAX_TOPK, experts))
        inp, probs, dy, dy_probs = (
            case_config.inputs[0],
            case_config.inputs[1],
            case_config.inputs[2],
            case_config.inputs[3],
        )
        (spec_attr, experts_attr, topk_attr, routing_dtype_attr, nout_attr) = case_config.inputs[
            4:9
        ]

        counts = self._allocate(tokens, experts)
        spec = _routing_spec(counts, topk)
        nout = sum(counts) * topk

        inp.shape = [tokens, hidden]
        inp.dtype = dtype
        probs.shape = [tokens, experts]
        probs.dtype = dtype
        dy.shape = [nout, hidden]
        dy.dtype = dtype
        dy_probs.shape = [nout]
        dy_probs.dtype = dtype
        spec_attr.range_values = spec
        experts_attr.range_values = experts
        topk_attr.range_values = topk
        routing_dtype_attr.range_values = self._routing_dtype
        nout_attr.range_values = nout

        rows = spec.split(";")
        targets = [int(x) for r in rows if r for x in r.split(",")]
        problems = []
        if len(rows) != tokens:
            problems.append("spec rows {} != tokens {}".format(len(rows), tokens))
        if any(r and len(r.split(",")) != topk for r in rows):
            problems.append("a row does not route to exactly topk experts")
        if targets and max(targets) >= experts:
            problems.append("a target is outside [0, {})".format(experts))
        if nout <= 0:
            problems.append("num_out_tokens must be > 0")
        if dy.shape != [nout, hidden] or dy_probs.shape != [nout]:
            problems.append("dy/dy_probs do not mirror the routed output")
        if {inp.dtype, probs.dtype, dy.dtype, dy_probs.dtype} != {dtype}:
            problems.append("tensor dtypes are not uniform")
        if problems:
            raise ValueError(
                "generate_mp_with_probs_hook: assembled case violates the "
                "coupling contract (style={}, tokens={}, experts={}, topk={}): "
                "{}".format(self._style, tokens, experts, topk, "; ".join(problems))
            )
        return case_config
