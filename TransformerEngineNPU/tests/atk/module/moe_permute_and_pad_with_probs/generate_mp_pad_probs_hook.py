# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``moe_permute_and_pad_with_probs``.

Single route (2026-09-20): the native ATK engine drives sampling from the yaml;
this subclass renders the two semantic strings (``routing_spec`` and
``tokens_per_expert_spec``) and derives everything that depends on them.

Adapter layout (``function_mp_pad_probs.run_mp_pad_probs_case``):

  inp [T, H], probs [T, E], dy [padded, H], dy_probs [padded], routing_spec,
  num_experts, topk, routing_dtype, tokens_per_expert_spec, align_size

Coupling this hook owns:

  1. ``routing_spec`` := T rows, one per token, each routing to ``topk``
     experts (owner + k mod E), deterministically scrambled.
     Dropped tokens (empty rows) are NOT generated: the curated matrix excluded
     them and re-adding them failed on device (see the moe_permute hook).
  2. ``tokens_per_expert_spec`` := the per-expert RECEIVED-token counts, counted
     from the same rendered spec (a token routes to topk experts, so this is not
     the owner allocation) — it must have exactly E non-negative values, and the
     adapter derives ``capacity`` from it.
  3. ``align_size`` := the smallest value in a small pool for which the padded
     total ``sum(ceil(t_i / align) * align)`` is divisible by E — otherwise the
     adapter raises (``num_out_tokens must be the same as required``). ``align
     == E`` always works (every term is then a multiple of E), so the search is
     total; it is still worth starting small, because the padded total is the
     row count of dy / dy_probs.
  4. ``dy`` := [padded, H] and ``dy_probs`` := [padded], plus ``probs`` := [T, E]
     and every tensor mirroring inp's dtype.

Determinism: identical yaml + identical ``atk case -s <seed>`` + identical ATK
version ⇒ identical cases.
"""

import random
from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# inp, probs, dy, dy_probs, routing_spec, num_experts, topk, routing_dtype,
# tokens_per_expert_spec, align_size.
_EXPECTED_TYPES = ("tensor",) * 4 + ("attr",) * 6

_KNOWN_ROUTING_DTYPES = ("bool", "int32")
_KNOWN_STYLES = ("uniform", "uneven", "skewed", "empty_expert")

_MAX_EXPERTS = 128
_MAX_TOPK = 8
_ALIGN_POOL = (1, 2, 4, 8, 16, 32, 64, 128)
# NPU kernel limit, measured 2026-09-20: "numOutTokens should not great than
# 1024" (aclnnMoeTokenPermuteWithRoutingMap, error 161002). The padded row count
# dy/dy_probs must feed is what the kernel calls numOutTokens, so the align
# search has to respect it on top of the divisibility rule.
_MAX_NUM_OUT_TOKENS = 1024


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


def _routing_rows(counts: List[int], topk: int) -> List[List[int]]:
    """Per-token routed-expert target lists (before encoding/scrambling)."""
    num_experts = len(counts)
    return [[(owner + k) % num_experts for k in range(topk)] for owner in _owner_sequence(counts)]


@GENERATOR_REGISTRY.register("generate_mp_pad_probs_hook")
class GenerateMoePermuteAndPadWithProbsHook(CaseGenerator):
    """Native sampling + spec rendering and dependent shapes."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_mp_pad_probs_hook expects the declared layout (inp, "
                "probs, dy, dy_probs, routing_spec, num_experts, topk, "
                "routing_dtype, tokens_per_expert_spec, align_size) with types "
                "tensor x4 + attr x6, got {}".format(got)
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
        # The attrs fire after the tensors; the specs need all of them, so
        # everything is assembled in after_case_config.
        if index == 0:
            shape = list(input_case.shape)
            if len(shape) != 2 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_mp_pad_probs_hook: inp must be 2-D [tokens, "
                    "hidden] with every dim >= 1, got {} — keep extra_numbers: 0 "
                    "and has_empty/has_scalar/has_infnan: false".format(input_case.shape)
                )
            self._inp_shape = shape
            self._dtype = input_case.dtype
        elif index == 4:
            style = str(input_case.range_values)
            if style not in _KNOWN_STYLES:
                raise ValueError(
                    "generate_mp_pad_probs_hook: unknown routing style {!r} — "
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
                    "generate_mp_pad_probs_hook: unknown routing_dtype {!r} — "
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
        """Per-expert owner token counts summing to ``tokens`` (style-driven)."""
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

    @staticmethod
    def _padded_total(received: List[int], align: int) -> int:
        return sum((count + align - 1) // align * align for count in received)

    @classmethod
    def _pick_align(cls, received: List[int], experts: int):
        """Smallest align with a divisible padded total that also fits the kernel cap.

        Returns ``(align, padded)``; ``padded`` is None when nothing fits, which
        makes the caller shrink the token count and retry.
        """
        for align in _ALIGN_POOL:
            padded = cls._padded_total(received, align)
            if padded % experts == 0 and padded <= _MAX_NUM_OUT_TOKENS:
                return align, padded
        return None, None

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
        (spec_attr, experts_attr, topk_attr, routing_dtype_attr, tpe_attr, align_attr) = (
            case_config.inputs[4:10]
        )

        # Shrink the token count until the padded total fits the kernel cap: with
        # tokens * topk <= 1024 the align search has a solution (a multiple of E
        # not above the cap always exists for E <= 1024).
        align, padded, counts, received = None, None, None, None
        for attempt in range(4):
            counts = self._allocate(tokens, experts)
            rows = _routing_rows(counts, topk)
            # A token routes to topk experts, so the RECEIVED counts are counted
            # off the routes, not off the owner allocation.
            received = [0] * experts
            for targets in rows:
                for target in targets:
                    received[target] += 1
            align, padded = self._pick_align(received, experts)
            if padded is not None:
                break
            tokens = max(1, (_MAX_NUM_OUT_TOKENS // max(1, topk)) // (attempt + 2))
        if padded is None:
            raise ValueError(
                "generate_mp_pad_probs_hook: no align_size in {} yields a padded "
                "total that is both divisible by num_experts ({}) and <= {} "
                "(tokens={}, topk={})".format(
                    list(_ALIGN_POOL), experts, _MAX_NUM_OUT_TOKENS, tokens, topk
                )
            )

        row_strings = [",".join(str(t) for t in targets) for targets in rows]
        spec = ";".join(_scramble(row_strings, _spec_seed(counts, topk, 0)))
        spec_attr.range_values = spec
        experts_attr.range_values = experts
        topk_attr.range_values = topk
        routing_dtype_attr.range_values = self._routing_dtype
        tpe_attr.range_values = ",".join(str(c) for c in received)
        align_attr.range_values = align

        inp.shape = [tokens, hidden]
        inp.dtype = dtype
        probs.shape = [tokens, experts]
        probs.dtype = dtype
        dy.shape = [padded, hidden]
        dy.dtype = dtype
        dy_probs.shape = [padded]
        dy_probs.dtype = dtype

        problems = []
        if len(spec.split(";")) != tokens:
            problems.append("spec rows != tokens")
        if len(received) != experts or any(c < 0 for c in received):
            problems.append("tokens_per_expert is not E non-negative values")
        if padded % experts:
            problems.append("padded total is not divisible by E")
        if padded > _MAX_NUM_OUT_TOKENS:
            problems.append(
                "padded total {} exceeds the kernel cap {}".format(padded, _MAX_NUM_OUT_TOKENS)
            )
        if dy.shape != [padded, hidden] or dy_probs.shape != [padded]:
            problems.append("dy/dy_probs do not match the padded rows")
        if {inp.dtype, probs.dtype, dy.dtype, dy_probs.dtype} != {dtype}:
            problems.append("tensor dtypes are not uniform")
        if problems:
            raise ValueError(
                "generate_mp_pad_probs_hook: assembled case violates the "
                "coupling contract (style={}, tokens={}, experts={}, topk={}, "
                "align={}, padded={}): {}".format(
                    self._style, tokens, experts, topk, align, padded, "; ".join(problems)
                )
            )
        return case_config
