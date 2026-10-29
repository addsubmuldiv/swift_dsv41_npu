# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``moe_permute``.

Single route (2026-09-20): the native ATK engine drives sampling from the yaml;
this subclass renders the semantic ``routing_spec`` and derives everything that
depends on it. The spec synthesis below (``_owner_sequence`` / ``_spec_seed`` /
``_scramble`` / ``_routing_spec``) is moved verbatim from the curated
generator — it is the module's semantic contract, not a sampling detail: the
routing map is rebuilt from this string on both backends, so its content must
stay bit-identical between them.

Adapter layout (``function_moe_permute.run_moe_permute_case``):

  inp [T, H], dy [nout, H], map_type, routing_spec, num_experts, topk,
  routing_dtype, num_out_tokens, max_token_num

Coupling this hook owns (all of it is only enforced by construction — the
adapter checks the spec's internal consistency, nothing else):

  1. ``routing_spec`` := T rows, one per token, each routing to ``topk``
     experts (owner + k mod E), the whole list deterministically scrambled so
     the same expert's tokens are scattered — otherwise a correct kernel could
     pass by returning the input order.
  2. ``routing_dtype`` := "int32" for the index path (the adapter raises
     otherwise); the mask path keeps the sampled value (bool or the legacy
     int32).
  3. ``num_out_tokens`` := tokens * topk for mask (> 0, TE asserts); 0
     ("infer") for index.
  4. ``dy`` := [num_out_tokens or tokens * topk, H] — the row count the operator
     actually produces, which is what the loss multiplies.
  5. Dropped tokens (empty spec rows) are not generated at all: the curated
     matrix excluded them and re-adding them failed on device — with dropped
     rows the operator's output row count does not follow the
     routed_tokens * topk rule the mask path uses without them.

Determinism: identical yaml + identical ``atk case -s <seed>`` + identical ATK
version ⇒ identical cases.
"""

import random
from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# inp, dy, map_type, routing_spec, num_experts, topk, routing_dtype,
# num_out_tokens, max_token_num.
_EXPECTED_TYPES = ("tensor", "tensor") + ("attr",) * 7

_KNOWN_MAP_TYPES = ("mask", "index")
_KNOWN_ROUTING_DTYPES = ("bool", "int32")
_KNOWN_STYLES = ("uniform", "uneven", "skewed", "empty_expert")

_MAX_EXPERTS = 128
_MAX_TOPK = 8


def _owner_sequence(counts: List[int]) -> List[int]:
    """Expand per-expert token counts into an owner-per-token list (expert-major).

    ``_routing_spec`` scrambles this before encoding, so the final input order
    is disorganized with respect to experts.
    """
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
    """Deterministic shuffle of the per-token rows (regression guard).

    A fixed-seed Fisher-Yates makes the token->expert assignment independent of
    the input row order while staying reproducible across runs. If a shuffle
    lands on the original order (possible for tiny lists) the next seed is
    tried, so the encoded spec is guaranteed to be disorganized.
    """
    ordered = list(rows)
    rng = random.Random(seed)
    for _ in range(8):
        rng.shuffle(rows)
        if rows != ordered:
            return rows
        seed += 1
        rng = random.Random(seed)
    return rows


def _routing_spec(counts: List[int], topk: int, dropped: int) -> str:
    """Encode per-row routed-expert targets as a deterministic attr string."""
    num_experts = len(counts)
    rows = [
        ",".join(str((owner + k) % num_experts) for k in range(topk))
        for owner in _owner_sequence(counts)
    ]
    rows.extend([""] * dropped)
    rows = _scramble(rows, _spec_seed(counts, topk, dropped))
    return ";".join(rows)


@GENERATOR_REGISTRY.register("generate_moe_permute_hook")
class GenerateMoePermuteHook(CaseGenerator):
    """Native sampling + spec rendering and dependent shapes."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_moe_permute_hook expects the declared layout (inp, dy, "
                "map_type, routing_spec, num_experts, topk, routing_dtype, "
                "num_out_tokens, max_token_num) with types tensor x2 + attr x7, "
                "got {}".format(got)
            )
        super().__init__(config)
        self._inp_shape = None
        self._dtype = None
        self._map_type = None
        self._style = None
        self._num_experts = None
        self._topk = None
        self._routing_dtype = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        # The attrs fire after the tensors, and the spec needs all of them —
        # everything is assembled in after_case_config.
        if index == 0:
            shape = list(input_case.shape)
            if len(shape) != 2 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_moe_permute_hook: inp must be 2-D [tokens, hidden] "
                    "with every dim >= 1, got {} — keep extra_numbers: 0 and "
                    "has_empty/has_scalar/has_infnan: false".format(input_case.shape)
                )
            self._inp_shape = shape
            self._dtype = input_case.dtype
        elif index == 2:
            map_type = str(input_case.range_values)
            if map_type not in _KNOWN_MAP_TYPES:
                raise ValueError(
                    "generate_moe_permute_hook: unknown map_type {!r} — expected "
                    "one of {} (check the yaml map_type pool)".format(
                        map_type, list(_KNOWN_MAP_TYPES)
                    )
                )
            self._map_type = map_type
        elif index == 3:
            style = str(input_case.range_values)
            if style not in _KNOWN_STYLES:
                raise ValueError(
                    "generate_moe_permute_hook: unknown routing style {!r} — "
                    "expected one of {} (check the yaml routing_spec pool)".format(
                        style, list(_KNOWN_STYLES)
                    )
                )
            self._style = style
        elif index == 4:
            self._num_experts = int(input_case.range_values)
        elif index == 5:
            self._topk = int(input_case.range_values)
        elif index == 6:
            routing_dtype = str(input_case.range_values)
            if routing_dtype not in _KNOWN_ROUTING_DTYPES:
                raise ValueError(
                    "generate_moe_permute_hook: unknown routing_dtype {!r} — "
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
            # One expert takes ~60% (router collapse), the rest share the rest.
            head = max(0, min(tokens, int(round(tokens * 0.6))))
            return [head] + self._split_evenly(tokens - head, experts - 1)
        # empty_expert: two experts get nothing (a real underloaded expert).
        active = max(1, experts - 2)
        return self._split_evenly(tokens, active) + [0, 0]

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tokens, hidden = self._inp_shape
        # The index path runs NVTE's CUDA permute kernel, which accesses the
        # hidden dim through float4 (16-byte) vectorized loads/stores and so
        # requires hidden * sizeof(dtype) % 16 == 0; hidden = 17 crashes the
        # golden CUDA side with a misaligned address. The mask path runs the
        # Triton kernel (no vectorization constraint), so only remap hidden on
        # the index path. tokens = 17 is harmless — the constraint is on the
        # hidden dim only.
        if self._map_type == "index" and hidden == 17:
            hidden = 16
        # Enforce the equivalent of max_length = 33554432 (GENERALIZATION
        # §5.3): the yaml dim pool's max product stays far below this bound,
        # so it never binds normal sampling — it only guards the assembled
        # case against pathological element counts the way the yaml cap does.
        if tokens * hidden > 33554432:
            hidden = max(1, 33554432 // tokens)
        self._inp_shape = [tokens, hidden]
        dtype = self._dtype
        experts = max(2, min(int(self._num_experts), _MAX_EXPERTS))
        topk = max(1, min(int(self._topk), _MAX_TOPK, experts))
        inp, dy = case_config.inputs[0], case_config.inputs[1]
        (
            map_type_attr,
            spec_attr,
            experts_attr,
            topk_attr,
            routing_dtype_attr,
            nout_attr,
            max_token_attr,
        ) = case_config.inputs[2:9]

        # No dropped tokens: the curated generator excluded them ("either
        # violate the NPU/NVTE kernel contract or do not exercise floating-point
        # behavior"), and re-adding them here failed on device — with dropped
        # rows the operator's output row count does not follow the
        # routed_tokens * topk rule the mask path uses without them.
        dropped = 0
        counts = self._allocate(tokens, experts)
        spec = _routing_spec(counts, topk, dropped)

        routing_dtype = "int32" if self._map_type == "index" else self._routing_dtype
        routed_tokens = sum(counts)
        nout = routed_tokens * topk if self._map_type == "mask" else 0
        dy_rows = nout if nout > 0 else tokens * topk

        inp.shape = [tokens, hidden]
        inp.dtype = dtype
        dy.shape = [dy_rows, hidden]
        dy.dtype = dtype
        map_type_attr.range_values = self._map_type
        spec_attr.range_values = spec
        experts_attr.range_values = experts
        topk_attr.range_values = topk
        routing_dtype_attr.range_values = routing_dtype
        nout_attr.range_values = nout
        max_token_attr.range_values = -1

        rows = spec.split(";")
        problems = []
        if len(rows) != tokens:
            problems.append("spec rows {} != tokens {}".format(len(rows), tokens))
        if any(r and len(r.split(",")) != topk for r in rows):
            problems.append("a row does not route to exactly topk experts")
        if any(r == "" for r in rows) and self._map_type == "index":
            problems.append("index spec has an empty row")
        targets = [int(x) for r in rows if r for x in r.split(",")]
        if targets and (min(targets) < 0 or max(targets) >= experts):
            problems.append("a target is outside [0, {})".format(experts))
        if self._map_type == "mask" and nout <= 0:
            problems.append("mask path needs num_out_tokens > 0")
        if self._map_type == "index" and nout != 0:
            problems.append("index path must leave num_out_tokens = 0")
        if dy.shape != [dy_rows, hidden] or dy.dtype != dtype:
            problems.append("dy {} does not mirror the routed output".format(dy.shape))
        if problems:
            raise ValueError(
                "generate_moe_permute_hook: assembled case violates the coupling "
                "contract (map_type={}, style={}, tokens={}, experts={}, topk={}, "
                "dropped={}): {}".format(
                    self._map_type, self._style, tokens, experts, topk, dropped, "; ".join(problems)
                )
            )
        return case_config
