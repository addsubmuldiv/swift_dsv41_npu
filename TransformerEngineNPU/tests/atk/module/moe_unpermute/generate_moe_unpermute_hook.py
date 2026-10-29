# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``moe_unpermute``.

Single route (2026-09-20): the native ATK engine drives sampling from the
yaml; this subclass rebuilds each case into the slot layout its ``mode``
requires and renders the semantic spec strings.

The adapter runs the matching permute API first and feeds its output straight
into ``moe_unpermute``, so a case carries the PERMUTE side's inputs — see the
adapter docstring (``function_moe_unpermute.py``) and the module README.
``inp`` is therefore the original token tensor [T, H] and ``dy`` is the gradient
of the UNPERMUTED output [T, H].

Declared union (3 tensors + 8 attrs); the case layout differs per mode, and the
adapter locates ``mode`` at slot 2 (permute) or slot 3 (with_probs / pad):

  permute        [inp, dy, mode, map_type, spec, E, topk, routing_dtype,
                  num_out_tokens, max_token_num]                        (10)
  with_probs     [inp, probs, dy, mode, spec, E, topk, routing_dtype,
                  num_out_tokens]                                       (9)
  pad_with_probs [inp, probs, dy, mode, spec, E, topk, routing_dtype,
                  tokens_per_expert_spec, align_size]                   (10)

Coupling this hook owns:

  1. ``routing_spec`` := T rows, one per token, each routing to ``topk``
     experts (owner + k mod E), deterministically scrambled. Dropped tokens
     (empty rows) are NOT generated — the curated matrix excluded them and
     re-adding them failed on device (see the moe_permute hook).
  2. ``map_type`` is permute-mode only; the other modes are always mask, and the
     index path requires ``topk == 1`` (without merging_probs the unpermute side
     takes ``row_id_map.size(0)`` as the token count), so topk is forced to 1
     there.
  3. ``probs`` := [T, E] (the 2-D original probabilities, which is what
     ``moe_unpermute`` reads as ``merging_probs`` — never the 1-D permuted
     output of the permute API).
  4. ``num_out_tokens`` := tokens * topk (permute / with_probs); the pad mode
     derives ``tokens_per_expert_spec`` and an ``align_size`` whose padded total
     is divisible by E and inside the kernel's numOutTokens cap (same rules as
     the mp_pad_probs hook).
  5. ``dy`` := [T, H] in every mode (gradient of the unpermuted output), and
     every tensor mirrors inp's dtype.

Determinism: identical yaml + identical ``atk case -s <seed>`` + identical ATK
version ⇒ identical cases.
"""

import random
from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# inp, probs, dy, mode, map_type, routing_spec, num_experts, topk,
# routing_dtype, num_out_tokens_or_tpe, max_token_num_or_align.
_EXPECTED_TYPES = ("tensor",) * 3 + ("attr",) * 8

_KNOWN_MODES = ("permute", "with_probs", "pad_with_probs")
_KNOWN_MAP_TYPES = ("mask", "index")
_KNOWN_ROUTING_DTYPES = ("bool", "int32")
_KNOWN_STYLES = ("uniform", "uneven", "skewed", "empty_expert")

_MAX_EXPERTS = 128
_ALIGN_POOL = (1, 2, 4, 8, 16, 32, 64, 128)
_MAX_NUM_OUT_TOKENS = 1024  # NPU kernel: "numOutTokens should not great than 1024"


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


@GENERATOR_REGISTRY.register("generate_moe_unpermute_hook")
class GenerateMoeUnpermuteHook(CaseGenerator):
    """Native sampling + per-mode slot rebuild and spec rendering."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_moe_unpermute_hook expects the declared union layout "
                "(inp, probs, dy, mode, map_type, routing_spec, num_experts, "
                "topk, routing_dtype, num_out_tokens_or_tpe, "
                "max_token_num_or_align) with types tensor x3 + attr x8, got "
                "{}".format(got)
            )
        super().__init__(config)
        self._inp_shape = None
        self._dtype = None
        self._mode = None
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
        # Everything is assembled in after_case_config (the spec needs the shape
        # AND the attrs, and the attrs fire last).
        if index == 0:
            shape = list(input_case.shape)
            if len(shape) != 2 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_moe_unpermute_hook: inp must be 2-D [tokens, "
                    "hidden] with every dim >= 1, got {} — keep extra_numbers: 0 "
                    "and has_empty/has_scalar/has_infnan: false".format(input_case.shape)
                )
            self._inp_shape = shape
            self._dtype = input_case.dtype
        elif index == 3:
            mode = str(input_case.range_values)
            if mode not in _KNOWN_MODES:
                raise ValueError(
                    "generate_moe_unpermute_hook: unknown mode {!r} — expected "
                    "one of {} (check the yaml mode pool)".format(mode, list(_KNOWN_MODES))
                )
            self._mode = mode
        elif index == 4:
            map_type = str(input_case.range_values)
            if map_type not in _KNOWN_MAP_TYPES:
                raise ValueError(
                    "generate_moe_unpermute_hook: unknown map_type {!r} — "
                    "expected one of {} (check the yaml map_type pool)".format(
                        map_type, list(_KNOWN_MAP_TYPES)
                    )
                )
            self._map_type = map_type
        elif index == 5:
            style = str(input_case.range_values)
            if style not in _KNOWN_STYLES:
                raise ValueError(
                    "generate_moe_unpermute_hook: unknown routing style {!r} — "
                    "expected one of {} (check the yaml routing_spec pool)".format(
                        style, list(_KNOWN_STYLES)
                    )
                )
            self._style = style
        elif index == 6:
            self._num_experts = int(input_case.range_values)
        elif index == 7:
            self._topk = int(input_case.range_values)
        elif index == 8:
            routing_dtype = str(input_case.range_values)
            if routing_dtype not in _KNOWN_ROUTING_DTYPES:
                raise ValueError(
                    "generate_moe_unpermute_hook: unknown routing_dtype {!r} — "
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
        """Smallest align with a divisible padded total inside the kernel cap."""
        for align in _ALIGN_POOL:
            padded = cls._padded_total(received, align)
            if padded % experts == 0 and padded <= _MAX_NUM_OUT_TOKENS:
                return align, padded
        return None, None

    @staticmethod
    def _attr(value, dtype: str = "string") -> InputCaseConfig:
        return InputCaseConfig(
            name=None,
            type="attr",
            dtype=dtype,
            shape=None,
            range_values=value,
            backward=False,
        )

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tokens, hidden = self._inp_shape
        # Every unpermute case runs NVTE's CUDA unpermute kernel (both mask
        # and index paths), which accesses the hidden dim through float4
        # (16-byte) vectorized loads/stores and so requires
        # hidden * sizeof(dtype) % 16 == 0; hidden = 17 crashes the golden
        # CUDA side with a misaligned address. Remap hidden unconditionally.
        # tokens = 17 is harmless — the constraint is on the hidden dim only.
        if hidden == 17:
            hidden = 16
        # Enforce the equivalent of max_length = 33554432 (GENERALIZATION
        # §5.3): the yaml dim pool's max product stays far below this bound,
        # so it never binds normal sampling — it only guards the assembled
        # case against pathological element counts the way the yaml cap does.
        if tokens * hidden > 33554432:
            hidden = max(1, 33554432 // tokens)
        self._inp_shape = [tokens, hidden]
        dtype = self._dtype
        mode = self._mode
        experts = max(2, min(int(self._num_experts), _MAX_EXPERTS))
        topk = max(1, min(int(self._topk), 8, experts))
        inp, probs, dy = case_config.inputs[0], case_config.inputs[1], case_config.inputs[2]
        (
            mode_attr,
            map_type_attr,
            spec_attr,
            experts_attr,
            topk_attr,
            routing_dtype_attr,
            tail_attr,
            last_attr,
        ) = case_config.inputs[3:11]

        # The index path is permute-mode only and needs topk == 1 (without
        # merging_probs the unpermute side takes row_id_map.size(0) as the token
        # count).
        map_type = self._map_type if mode == "permute" else "mask"
        if map_type == "index":
            topk = 1
        routing_dtype = "int32" if map_type == "index" else self._routing_dtype

        counts = self._allocate(tokens, experts)
        # Per-token target lists for the spec (and the received counts).
        row_targets = [
            [(owner + k) % experts for k in range(topk)] for owner in _owner_sequence(counts)
        ]
        received = [0] * experts
        for targets in row_targets:
            for target in targets:
                received[target] += 1
        spec = ";".join(
            _scramble(
                [",".join(str(t) for t in targets) for targets in row_targets],
                _spec_seed(counts, topk, 0),
            )
        )

        inp.shape = [tokens, hidden]
        inp.dtype = dtype
        dy.shape = [tokens, hidden]
        dy.dtype = dtype
        probs.shape = [tokens, experts]
        probs.dtype = dtype
        mode_attr.range_values = mode
        spec_attr.range_values = spec
        experts_attr.range_values = experts
        topk_attr.range_values = topk
        routing_dtype_attr.range_values = routing_dtype

        problems = []
        if mode == "permute":
            nout = tokens * topk if map_type == "mask" else 0
            map_type_attr.range_values = map_type
            tail_attr.range_values = str(nout)
            last_attr.range_values = -1
            assembled = [
                inp,
                dy,
                mode_attr,
                map_type_attr,
                spec_attr,
                experts_attr,
                topk_attr,
                routing_dtype_attr,
                tail_attr,
                last_attr,
            ]
        elif mode == "with_probs":
            map_type_attr.range_values = "mask"
            tail_attr.range_values = str(tokens * topk)
            assembled = [
                inp,
                probs,
                dy,
                mode_attr,
                spec_attr,
                experts_attr,
                topk_attr,
                routing_dtype_attr,
                tail_attr,
            ]
        else:  # pad_with_probs
            map_type_attr.range_values = "mask"
            align, padded = None, None
            for attempt in range(4):
                align, padded = self._pick_align(received, experts)
                if padded is not None:
                    break
                tokens = max(1, (_MAX_NUM_OUT_TOKENS // max(1, topk)) // (attempt + 2))
                inp.shape = [tokens, hidden]
                dy.shape = [tokens, hidden]
                probs.shape = [tokens, experts]
                counts = self._allocate(tokens, experts)
                row_targets = [
                    [(owner + k) % experts for k in range(topk)]
                    for owner in _owner_sequence(counts)
                ]
                received = [0] * experts
                for targets in row_targets:
                    for target in targets:
                        received[target] += 1
                spec_attr.range_values = ";".join(
                    _scramble(
                        [",".join(str(t) for t in targets) for targets in row_targets],
                        _spec_seed(counts, topk, 0),
                    )
                )
            if padded is None:
                problems.append("no align_size fits the divisibility rule and the numOutTokens cap")
            tail_attr.range_values = ",".join(str(c) for c in received)
            last_attr.range_values = align if align is not None else 1
            assembled = [
                inp,
                probs,
                dy,
                mode_attr,
                spec_attr,
                experts_attr,
                topk_attr,
                routing_dtype_attr,
                tail_attr,
                last_attr,
            ]

        case_config.inputs = assembled

        # Fail-closed consistency check on the assembled case.
        if len(spec_attr.range_values.split(";")) != tokens:
            problems.append("spec rows != tokens")
        if dy.shape != [tokens, hidden]:
            problems.append("dy must be the unpermuted-output gradient [T, H]")
        if mode != "permute" and list(probs.shape) != [tokens, experts]:
            problems.append("probs must be the 2-D [T, E] merging_probs")
        if mode == "pad_with_probs" and len(str(tail_attr.range_values).split(",")) != experts:
            problems.append("tokens_per_expert must have E values")
        if {inp.dtype, dy.dtype} != {dtype} or (mode != "permute" and probs.dtype != dtype):
            problems.append("tensor dtypes are not uniform")
        if problems:
            raise ValueError(
                "generate_moe_unpermute_hook: assembled case violates the "
                "coupling contract (mode={}, map_type={}, style={}, tokens={}, "
                "experts={}, topk={}): {}".format(
                    mode, map_type, self._style, tokens, experts, topk, "; ".join(problems)
                )
            )
        return case_config
