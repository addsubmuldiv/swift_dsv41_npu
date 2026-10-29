# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``moe_sort_chunks_by_index_with_probs``.

Single route (2026-09-18): this generator replaces the curated P0 matrix — the
module keeps one generator only, so the yaml ``generate:`` field and the
``atk case -p`` argument both point here.

Route: the native ATK engine drives sampling from the yaml; this subclass
renders the two semantic spec strings the adapter parses and mirrors the shapes
the sampler cannot know.

Adapter layout
(``function_moe_sort_chunks_by_index_with_probs.run_moe_sort_chunks_by_index_with_probs_case``):

  inp [T, H], probs [T], dy [T, H], dy_probs [T],
  split_spec "a,b,c", sorted_spec "i,j,k"

Coupling this hook owns:

  1. ``dy`` mirrors inp and ``dy_probs`` mirrors probs — the adapter relies on
     broadcasting in ``(output*dy).sum() + (permuted_probs*dy_probs).sum()``, so
     a mismatch would be silently accepted rather than rejected. ``probs`` is
     1-D (one probability per token), which the yaml cannot express per slot.
  2. ``split_spec`` := the sampled "<pattern>:<num_chunks>" style rendered into
     chunk sizes that sum EXACTLY to the sampled token count, with at most one
     empty chunk (the curated matrix covered empty chunks as a real MoE
     condition). A literal spec in the yaml cannot work: the row count is
     sampled in slot 0.
  3. ``sorted_spec`` := a non-identity permutation of ``0..num_chunks-1`` whose
     length equals the split's — the adapter asserts both lengths are equal, and
     the curated generator additionally required non-identity ("a no-op sort
     would not be a regression guard"). Both invariants are only enforced by
     construction here.

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI default
1234) + identical ATK version ⇒ identical cases.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# inp, probs, dy, dy_probs, split_spec, sorted_spec.
_EXPECTED_TYPES = ("tensor",) * 4 + ("attr",) * 2

_SPLIT_PATTERNS = ("uniform", "uneven", "skewed", "one_empty")
_ORDER_STYLES = ("reverse", "rotate", "interleave")


@GENERATOR_REGISTRY.register("generate_moe_sort_chunks_by_index_with_probs_hook")
class GenerateMoeSortChunksByIndexWithProbsHook(CaseGenerator):
    """Native sampling + spec rendering and tensor mirroring."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_moe_sort_chunks_by_index_with_probs_hook expects the "
                "declared layout (inp, probs, dy, dy_probs, split_spec_style, "
                "sorted_spec_style) with types tensor x4 + attr x2, got {}".format(got)
            )
        super().__init__(config)
        self._inp_shape = None
        self._dtype = None
        self._split_style = None
        self._order_style = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        # The attrs fire after the tensors; everything is assembled in
        # after_case_config, so only capture here.
        if index == 0:
            shape = list(input_case.shape)
            if len(shape) != 2 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_moe_sort_chunks_by_index_with_probs_hook: inp must "
                    "be 2-D with every dim >= 1, got {} — keep extra_numbers: 0 "
                    "and has_empty/has_scalar/has_infnan: false".format(input_case.shape)
                )
            self._inp_shape = shape
            self._dtype = input_case.dtype
        elif index == 4:
            style = str(input_case.range_values)
            if ":" not in style or style.split(":", maxsplit=1)[0] not in _SPLIT_PATTERNS:
                raise ValueError(
                    "generate_moe_sort_chunks_by_index_with_probs_hook: unknown "
                    "split style {!r} — expected '<pattern>:<num_chunks>' with "
                    "pattern in {} (check the yaml split_spec pool)".format(
                        style, list(_SPLIT_PATTERNS)
                    )
                )
            self._split_style = style
        elif index == 5:
            style = str(input_case.range_values)
            if style not in _ORDER_STYLES:
                raise ValueError(
                    "generate_moe_sort_chunks_by_index_with_probs_hook: unknown "
                    "order style {!r} — expected one of {} (check the yaml "
                    "sorted_spec pool)".format(style, list(_ORDER_STYLES))
                )
            self._order_style = style
        return input_case

    @staticmethod
    def _split_evenly(total: int, groups: int) -> List[int]:
        base, remainder = divmod(total, groups)
        return [base + (1 if i < remainder else 0) for i in range(groups)]

    def _render_sizes(self, pattern: str, chunks: int, tokens: int) -> List[int]:
        """Chunk sizes summing exactly to ``tokens``; at most one empty chunk."""
        if pattern == "uniform":
            return self._split_evenly(tokens, chunks)
        if pattern == "uneven":
            # +/-3% sawtooth (largest-remainder), floor of one token.
            weights = [1.0 + 0.03 * (1 if i % 2 == 0 else -1) for i in range(chunks)]
            scale = sum(weights)
            raw = [w / scale * tokens for w in weights]
            sizes = [max(1, int(r)) for r in raw]
            missing = tokens - sum(sizes)
            order = sorted(range(chunks), key=lambda i: (-(raw[i] % 1.0), i))
            for k in range(missing):
                sizes[order[k % chunks]] += 1
            return sizes if sum(sizes) == tokens else self._split_evenly(tokens, chunks)
        if pattern == "skewed":
            # One chunk takes ~60% (router collapse), the rest share the remainder.
            head = max(1, min(tokens - (chunks - 1), int(round(tokens * 0.6))))
            return [head] + self._split_evenly(tokens - head, chunks - 1)
        # one_empty: everything to the first chunks-1, the last chunk is empty.
        return self._split_evenly(tokens, chunks - 1) + [0]

    @staticmethod
    def _render_order(style: str, chunks: int) -> List[int]:
        """A non-identity permutation of 0..chunks-1 (chunks >= 2)."""
        if style == "reverse":
            order = list(range(chunks - 1, -1, -1))
        elif style == "rotate":
            order = list(range(1, chunks)) + [0]
        else:
            # interleave: even indices first, then odd ones — which collapses to
            # the identity at chunks == 2 (a no-op sort is not a regression guard).
            order = list(range(0, chunks, 2)) + list(range(1, chunks, 2))
        if order == list(range(chunks)):
            order = order[1:] + order[:1]
        return order

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tokens, hidden = self._inp_shape
        dtype = self._dtype
        pattern, chunks = self._split_style.split(":")
        chunks = int(chunks)
        inp, probs, dy, dy_probs, split_attr, order_attr = case_config.inputs

        # A split needs one token per non-empty chunk; clamp the requested chunk
        # count and always keep at least two chunks (a non-identity permutation
        # needs two). one_empty's floor is 2 (one real + one empty), not 3 —
        # at tokens == 1 a floor of 3 overrode the tokens + 1 cap and rendered
        # [1, 0, 0], which the contract check below then rightly rejected.
        if pattern == "one_empty":
            chunks = max(2, min(chunks, tokens + 1))
        else:
            chunks = max(2, min(chunks, tokens))
        sizes = self._render_sizes(pattern, chunks, tokens)
        order = self._render_order(self._order_style, chunks)

        inp.shape = [tokens, hidden]
        inp.dtype = dtype
        probs.shape = [tokens]
        probs.dtype = dtype
        dy.shape = [tokens, hidden]
        dy.dtype = dtype
        dy_probs.shape = [tokens]
        dy_probs.dtype = dtype
        split_attr.dtype = "string"
        split_attr.range_values = ",".join(str(x) for x in sizes)
        order_attr.dtype = "string"
        order_attr.range_values = ",".join(str(x) for x in order)

        # Fail-closed consistency check on the assembled case: every invariant
        # below is otherwise only enforced by construction (the adapter checks
        # the two spec lengths against each other and nothing else).
        problems = []
        if len(sizes) != len(order):
            problems.append("spec lengths {} != {}".format(len(sizes), len(order)))
        if sum(sizes) != tokens:
            problems.append("split sums to {} != tokens {}".format(sum(sizes), tokens))
        if any(size < 0 for size in sizes) or sum(1 for s in sizes if s == 0) > 1:
            problems.append("negative or multiple empty chunks: {}".format(sizes))
        if sorted(order) != list(range(chunks)):
            problems.append("order {} is not a permutation of 0..{}".format(order, chunks - 1))
        if order == list(range(chunks)):
            problems.append("order {} is the identity permutation".format(order))
        if dy.shape != inp.shape or dy.dtype != inp.dtype:
            problems.append("dy {} does not mirror inp {}".format(dy.shape, inp.shape))
        if dy_probs.shape != probs.shape or dy_probs.dtype != probs.dtype:
            problems.append(
                "dy_probs {} does not mirror probs {}".format(dy_probs.shape, probs.shape)
            )
        if {inp.dtype, probs.dtype, dy.dtype, dy_probs.dtype} != {dtype}:
            problems.append("tensor dtypes are not uniform")
        if problems:
            raise ValueError(
                "generate_moe_sort_chunks_by_index_with_probs_hook: assembled "
                "case violates the coupling contract (pattern={}, chunks={}, "
                "tokens={}): {}".format(pattern, chunks, tokens, "; ".join(problems))
            )
        return case_config
