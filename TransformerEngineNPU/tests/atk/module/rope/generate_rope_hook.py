# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for the two RoPE entry points.

Single route (2026-09-18): this generator replaces the curated P0 matrix — RoPE
has no second generator, so the yaml ``generate:`` field and the ``atk case -p``
argument both point here.

Route: the native ATK engine drives sampling from ``rope.yaml``; this subclass
rebuilds each case into the per-variant slot layout the adapter expects and
derives every shape/attr the sampler cannot know.

Two variants share the declared 8 slots (union layout, dispatch on slot 4):

  rope (``apply_rotary_pos_emb``, 3 outputs)
      [t, freqs, dy, "rope", tensor_format, interleaved, fused, cu_seqlens]
  rope_fused_qkv (``apply_fused_qkv_rotary_pos_emb``, 5 outputs)
      [qkv, q_freqs, k_freqs, dy, "rope_fused_qkv", tensor_format, interleaved,
       qkv_split_arg_list]

The adapter finds the variant by probing slots 3 and 4 for the first string
(``function_rope.py``), so the hook must guarantee slot 3 is a plain tensor for
the fused variant and the variant string for the plain one.

Coupling this hook owns:

  1. Plain variant: drop the declared slot 3 (unused there) and put the variant
     attr in its place; append a cu_seqlens attr (the declared schema has no
     block for it — it is constructed here).
  2. Shapes: freqs := [s,1,1,d/2], dy := t's shape; thd flattens t's first two
     dims into a token count. Head dim is floored at 4 because the adapter
     rebuilds freqs by mirroring the first half of its last dim, which needs a
     non-empty half.
  3. thd: cu_seqlens is derived from the token count (3 ragged segments, the
     same rule the DPA module uses) and ``fused`` is forced True — TENPU
     documents "thd is only supported when fused is True", and NPU refuses the
     combination otherwise.
  4. Fused variant: tensor_format is reconciled to sbhd when the sampler picked
     thd, because ``apply_fused_qkv_rotary_pos_emb`` asserts thd is unsupported.
     Head dim is snapped to the de-facto LLM set {64,128}; heads-per-group is
     derived from the sampled fused width and the width is rewritten to
     ``(np/ng + 2) * hn``, so the split list always sums to it by construction.
     Slot 7 is replaced by a constructed split attr.
  5. The fused-qkv slot 3 is dy (it carries a tensor there), which is what makes
     the adapter's string probe land on slot 4.

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI default
1234) + identical ATK version ⇒ identical cases.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Declared schema: t/qkv, freqs/q_freqs, dy/k_freqs, unused/dy, variant,
# tensor_format, interleaved, fused/split.
_EXPECTED_TYPES = ("tensor",) * 4 + ("attr",) * 4

_T_INDEX = 0
_FREQS_INDEX = 1
_VARIANT_INDEX = 4
_FORMAT_INDEX = 5
_INTERLEAVED_INDEX = 6
_FUSED_INDEX = 7

_KNOWN_VARIANTS = ("rope", "rope_fused_qkv")
_KNOWN_FORMATS = ("sbhd", "bshd", "thd")

_FUSED_HEAD_DIMS = (64, 128)
_MIN_HEAD_DIM = 4  # freqs 末维 = head_dim // 2 must stay non-empty
_THD_MIN_TOKENS = 3  # the 3 ragged cu_seqlens segments need 3 tokens

# Per-tensor budget for the DERIVED shapes. The sampler's max_length only bounds
# the raw sample, and this hook rewrites the shapes (thd flattening, and
# (np/ng + 2) * hn for the fused variant), so a case can come out far above any
# intended size — measured 2026-09-18: one 4-D draw became 4.5e8 elements
# (3.6 GB in fp32) before this guard. The sequence dim is shrunk
# deterministically to fit instead, which keeps every dependent shape (freqs,
# dy, cu_seqlens) consistent.
_MAX_ELEMENTS = 1 << 24


@GENERATOR_REGISTRY.register("generate_rope_hook")
class GenerateRopeHook(CaseGenerator):
    """Native sampling + per-variant slot rebuild for the RoPE matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_rope_hook expects the declared layout (t, freqs, "
                "dy_or_k_freqs, unused_or_dy, variant, tensor_format, "
                "interleaved, fused_or_split) with types tensor x4 + attr x4, "
                "got {}".format(got)
            )
        super().__init__(config)
        self._t_shape = None
        self._t_dtype = None
        self._head_source = None
        self._variant = None
        self._tensor_format = None
        self._interleaved = None
        self._fused = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        # Slots 4-7 fire after the tensors; the whole case is assembled in
        # after_case_config, so only capture here.
        if index == _T_INDEX:
            shape = list(input_case.shape)
            if len(shape) != 4 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_rope_hook: slot 0 must be 4-D with every dim >= 1, "
                    "got {} — the empty track is not usable for RoPE in this "
                    "harness: thd empty (T<3) cannot derive the 3 ragged "
                    "cu_seqlens segments, head_dim 0 leaves the freqs "
                    "half-width empty (the adapter mirrors it to rebuild "
                    "freqs), and only S=0/B=0 was NPU-probed (GPU "
                    "unverified; ATK zeroes a RANDOM dim). Keep has_empty/"
                    "has_scalar/has_infnan: false".format(input_case.shape)
                )
            self._t_shape = shape
            self._t_dtype = input_case.dtype
        elif index == _FREQS_INDEX:
            self._head_source = list(input_case.shape)[-1]
        elif index == _VARIANT_INDEX:
            variant = str(input_case.range_values)
            if variant not in _KNOWN_VARIANTS:
                raise ValueError(
                    "generate_rope_hook: unknown variant {!r} — expected one of "
                    "{} (check rope.yaml slot 4)".format(variant, list(_KNOWN_VARIANTS))
                )
            self._variant = variant
        elif index == _FORMAT_INDEX:
            tensor_format = str(input_case.range_values)
            if tensor_format not in _KNOWN_FORMATS:
                raise ValueError(
                    "generate_rope_hook: unknown tensor_format {!r} — expected "
                    "one of {} (check rope.yaml slot 5)".format(tensor_format, list(_KNOWN_FORMATS))
                )
            self._tensor_format = tensor_format
        elif index == _INTERLEAVED_INDEX:
            self._interleaved = bool(input_case.range_values)
        elif index == _FUSED_INDEX:
            self._fused = bool(input_case.range_values)
        return input_case

    @staticmethod
    def _fit_sequence(shape, seq_axis):
        """Shrink the sequence dim so the derived tensor stays inside budget."""
        total = 1
        for dim in shape:
            total *= dim
        if total <= _MAX_ELEMENTS:
            return shape, shape[seq_axis]
        others = total // shape[seq_axis]
        shape = list(shape)
        shape[seq_axis] = max(1, _MAX_ELEMENTS // others) if others else 1
        return shape, shape[seq_axis]

    @staticmethod
    def _attr(value: str) -> InputCaseConfig:
        return InputCaseConfig(
            name=None,
            type="attr",
            dtype="string",
            shape=None,
            range_values=value,
            backward=False,
        )

    @staticmethod
    def _cu_seqlens(tokens: int) -> List[int]:
        """Three ragged segments — same rule as the DPA adapter."""
        chunk = tokens // _THD_MIN_TOKENS
        return [chunk * i for i in range(_THD_MIN_TOKENS)] + [tokens]

    def _plain_case(self, inputs: List[InputCaseConfig]):
        first, second, third, fourth = self._t_shape
        dtype = self._t_dtype
        head_dim = max(fourth, _MIN_HEAD_DIM)
        t, freqs, dy, _unused, variant, tensor_format, interleaved, fused = inputs

        if self._tensor_format == "thd":
            shape, tokens = self._fit_sequence(
                [max(first * second, _THD_MIN_TOKENS), third, head_dim], 0
            )
            seq = tokens
            t.shape = shape
            cu_attr = self._attr(",".join(str(x) for x in self._cu_seqlens(tokens)))
            # "thd is only supported when fused is True"
            fused_value = True
        else:
            # The sampled dims keep their positions; the format says which one
            # is the sequence (freqs are indexed by it, and the bshd path
            # transposes freqs internally to match [b, s, h, d]).
            seq_axis = 0 if self._tensor_format == "sbhd" else 1
            t.shape, seq = self._fit_sequence([first, second, third, head_dim], seq_axis)
            cu_attr = self._attr("none")
            fused_value = self._fused

        t.dtype = dtype
        freqs.shape = [seq, 1, 1, head_dim // 2]
        freqs.dtype = "fp32"
        dy.shape = list(t.shape)
        dy.dtype = dtype
        variant.range_values = "rope"
        fused.dtype = "attr_bool"
        fused.range_values = bool(fused_value)
        interleaved.range_values = bool(self._interleaved)
        return [t, freqs, dy, variant, tensor_format, interleaved, fused, cu_attr]

    def _fused_qkv_case(self, inputs: List[InputCaseConfig]):
        first, second, third, fourth = self._t_shape
        dtype = self._t_dtype
        qkv, q_freqs, k_freqs, dy, variant, tensor_format, interleaved, _split = inputs

        if self._tensor_format == "thd":
            # apply_fused_qkv_rotary_pos_emb asserts thd is not supported.
            tensor_format.range_values = "sbhd"
        head_dim = min(_FUSED_HEAD_DIMS, key=lambda h: (abs(h - self._head_source), h))
        groups = max(third, 1)
        heads_per_group = max(1, int(round(fourth / head_dim)) - 2)
        last_dim = (heads_per_group + 2) * head_dim
        # Same convention as the plain variant: sampled dims stay in place, the
        # format picks the sequence dim.
        qkv_shape, seq = self._fit_sequence(
            [first, second, groups, last_dim], 0 if tensor_format.range_values == "sbhd" else 1
        )

        qkv.shape = qkv_shape
        qkv.dtype = dtype
        for freqs_slot in (q_freqs, k_freqs):
            freqs_slot.shape = [seq, 1, 1, head_dim]
            freqs_slot.dtype = "fp32"
        dy.shape = list(qkv_shape)
        dy.dtype = dtype
        variant.range_values = "rope_fused_qkv"
        interleaved.range_values = bool(self._interleaved)
        split = [heads_per_group * head_dim, head_dim, head_dim]
        split_attr = self._attr(",".join(str(x) for x in split))
        return [qkv, q_freqs, k_freqs, dy, variant, tensor_format, interleaved, split_attr]

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = case_config.inputs
        if self._variant == "rope":
            assembled = self._plain_case(inputs)
        else:
            assembled = self._fused_qkv_case(inputs)
        case_config.inputs = assembled

        # Fail-closed consistency check on the assembled case. NOTE the two
        # layouts put the attrs at different indices — that is the point of the
        # union schema, so the check is written per variant.
        problems = []
        if len(assembled) != 8:
            problems.append("arity {}".format(len(assembled)))
        freqs = assembled[1]
        if freqs.dtype != "fp32" or len(freqs.shape) != 4 or freqs.shape[-1] < 2:
            problems.append("freqs {} {}".format(freqs.shape, freqs.dtype))
        if self._variant == "rope":
            t, dy = assembled[0], assembled[2]
            variant_attr, format_attr, interleaved_attr, fused_attr, cu_attr = assembled[3:8]
            if str(variant_attr.range_values) != "rope":
                problems.append("variant {!r}".format(variant_attr.range_values))
            if format_attr.range_values not in _KNOWN_FORMATS:
                problems.append("tensor_format {!r}".format(format_attr.range_values))
            if not isinstance(interleaved_attr.range_values, bool):
                problems.append("interleaved is not a bool")
            if not isinstance(fused_attr.range_values, bool):
                problems.append("fused is not a bool")
            if not isinstance(cu_attr.range_values, str):
                problems.append("cu_seqlens is not a string")
            if dy.shape != t.shape:
                problems.append("dy {} != t {}".format(dy.shape, t.shape))
            if format_attr.range_values == "thd":
                if cu_attr.range_values == "none":
                    problems.append("thd without cu_seqlens")
                if fused_attr.range_values is not True:
                    problems.append("thd without fused=True")
        else:
            qkv, q_freqs, k_freqs, dy = assembled[0], assembled[1], assembled[2], assembled[3]
            variant_attr, format_attr, interleaved_attr, split_attr = assembled[4:8]
            if str(variant_attr.range_values) != "rope_fused_qkv":
                problems.append("variant {!r}".format(variant_attr.range_values))
            if format_attr.range_values not in ("sbhd", "bshd"):
                problems.append("fused-qkv tensor_format {!r}".format(format_attr.range_values))
            if q_freqs.shape[-1] not in _FUSED_HEAD_DIMS:
                problems.append("head_dim {}".format(q_freqs.shape[-1]))
            if k_freqs.shape != q_freqs.shape:
                problems.append("k_freqs {} != q_freqs {}".format(k_freqs.shape, q_freqs.shape))
            if dy.shape != qkv.shape:
                problems.append("dy {} != qkv {}".format(dy.shape, qkv.shape))
            split = [int(x) for x in str(split_attr.range_values).split(",")]
            if len(split) != 3 or sum(split) != qkv.shape[-1]:
                problems.append("split {} does not sum to {}".format(split, qkv.shape[-1]))
        if problems:
            raise ValueError(
                "generate_rope_hook: assembled case violates the RoPE coupling "
                "contract (variant={}, t={}, freqs={}): {}".format(
                    self._variant, assembled[0].shape, freqs.shape, "; ".join(problems)
                )
            )
        return case_config
