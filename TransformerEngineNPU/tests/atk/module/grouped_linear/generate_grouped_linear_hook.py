# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``transformer_engine.pytorch.GroupedLinear``.

Single route (2026-09-18): this generator replaces the curated P0 matrix —
GroupedLinear has no second generator, so the yaml ``generate:`` field and the
``atk case -p`` argument both point here.

Route: the native ATK engine drives sampling from ``grouped_linear.yaml``; this
subclass compacts the declared 37 slots down to the arity a case actually needs
and derives every shape the sampler cannot know.

Layout the adapter (``function_grouped_linear.run_grouped_linear_case``) unpacks
positionally, with ``g = num_gemms``:

  inp [T, in_features]                     rows must equal sum(m_splits)
  dy  [T, out_features]
  weight_0..g-1  [out_features, in_features]
  bias_0..g-1    [out_features]
  num_gemms(int), m_splits("a,b,c"), bias_enabled(bool)

Coupling this hook owns:

  1. Arity. The sampler creates one slot per declared block (37); the case is
     rebuilt as ``[inp, dy] + weights + biases + attrs`` = 2g + 5 slots. The
     yaml declares the maximum arity for g <= 16 because a fixed schema is the
     only thing the native sampler can consume.
  2. Shapes. ``in_features`` comes from inp, ``out_features`` from dy's second
     dim (both sampled); weights/biases are rewritten from that pair and every
     tensor's dtype is mirrored from inp — the adapter builds the module with
     ``params_dtype=inp.dtype`` and copies the weights in, so a mismatched dtype
     would be silently cast rather than rejected.
  3. m_splits. Its row count is fixed by inp (sampled in slot 0), so the yaml
     samples a split STYLE and the rendering happens here: uniform / uneven /
     collapse / tiny_tail, all summing exactly to the token count with at least
     one token per expert (a zero-token expert is not a probed configuration).
  4. num_gemms is clamped to [1, min(16, tokens)] — with fewer tokens than
     experts no positive split can sum to the row count.

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI default
1234) + identical ATK version ⇒ identical cases.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Declared schema: inp, dy, 16 weights, 16 biases, 3 attrs.
_EXPECTED_TYPES = ("tensor", "tensor") + ("tensor",) * 32 + ("attr",) * 3

_MAX_GEMMS = 16
_WEIGHT_START = 2  # 16 declared weight slots follow inp/dy
_BIAS_START = 18  # 16 declared bias slots follow the weights
_NUM_GEM_ATTR, _SPLIT_STYLE_ATTR, _BIAS_ATTR = 34, 35, 36

_SPLIT_STYLES = ("uniform", "uneven", "collapse", "tiny_tail")


@GENERATOR_REGISTRY.register("generate_grouped_linear_hook")
class GenerateGroupedLinearHook(CaseGenerator):
    """Native sampling + arity compaction and shape derivation."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_grouped_linear_hook expects the declared layout "
                "(inp, dy, weight x16, bias x16, num_gemms, m_splits_style, "
                "bias_enabled) with types tensor x34 + attr x3, got {}".format(got)
            )
        super().__init__(config)
        self._inp_shape = None
        self._dtype = None
        self._out_features = None
        self._num_gemms = None
        self._split_style = None
        self._bias_enabled = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        # Slots 34/35/36 fire last, and the whole case is assembled in
        # after_case_config — so capture the sampled values here.
        if index == 0:
            shape = list(input_case.shape)
            # Fail-closed: [tokens, in_features], both >= 1. The empty track
            # is not usable here: rows=0 (with all-zero m_splits) was only
            # NPU-probed — the GPU side goes through NVTE's C++ tensor
            # checks, which are expected to reject the null data_ptr of
            # 0-byte CUDA allocations (the §3#28 mechanism in ATK_REPORT) —
            # and ATK zeroes a RANDOM dim (in_features=0 untested).
            if len(shape) != 2 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_grouped_linear_hook: inp must be 2-D "
                    "[tokens, in_features], got {} — the empty track is not "
                    "usable for grouped_linear: rows=0 was only NPU-probed "
                    "(GPU-side C++ tensor checks expected to reject it, see "
                    "the §3#28 lesson in ATK_REPORT) and ATK zeroes a RANDOM "
                    "dim (in_features=0 untested). Keep has_empty/has_scalar/"
                    "has_infnan: false".format(input_case.shape)
                )
            self._inp_shape = shape
            self._dtype = input_case.dtype
        elif index == 1:
            shape = list(input_case.shape)
            if len(shape) != 2 or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_grouped_linear_hook: dy must be 2-D, its second "
                    "dim carries out_features, got {}".format(input_case.shape)
                )
            self._out_features = shape[1]
        elif index == _NUM_GEM_ATTR:
            self._num_gemms = int(input_case.range_values)
        elif index == _SPLIT_STYLE_ATTR:
            style = str(input_case.range_values)
            if style not in _SPLIT_STYLES:
                raise ValueError(
                    "generate_grouped_linear_hook: unknown split style {!r} — "
                    "expected one of {} (check grouped_linear.yaml slot "
                    "35)".format(style, list(_SPLIT_STYLES))
                )
            self._split_style = style
        elif index == _BIAS_ATTR:
            self._bias_enabled = bool(input_case.range_values)
        return input_case

    @staticmethod
    def _split_evenly(total: int, groups: int) -> List[int]:
        base, remainder = divmod(total, groups)
        return [base + (1 if i < remainder else 0) for i in range(groups)]

    def _split_proportionally(self, total: int, groups: int) -> List[int]:
        """+/-3% sawtooth split (largest-remainder, floor of one token)."""
        weights = [1.0 + 0.03 * (1 if i % 2 == 0 else -1) for i in range(groups)]
        scale = sum(weights)
        raw = [w / scale * total for w in weights]
        splits = [max(1, int(r)) for r in raw]
        missing = total - sum(splits)
        if missing > 0:
            order = sorted(range(groups), key=lambda i: (-(raw[i] % 1.0), i))
            for k in range(missing):
                splits[order[k % groups]] += 1
        return splits if sum(splits) == total else self._split_evenly(total, groups)

    def _render_splits(self, style: str, tokens: int, gemms: int) -> List[int]:
        if gemms == 1:
            return [tokens]
        if style == "uniform":
            return self._split_evenly(tokens, gemms)
        if style == "uneven":
            return self._split_proportionally(tokens, gemms)
        if style == "collapse":
            # One expert takes ~60%; the rest share what is left, always >= 1.
            head = max(1, min(tokens - (gemms - 1), int(round(tokens * 0.6))))
            return [head] + self._split_evenly(tokens - head, gemms - 1)
        # tiny_tail: the last expert is under-loaded (~2%, at least one token).
        tail = max(1, min(tokens - (gemms - 1), tokens // 50))
        return self._split_evenly(tokens - tail, gemms - 1) + [tail]

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = case_config.inputs
        tokens, in_features = self._inp_shape
        dtype = self._dtype
        out_features = self._out_features
        gemms = max(1, min(int(self._num_gemms), _MAX_GEMMS, tokens))
        bias_enabled = bool(self._bias_enabled)

        weights = inputs[_WEIGHT_START : _WEIGHT_START + gemms]
        biases = inputs[_BIAS_START : _BIAS_START + gemms]
        inp, dy = inputs[0], inputs[1]
        inp.shape = [tokens, in_features]
        inp.dtype = dtype
        dy.shape = [tokens, out_features]
        dy.dtype = dtype
        for weight in weights:
            weight.shape = [out_features, in_features]
            weight.dtype = dtype
        for bias in biases:
            bias.shape = [out_features]
            bias.dtype = dtype

        splits = self._render_splits(self._split_style, tokens, gemms)
        num_gem_attr = inputs[_NUM_GEM_ATTR]
        num_gem_attr.dtype = "int"
        num_gem_attr.range_values = gemms
        split_attr = inputs[_SPLIT_STYLE_ATTR]
        split_attr.dtype = "string"
        split_attr.range_values = ",".join(str(x) for x in splits)
        bias_attr = inputs[_BIAS_ATTR]
        bias_attr.dtype = "attr_bool"
        bias_attr.range_values = bias_enabled

        attrs = inputs[_NUM_GEM_ATTR : _NUM_GEM_ATTR + 3]
        case_config.inputs = [inp, dy] + list(weights) + list(biases) + list(attrs)

        # Fail-closed consistency check on the assembled case.
        assembled = case_config.inputs
        rendered = [int(x) for x in str(assembled[-2].range_values).split(",")]
        if not (
            len(assembled) == 2 + 2 * gemms + 3
            and len(rendered) == gemms
            and sum(rendered) == tokens
            and all(x >= 1 for x in rendered)
            and int(assembled[-3].range_values) == gemms
            and isinstance(assembled[-1].range_values, bool)
            and assembled[0].shape == [tokens, in_features]
            and assembled[1].shape == [tokens, out_features]
            and assembled[2].shape == [out_features, in_features]
            and assembled[2 + gemms].shape == [out_features]
            and len({c.dtype for c in assembled[: 2 + 2 * gemms]}) == 1
            and 1 <= gemms <= _MAX_GEMMS
        ):
            raise ValueError(
                "generate_grouped_linear_hook: assembled case violates the "
                "GroupedLinear coupling contract (g={}, tokens={}, in={}, "
                "out={}, splits={}, bias={})".format(
                    gemms, tokens, in_features, out_features, rendered, bias_enabled
                )
            )
        return case_config
