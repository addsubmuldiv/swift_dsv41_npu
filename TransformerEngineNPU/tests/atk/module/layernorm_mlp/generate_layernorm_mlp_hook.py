# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``LayerNormMLP``.

Single route (2026-09-20): the native ATK engine drives sampling from the yaml;
this subclass derives the seven tensors that hang off ``x``.

Adapter layout (``function_layernorm_mlp.run_layernorm_mlp_case``):

  x, dy, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b,
  eps, normalization, activation, bias_enabled, zero_centered_gamma

The adapter reads ``hidden_size = x.shape[-1]`` and
``ffn_hidden_size = fc1_w.shape[0]`` and copies every tensor into the module, so
the shapes below are the whole contract (it relies on broadcasting for ``dy``,
which means a wrong shape would be silently accepted rather than rejected):

  x      [..., hidden]          (2/3/4-D, last dim = hidden)
  dy     same as x              (the output has x's trailing dim)
  ln_w   [hidden]
  ln_b   [hidden]               (dead for RMSNorm)
  fc1_w  [ffn, hidden]          (ffn is the free dimension)
  fc1_b  [ffn]                  (dead when bias is off)
  fc2_w  [hidden, ffn]
  fc2_b  [hidden]               (dead when bias is off)

``ffn`` comes from the sampled placeholder in slot 4 (the same trick as the
other derived-shape hooks: the yaml pool is the source, the final shape is
written here), clamped so ``ffn * hidden`` stays inside a per-tensor budget.

Two sample-space guards run in ``after_case_config``: the eps/rsigma clamp on
x (shared module), and the fp16 pre-activation clamp on the fc1 weight/bias
ranges. The latter exists because x-hat is unit-RMS per token BY CONSTRUCTION,
so no x-range clamp can bound the fc1 pre-activation that the activation
backward feeds into exp/tanh -- and fp16's 5-bit exponent overflows there
~33 orders of magnitude earlier than bf16/fp32's 8-bit one. Containment of a
device-side defect, not a well-posedness carve-out: see the shared module's
docstring for the reach formula and the removal condition.

Determinism: identical yaml + identical ``atk case -s <seed>`` + identical ATK
version ⇒ identical cases.
"""

import os
import sys
from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Shared sample-space guards (the eps/rsigma coupling) live one level up.
_CONSTRAINTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CONSTRAINTS_DIR not in sys.path:
    sys.path.insert(0, _CONSTRAINTS_DIR)
from _range_constraints import (  # noqa: E402
    MAX_RSIGMA,
    clamp_fc1_range_for_preact,
    clamp_x_range_for_eps,
    preact_violations,
    rsigma_violations,
)

# x, dy, ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b + 5 attrs.
_EXPECTED_TYPES = ("tensor",) * 8 + ("attr",) * 5

_KNOWN_NORMALIZATIONS = ("LayerNorm", "RMSNorm")
_KNOWN_ACTIVATIONS = ("gelu", "silu")

# Per-tensor budget for the derived GEMM weights (ffn * hidden).
_MAX_WEIGHT_ELEMENTS = 1 << 23


@GENERATOR_REGISTRY.register("generate_layernorm_mlp_hook")
class GenerateLayerNormMLPHook(CaseGenerator):
    """Native sampling + shape derivation for the LayerNormMLP matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_layernorm_mlp_hook expects the declared layout (x, dy, "
                "ln_w, ln_b, fc1_w, fc1_b, fc2_w, fc2_b + eps, normalization, "
                "activation, bias, zero_centered_gamma) with types tensor x8 + "
                "attr x5, got {}".format(got)
            )
        super().__init__(config)
        self._x_shape = None
        self._dtype = None
        self._ffn_source = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        if index == 0:
            shape = list(input_case.shape)
            # Fail-closed: 2/3/4-D with every dim >= 1. Keep the boundary track
            # off — an empty x is a different (untested) contract for this
            # module.
            if len(shape) not in (2, 3, 4) or any(d < 1 for d in shape):
                raise ValueError(
                    "generate_layernorm_mlp_hook: x must be 2/3/4-D with every "
                    "dim >= 1, got {} — keep extra_numbers: 0 and "
                    "has_empty/has_scalar/has_infnan: false".format(input_case.shape)
                )
            self._x_shape = shape
            self._dtype = input_case.dtype
        elif index == 4:
            # fc1_weight's placeholder carries the ffn source dimension.
            self._ffn_source = list(input_case.shape)[-1]
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        inputs = case_config.inputs
        hidden = self._x_shape[-1]
        dtype = self._dtype
        # The yaml pool is the ffn source; shrink it deterministically if the
        # weight budget would be exceeded.
        ffn = max(1, int(self._ffn_source))
        if ffn * hidden > _MAX_WEIGHT_ELEMENTS:
            ffn = max(1, _MAX_WEIGHT_ELEMENTS // hidden)

        shapes = {
            0: list(self._x_shape),  # x
            1: list(self._x_shape),  # dy
            2: [hidden],  # ln_w
            3: [hidden],  # ln_b
            4: [ffn, hidden],  # fc1_w
            5: [ffn],  # fc1_b
            6: [hidden, ffn],  # fc2_w
            7: [hidden],  # fc2_b
        }
        for slot, shape in shapes.items():
            inputs[slot].shape = shape
            inputs[slot].dtype = dtype

        problems = []
        # eps (slot 8) and x's range (slot 0) are both in hand only here, so
        # the eps/rsigma coupling is decided here. Clamped rather than
        # rejected: ATK's generator has no per-case skip and raising aborts
        # the whole `atk case` run (the DPA head_dim lesson).
        clamp_x_range_for_eps(inputs[0], dtype, inputs[8].range_values)
        for std, rsigma in rsigma_violations(inputs[0].range_values, dtype, inputs[8].range_values):
            problems.append(
                "x std {:.4g} with eps {:.4g} gives rsigma {:.4g} > {}".format(
                    std, float(inputs[8].range_values), rsigma, MAX_RSIGMA
                )
            )
        # fp16 activation-backward overflow: the fc1 pre-activation's k-sigma
        # reach must fit the per-activation bound (bf16/fp32 never see the
        # corner -- 8-bit exponents). x-hat is unit-RMS, so the rsigma clamp
        # above cannot bound it; the lever is the fc1 weight/bias range.
        preact_kwargs = dict(
            hidden=hidden,
            ln_w_range=inputs[2].range_values,
            ln_b_range=inputs[3].range_values,
            normalization=inputs[9].range_values,
            zero_centered_gamma=inputs[12].range_values,
            bias_enabled=inputs[11].range_values,
            dtype=dtype,
            activation=inputs[10].range_values,
        )
        clamp_fc1_range_for_preact(inputs[4], inputs[5], **preact_kwargs)
        problems.extend(
            preact_violations(inputs[4].range_values, inputs[5].range_values, **preact_kwargs)
        )
        if inputs[9].range_values not in _KNOWN_NORMALIZATIONS:
            problems.append("normalization {!r}".format(inputs[9].range_values))
        if inputs[10].range_values not in _KNOWN_ACTIVATIONS:
            problems.append("activation {!r}".format(inputs[10].range_values))
        for slot in (11, 12):
            if not isinstance(inputs[slot].range_values, bool):
                problems.append("slot {} is not a bool".format(slot))
        if inputs[4].shape != [ffn, hidden] or inputs[6].shape != [hidden, ffn]:
            problems.append("fc1/fc2 weight shapes disagree")
        if inputs[2].shape != [hidden] or inputs[7].shape != [hidden]:
            problems.append("norm/fc2 bias shapes disagree")
        if inputs[1].shape != inputs[0].shape:
            problems.append("dy does not mirror x")
        if {c.dtype for c in inputs[:8]} != {dtype}:
            problems.append("tensor dtypes are not uniform")
        if ffn * hidden > _MAX_WEIGHT_ELEMENTS:
            problems.append("weight budget exceeded")
        if problems:
            raise ValueError(
                "generate_layernorm_mlp_hook: assembled case violates the "
                "coupling contract (x={}, hidden={}, ffn={}): {}".format(
                    self._x_shape, hidden, ffn, "; ".join(problems)
                )
            )
        return case_config
