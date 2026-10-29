# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for the multi-tensor utility ops.

ITERATION STATUS: this file is the hook-route candidate, coexisting with
the default full-build generator (``generate_multi_tensor.py``, registry
key ``generate_multi_tensor_p0`` — curated two-tier matrix,
golden-compatible). Once this route is validated through iteration, the
full-build file gets deleted and this one becomes the only generator.
Until then, switch routes by pointing the yaml ``generate:`` field AND
the ``atk case -p`` argument at this file.

Route: the native ATK engine drives sampling; this subclass patches the
one coupling via the official ``after_input_config`` extension point:

    slot 0  t0   → recorded; dtype mirrored onto t1/t2
    slot 1  t1   := t0.shape / t0.dtype
    slot 2  t2   := t0.shape / t0.dtype
    slot 3  op   ∈ {scale, scale_tensor, l2norm} — validated fail-closed
                   in after_case_config (drives which attrs are live)
    slot 4  scale(float)      — live for scale ops, dead for l2norm
    slot 5  per_tensor(bool)  — live for l2norm, dead for scale ops
    slot 6  chunk_size(int)   — chunking hint; ignored on NPU, does not
                                affect numerics (NVTE kernel param)

Probe-verified boundaries (2026-09-16, NPU, all three ops): any ndim is
legal (1-4-D tested, elementwise semantics); EMPTY inputs are fully legal
(any zero dim incl. last dim); MIXED shapes inside the tensor list are
legal at the API level — the equal-shape mirror below is a CASE-DESIGN
convention (contract symmetry + golden comparability), not an API
constraint; bf16 also runs (fp32 is the established case dtype). Because
nothing is unconditionally illegal, the hook has no shape guard.

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI
default 1234) + identical ATK version ⇒ identical cases.

Yaml prerequisites (already encoded in multi_tensor.yaml —
GENERALIZATION.md §5.3): ``shape_distributions`` tuned to module scale;
``max_length`` ≈ 2x the largest intended shape; minimal pinned
``shapes`` on t1/t2. The [2, 3] dim pool is a sample choice matching the
curated coverage, not an API limit.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Fail-closed layout check: the hook addresses slots by index, so a yaml
# whose inputs don't match the 7-slot adapter contract must raise here.
_EXPECTED_TYPES = ("tensor", "tensor", "tensor", "attr", "attr", "attr", "attr")

_KNOWN_OPS = ("scale", "scale_tensor", "l2norm")


@GENERATOR_REGISTRY.register("generate_multi_tensor_hook")
class GenerateMultiTensorHook(CaseGenerator):
    """Native sampling + coupling patch for the multi-tensor matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_multi_tensor_hook expects the input layout "
                "(t0, t1, t2, op, scale, per_tensor, chunk_size) with types "
                "tensor x3 + attr x4, got {}".format(got)
            )
        super().__init__(config)
        self._t0_shape = None
        self._t0_dtype = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        if index == 0:
            # t0 — record only. No shape guard: any ndim and empty inputs
            # are legal (probe 2026-09-16); the equal-shape mirror below is
            # a case-design convention, not an API constraint.
            self._t0_shape = list(input_case.shape)
            self._t0_dtype = input_case.dtype
        elif index in (1, 2):
            # t1 / t2 := t0.shape / t0.dtype (case-design convention; the
            # API itself accepts mixed shapes inside the list)
            input_case.shape = list(self._t0_shape)
            input_case.dtype = self._t0_dtype
        # index 3-6 (op / scale / per_tensor / chunk_size): free attrs.
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        op = str(case_config.inputs[3].range_values)
        if op not in _KNOWN_OPS:
            raise ValueError(
                "generate_multi_tensor_hook: unknown op {!r} — expected one "
                "of {} (check the yaml op attr values)".format(op, list(_KNOWN_OPS))
            )
        # Fail-closed consistency check on the assembled case.
        if not (
            case_config.inputs[1].shape == self._t0_shape
            and case_config.inputs[2].shape == self._t0_shape
            and len({c.dtype for c in case_config.inputs[:3]}) == 1
        ):
            raise ValueError(
                "generate_multi_tensor_hook: assembled case violates the "
                "coupling contract (t0={}, t1={}, t2={})".format(
                    self._t0_shape, case_config.inputs[1].shape, case_config.inputs[2].shape
                )
            )
        return case_config
