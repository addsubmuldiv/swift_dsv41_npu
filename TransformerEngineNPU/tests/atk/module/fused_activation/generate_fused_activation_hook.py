# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for the fusible-activation ops.

ITERATION STATUS: this file is the hook-route candidate, coexisting with
the default full-build generator (``generate_fused_activation.py``,
registry key ``generate_fused_activation_p0`` — curated per-op matrix,
golden-compatible). Once this route is validated through iteration, the
full-build file gets deleted and this one becomes the only generator.
Until then, switch routes by pointing the yaml ``generate:`` field AND
the ``atk case -p`` argument at this file.

Route: the native ATK engine drives sampling; this subclass patches the
cross-input coupling via the official extension points:

    slot 0  x    → recorded; dtype mirrored onto dy/scale
    slot 1  dy   := op OUTPUT shape — DEFERRED to after_case_config (the
                    op_name dependency sits at slot 3, after dy)
    slot 2  scale := x.shape[:-1] / x.dtype (per-row scale; live input for
                    the scaled variants, dead for the rest)
    slot 3  op_name ∈ 9 ops — free attr, but drives everything below

Op taxonomy (mirror of function_fused_activation.py):
    unary  gelu/silu/relu, scaled_srelu        → out = x.shape
    gated  geglu/reglu/swiglu,                 → out = x[:-1] + [x[-1]//2]
           scaled_swiglu/scaled_clamped_qgeglu

Boundaries: any ndim is legal for unary ops (elementwise semantics).
0-dim is legal on both backends for the plain unary ops, but NOT for
scaled_srelu: its forward computes srelu(x) * scale.unsqueeze(-1), whose
broadcast promotes a 0-dim x to output [1], and NVTE's tex.dsrelu then
asserts grad [1] != cached input [] — the hook reconciles 0-dim → [1]
(gated 0-dim → [2], as before). Gated ops carry two kernel hard
constraints — >= 1 dim and an EVEN last dim (odd fails at runtime:
op-plugin kernel error on swiglu family, torch broadcast error on
geglu/qgeglu family) — reconciled in after_case_config (odd last dim →
+1, syncing dy). EMPTY inputs are NPU-only legal: PyTorch's CUDA
allocator returns a null data_ptr for 0-byte allocations, so every
tex.* entry fails NVTE's CheckInputTensor ("Input ... is not
allocated"); the hook rejects empty shapes fail-closed — keep
has_empty: false in the yaml.

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI
default 1234) + identical ATK version ⇒ identical cases.

Yaml prerequisites (already encoded in fused_activation.yaml —
GENERALIZATION.md §5.3): ``shape_distributions`` tuned to module scale;
``max_length`` ≈ 2x the largest intended shape; minimal pinned
``shapes`` on dy/scale. The dim pool [2,3,4] is a sample choice, not an
API limit (any ndim is legal).
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Fail-closed layout check: the hook addresses slots by index, so a yaml
# whose inputs don't match the 4-slot adapter contract must raise here.
_EXPECTED_TYPES = ("tensor", "tensor", "tensor", "attr")

# Mirror of function_fused_activation.py's op registry. Kept local (not
# imported) because ATK loads plugin files independently.
_KNOWN_OPS = (
    "gelu",
    "silu",
    "relu",
    "geglu",
    "reglu",
    "swiglu",
    "scaled_swiglu",
    "scaled_srelu",
    "scaled_clamped_qgeglu",
)
_GATED_OPS = {"geglu", "reglu", "swiglu", "scaled_swiglu", "scaled_clamped_qgeglu"}


@GENERATOR_REGISTRY.register("generate_fused_activation_hook")
class GenerateFusedActivationHook(CaseGenerator):
    """Native sampling + coupling patch for the fusible-activation matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_fused_activation_hook expects the input layout "
                "(x, dy, scale, op_name) with types tensor x3 + attr x1, "
                "got {}".format(got)
            )
        super().__init__(config)
        self._x_shape = None
        self._x_dtype = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        if index == 0:
            # x — 0-dim is recorded as-is (legal for plain unary ops; the
            # gated/scaled_srelu variants are reconciled in
            # after_case_config once op_name is known). Empty shapes are
            # rejected fail-closed: NPU runs them, NVTE cannot (CUDA 0-byte
            # allocations carry a null data_ptr, and every tex.* entry
            # asserts "Input ... is not allocated" in CheckInputTensor).
            if any(dim < 1 for dim in input_case.shape):
                raise ValueError(
                    "generate_fused_activation_hook: x must have every dim >= 1, "
                    "got {} — empty inputs run on NPU only (NVTE: null data_ptr "
                    "for 0-byte CUDA allocations, CheckInputTensor asserts); keep "
                    "has_empty: false".format(input_case.shape)
                )
            self._x_shape = list(input_case.shape)
            self._x_dtype = input_case.dtype
        elif index == 2:
            # scale := x.shape[:-1] / x.dtype (per-row scale; 0-dim for 0/1-D
            # x). The reconciles below only ever touch x's LAST dim (or
            # 0-dim → [1]/[2]), which never changes x.shape[:-1], so this
            # patch stays valid.
            input_case.shape = list(self._x_shape[:-1])
            input_case.dtype = self._x_dtype
        # index 1 (dy) is patched in after_case_config — its op_name
        # dependency (slot 3) comes later in the input order.
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        op_name = str(case_config.inputs[3].range_values)
        if op_name not in _KNOWN_OPS:
            raise ValueError(
                "generate_fused_activation_hook: unknown op {!r} — expected "
                "one of {} (check the yaml op attr values)".format(op_name, sorted(_KNOWN_OPS))
            )
        shape = self._x_shape
        if op_name in _GATED_OPS:
            # Reconcile the gated kernel constraints: >= 1 dim and an even
            # last dim. 0-dim → [2] (minimal legal gated input); odd last
            # dim → +1. x itself must be rewritten (the adapter consumes
            # the reconciled shape); only dy depends on the last dim
            # (scale = x[:-1] is unaffected by both bumps).
            if len(shape) == 0:
                shape = [2]
            elif shape[-1] % 2 == 1:
                shape = shape[:-1] + [shape[-1] + 1]
            self._x_shape = shape
            case_config.inputs[0].shape = list(shape)
            out_shape = shape[:-1] + [shape[-1] // 2]
        elif op_name == "scaled_srelu" and len(shape) == 0:
            # The only SCALED unary op breaks on 0-dim x in NVTE's
            # backward: the forward computes srelu(x) *
            # scale.unsqueeze(-1), whose broadcast promotes the output to
            # [1] while the cached input stays [], so tex.dsrelu asserts
            # on the grad/input shape mismatch. [1] is the nearest legal
            # shape (scale stays 0-dim; dy mirrors x). Plain unary ops
            # keep true 0-dim — legal on both backends.
            shape = [1]
            self._x_shape = shape
            case_config.inputs[0].shape = list(shape)
            out_shape = shape
        else:
            out_shape = shape
        # dy := op output shape / x.dtype
        case_config.inputs[1].shape = out_shape
        case_config.inputs[1].dtype = self._x_dtype
        # Fail-closed consistency check on the assembled case.
        if not (
            case_config.inputs[1].shape == out_shape
            and case_config.inputs[2].shape == list(shape[:-1])
            and len({c.dtype for c in case_config.inputs[:3]}) == 1
            and (op_name not in _GATED_OPS or (len(shape) >= 1 and shape[-1] % 2 == 0))
        ):
            raise ValueError(
                "generate_fused_activation_hook: assembled case violates the "
                "coupling contract (x={}, dy={}, scale={}, op={})".format(
                    shape, case_config.inputs[1].shape, case_config.inputs[2].shape, op_name
                )
            )
        return case_config
