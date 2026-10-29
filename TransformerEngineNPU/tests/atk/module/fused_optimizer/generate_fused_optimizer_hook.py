# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for the fused optimizers.

ITERATION STATUS: this file is the hook-route candidate, coexisting with
the default full-build generator (``generate_fused_optimizer.py``,
registry key ``generate_fused_optimizer_p0`` — curated matrix,
golden-compatible). Once this route is validated through iteration, the
full-build file gets deleted and this one becomes the only generator.
Until then, switch routes by pointing the yaml ``generate:`` field AND
the ``atk case -p`` argument at this file.

Route: the native ATK engine drives sampling; this subclass patches the
coupling via the official extension points:

    slot 0  params → recorded; dtype mirrored onto grads
    slot 1  grads  := params.shape / params.dtype
    slot 2  optim_type ∈ {adam, sgd} — validated fail-closed
    slot 3  lr / slot 4 weight_decay / slot 5 steps — live for both
    slot 6  b0 / slot 7 b1 / slot 8 eps     — live for adam, dead for sgd
    slot 9  momentum / slot 10 nesterov     — live for sgd, dead for adam

Probe-verified boundaries (2026-09-16, NPU, fwd+steps): any ndim is legal
(1-4-D tested); EMPTY params are legal (steps no-op cleanly). Two SGD
hyperparameter hard constraints (both raise at runtime):
  - momentum < 0 is unconditionally illegal ("Invalid momentum value");
  - nesterov=True requires momentum > 0 ("Nesterov momentum requires a
    momentum and zero dampening").
Because nothing is unconditionally illegal in the TENSOR shape, the hook
has no shape guard; the two attr constraints are reconciled in
after_case_config (negative momentum → 0.9 with nesterov else 0.0;
nesterov with zero momentum → 0.9), preserving the nesterov branch
instead of erroring out. The yaml sample range deliberately includes
momentum=0.0 so the reconcile path is exercised routinely.

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI
default 1234) + identical ATK version ⇒ identical cases.

Yaml prerequisites (already encoded in fused_optimizer.yaml —
GENERALIZATION.md §5.3): ``shape_distributions`` tuned to module scale;
``max_length`` ≈ 2x the largest intended shape; minimal pinned
``shapes`` on grads. The [2] dim pool is a sample choice (optimizer
params are 2-D weights; the API accepts any ndim), not an API limit.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Fail-closed layout check: the hook addresses slots by index, so a yaml
# whose inputs don't match the 11-slot adapter contract must raise here.
_EXPECTED_TYPES = ("tensor", "tensor") + ("attr",) * 9

_KNOWN_TYPES = ("adam", "sgd")


@GENERATOR_REGISTRY.register("generate_fused_optimizer_hook")
class GenerateFusedOptimizerHook(CaseGenerator):
    """Native sampling + coupling patch for the fused-optimizer matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_fused_optimizer_hook expects the input layout "
                "(params, grads, optim_type, lr, weight_decay, steps, b0, "
                "b1, eps, momentum, nesterov) with types tensor x2 + "
                "attr x9, got {}".format(got)
            )
        super().__init__(config)
        self._p_shape = None
        self._p_dtype = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        if index == 0:
            # params — record only. No shape guard: any ndim and empty
            # params are legal (probe 2026-09-16).
            self._p_shape = list(input_case.shape)
            self._p_dtype = input_case.dtype
        elif index == 1:
            # grads := params.shape / params.dtype
            input_case.shape = list(self._p_shape)
            input_case.dtype = self._p_dtype
        # index 2-10 (attrs): free, reconciled in after_case_config.
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        optim_type = str(case_config.inputs[2].range_values)
        if optim_type not in _KNOWN_TYPES:
            raise ValueError(
                "generate_fused_optimizer_hook: unknown optim_type {!r} — "
                "expected one of {} (check the yaml optim_type attr "
                "values)".format(optim_type, list(_KNOWN_TYPES))
            )
        # SGD hyperparameter reconcile (probe-verified hard constraints):
        # momentum < 0 is unconditionally illegal; nesterov needs
        # momentum > 0. Reconcile to the nearest legal value, preserving
        # the nesterov branch when it was sampled. Attr values may arrive
        # as intervals (list form) when the yaml omits ranges — ATK's
        # default [-5, 5] contains illegal values and the eventual draw is
        # not fixed at generation time, so collapse to a known-legal
        # scalar instead of crashing on float(list).
        if optim_type == "sgd":
            raw_momentum = case_config.inputs[9].range_values
            raw_nesterov = case_config.inputs[10].range_values
            if isinstance(raw_nesterov, (list, tuple)):
                nesterov = False
                case_config.inputs[10].range_values = False
            else:
                nesterov = bool(raw_nesterov)
            if isinstance(raw_momentum, (list, tuple)):
                momentum = 0.9
            else:
                momentum = float(raw_momentum)
                if momentum < 0:
                    momentum = 0.9 if nesterov else 0.0
                elif nesterov and momentum == 0:
                    momentum = 0.9
            case_config.inputs[9].range_values = momentum
        # Fail-closed consistency check on the assembled case.
        if not (
            case_config.inputs[1].shape == self._p_shape
            and case_config.inputs[1].dtype == self._p_dtype
        ):
            raise ValueError(
                "generate_fused_optimizer_hook: assembled case violates the "
                "coupling contract (params={}, grads={})".format(
                    self._p_shape, case_config.inputs[1].shape
                )
            )
        return case_config
