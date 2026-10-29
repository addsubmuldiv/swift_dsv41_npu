# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``transformer_engine.pytorch.router``.

Single route (2026-09-18): this generator replaces the curated P0 matrix —
router has no second generator, so the yaml ``generate:`` field and the
``atk case -p`` argument both point here.

Route: the native ATK engine drives sampling from ``router.yaml``; this
subclass only patches the cross-slot coupling the three variants require.

Three variants share one 9-slot layout (dispatch on slot 4). Slots 0-3 are
tensors, 4-8 attrs; the adapter (``function_router.run_router_case``) reads
them positionally:

  topk      t0=logits[T,E]  t1=expert_bias[E]  g0=[T,E]              g1 dead
            a2=use_pre_softmax(bool)  a3=topk  a4=(groups<<8)|group_topk
  score     t0=logits[T,E]  t1 dead            g0=[T,E]              g1 dead
            a2=topk(int)              a3=0     a4=0
  aux_loss  t0=probs[T,E]   t1=tokens_per_expert seed  g0=[1]         g1 dead
            a2=topk(int)              a3=coeff(float)  a4=0

Coupling this hook owns (everything the native sampler cannot know):

  1. t1 shape := [E]                                     (E = t0.shape[1])
  2. g0/g1 shape := [T, E], or [1] for aux_loss          (T = t0.shape[0])
     g1 is dead in all three variants — kept only so the declared arity
     matches the adapter's 9 positional inputs.
  3. slot 6 dtype is variant-dependent: ``attr_bool`` (use_pre_softmax) for
     topk, ``int`` (topk) otherwise — the same sampled integer drives both
     roles, so the bool role reads its parity (even → True).
     use_pre_softmax is forced False for non-softmax score functions (TENPU's
     ``router.py`` only branches on it for softmax; Megatron never sets it
     elsewhere).
  4. topk is clamped into [1, E] — it is sampled from a shared integer pool
     and must never exceed the expert count.
  5. a4 (topk variant only) := 0 or ``(num_groups << 8) | group_topk`` with
     ``num_groups | E`` and ``group_topk`` the largest divisor of ``topk``
     not exceeding ``num_groups``. Two kernel constraints are packed into
     that rule: group-limited topk requires
     ``group_topk <= topk <= group_topk * (E // num_groups)``
     (``_compute_topk_with_group_limited`` views E as G groups of E/G), and
     the GPU kernel scores each group by its top-``topk // group_topk``
     experts, so ``topk % group_topk == 0``. If no divisor of ``topk`` fits
     the window, the case falls back to plain top-k (a4 = 0). Sampling ``a4``
     directly is not an option: the adapter raises on an unpaired
     groups/group_topk encoding.
  6. aux_loss coeff := float(sampled positive int); 0 would zero the loss and
     its gradient, making the comparison trivial.

Not in the case (derived inside the adapter, identical on both backends):
scaling_factor comes from a fixed table keyed by
``(score_function, bias_enabled)``, and ``bias_enabled`` is read off
``t1.reshape(-1)[0] > 0`` for sigmoid/sqrtsoftplus. The yaml therefore samples
t1's range from two all-positive/all-negative intervals, so the expert_bias
branch is covered deterministically rather than by luck of the draw.

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI default
1234) + identical ATK version ⇒ identical cases.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# 9 slots: t0, t1, g0, g1 (tensors) + variant, score_function, a2, a3, a4.
_EXPECTED_TYPES = ("tensor", "tensor", "tensor", "tensor", "attr", "attr", "attr", "attr", "attr")

_KNOWN_VARIANTS = ("topk", "score", "aux_loss")
_KNOWN_SCORE_FUNCTIONS = ("softmax", "sigmoid", "sqrtsoftplus")

# Low-precision + sparse top-k is not a well-defined observable across backends:
# two identical logits in a row let torch.topk (TENPU) and the CUDA kernel pick
# different experts, and routing_map is compared exactly. Measured on the NPU
# host (2026-09-18) on bf16: E in {32, 64, 1024} produced 5-137 tied rows per
# case, and even E = 16 still produced one tied row in 7 seeds — it is a
# property of the sampled data, not of either implementation, so no expert-count
# threshold makes it safe. fp16 has a wider mantissa than bf16 but is still
# finite-precision, so the same rule applies to every non-fp32 case: selection
# cases are forced to a DENSE selection (topk == E), where the top-k boundary
# disappears and tie-breaking cannot matter. The dtype-specific parts that ARE
# comparable — the fp32 upcast of the logits, the cast of the outputs back to
# the low-precision dtype, and the contract's dtype fields — are still exercised,
# and sparse selection stays covered on the fp32 cases.


@GENERATOR_REGISTRY.register("generate_router_hook")
class GenerateRouterHook(CaseGenerator):
    """Native sampling + per-variant coupling patch for the router matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_router_hook expects the input layout "
                "(t0, t1, g0, g1, variant, score_function, a2, a3, a4) with "
                "types tensor x4 + attr x5, got {}".format(got)
            )
        super().__init__(config)
        self._t0_shape = None
        self._t0_dtype = None
        self._variant = None
        self._score_function = None

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        # Slots 4/5 (variant, score_function) fire after t0, and every patch
        # below needs all three — so capture here, patch in after_case_config.
        if index == 0:
            shape = list(input_case.shape)
            # Fail-closed: [T, E] with both dims >= 1 (E=0 is illegal
            # outright, T=0 was only NPU-probed). Selection legality
            # (1 <= topk <= E) is enforced at case assembly — no separate
            # expert-count floor. Keep the boundary track off — an empty
            # or 0-D t0 is untested for this module.
            if len(shape) != 2:
                raise ValueError(
                    "generate_router_hook: t0 must be 2-D [tokens, experts], "
                    "got shape {} — keep extra_numbers: 0 and "
                    "has_empty/has_scalar/has_infnan: false".format(input_case.shape)
                )
            if any(d < 1 for d in shape):
                raise ValueError(
                    "generate_router_hook: degenerate t0 shape {} — the empty "
                    "track is not usable for router: E=0 is illegal outright "
                    "(top-k has no expert to select) and ATK zeroes a RANDOM "
                    "dim, while T=0 was only NPU-probed (GPU unverified — "
                    "the §3#28 lesson in ATK_REPORT)".format(input_case.shape)
                )
            self._t0_shape = shape
            self._t0_dtype = input_case.dtype
        elif index == 4:
            variant = str(input_case.range_values)
            if variant not in _KNOWN_VARIANTS:
                raise ValueError(
                    "generate_router_hook: unknown variant {!r} — expected one "
                    "of {} (check router.yaml slot 4)".format(variant, list(_KNOWN_VARIANTS))
                )
            self._variant = variant
        elif index == 5:
            score_function = str(input_case.range_values)
            if score_function not in _KNOWN_SCORE_FUNCTIONS:
                raise ValueError(
                    "generate_router_hook: unknown score_function {!r} — "
                    "expected one of {} (check router.yaml slot 5)".format(
                        score_function, list(_KNOWN_SCORE_FUNCTIONS)
                    )
                )
            self._score_function = score_function
        return input_case

    @staticmethod
    def _clamp_topk(raw: int, experts: int) -> int:
        return max(1, min(int(raw), experts))

    @staticmethod
    def _reconcile_groups(raw: int, experts: int) -> int:
        """0 (no group routing) or a divisor of E, nearest to the sampled value."""
        if raw <= 0:
            return 0
        groups = min(int(raw), experts)
        while groups > 1 and experts % groups:
            groups -= 1
        return groups if groups > 1 and experts % groups == 0 else 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        variant = self._variant
        score_function = self._score_function
        tokens, experts = self._t0_shape
        dtype = self._t0_dtype
        aux_loss = variant == "aux_loss"
        # bf16 selection cases go dense (module-level note above).
        dense_selection = dtype != "fp32" and variant in ("topk", "score")

        # t1 mirrors the expert count; g0/g1 mirror t0 (aux_loss seeds a scalar
        # loss instead). All three are fp32 on both backends: expert_bias is
        # consumed as fp32 by the kernel, the grad seeds are ATK-side only, and
        # the aux tokens-seed is synthesized as fp32 by the adapter.
        for slot in (1, 2, 3):
            case_config.inputs[slot].shape = (
                [experts] if slot == 1 else ([1] if aux_loss else [tokens, experts])
            )
            case_config.inputs[slot].dtype = "fp32"

        if variant == "topk":
            topk = (
                experts
                if dense_selection
                else self._clamp_topk(case_config.inputs[7].range_values, experts)
            )
            # One sampled integer carries two roles across variants: its parity
            # is the bool role (even → True), its value the int role.
            parity_source = int(case_config.inputs[6].range_values)
            use_pre_softmax = score_function == "softmax" and parity_source % 2 == 0
            case_config.inputs[6].dtype = "attr_bool"
            case_config.inputs[6].range_values = use_pre_softmax
            case_config.inputs[7].dtype = "int"
            case_config.inputs[7].range_values = topk
            groups = self._reconcile_groups(int(case_config.inputs[8].range_values), experts)
            # The GPU kernel scores each group by its top-(topk // group_topk)
            # experts (see _compute_topk_with_group_limited), so topk must be a
            # multiple of group_topk: pick the largest divisor of topk not
            # exceeding the group count. If even that breaks the selection
            # window, no divisor can satisfy it (the window widens with
            # group_topk) and the case falls back to plain top-k.
            group_topk = 0
            if groups:
                group_topk = min(groups, topk)
                while topk % group_topk:
                    group_topk -= 1
                if topk > group_topk * (experts // groups):
                    groups = 0
                    group_topk = 0
            case_config.inputs[8].dtype = "int"
            case_config.inputs[8].range_values = (groups << 8) | group_topk if groups else 0
        else:
            topk = (
                experts
                if dense_selection
                else self._clamp_topk(case_config.inputs[6].range_values, experts)
            )
            case_config.inputs[6].dtype = "int"
            case_config.inputs[6].range_values = topk
            case_config.inputs[8].dtype = "int"
            case_config.inputs[8].range_values = 0
            if aux_loss:
                # The pool is a positive integer; 0 would zero the loss and its
                # gradient.
                coeff = float(case_config.inputs[7].range_values)
                case_config.inputs[7].dtype = "float"
                case_config.inputs[7].range_values = coeff if coeff > 0 else 1.0
            else:
                case_config.inputs[7].dtype = "int"
                case_config.inputs[7].range_values = 0

        inputs = case_config.inputs
        groups = inputs[8].range_values >> 8
        group_topk = inputs[8].range_values & 0xFF
        group_window_ok = not groups or (
            1 <= group_topk <= groups <= experts
            and experts % groups == 0
            and group_topk <= topk <= group_topk * (experts // groups)
            and topk % group_topk == 0
        )
        # Fail-closed consistency check on the assembled case.
        if not (
            inputs[1].shape == [experts]
            and inputs[2].shape == inputs[3].shape
            and inputs[2].shape == ([1] if aux_loss else [tokens, experts])
            and len({inputs[i].dtype for i in (1, 2, 3)}) == 1
            and 1 <= topk <= experts
            and (
                variant != "topk"
                or (
                    inputs[6].dtype == "attr_bool"
                    and isinstance(inputs[6].range_values, bool)
                    and inputs[6].range_values == use_pre_softmax
                )
            )
            and group_window_ok
            and (not aux_loss or inputs[7].dtype == "float")
        ):
            raise ValueError(
                "generate_router_hook: assembled case violates the router "
                "coupling contract (variant={}, score={}, t0={}, t1={}, "
                "g0={}, topk={}, a4={})".format(
                    variant,
                    score_function,
                    self._t0_shape,
                    inputs[1].shape,
                    inputs[2].shape,
                    topk,
                    inputs[8].range_values,
                )
            )
        return case_config
