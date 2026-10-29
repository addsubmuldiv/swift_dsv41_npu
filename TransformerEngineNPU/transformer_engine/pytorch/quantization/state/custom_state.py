# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import warnings
from typing import List, Optional

import torch

from transformer_engine.common.recipe import CustomRecipe, DelayedScaling

from ..quantizer_role import DelayedScalingRequest, QuantizerRole
from ..utils import get_fp8_torch_dtype
from .base import RecipeState
from .delayed_state import DelayedScalingRecipeState


def _handle_delayed_scaling_requests(
    raw_quantizers: list,
    device: torch.device,
    mode: str,
    existing_ds_state: Optional[DelayedScalingRecipeState] = None,
) -> Optional[DelayedScalingRecipeState]:
    """Replace ``DelayedScalingRequest`` entries in ``raw_quantizers``
    with real ``Float8Quantizer`` instances managed by a shared ``DelayedScalingRecipeState``.

    Returns the ``DelayedScalingRecipeState`` (or ``None`` if no DS requests found).
    """
    ds_indices = [i for i, q in enumerate(raw_quantizers) if isinstance(q, DelayedScalingRequest)]
    if not ds_indices:
        return None

    # All DS requests must share identical parameters.
    ref = raw_quantizers[ds_indices[0]]
    for i in ds_indices[1:]:
        req = raw_quantizers[i]
        if req != ref:
            raise ValueError(
                "All DelayedScalingRequest instances within a CustomRecipeState "
                "must have identical parameters."
            )

    # Build a DelayedScaling recipe from the request params.
    ds_recipe = DelayedScaling(
        margin=ref.margin,
        fp8_format=ref.fp8_format,
        amax_history_len=ref.amax_history_len,
        amax_compute_algo=ref.amax_compute_algo,
        scaling_factor_compute_algo=ref.scaling_factor_compute_algo,
        reduce_amax=ref.reduce_amax,
    )

    ds_state: Optional[DelayedScalingRecipeState] = None
    expected_dtype = get_fp8_torch_dtype(ds_recipe, mode == "forward")
    n_ds_quantizers = len(ds_indices)
    if existing_ds_state is not None:
        # Reuse a compatible predecessor DS state. All three of dtype,
        # number of DS slots, and amax_history_len must match -- otherwise
        # accumulated scale / amax_history would be silently kept under a
        # new recipe with a different shape or format, producing wrong
        # scaling factors on the next forward pass.
        if (
            existing_ds_state.num_quantizers == n_ds_quantizers
            and existing_ds_state.dtype == expected_dtype
            and existing_ds_state.amax_history.shape[0] == ds_recipe.amax_history_len
        ):
            ds_state = existing_ds_state
            # Buffers survive the rebuild, but scalar parameters must follow
            # the latest requests. Without this rebind, a changed margin /
            # amax_compute_algo / reduce_amax / scaling_factor_compute_algo
            # would be silently ignored because reductions read them from
            # ``ds_state.recipe`` (via ``_inner_delayed_scaling_recipe``).
            # This mirrors upstream TE, where persistent scale/amax buffers
            # are kept while recipe parameters always track the recipe
            # passed to the active autocast context.
            ds_state.recipe = ds_recipe
    if ds_state is None:
        ds_state = DelayedScalingRecipeState(
            ds_recipe,
            mode=mode,
            num_quantizers=n_ds_quantizers,
            device=device,
        )

    # Replace DelayedScalingRequest entries with real quantizers.
    quantizers = ds_state.make_quantizers()
    for idx, q in zip(ds_indices, quantizers):
        raw_quantizers[idx] = q

    # ``dtype`` is set by ``DelayedScalingRecipeState.__init__`` for fresh
    # states and verified equal to ``expected_dtype`` above for reused
    # states, so no extra assignment is needed here.
    return ds_state


class CustomRecipeState(RecipeState):
    """State for CustomRecipe: produce quantizers per tensor role.

    Supports stateless quantizers (directly returned by ``qfactory``) and
    stateful quantizers (e.g. delayed scaling) via ``DelayedScalingRequest``.
    """

    recipe: CustomRecipe
    mode: str
    num_quantizers: int
    device: Optional[torch.device]
    roles: Optional[List[QuantizerRole]]

    # Composed delayed-scaling sub-state.
    _ds_state: Optional[DelayedScalingRecipeState]
    _ds_state_to_inherit: Optional[DelayedScalingRecipeState]

    def __init__(
        self,
        recipe: CustomRecipe,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[QuantizerRole]] = None,
    ) -> None:
        self._validate_roles(roles, num_quantizers)
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = roles
        if device is None:
            device = torch.device("npu")
        self.device = device

        self._ds_state = None
        self._ds_state_to_inherit = None

        if getattr(recipe, "qfactory", None) is None:
            raise ValueError("CustomRecipe requires `qfactory`.")

    def make_quantizers(self) -> list:
        qfactory = self.recipe.qfactory

        roles = self.roles
        if roles is None:
            warnings.warn(
                "CustomRecipeState: no QuantizerRole list provided. "
                "Falling back to bare QuantizerRole() defaults.",
                stacklevel=2,
            )
            roles = [QuantizerRole() for _ in range(self.num_quantizers)]

        raw = [qfactory(roles[i]) for i in range(self.num_quantizers)]
        for i, q in enumerate(raw):
            if q is None:
                raise ValueError(
                    f"CustomRecipe qfactory returned None for slot {i} "
                    f"(role={roles[i]}). "
                    "Every slot must return a Quantizer instance or QuantizerRequest."
                )

        # Handle stateful requests (DelayedScalingRequest).
        existing_ds_state = self._ds_state_to_inherit
        self._ds_state_to_inherit = None
        self._ds_state = _handle_delayed_scaling_requests(
            raw,
            self.device,
            self.mode,
            existing_ds_state=existing_ds_state,
        )

        return raw

    def inherit_state_from(self, other: "RecipeState") -> bool:
        if not isinstance(other, CustomRecipeState):
            return False
        inherited_any = False
        if other._ds_state is not None:
            self._ds_state_to_inherit = other._ds_state
            inherited_any = True
        return inherited_any

    @property
    def _has_delayed_scaling(self) -> bool:
        return self._ds_state is not None

    @property
    def amax_history(self) -> Optional[torch.Tensor]:
        return self._ds_state.amax_history if self._ds_state else None

    @property
    def scale(self) -> Optional[torch.Tensor]:
        return self._ds_state.scale if self._ds_state else None

    @property
    def _inner_delayed_scaling_recipe(self) -> Optional[DelayedScaling]:
        return self._ds_state.recipe if self._ds_state else None
