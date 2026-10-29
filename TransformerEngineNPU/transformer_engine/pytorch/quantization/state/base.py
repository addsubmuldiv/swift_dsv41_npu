# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import abc
import importlib
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from transformer_engine.common.recipe import Recipe
    from ..quantizer_role import QuantizerRole


_QUANTIZATION_SCOPE_MODULE_TYPES = {"moe-only": "grouped_linear", "linear-only": "linear"}


class RecipeState(abc.ABC):
    """Base class for quantization recipe state.

    Subclasses that own mutable training buffers (e.g. delayed scaling's
    ``scale`` / ``amax_history``) MUST list them in
    :attr:`_persistent_state_buffers`. These buffers are preserved across
    role-driven rebuilds and post-checkpoint resume via
    :meth:`inherit_state_from`. Stateless subclasses leave the attribute
    empty.
    """

    roles: Optional[List["QuantizerRole"]]
    mode: str
    recipe: "Recipe"

    # Names of mutable torch.Tensor attributes that represent persistent
    # training state (e.g. running scale, amax history). The default
    # ``inherit_state_from`` rebinds these from a predecessor RecipeState
    # so external references (e.g. ``FP8GlobalStateManager`` reduction
    # buffers) keep pointing at the same backing tensor.
    _persistent_state_buffers: Tuple[str, ...] = ()

    @staticmethod
    def _validate_roles(
        roles: Optional[List["QuantizerRole"]],
        num_quantizers: int,
    ) -> None:
        if roles is not None and len(roles) != num_quantizers:
            raise ValueError(
                "RecipeState requires roles to match num_quantizers "
                f"({len(roles)=} vs {num_quantizers=})"
            )

    @staticmethod
    def create(
        recipe: "Recipe",
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional["torch.device"] = None,
        roles: Optional[List["QuantizerRole"]] = None,
    ) -> "RecipeState":
        """Factory method to create the state for a quantization recipe

        Parameters
        ----------
        recipe: Recipe
            Quantization recipe.
        mode: {"forward", "backward"}
            Training stage where quantization will be performed.
        num_quantizers: int, default = 1
            Number of quantizers to create state for.
        device: torch.device, default = default NPU device
            Device for quantized tensors.
        roles: list of QuantizerRole, optional
            Semantic roles for each quantizer slot. When provided, must
            have length ``num_quantizers``.

        Returns
        -------
        RecipeState:
            Quantization recipe state.

        """
        if recipe.hif8_delayed():
            module_name, cls_name = "hif8_delayed_state", "HIF8DelayedScalingRecipeState"
        elif recipe.delayed():
            module_name, cls_name = "delayed_state", "DelayedScalingRecipeState"
        elif recipe.mxfp4():
            module_name, cls_name = "mxfp4_state", "MXFP4BlockScalingRecipeState"
        elif recipe.w4a8():
            module_name, cls_name = "w4a8_state", "W4A8BlockScalingRecipeState"
        elif recipe.w4a16():
            module_name, cls_name = "w4a16_state", "W4A16BlockScalingRecipeState"
        elif recipe.mxfp8_32x32():
            module_name, cls_name = "mxfp8_32x32_state", "MXFP832x32BlockScalingRecipeState"
        elif recipe.mxfp8():
            module_name, cls_name = "mxfp8_state", "MXFP8BlockScalingRecipeState"
        elif recipe.float8_current_scaling():
            module_name, cls_name = "current_state", "Float8CurrentScalingRecipeState"
        elif recipe.float8_block_scaling():
            module_name, cls_name = "float8_block_state", "Float8BlockScalingRecipeState"
        elif recipe.custom():
            module_name, cls_name = "custom_state", "CustomRecipeState"
        else:
            raise ValueError(f"{recipe.__class__.__name__} is not supported")
        cls = getattr(importlib.import_module(f"{__package__}.{module_name}"), cls_name)
        return cls(
            recipe,
            mode=mode,
            num_quantizers=num_quantizers,
            device=device,
            roles=roles,
        )

    def _quantization_scope_mask(self, num_quantizers: int) -> List[bool]:
        """Select slots before constructing quantizers to avoid unused objects."""
        scope = self.recipe.quantization_scope
        if scope == "all":
            return [True] * num_quantizers
        if scope == "close":
            return [False] * num_quantizers
        if self.roles is None:
            raise ValueError("Scoped quantization requires quantizer roles")
        self._validate_roles(self.roles, num_quantizers)
        module_type = _QUANTIZATION_SCOPE_MODULE_TYPES[scope]
        return [role.module_type == module_type for role in self.roles]

    def inherit_state_from(self, other: "RecipeState") -> bool:
        """Take over persistent training buffers from a predecessor state.

        Used when a :class:`RecipeState` is being replaced (e.g. role-driven
        rebuild, post-checkpoint resume) but its mutable buffers must
        survive. The default implementation rebinds attributes listed in
        :attr:`_persistent_state_buffers` to ``other``'s tensor objects.
        Rebinding (rather than copying values) ensures any external
        references -- most importantly the
        :class:`FP8GlobalStateManager` reduction buffers -- keep pointing
        at storage that is also visible to this state's quantizers, so
        amax reductions and quantization stay consistent.

        Subclasses with composed sub-states (e.g. :class:`CustomRecipeState`
        owning an inner :class:`DelayedScalingRecipeState`) override this
        to recurse / stash for later use during ``make_quantizers``.

        Returns
        -------
        bool
            ``True`` if any persistent buffer was inherited; ``False`` if
            the states are incompatible (different class, mismatched
            shapes / dtypes) and a fresh state should be used instead.
        """
        if type(self) is not type(other):
            return False
        if not self._persistent_state_buffers:
            return False
        # Two-pass: first validate every named buffer, then rebind. This
        # avoids leaving ``self`` half-populated when an incompatible
        # predecessor is encountered mid-loop.
        for buf_name in self._persistent_state_buffers:
            src = getattr(other, buf_name)
            dst = getattr(self, buf_name)
            if src.shape != dst.shape or src.dtype != dst.dtype:
                return False
        for buf_name in self._persistent_state_buffers:
            setattr(self, buf_name, getattr(other, buf_name))
        return True

    @abc.abstractmethod
    def make_quantizers(self) -> list:
        pass
