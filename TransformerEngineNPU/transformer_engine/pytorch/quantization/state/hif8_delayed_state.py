# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Independent state machine for HiF8 delayed scaling."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Optional

import torch

from transformer_engine.common.recipe import (
    HIF8DelayedScaling,
    QuantizerRole as HIF8QuantizerRole,
)

from ..utils import get_fp8_max, get_fp8_torch_dtype
from .base import RecipeState


class HIF8DelayedScalingRecipeState(RecipeState):
    """Per-module HiF8 state shared by all copies of its quantizers."""

    _persistent_state_buffers = (
        "scale",
        "scale_inv",
        "amax_history",
        "history_write_index",
        "pending_amax",
        "completed_steps",
        "region_active",
        "region_sample",
        "region_warmup",
    )
    _recompute_state_buffers = (
        "scale",
        "scale_inv",
        "pending_amax",
        "completed_steps",
        "region_active",
        "region_sample",
        "region_warmup",
    )

    def __init__(
        self,
        recipe: HIF8DelayedScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
        roles: Optional[List[Any]] = None,
    ) -> None:
        if mode not in ("forward", "backward"):
            raise ValueError(f"Unsupported quantization mode: {mode}")
        if roles is None or len(roles) != num_quantizers:
            raise ValueError("HIF8DelayedScaling requires one explicit QuantizerRole per quantizer")
        if device is None:
            device = torch.device("npu")

        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.roles = list(roles)
        self.margin_roles = tuple(self._normalize_margin_role(role) for role in roles)
        self.dtype = get_fp8_torch_dtype(recipe, mode == "forward")
        self.device = device

        self.scale = torch.ones(num_quantizers, dtype=torch.float32, device=device)
        self.scale_inv = torch.ones(num_quantizers, dtype=torch.float32, device=device)
        self.amax_history = torch.zeros(
            recipe.amax_history_len,
            num_quantizers,
            dtype=torch.float32,
            device=device,
        )
        self.history_write_index = torch.zeros(num_quantizers, dtype=torch.int64)
        self.pending_amax = torch.zeros(num_quantizers, dtype=torch.float32, device=device)

        margin_factors = [recipe.margin_factor(role) for role in self.margin_roles]
        self.margin_factors = torch.tensor(margin_factors, dtype=torch.float32, device=device)
        fp8_max = float(get_fp8_max(recipe, mode == "forward"))
        self.scale_numerator = torch.full(
            (num_quantizers,),
            fp8_max,
            dtype=torch.float32,
            device=device,
        )
        self.scale_numerator.div_(self.margin_factors)

        # Cadence is host-side policy state. Quantizers share this RecipeState,
        # so copies cannot accidentally advance independent counters.
        self.completed_steps = torch.zeros(num_quantizers, dtype=torch.int64)
        self.region_active = torch.zeros(num_quantizers, dtype=torch.bool)
        self.region_sample = torch.zeros(num_quantizers, dtype=torch.bool)
        self.region_warmup = torch.zeros(num_quantizers, dtype=torch.bool)
        self.tracking_enabled = False

    @staticmethod
    def _normalize_margin_role(role: Any) -> HIF8QuantizerRole:
        """Map TE's rich quantizer role to the three HiF8 margin classes."""
        if isinstance(role, HIF8QuantizerRole):
            return role

        tensor_type = getattr(role, "tensor_type", "")
        if tensor_type == "weight":
            return HIF8QuantizerRole.WEIGHT
        if tensor_type.startswith("grad"):
            return HIF8QuantizerRole.GRAD
        if tensor_type in ("input", "output"):
            return HIF8QuantizerRole.INPUT
        raise ValueError(f"Unsupported HiF8 quantizer role: {role!r}")

    def _matches_except_history_len(self, recipe: HIF8DelayedScaling) -> bool:
        """Whether ``recipe`` differs from the tracked policy only in history length."""
        if type(recipe) is not type(self.recipe):
            return False
        if recipe.amax_history_len == self.recipe.amax_history_len:
            return False
        return all(
            getattr(recipe, field.name) == getattr(self.recipe, field.name)
            for field in fields(recipe)
            if field.name != "amax_history_len"
        )

    def inherit_state_from(self, other: RecipeState) -> bool:
        """Preserve cadence when recipe and HiF8 margin roles still match.

        A history-length-only change resizes the predecessor's ring buffer in
        place instead of discarding the running scale and cadence.
        """
        if not isinstance(other, HIF8DelayedScalingRecipeState):
            return False
        if self.margin_roles != other.margin_roles:
            return False
        if self.recipe != other.recipe:
            if not self._matches_except_history_len(other.recipe):
                return False
            other.resize_amax_history(self.recipe.amax_history_len)
        inherited = super().inherit_state_from(other)
        if inherited:
            self.tracking_enabled = other.tracking_enabled
        return inherited

    def make_quantizers(self) -> list:
        from ...tensor.hif8_tensor import HIF8Quantizer

        return [HIF8Quantizer(self, i) for i in range(self.num_quantizers)]

    def _activate_region(self, index: int) -> tuple[bool, bool]:
        if not bool(self.region_active[index]):
            region = int(self.completed_steps[index]) + 1
            warmup = region <= self.recipe.amax_collect_interval
            sample = warmup or (
                (region - self.recipe.amax_collect_interval) % self.recipe.scale_update_interval
                == 0
            )
            self.region_active[index] = True
            self.region_warmup[index] = warmup
            self.region_sample[index] = sample
        return bool(self.region_warmup[index]), bool(self.region_sample[index])

    def enable_tracking(self) -> None:
        """Enable cadence tracking once the state is attached to an FP8 region."""
        self.tracking_enabled = True

    @staticmethod
    def _tensor_amax(tensor: torch.Tensor) -> torch.Tensor:
        amin, amax = tensor.aminmax()
        return torch.maximum(-amin, amax).to(dtype=torch.float32).reshape(())

    def observe_and_get_scale_inv(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        """Observe one quantization and return the multiplicative NPU scale."""
        if not self.tracking_enabled:
            current_amax = self._tensor_amax(tensor)
            scale_inv = current_amax / self.scale_numerator[index]
            valid = (current_amax > 0.0) & torch.isfinite(current_amax)
            return torch.where(valid, scale_inv, torch.ones_like(scale_inv)).reshape((1,))
        warmup, sample = self._activate_region(index)
        current_amax = None
        if sample:
            current_amax = self._tensor_amax(tensor)
            self.pending_amax[index].copy_(torch.maximum(self.pending_amax[index], current_amax))

        if warmup:
            numerator = self.scale_numerator[index]
            scale_inv = current_amax / numerator
            valid = (current_amax > 0.0) & torch.isfinite(current_amax)
            scale_inv = torch.where(valid, scale_inv, torch.ones_like(scale_inv))
            return scale_inv.reshape((1,))

        return self.scale_inv[index].reshape((1,))

    def observe_for_calibration(self, index: int, tensor: torch.Tensor) -> None:
        """Advance region bookkeeping and collect amax without quantizing."""
        if not self.tracking_enabled:
            return
        _, sample = self._activate_region(index)
        if sample:
            current_amax = self._tensor_amax(tensor)
            self.pending_amax[index].copy_(torch.maximum(self.pending_amax[index], current_amax))

    def has_active_region(self) -> bool:
        return bool(torch.any(self.region_active))

    def sampled_indices(self) -> List[int]:
        return [
            index
            for index in range(self.num_quantizers)
            if bool(self.region_active[index]) and bool(self.region_sample[index])
        ]

    def _get_amax(self, index: int) -> torch.Tensor:
        history = self.amax_history[:, index]
        algo = self.recipe.amax_compute_algo
        if callable(algo):
            return algo(history)
        if algo == "max":
            return torch.max(history)
        if algo == "mean":
            return torch.mean(history)
        return self.pending_amax[index].clone()

    def _update_scale(self, index: int) -> torch.Tensor:
        amax = self._get_amax(index)
        old_scale = self.scale[index]
        new_scale = self.scale_numerator[index] / amax
        new_scale = torch.where((amax > 0.0) & torch.isfinite(amax), new_scale, old_scale)
        new_scale = torch.where(
            torch.isinf(new_scale),
            torch.full_like(new_scale, torch.finfo(torch.float32).max),
            new_scale,
        )
        self.scale[index].copy_(new_scale)
        return torch.reciprocal(new_scale)

    def _record_sample(self, index: int, update_scale: bool) -> Optional[torch.Tensor]:
        history = self.amax_history[:, index]
        if callable(self.recipe.amax_compute_algo):
            # Preserve the legacy callable contract: newest sample at row 0,
            # followed by the existing chronological history.
            history[0].copy_(self.pending_amax[index])
            updated_scale_inv = None
            if update_scale:
                updated_scale_inv = self._update_scale(index)
            if history.shape[0] > 1:
                history.copy_(torch.roll(history, -1, 0))
            history[0].zero_()
            return updated_scale_inv

        write_index = int(self.history_write_index[index])
        history[write_index].copy_(self.pending_amax[index])
        updated_scale_inv = None
        if update_scale:
            updated_scale_inv = self._update_scale(index)
        self.history_write_index[index] = (write_index + 1) % history.shape[0]
        return updated_scale_inv

    def commit_region(self) -> None:
        """Commit one outer FP8 region after any distributed amax reduction."""
        new_scale_inv = None
        for index in range(self.num_quantizers):
            if not bool(self.region_active[index]):
                continue
            region = int(self.completed_steps[index]) + 1
            if bool(self.region_sample[index]):
                update_scale = region >= self.recipe.amax_collect_interval
                updated_scale_inv = self._record_sample(index, update_scale)
                if updated_scale_inv is not None:
                    if new_scale_inv is None:
                        # Quantized tensors retain their scale-inverse tensor.
                        # Create one new cache version per region so no update
                        # can mutate metadata held by older outputs.
                        new_scale_inv = self.scale_inv.clone()
                    new_scale_inv[index].copy_(updated_scale_inv)
            self.completed_steps[index] = region
            self.pending_amax[index].zero_()
            self.region_active[index] = False
            self.region_sample[index] = False
            self.region_warmup[index] = False
        if new_scale_inv is not None:
            self.scale_inv = new_scale_inv

    def reset_amax_history(self, *, preserve_scale: bool = True) -> None:
        """Clear history and restart current-scaling warmup."""
        self.amax_history.zero_()
        self.history_write_index.zero_()
        self.pending_amax.zero_()
        self.completed_steps.zero_()
        self.region_active.zero_()
        self.region_sample.zero_()
        self.region_warmup.zero_()
        if not preserve_scale:
            self.scale.fill_(1.0)
            self.scale_inv = torch.ones_like(self.scale_inv)

    def resize_amax_history(self, length: int) -> None:
        """Resize amax history while preserving scale and region cadence.

        HiF8 keeps history in a per-quantizer ring buffer, so shrinking keeps
        the newest samples and growing zero-fills older rows. Callable
        ``amax_compute_algo`` keeps the ``DelayedScaling`` layout where row 0
        is the newest sample.
        """
        current = self.amax_history.shape[0]
        if length == current:
            return
        if callable(self.recipe.amax_compute_algo):
            if length < current:
                self.amax_history = self.amax_history[:length].clone()
            else:
                self.amax_history = torch.nn.functional.pad(
                    self.amax_history, (0, 0, 0, length - current)
                )
            return

        device = self.amax_history.device
        write_index = self.history_write_index.to(device=device, dtype=torch.int64) % current
        columns = torch.arange(self.num_quantizers, device=device).unsqueeze(1)
        order = (write_index.unsqueeze(1) + torch.arange(current, device=device)) % current
        # Oldest-to-newest samples for each quantizer's own ring position.
        chronological = self.amax_history[order, columns]
        if length < current:
            resized = chronological[:, -length:]
            next_write = torch.zeros(self.num_quantizers, dtype=torch.int64)
        else:
            resized = torch.cat(
                (
                    chronological,
                    torch.zeros(self.num_quantizers, length - current, device=device),
                ),
                dim=1,
            )
            next_write = torch.full((self.num_quantizers,), current, dtype=torch.int64)
        self.amax_history = resized.t().contiguous()
        self.history_write_index = next_write

    def snapshot(self) -> Dict[str, torch.Tensor]:
        """Clone all mutable policy state for a persistent checkpoint."""
        return {
            "scale": self.scale.clone(),
            "scale_inv": self.scale_inv.clone(),
            "amax_history": self.amax_history.clone(),
            "history_write_index": self.history_write_index.clone(),
            "pending_amax": self.pending_amax.clone(),
            "completed_steps": self.completed_steps.clone(),
            "region_active": self.region_active.clone(),
            "region_sample": self.region_sample.clone(),
            "region_warmup": self.region_warmup.clone(),
        }

    def snapshot_for_recompute(self) -> Dict[str, torch.Tensor]:
        """Clone only state read or mutated during activation recompute."""
        return {name: getattr(self, name).clone() for name in self._recompute_state_buffers}

    def load_recompute_snapshot(self, snapshot: Dict[str, torch.Tensor]) -> None:
        """Restore a lightweight activation-recompute snapshot."""
        self.scale.copy_(snapshot["scale"])
        new_scale_inv = torch.empty_like(self.scale_inv)
        new_scale_inv.copy_(snapshot["scale_inv"])
        self.scale_inv = new_scale_inv
        for name in (
            "pending_amax",
            "completed_steps",
            "region_active",
            "region_sample",
            "region_warmup",
        ):
            getattr(self, name).copy_(snapshot[name])

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore a snapshot created by :meth:`snapshot`."""
        self.scale.copy_(snapshot["scale"])
        if "scale_inv" in snapshot:
            new_scale_inv = torch.empty_like(self.scale_inv)
            new_scale_inv.copy_(snapshot["scale_inv"])
            self.scale_inv = new_scale_inv
        else:
            # Checkpoints created before scale-inverse caching only contain scale.
            self.scale_inv = torch.reciprocal(self.scale)
        if "history_write_index" in snapshot:
            self.history_write_index.copy_(snapshot["history_write_index"])
        else:
            # Legacy history is still valid ring storage for max/mean; start
            # overwriting from row 0 on the next sample.
            self.history_write_index.zero_()
        for name in (
            "amax_history",
            "pending_amax",
            "completed_steps",
            "region_active",
            "region_sample",
            "region_warmup",
        ):
            getattr(self, name).copy_(snapshot[name])


# Compatibility with the spelling used by the earlier prototype.
Hif8DelayedScalingRecipeState = HIF8DelayedScalingRecipeState
