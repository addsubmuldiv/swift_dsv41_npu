# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Parallel Cross Entropy Loss API for Ascend NPU (NVTE v2.17 semantics)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional
import warnings

import torch
import torch.nn.functional as F

__all__ = [
    "parallel_cross_entropy",
    "set_force_triton",
]

_TRITON_MIN_VOCAB = int(os.environ.get("TE_PARALLEL_CE_TRITON_MIN_VOCAB", "16384"))
# UT toggles (no os.environ on ACL perf hot path).
ACL_CHECK_TARGETS = False
# Default True: mean CE with all-ignored labels returns 0 (NVTE / Triton semantics),
# not NaN from bare F.cross_entropy. Keep the name for UT that still toggles it.
ACL_ZERO_ALL_IGNORED = True
# None = follow env (lru_cache); bool = UT/bench override.
_FORCE_TRITON_OVERRIDE = None  # type: Optional[bool]


@lru_cache(maxsize=1)
def _env_force_triton() -> bool:
    return os.environ.get("TE_PARALLEL_CE_FORCE_TRITON", "0") == "1"


def set_force_triton(enabled: bool) -> None:
    """UT/benchmark toggle for Triton route (does not require process restart)."""
    global _FORCE_TRITON_OVERRIDE
    _FORCE_TRITON_OVERRIDE = bool(enabled)
    _env_force_triton.cache_clear()


def _force_triton() -> bool:
    if _FORCE_TRITON_OVERRIDE is not None:
        return bool(_FORCE_TRITON_OVERRIDE)
    return _env_force_triton()


def _prefer_triton(inp: torch.Tensor, dist_process_group) -> bool:
    if _force_triton():
        return True
    if dist_process_group is not None:
        return True
    return False


def _validate_targets(target: torch.Tensor, vocab_size: int, ignore_idx: int) -> None:
    flat = target.reshape(-1)
    valid = (flat == ignore_idx) | ((flat >= 0) & (flat < vocab_size))
    if not bool(valid.all().item()):
        raise ValueError(
            f"target contains out-of-range class ids; expected ignore_idx={ignore_idx} "
            f"or 0 <= y < {vocab_size}"
        )


class CrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp,
        target,
        label_smoothing=0.0,
        reduce_loss=False,
        dist_process_group=None,
        ignore_idx=-100,
        is_cg_capturable=False,
    ):
        import transformer_engine.pytorch.triton.cross_entropy as triton_cross_entropy

        loss, grad_buf = triton_cross_entropy.cross_entropy_forward(
            inp,
            target,
            label_smoothing,
            reduce_loss,
            dist_process_group,
            ignore_idx,
        )
        ctx.save_for_backward(grad_buf.detach())
        ctx.is_cg_capturable = is_cg_capturable
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        import transformer_engine.pytorch.triton.cross_entropy as triton_cross_entropy

        (grad_buf,) = ctx.saved_tensors
        grad_buf = triton_cross_entropy.cross_entropy_backward(
            grad_buf, grad_output, ctx.is_cg_capturable
        )
        return grad_buf, None, None, None, None, None, None


def parallel_cross_entropy(
    inp: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    reduce_loss: bool = False,
    dist_process_group: Optional[torch.distributed.ProcessGroup] = None,
    ignore_idx: int = -100,
    is_cg_capturable: bool = False,
    *,
    _input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross Entropy with optional vocabulary-parallel reduction.

    Single-device default: thin PyTorch/ACL CE (task-book latency).
    Triton: vocab-parallel or ``TE_PARALLEL_CE_FORCE_TRITON=1``.
    Host target-range checks run only on the Triton path (and ACL when
    ``TE_PARALLEL_CE_CHECK_TARGETS=1`` for UT).
    """
    if _input is not None:
        warnings.warn(
            "The '_input' parameter is deprecated. Please use 'inp' instead.",
            FutureWarning,
        )
        inp = _input

    if inp.ndim != 3 or target.ndim != 2:
        raise ValueError("inp must be 3D and target 2D")

    # Hot path first: single-device ACL/PyTorch CE — match bare F.cross_entropy.
    if not _prefer_triton(inp, dist_process_group):
        b0, b1, v = inp.shape
        if target.numel() != b0 * b1:
            raise ValueError("Each token needs a target token ID.")
        if ACL_CHECK_TARGETS:
            _validate_targets(target, v, ignore_idx)
        loss = F.cross_entropy(
            inp.reshape(b0 * b1, v).float(),
            target.reshape(b0 * b1),
            ignore_index=ignore_idx,
            label_smoothing=label_smoothing,
            reduction="mean" if reduce_loss else "none",
        )
        if not reduce_loss:
            loss = loss.reshape(b0, b1)
        elif ACL_ZERO_ALL_IGNORED:
            # All-ignored mean CE is NaN in bare PyTorch; map to 0 like Triton/NVTE.
            loss = torch.nan_to_num(loss, nan=0.0)
        return loss

    local_v = inp.shape[-1]
    if dist_process_group is None:
        global_vocab = local_v
    else:
        global_vocab = local_v * torch.distributed.get_world_size(dist_process_group)
    _validate_targets(target, global_vocab, ignore_idx)
    return CrossEntropyFunction.apply(
        inp,
        target,
        label_smoothing,
        reduce_loss,
        dist_process_group,
        ignore_idx,
        is_cg_capturable,
    )


def __getattr__(name: str):
    # Compat: older UT reads ``_FORCE_TRITON`` as a module attribute.
    if name == "_FORCE_TRITON":
        return _force_triton()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
