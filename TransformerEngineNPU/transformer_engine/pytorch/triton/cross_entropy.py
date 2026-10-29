# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrappers for Parallel Cross Entropy Triton kernels (Ascend NPU)."""

from __future__ import annotations

import os
from functools import reduce
from operator import mul
from typing import Union

import torch
import torch.distributed as dist
import triton

from transformer_engine.pytorch.triton.cross_entropy_kernels import (
    cross_entropy_fwd_single_kernel,
    cross_entropy_kernel,
    element_mul_kernel,
    online_softmax_kernel,
)

# Ascend UB is limited; 4k tiles compile safely on 910B3.
MAX_FUSED_SIZE = 4096
_DEFAULT_NUM_WARPS = 4


def _validate_targets(target: torch.Tensor, vocab_size: int, ignore_idx: int) -> None:
    """Require each label in [0, vocab_size) or equal to ignore_idx."""
    flat = target.reshape(-1)
    valid = (flat == ignore_idx) | ((flat >= 0) & (flat < vocab_size))
    if not bool(valid.all().item()):
        raise ValueError(
            f"target contains out-of-range class ids; expected ignore_idx={ignore_idx} "
            f"or 0 <= y < {vocab_size}"
        )


def _apply_label_smoothing(
    hard_loss: torch.Tensor,
    scaled: torch.Tensor,
    mld: torch.Tensor,
    label_smoothing: float,
    dist_process_group: Union[dist.ProcessGroup, None],
) -> torch.Tensor:
    """Combine hard CE with global scaled_x_sum (all-reduced under TP)."""
    if label_smoothing <= 0:
        return hard_loss
    if dist_process_group is not None:
        dist.all_reduce(scaled, op=dist.ReduceOp.SUM, group=dist_process_group)
    return (1.0 - label_smoothing) * hard_loss + scaled + label_smoothing * mld


def cross_entropy_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
):
    """Forward CE; stage gradients into a separate buffer (inputs unchanged)."""
    if _input.ndim != 3:
        raise ValueError(f"input must be 3D (B,SQ,V) or (SQ,B,V), got shape={tuple(_input.shape)}")
    if target.ndim != 2:
        raise ValueError(
            f"target must be 2D matching input leading dims, got shape={tuple(target.shape)}"
        )

    b0, b1, v = _input.shape
    n_rows = b0 * b1

    if reduce(mul, list(target.size())) != n_rows:
        raise ValueError("Each token needs a target token ID.")

    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)
    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    global_vocab = v * world_size
    _validate_targets(target, global_vocab, ignore_idx)

    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(v))

    loss_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    scaled_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)
    mld_1d = torch.zeros(n_rows, dtype=torch.float32, device=_input.device)

    # Clamp for kernel grad scale only; mean reduction uses raw count below.
    n_non_ignore_raw = (target.reshape(-1) != ignore_idx).to(torch.int64).sum()
    n_non_ignore = torch.clamp(n_non_ignore_raw, min=1).to(device=_input.device)

    # Kernel indexes rows as program_id * stride(-2). Batch slices like
    # base[::2] keep last-2D strides looking dense while skipping rows; always
    # materialize a true (n_rows, V) row-major buffer when needed.
    flat = _input.reshape(n_rows, v)
    if flat.stride(0) != v or flat.stride(1) != 1:
        _input = flat.contiguous().view(b0, b1, v)
    else:
        _input = flat.view(b0, b1, v)
    target = target.reshape(n_rows).contiguous().view(b0, b1)

    # Separate buffer so caller logits are not overwritten (NVTE contract).
    grad_buf = torch.empty_like(_input)

    if world_size == 1:
        cross_entropy_fwd_single_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Grad_ptr=grad_buf,
            Grad_stride=grad_buf.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(-1),
            scaled_ptr=scaled_1d,
            scaled_stride=scaled_1d.stride(-1),
            mld_ptr=mld_1d,
            mld_stride=mld_1d.stride(-1),
            ignore_idx=ignore_idx,
            n_cols=v,
            n_non_ignore=n_non_ignore,
            reduce_loss=reduce_loss,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=block_size,
            num_warps=_DEFAULT_NUM_WARPS,
        )
    else:
        m_d_x_y = torch.zeros(n_rows * 3, dtype=torch.float32, device=_input.device)
        online_softmax_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),
            m_d_X_y_ptr=m_d_x_y,
            m_d_X_y_stride=m_d_x_y.stride(-1),
            rank=rank,
            n_cols=v,
            ignore_idx=ignore_idx,
            BLOCK_SIZE=block_size,
            num_warps=_DEFAULT_NUM_WARPS,
        )
        # Default: Megatron-native style all_reduce of online-softmax stats.
        # Set TE_PARALLEL_CE_ALL_GATHER=1 to keep the legacy all_gather path.
        use_all_gather = os.environ.get("TE_PARALLEL_CE_ALL_GATHER", "0") == "1"
        if use_all_gather:
            m_d_x_y_buf = torch.zeros(
                n_rows * 3 * world_size, dtype=torch.float32, device=_input.device
            )
            dist.all_gather_into_tensor(m_d_x_y_buf, m_d_x_y, group=dist_process_group)
            num_stat_shards = world_size
        else:
            stats = m_d_x_y.view(n_rows, 3)
            m_loc = stats[:, 0].contiguous()
            d_loc = stats[:, 1].contiguous()
            xy_loc = stats[:, 2].contiguous()
            # One MAX all_reduce for (m, X_y); non-owners keep X_y=-inf.
            mx = torch.stack((m_loc, xy_loc), dim=0)
            dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=dist_process_group)
            m_g, xy_g = mx[0], mx[1]
            # Rescale local d to global max, then SUM-reduce (online-softmax merge).
            d_g = d_loc * torch.exp(m_loc - m_g)
            d_g = torch.nan_to_num(d_g, nan=0.0, posinf=0.0, neginf=0.0)
            dist.all_reduce(d_g, op=dist.ReduceOp.SUM, group=dist_process_group)
            m_d_x_y_buf = torch.stack((m_g, d_g, xy_g), dim=1).reshape(-1).contiguous()
            num_stat_shards = 1
        cross_entropy_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride=_input.stride(-2),
            Grad_ptr=grad_buf,
            Grad_stride=grad_buf.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(-1),
            scaled_ptr=scaled_1d,
            scaled_stride=scaled_1d.stride(-1),
            mld_ptr=mld_1d,
            mld_stride=mld_1d.stride(-1),
            m_d_X_y_ptr=m_d_x_y_buf,
            m_d_X_y_stride=m_d_x_y_buf.stride(-1),
            rank=rank,
            world_size=world_size,
            num_stat_shards=num_stat_shards,
            ignore_idx=ignore_idx,
            n_cols=v,
            n_rows=n_rows,
            n_non_ignore=n_non_ignore,
            reduce_loss=reduce_loss,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=block_size,
            num_warps=_DEFAULT_NUM_WARPS,
        )

    loss_1d = _apply_label_smoothing(
        loss_1d, scaled_1d, mld_1d, label_smoothing, dist_process_group
    )

    if reduce_loss:
        # Kernel uses clamped n_non_ignore (>=1) for grad scale; mean uses raw count.
        # All-ignored -> 0 (matches nan_to_num of PyTorch mean CE).
        loss = torch.where(
            n_non_ignore_raw > 0,
            torch.sum(loss_1d) / n_non_ignore_raw.to(dtype=loss_1d.dtype),
            loss_1d.new_zeros(()),
        )
    else:
        loss = torch.reshape(loss_1d, (b0, b1))

    return loss, grad_buf


def cross_entropy_backward(
    _input: torch.Tensor, grad_output: torch.Tensor, is_cg_capturable: bool = False
):
    """Scale staged gradients by upstream grad_output.

    Copies the saved template first so retain_graph / repeated backward does not
    mutate the autograd-saved buffer in place.
    """
    if (not is_cg_capturable) and grad_output.numel() == 1 and float(grad_output.item()) == 1.0:
        return _input.clone()

    # Preserve saved grad template for multi-backward (retain_graph=True).
    out = _input.clone()
    b0, b1, v = out.shape
    n_rows = b0 * b1
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(v))
    go = grad_output.contiguous()
    go_stride = 1 if go.numel() > 1 else 0

    element_mul_kernel[(n_rows,)](
        out,
        out.stride(-2),
        go,
        go_stride,
        v,
        BLOCK_SIZE=block_size,
        num_warps=_DEFAULT_NUM_WARPS,
    )
    return out
