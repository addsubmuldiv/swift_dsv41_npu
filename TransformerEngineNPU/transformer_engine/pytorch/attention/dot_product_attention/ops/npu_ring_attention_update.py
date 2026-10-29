# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Provider bridge for the fused CANN RingAttentionUpdate operator."""

from functools import partial

import torch_npu


def _resolve_ring_attention_update(actual_seq_qlen=None, layout='SBH'):
    """Return a six-tensor callable with the requested layout options bound."""
    torch_npu_op = getattr(torch_npu, 'npu_ring_attention_update', None)
    # Only rely on the six-tensor torch_npu interface for default SBH updates.
    # Use the MindSpeed extension to preserve explicit layout/sequence options.
    if torch_npu_op is not None and layout == 'SBH' and actual_seq_qlen is None:
        return torch_npu_op

    try:
        from mindspeed.ops.npu_ring_attention_update import (
            npu_ring_attention_update as mindspeed_ring_attention_update,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f'No compatible fused ring-attention update provider is available for layout={layout!r}. '
            'Install MindSpeed with its npu_ring_attention_update extension for layout and '
            'sequence-length options, or use torch_npu.npu_ring_attention_update for SBH '
            'without actual_seq_qlen.'
        ) from exc
    return partial(mindspeed_ring_attention_update, actual_seq_qlen=actual_seq_qlen, layout=layout)


def npu_ring_attention_update(
    prev_attn_out,
    prev_softmax_max,
    prev_softmax_sum,
    cur_attn_out,
    cur_softmax_max,
    cur_softmax_sum,
    actual_seq_qlen=None,
    layout='SBH',
):
    """Run the fused update through the first available supported provider."""
    op = _resolve_ring_attention_update(actual_seq_qlen=actual_seq_qlen, layout=layout)
    return op(
        prev_attn_out,
        prev_softmax_max,
        prev_softmax_sum,
        cur_attn_out,
        cur_softmax_max,
        cur_softmax_sum,
    )
