# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Regression tests for attn_mask_type='padding_causal' on NPU.

Megatron-core passes attn_mask_type='padding_causal' for packed (qkv_format='thd')
sequences. It must apply causal masking within each packed segment, identical to
'causal'. These tests guard against the regression where get_fa_config() let
'padding_causal' fall through to sparse_mode=0 and DotProductAttention skipped the
compressed causal mask, making packed/thd attention non-causal (full bidirectional).
"""

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention.backends import get_fa_config
from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
    DotProductAttention,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")


def test_get_fa_config_padding_causal_maps_to_causal():
    # padding_causal must use top-left causal (sparse_mode=2), same as causal
    assert get_fa_config("causal")["sparse_mode"] == 2
    assert get_fa_config("padding_causal")["sparse_mode"] == 2
    assert get_fa_config("padding,causal")["sparse_mode"] == 2
    assert get_fa_config("causal,padding")["sparse_mode"] == 2
    # unchanged mappings
    assert get_fa_config("no_mask")["sparse_mode"] == 0
    assert get_fa_config("general")["sparse_mode"] == 1


def _ref_per_segment_causal(q, k, v, hq, hkv, d, scale, segs):
    """Pure-fp32 per-segment causal attention ground truth."""
    rep = hq // hkv
    outs = []
    for a, b in segs:
        s = b - a
        qf, kf, vf = q[a:b].float(), k[a:b].float(), v[a:b].float()
        o = torch.empty(s, hq, d, device=q.device, dtype=torch.float32)
        cm = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        for h in range(hq):
            sc = (qf[:, h] @ kf[:, h // rep].transpose(0, 1)) * scale
            sc = sc.masked_fill(cm, float("-inf"))
            o[:, h] = torch.softmax(sc, dim=-1) @ vf[:, h // rep]
        outs.append(o.reshape(s, hq * d))
    return torch.cat(outs, dim=0)


def test_thd_padding_causal_matches_per_segment_causal():
    """Packed (thd) padding_causal forward == per-segment causal ground truth.

    Before the fix, attention was bidirectional and the first token of each segment
    attended to future tokens, giving an O(1) error vs. this reference.
    """
    torch.manual_seed(0)
    dev = "npu"
    hq, hkv, d = 4, 2, 64  # GQA, like real decoder layers
    scale = 1.0 / (d**0.5)
    L1, L2 = 3, 5
    T = L1 + L2
    segs = [(0, L1), (L1, T)]
    cu = torch.tensor([0, L1, T], dtype=torch.int32, device=dev)

    q = torch.randn(T, hq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(T, hkv, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(T, hkv, d, device=dev, dtype=torch.bfloat16)

    dpa = DotProductAttention(
        num_attention_heads=hq,
        kv_channels=d,
        num_gqa_groups=hkv,
        qkv_format="thd",
        attn_mask_type="padding_causal",
        softmax_scale=scale,
    )
    out = dpa(
        q,
        k,
        v,
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(L1, L2),
        max_seqlen_kv=max(L1, L2),
        attn_mask_type="padding_causal",
    ).reshape(T, hq * d)

    ref = _ref_per_segment_causal(q, k, v, hq, hkv, d, scale, segs)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
