# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct UT for context-parallel helper error paths.

Covers:
- ring_context_parallel: flash_attention_with_alibi_pse NotImplementedError paths
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention.ring_context_parallel import (
    flash_attention_with_alibi_pse,
    flash_attention_with_alibi_pse_grad,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


# ═══════════════════════════════════════════════════════════════════════════════
# Ring Context Parallel - NotImplementedError paths (no distributed needed)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRingCPNotImplemented:
    """Test Ring CP paths that raise NotImplementedError. These don't need DistributedTest."""

    def test_flash_attention_with_alibi_pse_raises(self) -> None:
        """flash_attention_with_alibi_pse should raise NotImplementedError."""
        # Function signature: flash_attention_with_alibi_pse(q_block_id, kv_block_id, cur_qkv, attn_info, s)
        q_block_id = 0
        kv_block_id = 0
        cur_qkv = torch.randn(2, 8, 4, 16, device="npu")
        attn_info = None
        s = None
        with pytest.raises(NotImplementedError):
            flash_attention_with_alibi_pse(q_block_id, kv_block_id, cur_qkv, attn_info, s)

    def test_flash_attention_with_alibi_pse_grad_raises(self) -> None:
        """flash_attention_with_alibi_pse_grad should raise NotImplementedError."""
        # Function signature: flash_attention_with_alibi_pse_grad(q_block_id, kv_block_id, cur_qkv, cur_dout, cur_attn_out, cur_softmax_max, cur_softmax_sum, attn_grad_info, s)
        q_block_id = 0
        kv_block_id = 0
        cur_qkv = torch.randn(2, 8, 4, 16, device="npu")
        cur_dout = torch.randn(2, 8, 4, 16, device="npu")
        cur_attn_out = torch.randn(2, 8, 4, 16, device="npu")
        cur_softmax_max = torch.randn(2, 8, 4, 1, device="npu")
        cur_softmax_sum = torch.randn(2, 8, 4, 1, device="npu")
        attn_grad_info = None
        s = None
        with pytest.raises(NotImplementedError):
            flash_attention_with_alibi_pse_grad(
                q_block_id,
                kv_block_id,
                cur_qkv,
                cur_dout,
                cur_attn_out,
                cur_softmax_max,
                cur_softmax_sum,
                attn_grad_info,
                s,
            )
