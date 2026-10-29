# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Extra coverage for ``permutation.py`` MoE operations, complementing the
existing ``test_permutation.py`` which focuses on permute/unpermute
end-to-end via te_moe aten ops.
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.permutation import (
    _convert_tensors_to_fp32_if_needed,
    _restore_original_dtype,
    moe_permute,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# Dtype conversion helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestDtypeConversion:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_convert_fp32_passthrough(self) -> None:
        """Both fp32 → no conversion, returns (t1, t2, None)."""
        t1 = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        t2 = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        out1, out2, orig_dtype = _convert_tensors_to_fp32_if_needed(t1, t2)
        assert out1.dtype == torch.float32
        assert out2.dtype == torch.float32
        assert orig_dtype is None

    def test_convert_bf16_with_fp32_to_fp32(self) -> None:
        """Mixed (bf16, fp32) → both fp32, returns bf16 as orig. Shape preserved."""
        t1 = torch.randn(3, 4, device=_NPU, dtype=torch.bfloat16)
        t2 = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        out1, out2, orig_dtype = _convert_tensors_to_fp32_if_needed(t1, t2)
        assert out1.dtype == torch.float32
        assert out2.dtype == torch.float32
        assert orig_dtype == torch.bfloat16
        # Shape preserved
        assert out1.shape == t1.shape
        assert out2.shape == t2.shape

    def test_convert_fp16_with_fp32_to_fp32(self) -> None:
        """Mixed (fp16, fp32) → both fp32, returns fp16 as orig."""
        t1 = torch.randn(3, 4, device=_NPU, dtype=torch.float16)
        t2 = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        out1, out2, orig_dtype = _convert_tensors_to_fp32_if_needed(t1, t2)
        assert out1.dtype == torch.float32
        assert out2.dtype == torch.float32
        assert orig_dtype == torch.float16
        assert out1.shape == t1.shape

    def test_restore_dtype_bf16(self) -> None:
        """Restore converts both back to bf16. Shape preserved."""
        t1 = torch.randn(3, 4, device=_NPU, dtype=torch.bfloat16)
        t2 = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        out1, out2, _ = _convert_tensors_to_fp32_if_needed(t1, t2)
        rest1, rest2 = _restore_original_dtype(out1, out2, torch.bfloat16)
        assert rest1.dtype == torch.bfloat16
        assert rest2.dtype == torch.bfloat16
        assert rest1.shape == t1.shape

    def test_restore_dtype_none_passthrough(self) -> None:
        """original_dtype=None → _restore_original_dtype returns tensors unchanged."""
        t1 = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        t2 = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        out1, out2, _ = _convert_tensors_to_fp32_if_needed(t1, t2)
        rest1, rest2 = _restore_original_dtype(out1, out2, None)
        assert rest1.dtype == torch.float32
        assert rest2.dtype == torch.float32

    def test_one_tensor_none_returns_both_none(self) -> None:
        """Mixed (Tensor, None) → returns (tensor, None, None)."""
        t1 = torch.randn(3, 4, device=_NPU, dtype=torch.bfloat16)
        out1, out2, orig = _convert_tensors_to_fp32_if_needed(t1, None)
        assert out1 is t1
        assert out2 is None
        assert orig is None


# ═══════════════════════════════════════════════════════════════════════════════
# moe_permute invalid map_type
# ═══════════════════════════════════════════════════════════════════════════════


class TestPermuteInvalidMapType:
    def test_invalid_map_type_raises(self) -> None:
        inp = torch.randn(4, 8, device=_NPU)
        routing_map = torch.zeros(4, 2, device=_NPU, dtype=torch.int32)
        with pytest.raises(ValueError, match="map_type"):
            moe_permute(inp, routing_map, num_out_tokens=4, map_type="invalid")
