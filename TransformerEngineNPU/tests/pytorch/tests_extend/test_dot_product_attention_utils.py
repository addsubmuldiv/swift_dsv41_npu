# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""B-class UT for attention utils: get_cache_policy, prepare_sbhd/thd_format."""

from __future__ import annotations

import pytest
import torch
from packaging.version import Version as PkgVersion
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils,
    get_cache_policy,
    prepare_sbhd_format,
    prepare_thd_format,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


class TestFlashAttentionUtils:
    """TENPU never ships flash-attn, so every capability flag stays False."""

    def test_flash_attention_is_disabled(self) -> None:
        assert FlashAttentionUtils.is_installed is False
        assert FlashAttentionUtils.v2_plus is False
        assert FlashAttentionUtils.v2_1_plus is False
        assert FlashAttentionUtils.v2_3_plus is False
        assert FlashAttentionUtils.v2_4_plus is False
        assert FlashAttentionUtils.v2_4_1_plus is False
        assert FlashAttentionUtils.v2_5_plus is False
        assert FlashAttentionUtils.v2_5_7_plus is False
        assert FlashAttentionUtils.v2_6_0_plus is False
        assert FlashAttentionUtils.v2_7_0_plus is False
        assert FlashAttentionUtils.v3_is_installed is False
        assert FlashAttentionUtils.v4_is_installed is False
        assert FlashAttentionUtils.version == PkgVersion("0")
        assert FlashAttentionUtils.fa3_version == PkgVersion("0")
        assert FlashAttentionUtils.fa4_version == PkgVersion("0")

    def test_version_setters_are_noop(self) -> None:
        FlashAttentionUtils.set_flash_attention_version()
        FlashAttentionUtils.set_flash_attention_3_params()
        FlashAttentionUtils.set_flash_attention_4_params()
        assert FlashAttentionUtils.is_installed is False
        assert FlashAttentionUtils.v3_is_installed is False
        assert FlashAttentionUtils.v4_is_installed is False

    def test_is_version_supported(self) -> None:
        assert FlashAttentionUtils.is_version_supported(
            PkgVersion("2.5.0"), FlashAttentionUtils.version_required
        )
        assert not FlashAttentionUtils.is_version_supported(
            PkgVersion("2.0.0"), FlashAttentionUtils.version_required
        )
        assert not FlashAttentionUtils.is_version_supported(
            FlashAttentionUtils.max_version, FlashAttentionUtils.version_required
        )


class TestCachePolicy:
    def test_cache_policy_init_no_interval(self) -> None:
        assert get_cache_policy(0, "always", 0) == "always"
        assert get_cache_policy(0, "never", 0) == "never"

    def test_cache_policy_interval_hit(self) -> None:
        # layer_number=1, interval=3 → 1 % 4 == 1 → hit, keeps policy
        assert get_cache_policy(1, "always", 3) == "always"

    def test_cache_policy_interval_miss(self) -> None:
        # layer_number=2, interval=3 → 2 % 4 == 2 != 1 → miss, returns None
        assert get_cache_policy(2, "always", 3) is None

    def test_cache_policy_layer5_interval4(self) -> None:
        # layer=5, interval=4 → 5%5=0 != 1 → None
        assert get_cache_policy(5, "always", 4) is None


class TestPrepareFormats:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_prepare_sbhd_format(self) -> None:
        q = torch.randn(2, 8, 4, 16, device=_NPU)
        k = torch.randn(2, 8, 4, 16, device=_NPU)
        v = torch.randn(2, 8, 4, 16, device=_NPU)
        q_out, k_out, v_out, n_head = prepare_sbhd_format(q, k, v)
        assert q_out.shape == (2, 8, 64)  # rearranged s b (h d)
        assert n_head == 4

    def test_prepare_thd_format_tensor(self) -> None:
        q = torch.randn(8, 4, 16, device=_NPU)
        cu_q = torch.tensor([0, 4, 8], dtype=torch.int32, device=_NPU)
        cu_kv = torch.tensor([0, 4, 8], dtype=torch.int32, device=_NPU)
        n_head, cu_q_out, cu_kv_out = prepare_thd_format(q, cu_q, cu_kv)
        assert n_head == 4
        assert cu_q_out == [0, 4, 8]

    def test_prepare_thd_format_list(self) -> None:
        q = torch.randn(8, 4, 16, device=_NPU)
        n_head, cu_q_out, cu_kv_out = prepare_thd_format(q, [0, 4, 8], [0, 4, 8])
        assert n_head == 4
