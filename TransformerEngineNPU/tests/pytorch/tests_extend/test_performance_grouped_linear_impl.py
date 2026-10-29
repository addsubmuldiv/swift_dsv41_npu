# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``module/performance_grouped_linear_impl.py``:
- ``GroupedLinearArgs`` dataclass: required fields + 6 single-quantizer properties
- ``__post_init__`` derivation of group_split / group_list
- ``_PerformanceGroupedLinear`` static forward shape validation
- Property accessor edge cases (empty list → None)
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.module.performance_grouped_linear_impl import (
    GroupedLinearArgs,
    _PerformanceGroupedLinear,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


def _make_args(m_splits=(4, 4, 4)) -> GroupedLinearArgs:
    """Build a GroupedLinearArgs with sensible defaults."""
    return GroupedLinearArgs(
        m_splits=list(m_splits),
        use_bias=True,
        is_first_microbatch=True,
        fp8=False,
        fp8_calibration=False,
        wgrad_store=None,
        input_quantizers=[],
        weight_quantizers=[],
        output_quantizers=[],
        grad_input_quantizers=[],
        grad_weight_quantizers=[],
        grad_output_quantizers=[],
        fuse_wgrad_accumulation=False,
        cpu_offloading=False,
        sequence_parallel=False,
        activation_dtype=torch.float32,
        is_grad_enabled=True,
        weight_workspaces=[],
        cache_weight=False,
        skip_fp8_weight_update=False,
        save_original_input=False,
        debug=False,
        backward_override=None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GroupedLinearArgs: field assignment + post_init derivation
# ═══════════════════════════════════════════════════════════════════════════════


class TestGroupedLinearArgsFields:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_basic_construction(self) -> None:
        args = _make_args()
        assert args.use_bias is True
        assert args.fp8 is False
        assert args.fp8_calibration is False
        assert args.fuse_wgrad_accumulation is False
        assert args.cpu_offloading is False
        assert args.sequence_parallel is False
        assert args.is_grad_enabled is True
        assert args.debug is False
        assert args.skip_fp8_weight_update is False
        assert args.save_original_input is False
        assert args.cache_weight is False

    def test_activation_dtype_preserved(self) -> None:
        args = _make_args()
        assert args.activation_dtype == torch.float32

    def test_post_init_derives_group_split(self) -> None:
        """__post_init__ converts m_splits to a tensor on npu:int64."""
        args = _make_args(m_splits=(2, 3, 5))
        assert isinstance(args.group_split, torch.Tensor)
        assert args.group_split.device.type == "npu"
        assert args.group_split.dtype == torch.int64
        torch.testing.assert_close(
            args.group_split.cpu(), torch.tensor([2, 3, 5], dtype=torch.int64)
        )

    def test_post_init_derives_group_list(self) -> None:
        """__post_init__ builds group_list as cumsum of m_splits."""
        args = _make_args(m_splits=(2, 3, 5))
        # cumsum = [2, 5, 10]
        torch.testing.assert_close(
            args.group_list.cpu(), torch.tensor([2, 5, 10], dtype=torch.int64)
        )

    def test_post_init_single_gemm(self) -> None:
        """m_splits with one element → group_list = [m_splits[0]]."""
        args = _make_args(m_splits=(8,))
        torch.testing.assert_close(args.group_list.cpu(), torch.tensor([8], dtype=torch.int64))

    def test_post_init_zero_splits(self) -> None:
        """Empty m_splits → group_split and group_list are empty tensors (no error)."""
        args = _make_args(m_splits=())
        assert args.group_split.numel() == 0
        assert args.group_list.numel() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# GroupedLinearArgs: single-quantizer properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestGroupedLinearArgsQuantizerProperties:
    def test_input_quantizer_empty_returns_none(self) -> None:
        args = _make_args()
        assert args.input_quantizer is None

    def test_weight_quantizer_empty_returns_none(self) -> None:
        args = _make_args()
        assert args.weight_quantizer is None

    def test_output_quantizer_empty_returns_none(self) -> None:
        args = _make_args()
        assert args.output_quantizer is None

    def test_grad_input_quantizer_empty_returns_none(self) -> None:
        args = _make_args()
        assert args.grad_input_quantizer is None

    def test_grad_output_quantizer_empty_returns_none(self) -> None:
        args = _make_args()
        assert args.grad_output_quantizer is None

    def test_grad_weight_quantizer_empty_returns_none(self) -> None:
        args = _make_args()
        assert args.grad_weight_quantizer is None

    def test_input_quantizer_with_one_returns_first(self) -> None:
        sentinel = object()
        args = _make_args()
        args.input_quantizers = [sentinel]
        assert args.input_quantizer is sentinel

    def test_weight_quantizer_with_one_returns_first(self) -> None:
        sentinel = object()
        args = _make_args()
        args.weight_quantizers = [sentinel]
        assert args.weight_quantizer is sentinel

    def test_output_quantizer_with_one_returns_first(self) -> None:
        sentinel = object()
        args = _make_args()
        args.output_quantizers = [sentinel]
        assert args.output_quantizer is sentinel

    def test_weight_workspace_empty_returns_none(self) -> None:
        args = _make_args()
        assert args.weight_workspace is None

    def test_weight_workspace_with_one_returns_first(self) -> None:
        sentinel = object()
        args = _make_args()
        args.weight_workspaces = [sentinel]
        assert args.weight_workspace is sentinel


# ═══════════════════════════════════════════════════════════════════════════════
# _PerformanceGroupedLinear.forward shape validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceGroupedLinearShapeCheck:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_in_features_mismatch_raises(self) -> None:
        """Input last dim != weight in_features → ValueError."""
        ctx = None
        args = _make_args()
        weight = torch.randn(2, 4, 8, device=_NPU)  # num_gemms=2, out=4, in=8
        bias = torch.randn(2, 4, device=_NPU)
        inp = torch.randn(3, 6, device=_NPU)  # last dim 6 != in_features 8
        with pytest.raises(ValueError, match="not compatible"):
            _PerformanceGroupedLinear.forward(ctx, inp, args, weight, bias)
