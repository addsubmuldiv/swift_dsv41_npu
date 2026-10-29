# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``ops/_common.py`` helpers:
- ``is_quantized_tensor``
- ``maybe_dequantize``
- ``maybe_autocast_dtype``
- ``get_main_grad_from_param``
- ``get_accumulate_flag_in_param``
- ``get_dummy_wgrads_for_params``
"""

from __future__ import annotations

import pytest
import torch

from transformer_engine.pytorch.ops._common import (
    get_accumulate_flag_in_param,
    get_dummy_wgrads_for_params,
    get_main_grad_from_param,
    is_quantized_tensor,
    maybe_autocast_dtype,
    maybe_dequantize,
)
from utils import npu_available

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# is_quantized_tensor
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsQuantizedTensor:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_plain_tensor_returns_false(self) -> None:
        t = torch.randn(3, 4, device=_NPU)
        assert is_quantized_tensor(t) is False

    def test_int_tensor_returns_false(self) -> None:
        t = torch.zeros(5, dtype=torch.long, device="cpu")
        assert is_quantized_tensor(t) is False


# ═══════════════════════════════════════════════════════════════════════════════
# maybe_dequantize
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaybeDequantize:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_plain_tensor_dtype_preserved(self) -> None:
        t = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        out = maybe_dequantize(t)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, t)

    def test_plain_tensor_dtype_cast(self) -> None:
        t = torch.randn(3, 4, device=_NPU, dtype=torch.float32)
        out = maybe_dequantize(t, dtype=torch.float16)
        assert out.dtype == torch.float16
        torch.testing.assert_close(out, t.to(torch.float16))

    def test_plain_tensor_no_dtype_returns_as_is(self) -> None:
        t = torch.randn(3, 4, device=_NPU, dtype=torch.bfloat16)
        out = maybe_dequantize(t)
        assert out.dtype == torch.bfloat16

    def test_non_contiguous_tensor_made_contiguous(self) -> None:
        t = torch.randn(4, 3, device=_NPU)
        t_t = t.t()  # Non-contiguous view
        assert not t_t.is_contiguous()
        out = maybe_dequantize(t_t)
        assert out.is_contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
# maybe_autocast_dtype
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaybeAutocastDtype:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_returns_default_when_autocast_disabled(self) -> None:
        out = maybe_autocast_dtype(default_dtype=torch.float16)
        assert out == torch.float16

    def test_returns_default_none_canonicalized(self) -> None:
        out = maybe_autocast_dtype()
        # canonicalize_dtype(None) → torch.get_default_dtype()
        assert out == torch.get_default_dtype()

    def test_explicit_default_bfloat16(self) -> None:
        out = maybe_autocast_dtype(default_dtype=torch.bfloat16)
        assert out == torch.bfloat16


# ═══════════════════════════════════════════════════════════════════════════════
# get_main_grad_from_param
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetMainGradFromParam:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_param_without_main_grad_raises(self) -> None:
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        with pytest.raises(RuntimeError, match="does not have a valid main_grad"):
            get_main_grad_from_param(p)

    def test_param_with_none_main_grad_raises(self) -> None:
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        p.main_grad = None
        with pytest.raises(RuntimeError, match="does not have a valid main_grad"):
            get_main_grad_from_param(p)

    def test_param_with_valid_main_grad_returns_it(self) -> None:
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        p.main_grad = torch.randn(3, 4, device=_NPU)
        out = get_main_grad_from_param(p)
        # Function must return the exact same tensor object, not a copy
        assert out is p.main_grad

    def test_op_label_in_error_message(self) -> None:
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        with pytest.raises(RuntimeError, match="my_op operation"):
            get_main_grad_from_param(p, op_label="my_op")


# ═══════════════════════════════════════════════════════════════════════════════
# get_accumulate_flag_in_param
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetAccumulateFlagInParam:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_default_returns_true(self) -> None:
        """Without overwrite_main_grad flag set → accumulate."""
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        assert get_accumulate_flag_in_param(p) is True

    def test_with_overwrite_returns_false(self) -> None:
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        p.overwrite_main_grad = True
        assert get_accumulate_flag_in_param(p) is False

    def test_explicit_overwrite_false_returns_true(self) -> None:
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        p.overwrite_main_grad = False
        assert get_accumulate_flag_in_param(p) is True


# ═══════════════════════════════════════════════════════════════════════════════
# get_dummy_wgrads_for_params
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetDummyWgradsForParams:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_param_without_grad_added_returns_none(self) -> None:
        """Param lacking ``grad_added_to_main_grad`` attr → entry is None."""
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        result = get_dummy_wgrads_for_params([p])
        assert result == [None]

    def test_param_with_grad_added_returns_dummy(self) -> None:
        p = torch.nn.Parameter(torch.randn(3, 4, device=_NPU))
        p.grad_added_to_main_grad = True
        p.main_grad = torch.randn(3, 4, device=_NPU)
        result = get_dummy_wgrads_for_params([p])
        assert len(result) == 1
        assert result[0] is not None
        assert result[0].shape == (3, 4)

    def test_mixed_params(self) -> None:
        """List of mixed params — first with marker, second without."""
        p1 = torch.nn.Parameter(torch.randn(2, 3, device=_NPU))
        p1.grad_added_to_main_grad = True
        p1.main_grad = torch.randn(2, 3, device=_NPU)
        p2 = torch.nn.Parameter(torch.randn(2, 3, device=_NPU))
        # p2 has no grad_added_to_main_grad
        result = get_dummy_wgrads_for_params([p1, p2])
        assert len(result) == 2
        assert result[0] is not None
        assert result[1] is None

    def test_empty_list_returns_empty(self) -> None:
        assert not get_dummy_wgrads_for_params([])
