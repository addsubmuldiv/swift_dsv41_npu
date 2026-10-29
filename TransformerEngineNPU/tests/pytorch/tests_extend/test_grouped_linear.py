# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Direct coverage for ``module/grouped_linear.py``:
- ``GroupedLinear.__init__`` parameter handling
- ``_has_packed_grouped_weight`` / ``_use_performance_grouped_linear``
- ``_get_weight_tensors`` / ``_get_bias_tensors``
- ``reset_parameters`` / ``set_tensor_parallel_attributes``
- ``make_grouped_weights`` (A-class: packed FP8 path)
- Parameter count and shape assertions
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch.module.grouped_linear import GroupedLinear
from utils import assert_close, dtype_tols, npu_available, to_cpu

pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")

_NPU = torch.device("npu")


def _seed():
    torch.manual_seed(42)
    torch.npu.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# __init__
# ═══════════════════════════════════════════════════════════════════════════════


class TestGroupedLinearInit:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_default_construction(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        assert m.num_gemms == 2
        assert m.in_features == 8
        assert m.out_features == 4
        assert m.use_bias is True
        assert m.return_bias is False
        assert m.parallel_mode is None
        assert m.tp_size == 1

    def test_parameters_registered(self) -> None:
        """num_gemms weight + bias parameters all present."""
        m = GroupedLinear(num_gemms=3, in_features=8, out_features=4)
        for i in range(3):
            assert hasattr(m, f"weight{i}"), f"weight{i} missing"
            assert hasattr(m, f"bias{i}"), f"bias{i} missing"
        # Packed grouped weight attribute does not exist by default
        assert not hasattr(m, "weight")

    def test_parameter_shapes(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        for i in range(2):
            weight = getattr(m, f"weight{i}")
            assert weight.shape == (4, 8)  # out_features × in_features
            bias = getattr(m, f"bias{i}")
            assert bias.shape == (4,)

    def test_no_bias(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4, bias=False)
        assert m.use_bias is False
        # bias attribute registered but with 0 elements
        for i in range(2):
            assert hasattr(m, f"bias{i}")
            assert getattr(m, f"bias{i}").numel() == 0

    def test_ub_overlap_unsupported(self) -> None:
        """ub_overlap_rs and ub_overlap_ag both raise on construction."""
        with pytest.raises(AssertionError, match="Userbuffer overlap"):
            GroupedLinear(num_gemms=2, in_features=8, out_features=4, ub_overlap_rs=True)
        with pytest.raises(AssertionError, match="Userbuffer overlap"):
            GroupedLinear(num_gemms=2, in_features=8, out_features=4, ub_overlap_ag=True)

    def test_bias_with_tp_raises(self) -> None:
        """tp_size>1 + bias=True raises (TP comm handled outside)."""
        with pytest.raises(ValueError, match="doesn't support bias"):
            GroupedLinear(
                num_gemms=2,
                in_features=8,
                out_features=4,
                parallel_mode="column",
                tp_size=2,
                bias=True,
            )

    def test_parallel_mode_column_divides_out(self) -> None:
        m = GroupedLinear(
            num_gemms=2,
            in_features=8,
            out_features=4,
            parallel_mode="column",
            tp_size=2,
            bias=False,
        )
        assert m.out_features == 2  # 4 // 2

    def test_parallel_mode_row_divides_in(self) -> None:
        m = GroupedLinear(
            num_gemms=2, in_features=8, out_features=4, parallel_mode="row", tp_size=2, bias=False
        )
        assert m.in_features == 4  # 8 // 2

    def test_invalid_parallel_mode_raises(self) -> None:
        with pytest.raises(AssertionError, match="parallel_mode"):
            GroupedLinear(num_gemms=2, in_features=8, out_features=4, parallel_mode="invalid")

    def test_sequence_parallel_default_false(self) -> None:
        """tp_size=1 → sequence_parallel stays False even if user passes True."""
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4, sequence_parallel=True)
        assert m.sequence_parallel is False

    def test_sequence_parallel_with_tp(self) -> None:
        m = GroupedLinear(
            num_gemms=2,
            in_features=8,
            out_features=4,
            parallel_mode="column",
            tp_size=2,
            sequence_parallel=True,
            bias=False,
        )
        assert m.sequence_parallel is True

    def test_num_gemms_one(self) -> None:
        """GroupedLinear with num_gemms=1 still works (degenerate case)."""
        m = GroupedLinear(num_gemms=1, in_features=8, out_features=4)
        assert hasattr(m, "weight0")
        assert hasattr(m, "bias0")
        assert not hasattr(m, "weight1")

    def test_offsets_dict_initialized(self) -> None:
        """Internal offset dicts are set up."""
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        assert "input" in m._offsets
        assert "weight" in m._offsets
        assert m._num_fp8_tensors_per_gemm["fwd"] == 3
        assert m._num_fp8_tensors_per_gemm["bwd"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Query methods
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueryMethods:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_has_packed_grouped_weight_default_false(self) -> None:
        """Default construction → no packed 'weight' attribute."""
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        assert m._has_packed_grouped_weight() is False

    def test_use_performance_grouped_linear_default_false(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        assert m._use_performance_grouped_linear() is False

    def test_get_weight_tensors_returns_list(self) -> None:
        m = GroupedLinear(num_gemms=3, in_features=8, out_features=4)
        weights = m._get_weight_tensors()
        assert isinstance(weights, list)
        assert len(weights) == 3
        for w in weights:
            assert w.shape == (4, 8)

    def test_get_bias_tensors_returns_list(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        biases = m._get_bias_tensors(for_linear=True)
        assert isinstance(biases, list)
        assert len(biases) == 2
        for b in biases:
            assert b.shape == (4,)

    def test_get_bias_tensors_no_bias_case(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4, bias=False)
        biases = m._get_bias_tensors(for_linear=True)
        # Each entry is an empty tensor (bias=False → bias.numel() == 0)
        assert len(biases) == 2
        for b in biases:
            assert b.numel() == 0

    def test_get_bias_tensors_for_linear_false(self) -> None:
        """Default for_linear=False path: also returns 2 tensors for num_gemms=2."""
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4, bias=False)
        biases = m._get_bias_tensors(for_linear=False)
        assert len(biases) == 2
        for b in biases:
            assert b.numel() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# reset_parameters / set_tensor_parallel_attributes
# ═══════════════════════════════════════════════════════════════════════════════


class TestResetAndSetTP:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_reset_parameters_runs(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        # Just verify it doesn't raise and parameters remain well-formed
        m.reset_parameters(defer_init=False)
        for i in range(2):
            assert torch.isfinite(getattr(m, f"weight{i}")).all().item()

    def test_set_tensor_parallel_attributes_runs(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        m.set_tensor_parallel_attributes(defer_init=False)
        for i in range(2):
            assert getattr(m, f"weight{i}").tensor_model_parallel is True
            assert getattr(m, f"weight{i}").partition_dim == 0
            assert getattr(m, f"weight{i}").partition_stride == 1

    def test_set_tp_attributes_row_mode_sets_bias_sp(self) -> None:
        """row parallel + sp=True → bias params get sequence_parallel=True attribute."""
        m = GroupedLinear(
            num_gemms=2,
            in_features=8,
            out_features=4,
            parallel_mode="row",
            tp_size=2,
            sequence_parallel=True,
            bias=False,
        )
        # bias=False → no bias attribute at all (not registered as param)
        m.set_tensor_parallel_attributes(defer_init=False)
        for i in range(2):
            assert getattr(m, f"weight{i}").tensor_model_parallel is True
            assert getattr(m, f"weight{i}").partition_dim == 1  # row → dim=1
            assert getattr(m, f"weight{i}").partition_stride == 1


# ═══════════════════════════════════════════════════════════════════════════════
# make_grouped_weights (A-class: FP8 packing)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMakeGroupedWeights:
    def test_defer_init_skips(self) -> None:
        m = GroupedLinear(num_gemms=2, in_features=8, out_features=4)
        m.make_grouped_weights(defer_init=True)
        # No state change (deferred)
        for i in range(2):
            assert hasattr(m, f"weight{i}")


class TestGroupedLinearForward:
    @staticmethod
    def setup_method() -> None:
        _seed()

    def test_forward_and_backward_match_per_group_linear_reference(self) -> None:
        module = GroupedLinear(
            num_gemms=2,
            in_features=8,
            out_features=4,
            device=_NPU,
            params_dtype=torch.float32,
        )
        splits = [2, 3]
        inp = torch.randn(sum(splits), 8, device="cpu").to(_NPU).requires_grad_(True)

        actual = module(inp, splits)
        reference_outputs = []
        start = 0
        for index, split_size in enumerate(splits):
            end = start + split_size
            reference_outputs.append(
                torch.nn.functional.linear(
                    inp[start:end],
                    getattr(module, f"weight{index}"),
                    getattr(module, f"bias{index}"),
                )
            )
            start = end
        expected = torch.cat(reference_outputs, dim=0)
        assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(torch.float32))

        actual.square().mean().backward()
        torch.npu.synchronize()
        assert inp.grad is not None
        assert module.weight0.grad is not None
        assert module.weight1.grad is not None
        assert module.bias0.grad is not None
        assert module.bias1.grad is not None

    def test_dense_fp32_bias_is_not_rounded_to_bf16(self) -> None:
        """Dense FP32 GroupedLinear must preserve the parameter bias exactly."""
        module = GroupedLinear(
            num_gemms=2,
            in_features=8,
            out_features=4,
            device=_NPU,
            params_dtype=torch.float32,
        )
        bias_values = (
            torch.tensor([0.1234567, -0.2345678, 0.3456789, -0.4567891]),
            torch.tensor([-0.1357911, 0.2468023, -0.3579135, 0.4680246]),
        )
        with torch.no_grad():
            module.weight0.zero_()
            module.weight1.zero_()
            module.bias0.copy_(bias_values[0].to(_NPU))
            module.bias1.copy_(bias_values[1].to(_NPU))

        actual = module(torch.zeros(5, 8, dtype=torch.float32, device=_NPU), [2, 3])
        expected = torch.cat((bias_values[0].expand(2, -1), bias_values[1].expand(3, -1)), dim=0)
        assert_close(to_cpu(actual), expected, rtol=0, atol=0)


class TestPerformanceGroupedLinear:
    @staticmethod
    def setup_method() -> None:
        _seed()

    @pytest.mark.parametrize("use_bias", [False, True])
    def test_packed_weight_forward_and_backward_match_per_group_reference(
        self, use_bias: bool
    ) -> None:
        """The packed-weight implementation must match independent per-group linear ops."""
        module = GroupedLinear(
            num_gemms=2,
            in_features=8,
            out_features=4,
            bias=use_bias,
            single_grouped_weight=True,
            device=_NPU,
            params_dtype=torch.float32,
        )
        assert module._use_performance_grouped_linear()

        splits = [2, 3]
        inp = torch.randn(sum(splits), 8, device="cpu").to(_NPU).requires_grad_(True)
        reference_inp = inp.detach().clone().requires_grad_(True)
        reference_weight = module.weight.detach().clone().requires_grad_(True)
        reference_bias = (
            [getattr(module, f"bias{i}").detach().clone().requires_grad_(True) for i in range(2)]
            if use_bias
            else [None, None]
        )

        actual = module(inp, splits)
        expected = torch.cat(
            [
                F.linear(
                    reference_inp[start:end],
                    reference_weight[index],
                    reference_bias[index],
                )
                for index, (start, end) in enumerate(((0, 2), (2, 5)))
            ],
            dim=0,
        )
        assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(torch.float32))

        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.npu.synchronize()
        assert_close(to_cpu(inp.grad), to_cpu(reference_inp.grad), **dtype_tols(torch.float32))
        assert_close(
            to_cpu(module.weight.grad), to_cpu(reference_weight.grad), **dtype_tols(torch.float32)
        )
        if use_bias:
            for index, bias in enumerate(reference_bias):
                assert_close(
                    to_cpu(getattr(module, f"bias{index}").grad),
                    to_cpu(bias.grad),
                    **dtype_tols(torch.float32),
                )

    def test_packed_dense_fp32_bias_is_not_rounded_to_bf16(self) -> None:
        """The packed/performance path must also keep dense FP32 bias."""
        module = GroupedLinear(
            num_gemms=2,
            in_features=8,
            out_features=4,
            single_grouped_weight=True,
            device=_NPU,
            params_dtype=torch.float32,
        )
        assert module._use_performance_grouped_linear()
        bias_values = (
            torch.tensor([0.1234567, -0.2345678, 0.3456789, -0.4567891]),
            torch.tensor([-0.1357911, 0.2468023, -0.3579135, 0.4680246]),
        )
        with torch.no_grad():
            module.weight.zero_()
            module.bias0.copy_(bias_values[0].to(_NPU))
            module.bias1.copy_(bias_values[1].to(_NPU))

        actual = module(torch.zeros(5, 8, dtype=torch.float32, device=_NPU), [2, 3])
        expected = torch.cat((bias_values[0].expand(2, -1), bias_values[1].expand(3, -1)), dim=0)
        assert_close(to_cpu(actual), expected, rtol=0, atol=0)
