# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Multi-card tests for ``backward_override`` under TP + sequence parallelism.

Exercises the CommOverlapOps fused comm+GEMM paths (allgather_matmul fprop,
matmul_reduce_scatter dgrad) with unquantized override operands, which is the
configuration where the override previously force-disabled every overlap.
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from distributed_testing import DistributedTest
from test_backward_override import _layernorm_backward_reference
from utils import assert_close

from transformer_engine.common import recipe
from transformer_engine.pytorch import LayerNormLinear, Linear, autocast
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    is_mxfp8_available,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

_mxfp8_available, _reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)


def _reset_fp8_state() -> None:
    FP8GlobalStateManager.reset()


class TestLinearBackwardOverrideTP(DistributedTest):
    """Column-parallel + sequence-parallel Linear under backward_override.

    Module-level fused comm configuration (ub_overlap_ag / ub_overlap_rs_dgrad
    both enabled). Per upstream NVTE alignment, the override keeps the fused AG
    fprop for the forward GEMM but recomputes the backward in plain precision,
    so dgrad/wgrad take the unfused path.

    Validation: forward is compared bitwise between the default (quantized)
    run and the override run — identical quantized operands through the same
    fused kernel. Dgrad/wgrad differ numerically by design (unquantized
    operands), so they are validated against per-rank plain-math references
    built from this rank's shards with FP8-scale tolerances.
    """

    world_size = 2

    def _make_layer(self, override, ub_overlap_ag, ub_overlap_rs_dgrad):
        # Fused comm+GEMM kernels (aclnnAllGatherMatmulV2 / mm_reduce_scatter)
        # require every contraction axis >= 256: in_features for fprop,
        # out_features / tp_size for dgrad.
        torch.manual_seed(1234)  # identical params on every rank
        layer = Linear(
            256,
            512,
            bias=False,
            parallel_mode="column",
            sequence_parallel=True,
            tp_size=dist.get_world_size(),
            ub_overlap_ag=ub_overlap_ag,
            ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
            params_dtype=torch.bfloat16,
            device="npu",
        )
        layer.set_tensor_parallel_group(dist.group.WORLD)
        return layer, recipe.MXFP8BlockScaling(backward_override=override)

    @pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8)
    @pytest.mark.parametrize("override", ["high_precision", "dequantized"])
    def test_column_sp_fused_comm_under_override(self, override):
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.manual_seed(7)

        m_default, recipe_default = self._make_layer(None, True, True)
        m_test, recipe_ovr = self._make_layer(override, True, True)

        # Full tensors seeded identically on every rank; each rank feeds its
        # sequence shard (column parallel shards activations along M under SP).
        # The per-rank shard must be a multiple of the MX quant block (64):
        # allgather_matmul gathers per-rank scales independently, so
        # world * ceil(M_local / 64) == ceil(M_full / 64) only holds then,
        # and the wgrad quant matmul validates exactly that ratio.
        x_full = torch.randn(128, 256, dtype=torch.bfloat16, device="npu")
        x_shard = x_full.chunk(world, dim=0)[rank].detach().clone().requires_grad_(True)

        try:
            # Default run. Under SP the forward returns the full sequence
            # length with a local N shard, so grad_output is full-M.
            x_def = x_shard.detach().clone().requires_grad_(True)
            with autocast(enabled=True, recipe=recipe_default):
                y_def = m_default(x_def)
            dy_full = torch.randn_like(y_def)
            y_def.backward(dy_full)
            dx_def = x_def.grad.detach().clone()  # noqa: F841
            dw_def = m_default.weight.grad.detach().clone()  # noqa: F841

            # Override run through the same fused-comm configuration.
            x_tst = x_shard.detach().clone().requires_grad_(True)
            with autocast(enabled=True, recipe=recipe_ovr):
                y_tst = m_test(x_tst)
            y_tst.backward(dy_full)
            dx_tst = x_tst.grad.detach().clone()
            dw_tst = m_test.weight.grad.detach().clone()
        finally:
            _reset_fp8_state()

        # Forward is unaffected by the flag: both runs quantize the operands
        # identically and go through the same fused AG GEMM kernel.
        assert_close(y_tst.detach(), y_def.detach(), rtol=0, atol=0, check_dtype=True)

        # Dgrad/wgrad differ numerically by design (override recomputes in
        # plain precision), so validate them against plain-math references.
        # "dequantized" recomputes the backward from MXFP8-rounded operands,
        # so every reference goes through the same quantization round trips
        # (MX blocks tile the K axis and rows are independent, so rounding
        # the full tensor equals rounding the per-rank shards).
        if override == "dequantized":
            w_local = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(
                m_test.weight.detach()
            ).dequantize()
            operand = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(x_full).dequantize()
        else:
            w_local = m_test.weight.detach()
            operand = x_full
        contrib = dy_full.float() @ w_local.float()
        dist.all_reduce(contrib, op=dist.ReduceOp.SUM)
        dx_ref = contrib.chunk(world, dim=0)[rank]
        dw_ref = dy_full.reshape(-1, dy_full.shape[-1]).float().T @ operand.float()
        assert_close(dx_tst.float(), dx_ref, rtol=0.125, atol=0.0675)
        assert_close(dw_tst.float(), dw_ref, rtol=0.125, atol=0.0675)


class TestLayerNormLinearBackwardOverrideTP(DistributedTest):
    """Column-parallel + sequence-parallel LayerNormLinear under backward_override.

    Mirrors ``TestLinearBackwardOverrideTP`` for the LayerNormLinear variant,
    whose fused AG fprop and override wgrad-gather paths live in
    ``layernorm_linear.py``. The GEMM operand is the norm output, so the
    plain-math references apply layer normalization to the full input first;
    dgrad through the GEMM is chained into the layer-norm backward reference
    to obtain dx/dgamma.
    """

    world_size = 2

    _IN = 256  # fused comm kernels require contraction axes >= 256
    _OUT = 512  # / tp_size must also stay >= 256

    def _make_layer(self, override):
        torch.manual_seed(1234)  # identical params on every rank
        layer = LayerNormLinear(
            self._IN,
            self._OUT,
            eps=1e-5,
            bias=False,
            parallel_mode="column",
            sequence_parallel=True,
            tp_size=dist.get_world_size(),
            ub_overlap_ag=True,
            ub_overlap_rs_dgrad=True,
            params_dtype=torch.bfloat16,
            device="npu",
        )
        layer.set_tensor_parallel_group(dist.group.WORLD)
        return layer, recipe.MXFP8BlockScaling(backward_override=override)

    @pytest.mark.skipif(not _mxfp8_available, reason=_reason_for_no_mxfp8)
    @pytest.mark.parametrize("override", ["high_precision", "dequantized"])
    def test_column_sp_fused_comm_under_override(self, override):
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.manual_seed(7)

        m_default, recipe_default = self._make_layer(None)
        m_test, recipe_ovr = self._make_layer(override)

        x_full = torch.randn(128, self._IN, dtype=torch.bfloat16, device="npu")
        x_shard = x_full.chunk(world, dim=0)[rank].detach().clone().requires_grad_(True)

        try:
            # Default run (quantized operands, no override).
            x_def = x_shard.detach().clone().requires_grad_(True)
            with autocast(enabled=True, recipe=recipe_default):
                y_def = m_default(x_def)
            dy_full = torch.randn_like(y_def)
            y_def.backward(dy_full)

            # Override run through the same module configuration.
            x_tst = x_shard.detach().clone().requires_grad_(True)
            with autocast(enabled=True, recipe=recipe_ovr):
                y_tst = m_test(x_tst)
            y_tst.backward(dy_full)
            dx_tst = x_tst.grad.detach().clone()
            dw_tst = m_test.weight.grad.detach().clone()
            dgamma_tst = m_test.layer_norm_weight.grad.detach().clone()
        finally:
            _reset_fp8_state()

        # Forward is unaffected by the flag: both runs quantize the operands
        # identically and go through the same fused AG GEMM kernel.
        assert_close(y_tst.detach(), y_def.detach(), rtol=0, atol=0, check_dtype=True)

        # Plain-math references. d_ln_out is the gradient w.r.t. the norm
        # output: each rank contributes dy @ W_local, summed across ranks and
        # reduce-scattered along the sequence dim. "dequantized" recomputes
        # the backward from MXFP8-rounded operands (dequantized weight for
        # dgrad, rounded norm output for wgrad), so every reference goes
        # through the same quantization round trips (MX blocks tile the K
        # axis and rows are independent, so rounding the full tensor equals
        # rounding the per-rank shards).
        gamma = m_test.layer_norm_weight.detach()
        ln_out_full = F.layer_norm(x_full.float(), (self._IN,), weight=gamma.float(), eps=1e-5).to(
            torch.bfloat16
        )
        if override == "dequantized":
            w_local = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(
                m_test.weight.detach()
            ).dequantize()
            operand = MXFP8Quantizer(fp8_dtype=torch.float8_e4m3fn)(ln_out_full).dequantize()
        else:
            w_local = m_test.weight.detach()
            operand = ln_out_full
        contrib = dy_full.float() @ w_local.float()
        dist.all_reduce(contrib, op=dist.ReduceOp.SUM)
        d_ln_out_rows = contrib.chunk(world, dim=0)[rank]

        dx_ref, dgamma_ref = _layernorm_backward_reference(
            d_ln_out_rows, x_shard.detach(), gamma, 1e-5
        )
        dw_ref = dy_full.reshape(-1, dy_full.shape[-1]).float().T @ operand.float()

        assert_close(dx_tst.float(), dx_ref, rtol=0.125, atol=0.0675)
        assert_close(dw_tst.float(), dw_ref, rtol=0.125, atol=0.0675)
        assert_close(dgamma_tst.float(), dgamma_ref, rtol=0.125, atol=0.0675)
