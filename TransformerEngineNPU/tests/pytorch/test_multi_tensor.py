# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for multi_tensor operations in TransformerEngineNPU.

Aligned with upstream TransformerEngine test_multi_tensor.py.
"""

import importlib
import sys
from typing import Tuple

import pytest
import torch

# Adjust path so we can import transformer_engine from TransformerEngineNPU
sys.path.insert(0, "transformer_engine")

from transformer_engine.pytorch.optimizers import (
    MultiTensorApply,
    multi_tensor_applier,
    multi_tensor_scale,
    multi_tensor_scale_tensor,
    multi_tensor_l2norm,
    multi_tensor_unscale_l2norm,
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)

from transformer_engine.pytorch.quantization import is_mxfp8_available

# ---------------------------------------------------------------------------
# Device detection: prefer NPU, fall back to CUDA, then CPU
# ---------------------------------------------------------------------------
_device = None
if hasattr(torch, "npu") and torch.npu.is_available():
    _device = torch.device("npu")
elif torch.cuda.is_available():
    _device = torch.device("cuda")
else:
    _device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
input_size_pairs = [
    (7777 * 77, 555 * 555),
    (777, 555),
    (555, 2048 * 32 + 1),
    (2048 * 32 + 1, 555),
    (555, 2048 * 32),
    (2048 * 32, 555),
    (33333, 555),
    (555, 33333),
]
appliers = [MultiTensorApply(2048 * 32), MultiTensorApply(333), MultiTensorApply(33333)]
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def _scale_from_amax_tensor(
    amax: torch.Tensor,
    fp8_dtype: torch.dtype,
    eps: float,
    pow_2_scales: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for scale/scale_inv computation."""
    assert amax.dtype == torch.float32, "amax must be a float32 tensor."
    fp8_max = torch.finfo(fp8_dtype).max
    clamped = torch.max(amax, torch.tensor(eps, dtype=torch.float32, device=amax.device))

    scale = torch.div(torch.tensor(fp8_max, dtype=torch.float32, device=amax.device), clamped)
    scale = torch.where(
        torch.isinf(scale),
        torch.tensor(torch.finfo(torch.float32).max, dtype=torch.float32, device=amax.device),
        scale,
    )

    if pow_2_scales:
        _, exp = torch.frexp(scale)
        exp = exp - 1
        unity = torch.tensor(1.0, device=exp.device)
        torch.ldexp(unity, exp, out=scale)
        scale = torch.where(amax == float("inf"), 0.0, scale)

    scale = torch.where(amax == 0.0, 1.0, scale)
    scale_inv = torch.reciprocal(scale)
    return scale, scale_inv


# ===========================================================================
class TestMultiTensorScale:
    """Tests for multi_tensor_scale."""

    @pytest.mark.parametrize("input_size_pair", input_size_pairs)
    @pytest.mark.parametrize("applier", appliers)
    @pytest.mark.parametrize("repeat", [1, 55])
    @pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("out_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_multi_tensor_scale(self, input_size_pair, applier, repeat, in_type, out_type, inplace):
        if inplace and out_type != in_type:
            pytest.skip("inplace=True and out_type != in_type is not supported")
        if (in_type == torch.float16 and out_type == torch.bfloat16) or (
            in_type == torch.bfloat16 and out_type == torch.float16
        ):
            pytest.skip("float16 to bfloat16 is not necessary and vice versa")

        scale = 4.0
        overflow_buf = torch.zeros(1, dtype=torch.int32, device=_device)
        ref = torch.tensor([1.0], dtype=torch.float32, device=_device)
        sizea, sizeb = input_size_pair

        def downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=False):
            overflow_buf.zero_()
            a = torch.full([sizea], scale, dtype=torch.float32, device=_device)
            b = torch.full([sizeb], scale, dtype=torch.float32, device=_device)

            out_list = []
            for i in range(repeat):
                out_list += [a.clone().to(out_type), b.clone().to(out_type)]

            if inplace:
                in_list = out_list
            else:
                in_list = [out.clone().to(in_type) for out in out_list]

            applier(multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)

            assert all(torch.allclose(out, ref.to(out_type)) for out in out_list)
            assert overflow_buf.item() == 0

        def find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            t,
            ind,
            val,
            inplace=False,
        ):
            overflow_buf.zero_()
            a = torch.full([sizea], scale, dtype=torch.float32, device=_device)
            b = torch.full([sizeb], scale, dtype=torch.float32, device=_device)

            out_list = []
            for i in range(repeat):
                out_list += [a.clone().to(out_type), b.clone().to(out_type)]

            if inplace:
                in_list = out_list
            else:
                in_list = [out.clone().to(in_type) for out in out_list]

            applier(multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)

            overflow_buf.zero_()
            in_list[t][ind] = val
            applier(multi_tensor_scale, overflow_buf, [in_list, out_list], 1.0 / scale)
            assert overflow_buf.item() > 0

        downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=inplace)
        find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            0,
            0,
            float("nan"),
            inplace=inplace,
        )
        find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            2 * repeat - 1,
            sizeb - 1,
            float("inf"),
            inplace=inplace,
        )
        find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            2 * (repeat // 2),
            sizea // 2,
            float("inf"),
            inplace=inplace,
        )


class TestMultiTensorScaleTensor:
    """Tests for multi_tensor_scale_tensor."""

    @pytest.mark.parametrize("input_size_pair", input_size_pairs)
    @pytest.mark.parametrize("applier", appliers)
    @pytest.mark.parametrize("repeat", [1, 55])
    @pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("out_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_multi_tensor_scale_tensor(
        self, input_size_pair, applier, repeat, in_type, out_type, inplace
    ):
        if inplace and out_type != in_type:
            pytest.skip("inplace=True and out_type != in_type is not supported")
        if (in_type == torch.float16 and out_type == torch.bfloat16) or (
            in_type == torch.bfloat16 and out_type == torch.float16
        ):
            pytest.skip("float16 to bfloat16 is not necessary and vice versa")

        scale = 4.0
        inv_scale_tensor = torch.tensor([1.0 / scale], dtype=torch.float32, device=_device)
        overflow_buf = torch.zeros(1, dtype=torch.int32, device=_device)
        ref = torch.tensor([1.0], dtype=torch.float32, device=_device)
        sizea, sizeb = input_size_pair

        def downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=False):
            overflow_buf.zero_()
            a = torch.full([sizea], scale, dtype=torch.float32, device=_device)
            b = torch.full([sizeb], scale, dtype=torch.float32, device=_device)

            out_list = []
            for _ in range(repeat):
                out_list += [a.clone().to(out_type), b.clone().to(out_type)]

            if inplace:
                in_list = out_list
            else:
                in_list = [out.clone().to(in_type) for out in out_list]

            applier(
                multi_tensor_scale_tensor,
                overflow_buf,
                [in_list, out_list],
                inv_scale_tensor,
            )

            assert all(torch.allclose(out, ref.to(out_type)) for out in out_list)
            assert overflow_buf.item() == 0

        def find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            t,
            ind,
            val,
            inplace=False,
        ):
            overflow_buf.zero_()
            a = torch.full([sizea], scale, dtype=torch.float32, device=_device)
            b = torch.full([sizeb], scale, dtype=torch.float32, device=_device)

            out_list = []
            for _ in range(repeat):
                out_list += [a.clone().to(out_type), b.clone().to(out_type)]

            if inplace:
                in_list = out_list
            else:
                in_list = [out.clone().to(in_type) for out in out_list]

            applier(
                multi_tensor_scale_tensor,
                overflow_buf,
                [in_list, out_list],
                inv_scale_tensor,
            )

            overflow_buf.zero_()
            in_list[t][ind] = val
            applier(
                multi_tensor_scale_tensor,
                overflow_buf,
                [in_list, out_list],
                inv_scale_tensor,
            )
            assert overflow_buf.item() > 0

        downscale(sizea, sizeb, applier, repeat, in_type, out_type, inplace=inplace)
        find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            0,
            0,
            float("nan"),
            inplace=inplace,
        )
        find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            2 * repeat - 1,
            sizeb - 1,
            float("inf"),
            inplace=inplace,
        )
        find_inf(
            sizea,
            sizeb,
            applier,
            repeat,
            in_type,
            out_type,
            2 * (repeat // 2),
            sizea // 2,
            float("inf"),
            inplace=inplace,
        )


class TestMultiTensorL2Norm:
    """Tests for multi_tensor_l2norm."""

    @pytest.mark.parametrize("input_size_pair", input_size_pairs)
    @pytest.mark.parametrize("applier", appliers)
    @pytest.mark.parametrize("repeat", [1, 55])
    @pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("per_tensor", [False, True])
    def test_multi_tensor_l2norm(self, input_size_pair, applier, repeat, in_type, per_tensor):
        sizea, sizeb = input_size_pair
        val = 4.0
        overflow_buf = torch.zeros(1, dtype=torch.int32, device=_device)

        overflow_buf.zero_()
        a = torch.full([sizea], val, dtype=torch.float32, device=_device)
        b = torch.full([sizeb], val, dtype=torch.float32, device=_device)

        in_list = []
        for i in range(repeat):
            in_list += [a.clone().to(in_type), b.clone().to(in_type)]

        if per_tensor:
            norm, norm_per_tensor = applier(
                multi_tensor_l2norm,
                overflow_buf,
                [in_list],
                True,
            )
            normab = torch.cat((a.norm().view(1), b.norm().view(1)))
            norm_per_tensor = norm_per_tensor.view(-1, 2)
        else:
            norm, _ = applier(
                multi_tensor_l2norm,
                overflow_buf,
                [in_list],
                False,
            )

        reference = torch.full(
            [(sizea + sizeb) * repeat],
            val,
            dtype=torch.float32,
            device=_device,
        ).norm()

        torch.testing.assert_close(norm, reference.broadcast_to(norm.shape))
        if per_tensor:
            torch.testing.assert_close(
                norm_per_tensor,
                normab.broadcast_to(norm_per_tensor.shape),
            )
        assert overflow_buf.item() == 0


class TestMultiTensorUnscaleL2Norm:
    """Tests for multi_tensor_unscale_l2norm."""

    @pytest.mark.parametrize("input_size_pair", input_size_pairs)
    @pytest.mark.parametrize("applier", appliers)
    @pytest.mark.parametrize("repeat", [1, 55])
    @pytest.mark.parametrize("in_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("per_tensor", [False, True])
    def test_multi_tensor_unscale_l2norm(
        self, input_size_pair, applier, repeat, in_type, per_tensor
    ):
        sizea, sizeb = input_size_pair
        val = 4.0
        inv_scale = 0.5
        inv_scale_tensor = torch.tensor([inv_scale], dtype=torch.float32, device=_device)
        overflow_buf = torch.zeros(1, dtype=torch.int32, device=_device)

        overflow_buf.zero_()
        a = torch.full([sizea], val, dtype=torch.float32, device=_device)
        b = torch.full([sizeb], val, dtype=torch.float32, device=_device)

        in_list = []
        for i in range(repeat):
            in_list += [a.clone().to(in_type), b.clone().to(in_type)]

        norm, norm_per_tensor = applier(
            multi_tensor_unscale_l2norm,
            overflow_buf,
            [in_list],
            inv_scale_tensor,
            per_tensor,
        )

        reference = torch.full(
            [(sizea + sizeb) * repeat],
            val * inv_scale,
            dtype=torch.float32,
            device=_device,
        ).norm()

        torch.testing.assert_close(norm, reference.broadcast_to(norm.shape))
        if per_tensor:
            normab = torch.cat(
                ((a * inv_scale).norm().view(1), (b * inv_scale).norm().view(1)),
            )
            norm_per_tensor = norm_per_tensor.view(-1, 2)
            torch.testing.assert_close(
                norm_per_tensor,
                normab.broadcast_to(norm_per_tensor.shape),
            )
        assert overflow_buf.item() == 0


class TestMultiTensorComputeScale:
    """Tests for multi_tensor_compute_scale_and_scale_inv."""

    @pytest.mark.parametrize("input_size_pair", input_size_pairs + [(1, 1)])
    @pytest.mark.parametrize("applier", appliers)
    @pytest.mark.parametrize("repeat", [1, 55])
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    @pytest.mark.parametrize("pow_2_scales", [False, True])
    @pytest.mark.parametrize("epsilon", [0.0, 100.0])
    def test_multi_tensor_compute_scale_and_scale_inv(
        self,
        input_size_pair,
        applier,
        repeat,
        fp8_dtype,
        pow_2_scales,
        epsilon,
    ):
        sizea, sizeb = input_size_pair
        overflow_buf = torch.zeros(1, dtype=torch.int32, device=_device)
        a = torch.randn([sizea], dtype=torch.float32, device=_device).abs()
        b = torch.randn([sizeb], dtype=torch.float32, device=_device).abs()
        max_fp8 = torch.finfo(fp8_dtype).max

        amax_list = []
        for i in range(repeat):
            amax_list += [a.clone(), b.clone()]

        scale_list = [torch.empty_like(x) for x in amax_list]
        scale_inv_list = [torch.empty_like(x) for x in amax_list]

        applier(
            multi_tensor_compute_scale_and_scale_inv,
            overflow_buf,
            [amax_list, scale_list, scale_inv_list],
            max_fp8,
            pow_2_scales,
            epsilon,
        )

        for amax, scale, scale_inv in zip(amax_list, scale_list, scale_inv_list):
            scale_ref, scale_inv_ref = _scale_from_amax_tensor(
                amax,
                fp8_dtype,
                eps=epsilon,
                pow_2_scales=pow_2_scales,
            )
            # triton-ascend 的 fp32 除法与 aclnn 除法在部分硬件上相差 1 ULP
            # （非 IEEE 正确舍入；950 上实测 Triton 路径偏离正确舍入，aclnn 一致），
            # scale 本身最多差 1 ULP，故允许 2 ULP；scale_inv = 1/scale 中 scale 的
            # 1 ULP 误差会传播，且两侧倒数运算各自再舍入，最坏约 2.5 ULP，故允许 4 ULP。
            _fp32_ulp = 2.0**-23
            torch.testing.assert_close(scale, scale_ref, rtol=2 * _fp32_ulp, atol=0)
            torch.testing.assert_close(scale_inv, scale_inv_ref, rtol=4 * _fp32_ulp, atol=0)


class TestMultiTensorComputeScaleInvE8M0:
    """Tests for multi_tensor_compute_scale_inv_e8m0."""

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    @pytest.mark.parametrize("input_size_pair", input_size_pairs + [(1, 1)])
    @pytest.mark.parametrize("applier", appliers)
    @pytest.mark.parametrize("repeat", [1, 55])
    def test_multi_tensor_compute_scale_inv_e8m0(
        self,
        input_size_pair,
        applier,
        repeat,
    ):
        sizea, sizeb = input_size_pair
        a = torch.randn([sizea], dtype=torch.bfloat16, device=_device).abs()
        b = torch.randn([sizeb], dtype=torch.bfloat16, device=_device).abs()

        amax_list = []
        for _ in range(repeat):
            amax_list += [a.clone(), b.clone()]
        scale_inv_list = [torch.empty_like(x).to(torch.uint8) for x in amax_list]

        applier(
            multi_tensor_compute_scale_inv_e8m0,
            None,
            [amax_list, scale_inv_list],
        )

        for amax, scale_inv in zip(amax_list, scale_inv_list):
            # Keep the independent bit-level oracle on CPU so unsupported NPU bitwise
            # operators cannot affect the reference result.
            fp32 = amax.float().cpu()
            fp32_bits = fp32.view(torch.int32)
            biased_exponent = torch.bitwise_and(
                torch.bitwise_right_shift(fp32_bits, 23),
                0xFF,
            )
            expected = (biased_exponent - 8).clamp_(0, 254).to(torch.uint8)
            expected = torch.where(fp32 > 0, expected, 0)
            torch.testing.assert_close(expected, scale_inv.cpu())

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_multi_tensor_compute_scale_inv_e8m0_boundaries(self):
        # Include an exact power of two and its adjacent FP32 values.
        amax = torch.tensor(
            [
                0.0,
                2**-13 - 2**-37,
                2**-13,
                2**-13 + 2**-36,
                224.0,
                226.0,
                448.0,
                450.0,
                500.0,
                512.0,
                896.0,
                900.0,
            ],
            dtype=torch.float32,
            device=_device,
        )
        scale_inv = torch.empty_like(amax, dtype=torch.uint8)

        multi_tensor_applier(
            multi_tensor_compute_scale_inv_e8m0,
            None,
            [[amax], [scale_inv]],
        )

        # NPU scale_alg=0 uses floor-power-of-two E8M0 scales. This intentionally differs
        # from CUDA upstream's amax / 448 round-up rule around values such as 450 and 900.
        expected = torch.tensor(
            [0, 105, 106, 106, 126, 126, 127, 127, 127, 128, 128, 128],
            dtype=torch.uint8,
            device=_device,
        )
        torch.testing.assert_close(scale_inv, expected, rtol=0, atol=0)


class TestMultiTensorApply:
    """Tests for MultiTensorApply class."""

    def test_chunk_size_injection(self):
        applier = MultiTensorApply(12345)
        called_with = {}

        def capture_op(chunk_size, buf, lists, *args):
            called_with["chunk_size"] = chunk_size
            called_with["buf"] = buf
            called_with["lists"] = lists
            called_with["args"] = args

        buf = torch.zeros(1, dtype=torch.int32)
        lists = [[torch.zeros(10), torch.zeros(20)]]
        applier(capture_op, buf, lists, "extra")

        assert called_with["chunk_size"] == 12345
        assert called_with["buf"] is buf
        assert called_with["lists"] is lists
        assert called_with["args"] == ("extra",)

    def test_default_applier_exists(self):
        assert hasattr(multi_tensor_applier, "chunk_size")
        assert multi_tensor_applier.chunk_size == 2048 * 32


class TestImportPaths:
    """Verify all expected import paths work."""

    def test_direct_import(self):
        optimizers = importlib.import_module("transformer_engine.pytorch.optimizers")

        assert optimizers.multi_tensor_scale is multi_tensor_scale

    def test_from_pytorch(self):
        pytorch = importlib.import_module("transformer_engine.pytorch")

        assert pytorch.multi_tensor_scale is multi_tensor_scale

    def test_from_pytorch_applier(self):
        pytorch = importlib.import_module("transformer_engine.pytorch")

        assert pytorch.multi_tensor_applier is multi_tensor_applier
        assert callable(pytorch.multi_tensor_applier)

    def test_from_pytorch_mtapply(self):
        pytorch = importlib.import_module("transformer_engine.pytorch")

        assert pytorch.MultiTensorApply is MultiTensorApply

    def test_import_all_exports(self):
        optimizers = importlib.import_module("transformer_engine.pytorch.optimizers")

        assert optimizers.multi_tensor_scale is multi_tensor_scale
        assert optimizers.multi_tensor_scale_tensor is multi_tensor_scale_tensor
        assert optimizers.multi_tensor_l2norm is multi_tensor_l2norm
        assert optimizers.multi_tensor_unscale_l2norm is multi_tensor_unscale_l2norm
        assert (
            optimizers.multi_tensor_compute_scale_and_scale_inv
            is multi_tensor_compute_scale_and_scale_inv
        )
        assert optimizers.multi_tensor_compute_scale_inv_e8m0 is multi_tensor_compute_scale_inv_e8m0
        assert optimizers.MultiTensorApply is MultiTensorApply
        assert optimizers.multi_tensor_applier is multi_tensor_applier
