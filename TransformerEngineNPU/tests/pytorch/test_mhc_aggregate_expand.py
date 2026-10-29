# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations
import os
import pytest
import torch
import torch.nn.functional as F

import torch_npu  # noqa: F401

os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
os.environ.setdefault("NVTE_DISABLE_TRITON_AUTOTUNING", "1")
os.environ.setdefault("TE_NPU_ALLOW_NONDETERMINISTIC_ALGO", "1")

DEVICE = (
    torch.device("npu:0")
    if hasattr(torch, "npu") and torch.npu.is_available()
    else torch.device("cpu")
)
from transformer_engine.pytorch.triton.mhc import (  # noqa: E402
    mhc_fused_aggregate,
    mhc_fused_expand_combine,
)

N = 4


def _atol_rtol(dtype):
    return (5e-3, 5e-3) if dtype == torch.float32 else (2.5e-2, 2.5e-2)


def random_legal_H(s, b, dtype, device):
    H_pre = torch.sigmoid(torch.randn(s, b, N, device=device, dtype=torch.float32)).to(dtype)
    H_post = (2.0 * torch.sigmoid(torch.randn(s, b, N, device=device, dtype=torch.float32))).to(
        dtype
    )
    raw = torch.randn(s, b, N, N, device=device, dtype=torch.float32)
    H_res = F.softmax(raw, dim=-1)
    H_res = (H_res / H_res.sum(dim=-2, keepdim=True).clamp_min(1e-6)).to(dtype)
    return H_pre, H_post, H_res


def ref_aggregate(x, H_pre):
    return (x * H_pre.unsqueeze(-2)).sum(dim=-1)


def ref_expand_combine(f, bias, H_post, x, H_res):
    if bias is not None:
        f = f + bias.view(1, 1, -1)
    return f.unsqueeze(-1) * H_post.unsqueeze(-2) + torch.einsum("sbci,sbij->sbcj", x, H_res)


SHAPES = [(8, 2, 128), (16, 1, 256), (4, 4, 512), (1, 1, 64), (3, 2, 96), (7, 1, 192)]


@pytest.mark.parametrize("s,b,C", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aggregate_fwd(s, b, C, dtype):
    torch.manual_seed(0)
    x = torch.randn(s, b, C, N, device=DEVICE, dtype=dtype)
    H_pre, _, _ = random_legal_H(s, b, dtype, DEVICE)
    out = mhc_fused_aggregate(x, H_pre, N, use_tf32=False)
    ref = ref_aggregate(x.float(), H_pre.float())
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("s,b,C", SHAPES[:4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aggregate_bwd(s, b, C, dtype):
    torch.manual_seed(1)
    x = torch.randn(s, b, C, N, device=DEVICE, dtype=dtype, requires_grad=True)
    H_pre, _, _ = random_legal_H(s, b, dtype, DEVICE)
    H_pre = H_pre.detach().requires_grad_(True)
    mhc_fused_aggregate(x, H_pre, N, use_tf32=False).float().square().mean().backward()
    x2 = x.detach().float().requires_grad_(True)
    H2 = H_pre.detach().float().requires_grad_(True)
    ref_aggregate(x2, H2).square().mean().backward()
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(x.grad.float(), x2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(H_pre.grad.float(), H2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("s,b,C", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_expand_combine_fwd(s, b, C, dtype, with_bias):
    torch.manual_seed(2)
    f = torch.randn(s, b, C, device=DEVICE, dtype=dtype)
    x = torch.randn(s, b, C, N, device=DEVICE, dtype=dtype)
    _, H_post, H_res = random_legal_H(s, b, dtype, DEVICE)
    bias = torch.randn(C, device=DEVICE, dtype=dtype) if with_bias else None
    out = mhc_fused_expand_combine(f, bias, H_post, x, H_res, N, use_tf32=False)
    ref = ref_expand_combine(
        f.float(), None if bias is None else bias.float(), H_post.float(), x.float(), H_res.float()
    )
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("s,b,C", [(8, 2, 128), (4, 1, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_expand_combine_bwd(s, b, C, dtype, with_bias):
    torch.manual_seed(3)
    f = torch.randn(s, b, C, device=DEVICE, dtype=dtype, requires_grad=True)
    x = torch.randn(s, b, C, N, device=DEVICE, dtype=dtype, requires_grad=True)
    _, H_post, H_res = random_legal_H(s, b, dtype, DEVICE)
    H_post = H_post.detach().requires_grad_(True)
    H_res = H_res.detach().requires_grad_(True)
    bias = torch.randn(C, device=DEVICE, dtype=dtype, requires_grad=True) if with_bias else None
    mhc_fused_expand_combine(
        f, bias, H_post, x, H_res, N, use_tf32=False
    ).float().square().mean().backward()
    f2 = f.detach().float().requires_grad_(True)
    x2 = x.detach().float().requires_grad_(True)
    Hp2 = H_post.detach().float().requires_grad_(True)
    Hr2 = H_res.detach().float().requires_grad_(True)
    b2 = None if bias is None else bias.detach().float().requires_grad_(True)
    ref_expand_combine(f2, b2, Hp2, x2, Hr2).square().mean().backward()
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(f.grad.float(), f2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad.float(), x2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(H_post.grad.float(), Hp2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(H_res.grad.float(), Hr2.grad, atol=atol, rtol=rtol)
    if with_bias:
        torch.testing.assert_close(bias.grad.float(), b2.grad, atol=atol, rtol=rtol)
