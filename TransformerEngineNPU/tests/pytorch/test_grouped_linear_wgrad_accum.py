# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for ``GroupedLinear`` weight-gradient accumulation paths.

Covers the ``gradient_accumulation_fusion`` implementation on NPU:

- Basic ops: ``matmul_add`` (the shared wgrad-accum primitive) and its
  thin wrappers ``wgrad_gemm_accum_fp32`` / ``wgrad_gemm_accum_fp16``.
- Integration: ``GroupedLinear`` backward with ``fuse_wgrad_accumulation=True``
  (path A: per-expert ``addmm_`` into ``main_grad``) vs ``fuse_wgrad_accumulation=False``
  (path B2: ``general_grouped_gemm`` returns wgrad to ``weight{i}.grad``).

Path B1 (GroupedTensor + fuse, ``add_`` loop) is only reachable under FP8 /
MXFP8 quantization, which is not exercised here.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from transformer_engine.pytorch import GroupedLinear
from transformer_engine.pytorch.ops.gemm import (
    matmul_add,
    wgrad_gemm_accum_fp32,
    wgrad_gemm_accum_fp16,
)

from utils import npu_available


pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")


def _npu() -> torch.device:
    return torch.device("npu")


# Per-input-dtype tolerances for the basic ops. ``main_grad`` is always FP32,
# so the output precision is at least as good as the input dtype allows.
_BASIC_TOLS = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}

# Integration tols: path A (addmm_, fp32 accumulate) vs path B2
# (npu_grouped_matmul, bf16 output). The bf16 output caps the achievable
# precision, so we use a relaxed tolerance.
_INTEGRATION_TOLS = dict(rtol=5e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Basic ops: matmul_add / wgrad_gemm_accum_fp32 / wgrad_gemm_accum_fp16
# ---------------------------------------------------------------------------
# Reference: main_grad += grad_out.T @ inp   (computed in fp32 on NPU)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,n,k", [(4, 8, 16), (1, 3, 5), (7, 32, 12)])
def test_matmul_add_matches_pytorch_reference(dtype, m, n, k):
    """``main_grad += grad_out^T @ inp`` must match a plain fp32 PyTorch accumulation."""
    torch.manual_seed(0)
    inp = torch.randn(m, k, dtype=dtype, device=_npu())
    grad_out = torch.randn(m, n, dtype=dtype, device=_npu())
    main_grad = torch.randn(n, k, dtype=torch.float32, device=_npu())
    main_grad_ref = main_grad.detach().clone()

    matmul_add(main_grad, inp, grad_out)

    main_grad_ref.addmm_(grad_out.float().t(), inp.float())
    torch.testing.assert_close(main_grad, main_grad_ref, **_BASIC_TOLS[dtype])


def test_wgrad_gemm_accum_fp32_equivalent_to_matmul_add():
    """``wgrad_gemm_accum_fp32`` must be a thin wrapper over ``matmul_add``."""
    torch.manual_seed(1)
    inp = torch.randn(7, 16, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(7, 32, dtype=torch.bfloat16, device=_npu())
    main_a = torch.randn(32, 16, dtype=torch.float32, device=_npu())
    main_b = main_a.detach().clone()

    wgrad_gemm_accum_fp32(inp, grad_out, main_a)
    matmul_add(main_b, inp, grad_out)
    torch.testing.assert_close(main_a, main_b, rtol=0, atol=0)


def test_wgrad_gemm_accum_fp16_equivalent_to_matmul_add():
    """``wgrad_gemm_accum_fp16`` must be a thin wrapper over ``matmul_add``."""
    torch.manual_seed(2)
    inp = torch.randn(5, 12, dtype=torch.float16, device=_npu())
    grad_out = torch.randn(5, 24, dtype=torch.float16, device=_npu())
    main_a = torch.randn(24, 12, dtype=torch.float32, device=_npu())
    main_b = main_a.detach().clone()

    wgrad_gemm_accum_fp16(inp, grad_out, main_a)
    matmul_add(main_b, inp, grad_out)
    torch.testing.assert_close(main_a, main_b, rtol=0, atol=0)


def test_matmul_add_empty_input_is_noop():
    """Empty input must short-circuit without modifying ``main_grad``."""
    torch.manual_seed(3)
    inp = torch.randn(0, 16, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(0, 32, dtype=torch.bfloat16, device=_npu())
    main_grad = torch.randn(32, 16, dtype=torch.float32, device=_npu())
    main_grad_ref = main_grad.detach().clone()

    matmul_add(main_grad, inp, grad_out)

    torch.testing.assert_close(main_grad, main_grad_ref, rtol=0, atol=0)


def test_matmul_add_accumulates_repeated_calls():
    """Repeated calls must additively accumulate, matching a single fused call."""
    torch.manual_seed(4)
    inp1 = torch.randn(3, 8, dtype=torch.bfloat16, device=_npu())
    inp2 = torch.randn(2, 8, dtype=torch.bfloat16, device=_npu())
    grad_out1 = torch.randn(3, 4, dtype=torch.bfloat16, device=_npu())
    grad_out2 = torch.randn(2, 4, dtype=torch.bfloat16, device=_npu())

    main_split = torch.randn(4, 8, dtype=torch.float32, device=_npu())
    main_fused = main_split.detach().clone()

    # Split path: two addmm_ calls
    matmul_add(main_split, inp1, grad_out1)
    matmul_add(main_split, inp2, grad_out2)

    # Fused path: one addmm_ over the concatenated tensors
    matmul_add(
        main_fused,
        torch.cat([inp1, inp2], dim=0),
        torch.cat([grad_out1, grad_out2], dim=0),
    )

    torch.testing.assert_close(main_split, main_fused, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Integration: GroupedLinear backward wgrad paths
# ---------------------------------------------------------------------------

_NUM_GEMMS = 2
_IN_FEATURES = 16
_OUT_FEATURES = 32
_SEQ_LEN = 12  # total M; split among experts


def _make_grouped_linear(
    *,
    num_gemms: int,
    in_features: int,
    out_features: int,
    bias: bool,
    fuse_wgrad_accumulation: bool,
    parallel_mode: str = "column",
    dtype: torch.dtype = torch.bfloat16,
    delay_wgrad_compute: bool = False,
) -> GroupedLinear:
    return GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=bias,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        params_dtype=dtype,
        parallel_mode=parallel_mode,
        delay_wgrad_compute=delay_wgrad_compute,
        device=_npu(),
    ).eval()


def _share_weights(src: GroupedLinear, dst: GroupedLinear) -> None:
    """Copy ``weight{i}`` (and ``bias{i}``) from ``src`` into ``dst``."""
    with torch.no_grad():
        for i in range(src.num_gemms):
            getattr(dst, f"weight{i}").copy_(getattr(src, f"weight{i}"))
            if src.use_bias:
                getattr(dst, f"bias{i}").copy_(getattr(src, f"bias{i}"))


def _even_m_splits(seq_len: int, num_gemms: int) -> List[int]:
    """Split ``seq_len`` into ``num_gemms`` parts as evenly as possible."""
    base, rem = divmod(seq_len, num_gemms)
    return [base + (1 if i < rem else 0) for i in range(num_gemms)]


def _run_forward_backward(
    layer: GroupedLinear, inp: torch.Tensor, m_splits, grad_out: torch.Tensor
) -> None:
    out = layer(inp, m_splits)
    out.backward(grad_out)


def _fp32_wgrad_reference(
    inp: torch.Tensor, grad_out: torch.Tensor, m_splits: List[int]
) -> List[torch.Tensor]:
    """Per-expert fp32 reference ``wgrad_i = grad_output_i^T @ input_i``."""
    return [
        torch.matmul(dy.float().t(), x.float())
        for x, dy in zip(torch.split(inp, m_splits), torch.split(grad_out, m_splits))
    ]


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("parallel_mode", ["column", "row"])
def test_grouped_linear_fuse_wgrad_matches_no_fuse(bias, parallel_mode):
    """Path A (fuse=True, ``addmm_`` into ``main_grad``) must match path B2
    (fuse=False, wgrad returned to ``weight{i}.grad``).

    Both paths compute ``wgrad_i = grad_output_i^T @ input_i`` per expert; the
    only differences are the accumulation dtype (fp32 vs bf16) and the kernel
    (``addmm_`` vs ``npu_grouped_matmul``).
    """
    torch.manual_seed(10)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    fuse_layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        fuse_wgrad_accumulation=True,
        parallel_mode=parallel_mode,
    )
    no_fuse_layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        fuse_wgrad_accumulation=False,
        parallel_mode=parallel_mode,
    )
    _share_weights(fuse_layer, no_fuse_layer)

    # Set fp32 main_grad on the fuse layer so path A (per-expert addmm_) is taken.
    main_grad_init = []
    for i in range(num_gemms):
        w = getattr(fuse_layer, f"weight{i}")
        init = torch.randn_like(w, dtype=torch.float32)
        w.main_grad = init.clone()
        main_grad_init.append(init)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())

    _run_forward_backward(fuse_layer, inp.detach().clone(), m_splits, grad_out)
    _run_forward_backward(no_fuse_layer, inp.detach().clone(), m_splits, grad_out)

    for i in range(num_gemms):
        fuse_wgrad = getattr(fuse_layer, f"weight{i}").main_grad - main_grad_init[i]
        no_fuse_wgrad = getattr(no_fuse_layer, f"weight{i}").grad
        assert no_fuse_wgrad is not None, f"weight{i}.grad should be set when fuse=False"
        torch.testing.assert_close(
            fuse_wgrad,
            no_fuse_wgrad.float(),
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: f"weight{i} wgrad mismatch (fuse vs no-fuse)\n{msg}",
        )


def test_grouped_linear_fuse_wgrad_accumulates_across_microbatches():
    """Two consecutive backward passes must additively accumulate into ``main_grad``.

    The fused ``main_grad`` after two microbatches must equal
    ``init + wgrad_mb1 + wgrad_mb2``, where each ``wgrad_mb*`` is independently
    computed via the no-fuse path.
    """
    torch.manual_seed(20)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    fuse_layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
    )
    no_fuse_mb1 = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=False,
        parallel_mode="column",
    )
    no_fuse_mb2 = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=False,
        parallel_mode="column",
    )
    _share_weights(fuse_layer, no_fuse_mb1)
    _share_weights(fuse_layer, no_fuse_mb2)

    main_grad_init = []
    for i in range(num_gemms):
        w = getattr(fuse_layer, f"weight{i}")
        init = torch.randn_like(w, dtype=torch.float32)
        w.main_grad = init.clone()
        main_grad_init.append(init)

    # Microbatch 1
    inp1 = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out1 = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    _run_forward_backward(fuse_layer, inp1.detach().clone(), m_splits, grad_out1)
    _run_forward_backward(no_fuse_mb1, inp1.detach().clone(), m_splits, grad_out1)

    # Microbatch 2
    inp2 = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out2 = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    _run_forward_backward(fuse_layer, inp2.detach().clone(), m_splits, grad_out2)
    _run_forward_backward(no_fuse_mb2, inp2.detach().clone(), m_splits, grad_out2)

    for i in range(num_gemms):
        expected = (
            main_grad_init[i]
            + getattr(no_fuse_mb1, f"weight{i}").grad.float()
            + getattr(no_fuse_mb2, f"weight{i}").grad.float()
        )
        actual = getattr(fuse_layer, f"weight{i}").main_grad
        torch.testing.assert_close(
            actual,
            expected,
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: f"weight{i} accumulated main_grad mismatch\n{msg}",
        )


def test_grouped_linear_fuse_wgrad_grad_attribute_is_none():
    """When ``fuse_wgrad_accumulation=True``, ``weight{i}.grad`` must stay ``None``.

    The wgrad is accumulated into ``main_grad``; PyTorch's ``.grad`` slot is
    intentionally left untouched so the optimizer can consume ``main_grad``.
    """
    torch.manual_seed(30)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    fuse_layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
    )
    for i in range(num_gemms):
        w = getattr(fuse_layer, f"weight{i}")
        w.main_grad = torch.zeros_like(w, dtype=torch.float32)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    _run_forward_backward(fuse_layer, inp.detach().clone(), m_splits, grad_out)

    for i in range(num_gemms):
        w = getattr(fuse_layer, f"weight{i}")
        assert w.grad is None, f"weight{i}.grad must be None when fuse=True, got {w.grad}"
        assert w.main_grad is not None, f"weight{i}.main_grad must be populated"


@pytest.mark.parametrize(
    "split_type",
    ["list", "tensor", "metadata"],
    ids=["list-splits", "tensor-splits", "metadata-splits"],
)
def test_grouped_linear_fuse_wgrad_path_a_with_delay_wgrad_compute(split_type):
    """Delayed path A must support every public ``m_splits`` representation.

    When ``delay_wgrad_compute=True``, path A defers the per-expert ``addmm_``
    (and DDP flag setup) to ``backward_dw()`` via ``wgrad_store``, consistent
    with the B1/B2 delayed path. The accumulated ``main_grad`` must match the
    no-delay reference where ``addmm_`` runs eagerly in ``backward``.
    """
    torch.manual_seed(40)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    delay_layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
        delay_wgrad_compute=True,
    )
    ref_layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
        delay_wgrad_compute=False,
    )
    _share_weights(delay_layer, ref_layer)

    main_grad_init = []
    for i in range(num_gemms):
        w_delay = getattr(delay_layer, f"weight{i}")
        w_ref = getattr(ref_layer, f"weight{i}")
        init = torch.randn_like(w_delay, dtype=torch.float32)
        w_delay.main_grad = init.clone()
        w_ref.main_grad = init.clone()
        main_grad_init.append(init)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    split_tensor = torch.tensor(m_splits, dtype=torch.int64, device=_npu())
    if split_type == "list":
        delay_splits = m_splits
    elif split_type == "tensor":
        delay_splits = split_tensor
    else:
        delay_splits = delay_layer.make_grouped_linear_metadata(split_tensor)

    # Candidate: delayed wgrad with the selected split representation.
    _run_forward_backward(delay_layer, inp.detach().clone(), delay_splits, grad_out)
    assert delay_layer.need_backward_dw(), "delay_wgrad_compute=True must set need_backward_dw"
    delay_layer.backward_dw()

    # Reference: delay=False, wgrad accumulated directly in backward
    _run_forward_backward(ref_layer, inp.detach().clone(), m_splits, grad_out)

    for i in range(num_gemms):
        delay_main = getattr(delay_layer, f"weight{i}").main_grad
        ref_main = getattr(ref_layer, f"weight{i}").main_grad
        torch.testing.assert_close(
            delay_main,
            ref_main,
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad mismatch (split_type={split_type}, delay vs no-delay)\n{msg}"
            ),
        )


# ---------------------------------------------------------------------------
# Integration: fused wgrad overwrite semantics
# ---------------------------------------------------------------------------
# Mirrors ``linear.py``: the fused bf16 wgrad must overwrite ``main_grad`` when
# ``overwrite_main_grad`` is set or on the first microbatch, and must accumulate
# otherwise. ``main_grad`` is pre-filled with a non-zero sentinel so that a
# wrong accumulate/lose is visible instead of being masked by zero-init.

_PREFILL = 7.0


def _fill_main_grad(layer: GroupedLinear, value: float) -> None:
    for i in range(layer.num_gemms):
        w = getattr(layer, f"weight{i}")
        w.main_grad = torch.full_like(w, value, dtype=torch.float32)


@pytest.mark.parametrize("parallel_mode", ["column", "row"])
@pytest.mark.parametrize("overwrite_main_grad", [True, False])
def test_grouped_linear_fuse_wgrad_respects_overwrite_main_grad(overwrite_main_grad, parallel_mode):
    """``overwrite_main_grad=True`` must write ``main_grad`` instead of accumulating.

    ``linear.py`` computes ``accumulate = (is_first_microbatch is None or not
    is_first_microbatch) and not overwrite_main_grad`` and zeroes ``main_grad``
    when not accumulating. The bf16 fused branch of ``GroupedLinear`` must follow
    the same contract: with the flag set the stale ``main_grad`` content must be
    discarded, while the default (flag unset) keeps additive accumulation.
    """
    torch.manual_seed(50)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode=parallel_mode,
    )
    for i in range(num_gemms):
        getattr(layer, f"weight{i}").overwrite_main_grad = overwrite_main_grad
    _fill_main_grad(layer, _PREFILL)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    _run_forward_backward(layer, inp, m_splits, grad_out)

    reference = _fp32_wgrad_reference(inp, grad_out, m_splits)
    for i in range(num_gemms):
        expected = reference[i] if overwrite_main_grad else reference[i] + _PREFILL
        actual = getattr(layer, f"weight{i}").main_grad
        torch.testing.assert_close(
            actual,
            expected,
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad mismatch (overwrite_main_grad={overwrite_main_grad})\n{msg}"
            ),
        )


def test_grouped_linear_fuse_wgrad_first_microbatch_overwrites():
    """``is_first_microbatch=True`` must overwrite ``main_grad``.

    Same contract as the dense ``Linear`` path: the first microbatch of a new
    gradient window writes the wgrad instead of accumulating onto a buffer that
    may be reused without being zeroed.
    """
    torch.manual_seed(51)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
    )
    _fill_main_grad(layer, _PREFILL)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    out = layer(inp, m_splits, is_first_microbatch=True)
    out.backward(grad_out)

    reference = _fp32_wgrad_reference(inp, grad_out, m_splits)
    for i in range(num_gemms):
        actual = getattr(layer, f"weight{i}").main_grad
        torch.testing.assert_close(
            actual,
            reference[i],
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad mismatch (is_first_microbatch=True)\n{msg}"
            ),
        )


def test_grouped_linear_fuse_wgrad_non_first_microbatch_accumulates():
    """``is_first_microbatch=False`` must keep additive accumulation.

    Completes the three-way contract of the dense ``Linear`` path: only the
    explicit first microbatch (or ``overwrite_main_grad``) overwrites; an
    explicit non-first microbatch keeps accumulating onto the sentinel.
    """
    torch.manual_seed(54)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
    )
    _fill_main_grad(layer, _PREFILL)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    out = layer(inp, m_splits, is_first_microbatch=False)
    out.backward(grad_out)

    reference = _fp32_wgrad_reference(inp, grad_out, m_splits)
    for i in range(num_gemms):
        actual = getattr(layer, f"weight{i}").main_grad
        torch.testing.assert_close(
            actual,
            reference[i] + _PREFILL,
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad mismatch (is_first_microbatch=False)\n{msg}"
            ),
        )


@pytest.mark.parametrize("overwrite_main_grad", [True, False])
def test_grouped_linear_fuse_wgrad_delayed_respects_overwrite_main_grad(overwrite_main_grad):
    """Delayed wgrad must apply the overwrite decision inside ``backward_dw``.

    ``delay_wgrad_compute=True`` defers the fused ``addmm_`` to ``backward_dw()``,
    so the overwrite decision is taken at flush time and ``main_grad`` must stay
    untouched until then.
    """
    torch.manual_seed(52)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
        delay_wgrad_compute=True,
    )
    for i in range(num_gemms):
        getattr(layer, f"weight{i}").overwrite_main_grad = overwrite_main_grad
    _fill_main_grad(layer, _PREFILL)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    _run_forward_backward(layer, inp, m_splits, grad_out)

    assert layer.need_backward_dw(), "delay_wgrad_compute=True must set need_backward_dw"
    for i in range(num_gemms):
        weight = getattr(layer, f"weight{i}")
        torch.testing.assert_close(
            weight.main_grad,
            torch.full_like(weight, _PREFILL, dtype=torch.float32),
            rtol=0,
            atol=0,
            msg=lambda msg, i=i: f"weight{i} main_grad changed before backward_dw\n{msg}",
        )

    layer.backward_dw()

    reference = _fp32_wgrad_reference(inp, grad_out, m_splits)
    for i in range(num_gemms):
        expected = reference[i] if overwrite_main_grad else reference[i] + _PREFILL
        torch.testing.assert_close(
            getattr(layer, f"weight{i}").main_grad,
            expected,
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad mismatch "
                f"(delay, overwrite_main_grad={overwrite_main_grad})\n{msg}"
            ),
        )


def test_grouped_linear_fuse_wgrad_delayed_first_microbatch_overwrites():
    """Delayed wgrad must honor ``is_first_microbatch=True`` at flush time.

    The overwrite decision is evaluated inside the deferred closure, so a first
    microbatch with ``delay_wgrad_compute=True`` must still discard the stale
    ``main_grad`` content when ``backward_dw()`` flushes, and must leave it
    untouched before the flush.
    """
    torch.manual_seed(55)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)

    layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
        delay_wgrad_compute=True,
    )
    _fill_main_grad(layer, _PREFILL)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    out = layer(inp, m_splits, is_first_microbatch=True)
    out.backward(grad_out)

    assert layer.need_backward_dw(), "delay_wgrad_compute=True must set need_backward_dw"
    for i in range(num_gemms):
        weight = getattr(layer, f"weight{i}")
        torch.testing.assert_close(
            weight.main_grad,
            torch.full_like(weight, _PREFILL, dtype=torch.float32),
            rtol=0,
            atol=0,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad changed before backward_dw "
                f"(delay, is_first_microbatch=True)\n{msg}"
            ),
        )

    layer.backward_dw()

    reference = _fp32_wgrad_reference(inp, grad_out, m_splits)
    for i in range(num_gemms):
        torch.testing.assert_close(
            getattr(layer, f"weight{i}").main_grad,
            reference[i],
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad mismatch (delay, is_first_microbatch=True)\n{msg}"
            ),
        )


def test_grouped_linear_fuse_wgrad_overwrite_flag_is_layer_wide():
    """The layer-wide ``overwrite_main_grad`` flag is read from ``weight0``.

    Consistent with ``_PerformanceGroupedLinear``: the flag is cached once at
    forward time from ``weight0`` and applies to every expert, so per-expert
    flag differences are not honored. With ``weight0`` flagged, all experts
    must overwrite the sentinel.
    """
    torch.manual_seed(53)
    num_gemms = _NUM_GEMMS
    in_features = _IN_FEATURES
    out_features = _OUT_FEATURES
    seq_len = _SEQ_LEN
    m_splits = _even_m_splits(seq_len, num_gemms)
    overwrite_flags = [True, False]

    layer = _make_grouped_linear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        fuse_wgrad_accumulation=True,
        parallel_mode="column",
    )
    for i, flag in enumerate(overwrite_flags):
        getattr(layer, f"weight{i}").overwrite_main_grad = flag
    _fill_main_grad(layer, _PREFILL)

    inp = torch.randn(seq_len, in_features, dtype=torch.bfloat16, device=_npu())
    grad_out = torch.randn(seq_len, out_features, dtype=torch.bfloat16, device=_npu())
    _run_forward_backward(layer, inp, m_splits, grad_out)

    reference = _fp32_wgrad_reference(inp, grad_out, m_splits)
    for i in range(num_gemms):
        torch.testing.assert_close(
            getattr(layer, f"weight{i}").main_grad,
            reference[i],
            **_INTEGRATION_TOLS,
            msg=lambda msg, i=i: (
                f"weight{i} main_grad mismatch (layer-wide overwrite from weight0)\n{msg}"
            ),
        )
