# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for the Float8Blockwise precision guard.

Golden-free module (see the workspace-root ATK_REPORT.md (outside this repo) section 4-P2(2)): NVTE's
blockwise quantize kernel (``quantize_transpose_square_blockwise.cu``) is a TMA
implementation compiled behind ``__CUDA_ARCH__ >= 900`` (Hopper) — the intranet
A100-class machines cannot collect a golden for it either. Same anchoring as
the MXFP8 module: hardware-independent invariants + a CPU-computed reference.

Structure mirrors ``../mxfp8_quantization``; differences specific to 128x128
blockwise scaling:

- Block size is 128 (``Float8BlockQuantizer.block_len``); the default
  ``block_scaling_dim=2`` uses 128x128 blocks for BOTH usages (TENPU calls
  ``npu_dynamic_block_quant`` twice with the same block dims — rowwise
  1x128/128x128 per ``get_block_dim`` — and the guard verifies both payloads
  round-trip independently).
- Scales are **fp32 values, not E8M0 bytes** — no exponent decoding. The
  canonicity reference is the exact CPU ``amax_block / fp8_max`` (unrounded
  fp32, no pow-2 constraint on TENPU: ``force_pow_2_scales`` is a dead knob
  that never reaches ``npu_dynamic_block_quant``), and the legal window is a
  plain relative tolerance: any scale the kernel produces that reconstructs
  the block without saturation satisfies ``s >= amax / fp8_max``; combined
  with the round-trip outputs, the same trap-closure argument as MXFP8
  applies (a too-small scale saturates the amax element and fails round-trip;
  a legal scale is within ``[amax/fp8_max, ...)``).
- The stored scale layout differs across implementations (NVTE pads the
  untiled dim to 4 and stores columnwise data transposed; TENPU uses plain
  ceil grids and stores columnwise untransposed), so — exactly as with the
  MXFP8 module — the guard compares **decoded dequantize outputs and the
  logical-grid scale values**, never raw buffer bytes.

Round-trip theoretical bound: identical to MXFP8 (fp8 mantissa rounding
dominates; blockwise fp32 scales lose NO headroom, so the bound is actually
tighter in practice, but the same 0.125/0.25 thresholds apply).
"""

import json
import math
from typing import Dict, Tuple

import torch

try:
    from atk.tasks.api_execute import register

    # ATK's API plugin loader resolves both BaseApi symbols from this namespace.
    from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi  # noqa: F401
    from atk.tasks.api_execute.base_api import BaseApi

    _ATK_IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - plain-python tooling mode
    # The CPU-reference collector (collect_cpu_reference.py) imports this
    # module for the pure math helpers (canonical scales, bound factor,
    # contract encoder) on hosts whose ATK install has a broken plugin
    # registry. Only the adapter class needs ATK; everything else works
    # without it, so degrade to a no-op registration instead of failing.
    def register(*_args, **_kwargs):  # noqa: E731  (no-op fallback, see below)
        """No-op decorator when ATK is absent (plain-python tooling mode)."""

        def _wrap(cls):
            return cls

        return _wrap

    AclnnBaseApi = BaseApi = object  # noqa: F401
    _ATK_IMPORT_ERROR = _exc


# Shared accuracy comparators live in ../_comparators.py.
import os
import sys

if _ATK_IMPORT_ERROR is None:
    # Comparator registration is only needed when running as an ATK plugin.
    _comparators_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _comparators_dir not in sys.path:
        sys.path.insert(0, _comparators_dir)
    from _comparators import register_module_comparators  # noqa: E402

    register_module_comparators("blockwise_quantization")

RESULT_SCHEMA = [
    "dequant_rowwise",  # dequantize(rowwise payload)/bound, fp32
    "dequant_columnwise",  # dequantize(columnwise payload)/bound, fp32
    "cross_consistency",  # |rowwise - columnwise|/bound, fp32 (ref: zeros)
    "scale_rowwise_logical",  # logical-grid rowwise scales (ref: CPU canonical)
    "scale_columnwise_logical",  # logical-grid columnwise scales (ref: CPU canonical)
    "api_contract_json_utf8",
]

# 128x128 blockwise scaling (Float8BlockQuantizer.block_len; dim=2 means both
# rowwise and columnwise use 128x128 tiles).
BLOCK = 128
_FP8_MAX = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}

# Theoretical round-trip relative bounds: 2^-3 (e4m3) / 2^-2 (e5m2) — same
# mantissa-rounding derivation as the MXFP8 module (blockwise fp32 scales lose
# no E8M0 headroom, so real errors are strictly tighter). Emitted tensors are
# divided by these so one fp32 rtol (0.125) covers both formats.
_RTOL = {
    "e4m3": 0.125,
    "e5m2": 0.25,
}


def bound_factor(fp8_format: str) -> float:
    """Emit-side normalization factor (_RTOL / 0.125)."""
    return _RTOL[fp8_format] / 0.125


def _resolve_fp8_dtype(fp8_format: str):
    mapping = {
        "e4m3": torch.float8_e4m3fn,
        "e5m2": torch.float8_e5m2,
    }
    return mapping[fp8_format]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _make_contract(x, fp8_format) -> Dict:
    """Structural / format fields only — data-dependent floats excluded."""
    return {
        "fp8_format": fp8_format,
        "fp8_dtype": "torch.float8_e4m3fn" if fp8_format == "e4m3" else "torch.float8_e5m2",
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "block": BLOCK,
        "block_scaling_dim": 2,
        "bound_factor": bound_factor(fp8_format),
        "guard_mode": "golden_free",
        "rowwise_usage": True,
        "columnwise_usage": True,
    }


def _logical_grid(scale_inv: torch.Tensor, m_dim: int, k_dim: int) -> torch.Tensor:
    """Bring a stored blockwise scale tensor onto the [M/128, K/128] logical grid.

    TENPU stores plain ceil grids (``get_scale_shape``); NVTE pads the untiled
    dim to a multiple of 4 and stores the columnwise grid transposed. Both are
    legitimate stored-layout choices — the guard slices/normalizes onto the
    logical grid so the VALUES compare, and raises on anything genuinely
    incompatible (numel too small).
    """
    m_tiles = math.ceil(m_dim / BLOCK)
    k_tiles = math.ceil(k_dim / BLOCK)
    if scale_inv.ndim == 3:
        scale_inv = scale_inv.reshape(scale_inv.shape[0], scale_inv.shape[-2], -1)[:, :, 0]
    if tuple(scale_inv.shape) == (k_tiles, m_tiles):
        scale_inv = scale_inv.transpose(0, 1)
    if scale_inv.ndim != 2 or scale_inv.numel() < m_tiles * k_tiles:
        raise RuntimeError(
            "blockwise scale layout divergence: stored shape {} cannot be brought "
            "onto the logical grid ({}, {})".format(tuple(scale_inv.shape), m_tiles, k_tiles)
        )
    return scale_inv.reshape(m_tiles, -1)[:, :k_tiles].contiguous()


def cpu_canonical_scales(x_2d: torch.Tensor, fp8_format: str) -> torch.Tensor:
    """Exact CPU canonical 128x128 blockwise scale (amax / fp8_max, fp32).

    Unlike MXFP8's E8M0 scales there is no power-of-two constraint on TENPU
    (``force_pow_2_scales`` never reaches ``npu_dynamic_block_quant``), so the
    canonical value is the unrounded fp32 ``amax_block / fp8_max``. A legal
    kernel scale satisfies ``s >= canonical`` (no saturation); the comparator
    window (rtol 1.0 vs canonical) accepts it, and a below-canonical scale
    saturates the amax element, which the round-trip outputs catch.
    """
    x = x_2d.to(torch.float64)
    m, k = x.shape
    fp8_max = _FP8_MAX[fp8_format]
    pad_m = (-m) % BLOCK
    pad_k = (-k) % BLOCK
    if pad_m or pad_k:
        x = torch.nn.functional.pad(x, (0, pad_k, 0, pad_m), value=0.0)
    tiles = x.reshape(x.shape[0] // BLOCK, BLOCK, x.shape[1] // BLOCK, BLOCK)
    amax = tiles.abs().amax(dim=(1, 3))  # [M/128, K/128]
    canonical = amax / fp8_max
    return torch.where(amax > 0, canonical, torch.ones_like(canonical)).to(torch.float32)


def cpu_reference_outputs(x, fp8_format, out_device=None):
    """Golden-free CPU reference outputs for one case — the single source.

    Both the offline collector (``collect_cpu_reference.py``) and the ONLINE
    GPU-side adapter branch call this function, so the two reference paths
    cannot drift. Always computed on plain CPU tensors: the reference is
    exact math derived from the input (round-trip reference = the input
    itself; scale reference = the canonical amax/fp8_max), never a kernel
    simulation — ``out_device`` only decides where the RESULT tensors land.
    """
    fp8_format = str(fp8_format)
    bound = bound_factor(fp8_format)
    x32 = x.detach().to(device="cpu", dtype=torch.float32)
    m_dim = 1
    for dim in list(x32.shape)[:-1]:
        m_dim *= dim
    k_dim = x32.shape[-1] if x32.dim() > 0 else 1
    x_2d = x32.reshape(m_dim, k_dim)

    canonical = cpu_canonical_scales(x_2d, fp8_format)
    # Structural fields only (shape/dtype of the ORIGINAL x, format, bound) —
    # device-independent, so the online CUDA-side contract is byte-identical
    # to the offline one.
    contract = _make_contract(x, fp8_format)
    outputs = (
        x32 / bound,  # round-trip reference (rowwise payload)
        x32 / bound,  # round-trip reference (columnwise payload)
        torch.zeros_like(x32),  # cross-consistency reference
        canonical,
        canonical,
        _encode_contract(contract, torch.device("cpu")),
    )
    if out_device is None:
        return outputs
    return tuple(t.to(out_device) for t in outputs)


def run_blockwise_quantization_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one golden-free Float8Blockwise quantize/dequantize case on the device.

    Input layout: ``x, fp8_format(str)``. On the NPU the kernel runs and the
    block scales are read off the resulting tensor; on the GPU side of the
    ONLINE comparison the CPU reference is carried instead (see the cuda
    branch) — the guard compares "NPU kernel vs exact math" on both paths.
    """
    values = list(input_data.args)
    if len(values) != 2:
        raise ValueError(
            "blockwise_quantization ATK case expects 2 positional inputs (x, fp8_format), "
            "got {}".format(len(values))
        )
    x, fp8_format = values
    fp8_format = str(fp8_format)
    if fp8_format not in _FP8_MAX:
        raise ValueError("unsupported fp8_format: {} (want e4m3/e5m2)".format(fp8_format))

    if device in ("gpu", "cuda"):
        # Golden-free ONLINE reference. NVTE's blockwise kernel is a TMA
        # implementation compiled behind __CUDA_ARCH__ >= 900 (Hopper, and
        # needs CUDA >= 12.9) — on smaller GPUs the kernel body is empty and
        # the launch "succeeds" with uninitialized output, so this branch used
        # to fail closed. It now carries the CPU reference instead: the exact
        # artifact set the offline collector produces, from the same
        # single-source function. The reference is derived from the case
        # input, so a regenerated case set can never race a stale baseline,
        # and no reference artifacts need to be stored or staged.
        return cpu_reference_outputs(x, fp8_format, out_device=x.device)

    # Clear, early diagnostic when the device cannot run the kernel path
    # (blockwise scaling needs an A5-class NPU).
    try:
        from transformer_engine.pytorch.quantization.utils import (
            is_fp8_block_scaling_available,
        )

        available, reason = is_fp8_block_scaling_available(return_reason=True)
        if not available:
            raise RuntimeError(
                "blockwise_quantization guard cannot run on this device: {}".format(reason)
            )
    except ImportError:
        pass

    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    x = x.detach().clone()
    fp8_dtype = _resolve_fp8_dtype(fp8_format)
    quantizer = Float8BlockQuantizer(
        fp8_dtype=fp8_dtype, rowwise=True, columnwise=True, block_scaling_dim=2
    )
    q = quantizer(x)

    m_dim = 1
    for dim in list(x.shape)[:-1]:
        m_dim *= dim
    k_dim = x.shape[-1]
    bound = bound_factor(fp8_format)

    # Order matters: update_usage(rowwise_usage=False) frees the rowwise
    # payload, so dequantize-from-rowwise and both scale reads come first.
    deq_row = q.dequantize(dtype=torch.float32)
    row_scale = _logical_grid(q._rowwise_scale_inv.detach().cpu(), m_dim, k_dim)
    col_scale = _logical_grid(q._columnwise_scale_inv.detach().cpu(), m_dim, k_dim)

    q.update_usage(rowwise_usage=False, columnwise_usage=True)
    deq_col = q.dequantize(dtype=torch.float32)

    cross = (deq_row - deq_col).abs().to(torch.float32)

    contract = _make_contract(x, fp8_format)
    return (
        deq_row.to(torch.float32) / bound,
        deq_col.to(torch.float32) / bound,
        cross / bound,
        row_scale.to(x.device),
        col_scale.to(x.device),
        _encode_contract(contract, deq_row.device),
    )


@register("function_blockwise_quantization")
class FunctionBlockwiseQuantization(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend has no role: the NPU runs the kernel under test, and
    the GPU side of the ONLINE comparison carries the golden-free CPU
    reference (``cpu_reference_outputs`` — round-trip reference = the input
    itself, scale reference = exact canonical values), because NVTE's
    blockwise kernel is SM90+-only. The same single-source function backs the
    offline collector (``collect_cpu_reference.py``).
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_blockwise_quantization_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server): the case function's cuda
            # path carries the CPU reference (NVTE's kernel is SM90+-only) —
            # see run_blockwise_quantization_case. 'gpu' is the ATK backend label.
            result = run_blockwise_quantization_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "blockwise_quantization supports 'npu' (kernel under test) and "
                "'gpu'/'cuda' (golden-free CPU reference for the online "
                "comparison); the 'cpu' backend has no role. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
