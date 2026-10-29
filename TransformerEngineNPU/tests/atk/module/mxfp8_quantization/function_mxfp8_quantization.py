# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for the MXFP8 quantization precision guard.

Golden-free module (see the workspace-root ATK_REPORT.md (outside this repo) section 4-P2(2)): NVTE's MXFP8
cast kernels (``quantize_mxfp8.cuh`` / ``dequantize_mxfp8.cuh``) are entirely
compiled behind ``__CUDA_ARCH__ >= 1000`` with no fallback, so no intranet CUDA
machine (A100-class) can collect an NVTE golden. The guard therefore anchors on
hardware-independent invariants, with a CPU-computed reference collected by
``collect_cpu_reference.py``:

1. **Round-trip theoretical bound** (per element, universal for ANY conforming
   MXFP8 codec regardless of E8M0 rounding convention):
   ``|dequant(quantize(x)) - x| <= atol + rtol * |x|`` with ``rtol = 0.125``
   (e4m3: 3-bit mantissa -> 2^-4 rounding + 1 bit E8M0 headroom -> 2^-3) and
   ``rtol = 0.25`` (e5m2). This is the methodology of NVTE's own
   ``TestMXFP8Tensor`` — comparing against the original input, never against a
   simulated quantization (a CPU-simulated rounding schedule is just as
   convention-specific as the kernel and would misjudge legal differences).
2. **Rowwise <-> columnwise cross consistency**: the dual-axis kernel emits two
   independent payloads (K-blocked and M-blocked 1x32/32x1); their dequantized
   results must agree with each other within twice the round-trip bound.
3. **Scale canonicity vs CPU canonical value**: the decoded E8M0 scale of each
   block must lie within ONE exponent (factor of 2) of the exact CPU value
   ``amax_block / fp8_max``. The window deliberately covers ceil / nearest /
   floor E8M0 conventions (all legal) while rejecting off-by-one-exponent bugs,
   saturated scales and mis-decoded packed layouts.

**Bound normalization**: the mixed-tolerance comparator thresholds are keyed by
the OUTPUT dtype (always fp32 here), not by the FP8 format. To keep one set of
yaml thresholds correct for both formats, every round-trip / cross output is
emitted divided by ``bound = 1.0 (e4m3) / 2.0 (e5m2)`` and the CPU reference is
divided by the same factor, so ``rtol = 0.125`` applies uniformly. The
transformation is exact (both sides scale identically) and recorded in the
contract (``bound_factor``).
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


# Shared accuracy comparators (allclose / mare / mixed_tolerance) live in
# ../_comparators.py and are registered here under this module's prefix so
# the yaml standard.acc block can select any of the three standards.
import os
import sys

if _ATK_IMPORT_ERROR is None:
    # Comparator registration is only needed when running as an ATK plugin.
    _comparators_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _comparators_dir not in sys.path:
        sys.path.insert(0, _comparators_dir)
    from _comparators import register_module_comparators  # noqa: E402

    register_module_comparators("mxfp8_quantization")

RESULT_SCHEMA = [
    "dequant_rowwise",  # dequantize(rowwise payload)/bound, fp32
    "dequant_columnwise",  # dequantize(columnwise payload)/bound, fp32
    "cross_consistency",  # |rowwise - columnwise|/bound, fp32 (ref: zeros)
    "scale_rowwise_decoded",  # decoded rowwise scales, fp32 [M, K/32] (ref: CPU canonical)
    "scale_columnwise_decoded",  # decoded columnwise scales, fp32 [M/32, K] (ref: CPU canonical)
    "api_contract_json_utf8",
]

# MXFP8 block size (both axes) and FP8 max finite values.
MX_BLOCK = 32
_FP8_MAX = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}

# Theoretical round-trip relative bounds: 2^-3 (e4m3) / 2^-2 (e5m2). They hold
# for any conforming codec independent of the E8M0 rounding convention — the
# property that makes them usable without an NVTE golden. The emitted tensors
# are divided by these so a single fp32 rtol (0.125) applies to both formats.
_RTOL = {
    "e4m3": 0.125,
    "e5m2": 0.25,
}


def bound_factor(fp8_format: str) -> float:
    """Emit-side normalization factor (_RTOL / 0.125)."""
    return _RTOL[fp8_format] / 0.125


def _resolve_fp8_dtype(fp8_format: str):
    """Return the FP8 dtype tag accepted by the running implementation."""
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
        "mx_block": MX_BLOCK,
        "bound_factor": bound_factor(fp8_format),
        "guard_mode": "golden_free",
        "rowwise_usage": True,
        "columnwise_usage": True,
    }


def _decode_scale_inv(scale_inv: torch.Tensor) -> torch.Tensor:
    """Decode an E8M0 scale buffer to fp32 values (byte b -> 2^(b-127)).

    Mirrors ``MXFP8TensorStorage._decode_scale_inv``: non-uint8 buffers are
    already numeric (some kernels emit e8m0fnu / fp32) and pass through.
    """
    if scale_inv.dtype != torch.uint8:
        try:
            return scale_inv.to(torch.float32)
        except (TypeError, RuntimeError):
            scale_inv = scale_inv.to(torch.uint8)
    return torch.pow(2.0, scale_inv.to(torch.float32) - 127.0)


def _logical_scale_matrix(
    scale_inv: torch.Tensor, m_dim: int, k_dim: int, *, axis: str
) -> torch.Tensor:
    """Decode TENPU's packed E8M0 scale buffer into the logical fp32 grid.

    Mirrors the production decode order (``_decode_scale_inv`` +
    ``_reshape_scale_inv(keep_first_dim=True)`` + the packed-3D permute in
    ``_dequantize_axis_m_blocks``), so the guard observes the layout the
    dequantize path itself would use:

    - axis "k" (rowwise payload): one scale per 1x32 block along K ->
      logical [m_dim, ceil(k_dim/32)]
    - axis "m" (columnwise payload): one scale per 32x1 block along M ->
      logical [ceil(m_dim/32), k_dim]
    """
    decoded = _decode_scale_inv(scale_inv)
    if axis == "k":
        k_blocks = math.ceil(k_dim / MX_BLOCK)
        if decoded.ndim >= 2 and decoded.numel() >= m_dim * k_blocks:
            return decoded.reshape(m_dim, -1)[:, :k_blocks].contiguous()
        return decoded.reshape(-1, k_blocks)[:m_dim, :].contiguous()
    # axis == "m": production permute for packed 3D layouts
    m_blocks = math.ceil(m_dim / MX_BLOCK)
    if (
        decoded.ndim >= 3
        and decoded.shape[0] * decoded.shape[-1] >= m_blocks
        and decoded.shape[1] >= k_dim
    ):
        return (
            decoded.permute(0, 2, 1).reshape(-1, decoded.shape[1])[:m_blocks, :k_dim].contiguous()
        )
    if decoded.ndim >= 2 and decoded.numel() >= m_blocks * k_dim:
        return decoded.reshape(m_blocks, -1)[:, :k_dim].contiguous()
    return decoded.reshape(-1, k_dim)[:m_blocks, :].contiguous()


def _coerce_to_logical(actual: torch.Tensor, logical: torch.Tensor, label: str) -> torch.Tensor:
    """Bring a decoded kernel scale tensor onto the canonical logical grid.

    A shape mismatch vs the canonical grid is itself signal (layout
    divergence), but tile-grid transposition is a legitimate stored-layout
    choice — accept it explicitly so the VALUES stay comparable.
    """
    if tuple(actual.shape) == tuple(logical.shape):
        return actual
    if tuple(actual.shape) == tuple(logical.shape)[::-1]:
        return actual.transpose(0, 1).contiguous()
    if actual.numel() == logical.numel():
        return actual.reshape(logical.shape)
    raise RuntimeError(
        "mxfp8 scale layout divergence ({}): kernel-decoded shape {} cannot be "
        "brought onto the canonical logical grid {}".format(
            label, tuple(actual.shape), tuple(logical.shape)
        )
    )


def cpu_canonical_scales(
    x_2d: torch.Tensor, fp8_format: str, block: int, axis: str
) -> torch.Tensor:
    """Exact CPU canonical scale per block: ``2^ceil(log2(amax / fp8_max))``.

    Pure CPU math used as the scale-canonicity REFERENCE. Convention analysis
    (validated locally, see README "scale-convention analysis"):

    - A legal scale must satisfy ``s >= amax / fp8_max`` (else the block amax
      saturates when cast). For powers of two, ``2^floor(log2(c)) <= c`` ALWAYS,
      so a floor convention saturates in ~100% of random blocks — it is NOT a
      legal choice. The one universally legal power of two is
      ``2^ceil(log2(c))`` ("canonical"), and scales up to one exponent ABOVE
      canonical are also legal (they only trade 1 bit of mantissa headroom;
      verified to still meet the round-trip bound).
    - One exponent BELOW canonical saturates the amax element with ~31%
      error — that case is caught by the round-trip outputs (0/1), while the
      scale window (rtol 1.0 = one exponent) accepts every legal choice and
      rejects anything two exponents or further off.

    All-zero blocks have amax 0 where any scale is legal; the generator's
    [0.01, 0.5] range keeps every block non-zero.
    """
    x = x_2d.to(torch.float64)
    fp8_max = _FP8_MAX[fp8_format]
    if axis == "k":
        pad = (-x.shape[1]) % block
        if pad:
            x = torch.nn.functional.pad(x, (0, pad, 0, 0), value=0.0)
        amax = x.reshape(x.shape[0], -1, block).abs().amax(dim=-1)  # [M, K/32]
    else:
        pad = (-x.shape[0]) % block
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad), value=0.0)
        amax = x.reshape(-1, block, x.shape[1]).abs().amax(dim=1)  # [M/32, K]
    c = amax / fp8_max
    canonical = torch.pow(2.0, torch.ceil(torch.log2(c)))
    # amax == 0 blocks (all-zero) have no defined canonical scale; emit 1 so
    # the comparator window degrades gracefully instead of producing nan.
    canonical = torch.where(amax > 0, canonical, torch.ones_like(canonical))
    return canonical.to(torch.float32)


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

    canonical_row = cpu_canonical_scales(x_2d, fp8_format, MX_BLOCK, axis="k")
    canonical_col = cpu_canonical_scales(x_2d, fp8_format, MX_BLOCK, axis="m")
    # Structural fields only (shape/dtype of the ORIGINAL x, format, bound) —
    # device-independent, so the online CUDA-side contract is byte-identical
    # to the offline one.
    contract = _make_contract(x, fp8_format)
    outputs = (
        x32 / bound,  # round-trip reference (rowwise payload)
        x32 / bound,  # round-trip reference (columnwise payload)
        torch.zeros_like(x32),  # cross-consistency reference
        canonical_row,
        canonical_col,
        _encode_contract(contract, torch.device("cpu")),
    )
    if out_device is None:
        return outputs
    return tuple(t.to(out_device) for t in outputs)


def run_mxfp8_quantization_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one golden-free MXFP8 quantize/dequantize case on the device.

    Input layout: ``x, fp8_format(str)``. On the NPU the kernel runs and the
    block scales are read off the resulting tensor; on the GPU side of the
    ONLINE comparison the CPU reference is carried instead (see the cuda
    branch) — the guard compares "NPU kernel vs exact math" on both paths.
    """
    values = list(input_data.args)
    if len(values) != 2:
        raise ValueError(
            "mxfp8_quantization ATK case expects 2 positional inputs (x, fp8_format), "
            "got {}".format(len(values))
        )
    x, fp8_format = values
    fp8_format = str(fp8_format)
    if fp8_format not in _FP8_MAX:
        raise ValueError("unsupported fp8_format: {} (want e4m3/e5m2)".format(fp8_format))

    if device in ("gpu", "cuda"):
        # Golden-free ONLINE reference. NVTE's MXFP8 kernel is compiled behind
        # __CUDA_ARCH__ >= 1000 with NO fallback — on smaller GPUs the kernel
        # body is empty and the launch "succeeds" with uninitialized output,
        # so this branch used to fail closed. It now carries the CPU reference
        # instead: the exact artifact set the offline collector produces, from
        # the same single-source function. The reference is derived from the
        # case input, so a regenerated case set can never race a stale
        # baseline, and no reference artifacts need to be stored or staged.
        return cpu_reference_outputs(x, fp8_format, out_device=x.device)

    # Clear, early diagnostic when the device cannot run the kernel path at all
    # (MXFP8 needs an A5-class NPU); the raw ACL error would be far less useful.
    try:
        from transformer_engine.pytorch.quantization.utils import is_mxfp8_available

        available, reason = is_mxfp8_available(return_reason=True)
        if not available:
            raise RuntimeError(
                "mxfp8_quantization guard cannot run on this device: {}".format(reason)
            )
    except ImportError:
        pass

    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    x = x.detach().clone()
    fp8_dtype = _resolve_fp8_dtype(fp8_format)
    quantizer = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    q = quantizer(x)

    m_dim = 1
    for dim in list(x.shape)[:-1]:
        m_dim *= dim
    k_dim = x.shape[-1]
    x_2d = x.detach().reshape(m_dim, k_dim)
    bound = bound_factor(fp8_format)

    # Order matters: update_usage(rowwise_usage=False) frees the rowwise
    # payload, so dequantize-from-rowwise and both scale reads come first
    # (the same order NVTE's own TestMXFP8Tensor uses).
    deq_row = q.dequantize(dtype=torch.float32)
    row_scale = _logical_scale_matrix(q._rowwise_scale_inv.detach().cpu(), m_dim, k_dim, axis="k")
    col_scale = _logical_scale_matrix(
        q._columnwise_scale_inv.detach().cpu(), m_dim, k_dim, axis="m"
    )

    q.update_usage(rowwise_usage=False, columnwise_usage=True)
    deq_col = q.dequantize(dtype=torch.float32)

    cross = (deq_row - deq_col).abs().to(torch.float32)

    # Scale canonicity: bring the decoded kernel scales onto the canonical
    # logical grids so the VALUES compare elementwise against the CPU
    # reference computed by collect_cpu_reference.py.
    canonical_row = cpu_canonical_scales(x_2d.cpu(), fp8_format, MX_BLOCK, axis="k")
    canonical_col = cpu_canonical_scales(x_2d.cpu(), fp8_format, MX_BLOCK, axis="m")
    row_scale = _coerce_to_logical(row_scale, canonical_row, "rowwise")
    col_scale = _coerce_to_logical(col_scale, canonical_col, "columnwise")

    contract = _make_contract(x, fp8_format)
    return (
        deq_row.to(torch.float32) / bound,
        deq_col.to(torch.float32) / bound,
        cross / bound,
        row_scale.to(x.device),
        col_scale.to(x.device),
        _encode_contract(contract, deq_row.device),
    )


@register("function_mxfp8_quantization")
class FunctionMXFP8Quantization(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend has no role: the NPU runs the kernel under test, and
    the GPU side of the ONLINE comparison carries the golden-free CPU
    reference (``cpu_reference_outputs`` — round-trip reference = the input
    itself, scale reference = exact canonical values), because NVTE's MXFP8
    kernel is SM100-only. The same single-source function backs the offline
    collector (``collect_cpu_reference.py``).
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_mxfp8_quantization_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server): the case function's cuda
            # path carries the CPU reference (NVTE's kernel is SM100-only) —
            # see run_mxfp8_quantization_case. 'gpu' is the ATK backend label.
            result = run_mxfp8_quantization_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "mxfp8_quantization supports 'npu' (kernel under test) and "
                "'gpu'/'cuda' (golden-free CPU reference for the online "
                "comparison); the 'cpu' backend has no role. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
