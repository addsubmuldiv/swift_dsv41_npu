# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.
"""ATK FP8 quantize/dequantize comparison on NPU and GPU."""

import json
import torch
from typing import Dict, Tuple

import os
import sys

from atk.tasks.api_execute import register

# ATK's plugin loader resolves both BaseApi symbols from this namespace.
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi  # noqa: F401
from atk.tasks.api_execute.base_api import BaseApi

_module_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_module_dir)
for path in (_parent_dir, _module_dir):
    if path not in sys.path:
        sys.path.insert(0, path)
from _comparators import register_module_comparators  # noqa: E402
from _nonfinite_contract import special_value_signatures  # noqa: E402

register_module_comparators("float8_quantization")


RESULT_SCHEMA = [
    "dequantized",  # FP8 dequantize round-trip (float32) — tolerance compare
    "raw_fp8",  # uint8 raw FP8 payload — exact compare (non-float)
    "scale_metadata",  # fp8_scale_inv (float32) — tolerance compare
    "api_contract_json_utf8",
]

# FP8 max finite values (delayed-scaling init scale multiplier: fp8_max / amax).
_FP8_MAX = {
    "e4m3": 448.0,
    "e5m2": 57344.0,
}


def _resolve_fp8_dtype(fp8_format: str):
    """Return the FP8 dtype tag accepted by the running implementation.

    The two implementations represent FP8 dtypes differently:
    - NVTE (CUDA golden host): the compiled ``transformer_engine_torch`` pybind
      enum, ``tex.DType.kFloat8E4M3 / kFloat8E5M2`` (its ``Float8Quantizer``
      does ``DType.cast(fp8_dtype)`` which accepts this).
    - TENPU (NPU): a plain ``torch.dtype``, ``torch.float8_e4m3fn`` /
      ``torch.float8_e5m2`` (its ``Float8Quantizer`` takes ``fp8_dtype:
      torch.dtype``).

    ``transformer_engine_torch.DType`` exists on both sides now (TENPU ships a
    pure-torch mirror enum in the ``transformer_engine_torch`` package), so the
    try-branch succeeds on NPU as well; TENPU's ``Float8Quantizer`` accepts the
    enum and maps it to the matching ``torch.dtype`` internally. The
    torch.dtype fallback only covers builds without the compat package.
    """
    if fp8_format == "e4m3":
        try:
            import transformer_engine_torch as tex  # noqa: F401

            return tex.DType.kFloat8E4M3
        except ImportError:
            return torch.float8_e4m3fn
    try:
        import transformer_engine_torch as tex  # noqa: F401

        return tex.DType.kFloat8E5M2
    except ImportError:
        return torch.float8_e5m2


def _encode_api_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _build_api_contract(x, fp8_format, dequant, raw) -> Dict:
    # Every field here must serialize to identical bytes on both the NVTE and the
    # NPU side (the contract is exact-compared). Only structural/format fields are
    # kept — data-dependent floats (amax/scale) are deliberately excluded; they are
    # already verified by the dequant/raw/scale outputs.
    return {
        "fp8_format": fp8_format,
        "fp8_dtype": "torch.float8_e4m3fn" if fp8_format == "e4m3" else "torch.float8_e5m2",
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "dequant_dtype": str(dequant.dtype),
        "raw_dtype": str(raw.dtype),
        "raw_shape": list(raw.shape),
    }


def run_float8_quantization_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one FP8 quantize/dequantize round-trip case using canonical ATK inputs.

    Run the installed TE implementation on the input device. No CPU reference
    or ATK registration is involved in this execution function.

    Input layout: ``x, fp8_format(str)``. The FP8 dtype and per-tensor scale are
    derived deterministically from ``x`` (test initialization convention
    ``scale = fp8_max / amax``), so the golden and the NPU side quantize the
    identical value and the scale is not part of the case inputs.
    """
    values = list(input_data.args)
    if len(values) != 2:
        raise ValueError(
            "quantization ATK case expects 2 positional inputs (x, fp8_format), got {}".format(
                len(values)
            )
        )
    x, fp8_format = values
    fp8_format = str(fp8_format)
    if fp8_format not in _FP8_MAX:
        raise ValueError("unsupported fp8_format: {} (want e4m3/e5m2)".format(fp8_format))

    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer  # noqa: F401

    x = x.detach().clone()
    fp8_dtype = _resolve_fp8_dtype(fp8_format)
    if not x.numel() or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("FP8 guard expects a nonempty fp16/bf16/fp32 input")
    amax = x.abs().max().to(torch.float32).reshape(1)
    nonfinite_input = not bool(torch.isfinite(amax).all())
    # TE scale is a multiplier: q = cast_fp8(x * scale). Pass the same
    # convention on both devices; the quantizer stores its reciprocal.
    # Nonfinite extra cases test propagation at a fixed valid scale; amax
    # calibration itself is undefined for these inputs. Do not sanitize x.
    scale = (
        torch.ones_like(amax)
        if nonfinite_input
        else torch.where(amax > 0, _FP8_MAX[fp8_format] / amax, torch.ones_like(amax))
    )
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("FP8 guard scale is not representable as positive FP32")
    quantizer = Float8Quantizer(scale, amax.reshape(1), fp8_dtype)
    fp8_t = quantizer.quantize(x.reshape(1, 1) if x.ndim == 0 else x)

    dequant = fp8_t.dequantize(dtype=torch.float32).reshape(x.shape)
    raw = fp8_t._data  # raw FP8 payload (NVTE: uint8, TENPU: torch.float8_*)
    if raw.dtype != torch.uint8:
        # TENPU stores the raw payload as torch.float8_e4m3fn/e5m2; reinterpret the
        # bits as uint8 (same 1-byte itemsize, bit-preserving) so both sides emit
        # the same dtype and the exact raw compare / serialized contract match.
        raw = raw.contiguous().view(torch.uint8)
    if tuple(raw.shape) != tuple(x.shape):
        # TENPU stores a flattened 2D payload; compare the original logical
        # shape on both devices without changing the underlying bytes.
        raw = raw.reshape(x.shape)
    scale_inv = fp8_t._scale_inv  # reciprocal of the applied scale

    contract = _build_api_contract(x, fp8_format, dequant, raw)
    if nonfinite_input:
        contract["special_value_signatures"] = special_value_signatures((dequant, scale_inv))
    return (dequant, raw, scale_inv, _encode_api_contract(contract, dequant.device))


@register("function_float8_quantization")
class FunctionQuantization(BaseApi):
    """Execute on NPU or GPU; this adapter provides no CPU output reference."""

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_float8_quantization_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_float8_quantization_case(input_data, "cuda")
        else:
            raise RuntimeError(f"FP8 case requires npu/gpu/cuda, got {self.device}")
        return result if with_output else None
