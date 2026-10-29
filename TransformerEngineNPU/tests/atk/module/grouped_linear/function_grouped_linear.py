# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine GroupedLinear.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of
``transformer_engine.pytorch.GroupedLinear``. NVTE consistency comes from the
offline golden loop (``collect_nvte_golden.py`` + an ATK ``accuracy_load``
node), never by a local torch reference.

This module exercises the per-expert grouped GEMM that backs Megatron's MoE
``TEGroupedLinear``: ``num_gemms`` experts, each with its own ``weight{i}`` /
``bias{i}`` parameters, an input tensor whose rows are split across experts by
``m_splits``, and the corresponding parameter gradients concatenated per slot.

**Exact baselines (golden side only).** NVTE's grouped wgrad comes out of
``general_grouped_gemm_for_grouped_tensor(layout="NT", ...)`` — the grouped
sibling of the cuBLASLt wgrad path whose in-GEMM accumulation is coarser than
the NPU's fp32 accumulation on long token counts (the ``linear`` guard proved
that family: NPU 1.0000 against exact fp64, NVTE 0.8802-0.9689; see
``../_exact_reference.py``). The generalized ``all_grouped_linear`` run shows
the same signature here: only ``output_2`` (``weights_grad_concatenated``) red,
matched_ratio 0.96-0.98 against the native defaults, no hard-gate violations,
worse for fp16 (atol 1.95e-3) than bf16 (1.56e-2) — an absolute deviation on
the near-zero elements of the per-expert ``dy^T @ x`` token reduction, which
local simulation shows pure fp32 accumulation *order* cannot produce. The
golden therefore carries exact fp64 references for the wgrad slot (and the
bgrad slot when bias is on); the comparison becomes "device vs exact math" — a
baseline choice, not a tolerance relaxation. ``forward_output`` and
``input_grad`` stay device-raw: their reductions are only hidden/ffn wide and
they pass at the defaults on the whole generalized set. The two-sided judge
that attributes the deviation is ``diagnose_slot_vs_fp64.py``; the negative
controls are ``verify_exact_baselines.py``.
"""

import json
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from atk.tasks.api_execute import register

# ATK's API plugin loader resolves both BaseApi symbols from this namespace.
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi  # noqa: F401
from atk.tasks.api_execute.base_api import BaseApi


# Shared accuracy comparators (allclose / mare / mixed_tolerance) live in
# ../_comparators.py and are registered here under this module's prefix so
# the yaml standard.acc block can select any of the three standards.
import os
import sys

_comparators_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _comparators_dir not in sys.path:
    sys.path.insert(0, _comparators_dir)
from _comparators import register_module_comparators  # noqa: E402
from _exact_reference import chunked_bgrad, chunked_wgrad, to_slot  # noqa: E402

# Disable TF32 for fp32 GEMM on CUDA (ATK server online mode): cuBLAS
# defaults to TF32 (10-bit mantissa, ~1e-3 relative error) on Ampere+
# GPUs — 10000x worse than true fp32. No-op on NPU (torch_npu does not
# go through cuBLAS). Must be set before any te.GroupedLinear forward.
torch.backends.cuda.matmul.allow_tf32 = False

register_module_comparators("grouped_linear")

RESULT_SCHEMA = [
    "forward_output",
    "input_grad",
    "weights_grad_concatenated",
    "biases_grad_concatenated_or_empty",
    "api_contract_json_utf8",
]

# Slots whose golden is synthesised as an exact fp64 reference instead of the
# NVTE kernel output, on the golden side only. See `_apply_exact_baselines`.
_WGRAD_SLOT = RESULT_SCHEMA.index("weights_grad_concatenated")
_BGRAD_SLOT = RESULT_SCHEMA.index("biases_grad_concatenated_or_empty")


def _golden_side_guard(device: str, tensors: Dict[str, torch.Tensor]) -> None:
    """Refuse to synthesise an exact baseline anywhere but the golden host.

    The local (NPU) adapter and the remote (golden/GPU) producer share one case
    function. If the substitution ever ran on both sides the ATK comparison would
    degenerate into reference-vs-reference: every case green, the guard stone
    blind, a real regression shipped silently. Hence a hard error, and an error
    rather than an assert so that ``python -O`` cannot strip it.
    """
    if str(device).split(":", maxsplit=1)[0].lower() not in ("cuda", "gpu"):
        raise RuntimeError(
            "exact_baselines is GOLDEN-SIDE ONLY; got device={!r}. The NPU "
            "side must keep the raw device outputs, otherwise the comparison "
            "becomes reference-vs-reference and can never fail.".format(device)
        )
    for name, tensor in tensors.items():
        if not (torch.is_tensor(tensor) and tensor.is_cuda):
            raise RuntimeError(
                "exact_baselines requires CUDA-resident device outputs; {} is "
                "not on CUDA, so this host cannot be producing the golden. Pass "
                "exact_baselines=False to judge a device's raw output.".format(name)
            )


@lru_cache(maxsize=1)
def _emit_baseline_banner() -> None:
    """One line per process, so the online golden path is at least greppable."""
    print(
        "[grouped_linear] exact fp64 baseline ACTIVE (slots: wgrad {}, bgrad {})".format(
            _WGRAD_SLOT, _BGRAD_SLOT
        )
    )


def _row_spans(m_splits: Sequence[int]) -> List[Tuple[int, int]]:
    """Contiguous ``(start, stop)`` row span per expert, in ``m_splits`` order.

    ``te.GroupedLinear`` splits the input's rows contiguously: expert ``e`` owns
    rows ``[sum(m_splits[:e]), sum(m_splits[:e+1]))``. The reference, the slot
    assembly and the module all have to agree on that, so it is derived once
    here.
    """
    spans = []
    offset = 0
    for rows in m_splits:
        rows = int(rows)
        if rows < 0:
            raise ValueError("negative m_splits entry: {}".format(list(m_splits)))
        spans.append((offset, offset + rows))
        offset += rows
    return spans


def grouped_wgrad_reference(
    x2d: torch.Tensor, dy2d: torch.Tensor, m_splits: Sequence[int]
) -> torch.Tensor:
    """Exact fp64 per-expert ``dw_e = dy_e^T @ x_e``, flattened and concatenated.

    Pure host math (no guard): the golden-side baseline, ``diagnose_slot_vs_fp64.py``
    and ``verify_exact_baselines.py`` all call it, so the judge and the baseline
    cannot drift into two definitions. Each expert's reduction sees only its own
    ``m_splits[e]`` rows — feeding the whole tensor to one GEMM would smear every
    expert's gradient together.
    """
    if x2d.shape[0] != dy2d.shape[0]:
        raise ValueError(
            "x and dy disagree on row count: {} vs {}".format(x2d.shape[0], dy2d.shape[0])
        )
    blocks = []
    for start, stop in _row_spans(m_splits):
        blocks.append(chunked_wgrad(dy2d[start:stop], x2d[start:stop]).reshape(-1))
    return torch.cat(blocks)


def grouped_bgrad_reference(dy2d: torch.Tensor, m_splits: Sequence[int]) -> torch.Tensor:
    """Exact fp64 per-expert ``db_e = colsum(dy_e)``, concatenated in slot order."""
    blocks = []
    for start, stop in _row_spans(m_splits):
        blocks.append(chunked_bgrad(dy2d[start:stop]))
    return torch.cat(blocks)


def _empty_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=reference.dtype, device=reference.device)


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _concatenate_grads(module, names: List[str], empty_reference: torch.Tensor) -> torch.Tensor:
    grads = []
    for name in names:
        parameter = getattr(module, name)
        if parameter.grad is not None:
            grads.append(parameter.grad.reshape(-1))
    if not grads:
        return _empty_like(empty_reference)
    return torch.cat(grads)


def _make_contract(module, output, m_splits, bias_enabled) -> Dict:
    return {
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "num_gemms": len(m_splits),
        "m_splits": list(m_splits),
        "bias_enabled": bool(bias_enabled),
        "named_parameters": [name for name, _ in module.named_parameters()],
        "parameter_shapes": {
            name: list(parameter.shape) for name, parameter in module.named_parameters()
        },
        "state_dict_keys": list(module.state_dict().keys()),
    }


def run_grouped_linear_case(
    input_data,
    device: str,
    *,
    exact_baselines: bool = False,
    raw_slots_out: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward GroupedLinear case using canonical ATK inputs.

    Input layout: ``inp, dy, weight_0..weight_{g-1}, bias_0..bias_{g-1},
    num_gemms(int), m_splits(str), bias_enabled(bool)``.

    ``exact_baselines`` is **golden-side only** and keyword-only on purpose: it
    replaces the weight-grad (and bias-grad) slots with exact fp64 references
    (``_apply_exact_baselines``) when this run is producing the golden. Any judge
    that needs to see a device's *raw* output — in particular
    ``diagnose_slot_vs_fp64.py``, the arbiter for "which side deviates" — must
    leave it off, or it would compare a reference against itself. The device
    string alone cannot tell the two roles apart (both say ``"cuda"``), which is
    why this is an explicit switch and not a device rule.

    ``raw_slots_out``, when given, receives the pre-substitution device slots
    keyed by output index. The collector uses it to record how far NVTE's own
    output sits from the baseline it was replaced by; it changes no output.
    """
    values = list(input_data.args)
    if len(values) < 5:
        raise ValueError(
            "GroupedLinear ATK case expects at least 5 inputs, got {}".format(len(values))
        )
    num_gemms = int(values[-3])
    m_splits = [int(x) for x in str(values[-2]).split(",")]
    bias_enabled = bool(values[-1])
    if len(m_splits) != num_gemms:
        raise ValueError("m_splits count {} != num_gemms {}".format(len(m_splits), num_gemms))

    inp, dy = values[0], values[1]
    weights = values[2 : 2 + num_gemms]
    biases = values[2 + num_gemms : 2 + 2 * num_gemms]
    if len(weights) != num_gemms or len(biases) != num_gemms:
        raise ValueError("Expected {n} weights and {n} biases".format(n=num_gemms))
    if sum(m_splits) != inp.shape[0]:
        raise ValueError("m_splits sum {} != input rows {}".format(sum(m_splits), inp.shape[0]))

    import transformer_engine.pytorch as te

    inp = inp.detach().clone().requires_grad_(True)
    in_features = inp.shape[-1]
    out_features = weights[0].shape[0]
    module = te.GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=bias_enabled,
        params_dtype=inp.dtype,
        device=device,
    )
    with torch.no_grad():
        for i in range(num_gemms):
            getattr(module, "weight{}".format(i)).copy_(weights[i])
            if bias_enabled:
                getattr(module, "bias{}".format(i)).copy_(biases[i])

    module.zero_grad(set_to_none=True)
    output = module(inp, m_splits)
    loss = (output * dy).sum()
    loss.backward()

    weight_names = ["weight{}".format(i) for i in range(num_gemms)]
    bias_names = ["bias{}".format(i) for i in range(num_gemms)]
    contract = _make_contract(module, output, m_splits, bias_enabled)
    slots = [
        output,
        inp.grad,
        _concatenate_grads(module, weight_names, output),
        _concatenate_grads(module, bias_names, output) if bias_enabled else _empty_like(output),
        _encode_contract(contract, output.device),
    ]
    if len(slots) != len(RESULT_SCHEMA):
        raise RuntimeError(
            "result arity {} drifted from RESULT_SCHEMA {}".format(len(slots), len(RESULT_SCHEMA))
        )
    if exact_baselines:
        _apply_exact_baselines(
            slots=slots,
            inp=inp,
            dy=dy,
            m_splits=m_splits,
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            bias_enabled=bias_enabled,
            device=device,
            raw_slots_out=raw_slots_out,
        )
    return tuple(slots)


def _apply_exact_baselines(
    slots: List[torch.Tensor],
    inp: torch.Tensor,
    dy: torch.Tensor,
    m_splits: List[int],
    num_gemms: int,
    in_features: int,
    out_features: int,
    bias_enabled: bool,
    device: str,
    raw_slots_out: Optional[Dict[int, torch.Tensor]] = None,
) -> None:
    """Overwrite the wgrad/bgrad slots with exact fp64 references.

    Only reachable from the golden side (``_golden_side_guard``). The operands
    are the pristine case inputs (``inp`` / ``dy``), so — unlike the
    LayerNormLinear wgrad baseline — there is no derived-operand rounding to
    mirror: the reference is the unique correct answer, rounded once into the
    slot dtype.
    """
    _golden_side_guard(device, {"forward_output": slots[0], "grad_output": dy})
    _emit_baseline_banner()

    # Capture before substituting: the collector records how far the device's own
    # output sits from the baseline that replaced it.
    if raw_slots_out is not None:
        raw_slots_out[_WGRAD_SLOT] = slots[_WGRAD_SLOT]
        raw_slots_out[_BGRAD_SLOT] = slots[_BGRAD_SLOT]

    x2d = inp.detach().reshape(-1, in_features)
    dy2d = dy.detach().reshape(-1, out_features)

    # Fail closed on a silently short slot: `_concatenate_grads` skips a
    # parameter whose grad is None, and writing a full-length golden anyway
    # would persist a shape-mismatched artifact (the same rule as the linear
    # guard's `row_spans_from_grad_parameters`).
    if x2d.shape[0] != sum(m_splits):
        raise RuntimeError(
            "m_splits sum {} does not tile the {} input rows".format(sum(m_splits), x2d.shape[0])
        )

    slots[_WGRAD_SLOT] = to_slot(
        grouped_wgrad_reference(x2d, dy2d, m_splits),
        slots[_WGRAD_SLOT],
        RESULT_SCHEMA[_WGRAD_SLOT],
    )

    if bias_enabled and slots[_BGRAD_SLOT].numel():
        slots[_BGRAD_SLOT] = to_slot(
            grouped_bgrad_reference(dy2d, m_splits),
            slots[_BGRAD_SLOT],
            RESULT_SCHEMA[_BGRAD_SLOT],
        )
    # bias off: the slot is empty by construction and an empty slot makes no
    # numeric claim to replace.


@register("function_grouped_linear")
class FunctionGroupedLinear(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this module ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            # The device under test: raw outputs, never the synthesised baseline.
            # See _golden_side_guard for why this must not change.
            result = run_grouped_linear_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'. This side *is* the golden, so it hands
            # back exact references for the slots whose NVTE kernels are less
            # accurate than the device being judged (weight grads, and bias
            # grads when bias is on).
            result = run_grouped_linear_case(input_data, "cuda", exact_baselines=True)
        else:
            raise RuntimeError(
                "GroupedLinear template has no CPU reference; only 'npu' execution "
                "is supported. NVTE consistency comes from the accuracy_load golden "
                "node, not from a local torch implementation. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
