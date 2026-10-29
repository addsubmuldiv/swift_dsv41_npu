# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine Linear.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of
``transformer_engine.pytorch.Linear``. NVTE consistency comes from the
offline golden loop (``collect_nvte_golden.py`` + an ATK ``accuracy_load``
node), never by a local torch reference. (The first version of this module
carried a CPU smoke reference; it was removed when the no-CPU-reference
template was adopted — the golden loop subsumes it.)

Covered knobs: bias on/off, ``return_bias`` / ``consume_returned_bias``,
``parameters_split`` (none / tuple2 / dict2), 2D/3D/4D inputs; forward
output, returned bias, input grad, concatenated weight/bias grads, and a
serialized API contract.

**Exact baselines (golden side only).** NVTE's ``weight_grad`` and ``bias_grad``
both come out of one cuBLASLt call (``"bias": bias, "grad": True``
-> ``CUBLASLT_EPILOGUE_BGRADB``) whose in-GEMM accumulation is less accurate than
the NPU's fp32 accumulation on large-token shapes: judged against exact fp64, the
NPU scores 1.0000 on both slots while NVTE lands at 0.8802-0.9689 / 0.9033-1.0000.
The golden therefore carries exact fp64 references for those two grad slots and
the comparison becomes "device vs exact math" -- a baseline choice, not a
tolerance relaxation (REPORT atk 3#17 set the same precedent for ``db``). Details
and the measurements: ``../_exact_reference.py``.

``forward_output`` joined the same treatment last: it was the only slot still red
on the generalized 150-case ``all_linear`` set (fp16 ``id73`` / ``id137``), and the
per-slot judge puts that deviation on the golden side too -- NPU 1.0000 against
exact fp64 versus NVTE 0.9616 / 0.9851. The switch is therefore named
``exact_baselines`` (it is no longer grads-only), and the forward reference has to
respect ``apply_bias = bias_enabled and not return_bias``; see
``_apply_exact_baselines`` and ``_exact_reference.forward_reference``.
"""

import json
from collections import OrderedDict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

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
from _exact_reference import (  # noqa: E402
    bias_grad_reference,
    chunked_wgrad,
    concatenated_bias,
    concatenated_weight,
    flatten_row_blocks,
    forward_reference,
    row_spans_from_grad_parameters,
    to_slot,
)

# Disable TF32 for fp32 GEMM on CUDA (ATK server online mode): cuBLAS
# defaults to TF32 (10-bit mantissa, ~1e-3 relative error) on Ampere+
# GPUs — 10000x worse than true fp32. No-op on NPU (torch_npu does not
# go through cuBLAS). Must be set before any te.Linear forward.
torch.backends.cuda.matmul.allow_tf32 = False

register_module_comparators("linear")

RESULT_SCHEMA = [
    "forward_output",
    "returned_bias_or_empty",
    "input_grad",
    "weight_grad_concatenated",
    "bias_grad_concatenated_or_empty",
    "api_contract_json_utf8",
]

# Slots whose golden is synthesised as an exact fp64 reference instead of the
# NVTE kernel output, on the golden side only. See `_apply_exact_baselines` and
# `../_exact_reference.py` for the measurements behind this.
_FORWARD_SLOT = RESULT_SCHEMA.index("forward_output")
_WGRAD_SLOT = RESULT_SCHEMA.index("weight_grad_concatenated")
_BGRAD_SLOT = RESULT_SCHEMA.index("bias_grad_concatenated_or_empty")


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
    """One line per process, so the online golden path is at least greppable.

    The offline path records the substitution in the collection manifest; the
    online path (ATK server) has no manifest, and this banner in the server log is
    what stands in for it.
    """
    print(
        "[linear] exact fp64 baseline ACTIVE (slots: forward {}, wgrad {}, bgrad {})".format(
            _FORWARD_SLOT, _WGRAD_SLOT, _BGRAD_SLOT
        )
    )


def _parameters_split(kind: str, out_features: int):
    if kind == "none":
        return None
    left = out_features // 2
    right = out_features - left
    if kind == "tuple2":
        if left != right:
            raise ValueError("tuple2 requires an even out_features value")
        return ("proj_a", "proj_b")
    if kind == "dict2":
        return OrderedDict((("gate", left), ("up", right)))
    raise ValueError("Unsupported parameters_split kind: {}".format(kind))


def _empty_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), dtype=reference.dtype, device=reference.device)


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _concatenate_grads(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    suffix: str,
    empty_reference: torch.Tensor,
) -> torch.Tensor:
    grads = [
        parameter.grad.reshape(-1)
        for name, parameter in named_parameters
        if name.endswith(suffix) and parameter.grad is not None
    ]
    if not grads:
        return _empty_like(empty_reference)
    return torch.cat(grads)


def _make_contract(
    raw_output,
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    state_dict: Dict[str, torch.Tensor],
) -> Dict:
    parameters = list(named_parameters)
    if isinstance(raw_output, tuple):
        forward_output = raw_output[0]
        returned_bias = raw_output[1]
        output_kind = "tuple"
    else:
        forward_output = raw_output
        returned_bias = None
        output_kind = "tensor"
    return {
        "output_kind": output_kind,
        "output_arity": len(raw_output) if isinstance(raw_output, tuple) else 1,
        "forward_output_shape": list(forward_output.shape),
        "forward_output_dtype": str(forward_output.dtype),
        "returned_bias_kind": "tensor" if isinstance(returned_bias, torch.Tensor) else "none",
        "returned_bias_shape": list(returned_bias.shape)
        if isinstance(returned_bias, torch.Tensor)
        else None,
        "named_parameters": [name for name, _ in parameters],
        "parameter_shapes": {name: list(parameter.shape) for name, parameter in parameters},
        "state_dict_keys": list(state_dict.keys()),
    }


def run_linear_case(
    input_data,
    device: str,
    *,
    exact_baselines: bool = False,
    raw_slots_out: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward Linear case using canonical ATK inputs.

    Only the device implementation is provided; numerical consistency with NVTE
    is established by the offline golden loop, never by a local torch reference.

    ``exact_baselines`` is **golden-side only** and keyword-only on purpose: it
    replaces ``forward_output``, ``weight_grad`` and ``bias_grad`` with exact fp64
    references (``_apply_exact_baselines``) when this run is producing the golden.
    Any judge that needs to see a device's *raw* output -- in particular
    ``diagnose_slot_vs_fp64.py``, the arbiter for "which side deviates" -- must
    leave it off, or it would compare a reference against itself. The device
    string alone cannot tell the two roles apart (both say ``"cuda"``), which is
    why this is an explicit switch and not a device rule.

    ``raw_slots_out``, when given, receives the pre-substitution device slots
    keyed by output index. The collector uses it to record how far NVTE's own
    output sits from the baseline it was replaced by; it changes no output.
    """
    if len(input_data.args) != 9:
        raise ValueError(
            "Linear ATK case expects 9 positional inputs, got {}".format(len(input_data.args))
        )
    inp, weight, bias_value, grad_output, grad_returned_bias = list(input_data.args)[:5]
    bias_enabled, return_bias, consume_returned_bias, split_kind = [
        _coerce(value) for value in list(input_data.args)[5:8]
    ] + [str(list(input_data.args)[8])]

    import transformer_engine.pytorch as te

    inp = inp.detach().clone().requires_grad_(True)
    out_features, in_features = weight.shape
    module = te.Linear(
        in_features,
        out_features,
        bias=bias_enabled,
        return_bias=return_bias,
        params_dtype=inp.dtype,
        parameters_split=_parameters_split(split_kind, out_features),
        device=device,
    )

    with torch.no_grad():
        offset = 0
        for name in module.weight_names:
            parameter = getattr(module, name)
            rows = parameter.shape[0]
            parameter.copy_(weight[offset : offset + rows])
            offset += rows
        if bias_enabled:
            offset = 0
            for name in module.bias_names:
                parameter = getattr(module, name)
                rows = parameter.shape[0]
                parameter.copy_(bias_value[offset : offset + rows])
                offset += rows

    module.zero_grad(set_to_none=True)
    raw_output = module(inp)
    if isinstance(raw_output, tuple):
        output, returned_bias = raw_output
    else:
        output, returned_bias = raw_output, None

    loss = (output * grad_output).sum()
    if consume_returned_bias and isinstance(returned_bias, torch.Tensor) and returned_bias.numel():
        loss = loss + (returned_bias * grad_returned_bias).sum()
    loss.backward()

    named_parameters = list(module.named_parameters())
    contract = _make_contract(raw_output, named_parameters, module.state_dict())
    returned_bias_tensor = (
        returned_bias if isinstance(returned_bias, torch.Tensor) else _empty_like(output)
    )
    slots = [
        output,
        returned_bias_tensor,
        inp.grad,
        _concatenate_grads(named_parameters, "weight", output),
        _concatenate_grads(named_parameters, "bias", output),
        _encode_contract(contract, output.device),
    ]
    if len(slots) != len(RESULT_SCHEMA):
        raise RuntimeError(
            "result arity {} drifted from RESULT_SCHEMA {}".format(len(slots), len(RESULT_SCHEMA))
        )
    if exact_baselines:
        _apply_exact_baselines(
            slots=slots,
            named_parameters=named_parameters,
            inp=inp,
            grad_output=grad_output,
            grad_returned_bias=grad_returned_bias,
            out_features=out_features,
            in_features=in_features,
            bias_enabled=bias_enabled,
            return_bias=return_bias,
            consume_returned_bias=consume_returned_bias,
            device=device,
            raw_slots_out=raw_slots_out,
        )
    return tuple(slots)


def _apply_exact_baselines(
    slots: List[torch.Tensor],
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
    inp: torch.Tensor,
    grad_output: torch.Tensor,
    grad_returned_bias: torch.Tensor,
    out_features: int,
    in_features: int,
    bias_enabled: bool,
    return_bias: bool,
    consume_returned_bias: bool,
    device: str,
    raw_slots_out: Optional[Dict[int, torch.Tensor]] = None,
) -> None:
    """Overwrite the forward/wgrad/bias_grad slots with exact fp64 references.

    Only reachable from the golden side (``_golden_side_guard``). The flags are
    the ones already parsed by the caller -- never re-index ``input_data.args``,
    which is what let the collector's own copy of this policy drift into reading
    ``consume_returned_bias`` without ``return_bias``.
    """
    _golden_side_guard(device, {"forward_output": slots[_FORWARD_SLOT], "grad_output": grad_output})
    _emit_baseline_banner()

    # Capture before substituting: the collector records how far the device's own
    # output sits from the baseline that replaced it.
    if raw_slots_out is not None:
        raw_slots_out[_FORWARD_SLOT] = slots[_FORWARD_SLOT]
        raw_slots_out[_WGRAD_SLOT] = slots[_WGRAD_SLOT]
        raw_slots_out[_BGRAD_SLOT] = slots[_BGRAD_SLOT]

    x2d = inp.detach().reshape(-1, in_features)
    dy2d = grad_output.detach().reshape(-1, out_features)

    # forward_output: out = x @ W^T, with the bias only when the module applies it
    # (NVTE `apply_bias = bias and not return_bias`). Passing `bias_enabled` alone
    # would put a bias-sized error *into the golden* for every return_bias case --
    # all_linear id73 is one -- and blame the device for it.
    apply_bias = bias_enabled and not return_bias
    slots[_FORWARD_SLOT] = forward_reference(
        slots[_FORWARD_SLOT],
        x2d,
        concatenated_weight(named_parameters, out_features),
        concatenated_bias(named_parameters, out_features) if apply_bias else None,
        apply_bias,
    )

    weight_spans = row_spans_from_grad_parameters(named_parameters, "weight", out_features)
    wgrad = flatten_row_blocks(chunked_wgrad(dy2d, x2d), weight_spans)
    slots[_WGRAD_SLOT] = to_slot(wgrad, slots[_WGRAD_SLOT], RESULT_SCHEMA[_WGRAD_SLOT])

    bgrad = bias_grad_reference(
        dy2d, grad_returned_bias, bias_enabled, return_bias, consume_returned_bias
    )
    if bgrad is None:
        # Empty by construction; the device slot already carries that arity and
        # an empty slot makes no numeric claim to replace.
        return
    bias_spans = row_spans_from_grad_parameters(named_parameters, "bias", out_features)
    slots[_BGRAD_SLOT] = to_slot(
        flatten_row_blocks(bgrad, bias_spans), slots[_BGRAD_SLOT], RESULT_SCHEMA[_BGRAD_SLOT]
    )


def _coerce(value):
    """Coerce ATK attr payloads (may arrive as numpy bools) to Python bools."""
    return bool(value)


@register("function_te_linear")
class FunctionTELinear(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this module ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            # The device under test: raw outputs, never the synthesised baseline.
            # See _golden_side_guard for why this must not change.
            result = run_linear_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'. This side *is* the golden, so it hands
            # back exact references for the slots whose NVTE kernels are less
            # accurate than the device being judged (forward, wgrad, bias_grad).
            result = run_linear_case(input_data, "cuda", exact_baselines=True)
        else:
            raise RuntimeError(
                "Linear template has no CPU reference; only 'npu' execution is "
                "supported. NVTE consistency comes from the accuracy_load golden "
                "node, not from a local torch implementation. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
