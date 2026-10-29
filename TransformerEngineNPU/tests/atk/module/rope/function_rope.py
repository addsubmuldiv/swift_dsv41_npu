# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine
``apply_rotary_pos_emb`` and ``apply_fused_qkv_rotary_pos_emb``.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation. NVTE consistency comes from
the offline golden loop (``collect_nvte_golden.py`` + an ATK ``accuracy_load``
node), never from a local torch reference.

Covers ``sbhd`` / ``bshd`` / ``thd`` tensor formats, ``interleaved`` and
``fused``, with the serialized contract recording the format and freqs shape.
The fused-QKV variant covers ``sbhd`` / ``bshd`` with MHA/GQA splits
(``qkv_split_arg_list``), returning the rotated q, k and pass-through v.
``start_positions`` / context-parallel offsets are follow-ups.

Both case shapes dispatch on the first string attr: ``rope`` (8 inputs) vs
``rope_fused_qkv`` (8 inputs). The fused-QKV outputs carry their own schema
slots (see RESULT_SCHEMA_FUSED_QKV) so golden comparison stays per-output.
"""

import json
from typing import Dict, Tuple

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

register_module_comparators("rope")

RESULT_SCHEMA = [
    "forward_output",
    "input_grad",
    "api_contract_json_utf8",
]

RESULT_SCHEMA_FUSED_QKV = [
    "q_output",
    "k_output",
    "v_output_passthrough",
    "qkv_input_grad",
    "api_contract_json_utf8",
]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _make_contract(output, t, freqs, tensor_format, interleaved, fused, cu_seqlens) -> Dict:
    return {
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "tensor_format": tensor_format,
        "interleaved": bool(interleaved),
        "fused": bool(fused),
        "input_shape": list(t.shape),
        "freqs_shape": list(freqs.shape),
        "freqs_dtype": str(freqs.dtype),
        "cu_seqlens": list(cu_seqlens) if cu_seqlens is not None else None,
    }


def _make_fused_qkv_contract(
    q, k, v, qkv, q_freqs, k_freqs, tensor_format, interleaved, split
) -> Dict:
    return {
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "output_dtype": str(q.dtype),
        "tensor_format": tensor_format,
        "interleaved": bool(interleaved),
        "qkv_shape": list(qkv.shape),
        "q_freqs_shape": list(q_freqs.shape),
        "k_freqs_shape": list(k_freqs.shape),
        "freqs_dtype": str(q_freqs.dtype),
        "qkv_split_arg_list": list(split),
    }


def _split_dy(dy, q, k, v, split):
    """Build per-output dy tensors matching the shapes q/k/v actually returned.

    dy has the fused [s, b, ng, q_w + k_w + v_w] layout (same as the qkv
    input): each group's channels are [q_w | k_w | v_w] with q_w = hpg * hn.
    NVTE returns q head-expanded to [s, b, np, hn] and k/v as [s, b, ng, hn];
    the q slice per group is reshaped to its heads, then groups concatenate on
    the head axis. TENPU now matches that layout after the 3#19 fix; whatever
    shape an implementation returns, the seeds are matched to it so the case
    runs to completion and layout divergences surface via the contract.
    """
    q_w, k_w, v_w = split
    ng = dy.shape[-2]

    def _match(target, width, head_expand):
        parts = []
        for g in range(ng):
            # keep the 4D group axis so concatenation happens on the head dim
            chunk = dy[..., g : g + 1, :width]
            if head_expand and target.shape[-1] != width:
                heads = width // target.shape[-1]
                chunk = chunk.reshape(*dy.shape[:-2], heads, target.shape[-1])
            parts.append(chunk)
        if ng > 1:
            fused = torch.cat(parts, dim=-2)
        elif head_expand and target.shape[-1] != width:
            # single group already reshaped to head view — no axis to drop
            fused = parts[0]
        else:
            fused = parts[0].squeeze(-2)
            if target.dim() == 4 and fused.dim() == 3:
                # restore the singleton head axis the target still carries
                fused = fused.unsqueeze(-2)
        heads = target.shape[-2]
        if fused.shape[-2] != heads and heads % fused.shape[-2] == 0:
            fused = fused.repeat_interleave(heads // fused.shape[-2], dim=-2)
        return fused

    # q is head-expanded [s, b, np, hn]; k/v stay [s, b, ng, hn]
    dy_q = _match(q, q_w, head_expand=True)
    dy_k = _match(k, k_w, head_expand=False)
    dy_v = _match(v, v_w, head_expand=False)
    return dy_q, dy_k, dy_v


def _duplicated_angle_freqs(freqs: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """Rebuild freqs in the duplicated-angle format used by every rotary
    embedding constructor (cat(freqs, freqs) non-interleaved; stacked pairs
    interleaved) from arbitrary canonical angles.

    The first half of the last dim is kept and mirrored; the transform is
    deterministic and identical on both sides of the golden loop.
    """
    w = freqs.shape[-1]
    half = freqs[..., : w // 2].contiguous()
    if interleaved:
        out = torch.stack((half, half), dim=-1).flatten(-2)
        return out.reshape(freqs.shape)
    return torch.cat((half, half), dim=-1)


def run_rope_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward RoPE case using canonical ATK inputs.

    Input layout: ``t, freqs, dy, variant(str 'rope'), tensor_format(str),
    interleaved(bool), fused(bool), cu_seqlens(str, 'none' unless thd)``.
    """
    values = list(input_data.args)
    if len(values) != 8:
        raise ValueError("RoPE ATK case expects 8 positional inputs, got {}".format(len(values)))
    t, freqs, dy, variant, tensor_format, interleaved, fused, cu_str = values
    if str(variant) != "rope":
        raise ValueError("run_rope_case received variant {}".format(variant))
    tensor_format = str(tensor_format)
    interleaved = bool(interleaved)
    fused = bool(fused)

    # Re-derive the freqs in the duplicated-angle (production) format from the
    # canonical random input: keep the first half of the random angles and
    # mirror them into the second half. Megatron/TE rotary constructors always
    # feed cat(freqs, freqs) (non-interleaved) or stacked pairs (interleaved),
    # so this exercises the real training layout — including the fused-kernel
    # backward path (identical transform on both the NVTE golden side and here,
    # so the golden loop stays deterministic).
    freqs = _duplicated_angle_freqs(freqs, interleaved)

    # Canonical NVTE import path: NVTE 2.17 only exposes apply_rotary_pos_emb from
    # the rope submodule (attention/__init__.py does not re-export it), so importing
    # from the package fails on the NVTE side of the golden loop. TENPU exports both.
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

    t = t.detach().clone().requires_grad_(True)
    kwargs = dict(
        tensor_format=tensor_format,
        interleaved=interleaved,
        fused=fused,
    )
    cu_seqlens = None
    if tensor_format == "thd":
        cu_seqlens = [int(x) for x in str(cu_str).split(",")]
        kwargs["cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    output = apply_rotary_pos_emb(t, freqs, **kwargs)
    loss = (output * dy).sum()
    loss.backward()

    contract = _make_contract(output, t, freqs, tensor_format, interleaved, fused, cu_seqlens)
    return (
        output,
        t.grad,
        _encode_contract(contract, output.device),
    )


def run_rope_fused_qkv_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward fused-QKV RoPE case using canonical ATK inputs.

    Input layout (Megatron/NVTE convention, REPORT atk 3#19): ``qkv, q_freqs,
    k_freqs, dy, variant(str 'rope_fused_qkv'), tensor_format(str),
    interleaved(bool), qkv_split_arg_list(str, 3 comma ints)``.

    ``qkv`` is [s, b, ng, (np/ng + 2) * hn] — ng KV groups, each carrying the
    group's q channels followed by one k and one v head (Megatron
    attention.py:1591). ``qkv_split_arg_list`` = [np/ng * hn, hn, hn]. NVTE
    returns q reshaped to [s, b, np, hn] (apply_rope.cpp:123-125) and k/v as
    [s, b, ng, hn]; the backward is seeded with per-output dy slices taken
    from the fused dy tensor: q gets dy[..., :split[0]] repeated per group
    semantics — see _split_dy.
    """
    values = list(input_data.args)
    if len(values) != 8:
        raise ValueError(
            "Fused-QKV RoPE ATK case expects 8 positional inputs, got {}".format(len(values))
        )
    qkv, q_freqs, k_freqs, dy, variant, tensor_format, interleaved, split_str = values
    if str(variant) != "rope_fused_qkv":
        raise ValueError("run_rope_fused_qkv_case received variant {}".format(variant))
    tensor_format = str(tensor_format)
    interleaved = bool(interleaved)
    split = [int(x) for x in str(split_str).split(",")]
    if len(split) != 3:
        raise ValueError("qkv_split_arg_list must have 3 elements, got {}".format(split))

    # Duplicated-angle (production) freqs layout, same rationale as the plain
    # rope case: exercises the fused-kernel backward path both sides.
    q_freqs = _duplicated_angle_freqs(q_freqs, interleaved)
    k_freqs = _duplicated_angle_freqs(k_freqs, interleaved)

    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    qkv = qkv.detach().clone().requires_grad_(True)
    q, k, v = apply_fused_qkv_rotary_pos_emb(
        qkv,
        q_freqs,
        k_freqs,
        split,
        tensor_format=tensor_format,
        interleaved=interleaved,
    )
    # Seed the backward against each output's own shape. TENPU currently
    # splits the last dim whole instead of per group (REPORT atk 3#19), so
    # _split_dy derives same-shape seeds from the returned outputs; the
    # layout divergence itself is caught by the contract's shape fields.
    dy_q, dy_k, dy_v = _split_dy(dy, q, k, v, split)
    loss = (q * dy_q).sum() + (k * dy_k).sum() + (v * dy_v).sum()
    loss.backward()

    contract = _make_fused_qkv_contract(
        q, k, v, qkv, q_freqs, k_freqs, tensor_format, interleaved, split
    )
    return (
        q,
        k,
        v,
        qkv.grad,
        _encode_contract(contract, q.device),
    )


@register("function_rope")
class FunctionRope(BaseApi):
    """ATK module adapter with a device-only implementation.

    Dispatches on the first string attr: ``rope`` runs
    ``apply_rotary_pos_emb`` (3 outputs), ``rope_fused_qkv`` runs
    ``apply_fused_qkv_rotary_pos_emb`` (5 outputs). Output arity differs per
    variant; the golden collector flattens whatever the adapter returns and
    the result_schema documents both.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device in ("npu", "gpu", "cuda"):
            # 'gpu' is the ATK backend label for online GPU-side execution
            # (ATK server); torch needs 'cuda'.
            device = "cuda" if self.device in ("gpu", "cuda") else self.device
            values = list(input_data.args)
            # The plain shape carries the variant attr at position 3 (t, freqs,
            # dy, variant, ...); the fused-QKV shape carries it at position 4
            # (qkv, q_freqs, k_freqs, dy, variant, ...).
            variant = ""
            for probe in (3, 4):
                if len(values) > probe and isinstance(values[probe], str):
                    variant = str(values[probe])
                    break
            if variant == "rope_fused_qkv":
                result = run_rope_fused_qkv_case(input_data, device)
            else:
                result = run_rope_case(input_data, device)
        else:
            raise RuntimeError(
                "RoPE template has no CPU reference; only 'npu' execution is "
                "supported. NVTE consistency comes from the accuracy_load golden "
                "node, not from a local torch implementation. Got backend: "
                "{}".format(self.device)
            )
        return result if with_output else None
