# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for Transformer Engine DotProductAttention.

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of
``transformer_engine.pytorch.DotProductAttention``. NVTE consistency comes from
the offline golden loop (``collect_nvte_golden.py`` + an ATK ``accuracy_load``
node), never from a local torch reference.

P0 scope: the standard self-attention path — MHA/GQA/MQA, no dropout, in both
qkv formats the TENPU API accepts: ``sbhd`` (4-D, mask ∈ {causal, no_mask})
and ``thd`` (3-D packed, mask ∈ {padding, padding_causal}, cu_seqlens derived
from the token count). ``core_attention_bias``, ``alibi_slopes`` and windowing
are follow-ups; ``bshd`` is not implementable here (see below).

Output layouts are NOT identical across backends: both flatten the trailing
(h, d) dims for ``sbhd`` (3-D [s, b, h*d]), but for ``thd`` NVTE returns the
flattened 2-D [t, h*d] ("...hd -> ...(hd)" in its backends) while TENPU keeps
3-D [t, h, d] (TND goes straight to npu_fusion_attention). This adapter
canonicalizes the GPU-side thd output to the 3-D layout so both backends
return — and get compared on — identical shapes; see the thd branch in
``run_dot_product_attention_case``.

``DotProductAttention`` has no learnable parameters, so the outputs are the
forward result, the query/key/value input gradients, and the serialized
contract.
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

register_module_comparators("dot_product_attention")

# Disable TF32 for fp32 matmuls on CUDA. NVTE's flash-attention and cuDNN
# backends only accept bf16/fp16/fp8 (backend dtype filters in
# attention/dot_product_attention/utils.py), so every fp32 case falls through
# to UnfusedDotProductAttention, whose scores/context are torch.bmm calls
# (backends.py) — and cuBLAS defaults to TF32 (~1e-3 relative error) on
# Ampere+ GPUs. That error straddles the fp32 rtol 9.77e-4 gate and shows up
# as data-dependent matched_ratio dips (the 2026-09-22 extra-run case 2:
# fp32 thd, 3-10 of 256 elements failing, same config passing on another
# draw). No-op on NPU (torch_npu does not go through cuBLAS).
torch.backends.cuda.matmul.allow_tf32 = False

RESULT_SCHEMA = [
    "forward_output",
    "query_grad",
    "key_grad",
    "value_grad",
    "api_contract_json_utf8",
]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def _make_contract(
    output,
    num_heads,
    kv_channels,
    gqa,
    mask_type,
    q,
    k,
    v,
    qkv_format,
    cu_seqlens,
) -> Dict:
    # The contract intentionally records only input-derived API parameters and
    # the output shape/dtype. `module.state_dict()` is *not* included: NVTE
    # registers a persistent `_extra_state` buffer (FP8 metadata) that TENPU's
    # DotProductAttention does not, so the key list is implementation-specific
    # and would make the serialized contract differ byte-wise between backends.
    return {
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "num_attention_heads": int(num_heads),
        "kv_channels": int(kv_channels),
        "num_gqa_groups": gqa,
        "attn_mask_type": mask_type,
        "qkv_format": qkv_format,
        "cu_seqlens": None if cu_seqlens is None else [int(x) for x in cu_seqlens],
        "query_shape": list(q.shape),
        "key_shape": list(k.shape),
        "value_shape": list(v.shape),
    }


# Number of packed sequences a thd case is split into, and the split rule.
# The case carries no cu_seqlens input (the adapter sees only tensors), so
# the layout is derived deterministically from the total token count. Three
# ragged sequences (T//3, T//3, T-2*(T//3)) follow the 2026-09-17 NPU probe
# (probe_thd2.py) rather than an equal 2-way split: variable-length packing
# is what thd exists for, and the remainder exercises the ragged tail of the
# padding mask.
_THD_NUM_SEQUENCES = 3


def _derive_cu_seqlens(total_tokens: int, device: torch.device) -> torch.Tensor:
    """Deterministically pack ``total_tokens`` into 3 ragged sequences.

    NVTE asserts cu_seqlens is 1-D, of shape ``[batch_size + 1]`` and
    ``torch.int32`` (``dot_product_attention.py:1380``), and infers
    ``max_seqlen_q/kv`` from it when they are not passed.
    """
    if total_tokens < _THD_NUM_SEQUENCES:
        raise ValueError(
            "thd case needs at least {n} tokens to form {n} sequences, got {got}".format(
                n=_THD_NUM_SEQUENCES, got=total_tokens
            )
        )
    chunk = total_tokens // _THD_NUM_SEQUENCES
    bounds = [chunk * i for i in range(_THD_NUM_SEQUENCES)] + [total_tokens]
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def run_dot_product_attention_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run one forward/backward DotProductAttention case using canonical ATK inputs.

    Input layout (9 positional): ``query, key, value, dy, num_heads(int),
    kv_channels(int), num_gqa_groups(int, 0 means None), attn_mask_type(str),
    qkv_format(str, 'sbhd' or 'thd')``.

    For thd: query/key/value are 3-D [total_tokens, heads, head_dim], dy
    matches the 3-D output, and the module is called with cu_seqlens derived
    deterministically from the token count (3 ragged sequences, see
    ``_derive_cu_seqlens``). NVTE's thd output arrives flattened as 2-D
    [total_tokens, heads * head_dim]; the GPU branch feeds the loss in that
    native form and canonicalizes the returned output back to the 3-D layout
    (TENPU's native form, and the case's dy shape). For sbhd: the 4-D path,
    unchanged except that the format is now passed explicitly instead of
    relying on the module default.
    """
    values = list(input_data.args)
    if len(values) != 9:
        raise ValueError(
            "DotProductAttention ATK case expects 9 positional inputs, got {}".format(len(values))
        )
    q, k, v, dy, num_heads, kv_channels, gqa, mask_type, qkv_format = values
    num_heads = int(num_heads)
    kv_channels = int(kv_channels)
    gqa = None if int(gqa) == 0 else int(gqa)
    mask_type = str(mask_type)
    qkv_format = str(qkv_format)
    if qkv_format not in ("sbhd", "thd"):
        # Not bshd: it is documented in both backends but TENPU's
        # DotProductAttention asserts qkv_format ∈ {sbhd, thd}
        # (dot_product_attention.py:645) — registered as ATK_REPORT §3#21.
        raise ValueError("Unsupported qkv_format '{}'; expected 'sbhd' or 'thd'".format(qkv_format))

    from transformer_engine.pytorch import DotProductAttention

    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    # TENPU accepts an explicit `device` kwarg, but NVTE's DotProductAttention
    # does not (it owns no parameters and creates its state on the default
    # device). Pass it only on NPU so the same adapter drives both sides.
    init_kwargs = dict(
        num_attention_heads=num_heads,
        kv_channels=kv_channels,
        num_gqa_groups=gqa,
        attention_dropout=0.0,
        attn_mask_type=mask_type,
    )
    if device == "npu":
        init_kwargs["device"] = device
    module = DotProductAttention(**init_kwargs)

    if qkv_format == "thd":
        cu_seqlens = _derive_cu_seqlens(q.shape[0], q.device)
        output = module(
            q,
            k,
            v,
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
        )
        if device != "npu":
            # NVTE flattens the trailing (h, d) dims of the DPA output for
            # every qkv_format ("...hd -> ...(hd)"; the thd branch reads
            # "thd -> t(hd)", backends.py), so its thd output is 2-D
            # [T, H*D]. TENPU keeps the 3-D [T, H, D] layout (TND goes
            # straight to npu_fusion_attention). The case's dy carries the
            # semantic 3-D shape: feed the loss in the backend's native
            # form, then canonicalize the output back to 3-D so both
            # sides return identical shapes for comparison.
            t_dim, h_dim, d_dim = q.shape
            loss = (output * dy.reshape(t_dim, h_dim * d_dim)).sum()
            loss.backward()
            output = output.reshape(t_dim, h_dim, d_dim)
        else:
            loss = (output * dy).sum()
            loss.backward()
    else:
        cu_seqlens = None
        # `sbhd` is the module default, but pass it explicitly: slot 8 is
        # what selects the branch, and both backends must be driven by the
        # case rather than by their own defaults.
        output = module(q, k, v, qkv_format="sbhd")

        loss = (output * dy).sum()
        loss.backward()

    contract = _make_contract(
        output, num_heads, kv_channels, gqa, mask_type, q, k, v, qkv_format, cu_seqlens
    )
    return (
        output,
        q.grad,
        k.grad,
        v.grad,
        _encode_contract(contract, output.device),
    )


@register("function_dot_product_attention")
class FunctionDotProductAttention(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts
    and never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_dot_product_attention_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_dot_product_attention_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "DotProductAttention template has no CPU reference; only 'npu' "
                "execution is supported. NVTE consistency comes from the "
                "accuracy_load golden node, not from a local torch implementation. "
                "Got backend: {}".format(self.device)
            )
        return result if with_output else None
