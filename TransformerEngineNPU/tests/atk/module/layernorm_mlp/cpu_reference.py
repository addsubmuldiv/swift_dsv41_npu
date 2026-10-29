# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Pure-torch fp32 CPU reference for the LayerNormMLP ATK guard.

The TENPU module rejects CPU inputs (device guard in base.py), so the CPU
baseline is a standalone reference implementing the same math as
``te.LayerNormMLP``'s documented semantics: LN/RMS forward with fp32
statistics, fc1 GEMM + bias, gelu/silu activation, fc2 GEMM + bias, with
every intermediate computed in fp32 and the forward output rounded once to
the input dtype (matching TENPU's cast-back). The backward runs through
autograd on the fp32 chain; parameter/input grads are rounded once to the
input dtype on return.

Role: golden-side baseline option for the offline comparison (the same
positioning as linear's exact CPU column-sum baseline, REPORT atk 3#17).
The NVTE epilogue/kernel outputs are the alternative baseline; this module
exists so either side can be chosen per experiment without touching the
adapter.

Note: this is a *reference*, not the no-CPU-reference template's "local
smoke" - the adapter still refuses to run it on the device side (the
``cpu`` backend stays unsupported). It is consumed by
``collect_nvte_golden.py --baseline cpu-fp32`` and by
``diagnose_cpu_baseline.py``.
"""

from typing import Optional, Tuple

import torch


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)


_ACTIVATIONS = {"gelu": _gelu, "silu": _silu}


def cpu_reference_layernorm_mlp(
    x: torch.Tensor,
    dy: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: Optional[torch.Tensor],
    fc1_w: torch.Tensor,
    fc1_b: Optional[torch.Tensor],
    fc2_w: torch.Tensor,
    fc2_b: Optional[torch.Tensor],
    eps: float,
    normalization: str,
    activation: str,
    bias_enabled: bool,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, ...]:
    """fp32 CPU reference forward+backward; returns the adapter's 8 tensor slots.

    Input roles match ``run_layernorm_mlp_case``: x/dy [.., hidden], ln_w/ln_b
    [hidden], fc1_w [ffn, hidden], fc1_b [ffn], fc2_w [hidden, ffn], fc2_b
    [hidden]. All math runs in fp32 on CPU regardless of the input dtype; the
    forward output and each gradient are rounded once to x.dtype at the end.
    """
    dtype = x.dtype
    act = _ACTIVATIONS[activation]

    x_f = x.detach().clone().float().requires_grad_(True)
    ln_w_f = ln_w.detach().float().clone().requires_grad_(True)
    ln_b_f = (
        ln_b.detach().float().clone().requires_grad_(True)
        if (normalization != "RMSNorm" and ln_b is not None)
        else None
    )
    fc1_w_f = fc1_w.detach().float().clone().requires_grad_(True)
    fc1_b_f = (
        fc1_b.detach().float().clone().requires_grad_(True)
        if (bias_enabled and fc1_b is not None)
        else None
    )
    fc2_w_f = fc2_w.detach().float().clone().requires_grad_(True)
    fc2_b_f = (
        fc2_b.detach().float().clone().requires_grad_(True)
        if (bias_enabled and fc2_b is not None)
        else None
    )

    gamma = ln_w_f + (1.0 if zero_centered_gamma else 0.0)
    if normalization == "LayerNorm":
        ln_out = torch.nn.functional.layer_norm(x_f, (x_f.shape[-1],), gamma, ln_b_f, eps)
    elif normalization == "RMSNorm":
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        ln_out = x_f * torch.rsqrt(var + eps) * gamma
    else:
        raise ValueError("Unknown normalization: {}".format(normalization))

    h = ln_out @ fc1_w_f.t()
    if fc1_b_f is not None:
        h = h + fc1_b_f
    a = act(h)
    out = a @ fc2_w_f.t()
    if fc2_b_f is not None:
        out = out + fc2_b_f

    loss = (out.to(dtype).float() * dy.detach().float()).sum()
    loss.backward()

    def _empty():
        return torch.empty((0,), dtype=dtype)

    def _flat(grad):
        return grad.reshape(-1).to(dtype)

    return (
        out.detach().to(dtype),
        x_f.grad.to(dtype),
        _flat(ln_w_f.grad),
        _flat(ln_b_f.grad) if ln_b_f is not None else _empty(),
        _flat(fc1_w_f.grad),
        _flat(fc1_b_f.grad) if fc1_b_f is not None else _empty(),
        _flat(fc2_w_f.grad),
        _flat(fc2_b_f.grad) if fc2_b_f is not None else _empty(),
    )


def make_contract_from_reference(
    slots,
    normalization: str,
    activation: str,
    zero_centered_gamma: bool,
    bias_enabled: bool,
) -> dict:
    """Contract fields reproducible from the reference outputs alone.

    Mirrors the adapter's ``_make_contract`` fields that do not require the
    module object: shapes/dtypes are derivable from the reference slots, and
    the parameter-name/state-dict fields are reconstructed from the fixed
    LayerNormMLP parameter layout so the contract compares byte-for-byte
    against the device side.
    """
    forward_output = slots[0]
    # Parameter layout of te.LayerNormMLP (module.named_parameters() order):
    # layer_norm_weight, [layer_norm_bias], fc1_weight, [fc1_bias], fc2_weight,
    # [fc2_bias]; the grad slots are flattened by the adapter, so the shapes
    # are reconstructed from the weight matrices instead of the flat slots.
    ln_shape = [forward_output.shape[-1]]
    fc1_shape = [slots[4].numel() // ln_shape[0], ln_shape[0]]
    fc2_shape = [ln_shape[0], slots[6].numel() // ln_shape[0]]
    names = ["layer_norm_weight"]
    shapes = {"layer_norm_weight": ln_shape}
    if normalization != "RMSNorm":
        names.append("layer_norm_bias")
        shapes["layer_norm_bias"] = ln_shape
    names.append("fc1_weight")
    shapes["fc1_weight"] = fc1_shape
    if bias_enabled:
        names.append("fc1_bias")
        shapes["fc1_bias"] = [fc1_shape[0]]
    names.append("fc2_weight")
    shapes["fc2_weight"] = fc2_shape
    if bias_enabled:
        names.append("fc2_bias")
        shapes["fc2_bias"] = [fc2_shape[0]]
    return {
        "output_shape": list(forward_output.shape),
        "output_dtype": str(forward_output.dtype),
        "normalization": normalization,
        "activation": activation,
        "zero_centered_gamma": bool(zero_centered_gamma),
        "bias_enabled": bool(bias_enabled),
        "named_parameters": names,
        "parameter_shapes": shapes,
        "state_dict_keys": list(shapes.keys()) + ["_extra_state"],
    }


def cpu_reference_layernorm_mlp_bf16_chain(
    x: torch.Tensor,
    dy: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: Optional[torch.Tensor],
    fc1_w: torch.Tensor,
    fc1_b: Optional[torch.Tensor],
    fc2_w: torch.Tensor,
    fc2_b: Optional[torch.Tensor],
    eps: float,
    normalization: str,
    activation: str,
    bias_enabled: bool,
    zero_centered_gamma: bool,
) -> Tuple[torch.Tensor, ...]:
    """bf16-chain CPU reference: every intermediate rounded to x.dtype.

    Complements ``cpu_reference_layernorm_mlp`` (the exact fp32 reference):
    this variant mirrors the quantization schedule of a real bf16 kernel
    chain - the LN output, fc1 output, activation output and fc2 output are
    each rounded to x.dtype, with fp32 accumulation only inside each
    matmul/layernorm (CPU torch semantics). Against this baseline the
    standard 0.0156 gate passes on the real tier (measured 0.99+ forward /
    0.97-0.99+ dx on [4096,512], [2048,1024], [1024,2048]; the exact fp32
    reference fails at 0.85-0.92 because the fp32-vs-bf16 chain difference
    itself is ~1% on ~10% of elements - the chain-depth physics of REPORT
    atk 3#15, not a TENPU defect).

    Use ``--baseline cpu-bf16`` in the collector to store this variant.
    """
    dtype = x.dtype
    act = _ACTIVATIONS[activation]

    x_c = x.detach().clone().requires_grad_(True)
    ln_w_c = ln_w.detach().clone().requires_grad_(True)
    gamma = ln_w_c + (
        torch.tensor(1.0, dtype=ln_w.dtype)
        if zero_centered_gamma
        else torch.tensor(0.0, dtype=ln_w.dtype)
    )
    ln_b_c = (
        ln_b.detach().clone().requires_grad_(True)
        if (normalization != "RMSNorm" and ln_b is not None)
        else None
    )
    fc1_w_c = fc1_w.detach().clone().requires_grad_(True)
    fc1_b_c = (
        fc1_b.detach().clone().requires_grad_(True)
        if (bias_enabled and fc1_b is not None)
        else None
    )
    fc2_w_c = fc2_w.detach().clone().requires_grad_(True)
    fc2_b_c = (
        fc2_b.detach().clone().requires_grad_(True)
        if (bias_enabled and fc2_b is not None)
        else None
    )

    if normalization == "LayerNorm":
        ln_out = torch.nn.functional.layer_norm(x_c, (x_c.shape[-1],), gamma, ln_b_c, eps).to(dtype)
    elif normalization == "RMSNorm":
        rstd = torch.rsqrt(x_c.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        ln_out = (x_c.float() * rstd * gamma.float()).to(dtype)
    else:
        raise ValueError("Unknown normalization: {}".format(normalization))

    h = ln_out @ fc1_w_c.t()
    if fc1_b_c is not None:
        h = h + fc1_b_c
    h = h.to(dtype)
    a = act(h).to(dtype)
    out = a @ fc2_w_c.t()
    if fc2_b_c is not None:
        out = out + fc2_b_c
    out = out.to(dtype)

    loss = (out.float() * dy.detach().float()).sum()
    loss.backward()

    def _empty():
        return torch.empty((0,), dtype=dtype)

    def _flat(grad):
        return grad.reshape(-1).to(dtype)

    return (
        out.detach().to(dtype),
        x_c.grad.to(dtype),
        _flat(ln_w_c.grad),
        _flat(ln_b_c.grad) if ln_b_c is not None and ln_b_c.grad is not None else _empty(),
        _flat(fc1_w_c.grad),
        _flat(fc1_b_c.grad) if fc1_b_c is not None and fc1_b_c.grad is not None else _empty(),
        _flat(fc2_w_c.grad),
        _flat(fc2_b_c.grad) if fc2_b_c is not None and fc2_b_c.grad is not None else _empty(),
    )
