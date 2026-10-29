# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
"""Torch-native FLA interface fallbacks for GatedDeltaNet.

When flash-linear-attention is not installed, this module provides
pure-torch implementations of the three FLA symbols that GDN imports:

* ``causal_conv1d`` → ``F.conv1d`` (+ per-sequence unpack for packed seqs)
* ``chunk_gated_delta_rule`` → Megatron-LM built-in ``torch_chunk_gated_delta_rule``
* ``l2norm`` → ``F.normalize(p=2)``

These are conditionally registered by ``MegatronBasicFeature`` via
``create_dummy`` on the FLA package paths, allowing Megatron's optional
FLA imports to succeed. A real FLA installation is left untouched. When
MindSpeed is present it replaces the entire
``GatedDeltaNet`` class via ``force_patch``; the MindSpeed subclass imports
its own Triton-accelerated operators directly from ``mindspeed.ops.triton``
and never reads the FLA paths.
"""

import torch
import torch.nn.functional as F


def _unpack_sequence_per_seq(x, cu_seqlens, dim):
    """Split a packed tensor into per-sequence chunks along *dim*."""
    chunks = []
    for i in range(cu_seqlens.shape[0] - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        idx = [slice(None)] * x.dim()
        idx[dim] = slice(start, end)
        chunks.append(x[tuple(idx)])
    return chunks


def _conv1d_impl(x, weight_3d, bias):
    """Single (non-packed) causal conv1d via ``F.conv1d``.

    Computes in float32 and **keeps the result in float32** so that the
    caller can apply the activation function before casting back, avoiding
    an unnecessary fp32→bf16→fp32 round-trip.
    """
    seq_len = x.shape[1]
    x_t = x.float().transpose(1, 2).contiguous()  # b,s,d → b,d,s (fp32)
    out = F.conv1d(
        input=x_t,
        weight=weight_3d.float(),
        bias=bias.float() if bias is not None else None,
        stride=1,
        padding=weight_3d.shape[-1] - 1,
        groups=weight_3d.shape[0],
    )
    out = out[..., :seq_len]  # trim causal padding
    return out.transpose(1, 2)  # b,d,s → b,s,d (keep fp32)


def _npu_causal_conv1d(
    x, weight, bias=None, activation=None, initial_state=None, output_final_state=False, cu_seqlens=None
):
    """Torch-native ``causal_conv1d`` with FLA-compatible signature.

    Dispatch:
    * ``cu_seqlens is None`` → single ``F.conv1d``
    * ``cu_seqlens is not None`` → per-sequence unpack → F.conv1d → repack

    Args:
        x: ``[b, s, d]`` (FLA / bshd layout).
        weight: ``[d, w]`` — will be unsqueezed to ``[d, 1, w]``.
        bias: optional ``[d]``.
        activation: ``"silu"`` / ``"swish"`` or ``None``.
        initial_state: ignored (no Triton kernel to consume it).
        output_final_state: always returns ``None``.
        cu_seqlens: ``[num_seqs + 1]`` cumulative lengths, or ``None``.

    Returns:
        ``(y, final_state)`` where *y* is ``[b, s, d]`` and *final_state*
        is always ``None``.
    """
    weight_3d = weight.unsqueeze(1)  # [d, w] → [d, 1, w]

    if cu_seqlens is None:
        out = _conv1d_impl(x, weight_3d, bias)
    else:
        chunks = _unpack_sequence_per_seq(x.float(), cu_seqlens, dim=1)
        if not chunks:
            # Degenerate pack: keep shapes with an empty fp32 tensor.
            out = torch.zeros_like(x, dtype=torch.float)
        else:
            out_chunks = [_conv1d_impl(c, weight_3d, bias) for c in chunks]
            out = torch.cat(out_chunks, dim=1)

    # Apply activation while still in fp32, then cast to original dtype
    # once at the very end.  This avoids an fp32→bf16→fp32 round-trip
    # between conv1d and the activation function.
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation for NPU causal_conv1d: {activation}")

    return out.to(x.dtype), None


def _npu_l2norm(x, dim=-1, eps=1e-6):
    """Torch-native L2 normalization.

    Computes in float32 and casts back to the original dtype.
    """
    return F.normalize(x.float(), p=2, dim=dim, eps=eps).to(x.dtype)


def _npu_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    scale=None,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    head_first=False,
):
    """Delegates to Megatron-LM's built-in ``torch_chunk_gated_delta_rule``
    (the same implementation used for deterministic mode, computing in
    float32 for numeric stability).

    Packed sequences (*cu_seqlens* not None) are handled by per-sequence
    unpack → compute → repack.

    Args:
        query: ``[B, T, H, K]``
        key: ``[B, T, H, K]``
        value: ``[B, T, H, V]``
        g: ``[B, T, H]`` (gating in log space)
        beta: ``[B, T, H]``
        chunk_size: chunk size (default 64)
        initial_state: ``[N, H, K, V]`` or ``None``
        output_final_state: return final state if True
        use_qk_l2norm_in_kernel: apply l2norm internally if True
        cu_seqlens: ``[N+1]`` cumulative lengths or ``None``

    Returns:
        ``(o, final_state)``
    """
    from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule

    if cu_seqlens is None:
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    # Packed sequence: per-sequence unpack → compute → repack
    chunks_q = _unpack_sequence_per_seq(query, cu_seqlens, dim=1)
    chunks_k = _unpack_sequence_per_seq(key, cu_seqlens, dim=1)
    chunks_v = _unpack_sequence_per_seq(value, cu_seqlens, dim=1)
    chunks_g = _unpack_sequence_per_seq(g, cu_seqlens, dim=1)
    chunks_beta = _unpack_sequence_per_seq(beta, cu_seqlens, dim=1)

    outs = []
    for q_i, k_i, v_i, g_i, b_i in zip(chunks_q, chunks_k, chunks_v, chunks_g, chunks_beta):
        o_i, _ = torch_chunk_gated_delta_rule(
            q_i,
            k_i,
            v_i,
            g=g_i,
            beta=b_i,
            chunk_size=chunk_size,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )
        outs.append(o_i)
    if not outs:
        # Degenerate pack: return an empty tensor shaped like ``value``.
        return torch.zeros_like(value), None
    return torch.cat(outs, dim=1), None
