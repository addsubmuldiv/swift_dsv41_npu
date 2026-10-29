# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""
Rotary Position Embedding implementation of different types along with helper functions
"""

from typing import Optional, Tuple, Union, List
import torch
import torch_npu

__all__ = [
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",
    "apply_fused_qkv_rotary_pos_emb",
    "fused_apply_mla_rope_for_q",
    "fused_apply_mla_rope_for_kv",
    "MLARopeUnsupportedError",
]


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        dim: int,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[int] = None,
        pretrained_max_position_embeddings: Optional[int] = None,
        rotary_base: float = 10000.0,
        interleaved: bool = False,
    ):
        """
        Parameters
        ----------
        dim: int
            Rotary embedding dimension.
        rotary_percent: float, default = 1.0
            Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor: int, default = None
            If not None, discrete positions will be interpolated by this factor via the trick in
            https://arxiv.org/abs/2306.15595
        pretrained_max_position_embeddings: int, default = None
            Pre-trained max_position_embeddings before position interpolation.
        rotary_base: float, default = 10000.0
            Base of the rotary position embedding.
        interleaved: bool, default = False
            Whether to use interleaved rotary position embedding.
        """
        super().__init__()
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_base = rotary_base
        inv_freq = 1.0 / (
            self.rotary_base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.npu.current_device())
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings
        self.interleaved = interleaved

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies.

        This function is particularly sensitive to the use of mixed precision, so we disable the
        autocast context if it is enabled.

        Parameters
        ----------
        max_seq_len: int
            Sequence length of a sample.
        offset: int, default = 0
            Fixed offset for frequencies.
        """
        with torch.autocast(enabled=False, device_type="npu"):
            seq = (
                torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
                + offset
            )

            if (
                self.pretrained_max_position_embeddings is not None
                and self.seq_len_interpolation_factor is not None
            ):
                if (
                    max_seq_len
                    > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor
                ):
                    seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
                else:
                    seq *= 1 / self.seq_len_interpolation_factor

            freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        if not self.interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        return emb.reshape(emb.size(0), 1, 1, emb.size(1))


def _rotate_half_thd(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], -1)


def _npu_rotary_mul(
    t: torch.Tensor,
    cos_: torch.Tensor,
    sin_: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    rotary_mode = "interleave" if interleaved else "half"
    return torch_npu.npu_rotary_mul(
        t.contiguous(), cos_.contiguous(), sin_.contiguous(), rotary_mode
    )


def _rotary_mul_adjoint(
    grad: torch.Tensor,
    cos_: torch.Tensor,
    sin_: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """True adjoint of the ``npu_rotary_mul`` kernel transform.

    The kernel forward is asymmetric per pair (probed on NPU): for a pair
    (i, i+w/2) the first element uses ``sin[i]`` while the second uses
    ``sin[i+w/2]`` (half mode; interleave pairs (e, e+1) analogously).
    ``npu_rotary_mul(grad, cos, -sin)`` is therefore NOT the adjoint — it
    swaps the sin indices (correct only when freqs are duplicated angles,
    where sin[i] == sin[i+w/2]). With arbitrary freqs the mismatch shows up
    as the fused backward divergence (REPORT atk 3#10). This helper applies
    the exact transpose:

    half:      fwd  out[i] = c[i]t[i] - s[i]t[i+w/2]
                     out[i+w/2] = s[i+w/2]t[i] + c[i+w/2]t[i+w/2]
               adj  grad[i] = c[i]g[i] + s[i+w/2]g[i+w/2]
                     grad[i+w/2] = -s[i]g[i] + c[i+w/2]g[i+w/2]
    interleave: same structure on pairs (e, e+1).
    """
    w = grad.shape[-1]
    if interleaved:
        # pairs are (2j, 2j+1):
        #   fwd  out[e] = c[e]t[e] - s[e]t[o];  out[o] = s[o]t[e] + c[o]t[o]
        #   adj  grad[e] = c[e]g[e] + s[o]g[o]; grad[o] = -s[e]g[e] + c[o]g[o]
        g_e, g_o = grad[..., 0::2], grad[..., 1::2]
        c_e, s_e = cos_[..., 0::2], sin_[..., 0::2]
        c_o, s_o = cos_[..., 1::2], sin_[..., 1::2]
        grad_e = g_e * c_e + g_o * s_o
        grad_o = -g_e * s_e + g_o * c_o
        out = torch.stack((grad_e, grad_o), dim=-1).flatten(-2)
        return out.reshape(grad.shape)
    # half mode: pairs (i, i+w/2):
    #   fwd  out[i] = c[i]t[i] - s[i]t[i+w/2];  out[i+w/2] = s[i+w/2]t[i] + c[i+w/2]t[i+w/2]
    #   adj  grad[i] = c[i]g[i] + s[i+w/2]g[i+w/2];  grad[i+w/2] = -s[i]g[i] + c[i+w/2]g[i+w/2]
    g1, g2 = grad[..., : w // 2], grad[..., w // 2 :]
    c1, s1 = cos_[..., : w // 2], sin_[..., : w // 2]
    c2, s2 = cos_[..., w // 2 :], sin_[..., w // 2 :]
    grad_1 = g1 * c1 + g2 * s2
    grad_2 = -g1 * s1 + g2 * c2
    return torch.cat((grad_1, grad_2), dim=-1)


def _rotary_mul_bwd(
    grad: torch.Tensor,
    cos_: torch.Tensor,
    sin_: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Backward of the ``npu_rotary_mul`` kernel transform, two-path dispatch.

    Path 1 (fast, fused kernel): when the freqs are in the duplicated-angle
    format produced by every standard rotary embedding constructor
    (``cat(freqs, freqs)`` non-interleaved / stacked interleaved — Megatron
    ``RotaryEmbedding`` and TE ``RotaryPositionEmbedding`` alike), the halves
    of ``sin_`` are elementwise equal, so ``npu_rotary_mul(g, cos, -sin)``
    — whose only flaw is swapping the per-pair sin indices (REPORT atk
    3#10) — is numerically exact. Verified on NPU: bitwise identical to the
    adjoint in fp32; bf16 differs only by 1-ULP rounding order (kernel
    accumulates in one pass, the Python adjoint rounds between ops). This is
    the real-training path (Megatron always feeds duplicated angles).

    Path 2 (exact Python adjoint): arbitrary freqs where
    ``sin_[..., :w/2] != sin_[..., w/2:]`` — the kernel's index swap would be
    a real gradient error, so ``_rotary_mul_adjoint`` (the exact transpose)
    is used instead.

    The dispatch check is one equality over the freqs-width tensor (~1 MB at
    seq 4096) — negligible against the rope operand (~64 MB).
    """
    w = grad.shape[-1]
    if w >= 2:
        if interleaved:
            # stacked format: pairs (2j, 2j+1) duplicate within each pair
            dup = torch.equal(sin_[..., 0::2], sin_[..., 1::2])
        else:
            dup = torch.equal(sin_[..., : w // 2], sin_[..., w // 2 :])
        if dup:
            return _npu_rotary_mul(grad, cos_, -sin_, interleaved)
    return _rotary_mul_adjoint(grad, cos_, sin_, interleaved)


def _apply_fused_rope_4d(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str,
    interleaved: bool,
) -> torch.Tensor:
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)

    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]

    use_npu_kernel = True
    if interleaved:
        if tensor_format == "sbhd" and freqs.shape[1] != 1:
            use_npu_kernel = False
        elif tensor_format == "bshd" and freqs.shape[0] != 1:
            use_npu_kernel = False

    if use_npu_kernel:
        t_rot = _npu_rotary_mul(
            t_rot.contiguous(), cos_.contiguous(), sin_.contiguous(), interleaved
        )
    else:
        t_rot = (t_rot * cos_) + (_rotate_half(t_rot, interleaved) * sin_)

    return torch.cat([t_rot, t_pass], dim=-1)


def _apply_fused_rope_thd_single(
    x: torch.Tensor,
    freqs: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    x_4d = x.unsqueeze(1)
    cos_ = torch.cos(freqs).to(x_4d.dtype)
    sin_ = torch.sin(freqs).to(x_4d.dtype)

    rot_dim = freqs.shape[-1]
    t_rot = x_4d[..., :rot_dim].contiguous()
    t_pass = x_4d[..., rot_dim:]

    t_rot = _npu_rotary_mul(t_rot, cos_, sin_, interleaved)
    result = torch.cat([t_rot, t_pass], dim=-1)
    return result.squeeze(1)


class FusedRoPEFunc(torch.autograd.Function):
    """Function for FusedRoPE on NPU.

    Uses torch_npu.npu_rotary_mul as the fused kernel. The Python-level preprocessing
    (start_positions, context parallel, cu_seqlens) is handled before calling the kernel,
    and the kernel handles the core rotation computation with autograd support.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        start_positions: Union[torch.Tensor, None] = None,
        tensor_format: str = "sbhd",
        interleaved: bool = False,
        cu_seqlens: Union[torch.Tensor, None] = None,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        assert tensor_format in (
            "sbhd",
            "bshd",
            "thd",
        ), f"Unsupported tensor_format: {tensor_format}."

        if tensor_format == "thd":
            assert cu_seqlens is not None, (
                "cu_seqlens must not be None when tensor_format is 'thd'."
            )
            cu_seqlens_cp = cu_seqlens // cp_size
            seqlens = (cu_seqlens_cp[1:] - cu_seqlens_cp[:-1]).tolist()
            output = torch.cat(
                [
                    _apply_fused_rope_thd_single(
                        x,
                        _get_freqs_on_this_cp_rank(
                            freqs[start_positions[idx] :] if start_positions is not None else freqs,
                            x.size(0),
                            cp_size,
                            cp_rank,
                        ),
                        interleaved=interleaved,
                    )
                    for idx, x in enumerate(torch.split(t, seqlens))
                ]
            )
            ctx.save_for_backward(freqs, start_positions, cu_seqlens)
            ctx.seqlens = seqlens
        else:
            if tensor_format == "sbhd":
                seqlen = t.size(0)
            else:
                seqlen = t.size(1)

            if start_positions is not None:
                max_offset = torch.max(start_positions)
                assert max_offset + seqlen * cp_size <= freqs.shape[0], (
                    f"Rotary Embeddings only supported up to {freqs.shape[0]} sequence length!"
                )
                freqs = torch.concatenate(
                    [freqs[i : i + seqlen * cp_size] for i in start_positions], dim=1
                )

            freqs = _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank)
            output = _apply_fused_rope_4d(t, freqs, tensor_format, interleaved)
            ctx.save_for_backward(freqs)

        ctx.tensor_format = tensor_format
        ctx.interleaved = interleaved
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.t_shape = t.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        tensor_format = ctx.tensor_format
        interleaved = ctx.interleaved

        if tensor_format == "thd":
            freqs, start_positions, cu_seqlens = ctx.saved_tensors
            seqlens = ctx.seqlens
            cp_size = ctx.cp_size
            cp_rank = ctx.cp_rank

            grad_inputs = []
            for idx, x_grad in enumerate(torch.split(grad_output, seqlens)):
                seq_freqs = _get_freqs_on_this_cp_rank(
                    freqs[start_positions[idx] :] if start_positions is not None else freqs,
                    x_grad.size(0),
                    cp_size,
                    cp_rank,
                )
                cos_ = (
                    torch.cos(seq_freqs)
                    .reshape(seq_freqs.shape[0], 1, seq_freqs.shape[-1])
                    .to(x_grad.dtype)
                )
                sin_ = (
                    torch.sin(seq_freqs)
                    .reshape(seq_freqs.shape[0], 1, seq_freqs.shape[-1])
                    .to(x_grad.dtype)
                )
                rot_dim = seq_freqs.shape[-1]
                x_grad_rot = x_grad[..., :rot_dim]
                # The THD forward runs the asymmetric npu_rotary_mul kernel
                # (via _apply_fused_rope_thd_single); the exact transpose of
                # that transform, not the symmetric _rotate_half_thd formula,
                # is the correct adjoint (REPORT atk 3#10, same sin-index fix
                # as the sbhd/bshd kernel branches). _rotary_mul_bwd keeps the
                # fused kernel for duplicated-angle freqs (fast path) and
                # falls back to the exact adjoint for arbitrary freqs.
                x_grad_rot = _rotary_mul_bwd(x_grad_rot, cos_, sin_, interleaved)
                if rot_dim < x_grad.shape[-1]:
                    grad_inputs.append(torch.cat([x_grad_rot, x_grad[..., rot_dim:]], dim=-1))
                else:
                    grad_inputs.append(x_grad_rot)
            grad_input = torch.cat(grad_inputs)
            return grad_input, None, None, None, None, None, None, None, None

        (freqs,) = ctx.saved_tensors

        if tensor_format == "bshd":
            freqs = freqs.transpose(0, 1)

        cos_ = torch.cos(freqs).to(grad_output.dtype)
        sin_ = torch.sin(freqs).to(grad_output.dtype)

        rot_dim = freqs.shape[-1]
        grad_output_rot = grad_output[..., :rot_dim].contiguous()

        use_npu_kernel = True
        if interleaved:
            if tensor_format == "sbhd" and freqs.shape[1] != 1:
                use_npu_kernel = False
            elif tensor_format == "bshd" and freqs.shape[0] != 1:
                use_npu_kernel = False

        if use_npu_kernel:
            # Two-path backward (REPORT atk 3#10): duplicated-angle freqs
            # (every standard rotary constructor) keep the fused kernel —
            # numerically exact there; arbitrary freqs take the exact adjoint.
            grad_input_rot = _rotary_mul_bwd(grad_output_rot, cos_, sin_, interleaved)
        else:
            grad_input_rot = (grad_output_rot * cos_) - (
                _rotate_half(grad_output_rot, interleaved) * sin_
            )

        if rot_dim < grad_output.shape[-1]:
            grad_input = torch.cat([grad_input_rot, grad_output[..., rot_dim:]], dim=-1)
        else:
            grad_input = grad_input_rot

        return grad_input, None, None, None, None, None, None, None, None


class FusedQKVRoPEFunc(torch.autograd.Function):
    """Function for FusedQKVRoPE on NPU.

    Applies RoPE to Q and K separately using torch_npu.npu_rotary_mul,
    while V passes through unchanged.
    """

    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
        qkv_split_arg_list: List[int],
        start_positions: Union[torch.Tensor, None] = None,
        tensor_format: str = "sbhd",
        interleaved: bool = False,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        if q_freqs.dtype != torch.float32:
            q_freqs = q_freqs.float()
        if k_freqs.dtype != torch.float32:
            k_freqs = k_freqs.float()
        assert tensor_format in (
            "sbhd",
            "bshd",
        ), f"Unsupported tensor_format: {tensor_format}."
        assert qkv.is_contiguous(), "QKV Tensor should be contiguous."
        assert q_freqs.is_contiguous(), "q_freqs Tensor should be contiguous."
        assert k_freqs.is_contiguous(), "k_freqs Tensor should be contiguous."

        q_dim, k_dim, v_dim = qkv_split_arg_list
        # NVTE fused-QKV semantics (apply_rope.cpp:123-125 and NVTE's own
        # test_fused_rope.py:394-465): the fused tensor is [s, b, ng, q_dim +
        # k_dim + v_dim] where each group holds the group's q channels
        # (q_dim = heads_per_group * hn) followed by one k and one v head.
        # Split the last dim, then head-expand q to [s, b, ng * heads_per_group,
        # hn]; k/v stay [s, b, ng, hn]. q_freqs/k_freqs cover hn/2 rotary pairs
        # per head (not half of the group's q width).
        q, k, v = torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)
        ng = qkv.shape[-2]
        heads_per_group = q_dim // k_dim
        q = q.reshape(*q.shape[:-2], ng * heads_per_group, k_dim)

        if tensor_format == "sbhd":
            seqlen = qkv.size(0)
        else:
            seqlen = qkv.size(1)

        if start_positions is not None:
            max_offset = torch.max(start_positions)
            assert max_offset + seqlen * cp_size <= q_freqs.shape[0], (
                f"Rotary Embeddings only supported up to {q_freqs.shape[0]} sequence length!"
            )
            q_freqs = torch.concatenate(
                [q_freqs[i : i + seqlen * cp_size] for i in start_positions], dim=1
            )
            k_freqs = torch.concatenate(
                [k_freqs[i : i + seqlen * cp_size] for i in start_positions], dim=1
            )

        q_freqs = _get_freqs_on_this_cp_rank(q_freqs, seqlen, cp_size, cp_rank)
        k_freqs = _get_freqs_on_this_cp_rank(k_freqs, seqlen, cp_size, cp_rank)

        q = _apply_fused_rope_4d(q, q_freqs, tensor_format, interleaved)
        k = _apply_fused_rope_4d(k, k_freqs, tensor_format, interleaved)

        ctx.save_for_backward(q_freqs, k_freqs)
        ctx.tensor_format = tensor_format
        ctx.qkv_split_arg_list = qkv_split_arg_list
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.interleaved = interleaved

        return q, k, v

    @staticmethod
    def backward(
        ctx, grad_q: torch.Tensor, grad_k: torch.Tensor, grad_v: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        q_freqs, k_freqs = ctx.saved_tensors
        tensor_format = ctx.tensor_format
        interleaved = ctx.interleaved
        q_dim, k_dim, v_dim = ctx.qkv_split_arg_list

        if tensor_format == "bshd":
            q_freqs = q_freqs.transpose(0, 1)
            k_freqs = k_freqs.transpose(0, 1)

        q_cos = torch.cos(q_freqs).to(grad_q.dtype)
        q_sin = torch.sin(q_freqs).to(grad_q.dtype)
        k_cos = torch.cos(k_freqs).to(grad_k.dtype)
        k_sin = torch.sin(k_freqs).to(grad_k.dtype)

        q_rot_dim = q_freqs.shape[-1]
        k_rot_dim = k_freqs.shape[-1]

        use_npu_kernel = True
        if interleaved:
            if tensor_format == "sbhd" and q_freqs.shape[1] != 1:
                use_npu_kernel = False
            elif tensor_format == "bshd" and q_freqs.shape[0] != 1:
                use_npu_kernel = False

        grad_q_rot = grad_q[..., :q_rot_dim].contiguous()
        grad_q_pass = grad_q[..., q_rot_dim:]
        if use_npu_kernel:
            # Two-path backward (REPORT atk 3#10): duplicated-angle freqs keep
            # the fused kernel; arbitrary freqs take the exact adjoint.
            grad_q_rot = _rotary_mul_bwd(grad_q_rot, q_cos, q_sin, interleaved)
        else:
            grad_q_rot = (grad_q_rot * q_cos) - (_rotate_half(grad_q_rot, interleaved) * q_sin)
        grad_q_out = (
            torch.cat([grad_q_rot, grad_q_pass], dim=-1)
            if q_rot_dim < grad_q.shape[-1]
            else grad_q_rot
        )

        grad_k_rot = grad_k[..., :k_rot_dim].contiguous()
        grad_k_pass = grad_k[..., k_rot_dim:]
        if use_npu_kernel:
            grad_k_rot = _rotary_mul_bwd(grad_k_rot, k_cos, k_sin, interleaved)
        else:
            grad_k_rot = (grad_k_rot * k_cos) - (_rotate_half(grad_k_rot, interleaved) * k_sin)
        grad_k_out = (
            torch.cat([grad_k_rot, grad_k_pass], dim=-1)
            if k_rot_dim < grad_k.shape[-1]
            else grad_k_rot
        )

        # grad_q arrives head-expanded [s, b, np, hn] (the forward output
        # layout); fold it back to the fused group layout [s, b, ng, q_dim]
        # before concatenating with k/v (NVTE applies the same layout round-trip
        # in tex.fused_qkv_rope_backward).
        ng = grad_q_out.shape[-2] // (q_dim // k_dim)
        grad_q_out = grad_q_out.reshape(*grad_q_out.shape[:-2], ng, q_dim)
        grad_input = torch.cat([grad_q_out, grad_k_out, grad_v], dim=-1)

        return grad_input, None, None, None, None, None, None, None, None


def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x: torch.Tensor. Input tensor.
        interleaved: bool. Whether to use interleaved rotary position embedding.

    Returns:
        Tensor: Tensor rotated half.
    """
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_base(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Base implementation of applying rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape `[s, b, h, d]` or `[b, s, h, d]`, on which rotary positional
        embedding will be applied.
    freqs : torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` or `[s2, b, 1, d2]`
        and dtype 'float', with `s2 >= s` and `d2 <= d`.
    tensor_format : {'sbhd', 'bshd'}, default = 'sbhd'
        Should be `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is of shape
        `[seq, bs, ...]`.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    """
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)

    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _get_freqs_on_this_cp_rank(
    freqs: torch.Tensor, seqlen: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        freqs: torch.Tensor. Positional embedding tensor of shape `[s2, 1, 1, d2]`.
        seqlen: int. Length of the current sequence.
        cp_size: int. Context parallel world size.
        cp_rank: int. Context parallel rank.
    """
    if cp_size > 1:
        cp_seg = seqlen // 2
        full_seqlen = cp_size * seqlen
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )

    return freqs[:seqlen]


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Support matrix:
    Fused (NPU):
        qkv_formats:            "thd", "bshd", "sbhd"
        context parallel:       yes
        start_positions:        yes
        interleaving:           yes
    Unfused (NPU):
        qkv_formats:            "thd", "bshd", "sbhd"
        context parallel:       yes
        start_positions:        yes
        interleaving:           yes

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[b, s, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs : torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    start_positions : torch.Tensor, default = None.
        Tokens in a sequence `i` should be applied with position encoding offset by
        `start_positions[i]`. If `start_positions=None`, there's no offset.
    tensor_format : {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    fused : bool, default = False
        Whether to use a fused applying RoPE implementation.
    cu_seqlens : torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
        Should be `cu_seqlens_padded` when cp_size > 1.
    cp_size : int, default = 1.
        Context parallel world size. Only valid when `tensor_format` is 'thd' and `fused` is True.
    cp_rank : int, default = 0.
        Context parallel rank. Only valid when `tensor_format` is 'thd' and `fused` is True.
    """
    assert tensor_format != "thd" or cu_seqlens is not None, (
        "cu_seqlens must not be None when tensor_format is 'thd'."
    )

    if fused:
        return FusedRoPEFunc.apply(
            t, freqs, start_positions, tensor_format, interleaved, cu_seqlens, cp_size, cp_rank
        )

    if tensor_format == "thd":
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        return torch.cat(
            [
                _apply_rotary_pos_emb_base(
                    x.unsqueeze(1),
                    _get_freqs_on_this_cp_rank(
                        (freqs[start_positions[idx] :] if start_positions is not None else freqs),
                        x.size(0),
                        cp_size,
                        cp_rank,
                    ),
                    interleaved=interleaved,
                )
                for idx, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)

    if tensor_format == "sbhd":
        seqlen = t.size(0)
    elif tensor_format == "bshd":
        seqlen = t.size(1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert max_offset + seqlen * cp_size <= freqs.shape[0], (
            f"Rotary Embeddings only suppported up to {freqs.shape[0]} sequence length!"
        )

        freqs = torch.concatenate([freqs[i : i + seqlen * cp_size] for i in start_positions], dim=1)

    return _apply_rotary_pos_emb_base(
        t,
        _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank),
        tensor_format,
        interleaved=interleaved,
    )


def apply_fused_qkv_rotary_pos_emb(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding tensor to the input qkv tensor.

    Support matrix:
    Fused (NPU):
        Training:
            qkv_formats:            "bshd", "sbhd"
            context parallel:       yes
            start_positions:        no
            interleaving:           yes
        Inference:
            qkv_formats:            "bshd", "sbhd"
            context parallelism:    no
            start_positions:        yes
            interleaving:           yes

    Parameters
    ----------
    qkv : torch.Tensor
        Input tensor of shape `[s, b, h, d]` or `[b, s, h, d]`, on which
        rotary positional embedding will be applied. This tensor has q, k, v concatenated
        along the last dimension.
    q_freqs : torch.Tensor
        Rotary positional embedding Q tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    k_freqs : torch.Tensor
        Rotary positional embedding K tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    qkv_split_arg_list : List[int]
        List of integers that specify the split of the qkv tensor. The list should have 3 elements,
        the first element is the number of elements in the q tensor, the second element is the number
        of elements in the k tensor, and the third element is the number of elements in the v tensor.
        The sum of the elements in the list should be equal to the last dimension of the qkv tensor.
    start_positions : torch.Tensor, default = None.
        Tokens in a sequence `i` should be applied with position encoding offset by
        `start_positions[i]`. If `start_positions=None`, there's no offset.
    tensor_format : {'sbhd', 'bshd'}, default = 'sbhd'
        is `bshd` if `qkv` is of shape `[bs, seq, ...]`, or `sbhd` if `qkv` is of shape
        `[seq, bs, ...]`.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    cp_size : int, default = 1.
        Context parallel world size.
    cp_rank : int, default = 0.
        Context parallel rank.
    """
    assert not (cp_size > 1 and start_positions is not None), (
        "start_positions != None with CP SIZE > 1 is not supported!"
    )

    assert tensor_format != "thd", "'thd' tensor_format not supported currently."

    return FusedQKVRoPEFunc.apply(
        qkv,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        start_positions,
        tensor_format,
        interleaved,
        cp_size,
        cp_rank,
    )


class MLARopeUnsupportedError(NotImplementedError):
    """Raised when the NPU MLA rope backend cannot handle the given inputs.

    Callers (e.g. MegatronAdaptor) treat this as "fall back to the original
    implementation"; other exceptions are real errors and must propagate.
    """


def _mla_rope_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, seq: int, emb_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate and convert Megatron's MLA rope table for the native kernel.

    Megatron caches cos/sin as ``[seq, 1, 1, emb_dim]`` with the two halves holding
    the same angles (``cat(freqs, freqs)``) and rotates interleaved pairs as
    ``even <- cos[j]``, ``odd <- cos[emb_dim // 2 + j]``. ``npu_rotary_mul`` in
    interleave mode indexes ``cos[2j]`` for the even output and ``cos[2j + 1]`` for
    the odd one, so the two halves must be interleaved first. The kernel's registered
    backward is the exact adjoint only for duplicated angles, so tables whose halves
    differ (e.g. a stacked-layout cache) are rejected with ``MLARopeUnsupportedError``
    and the caller falls back to the upstream implementation.
    """
    half = emb_dim // 2
    cos = cos[:seq]
    sin = sin[:seq]
    if not (
        torch.equal(cos[..., :half], cos[..., half:])
        and torch.equal(sin[..., :half], sin[..., half:])
    ):
        raise MLARopeUnsupportedError(
            "MLA rope on Ascend requires Megatron's duplicated-angle rope table "
            "(cat(freqs, freqs)): the npu_rotary_mul backward is not the exact "
            "adjoint for other table layouts"
        )
    cos = torch.stack((cos[..., :half], cos[..., half:]), dim=-1).flatten(-2).contiguous()
    sin = torch.stack((sin[..., :half], sin[..., half:]), dim=-1).flatten(-2).contiguous()
    return cos, sin


def fused_apply_mla_rope_for_q(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qk_head_dim: int,
    emb_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
) -> torch.Tensor:
    """Apply YARN RoPE to MLA's query on Ascend.

    Mirrors the semantics of Megatron-LM's ``fused_apply_mla_rope_for_q``: only the
    trailing ``emb_dim`` channels of ``t`` are rotated, in the interleaved (GPT-J
    style) layout, while the leading ``qk_head_dim`` channels pass through.

    The output is element-wise identical to the Triton kernel, whose rotated channels
    are written back in stacked (first-half/second-half) order; the per-call fallback
    to the upstream implementation in MegatronAdaptor is therefore layout-transparent.

    Parameters
    ----------
    t : torch.Tensor
        Query of shape ``[seq, batch, head_num, qk_head_dim + emb_dim]``.
    cos, sin : torch.Tensor
        Cached rope tables of shape ``[max_seq, 1, 1, emb_dim]``, with duplicated
        halves (``cat(freqs, freqs)``, Megatron's MLA cache layout); anything else
        raises ``MLARopeUnsupportedError``. A longer table is truncated to
        ``t.shape[0]`` (context parallel slicing leaves it longer).
    qk_head_dim : int
        Number of non-rotary channels at the front of the last dimension.
    emb_dim : int
        Number of rotary channels at the end of the last dimension.
    cu_seqlens_q : torch.Tensor, default = None
        Packed-sequence offsets. Not supported: ``MLARopeUnsupportedError`` is raised.
    cp_rank / cp_size : int
        Accepted for signature compatibility; cos/sin are already sliced by the caller.
    rotary_interleaved : bool, default = False
        Must be False: MLA layouts are always interleaved internally.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape/dtype as ``t``.
    """
    if t.dim() != 4:
        raise MLARopeUnsupportedError(
            "MLA rope-for-q on Ascend expects a 4D [seq, batch, head, dim] input"
        )
    if cu_seqlens_q is not None or rotary_interleaved:
        raise MLARopeUnsupportedError(
            "MLA rope-for-q on Ascend does not support packed sequences or rotary_interleaved=True"
        )
    if qk_head_dim < 0 or qk_head_dim + emb_dim != t.shape[-1]:
        raise ValueError(
            f"qk_head_dim + emb_dim must match the last dimension of q "
            f"({qk_head_dim} + {emb_dim} != {t.shape[-1]})!"
        )

    seq = t.shape[0]
    cos, sin = _mla_rope_cos_sin(cos, sin, seq, emb_dim)
    rotated = _npu_rotary_mul(t[..., qk_head_dim:].contiguous(), cos, sin, interleaved=True)
    # npu_rotary_mul returns interleaved pairs; Megatron writes first-half/second-half.
    rotated = torch.cat((rotated[..., 0::2], rotated[..., 1::2]), dim=-1)
    if qk_head_dim == 0:
        return rotated
    return torch.cat((t[..., :qk_head_dim], rotated), dim=-1)


def fused_apply_mla_rope_for_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    emb_dim: int,
    k_dim: int,
    v_dim: int,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split MLA's compressed KV, rotate the shared positional key and build key/value.

    ``k_pos_emb`` holds a single head, so only ``seq * emb_dim`` elements are rotated;
    the result is broadcast into all heads and concatenated after the non-rotary key
    channels. ``value`` is returned as a view of ``kv`` (no copy). The rotated
    channels are written back in the same stacked order as Megatron's Triton kernel
    (see ``fused_apply_mla_rope_for_q``).

    Parameters
    ----------
    kv : torch.Tensor
        Compressed key/value of shape ``[seq, batch, head_num, k_dim + v_dim]``.
    k_pos_emb : torch.Tensor
        Positional key of shape ``[seq, batch, 1, emb_dim]``.
    cos, sin : torch.Tensor
        Cached rope tables of shape ``[max_seq, 1, 1, emb_dim]``.
    emb_dim / k_dim / v_dim : int
        Rotary / non-rotary key / value channel counts.
    cu_seqlens_kv : torch.Tensor, default = None
        Packed-sequence offsets. Not supported: ``MLARopeUnsupportedError`` is raised.

    Returns
    -------
    (key, value) : Tuple[torch.Tensor, torch.Tensor]
        ``key`` of shape ``[seq, batch, head_num, emb_dim + k_dim]`` and ``value`` of
        shape ``[seq, batch, head_num, v_dim]``.
    """
    if kv.dim() != 4:
        raise MLARopeUnsupportedError(
            "MLA rope-for-kv on Ascend expects a 4D [seq, batch, head, dim] input"
        )
    if cu_seqlens_kv is not None or rotary_interleaved:
        raise MLARopeUnsupportedError(
            "MLA rope-for-kv on Ascend does not support packed sequences or rotary_interleaved=True"
        )
    if kv.shape[-1] != k_dim + v_dim:
        raise ValueError(
            f"k_dim + v_dim must match the last dimension of kv "
            f"({k_dim} + {v_dim} != {kv.shape[-1]})!"
        )
    if (
        k_pos_emb.dim() != 4
        or k_pos_emb.shape[2] != 1
        or k_pos_emb.shape[0] != kv.shape[0]
        or k_pos_emb.shape[1] != kv.shape[1]
        or k_pos_emb.shape[-1] != emb_dim
    ):
        raise ValueError(
            "k_pos_emb must be [seq, batch, 1, emb_dim] with the same seq/batch as kv, "
            f"got {tuple(k_pos_emb.shape)} for kv {tuple(kv.shape)}"
        )

    seq, heads = kv.shape[0], kv.shape[2]
    cos, sin = _mla_rope_cos_sin(cos, sin, seq, emb_dim)
    k_rot = _npu_rotary_mul(k_pos_emb, cos, sin, interleaved=True)
    # npu_rotary_mul returns interleaved pairs; Megatron writes first-half/second-half.
    k_rot = torch.cat((k_rot[..., 0::2], k_rot[..., 1::2]), dim=-1)
    key = torch.cat((kv[..., :k_dim], k_rot.expand(-1, -1, heads, -1)), dim=-1)
    return key, kv[..., k_dim:]
