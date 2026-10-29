# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass
from einops import rearrange

import torch
import torch.distributed as dist
import torch_npu


def _get_args():
    return type('Args', (), {'prof_file': False})()


def _get_npu_ring_attention_update():
    from .ops.npu_ring_attention_update import npu_ring_attention_update

    return npu_ring_attention_update


def _get_batch_index(seq1d, seq_len):
    from bisect import bisect_right

    end_points = list(range(seq_len, seq1d[-1] + 1, seq_len))
    indexes = [0] + [bisect_right(seq1d, p) for p in end_points]
    seq_batch = [seq1d[indexes[i] : indexes[i + 1]] for i in range(len(indexes) - 1)]
    return [[elem - i * seq_len for elem in seq] for i, seq in enumerate(seq_batch)]


@lru_cache(maxsize=8)
def get_selection_indices_for_tnd_softmax_update(t, n, sub_seq_len):
    full_indices = list(range(t * n))
    indices = []
    seq_start = 0
    for seq_len in sub_seq_len:
        for i in range(n):
            start = seq_start + seq_len * 2 * i + seq_len
            end = seq_start + seq_len * 2 * (i + 1)
            indices.extend(full_indices[start:end])
        seq_start += seq_len * n * 2
    return torch.tensor(indices)


def flatten_softmax(x, sub_seq_len):
    orig_shape = x.shape
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(section_len, dim=0)
    merged = [item.view(orig_shape[1], -1, orig_shape[-1]).transpose(0, 1) for item in splits]
    merged = torch.cat(merged, dim=0)
    return merged


def unflatten_softmax(x, sub_seq_len):
    orig_shape = x.shape
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(section_len, dim=0)
    merged = [
        item.view(-1, orig_shape[1], orig_shape[-1]).transpose(0, 1).view(-1, orig_shape[-1])
        for item in splits
    ]
    merged = torch.cat(merged, dim=0)
    return merged.view(*orig_shape)


def _to_rank_local_cu_seqlens(cu_seqlens_q, cu_seqlens_kv, q, k, cp_size):
    """Normalize plain THD cu_seqlens to rank-local endpoint lists.

    Two caller conventions reach the non-``cu_seqlens_*_padded`` THD path:
    MindSpeed's EOD-reset pipeline feeds endpoints that are already
    rank-local, while Megatron-LM's native SFT pipeline (the upstream GPU-TE
    convention) feeds endpoints over the full CP group and expects the
    attention layer to convert them, as the CUDA kernels do. Distinguish the
    two by comparing the final endpoint with this rank's token count and
    rescale global inputs by ``cp_size``; the division is exact because the
    THD CP partition requires every segment to be padded to a multiple of
    ``2 * cp_size``.
    """

    def convert(cu_seqlens, local_tokens):
        if isinstance(cu_seqlens, torch.Tensor):
            endpoints = cu_seqlens.tolist()
        else:
            endpoints = [int(value) for value in cu_seqlens]
        total = endpoints[-1]
        if total == local_tokens:
            return endpoints
        if total == local_tokens * cp_size:
            if any(value % cp_size for value in endpoints):
                raise ValueError(
                    f"Global cu_seqlens {endpoints} has entries not divisible by "
                    f"cp_size={cp_size}; THD context parallel requires every segment padded "
                    "to a multiple of 2*cp_size."
                )
            return [value // cp_size for value in endpoints]
        raise ValueError(
            f"cu_seqlens[-1]={total} matches neither the rank-local token count "
            f"({local_tokens}) nor the global one ({local_tokens * cp_size})."
        )

    return convert(cu_seqlens_q, q.shape[0]), convert(cu_seqlens_kv, k.shape[0])


def causal_forward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_mask=None):
    cur_attn_mask = None
    if q_block_id == kv_block_id:
        # [2, s, b, h] -> [2s, b, h]
        cur_attn_mask = attn_mask
        cur_q, cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [q, cur_k, cur_v]]
    elif kv_block_id <= q_block_id:
        # [2, s, b, h] -> [2s, b, h]
        cur_q = q.view(-1, *q.shape[2:])
        # only k[0] v[0] need to be calculated
        cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
    else:
        # only q[1] need to be calculated
        cur_q = q[1]
        # [2, s, b, h] -> [2s, b, h]
        cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]

    return cur_q, cur_k, cur_v, cur_attn_mask


def tnd_forward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, fetch_ptrs, attn_mask=None):
    seqlen, half_seqlen, q_index, kv_index = fetch_ptrs
    actual_seq_qlen, actual_seq_kvlen, sub_out_seq_len = seqlen
    half_actual_seq_qlen, half_actual_seq_kvlen, half_sub_out_seq_len = half_seqlen

    cur_attn_mask = None
    if q_block_id == kv_block_id:
        cur_attn_mask = attn_mask
        cur_q = q
        cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, actual_seq_kvlen
        cur_sub_out_seq_len = sub_out_seq_len
    elif kv_block_id <= q_block_id:
        cur_q = q
        cur_k, cur_v = [torch.index_select(x, 0, kv_index) for x in [cur_k, cur_v]]
        cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, half_actual_seq_kvlen
        cur_sub_out_seq_len = sub_out_seq_len
    else:
        cur_q = torch.index_select(q, 0, q_index)
        cur_seq_qlen, cur_seq_kvlen = half_actual_seq_qlen, actual_seq_kvlen
        cur_sub_out_seq_len = half_sub_out_seq_len

    return cur_q, cur_k, cur_v, cur_attn_mask, (cur_seq_qlen, cur_seq_kvlen, cur_sub_out_seq_len)


def tnd_backward_fetch(
    q_block_id,
    kv_block_id,
    q,
    cur_k,
    cur_v,
    attn_out,
    dout,
    softmax_values,
    seq_lens,
    index_values,
    attn_mask=None,
):
    # fetch backward output
    actual_seq_qlen, actual_seq_kvlen, half_actual_seq_kvlen, half_actual_seq_qlen = seq_lens
    softmax_max, softmax_sum, half_softmax_max, half_softmax_sum = softmax_values
    q_index, kv_index = index_values
    cur_attn_mask = None
    if q_block_id >= kv_block_id:
        if q_block_id == kv_block_id:
            cur_attn_mask = attn_mask
            cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, actual_seq_kvlen
        else:
            cur_k, cur_v = [torch.index_select(x, 0, kv_index) for x in [cur_k, cur_v]]
            cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, half_actual_seq_kvlen

        cur_q, cur_attn_out, cur_dout = q, attn_out, dout
        cur_softmax_max, cur_softmax_sum = softmax_max, softmax_sum
    else:
        cur_q, cur_attn_out, cur_dout = [
            torch.index_select(x, 0, q_index) for x in [q, attn_out, dout]
        ]
        cur_softmax_max, cur_softmax_sum = half_softmax_max, half_softmax_sum
        cur_seq_qlen, cur_seq_kvlen = half_actual_seq_qlen, actual_seq_kvlen

    return (
        (cur_q, cur_k, cur_v),
        cur_attn_out,
        cur_dout,
        (cur_softmax_max, cur_softmax_sum),
        cur_attn_mask,
        (cur_seq_qlen, cur_seq_kvlen),
    )


def causal_backward_fetch(
    q_block_id,
    kv_block_id,
    q,
    cur_k,
    cur_v,
    attn_out,
    dout,
    softmax_max,
    softmax_sum,
    attn_mask=None,
):
    cur_attn_mask = None
    if q_block_id >= kv_block_id:
        # [b, n, 2, s, 8] -> [b, n, 2s, 8]
        cur_softmax_max = softmax_max.view(
            softmax_max.shape[0], softmax_max.shape[1], -1, softmax_max.shape[-1]
        )
        cur_softmax_sum = softmax_sum.view(
            softmax_sum.shape[0], softmax_sum.shape[1], -1, softmax_sum.shape[-1]
        )
        # [2, s, b, h] -> [2s, b, h]
        cur_q, cur_attn_out, cur_dout = [x.view(-1, *x.shape[2:]) for x in [q, attn_out, dout]]
        if q_block_id == kv_block_id:
            cur_attn_mask = attn_mask
            # [2, s, b, h] -> [2s, b, h]
            (
                cur_k,
                cur_v,
            ) = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
        else:
            cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
    else:
        # [2, s, b, h] -> [2s, b, h]
        cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
        # only q[1] attn_out[1] and dout[1] need to be calculated
        cur_q, cur_attn_out, cur_dout = [x[1] for x in [q, attn_out, dout]]
        cur_softmax_max, cur_softmax_sum = [x[:, :, 1, :, :] for x in [softmax_max, softmax_sum]]

    return (
        cur_q,
        cur_k,
        cur_v,
        cur_attn_out,
        cur_dout,
        cur_softmax_max,
        cur_softmax_sum,
        cur_attn_mask,
    )


def tnd_grad_update(q_block_id, kv_block_id, cur_attn_grads, global_attn_grads, q_index, kv_index):
    cur_dq, cur_dk, cur_dv = cur_attn_grads
    dq, dk, dv = global_attn_grads
    if q_block_id == kv_block_id:
        dq.add_(cur_dq)
        dk.add_(cur_dk)
        dv.add_(cur_dv)
    elif q_block_id > kv_block_id:
        dq.add_(cur_dq)
        dk.index_add_(0, kv_index, cur_dk)
        dv.index_add_(0, kv_index, cur_dv)
    else:
        dq.index_add_(0, q_index, cur_dq)
        dk.add_(cur_dk)
        dv.add_(cur_dv)

    return dq, dk, dv


def causal_grad_update(q_block_id, kv_block_id, cur_dq, cur_dk, cur_dv, dq, dk, dv):
    if q_block_id == kv_block_id:
        cur_dq = cur_dq.view(dq.shape)
        cur_dk = cur_dk.view(dk.shape)
        cur_dv = cur_dv.view(dv.shape)
        dq.add_(cur_dq)
        dk.add_(cur_dk)
        dv.add_(cur_dv)
    elif q_block_id > kv_block_id:
        cur_dq = cur_dq.view(dq.shape)
        dq.add_(cur_dq)
        dk[0].add_(cur_dk)
        dv[0].add_(cur_dv)
    else:
        dq[1].add_(cur_dq)
        cur_dk = cur_dk.view(dk.shape)  # [2s, b, h] -> [2, s, b, h]
        cur_dv = cur_dv.view(dv.shape)
        dk.add_(cur_dk)
        dv.add_(cur_dv)

    return dq, dk, dv


def get_unaligned_cp_shapes(shapes, block_id, next_block_id):
    if shapes is None:
        return None
    unaligned_cp_shapes = [shapes[block_id], shapes[next_block_id]]
    return unaligned_cp_shapes


@lru_cache(maxsize=8)
def compute_qkv_index(seq_lens):
    full_indices = list(range(seq_lens[-1]))
    prev_eod_pos = 0
    kv_indices = []
    q_indices = []
    for eod_pos in seq_lens:
        mid = (eod_pos + prev_eod_pos) // 2
        kv_indices.extend(full_indices[prev_eod_pos:mid])
        q_indices.extend(full_indices[mid:eod_pos])
        prev_eod_pos = eod_pos

    kv_index = torch.tensor(kv_indices, device=torch.npu.current_device())
    q_index = torch.tensor(q_indices, device=torch.npu.current_device())

    return q_index, kv_index


def flash_attention_with_alibi_pse(q_block_id, kv_block_id, cur_qkv, attn_info, s):
    raise NotImplementedError(
        "flash_attention_with_alibi_pse is not yet supported in TransformerEngineNPU. "
        "This function depends on mindspeed ops for PSE (position-specific encoding) "
        "in ring context parallel. Please ensure pse is not used with ring CP."
    )


def flash_attention_with_alibi_pse_grad(
    q_block_id,
    kv_block_id,
    cur_qkv,
    cur_dout,
    cur_attn_out,
    cur_softmax_max,
    cur_softmax_sum,
    attn_grad_info,
    s,
):
    raise NotImplementedError(
        "flash_attention_with_alibi_pse_grad is not yet supported in TransformerEngineNPU. "
        "This function depends on mindspeed ops for PSE (position-specific encoding) "
        "in ring context parallel. Please ensure pse is not used with ring CP."
    )


class RingP2P:
    def __init__(
        self, ring_global_ranks, group, group_for_send_recv_overlap=None, is_backward=False
    ) -> None:
        self.group = group
        self.group_for_send_recv_overlap = group
        if group_for_send_recv_overlap is not None:
            self.group_for_send_recv_overlap = group_for_send_recv_overlap

        global_rank = dist.get_rank()
        ring_rank = ring_global_ranks.index(global_rank)
        ring_size = len(ring_global_ranks)
        self.next = ring_global_ranks[(ring_rank + 1) % ring_size]
        self.prev = ring_global_ranks[(ring_rank + ring_size - 1) % ring_size]
        self.ring_rank = ring_rank
        if is_backward:
            self.next, self.prev = self.prev, self.next

        self.send_recv_ops = []

    def async_send_recv(self, orig_send_tensor, orig_recv_tensor, shapes=None):
        send_tensor, recv_tensor = orig_send_tensor, orig_recv_tensor

        enable_mla = isinstance(orig_send_tensor, (list, tuple))
        if enable_mla:
            if shapes is not None:
                raise ValueError("MLA context parallel does not support uneven shapes yet.")
            if len(orig_send_tensor) != 2 or len(orig_recv_tensor) != 2:
                raise ValueError(
                    f"Expected tensors of length 2 (k,v), got lengths: "
                    f"send={len(orig_send_tensor)}, recv={len(orig_recv_tensor)}"
                )
            k_send, v_send = orig_send_tensor
            k_recv, v_recv = orig_recv_tensor
            if k_send.shape != k_recv.shape or v_send.shape != v_recv.shape:
                raise ValueError(
                    "Shape mismatch in KV tensors:\n"
                    f"  k_send: {k_send.shape} vs k_recv: {k_recv.shape}\n"
                    f"  v_send: {v_send.shape} vs v_recv: {v_recv.shape}"
                )
            k_shape, v_shape = k_send.shape, v_send.shape
            k_numel = k_send.numel()
            send_tensor = torch.cat((k_send.view(-1), v_send.view(-1)), dim=-1)
            recv_tensor = torch.cat((k_recv.view(-1), v_recv.view(-1)), dim=-1)

        if self.ring_rank % 2 == 0:
            if shapes is not None:
                send_tensor_shape_list = list(send_tensor.shape)
                send_tensor_shape_list[-3] = shapes[0]
                send_tensor.resize_(send_tensor_shape_list)
            send_op = dist.isend(send_tensor, self.next, self.group)
            if shapes is not None:
                recv_tensor_shape_list = list(recv_tensor.shape)
                recv_tensor_shape_list[-3] = shapes[1]
                recv_tensor.resize_(recv_tensor_shape_list)
            recv_op = dist.irecv(recv_tensor, self.prev, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(send_op)
            self.send_recv_ops.append(recv_op)
        else:
            if shapes is not None:
                recv_tensor_shape_list = list(recv_tensor.shape)
                recv_tensor_shape_list[-3] = shapes[1]
                recv_tensor.resize_(recv_tensor_shape_list)
            recv_op = dist.irecv(recv_tensor, self.prev, self.group)
            if shapes is not None:
                send_tensor_shape_list = list(send_tensor.shape)
                send_tensor_shape_list[-3] = shapes[0]
                send_tensor.resize_(send_tensor_shape_list)
            send_op = dist.isend(send_tensor, self.next, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(recv_op)
            self.send_recv_ops.append(send_op)

        if enable_mla:
            orig_recv_tensor[0] = recv_tensor[:k_numel].view(*k_shape)
            orig_recv_tensor[1] = recv_tensor[k_numel:].view(*v_shape)

    def wait(self):
        if len(self.send_recv_ops) > 0:
            for op in self.send_recv_ops:
                op.wait()
            self.send_recv_ops = []
            return 1
        else:
            return 0


class ContextParallelKVCache:
    def __init__(self, cache_policy, outer_data, inner_data, k, v) -> None:
        self.outer_size, self.outer_ring_p2p = outer_data
        self.inner_size, self.inner_ring_p2p = inner_data
        self.cache_policy = cache_policy
        self.k = k
        self.v = v
        self.cp_size = self.outer_size * self.inner_size
        self.outer_index = 0

        enable_mla = k[-1].shape[-1] != v[-1].shape[-1]
        if not enable_mla:
            send_data = torch.zeros(
                (2, *self.k[-1].shape), dtype=self.k[-1].dtype, device=self.k[-1].device
            )
            send_data.copy_(torch.cat((self.k[-1].unsqueeze(0), self.v[-1].unsqueeze(0)), dim=0))
            outer_recv_data = send_data.clone()
            inner_recv_data = send_data.clone()
        else:
            send_data = [self.k[-1].clone(), self.v[-1].clone()]
            outer_recv_data = [self.k[-1].clone(), self.v[-1].clone()]
            inner_recv_data = [self.k[-1].clone(), self.v[-1].clone()]
        self.cur_kv, self.outer_next_kv, self.inner_next_kv = (
            send_data,
            outer_recv_data,
            inner_recv_data,
        )

        self.k_out, self.v_out = None, None

    def communicate_outer_ring_kv(self, index, shapes=None) -> None:
        self.outer_index = index

        if index > 0:
            if index == 1 and self.cache_policy == "half":
                self.outer_ring_p2p.wait()
                if self.inner_size > 1:
                    self.cur_kv.copy_(self.outer_next_kv[1])
                    self.outer_next_kv = self.outer_next_kv[1].clone()
                else:
                    self.cur_kv, self.outer_next_kv = self.outer_next_kv, self.cur_kv
                    self.k_out, self.v_out = self.cur_kv[0].clone(), self.cur_kv[1].clone()
                    self.cur_kv = self.cur_kv[1].clone()
                    self.outer_next_kv = self.outer_next_kv[1].clone()
            else:
                self.outer_ring_p2p.wait()
                self.cur_kv, self.outer_next_kv = self.outer_next_kv, self.cur_kv

        is_last_step = index + 1 == self.outer_size
        first_step_with_full_cache = self.cache_policy == "full" and index > 0

        if not first_step_with_full_cache and not is_last_step:
            if isinstance(self.cur_kv, (list, tuple)):
                send_tensor = [x.clone() for x in self.cur_kv]
            else:
                send_tensor = self.cur_kv.clone()
            self.outer_ring_p2p.async_send_recv(send_tensor, self.outer_next_kv, shapes=shapes)

    def communicate_inner_ring_kv(self, index, shapes=None):
        total_index = self.outer_index * self.inner_size + index

        if index > 0:
            if total_index == 1 and self.cache_policy == "half":
                self.inner_ring_p2p.wait()
                self.cur_kv, self.inner_next_kv = self.inner_next_kv, self.cur_kv
                self.k_out, self.v_out = self.cur_kv[0].clone(), self.cur_kv[1].clone()
                self.cur_kv = self.cur_kv[1].clone()
                self.inner_next_kv = self.inner_next_kv[1].clone()
            else:
                self.inner_ring_p2p.wait()
                self.cur_kv, self.inner_next_kv = self.inner_next_kv, self.cur_kv

        is_last_step = index + 1 == self.inner_size
        first_step_with_full_cache = self.cache_policy == "full" and total_index > 0

        if not first_step_with_full_cache and not is_last_step:
            self.inner_ring_p2p.async_send_recv(self.cur_kv, self.inner_next_kv, shapes=shapes)

        cache_index = self.cp_size - total_index - 1
        if self.cache_policy is None:
            self.k_out, self.v_out = self.cur_kv[0], self.cur_kv[1]

        elif self.cache_policy == "half":
            if total_index == 0:
                self.k_out, self.v_out = self.cur_kv[0], self.cur_kv[1]
            elif total_index > 1:
                self.k_out, self.v_out = self.k[cache_index], self.cur_kv

        elif self.cache_policy == "full":
            if total_index <= 1:
                self.k_out, self.v_out = self.cur_kv[0], self.cur_kv[1]
            else:
                self.k_out, self.v_out = self.k[cache_index], self.v[cache_index]

        return self.k_out, self.v_out


def forward_update_without_fused(
    prev_attn_out,
    prev_softmax_max,
    prev_softmax_sum,
    cur_attn_out,
    cur_softmax_max,
    cur_softmax_sum,
    actual_seq_qlen=None,
    layout='SBH',
):
    if layout == 'TND':
        cur_softmax_max = flatten_softmax(cur_softmax_max, actual_seq_qlen)
        cur_softmax_sum = flatten_softmax(cur_softmax_sum, actual_seq_qlen)
        prev_softmax_max = flatten_softmax(prev_softmax_max, actual_seq_qlen)
        prev_softmax_sum = flatten_softmax(prev_softmax_sum, actual_seq_qlen)

    origin_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    if layout == 'SBH':
        n = prev_out_scale.shape[1]
        h = prev_attn_out.shape[-1]
        d = h // n
        prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
        prev_out_scale = rearrange(prev_out_scale, 'b n s d -> s b (n d)').contiguous()
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
        cur_out_scale = rearrange(cur_out_scale, 'b n s d -> s b (n d)').contiguous()
    elif layout == 'TND':
        d = prev_attn_out.shape[-1]
        prev_out_scale = prev_out_scale[..., 0].unsqueeze(2).repeat(1, 1, d)
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(2).repeat(1, 1, d)

    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    if layout == 'TND':
        softmax_max = unflatten_softmax(softmax_max, actual_seq_qlen)
        softmax_sum = unflatten_softmax(softmax_sum, actual_seq_qlen)
    return attn_out, softmax_max, softmax_sum


def forward_update(
    prev_attn_out,
    prev_softmax_max,
    prev_softmax_sum,
    cur_attn_out,
    cur_softmax_max,
    cur_softmax_sum,
    actual_seq_qlen=None,
    layout='SBH',
    use_fused_ring_attention_update=False,
):
    if use_fused_ring_attention_update:
        npu_ring_attention_update = _get_npu_ring_attention_update()
        if layout == "TND":
            cumulative_seq_qlen = [0]
            for seq_len in actual_seq_qlen:
                cumulative_seq_qlen.append(cumulative_seq_qlen[-1] + seq_len)
            actual_seq_qlen = torch.tensor(
                cumulative_seq_qlen, dtype=torch.int64, device=prev_attn_out.device
            )
        return npu_ring_attention_update(
            prev_attn_out,
            prev_softmax_max,
            prev_softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            actual_seq_qlen,
            layout,
        )

    return forward_update_without_fused(
        prev_attn_out,
        prev_softmax_max,
        prev_softmax_sum,
        cur_attn_out,
        cur_softmax_max,
        cur_softmax_sum,
        actual_seq_qlen,
        layout,
    )


def tnd_out_update(
    q_block_id,
    kv_block_id,
    cur_attn_outs,
    global_attn_outs,
    q_index,
    softmax_indices,
    cur_sub_out_seq_len,
    use_fused_ring_attention_update=False,
):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = (
        cur_attn_outs[0],
        cur_attn_outs[1],
        cur_attn_outs[2],
    )
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs

    layout = 'TND'

    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out,
            softmax_max,
            softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            actual_seq_qlen=cur_sub_out_seq_len,
            layout=layout,
            use_fused_ring_attention_update=use_fused_ring_attention_update,
        )
        attn_out, softmax_max, softmax_sum = (
            attn_out_updated,
            softmax_max_updated,
            softmax_sum_updated,
        )
    else:
        n = attn_out.shape[1]
        prev_softmax_max = softmax_max.view(-1, 8)[softmax_indices].view(-1, n, 8)
        prev_softmax_sum = softmax_sum.view(-1, 8)[softmax_indices].view(-1, n, 8)

        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            torch.index_select(attn_out, 0, q_index),
            prev_softmax_max,
            prev_softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            actual_seq_qlen=cur_sub_out_seq_len,
            layout=layout,
            use_fused_ring_attention_update=use_fused_ring_attention_update,
        )
        attn_out.index_copy_(0, q_index, attn_out_updated)
        softmax_max = (
            softmax_max.view(-1, 8)
            .index_copy(0, softmax_indices, softmax_max_updated.view(-1, 8))
            .view(-1, n, 8)
        )
        softmax_sum = (
            softmax_sum.view(-1, 8)
            .index_copy(0, softmax_indices, softmax_sum_updated.view(-1, 8))
            .view(-1, n, 8)
        )

    return [attn_out, softmax_max, softmax_sum, rng_states]


def causal_out_update(
    q_block_id,
    kv_block_id,
    cur_attn_outs,
    global_attn_outs,
    use_fused_ring_attention_update=False,
):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = (
        cur_attn_outs[0],
        cur_attn_outs[1],
        cur_attn_outs[2],
    )
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out,
            softmax_max,
            softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            actual_seq_qlen=None,
            layout=layout,
            use_fused_ring_attention_update=use_fused_ring_attention_update,
        )
        attn_out, softmax_max, softmax_sum = (
            attn_out_updated,
            softmax_max_updated,
            softmax_sum_updated,
        )
    else:
        attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
        softmax_max = softmax_max.view(
            softmax_max.shape[0],
            softmax_max.shape[1],
            2,
            softmax_max.shape[2] // 2,
            softmax_max.shape[-1],
        )
        softmax_sum = softmax_sum.view(
            softmax_sum.shape[0],
            softmax_sum.shape[1],
            2,
            softmax_sum.shape[2] // 2,
            softmax_sum.shape[-1],
        )
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out[1],
            softmax_max[:, :, 1, :, :],
            softmax_sum[:, :, 1, :, :],
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            actual_seq_qlen=None,
            layout=layout,
            use_fused_ring_attention_update=use_fused_ring_attention_update,
        )
        attn_out[1].copy_(attn_out_updated)
        softmax_max[:, :, 1, :, :].copy_(softmax_max_updated)
        softmax_sum[:, :, 1, :, :].copy_(softmax_sum_updated)
        attn_out = attn_out.view(-1, *attn_out.shape[2:])
        softmax_max = softmax_max.view(
            softmax_max.shape[0], softmax_max.shape[1], -1, softmax_max.shape[-1]
        )
        softmax_sum = softmax_sum.view(
            softmax_sum.shape[0], softmax_sum.shape[1], -1, softmax_sum.shape[-1]
        )

    return [attn_out, softmax_max, softmax_sum, rng_states]


def general_out_update(
    q_block_id,
    kv_block_id,
    cur_attn_outs,
    global_attn_outs,
    use_fused_ring_attention_update=False,
):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = (
        cur_attn_outs[0],
        cur_attn_outs[1],
        cur_attn_outs[2],
    )
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])
    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    else:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out,
            softmax_max,
            softmax_sum,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            layout=layout,
            use_fused_ring_attention_update=use_fused_ring_attention_update,
        )
        attn_out, softmax_max, softmax_sum = (
            attn_out_updated,
            softmax_max_updated,
            softmax_sum_updated,
        )

    return [attn_out, softmax_max, softmax_sum, rng_states]


class AttentionStrategy(ABC):
    """Attention strategy base class"""

    @abstractmethod
    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        pass

    @abstractmethod
    def update_out(self, cp_config):
        pass

    @abstractmethod
    def compute_fused_attention_grad(
        self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
    ):
        pass


@dataclass
class AttentionWithRingCpConfig:
    """Config in CP attention"""

    # Config that init from cp_para
    causal: Any = None
    cp_group: Any = None
    cp_size: Any = None
    rank: Any = None
    cp_global_ranks: Any = None
    cp_inner_ranks: Any = None
    cp_outer_ranks: Any = None
    kv_block_id: Any = None
    keep_prob: Any = None
    cp_group_for_send_recv_overlap: Any = None
    cp_group_for_intra_window: Any = None
    cp_group_for_intra_window_send_recv_overlap: Any = None
    cp_in_bnsd: Any = None
    use_fused_ring_attention_update: Any = None
    cache_policy: Any = None
    pse: Any = None
    pse_type: Any = None

    # Config that compute derived
    actual_seq_qlen: Any = None
    actual_seq_kvlen: Any = None
    is_eod_reset: Any = None
    inner_size: Any = None
    outer_size: Any = None
    rng_states: Any = None

    # Temporary config
    attn_outs: Any = None
    cur_k: Any = None
    cur_sub_out_seq_len: Any = None
    cur_v: Any = None
    fetch_ptrs: Any = None
    q_block_id: Any = None
    q_index: Any = None
    kv_index: Any = None
    softmax_indices: Any = None
    n: Any = None
    q: Any = None
    softmax_scale: Any = None
    half_actual_seq_qlen: Any = None
    half_actual_seq_kvlen: Any = None
    half_sub_out_seq_len: Any = None
    dout: Any = None
    softmax_max: Any = None
    softmax_sum: Any = None
    attn_out: Any = None
    half_softmax_max: Any = None
    half_softmax_sum: Any = None

    inner_ring: Any = None
    outer_ring: Any = None
    intra_kv_comm: Any = None
    intra_dkv_comm: Any = None
    inter_kv_comm: Any = None
    inter_dkv_comm: Any = None

    @classmethod
    def init_from_para(cls, cp_para):
        # Config init from cp_para
        return cls(
            causal=cp_para['causal'],
            cp_group=cp_para.get("cp_group"),
            cp_size=cp_para.get("cp_size"),
            rank=cp_para.get("rank"),
            cp_global_ranks=cp_para.get("cp_global_ranks"),
            cp_inner_ranks=cp_para.get("cp_inner_ranks", [torch.distributed.get_rank()]),
            cp_group_for_send_recv_overlap=cp_para.get("cp_group_for_send_recv_overlap"),
            cp_group_for_intra_window=cp_para.get('cp_group_for_intra_window'),
            cp_group_for_intra_window_send_recv_overlap=cp_para.get(
                'cp_group_for_intra_window_send_recv_overlap'
            ),
            cp_in_bnsd=cp_para.get('cp_in_bnsd'),
            use_fused_ring_attention_update=cp_para.get('use_fused_ring_attention_update', False),
            cache_policy=cp_para.get("cache_policy"),
            pse=cp_para.get("pse"),
            pse_type=cp_para.get("pse_type"),
        )

    def get_backward_config(self):
        # Get the config need use in backward
        return AttentionWithRingCpConfig(
            # Config need use in backward
            causal=self.causal,
            cp_group=self.cp_group,
            cp_size=self.cp_size,
            rank=self.rank,
            cp_global_ranks=self.cp_global_ranks,
            cp_inner_ranks=self.cp_inner_ranks,
            cp_outer_ranks=self.cp_outer_ranks,
            kv_block_id=self.kv_block_id,
            keep_prob=self.keep_prob,
            rng_states=self.rng_states,
            pse=self.pse,
            pse_type=self.pse_type,
            cp_group_for_send_recv_overlap=self.cp_group_for_send_recv_overlap,
            cp_group_for_intra_window=self.cp_group_for_intra_window,
            cp_group_for_intra_window_send_recv_overlap=self.cp_group_for_intra_window_send_recv_overlap,
            actual_seq_qlen=self.actual_seq_qlen,
            actual_seq_kvlen=self.actual_seq_kvlen,
            is_eod_reset=self.is_eod_reset,
            cp_in_bnsd=self.cp_in_bnsd,
            cache_policy=self.cache_policy,
            # Config not need use in backward
            inner_size=None,
            outer_size=None,
            attn_outs=None,
            cur_k=None,
            cur_sub_out_seq_len=None,
            cur_v=None,
            fetch_ptrs=None,
            q_block_id=None,
            q_index=None,
            kv_index=None,
            softmax_indices=None,
            n=None,
            q=None,
            softmax_scale=None,
            half_actual_seq_qlen=None,
            half_actual_seq_kvlen=None,
            half_sub_out_seq_len=None,
            dout=None,
            softmax_max=None,
            softmax_sum=None,
            attn_out=None,
            half_softmax_max=None,
            half_softmax_sum=None,
            inner_ring=None,
            outer_ring=None,
            intra_kv_comm=None,
            intra_dkv_comm=None,
            inter_kv_comm=None,
            inter_dkv_comm=None,
        )


class AttnFuncWithCPAndRing(torch.autograd.Function):
    """Attention implementation with context parallelism"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        n,
        attn_mask=None,
        qkv_format: str = "thd",
        attn_mask_type=None,
        dropout_p=0.0,
        softmax_scale=None,
        deterministic=None,
        cp_group=None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[torch.Tensor] = None,
        max_seqlen_kv: Optional[torch.Tensor] = None,
        cp_para=None,
        cp_config=None,
        shapes=None,
    ):
        enable_mla = k.shape[-1] != v.shape[-1]
        ctx.enable_mla = enable_mla
        cp_config.keep_prob = 1.0 - dropout_p
        inner_ring = cp_config.inner_ring
        outer_ring = cp_config.outer_ring
        cp_config.is_eod_reset = (cu_seqlens_q is not None) and (cu_seqlens_kv is not None)

        # In verl, cu_seqlens_q_padded is used for CP setting.
        # cp_para may be a dict or an object; support both access patterns.
        cu_seqlens_q_padded = None
        q_index = None
        kv_index = None
        if isinstance(cp_para, dict):
            cu_seqlens_q_padded = cp_para.get('cu_seqlens_q_padded', None)
        elif hasattr(cp_para, 'cu_seqlens_q_padded'):
            cu_seqlens_q_padded = cp_para.cu_seqlens_q_padded

        if cu_seqlens_q_padded is not None:
            cu_seqlens_q_padded_div_cp = cu_seqlens_q_padded // cp_config.cp_size
            q_index, kv_index = compute_qkv_index(tuple(cu_seqlens_q_padded_div_cp))
            cp_config.actual_seq_kvlen = (
                cu_seqlens_q_padded_div_cp.tolist() if max_seqlen_kv else None
            )
            cp_config.actual_seq_qlen = (
                cu_seqlens_q_padded_div_cp.tolist() if max_seqlen_q else None
            )
        else:
            if cp_config.is_eod_reset:
                q_index, kv_index = None, None
                if qkv_format == 'thd':
                    # Accept both rank-local endpoints (MindSpeed EOD-reset
                    # convention) and global ones over the CP group (Megatron
                    # native SFT / upstream GPU-TE convention), normalized to
                    # the rank-local lists the ring kernels consume.
                    cu_seqlens_q, cu_seqlens_kv = _to_rank_local_cu_seqlens(
                        cu_seqlens_q, cu_seqlens_kv, q, k, cp_config.cp_size
                    )
                    q_index, kv_index = compute_qkv_index(tuple(cu_seqlens_q))
                cp_config.actual_seq_kvlen = cu_seqlens_kv if max_seqlen_kv else None
                cp_config.actual_seq_qlen = cu_seqlens_q if max_seqlen_q else None

        if softmax_scale is None:
            # THD or SBH
            head_dim = (
                q.shape[-1] if max_seqlen_q is not None and cp_para['causal'] else q.shape[-1] // n
            )
            softmax_scale = head_dim ** (-0.5)
        if cp_config.causal and attn_mask is None:
            attn_mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        if cp_config.causal:
            if cp_config.is_eod_reset:
                # only first half of each sub sequence KV block need to be calculated when i <= rank
                cp_config.kv_index = kv_index
                # only last half of each sub sequence q block need to be calculated when i > rank
                cp_config.q_index = q_index

                sub_out_seq_len = (
                    torch.tensor([0] + cp_config.actual_seq_qlen)[1:]
                    - torch.tensor([0] + cp_config.actual_seq_qlen)[:-1]
                ).tolist()
                seq_lens = (cp_config.actual_seq_qlen, cp_config.actual_seq_kvlen, sub_out_seq_len)
                half_seq_lens = [[x // 2 for x in lst] for lst in seq_lens]
                cp_config.fetch_ptrs = (
                    seq_lens,
                    half_seq_lens,
                    cp_config.q_index,
                    cp_config.kv_index,
                )

                cp_config.softmax_indices = get_selection_indices_for_tnd_softmax_update(
                    q.shape[0], q.shape[1], tuple(half_seq_lens[2])
                ).to(q.device)
            else:
                # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

        if enable_mla:
            cur_kv = [k, v]
            next_kv = [torch.empty_like(k), torch.empty_like(v)]
            next_round_kv = [torch.empty_like(k), torch.empty_like(v)]
        else:
            cur_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, 2, s, b, h]
            next_kv = torch.empty_like(cur_kv)
            next_round_kv = torch.empty_like(cur_kv)

        attn_out, softmax_max, softmax_sum = None, None, None
        # (seed, offset, numels) for dropout mask
        cp_config.rng_states = [[0, 0, 0] for _ in range(cp_config.cp_size)]
        cp_config.global_attn_outs = [attn_out, softmax_max, softmax_sum, cp_config.rng_states]
        cp_config.q_block_id, cp_config.kv_block_id, kv_block_id_outer = (
            cp_config.rank,
            cp_config.rank,
            cp_config.rank,
        )

        cache_manager = KVCacheManager(cp_config.cache_policy)

        # Not support general EoD ring.
        # In mindspeed, 'TND' equals 'thd'.
        if qkv_format == 'thd':
            if cp_config.causal:
                attention_strategy = CausalEodAttentionStrategy()
            else:
                raise NotImplementedError('General TND(THD) not support for TE Ring.')

        elif qkv_format == 'sbhd':
            if cp_config.causal:
                attention_strategy = CausalRegularAttentionStrategy()  # SBH
            else:
                attention_strategy = GeneralAttentionStrategy()  # SBH

        else:
            raise ValueError(f"Unsupported qkv_format: {qkv_format}")

        for j in range(cp_config.outer_size):
            if getattr(_get_args(), "prof_file", False):
                activation_func_1 = torch.nn.Hardshrink()
                v = activation_func_1(v)
            cp_config.kv_block_id = kv_block_id_outer
            kv_block_offset = (cp_config.kv_block_id // cp_config.inner_size) * cp_config.inner_size
            if j < cp_config.outer_size - 1:
                next_kv_block_id_outer = (
                    kv_block_id_outer + cp_config.cp_size - cp_config.inner_size
                ) % cp_config.cp_size
                outer_ring.async_send_recv(
                    cur_kv,
                    next_round_kv,
                    shapes=get_unaligned_cp_shapes(
                        shapes, kv_block_id_outer, next_kv_block_id_outer
                    ),
                )
            for i in range(cp_config.inner_size):
                # wait until KV is received from recv_src
                if i < cp_config.inner_size - 1:
                    next_kv_block_id = (
                        cp_config.kv_block_id + cp_config.inner_size - 1
                    ) % cp_config.inner_size + kv_block_offset
                    inner_ring.async_send_recv(
                        cur_kv,
                        next_kv,
                        shapes=get_unaligned_cp_shapes(
                            shapes, cp_config.kv_block_id, next_kv_block_id
                        ),
                    )

                cp_config.cur_k, cp_config.cur_v = cur_kv[0], cur_kv[1]  # [2, s, b, h]

                # cache kv or k
                if j * cp_config.inner_size + i + 2 != cp_config.cp_size:
                    cache_manager.update_cache(cur_kv)
                if cp_config.causal:
                    cp_config.attn_outs = None
                    cp_config.cur_sub_out_seq_len = None
                cp_config.attn_outs, cp_config.cur_sub_out_seq_len = (
                    attention_strategy.compute_fused_attention(
                        attn_mask, n, q, softmax_scale, cp_config
                    )
                )
                cp_config.global_attn_outs = attention_strategy.update_out(cp_config)

                if inner_ring.wait():
                    cur_kv, next_kv = next_kv, cur_kv  # double buffer
                    cp_config.kv_block_id = (
                        cp_config.kv_block_id + cp_config.inner_size - 1
                    ) % cp_config.inner_size + kv_block_offset

            if outer_ring.wait():
                cur_kv, next_round_kv = next_round_kv, cur_kv  # double buffer
                kv_block_id_outer = (
                    kv_block_id_outer + cp_config.cp_size - cp_config.inner_size
                ) % cp_config.cp_size

            if getattr(_get_args(), "prof_file", False):
                v = activation_func_1(v)

        attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]

        attn_out, softmax_max, softmax_sum, cp_config.rng_states = cp_config.global_attn_outs
        if cp_config.causal and not cp_config.is_eod_reset:
            k_stack, v_stack, q = cache_manager.get_cache_causal_regular(cur_kv, q)
        else:
            k_stack, v_stack = cache_manager.get_cache(cur_kv)

        ctx.intra_kv_comm = cp_config.intra_kv_comm
        ctx.intra_dkv_comm = cp_config.intra_dkv_comm
        ctx.inter_kv_comm = cp_config.inter_kv_comm
        ctx.inter_dkv_comm = cp_config.inter_dkv_comm
        ctx.save_for_backward(q, k_stack, v_stack, *attn_mask, attn_out, softmax_max, softmax_sum)
        ctx.n = n
        ctx.softmax_scale = softmax_scale
        ctx.cp_dkv_outer_ranks = cp_para.get('cp_dkv_outer_ranks', cp_config.cp_global_ranks)
        ctx.shapes = shapes
        ctx.cp_config = cp_config.get_backward_config()
        ctx.qkv_format = qkv_format
        if cp_config.causal and cp_config.is_eod_reset:
            # Config need use in backward when condition is causal and eod
            ctx.cp_config.q_index = cp_config.q_index
            ctx.cp_config.kv_index = cp_config.kv_index
            ctx.cp_config.half_actual_seq_qlen = half_seq_lens[0]
            ctx.cp_config.half_actual_seq_kvlen = half_seq_lens[1]
            ctx.cp_config.half_sub_out_seq_len = half_seq_lens[2]
            ctx.cp_config.sub_out_seq_len = sub_out_seq_len
            ctx.cp_config.softmax_indices = cp_config.softmax_indices
            return attn_out

        return attn_out

    @staticmethod
    def backward(ctx, dout):
        cp_config = ctx.cp_config
        q, k_stack, v_stack, *attn_mask, attn_out, cp_config.softmax_max, cp_config.softmax_sum = (
            ctx.saved_tensors
        )
        attn_mask = attn_mask[0] if len(attn_mask) == 1 else attn_mask

        n = ctx.n
        shapes = ctx.shapes
        softmax_scale = ctx.softmax_scale
        qkv_format = ctx.qkv_format

        # Reversed order of forward
        inner_size = len(cp_config.cp_inner_ranks)
        outer_size = len(cp_config.cp_outer_ranks)

        intra_kv_comm = ctx.intra_kv_comm
        intra_dkv_comm = ctx.intra_dkv_comm
        inter_kv_comm = ctx.inter_kv_comm
        inter_dkv_comm = ctx.inter_dkv_comm

        if cp_config.causal:
            if cp_config.is_eod_reset:
                cp_config.half_softmax_max = cp_config.softmax_max.view(-1, 8)[
                    cp_config.softmax_indices
                ].view(-1, n, 8)
                cp_config.half_softmax_sum = cp_config.softmax_sum.view(-1, 8)[
                    cp_config.softmax_indices
                ].view(-1, n, 8)
            else:
                # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1], [2s, b, h] -> [2, s, b, h]
                q, attn_out, dout = [
                    x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, attn_out, dout]
                ]
                k_stack = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in k_stack]
                v_stack = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in v_stack]
                # [b, n, 2s, 8] -> [b, n, 2, s, 8]
                cp_config.softmax_max = cp_config.softmax_max.view(
                    cp_config.softmax_max.shape[0],
                    cp_config.softmax_max.shape[1],
                    2,
                    cp_config.softmax_max.shape[2] // 2,
                    cp_config.softmax_max.shape[-1],
                )
                cp_config.softmax_sum = cp_config.softmax_sum.view(
                    cp_config.softmax_sum.shape[0],
                    cp_config.softmax_sum.shape[1],
                    2,
                    cp_config.softmax_sum.shape[2] // 2,
                    cp_config.softmax_sum.shape[-1],
                )

        cp_config.q_block_id, cp_config.kv_block_id, kv_block_id_outer = (
            cp_config.rank,
            cp_config.kv_block_id,
            cp_config.kv_block_id,
        )
        if shapes:
            cur_shapes_list = list(k_stack[-1].shape)
            cur_shapes_list[-3] = shapes[cp_config.kv_block_id]
            cur_dkv = torch.zeros(
                (2, *cur_shapes_list), dtype=k_stack[-1].dtype, device=k_stack[-1].device
            )
            next_dkv = cur_dkv.clone()
            next_round_dkv = cur_dkv.clone()
        elif not ctx.enable_mla:
            cur_dkv = torch.zeros(
                (2, *k_stack[-1].shape), dtype=k_stack[-1].dtype, device=k_stack[-1].device
            )
            next_dkv = cur_dkv.clone()
            next_round_dkv = cur_dkv.clone()
        else:
            k_tmp, v_tmp = k_stack[-1], v_stack[-1]
            cur_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]
            next_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]
            next_round_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]

        if getattr(_get_args(), "prof_file", False):
            activation_func_1 = torch.nn.Hardshrink()
            q = activation_func_1(q)

        outer_data = (outer_size, inter_kv_comm)
        inner_data = (inner_size, intra_kv_comm)
        cp_kv_cache = ContextParallelKVCache(
            cp_config.cache_policy, outer_data, inner_data, k_stack, v_stack
        )

        dq = torch.zeros_like(q)  # [2, s, b, h]
        for j in range(outer_size):
            cp_config.kv_block_id = kv_block_id_outer
            kv_block_offset = (cp_config.kv_block_id // inner_size) * inner_size

            next_kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_config.cp_size
            cp_kv_cache.communicate_outer_ring_kv(
                j,
                shapes=get_unaligned_cp_shapes(
                    shapes, cp_config.kv_block_id, next_kv_block_id_outer
                ),
            )

            for i in range(inner_size):
                next_kv_block_id = (cp_config.kv_block_id + 1) % inner_size + kv_block_offset
                cp_config.cur_k, cp_config.cur_v = cp_kv_cache.communicate_inner_ring_kv(
                    i, get_unaligned_cp_shapes(shapes, cp_config.kv_block_id, next_kv_block_id)
                )

                dq_step, dk_step, dv_step = AttnFuncWithCPAndRing.backward_step_helper(
                    attn_mask, n, q, softmax_scale, attn_out, dout, cp_config, qkv_format
                )

                if i == 0 and j > 0:  # receive dk dv from last window
                    inter_dkv_comm.wait()
                    cur_dkv, next_round_dkv = next_round_dkv, cur_dkv
                elif i > 0:  # receive dk dv from last step
                    intra_dkv_comm.wait()
                    cur_dkv, next_dkv = next_dkv, cur_dkv

                dk, dv = cur_dkv[0], cur_dkv[1]
                # update qkv grades
                if cp_config.is_eod_reset and cp_config.causal:
                    tnd_grad_update(
                        cp_config.q_block_id,
                        cp_config.kv_block_id,
                        (dq_step, dk_step, dv_step),
                        (dq, dk, dv),
                        cp_config.q_index,
                        cp_config.kv_index,
                    )
                elif cp_config.causal:
                    causal_grad_update(
                        cp_config.q_block_id,
                        cp_config.kv_block_id,
                        dq_step,
                        dk_step,
                        dv_step,
                        dq,
                        dk,
                        dv,
                    )
                else:
                    dq.add_(dq_step)
                    dk.add_(dk_step)
                    dv.add_(dv_step)

                next_kv_block_id = (cp_config.kv_block_id + 1) % inner_size + kv_block_offset
                if i + 1 != inner_size:
                    intra_dkv_comm.async_send_recv(
                        cur_dkv,
                        next_dkv,
                        shapes=get_unaligned_cp_shapes(
                            shapes, cp_config.kv_block_id, next_kv_block_id
                        ),
                    )

                cp_config.kv_block_id = next_kv_block_id

            if intra_dkv_comm.wait():
                cur_dkv, next_dkv = next_dkv, cur_dkv

            next_kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_config.cp_size
            if j + 1 != outer_size:
                inter_dkv_comm.async_send_recv(
                    cur_dkv,
                    next_round_dkv,
                    shapes=get_unaligned_cp_shapes(
                        shapes, cp_config.kv_block_id, next_kv_block_id_outer
                    ),
                )

            kv_block_id_outer = next_kv_block_id_outer

        if inter_dkv_comm.wait():
            cur_dkv, next_round_dkv = next_round_dkv, cur_dkv

        dk, dv = cur_dkv[0], cur_dkv[1]

        if getattr(_get_args(), "prof_file", False):
            dq = activation_func_1(dq)

        # [2, s, b, h] -> [2s, b, h]
        if cp_config.causal and not cp_config.is_eod_reset:
            dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @classmethod
    def backward_step_helper(
        cls, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config, qkv_format
    ):
        if qkv_format == 'thd':
            if cp_config.causal:
                attention_strategy = CausalEodAttentionStrategy()
            else:
                raise NotImplementedError('General TND(THD) not support for TE Ring.')

        elif qkv_format == 'sbhd':
            if cp_config.causal:
                attention_strategy = CausalRegularAttentionStrategy()  # SBH
            else:
                attention_strategy = GeneralAttentionStrategy()  # SBH

        else:
            raise ValueError(f"Unsupported qkv_format: {qkv_format}")

        attn_grad_outs = attention_strategy.compute_fused_attention_grad(
            attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
        )
        return attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

    @classmethod
    def compute_mask(cls, actual_seq_qlen, actual_seq_kvlen, q_block_id, kv_block_id, attn_mask):
        if actual_seq_qlen:
            seq_len = actual_seq_qlen[-1] // AttnFuncWithCPAndRing.batch_size
            actual_seq_qlen = _get_batch_index(actual_seq_qlen, seq_len)
            actual_seq_kvlen = _get_batch_index(actual_seq_kvlen, seq_len)
            block_size = cls.block_size
            actual_seq_qlen = [[0] + lst for lst in actual_seq_qlen]
            sub_seq_qlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_qlen]
            sub_seq_qid = torch.stack(
                [torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_qlen]
            ).npu()  # B S

            this_ids = sub_seq_qid[:, q_block_id * block_size : (q_block_id + 1) * block_size].npu()
            this_tile = this_ids.unsqueeze(dim=2)  # B S 1

            actual_seq_kvlen = [[0] + lst for lst in actual_seq_kvlen]
            sub_seq_kvlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_kvlen]
            sub_seq_kvid = torch.stack(
                [torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_kvlen]
            ).npu()  # B S
            other_ids = sub_seq_kvid[
                :, kv_block_id * block_size : (kv_block_id + 1) * block_size
            ].npu()
            other_tile = other_ids.unsqueeze(dim=1)  # B 1 S

            mask = this_tile == other_tile  # B S S
            if kv_block_id > q_block_id:
                mask = torch.zeros_like(mask)
            elif kv_block_id == q_block_id:
                mask = torch.tril(mask)

            return torch.logical_not(mask).unsqueeze(dim=1).npu()  # B 1 S S
        else:
            return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None


class CausalRegularAttentionStrategy(AttentionStrategy):
    """Causal attention strategy when using pse or layout is SBH and BNSD"""

    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        if cp_config.pse is None:
            # Causal attention when layout is SBH and BNSD
            return self._standard_causal_attention(attn_mask, n, q, softmax_scale, cp_config)
        else:
            # Causal attention when using pse
            return self._pse_causal_attention(attn_mask, n, q, softmax_scale, cp_config)

    def _standard_causal_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # Causal attention when layout is SBH and BNSD
        cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            attn_mask,
        )
        layout = "SBH"
        pre_tockens_value = cur_k.shape[0]
        if cp_config.cp_in_bnsd:
            cur_q = rearrange(cur_q, 's b (h d) -> b h s d', h=n).contiguous()
            kv_n = cur_v.shape[2] // cur_q.shape[3]
            cur_k, cur_v = [
                rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous() for x in [cur_k, cur_v]
            ]
            layout = "BNSD"
            pre_tockens_value = cur_k.shape[2]

        # todo:from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention, npu_fusion_attention_grad
        attn_outs = torch_npu.npu_fusion_attention(
            cur_q,
            cur_k,
            cur_v,
            n,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=0 if cur_attn_mask is not None else pre_tockens_value,
            keep_prob=cp_config.keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
        )
        if cp_config.cp_in_bnsd:
            attn_outs = (
                rearrange(attn_outs[0], 'b h s d -> s b (h d)').contiguous(),
                attn_outs[1],
                attn_outs[2],
            )
        return attn_outs, cp_config.cur_sub_out_seq_len

    def _pse_causal_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # Causal attention when using pse
        cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            attn_mask,
        )
        q_index_list = [cp_config.q_block_id, cp_config.cp_size * 2 - 1 - cp_config.q_block_id]
        kv_index_list = [cp_config.kv_block_id, cp_config.cp_size * 2 - 1 - cp_config.kv_block_id]
        attn_info = [
            n,
            cp_config.pse,
            cp_config.pse_type,
            cur_attn_mask,
            softmax_scale,
            cp_config.keep_prob,
            q_index_list,
            kv_index_list,
        ]
        s = q.shape[1]
        attn_outs = flash_attention_with_alibi_pse(
            cp_config.q_block_id, cp_config.kv_block_id, (cur_q, cur_k, cur_v), attn_info, s
        )
        return attn_outs, cp_config.cur_sub_out_seq_len

    def update_out(self, cp_config):
        return causal_out_update(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            cp_config.attn_outs,
            cp_config.global_attn_outs,
            cp_config.use_fused_ring_attention_update,
        )

    def compute_fused_attention_grad(
        self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
    ):
        if cp_config.pse is None:
            # Causal attention backward when layout is SBH and BNSD
            return self._standard_causal_backward(
                attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
            )
        else:
            # Causal attention backward when using pse
            return self._pse_causal_backward(
                attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
            )

    def _standard_causal_backward(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        # Causal attention backward when layout is SBH and BNSD
        step_inputs = causal_backward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            attn_out,
            dout,
            cp_config.softmax_max,
            cp_config.softmax_sum,
            attn_mask=attn_mask,
        )
        (
            cur_q,
            cur_k,
            cur_v,
            cur_attn_out,
            cur_dout,
            cur_softmax_max,
            cur_softmax_sum,
            cur_attn_mask,
        ) = step_inputs
        layout = "SBH"
        pre_tockens_value = cur_k.shape[0]
        if cp_config.cp_in_bnsd:
            cur_q, cur_dout, cur_attn_out = [
                rearrange(x1, 's b (h d) -> b h s d', h=n).contiguous()
                for x1 in [cur_q, cur_dout, cur_attn_out]
            ]
            kv_n = cur_v.shape[2] // cur_q.shape[3]
            cur_k, cur_v = [
                rearrange(x1, 's b (h d) -> b h s d', h=kv_n).contiguous() for x1 in [cur_k, cur_v]
            ]
            layout = "BNSD"
            pre_tockens_value = cur_k.shape[2]
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q,
            cur_k,
            cur_v,
            cur_dout,
            n,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            softmax_max=cur_softmax_max,
            softmax_sum=cur_softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=0 if cur_attn_mask is not None else pre_tockens_value,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )
        if cp_config.cp_in_bnsd:
            attn_grad_outs = [
                rearrange(x1, 'b h s d -> s b (h d)').contiguous()
                for x1 in [attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]]
            ]
        return attn_grad_outs

    def _pse_causal_backward(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        # Causal attention backward when using pse
        step_inputs = causal_backward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            attn_out,
            dout,
            cp_config.softmax_max,
            cp_config.softmax_sum,
            attn_mask=attn_mask,
        )
        (
            cur_q,
            cur_k,
            cur_v,
            cur_attn_out,
            cur_dout,
            cur_softmax_max,
            cur_softmax_sum,
            cur_attn_mask,
        ) = step_inputs
        q_index_list = [cp_config.q_block_id, cp_config.cp_size * 2 - 1 - cp_config.q_block_id]
        kv_index_list = [cp_config.kv_block_id, cp_config.cp_size * 2 - 1 - cp_config.kv_block_id]
        attn_grad_info = [
            n,
            cp_config.pse,
            cp_config.pse_type,
            cur_attn_mask,
            softmax_scale,
            cp_config.keep_prob,
            cp_config.rng_states,
            q_index_list,
            kv_index_list,
        ]
        s = q.shape[1]
        attn_grad_outs = flash_attention_with_alibi_pse_grad(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            (cur_q, cur_k, cur_v),
            cur_dout,
            cur_attn_out,
            cur_softmax_max,
            cur_softmax_sum,
            attn_grad_info,
            s,
        )
        return attn_grad_outs


class CausalEodAttentionStrategy(AttentionStrategy):
    """Causal eod attention strategy"""

    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # Causal eod attention
        cur_q, cur_k, cur_v, cur_attn_mask, cur_seq_lens = tnd_forward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            cp_config.fetch_ptrs,
            attn_mask,
        )
        cur_seq_qlen, cur_seq_kvlen, cur_sub_out_seq_len = cur_seq_lens
        # flash attention forward
        attn_outs = torch_npu.npu_fusion_attention(
            cur_q,
            cur_k,
            cur_v,
            n,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tockens=cur_k.shape[0],
            next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
            keep_prob=cp_config.keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            actual_seq_qlen=cur_seq_qlen,
            actual_seq_kvlen=cur_seq_kvlen,
        )

        return attn_outs, cur_sub_out_seq_len

    def update_out(self, cp_config):
        return tnd_out_update(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            cp_config.attn_outs,
            cp_config.global_attn_outs,
            cp_config.q_index,
            cp_config.softmax_indices,
            cp_config.cur_sub_out_seq_len,
            cp_config.use_fused_ring_attention_update,
        )

    def compute_fused_attention_grad(
        self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
    ):
        # Causal eod attention backward
        softmax_values = (
            cp_config.softmax_max,
            cp_config.softmax_sum,
            cp_config.half_softmax_max,
            cp_config.half_softmax_sum,
        )
        seq_lens = (
            cp_config.actual_seq_qlen,
            cp_config.actual_seq_kvlen,
            cp_config.half_actual_seq_qlen,
            cp_config.half_actual_seq_kvlen,
        )
        index_values = (cp_config.q_index, cp_config.kv_index)
        step_inputs = tnd_backward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            attn_out,
            dout,
            softmax_values,
            seq_lens,
            index_values,
            attn_mask=attn_mask,
        )
        qkv, cur_attn_out, cur_dout, cur_softmax_values, cur_attn_mask, cur_seq_lens = step_inputs
        cur_q, cur_k, cur_v = qkv
        cur_softmax_max, cur_softmax_sum = cur_softmax_values
        cur_seq_qlen, cur_seq_kvlen = cur_seq_lens
        # flash attention backward
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q,
            cur_k,
            cur_v,
            cur_dout,
            n,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            softmax_max=cur_softmax_max,
            softmax_sum=cur_softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            pre_tockens=cur_k.shape[0],
            next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            actual_seq_qlen=cur_seq_qlen,
            actual_seq_kvlen=cur_seq_kvlen,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )
        return attn_grad_outs


class GeneralEodAttentionStrategy(AttentionStrategy):
    """Causal eod attention strategy"""

    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # Causal eod attention
        cur_q, cur_k, cur_v, cur_attn_mask, cur_seq_lens = tnd_forward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            cp_config.fetch_ptrs,
            attn_mask,
        )
        cur_seq_qlen, cur_seq_kvlen, cur_sub_out_seq_len = cur_seq_lens

        this_mask = AttnFuncWithCPAndRing.compute_mask(
            cp_config.actual_seq_qlen,
            cp_config.actual_seq_kvlen,
            cp_config.q_block_id,
            cp_config.kv_block_id,
            attn_mask,
        )
        # flash attention forward
        attn_outs = torch_npu.npu_fusion_attention(
            cur_q,
            cur_k,
            cur_v,
            n,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=this_mask,
            scale=softmax_scale,
            pre_tockens=0,
            next_tockens=0,
            keep_prob=cp_config.keep_prob,
            sparse_mode=1,
            actual_seq_qlen=cur_seq_qlen,
            actual_seq_kvlen=cur_seq_kvlen,
        )

        return attn_outs, cur_sub_out_seq_len

    def update_out(self, cp_config):
        return tnd_out_update(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            cp_config.attn_outs,
            cp_config.global_attn_outs,
            cp_config.q_index,
            cp_config.softmax_indices,
            cp_config.cur_sub_out_seq_len,
            cp_config.use_fused_ring_attention_update,
        )

    def compute_fused_attention_grad(
        self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
    ):
        # Causal eod attention backward
        softmax_values = (
            cp_config.softmax_max,
            cp_config.softmax_sum,
            cp_config.half_softmax_max,
            cp_config.half_softmax_sum,
        )
        seq_lens = (
            cp_config.actual_seq_qlen,
            cp_config.actual_seq_kvlen,
            cp_config.half_actual_seq_qlen,
            cp_config.half_actual_seq_kvlen,
        )
        index_values = (cp_config.q_index, cp_config.kv_index)
        step_inputs = tnd_backward_fetch(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            q,
            cp_config.cur_k,
            cp_config.cur_v,
            attn_out,
            dout,
            softmax_values,
            seq_lens,
            index_values,
            attn_mask=attn_mask,
        )
        qkv, cur_attn_out, cur_dout, cur_softmax_values, cur_attn_mask, cur_seq_lens = step_inputs

        # General attention backward
        this_mask = AttnFuncWithCPAndRing.compute_mask(
            cp_config.actual_seq_qlen,
            cp_config.actual_seq_kvlen,
            cp_config.q_block_id,
            cp_config.kv_block_id,
            attn_mask,
        )

        cur_q, cur_k, cur_v = qkv
        cur_softmax_max, cur_softmax_sum = cur_softmax_values
        cur_seq_qlen, cur_seq_kvlen = cur_seq_lens
        # flash attention backward
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q,
            cur_k,
            cur_v,
            cur_dout,
            n,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=this_mask,
            softmax_max=cur_softmax_max,
            softmax_sum=cur_softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            pre_tockens=0,
            next_tockens=0,
            sparse_mode=1,
            actual_seq_qlen=cur_seq_qlen,
            actual_seq_kvlen=cur_seq_kvlen,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )
        return attn_grad_outs


class GeneralAttentionStrategy(AttentionStrategy):
    """General attention strategy"""

    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # General attention
        this_mask = AttnFuncWithCPAndRing.compute_mask(
            cp_config.actual_seq_qlen,
            cp_config.actual_seq_kvlen,
            cp_config.q_block_id,
            cp_config.kv_block_id,
            attn_mask,
        )

        layout = "SBH"
        pre_tockens_value = cp_config.cur_k.shape[0]

        cur_q = q
        if cp_config.cp_in_bnsd:
            cur_q = rearrange(q, 's b (h d) -> b h s d', h=n).contiguous()
            kv_n = cp_config.cur_v.shape[2] // cur_q.shape[3]
            cp_config.cur_k, cp_config.cur_v = [
                rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous()
                for x in [cp_config.cur_k, cp_config.cur_v]
            ]
            layout = "BNSD"
            pre_tockens_value = cp_config.cur_k.shape[2]

        attn_outs = torch_npu.npu_fusion_attention(
            cur_q,
            cp_config.cur_k,
            cp_config.cur_v,
            n,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=this_mask,
            scale=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=pre_tockens_value,
            keep_prob=cp_config.keep_prob,
            sparse_mode=1,
        )
        if cp_config.cp_in_bnsd:
            attn_outs = (
                rearrange(attn_outs[0], 'b h s d -> s b (h d)').contiguous(),
                attn_outs[1],
                attn_outs[2],
                attn_outs[3],
                attn_outs[4],
                attn_outs[5],
                attn_outs[6],
            )
        return attn_outs, cp_config.cur_sub_out_seq_len

    def update_out(self, cp_config):
        return general_out_update(
            cp_config.q_block_id,
            cp_config.kv_block_id,
            cp_config.attn_outs,
            cp_config.global_attn_outs,
            cp_config.use_fused_ring_attention_update,
        )

    def compute_fused_attention_grad(
        self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config
    ):
        # General attention backward
        this_mask = AttnFuncWithCPAndRing.compute_mask(
            cp_config.actual_seq_qlen,
            cp_config.actual_seq_kvlen,
            cp_config.q_block_id,
            cp_config.kv_block_id,
            attn_mask,
        )

        layout = "SBH"
        pre_tockens_value = cp_config.cur_k.shape[0]
        cur_q = q
        cur_dout = dout
        cur_attn_out = attn_out
        if cp_config.cp_in_bnsd:
            cur_q, cur_dout, cur_attn_out = [
                rearrange(x, 's b (h d) -> b h s d', h=n).contiguous() for x in [q, dout, attn_out]
            ]
            kv_n = cp_config.cur_v.shape[2] // cur_q.shape[3]
            cp_config.cur_k, cp_config.cur_v = [
                rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous()
                for x in [cp_config.cur_k, cp_config.cur_v]
            ]
            layout = "BNSD"
            pre_tockens_value = cp_config.cur_k.shape[2]

        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q,
            cp_config.cur_k,
            cp_config.cur_v,
            cur_dout,
            n,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=this_mask,
            softmax_max=cp_config.softmax_max,
            softmax_sum=cp_config.softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=pre_tockens_value,
            sparse_mode=1,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )
        if cp_config.cp_in_bnsd:
            attn_grad_outs = [
                rearrange(x, 'b h s d -> s b (h d)').contiguous()
                for x in [attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]]
            ]
        return attn_grad_outs


class KVCacheManager:
    """KV Cache Manager"""

    def __init__(self, cache_policy):
        self.cache_policy = cache_policy
        self.k_cache_list = []
        self.v_cache_list = []

    def update_cache(self, cur_kv):
        # update the cache by kv cache policy
        if self.cache_policy == "full":
            self.k_cache_list.append(cur_kv[0].clone())
            self.v_cache_list.append(cur_kv[1].clone())
        elif self.cache_policy == "half":
            self.k_cache_list.append(cur_kv[0].clone())

    def get_cache(self, cur_kv):
        # get kv cache when config is not causal or eod
        self.k_cache_list = self.k_cache_list if self.k_cache_list else [cur_kv[0].clone()]
        self.v_cache_list = self.v_cache_list if self.v_cache_list else [cur_kv[1].clone()]
        k_stack = torch.stack(self.k_cache_list)
        v_stack = torch.stack(self.v_cache_list)
        return k_stack, v_stack

    def get_cache_causal_regular(self, cur_kv, q):
        # get kv cache when config is causal and is not eod, as causal regular condition
        self.k_cache_list = self.k_cache_list if self.k_cache_list else [cur_kv[0].clone()]
        self.v_cache_list = self.v_cache_list if self.v_cache_list else [cur_kv[1].clone()]
        q = q.view(-1, *q.shape[2:])
        self.k_cache_list = [x.view(-1, *x.shape[2:]) for x in self.k_cache_list]
        self.v_cache_list = [x.view(-1, *x.shape[2:]) for x in self.v_cache_list]
        k_stack = torch.stack(self.k_cache_list)
        v_stack = torch.stack(self.v_cache_list)
        return k_stack, v_stack, q
