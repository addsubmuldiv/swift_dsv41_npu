# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Optional, Tuple

import torch
import torch_npu

from .utils import get_distributed_rank, get_distributed_world_size

_seq_chunk_ids_cache_for_reordering_before_attn = {}
_seq_chunk_ids_cache_for_reordering_after_attn = {}


def gather_along_first_dim(
        inp: torch.Tensor,
        process_group,
        async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return inp, None

    out_shape = list(inp.size())
    out_shape[0] *= world_size

    out = torch.empty(
        out_shape,
        dtype=inp.dtype,
        device=inp.device,
        memory_format=torch.contiguous_format,
    )
    handle = torch.distributed.all_gather_into_tensor(
        out,
        inp.contiguous(),
        group=process_group,
        async_op=async_op,
    )
    return out, handle


def reduce_scatter_along_first_dim(
        inp: torch.Tensor,
        process_group,
        async_op: bool = False
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return inp, None

    dim_size = list(inp.size())
    assert dim_size[0] % world_size == 0, "First dimension must be divisible by world size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=inp.dtype, device=inp.device)
    handle = torch.distributed.reduce_scatter_tensor(
        output, inp.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


def get_seq_chunk_ids_for_reordering_before_attn(cp_size, device):
    global _seq_chunk_ids_cache_for_reordering_before_attn
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_before_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[rank] = 2 * rank
            chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
        _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)]


def get_seq_chunk_ids_for_reordering_after_attn(cp_size, device):
    global _seq_chunk_ids_cache_for_reordering_after_attn
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_after_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[2 * rank] = rank
            chunk_ids[2 * rank + 1] = 2 * cp_size - rank - 1
        _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_after_attn[(cp_size, device)]


def get_kv_seq_info_after_all_gather(
        local_chunk_id, cp_size, max_seqlen_kv, causal
):
    local_chunk_end_idx = (local_chunk_id + 1) * max_seqlen_kv
    full_seq_end_idx = max_seqlen_kv * cp_size * 2

    seq_start_idx = 0

    if causal:
        seq_end_idx = local_chunk_end_idx
    else:
        seq_end_idx = full_seq_end_idx

    return (seq_start_idx, seq_end_idx)


class AttnFuncWithCPAndKVAllGatherForSBHD(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            n_head,
            attention_mask,
            qkv_format,
            attn_mask_type,
            attention_dropout,
            softmax_scale,
            deterministic,
            cp_group,
            cp_stream
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        causal = 'causal' in attn_mask_type
        if not causal:
            raise AssertionError("Only causal mask is supported for AllGatherContextParallel.")

        seq_dim = qkv_format.index("s")
        if not (q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0):
            raise AssertionError("Sequence length per GPU needs to be divisible by 2!")

        q = q.view(2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1):])

        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)

        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        kv_len = k_ag.shape[0] // (2 * cp_size)
        cp_stream.wait_stream(torch.npu.current_stream())

        flash_attn_streams = [torch.npu.current_stream(), cp_stream]

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]
        kv_seq_range_per_step = [None, None]
        out_per_step = [None, None]
        softmax_max = [None, None]
        softmax_sum = [None, None]
        if k.shape[-1] == v.shape[-1]:
            out = torch.empty_like(q)
        else:
            out = torch.empty(*q.shape[:-1], v.shape[-1], dtype=q.dtype, device=q.device)

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.npu.stream(flash_attn_streams[i]):
                    q_ = q.select(seq_dim, i).contiguous()
                    kv_seq_range_per_step[i] = (
                        get_kv_seq_info_after_all_gather(
                            local_seq_chunk_ids[i],
                            cp_size,
                            kv_len,
                            causal,
                        )
                    )
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]

                    attn_outs = torch_npu.npu_fusion_attention(
                        q_,
                        k_,
                        v_,
                        n_head,
                        'SBH',
                        pse=None,
                        padding_mask=None,
                        atten_mask=attention_mask,
                        scale=softmax_scale,
                        pre_tockens=65536,
                        next_tockens=0,
                        keep_prob=1 - attention_dropout,
                        inner_precise=0,
                        sparse_mode=3
                    )
                    out_per_step[i] = attn_outs[0]
                    softmax_max[i] = attn_outs[1]
                    softmax_sum[i] = attn_outs[2]

            if i > 0:
                with torch.npu.stream(flash_attn_streams[i - 1]):
                    out[i - 1].copy_(out_per_step[i - 1])

        torch.npu.current_stream().wait_stream(cp_stream)

        out = out.view(-1, *out.shape[-2:])
        q = q.view(-1, *q.shape[2:])

        ctx.save_for_backward(
            q,
            k,
            v,
            *out_per_step,
            *softmax_max,
            *softmax_sum
        )

        ctx.kv_seq_range_per_step = kv_seq_range_per_step
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.qkv_format = qkv_format
        ctx.n_head = n_head
        ctx.attention_dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.attention_mask = attention_mask
        return out

    @staticmethod
    def backward(ctx, dout):
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v) = saved_tensors[:3]
        out_per_step = saved_tensors[3:5]
        softmax_max = saved_tensors[5:7]
        softmax_sum = saved_tensors[7:9]
        kv_seq_range_per_step = ctx.kv_seq_range_per_step

        seq_dim = ctx.qkv_format.index("s")

        q = q.view(2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1):])
        if k.shape[-1] == v.shape[-1]:
            dout = dout.view(q.shape)
        else:
            dout = dout.view(*q.shape[:-1], v.shape[-1])

        dq = torch.empty_like(q)
        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dv = torch.zeros((v.shape[0] * cp_size, *v.shape[1:]), dtype=v.dtype, device=v.device)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        flash_attn_streams = [torch.npu.current_stream(), ctx.cp_stream]
        dkv_update_done = torch.npu.Event()

        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)

        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        ctx.cp_stream.wait_stream(torch.npu.current_stream())

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.npu.stream(flash_attn_streams[i]):
                    q_ = q.select(seq_dim, i).contiguous()
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i][0],
                        kv_seq_range_per_step[i][1],
                    )
                    k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]

                    out_ = out_per_step[i]
                    dout_ = dout.select(seq_dim, i).contiguous()

                    attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                        q_,
                        k_,
                        v_,
                        dout_,
                        ctx.n_head,
                        "SBH",
                        pse=None,
                        padding_mask=None,
                        atten_mask=ctx.attention_mask,
                        softmax_max=softmax_max[i],
                        softmax_sum=softmax_sum[i],
                        attention_in=out_,
                        scale_value=ctx.softmax_scale,
                        pre_tockens=65536,
                        next_tockens=0,
                        sparse_mode=3,
                        keep_prob=1 - ctx.attention_dropout
                    )
                    dq_per_step[i] = attn_grad_outs[0]
                    dk_per_step[i] = attn_grad_outs[1]
                    dv_per_step[i] = attn_grad_outs[2]

            if i > 0:
                with torch.npu.stream(flash_attn_streams[i - 1]):
                    dq[i - 1].copy_(dq_per_step[i - 1])

                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)
                    seq_start_idx, seq_end_idx = (
                        kv_seq_range_per_step[i - 1][0],
                        kv_seq_range_per_step[i - 1][1],
                    )
                    dk[seq_start_idx:seq_end_idx].add_(dk_per_step[i - 1])
                    dv[seq_start_idx:seq_end_idx].add_(dv_per_step[i - 1])
                    if i < len(local_seq_chunk_ids):
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.npu.current_stream().wait_stream(ctx.cp_stream)

        dk = dk.view(2 * cp_size, -1, *dk.shape[-2:])
        dv = dv.view(2 * cp_size, -1, *dv.shape[-2:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_after_attn(cp_size, dk.device)
        dk = torch.index_select(dk, dim=0, index=chunk_ids_for_kv_ag)
        dv = torch.index_select(dv, dim=0, index=chunk_ids_for_kv_ag)
        dk = dk.view(-1, *dk.shape[-2:])
        dv = dv.view(-1, *dv.shape[-2:])
        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        dq = dq.view(-1, *dq.shape[-2:])

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
        )


def get_cu_seqlens_qkv_before_attn(cu_seqlens_q, cp_size, rank):
    total_length = cu_seqlens_q[-1]
    chunk_size = total_length // cp_size
    start = rank * chunk_size
    end = start + chunk_size

    rank_cu_seqlens_q = []
    rank_cu_seqlens_kv = []
    cur_len_q = 0
    cur_len_kv = 0

    kv_seq_range = [start, end]

    for i in range(len(cu_seqlens_q)):
        if i == 0:
            seq_start = 0
        else:
            seq_start = cu_seqlens_q[i - 1]
        seq_end = cu_seqlens_q[i]

        if seq_end <= start:
            continue
        if seq_start >= end:
            break

        overlap_start_q = max(seq_start, start)
        overlap_end_q = min(seq_end, end)
        overlap_len_q = overlap_end_q - overlap_start_q

        if overlap_len_q > 0:
            cur_len_q += overlap_len_q
            rank_cu_seqlens_q.append(cur_len_q)

        overlap_start_kv = seq_start
        overlap_end_kv = min(seq_end, end)
        overlap_len_kv = overlap_end_kv - overlap_start_kv

        if overlap_len_kv > 0:
            cur_len_kv += overlap_len_kv
            rank_cu_seqlens_kv.append(cur_len_kv)

        if seq_start < start:
            kv_seq_range[0] = seq_start

    return rank_cu_seqlens_q, rank_cu_seqlens_kv, kv_seq_range


class AttnFuncWithCPAndKVAllGatherForTHD(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            n_head,
            attention_mask,
            qkv_format,
            attn_mask_type,
            attention_dropout,
            softmax_scale,
            deterministic,
            cp_group,
            cu_seqlens_q,
            cu_seqlens_kv
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        causal = 'causal' in attn_mask_type
        if not causal:
            raise AssertionError("Only causal mask is supported for AllGatherContextParallel.")

        if cu_seqlens_q != cu_seqlens_kv:
            raise AssertionError("cu_seqlens_q and cu_seqlens_kv must be the same for THD format.")

        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])

        actual_seq_qlen, actual_seq_kvlen, kv_seq_range = get_cu_seqlens_qkv_before_attn(
            cu_seqlens_q,
            cp_size,
            rank
        )
        seq_start_idx, seq_end_idx = (
            kv_seq_range[0],
            kv_seq_range[1],
        )
        k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]

        attn_outs = torch_npu.npu_fusion_attention(
            q,
            k_,
            v_,
            n_head,
            'TND',
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=softmax_scale,
            pre_tockens=65536,
            next_tockens=0,
            keep_prob=1 - attention_dropout,
            inner_precise=0,
            sparse_mode=3,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen
        )
        out = attn_outs[0]
        softmax_max = attn_outs[1]
        softmax_sum = attn_outs[2]

        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum
        )

        ctx.kv_seq_range = kv_seq_range
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.cp_group = cp_group
        ctx.n_head = n_head
        ctx.attention_dropout = attention_dropout
        ctx.softmax_scale = softmax_scale
        ctx.attention_mask = attention_mask
        return out

    @staticmethod
    def backward(ctx, dout):
        cp_size = get_distributed_world_size(ctx.cp_group)

        q, k, v, out, softmax_max, softmax_sum = ctx.saved_tensors
        kv_seq_range = ctx.kv_seq_range
        actual_seq_qlen = ctx.actual_seq_qlen
        actual_seq_kvlen = ctx.actual_seq_kvlen

        dq = torch.empty_like(q)
        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dv = torch.zeros((v.shape[0] * cp_size, *v.shape[1:]), dtype=v.dtype, device=v.device)

        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])

        seq_start_idx, seq_end_idx = (
            kv_seq_range[0],
            kv_seq_range[1],
        )
        k_, v_ = [x[seq_start_idx:seq_end_idx] for x in [k_ag, v_ag]]

        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            q,
            k_,
            v_,
            dout,
            ctx.n_head,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=ctx.attention_mask,
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=out,
            scale_value=ctx.softmax_scale,
            pre_tockens=65536,
            next_tockens=0,
            sparse_mode=3,
            keep_prob=1 - ctx.attention_dropout,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen
        )
        dq.copy_(attn_grad_outs[0])
        dk[seq_start_idx:seq_end_idx].copy_(attn_grad_outs[1])
        dv[seq_start_idx:seq_end_idx].copy_(attn_grad_outs[2])

        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

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
        )
