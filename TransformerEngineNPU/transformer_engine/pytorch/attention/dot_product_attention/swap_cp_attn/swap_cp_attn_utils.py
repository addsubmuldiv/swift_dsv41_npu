# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.


from typing import Optional

import torch
import torch.distributed as dist
import torch_npu


PERMUTE_DIMS1 = {
    4: (1, 2, 3, 0),
    5: (1, 2, 3, 0, 4),
}


PERMUTE_DIMS2 = {
    4: (1, 2, 0, 3),
    5: (1, 2, 0, 3, 4),
}


_COMPRESSED_MASK_SIZE = 2048

_COMPRESSED_CAUSAL_MASK = None


def get_compressed_causal_mask(device="npu"):
    global _COMPRESSED_CAUSAL_MASK
    if _COMPRESSED_CAUSAL_MASK is None:
        _COMPRESSED_CAUSAL_MASK = torch.triu(
            torch.ones(
                (_COMPRESSED_MASK_SIZE, _COMPRESSED_MASK_SIZE),
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
    return _COMPRESSED_CAUSAL_MASK


def to_pinned_cpu(tensor):
    tensor = tensor.detach().contiguous()
    if tensor.device.type == 'cpu':
        return tensor if tensor.is_pinned() else tensor.pin_memory()
    result = torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device='cpu',
        pin_memory=True,
    )
    result.copy_(tensor)
    return result


def select_head_chunk_size(h_q, h_kv, requested_size, CP):
    """Choose the largest native-GQA-compatible chunk no larger than requested."""
    group_size = h_q // h_kv
    for candidate in range(min(requested_size, h_q), 0, -1):
        if h_q % candidate or candidate % CP or candidate % group_size:
            continue
        kv_heads = candidate // group_size
        if kv_heads % CP == 0 or CP % kv_heads == 0:
            return candidate
    raise ValueError("No head chunk size is compatible with the current CP and GQA grouping")


def make_chunked_grad_cpu(
    head_chunk_num,
    seq_chunk_num,
    seq_chunk_size,
    b,
    head_chunk_size,
    d,
    dtype,
):
    return torch.empty(
        head_chunk_num,
        seq_chunk_num,
        seq_chunk_size,
        b,
        head_chunk_size,
        d,
        dtype=dtype,
        device='cpu',
        pin_memory=True,
    )


def materialize_chunked_grad_cpu(chunks):
    head_chunk_num, seq_chunk_num, seq_chunk_size, b, head_chunk_size, d = chunks.shape
    if head_chunk_num == 1:
        return chunks.view(seq_chunk_num * seq_chunk_size, b, head_chunk_size, d)

    output = torch.empty(
        seq_chunk_num * seq_chunk_size,
        b,
        head_chunk_num * head_chunk_size,
        d,
        dtype=chunks.dtype,
        device='cpu',
        pin_memory=True,
    )
    for hc_id in range(head_chunk_num):
        h_start = hc_id * head_chunk_size
        for sc_id in range(seq_chunk_num):
            s_start = sc_id * seq_chunk_size
            output[
                s_start : s_start + seq_chunk_size,
                :,
                h_start : h_start + head_chunk_size,
                :,
            ].copy_(chunks[hc_id, sc_id])
    return output


def _full_a2a_forward_into(
    input_,
    output,
    scratch,
    CP,
    seq_chunk_num,
    seq_chunk_size,
    group,
):
    """Full-sequence head-scatter A2A using fixed send/receive storage."""
    local_s, b, local_h, d = input_.shape
    gathered_h = local_h // CP
    send = output.view(CP, local_s, b, gathered_h, d)
    recv = scratch.view(CP, local_s, b, gathered_h, d)
    send.permute(1, 2, 0, 3, 4).copy_(input_.view(local_s, b, CP, gathered_h, d), non_blocking=True)
    dist.all_to_all_single(recv, send, group=group)
    output.view(seq_chunk_num, CP, seq_chunk_size, b, gathered_h, d).copy_(
        recv.view(CP, seq_chunk_num, seq_chunk_size, b, gathered_h, d).permute(1, 0, 2, 3, 4, 5),
        non_blocking=True,
    )
    return output.view(local_s * CP, b, gathered_h, d)


def _full_a2a_reverse_into(
    grad_full,
    send_buffer,
    recv_buffer,
    output,
    CP,
    seq_chunk_num,
    seq_chunk_size,
    group,
):
    """Reverse full-sequence A2A directly into the final local gradient."""
    _, b, gathered_h, d = grad_full.shape
    local_s = seq_chunk_num * seq_chunk_size
    send = send_buffer.view(CP, local_s, b, gathered_h, d)
    recv = recv_buffer.view(CP, local_s, b, gathered_h, d)
    send.view(CP, seq_chunk_num, seq_chunk_size, b, gathered_h, d).copy_(
        grad_full.view(seq_chunk_num, CP, seq_chunk_size, b, gathered_h, d).permute(
            1, 0, 2, 3, 4, 5
        ),
        non_blocking=True,
    )
    dist.all_to_all_single(recv, send, group=group)
    output.view(local_s, b, CP, gathered_h, d).copy_(recv.permute(1, 2, 0, 3, 4), non_blocking=True)
    return output


def _full_routed_kv_reverse_into(
    grad_full,
    output,
    scratch,
    kv_head_index,
    kv_head_chunk_size,
    CP,
    seq_chunk_num,
    seq_chunk_size,
    group,
):
    """Sum replicated dKV contributors and scatter the local sequence shard."""
    global_s, b, _, d = grad_full.shape
    expanded_numel = global_s * b * kv_head_chunk_size * d
    expanded = scratch[:expanded_numel].view(
        CP,
        seq_chunk_num,
        seq_chunk_size,
        b,
        kv_head_chunk_size,
        d,
    )
    expanded.zero_()
    expanded[..., kv_head_index : kv_head_index + 1, :].copy_(
        grad_full.view(
            seq_chunk_num,
            CP,
            seq_chunk_size,
            b,
            1,
            d,
        ).permute(1, 0, 2, 3, 4, 5)
    )
    dist.reduce_scatter_tensor(output, expanded, group=group)
    return output


def run_full_npu_backward_saved_qkv(
    ctx,
    query_layer,
    key_layer,
    value_layer,
    grad_output,
    saved_group_tensors,
    compute_device,
    current_stream,
):
    """Reuse full Q/K/V captured during forward; only dO needs A2A."""
    cp_group = ctx.cp_group
    CP = dist.get_world_size(cp_group)
    local_s, b, h_q, dim_qk = query_layer.shape
    seq_chunk_num = local_s // ctx.seq_chunk_size
    gathered_s = ctx.seq_chunk_size * CP
    global_s = local_s * CP
    fa_heads = h_q // CP
    full_q, full_k, full_v = ctx.saved_full_qkv
    first_offset = ctx.saved_group_index[(0, 0)] * 3
    first_o = saved_group_tensors[first_offset]
    first_max = saved_group_tensors[first_offset + 1]

    workspace = getattr(ctx, "saved_qkv_backward_workspace", None)
    if workspace is None:
        torch.npu.empty_cache()
        workspace_numel = query_layer.numel()
        if ctx.kv_is_routed:
            workspace_numel = max(
                workspace_numel,
                global_s * b * ctx.kv_head_chunk_size * dim_qk,
            )
        with torch.npu.stream(current_stream):
            d_o_buffer = torch.empty(
                workspace_numel, dtype=query_layer.dtype, device=compute_device
            )
            full_o_buffer = torch.empty_like(d_o_buffer)
            full_max_shape = (*first_max.shape[:2], global_s, *first_max.shape[3:])
            full_lse_numel = (
                first_max.numel() if ctx.full_state_is_packed else first_max.numel() * CP
            )
            lse_buffer = torch.empty(
                2 * full_lse_numel,
                dtype=first_max.dtype,
                device=compute_device,
            )
        workspace = (d_o_buffer, full_o_buffer, lse_buffer, full_max_shape)
        ctx.saved_qkv_backward_workspace = workspace
    d_o_buffer, full_o_buffer, lse_buffer, full_max_shape = workspace
    lse_numel = lse_buffer.numel() // 2
    full_max = lse_buffer[:lse_numel].view(full_max_shape)
    full_sum = lse_buffer[lse_numel:].view(full_max_shape)

    with torch.npu.stream(current_stream):
        q_numel = query_layer.numel()
        full_dout = _full_a2a_forward_into(
            grad_output,
            d_o_buffer[:q_numel],
            full_o_buffer[:q_numel],
            CP,
            seq_chunk_num,
            ctx.seq_chunk_size,
            cp_group,
        )
        full_o = full_o_buffer[:q_numel].view(global_s, *first_o.shape[1:])
        if ctx.full_state_is_packed:
            full_o.copy_(saved_group_tensors[0], non_blocking=True)
            full_max.copy_(saved_group_tensors[1], non_blocking=True)
            full_sum.copy_(saved_group_tensors[2], non_blocking=True)
        else:
            for q_sc_id in range(seq_chunk_num):
                saved_offset = ctx.saved_group_index[(0, q_sc_id)] * 3
                q_start = q_sc_id * gathered_s
                q_end = q_start + gathered_s
                full_o[q_start:q_end].copy_(saved_group_tensors[saved_offset], non_blocking=True)
                full_max[:, :, q_start:q_end].copy_(
                    saved_group_tensors[saved_offset + 1], non_blocking=True
                )
                full_sum[:, :, q_start:q_end].copy_(
                    saved_group_tensors[saved_offset + 2], non_blocking=True
                )
        grad_q_full, grad_k_full, grad_v_full = run_fa_backward(
            full_q,
            full_k,
            full_v,
            full_dout,
            fa_heads,
            True,
            ctx.attention_mask,
            ctx.softmax_scale,
            full_max,
            full_sum,
            full_o,
        )
        grad_query = torch.empty_like(query_layer, device=compute_device)
        grad_key = torch.empty_like(key_layer, device=compute_device)
        grad_value = torch.empty_like(value_layer, device=compute_device)
        _full_a2a_reverse_into(
            grad_q_full,
            d_o_buffer[: grad_q_full.numel()],
            full_o_buffer[: grad_q_full.numel()],
            grad_query,
            CP,
            seq_chunk_num,
            ctx.seq_chunk_size,
            cp_group,
        )
        del grad_q_full
        if ctx.kv_is_routed:
            _full_routed_kv_reverse_into(
                grad_k_full,
                grad_key,
                d_o_buffer,
                ctx.kv_head_index,
                ctx.kv_head_chunk_size,
                CP,
                seq_chunk_num,
                ctx.seq_chunk_size,
                cp_group,
            )
        else:
            _full_a2a_reverse_into(
                grad_k_full,
                d_o_buffer[: grad_k_full.numel()],
                full_o_buffer[: grad_k_full.numel()],
                grad_key,
                CP,
                seq_chunk_num,
                ctx.seq_chunk_size,
                cp_group,
            )
        del grad_k_full
        if ctx.kv_is_routed:
            _full_routed_kv_reverse_into(
                grad_v_full,
                grad_value,
                full_o_buffer,
                ctx.kv_head_index,
                ctx.kv_head_chunk_size,
                CP,
                seq_chunk_num,
                ctx.seq_chunk_size,
                cp_group,
            )
        else:
            _full_a2a_reverse_into(
                grad_v_full,
                d_o_buffer[: grad_v_full.numel()],
                full_o_buffer[: grad_v_full.numel()],
                grad_value,
                CP,
                seq_chunk_num,
                ctx.seq_chunk_size,
                cp_group,
            )
        del grad_v_full
    return grad_query, grad_key, grad_value


# load + d2d
def load_chunk(
    tensor,
    hsd_range,
    stream,
    compute_device,
    group,
    d2d_mode="alltoall",
    d2d_stream=None,
):
    """Slice a local SBHD chunk on the specified stream and run scatter-H/gather-S."""
    if d2d_mode != "alltoall":
        raise ValueError(f"Unsupported d2d_mode: {d2d_mode}")

    h_range, s_range, d_range = hsd_range
    d2d_stream = stream if d2d_stream is None else d2d_stream
    chunk_shape = (
        s_range[1] - s_range[0],
        tensor.shape[1],
        h_range[1] - h_range[0],
        d_range[1] - d_range[0],
    )

    with torch.npu.stream(stream):
        chunk_npu = torch.empty(
            chunk_shape,
            device=compute_device,
            dtype=tensor.dtype,
        )
        chunk_npu.copy_(
            tensor[
                s_range[0] : s_range[1],
                :,
                h_range[0] : h_range[1],
                d_range[0] : d_range[1],
            ],
            non_blocking=True,
        )

    if d2d_stream is not stream:
        d2d_stream.wait_stream(stream)
    with torch.npu.stream(d2d_stream):
        output = all_to_all(
            input_=chunk_npu,
            process_group=group,
            scatter_dim=2,
            gather_dim=0,
        )
        chunk_npu.record_stream(d2d_stream)
    return output, d2d_stream


# Online O update
def online_update_o_chunk(
    o_chunk_this_q,
    o_chunk,
    softmax_max,
    softmax_sum,
    prev_softmax_max=None,
    prev_softmax_sum=None,
):
    """Merge attention results from two KV compute steps in the post-A2A SBH layout."""
    if o_chunk_this_q is None:
        return o_chunk, softmax_max, softmax_sum

    merged_max = torch.maximum(prev_softmax_max, softmax_max)
    prev_sum_scaled = prev_softmax_sum * torch.exp(prev_softmax_max - merged_max)
    cur_sum_scaled = softmax_sum * torch.exp(softmax_max - merged_max)
    merged_sum = prev_sum_scaled + cur_sum_scaled

    prev_scale = (prev_sum_scaled / merged_sum)[..., 0]
    cur_scale = (cur_sum_scaled / merged_sum)[..., 0]
    prev_scale = prev_scale.permute(2, 0, 1).unsqueeze(-1)
    cur_scale = cur_scale.permute(2, 0, 1).unsqueeze(-1)

    s, b, hidden = o_chunk.shape
    h = prev_scale.shape[2]
    d_v = hidden // h
    prev_output = o_chunk_this_q.view(s, b, h, d_v)
    cur_output = o_chunk.view(s, b, h, d_v)
    merged_output = prev_output * prev_scale + cur_output * cur_scale
    return merged_output.to(o_chunk.dtype).flatten(2), merged_max, merged_sum


def assign_kv_chunk_cache_slots(compute_step_order, cache_slots):
    """Assign fixed cache slots to KV compute steps by LRU during planning."""
    if cache_slots < 1:
        raise ValueError("kv_chunk_cache_slots must be greater than 0")

    cache_slots = min(
        cache_slots,
        len({(step['hc_id'], step['k_sc_id']) for step in compute_step_order}),
    )
    cache_keys = [None] * cache_slots
    lru_slots = []

    for compute_step in compute_step_order:
        cache_key = (compute_step['hc_id'], compute_step['k_sc_id'])
        if cache_key in cache_keys:
            slot_id = cache_keys.index(cache_key)
            lru_slots.remove(slot_id)
        elif None in cache_keys:
            slot_id = cache_keys.index(None)
            cache_keys[slot_id] = cache_key
        else:
            slot_id = lru_slots.pop(0)
            cache_keys[slot_id] = cache_key

        lru_slots.append(slot_id)
        compute_step['kv_slot_id'] = slot_id

    return cache_slots


def assign_q_chunk_cache_slots(compute_step_order, cache_slots):
    """Assign fixed cache slots to Q compute steps by LRU during planning."""
    if cache_slots < 0:
        raise ValueError("q_chunk_cache_slots cannot be negative")
    if cache_slots == 0:
        for compute_step in compute_step_order:
            compute_step['q_slot_id'] = 0
        return 0

    cache_slots = min(
        cache_slots,
        len({(step['hc_id'], step['q_sc_id']) for step in compute_step_order}),
    )
    cache_keys = [None] * cache_slots
    last_steps = [None] * cache_slots
    lru_slots = []

    for step_index, compute_step in enumerate(compute_step_order):
        cache_key = (compute_step['hc_id'], compute_step['q_sc_id'])
        if cache_key in cache_keys:
            slot_id = cache_keys.index(cache_key)
            lru_slots.remove(slot_id)
        elif None in cache_keys:
            slot_id = cache_keys.index(None)
            cache_keys[slot_id] = cache_key
        else:
            slot_id = lru_slots.pop(0)
            last_steps[slot_id]['q_slot_last_use'] = True
            last_steps[slot_id]['q_prefetch_step_index'] = step_index
            cache_keys[slot_id] = cache_key

        lru_slots.append(slot_id)
        compute_step['q_slot_id'] = slot_id
        last_steps[slot_id] = compute_step

    return cache_slots


def run_fa(
    q_chunk,
    k_chunk,
    v_chunk,
    fa_heads,
    is_diagonal,
    attention_mask,
    softmax_scale,
):
    q_chunk = q_chunk.flatten(2)
    k_chunk = k_chunk.flatten(2)
    v_chunk = v_chunk.flatten(2)

    if is_diagonal:
        fa_attention_mask = attention_mask
        if fa_attention_mask is None:
            fa_attention_mask = get_compressed_causal_mask(q_chunk.device)
        pre_tokens = 65536
        next_tokens = 0
        sparse_mode = 2
    else:
        fa_attention_mask = None
        pre_tokens = k_chunk.shape[0]
        next_tokens = k_chunk.shape[0]
        sparse_mode = 0

    o_chunk, softmax_max, softmax_sum, *_ = torch_npu.npu_fusion_attention(
        q_chunk,
        k_chunk,
        v_chunk,
        fa_heads,
        "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=fa_attention_mask,
        scale=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=sparse_mode,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
    )
    return o_chunk, softmax_max, softmax_sum


def run_fa_backward(
    q_chunk,
    k_chunk,
    v_chunk,
    dout_chunk,
    fa_heads,
    is_diagonal,
    attention_mask,
    softmax_scale,
    softmax_max,
    softmax_sum,
    attention_in,
):
    """Compute explicit gradients for one Q/K pair using the final Q online-softmax state."""
    q_shape = q_chunk.shape
    k_shape = k_chunk.shape
    v_shape = v_chunk.shape
    q_chunk = q_chunk.flatten(2)
    k_chunk = k_chunk.flatten(2)
    v_chunk = v_chunk.flatten(2)
    dout_chunk = dout_chunk.flatten(2)
    attention_in = attention_in.flatten(2)

    if is_diagonal:
        fa_attention_mask = attention_mask
        if fa_attention_mask is None:
            fa_attention_mask = get_compressed_causal_mask(q_chunk.device)
        pre_tokens = 65536
        next_tokens = 0
        sparse_mode = 2
    else:
        fa_attention_mask = None
        pre_tokens = k_chunk.shape[0]
        next_tokens = k_chunk.shape[0]
        sparse_mode = 0

    d_q_chunk, d_k_chunk, d_v_chunk, *_ = torch_npu.npu_fusion_attention_grad(
        q_chunk,
        k_chunk,
        v_chunk,
        dout_chunk,
        fa_heads,
        "SBH",
        pse=None,
        padding_mask=None,
        atten_mask=fa_attention_mask,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attention_in,
        scale_value=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=sparse_mode,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
    )
    return (
        d_q_chunk.view(q_shape),
        d_k_chunk.view(k_shape),
        d_v_chunk.view(v_shape),
    )


def prefetch_q(
    query_layer,
    compute_step,
    qk_prefetch_stream,
    compute_device,
    group,
):
    return load_chunk(
        query_layer,
        compute_step['q_hsd_range'],
        qk_prefetch_stream,
        compute_device,
        group,
        d2d_stream=qk_prefetch_stream,
    )


def load_chunk_into(tensor, hsd_range, send, recv, stream, d2d_stream, group):
    """Load a local SBHD chunk and run A2A into fixed buffers."""
    h_range, s_range, d_range = hsd_range
    chunk = tensor[
        s_range[0] : s_range[1],
        :,
        h_range[0] : h_range[1],
        d_range[0] : d_range[1],
    ]
    local_s, b, h, d = chunk.shape
    world_size = dist.get_world_size(group)

    with torch.npu.stream(stream):
        send.view(
            world_size,
            local_s,
            b,
            h // world_size,
            d,
        ).permute(1, 2, 0, 3, 4).copy_(
            chunk.view(local_s, b, world_size, h // world_size, d),
            non_blocking=True,
        )

    if d2d_stream is not stream:
        d2d_stream.wait_stream(stream)
    with torch.npu.stream(d2d_stream):
        dist.all_to_all_single(recv, send, group=group)


def get_chunk_comm_shapes(tensor, hsd_range, group):
    h_range, s_range, d_range = hsd_range
    chunk_shape = (
        s_range[1] - s_range[0],
        tensor.shape[1],
        h_range[1] - h_range[0],
        d_range[1] - d_range[0],
    )
    world_size = dist.get_world_size(group)
    comm_shape = (
        world_size,
        chunk_shape[0] * chunk_shape[1],
        chunk_shape[2] // world_size,
        chunk_shape[3],
    )
    output_shape = (
        chunk_shape[0] * world_size,
        chunk_shape[1],
        chunk_shape[2] // world_size,
        chunk_shape[3],
    )
    return comm_shape, output_shape


def make_chunk_send_buffer(tensor, hsd_range, compute_device, group):
    comm_shape, output_shape = get_chunk_comm_shapes(
        tensor,
        hsd_range,
        group,
    )
    send = torch.empty(
        comm_shape,
        device=compute_device,
        dtype=tensor.dtype,
    )
    return send, output_shape


class KVChunkCache:
    """Fixed-slot KV cache supporting regular head-scatter and routed GQA/MQA."""

    def __init__(
        self,
        key_layer,
        value_layer,
        first_compute_step,
        cache_slots,
        qk_prefetch_stream,
        v_prefetch_stream,
        compute_device,
        group,
        comm_stream=None,
        q_head_chunk_size=None,
    ):
        self.world_size = dist.get_world_size(group)
        k_h_range, k_s_range, k_d_range = first_compute_step['k_hsd_range']
        v_h_range, v_s_range, v_d_range = first_compute_step['v_hsd_range']
        self.kv_head_chunk_size = k_h_range[1] - k_h_range[0]
        self.is_routed = self.kv_head_chunk_size % self.world_size != 0
        self.kv_head_index = None
        self.gathered_heads = 1 if self.is_routed else self.kv_head_chunk_size // self.world_size
        self.key_layer = key_layer
        self.value_layer = value_layer
        self.qk_prefetch_stream = qk_prefetch_stream
        self.v_prefetch_stream = v_prefetch_stream
        self.comm_stream = qk_prefetch_stream if comm_stream is None else comm_stream
        self.group = group

        if self.is_routed:
            if (
                self.kv_head_chunk_size <= 0
                or self.world_size % self.kv_head_chunk_size != 0
                or q_head_chunk_size is None
                or q_head_chunk_size % self.world_size != 0
                or q_head_chunk_size % self.kv_head_chunk_size != 0
            ):
                raise ValueError("The current head chunk cannot use routed KV A2A")
            if v_h_range[1] - v_h_range[0] != self.kv_head_chunk_size:
                raise ValueError("K/V head chunk sizes must match")

            self.replicas_per_kv_head = self.world_size // self.kv_head_chunk_size
            self.kv_head_index = dist.get_rank(group) // self.replicas_per_kv_head
            local_s = k_s_range[1] - k_s_range[0]
            b = key_layer.shape[1]
            self.k_local = torch.empty(
                local_s,
                b,
                self.kv_head_chunk_size,
                k_d_range[1] - k_d_range[0],
                device=compute_device,
                dtype=key_layer.dtype,
            )
            self.v_local = torch.empty(
                v_s_range[1] - v_s_range[0],
                b,
                self.kv_head_chunk_size,
                v_d_range[1] - v_d_range[0],
                device=compute_device,
                dtype=value_layer.dtype,
            )
            self.k_send = torch.empty(
                self.world_size,
                local_s,
                b,
                1,
                k_d_range[1] - k_d_range[0],
                device=compute_device,
                dtype=key_layer.dtype,
            )
            self.v_send = torch.empty(
                self.world_size,
                v_s_range[1] - v_s_range[0],
                b,
                1,
                v_d_range[1] - v_d_range[0],
                device=compute_device,
                dtype=value_layer.dtype,
            )
            self.k_shape = (
                local_s * self.world_size,
                b,
                1,
                k_d_range[1] - k_d_range[0],
            )
            self.v_shape = (
                (v_s_range[1] - v_s_range[0]) * self.world_size,
                b,
                1,
                v_d_range[1] - v_d_range[0],
            )
        else:
            self.k_send, self.k_shape = make_chunk_send_buffer(
                key_layer,
                first_compute_step['k_hsd_range'],
                compute_device,
                group,
            )
            self.v_send, self.v_shape = make_chunk_send_buffer(
                value_layer,
                first_compute_step['v_hsd_range'],
                compute_device,
                group,
            )
        self.slots = [
            {
                'key': None,
                'k': torch.empty_like(self.k_send),
                'v': torch.empty_like(self.v_send),
                'ready': torch.npu.Event(),
                'used': torch.npu.Event(),
            }
            for _ in range(cache_slots)
        ]
        self.send_used = torch.npu.Event()
        self.send_in_use = False

    def _load_routed_tensor(
        self,
        tensor,
        hsd_range,
        local,
        send,
        recv,
        stream,
    ):
        h_range, s_range, d_range = hsd_range
        with torch.npu.stream(stream):
            local.copy_(
                tensor[
                    s_range[0] : s_range[1],
                    :,
                    h_range[0] : h_range[1],
                    d_range[0] : d_range[1],
                ],
                non_blocking=True,
            )
            for kv_head_index in range(self.kv_head_chunk_size):
                dest_start = kv_head_index * self.replicas_per_kv_head
                dest_end = dest_start + self.replicas_per_kv_head
                routed_head = local[:, :, kv_head_index : kv_head_index + 1, :].unsqueeze(0)
                send[dest_start:dest_end].copy_(
                    routed_head.expand(self.replicas_per_kv_head, *routed_head.shape[1:]),
                    non_blocking=True,
                )

        if self.comm_stream is not stream:
            self.comm_stream.wait_stream(stream)
        with torch.npu.stream(self.comm_stream):
            dist.all_to_all_single(recv, send, group=self.group)

    def load(self, compute_step):
        slot_id = compute_step['kv_slot_id']
        slot = self.slots[slot_id]
        cache_key = (compute_step['hc_id'], compute_step['k_sc_id'])

        if slot['key'] != cache_key:
            if slot['key'] is not None:
                self.comm_stream.wait_event(slot['used'])
            if self.send_in_use:
                if self.qk_prefetch_stream is not self.comm_stream:
                    self.qk_prefetch_stream.wait_event(self.send_used)
                if (
                    self.v_prefetch_stream is not self.comm_stream
                    and self.v_prefetch_stream is not self.qk_prefetch_stream
                ):
                    self.v_prefetch_stream.wait_event(self.send_used)

            if self.is_routed:
                self._load_routed_tensor(
                    self.key_layer,
                    compute_step['k_hsd_range'],
                    self.k_local,
                    self.k_send,
                    slot['k'],
                    self.qk_prefetch_stream,
                )
                self._load_routed_tensor(
                    self.value_layer,
                    compute_step['v_hsd_range'],
                    self.v_local,
                    self.v_send,
                    slot['v'],
                    self.v_prefetch_stream,
                )
            else:
                load_chunk_into(
                    self.key_layer,
                    compute_step['k_hsd_range'],
                    self.k_send,
                    slot['k'],
                    self.qk_prefetch_stream,
                    self.comm_stream,
                    self.group,
                )
                load_chunk_into(
                    self.value_layer,
                    compute_step['v_hsd_range'],
                    self.v_send,
                    slot['v'],
                    self.v_prefetch_stream,
                    self.comm_stream,
                    self.group,
                )
            slot['ready'].record(self.comm_stream)
            self.send_used.record(self.comm_stream)
            self.send_in_use = True
            slot['key'] = cache_key

        return (
            slot['k'].view(self.k_shape),
            slot['v'].view(self.v_shape),
            slot['ready'],
            slot_id,
        )

    def record_use(self, slot_id, stream):
        self.slots[slot_id]['used'].record(stream)


class QChunkCache:
    """Fixed-slot backward Q-state cache."""

    def __init__(
        self,
        query_layer,
        grad_output,
        saved_group_tensors,
        saved_group_index,
        first_compute_step,
        cache_slots,
        q_prefetch_stream,
        comm_stream,
        compute_device,
        group,
    ):
        self.query_layer = query_layer
        self.grad_output = grad_output
        self.saved_group_tensors = saved_group_tensors
        self.saved_group_index = saved_group_index
        self.q_prefetch_stream = q_prefetch_stream
        self.comm_stream = comm_stream
        self.group = group
        self.cache_enabled = cache_slots > 0

        self.q_send, self.q_shape = make_chunk_send_buffer(
            query_layer,
            first_compute_step['q_hsd_range'],
            compute_device,
            group,
        )
        q_h_range, q_s_range, _ = first_compute_step['q_hsd_range']
        grad_output_hsd_range = [
            q_h_range,
            q_s_range,
            [0, grad_output.shape[-1]],
        ]
        self.grad_output_send, self.grad_output_shape = make_chunk_send_buffer(
            grad_output,
            grad_output_hsd_range,
            compute_device,
            group,
        )
        first_group_key = (
            first_compute_step['hc_id'],
            first_compute_step['q_sc_id'],
        )
        first_offset = saved_group_index[first_group_key] * 3
        first_o = saved_group_tensors[first_offset]
        first_softmax_max = saved_group_tensors[first_offset + 1]
        first_softmax_sum = saved_group_tensors[first_offset + 2]
        self.slots = [
            {
                'key': None,
                'q': torch.empty_like(self.q_send),
                'grad_output': torch.empty_like(self.grad_output_send),
                'o': torch.empty_like(first_o, device=compute_device),
                'softmax_max': torch.empty_like(
                    first_softmax_max,
                    device=compute_device,
                ),
                'softmax_sum': torch.empty_like(
                    first_softmax_sum,
                    device=compute_device,
                ),
                'ready': torch.npu.Event(),
                'used': torch.npu.Event(),
            }
            for _ in range(max(cache_slots, 1))
        ]
        self.send_used = torch.npu.Event()
        self.send_in_use = False

    def load(self, compute_step):
        slot_id = compute_step['q_slot_id']
        slot = self.slots[slot_id]
        cache_key = (compute_step['hc_id'], compute_step['q_sc_id'])
        needs_load = slot['key'] != cache_key or not self.cache_enabled

        if needs_load:
            if slot['key'] is not None:
                self.q_prefetch_stream.wait_event(slot['used'])
                if self.comm_stream is not self.q_prefetch_stream:
                    self.comm_stream.wait_event(slot['used'])
            if self.send_in_use and self.comm_stream is not self.q_prefetch_stream:
                self.q_prefetch_stream.wait_event(self.send_used)

            load_chunk_into(
                self.query_layer,
                compute_step['q_hsd_range'],
                self.q_send,
                slot['q'],
                self.q_prefetch_stream,
                self.comm_stream,
                self.group,
            )
            q_h_range, q_s_range, _ = compute_step['q_hsd_range']
            grad_output_hsd_range = [
                q_h_range,
                q_s_range,
                [0, self.grad_output.shape[-1]],
            ]
            load_chunk_into(
                self.grad_output,
                grad_output_hsd_range,
                self.grad_output_send,
                slot['grad_output'],
                self.q_prefetch_stream,
                self.comm_stream,
                self.group,
            )
            saved_offset = self.saved_group_index[cache_key] * 3
            with torch.npu.stream(self.q_prefetch_stream):
                slot['o'].copy_(
                    self.saved_group_tensors[saved_offset],
                    non_blocking=True,
                )
                slot['softmax_max'].copy_(
                    self.saved_group_tensors[saved_offset + 1],
                    non_blocking=True,
                )
                slot['softmax_sum'].copy_(
                    self.saved_group_tensors[saved_offset + 2],
                    non_blocking=True,
                )
            if self.comm_stream is not self.q_prefetch_stream:
                self.comm_stream.wait_stream(self.q_prefetch_stream)
            slot['ready'].record(self.comm_stream)
            self.send_used.record(self.comm_stream)
            self.send_in_use = True
            slot['key'] = cache_key

        return (
            slot['q'].view(self.q_shape),
            slot['grad_output'].view(self.grad_output_shape),
            slot['o'],
            slot['softmax_max'],
            slot['softmax_sum'],
            slot['ready'] if needs_load else None,
            slot_id,
        )

    def record_use(self, slot_id, stream):
        self.slots[slot_id]['used'].record(stream)


def prefetch_kv(kv_chunk_cache, compute_step):
    return kv_chunk_cache.load(compute_step)


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    gather_size: Optional[int] = None,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, gather_size)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, gather_size=None):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.scatter_size = input_.size(scatter_dim)
        ctx.gather_dim = gather_dim
        ctx.gather_size = gather_size
        output = _all_to_all(input_, process_group, scatter_dim, gather_dim, gather_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output, ctx.process_group, ctx.gather_dim, ctx.scatter_dim, ctx.scatter_size
        )
        return (grad_output, None, None, None, None, None)


def _all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None,
):
    if not 3 <= input_.dim() <= 4:
        raise ValueError(f"Input tensor must have 3 or 4 dimensions, got {input_.dim()}")
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size
    gather_mod = gather_size % world_size
    scatter_mod = scatter_size % world_size

    if gather_mod != 0 or scatter_mod != 0:
        raise ValueError(
            "For aligned data, gather_size and scatter_size must be divisible by world_size"
        )
    return _aligned_all_to_all(input_, group, scatter_dim, gather_dim)


def _aligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    world_size = dist.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // world_size
    if scatter_dim == 0:
        input_t = input_.reshape([world_size] + inp_shape).contiguous()
    else:
        input_t = (
            input_.reshape([-1, world_size] + inp_shape[scatter_dim:]).transpose(0, 1).contiguous()
        )

    output = torch.empty_like(input_t)

    dist.all_to_all_single(output, input_t, group=group)

    output = output.view([world_size] + inp_shape).contiguous()
    output_dim = output.dim()
    if gather_dim == 1:
        output = output.transpose(0, 1).contiguous()
    elif gather_dim == 2:
        output = output.permute(*PERMUTE_DIMS2[output_dim]).contiguous()
    elif gather_dim == 3:
        output = output.permute(*PERMUTE_DIMS1[output_dim]).contiguous()
    output = output.view(
        inp_shape[:gather_dim]
        + [
            inp_shape[gather_dim] * world_size,
        ]
        + inp_shape[gather_dim + 1 :]
    ).contiguous()

    return output


def parse_qk_id(compute_step):
    q_id, k_id = compute_step['qkv_sc_id'].removeprefix('q').split('k')
    return int(q_id), int(k_id)


def convert_compute_step_order(compute_order, d_v):
    if not compute_order:
        return []

    if compute_order[0]['phase'] == 'grad':
        compute_step_order = []
        for grad_step in compute_order:
            if grad_step['phase'] == 'grad_acc':
                continue
            if grad_step['phase'] != 'grad':
                raise RuntimeError("Expected grad planner steps")

            q_sc_id, k_sc_id = parse_qk_id(grad_step)
            v_hsd_range = [list(x) for x in grad_step['v_hsd_range']]
            v_hsd_range[2] = [0, d_v]
            compute_step_order.append(
                {
                    'hc_id': grad_step['hc_id'],
                    'q_sc_id': q_sc_id,
                    'k_sc_id': k_sc_id,
                    'q_hsd_range': grad_step['q_hsd_range'],
                    'k_hsd_range': grad_step['k_hsd_range'],
                    'v_hsd_range': v_hsd_range,
                }
            )
        return compute_step_order

    compute_step_order = []
    for i_step in range(0, len(compute_order), 3):
        score_step = compute_order[i_step]
        prob_step = compute_order[i_step + 1]
        output_step = compute_order[i_step + 2]
        if (
            score_step['phase'] != 'score'
            or prob_step['phase'] != 'prob'
            or output_step['phase'] != 'output'
        ):
            raise RuntimeError("Expected score/prob/output planner triplets")

        q_sc_id, k_sc_id = parse_qk_id(score_step)
        v_hsd_range = [list(x) for x in output_step['v_hsd_range']]
        v_hsd_range[2] = [0, d_v]
        compute_step_order.append(
            {
                'hc_id': score_step['hc_id'],
                'q_sc_id': q_sc_id,
                'k_sc_id': k_sc_id,
                'q_hsd_range': score_step['q_hsd_range'],
                'k_hsd_range': score_step['k_hsd_range'],
                'v_hsd_range': v_hsd_range,
            }
        )
    return compute_step_order


def make_compute_step_order(
    # q,k,v,
    b,
    h_q,
    h_kv,
    local_s,
    d,
    CP,
    start_q_seq_chunk_id=None,
    start_head_chunk_id=None,
    head_chunk_size=None,
    seq_chunk_size=None,
    dim_chunk_size=None,
    q_d2d_mode='alltoall',
    kv_d2d_mode='alltoall',
):
    """
    All shapes are for a single rank before all-to-all/D2D.
    b, h, h_kv, local_s, and d are single-rank dimensions before D2D.
    Chunk-related configuration values also use pre-D2D single-rank dimensions.

    Build the complete pipeline order directly from Q/K/V (causal attention with three phases).
    Supports MHA and GQA (Grouped Query Attention).

    Inputs:
    Q: [local_s, b, h_q, d]
    K: [local_s, b, h_kv, d]
    V: [local_s, b, h_kv, d]

    Definitions:
    - local_s: local sequence length on one rank
    - h_q/h_kv: Q/KV head counts on one rank
    - d: head dimension
    - seq_chunk_size: local sequence chunk size before D2D
    - head_chunk_size: local head chunk size before D2D
    - dim_chunk_size: local dimension chunk size before D2D
    - CP: D2D communication parallelism; it does not change planner ranges

    Parameters:
    start_q_seq_chunk_id: None or an integer
        - None: generate the complete order for all Q sequence chunks
        - Integer: generate the order only for the selected Q sequence chunk (reduces CPU planning cost)

    start_head_chunk_id: None or an integer
        - None: generate the complete order for all head chunks
        - Integer: generate the order only for the selected head chunk (reduces CPU planning cost)

    Computation flow (causal attention, three phases):
    Each (Q, K) pair has three phases:
    1. Score: Q @ K^T, accumulated over dimension chunks
    2. Probability: softmax/online-softmax update from the complete score
    3. Output: prob @ V, accumulated into output over dimension chunks

    K traversal order for each Q sequence chunk (Z-shaped traversal):
    - q0: [k0]
    - q1: [k0, k1] (forward order)
    - q2: [k1, k0, k2] (reverse old K chunks, then the new K chunk)
    - q3: [k2, k0, k1, k3] (start at the new K chunk, then traverse old and new K)
    - q4: [k3, k1, k0, k2, k4] (start at the new K chunk and traverse in reverse)

    The direction alternates by parity; each step starts with the K chunk added in the previous round and ends with the current new K chunk.

    Dimension chunks alternate by global step during the score phase and use the same order during the output phase.

    Return format: a list of dictionaries. Each dictionary contains:
    phase1: score computation:
    {
        'phase': 'score',
        'qkv_sc_id': f'q{q_sc_id}k{k_sc_id}',
        'q_hc_sc_dc_id': [hc_id, sc_id, dc_id],
        'k_hc_sc_dc_id': [hc_id, sc_id, dc_id],
        'q_range': [[h_start, h_end], [s_start, s_end], [d_start, d_end]],
        'k_range': [[h_start, h_end], [s_start, s_end], [d_start, d_end]],
        'score': f'+score_q{sc_id}k{sc_id}d{dc_id}',
        'score_acc': f'score_q{q_sc_id}k{k_sc_id}'
    }

    phase2: probability computation
    {
        'phase': 'prob',
        'qkv_sc_id': f'q{q_sc_id}k{k_sc_id}',
        'score': f'score_q{q_sc_id}k{k_sc_id}',
        'cur_lse': f'cur_lse_q{q_sc_id}',
        'old_lse': f'old_lse_q{q_sc_id}',
        'prob': f'prob_q{q_sc_id}k{k_sc_id}'
    }

    phase3: output computation
    {
        'phase': 'output',
        'qkv_sc_id': f'v{k_sc_id}',
        'prob': f'prob_q{q_sc_id}k{k_sc_id}',
        'v_hc_sc_dc_id': [hc_id, sc_id, dc_id],
        'v_range': [[h_start, h_end], [s_start, s_end], [d_start, d_end]],
        'output': f'output_s{q_sc_id}d{dc_id}' or f'+output_s{q_sc_id}d{dc_id}'
    }
    """

    # Validate the GQA configuration.
    if h_q % h_kv != 0:
        raise ValueError(f"Q head count {h_q} must be divisible by KV head count {h_kv} for GQA")

    group_size = h_q // h_kv  # group_size=1 for MHA and >1 for GQA

    # Compute chunk counts from the Q dimensions.
    head_chunk_num = (h_q + head_chunk_size - 1) // head_chunk_size
    seq_chunk_num = (local_s + seq_chunk_size - 1) // seq_chunk_size
    dim_chunk_num = (d + dim_chunk_size - 1) // dim_chunk_size

    # Determine the head chunk range to process.
    if start_head_chunk_id is not None:
        # Process only the selected head chunk.
        if start_head_chunk_id >= head_chunk_num or start_head_chunk_id < 0:
            raise ValueError(
                f"start_head_chunk_id={start_head_chunk_id} is outside [0, {head_chunk_num - 1}]"
            )
        head_range = [start_head_chunk_id]
    else:
        # Process all head chunks.
        head_range = range(head_chunk_num)

    # Determine the Q sequence chunk range to process.
    if start_q_seq_chunk_id is not None:
        # Process only the selected Q sequence chunk.
        if start_q_seq_chunk_id >= seq_chunk_num or start_q_seq_chunk_id < 0:
            raise ValueError(
                f"start_q_seq_chunk_id={start_q_seq_chunk_id} is outside [0, {seq_chunk_num - 1}]"
            )
        q_range = [start_q_seq_chunk_id]
    else:
        # Process all Q sequence chunks.
        q_range = range(seq_chunk_num)

    compute_order = []

    # Initialize global_step, which controls dimension chunk traversal.
    # For a selected q_sc_id, include all preceding steps.
    if start_q_seq_chunk_id is not None:
        # global_step is the sum of K counts for preceding Q chunks.
        # q_sc_id has q_sc_id + 1 K chunks in causal attention.
        global_step = sum(range(1, start_q_seq_chunk_id + 1))
    else:
        global_step = 0  # Global counter controlling dimension chunk order.

    # Adjust global_step when a head chunk is selected as well.
    if start_head_chunk_id is not None:
        # Each head chunk contains the same total number of sequence steps.
        if start_q_seq_chunk_id is None:
            # Accumulate steps from all preceding head chunks if no Q chunk is selected.
            # Steps per head chunk = seq_chunk_num * (seq_chunk_num + 1) / 2.
            steps_per_head_chunk = seq_chunk_num * (seq_chunk_num + 1) // 2
            global_step = start_head_chunk_id * steps_per_head_chunk

    # Traverse head chunks.
    for hc_id in head_range:
        # Traverse Q sequence chunks.
        for q_sc_id in q_range:
            # Causal attention with Z-shaped traversal
            # Determine the K sequence chunk order.
            if q_sc_id == 0:
                k_order = [0]
            elif q_sc_id == 1:
                # q1: [k0, k1] in forward order.
                k_order = [0, 1]
            else:
                # q >= 2: start with the K chunk added in the previous round,
                # traverse old K chunks, and finish with the current new K chunk.
                prev_new_k = q_sc_id - 1  # K chunk added in the previous round.
                old_k_range = list(range(q_sc_id - 1))  # Old K chunks: [0, ..., q_sc_id - 2].

                # Alternate the old-K traversal direction by parity.
                if q_sc_id % 2 == 1:
                    # Odd Q: traverse old K in forward order.
                    k_order = [prev_new_k] + old_k_range + [q_sc_id]
                else:
                    # Even Q: traverse old K in reverse order.
                    k_order = [prev_new_k] + old_k_range[::-1] + [q_sc_id]

            # Traverse K sequence chunks.
            for k_idx, k_sc_id in enumerate(k_order):
                # === Phase 1: Score computation ===
                # Dimension chunk order is controlled by global_step.
                if global_step % 2 == 0:
                    dim_range = list(range(dim_chunk_num))
                else:
                    dim_range = list(range(dim_chunk_num - 1, -1, -1))

                # Score computation: Q @ K^T over all dimension chunks.
                for dc_idx, dc_id in enumerate(dim_range):
                    # Compute the Q range from the Q shape.
                    q_h_start = hc_id * head_chunk_size
                    q_h_end = min((hc_id + 1) * head_chunk_size, h_q)
                    q_s_start = q_sc_id * seq_chunk_size
                    q_s_end = min((q_sc_id + 1) * seq_chunk_size, local_s)
                    q_d_start = dc_id * dim_chunk_size
                    q_d_end = min((dc_id + 1) * dim_chunk_size, d)

                    # Compute the K range using the GQA group mapping.
                    k_h_start = (hc_id * head_chunk_size) // group_size
                    k_h_end = min(((hc_id + 1) * head_chunk_size) // group_size, h_kv)
                    k_s_start = k_sc_id * seq_chunk_size
                    k_s_end = min((k_sc_id + 1) * seq_chunk_size, local_s)
                    k_d_start = dc_id * dim_chunk_size
                    k_d_end = min((dc_id + 1) * dim_chunk_size, d)

                    compute_order.append(
                        {
                            'phase': 'score',
                            'hc_id': hc_id,
                            'qkv_sc_id': f'q{q_sc_id}k{k_sc_id}',
                            'q_hc_sc_dc_id': [hc_id, q_sc_id, dc_id],
                            'k_hc_sc_dc_id': [hc_id, k_sc_id, dc_id],
                            'q_hsd_range': [
                                [q_h_start, q_h_end],
                                [q_s_start, q_s_end],
                                [q_d_start, q_d_end],
                            ],
                            'k_hsd_range': [
                                [k_h_start, k_h_end],
                                [k_s_start, k_s_end],
                                [k_d_start, k_d_end],
                            ],
                            'score': f'score_q{q_sc_id}k{k_sc_id}d{dc_id}',
                            'score_acc': f'score_q{q_sc_id}k{k_sc_id}',
                        }
                    )

                # === Phase 2: Probability computation ===
                # Update softmax/online-softmax state from the complete score_qk.
                compute_order.append(
                    {
                        'phase': 'prob',
                        'hc_id': hc_id,
                        'qkv_sc_id': f'q{q_sc_id}k{k_sc_id}',
                        'score': f'score_q{q_sc_id}k{k_sc_id}',
                        'cur_lse': f'cur_lse_q{q_sc_id}',
                        'old_lse': f'old_lse_q{q_sc_id}',
                        'prob': f'prob_q{q_sc_id}k{k_sc_id}',
                    }
                )

                # === Phase 3: Output computation ===
                # Compute prob @ V using the same dimension chunk order.
                for dc_idx, dc_id in enumerate(dim_range):
                    # Determine whether this is the first output accumulation for Q (first K).
                    is_first_output = k_idx == 0
                    output_prefix = '' if is_first_output else '+'

                    # Compute the V range, which follows K through the GQA group mapping.
                    v_h_start = (hc_id * head_chunk_size) // group_size
                    v_h_end = min(((hc_id + 1) * head_chunk_size) // group_size, h_kv)
                    v_s_start = k_sc_id * seq_chunk_size
                    v_s_end = min((k_sc_id + 1) * seq_chunk_size, local_s)
                    v_d_start = dc_id * dim_chunk_size
                    v_d_end = min((dc_id + 1) * dim_chunk_size, d)

                    compute_order.append(
                        {
                            'phase': 'output',
                            'hc_id': hc_id,
                            'qkv_sc_id': f'v{k_sc_id}',
                            'prob': f'prob_q{q_sc_id}k{k_sc_id}',
                            'v_hc_sc_dc_id': [hc_id, k_sc_id, dc_id],
                            'v_hsd_range': [
                                [v_h_start, v_h_end],
                                [v_s_start, v_s_end],
                                [v_d_start, v_d_end],
                            ],
                            'output': f'{output_prefix}output_s{q_sc_id}d{dc_id}',
                        }
                    )

                global_step += 1  # Advance after completing all three phases for a (Q, K) pair.

    return compute_order


def make_backward_compute_step_order(
    # q,k,v,
    b,
    h_q,
    h_kv,
    local_s,
    d,
    CP,
    is_k0_first: bool,
    start_q_seq_chunk_id=None,
    start_head_chunk_id=None,
    head_chunk_size=None,
    seq_chunk_size=None,
    dim_chunk_size=None,
    q_d2d_mode='alltoall',
    kv_d2d_mode='alltoall',
):
    """

    All shapes are for a single rank before all-to-all/D2D.
    b, h, h_kv, local_s, and d are single-rank dimensions before D2D.
    Chunk-related configuration values also use pre-D2D single-rank dimensions.

    Build the complete backward pipeline order directly from Q/K/V (causal attention with two phases).
    Supports MHA and GQA (Grouped Query Attention).

    Inputs:
    Q: [local_s, b, h_q, d]
    K: [local_s, b, h_kv, d]
    V: [local_s, b, h_kv, d]

    Definitions:
    - local_s: local sequence length on one rank
    - h_q/h_kv: Q/KV head counts on one rank
    - d: head dimension
    - seq_chunk_size: local sequence chunk size before D2D
    - head_chunk_size: local head chunk size before D2D
    - dim_chunk_size: local dimension chunk size before D2D
    - CP: D2D communication parallelism; it does not change planner ranges

    dKV-major traversal (Z-shaped) with is_k0_first=False:
    - dkv3: dq3
    - dkv2: dq3 dq2
    - dkv1: dq2 dq3 dq1
    - dkv0: dq1 dq3 dq2 dq0

    dKV-major traversal (Z-shaped) with is_k0_first=True:
    - dkv0: dq0 dq2 dq3 dq1
    - dkv1: dq1 dq3 dq2
    - dkv2: dq2 dq3
    - dkv3: dq3


    Each entry in backward_compute_order is a dictionary:
    {
        "phase": "grad", # grad  grad_acc
        "hc_id": hc_id,
        "qkv_sc_id": f"q{q_sc_id}k{k_sc_id}",
        "q_hsd_range": q_hsd_range,
        "k_hsd_range": k_hsd_range,
        "v_hsd_range": v_hsd_range,
    }
    """

    if h_q % h_kv != 0:
        raise ValueError(f"Q head count {h_q} must be divisible by KV head count {h_kv} for GQA")

    group_size = h_q // h_kv
    head_chunk_num = (h_q + head_chunk_size - 1) // head_chunk_size
    seq_chunk_num = (local_s + seq_chunk_size - 1) // seq_chunk_size
    dim_chunk_num = (d + dim_chunk_size - 1) // dim_chunk_size

    if start_head_chunk_id is not None:
        if start_head_chunk_id >= head_chunk_num or start_head_chunk_id < 0:
            raise ValueError(
                f"start_head_chunk_id={start_head_chunk_id} is outside [0, {head_chunk_num - 1}]"
            )
        head_range = [start_head_chunk_id]
    else:
        head_range = range(head_chunk_num)

    if start_q_seq_chunk_id is not None:
        if start_q_seq_chunk_id >= seq_chunk_num or start_q_seq_chunk_id < 0:
            raise ValueError(
                f"start_q_seq_chunk_id={start_q_seq_chunk_id} is outside [0, {seq_chunk_num - 1}]"
            )

    backward_compute_order = []
    global_step = 0

    for hc_id in head_range:
        # K0-first reverses the order of K groups and the Q order inside
        # each group independently; it is not a reverse of the flat list.
        k_sc_ids = range(seq_chunk_num) if is_k0_first else range(seq_chunk_num - 1, -1, -1)
        for k_sc_id in k_sc_ids:
            reverse_k_sc_id = seq_chunk_num - 1 - k_sc_id

            if reverse_k_sc_id == 0:
                reverse_q_order = [0]
            elif reverse_k_sc_id == 1:
                reverse_q_order = [0, 1]
            else:
                prev_new_q = reverse_k_sc_id - 1
                old_q_range = list(range(reverse_k_sc_id - 1))
                if reverse_k_sc_id % 2 == 1:
                    reverse_q_order = [prev_new_q] + old_q_range + [reverse_k_sc_id]
                else:
                    reverse_q_order = [prev_new_q] + old_q_range[::-1] + [reverse_k_sc_id]

            q_order = [seq_chunk_num - 1 - q_sc_id for q_sc_id in reverse_q_order]
            if is_k0_first:
                # Mirror only this K group's causal-Q traversal.
                q_order = list(reversed(q_order))
            if start_q_seq_chunk_id is not None:
                q_order = [q_sc_id for q_sc_id in q_order if q_sc_id == start_q_seq_chunk_id]

            for q_sc_id in q_order:
                if global_step % 2 == 0:
                    dim_range = list(range(dim_chunk_num))
                else:
                    dim_range = list(range(dim_chunk_num - 1, -1, -1))

                for dc_id in dim_range:
                    q_h_start = hc_id * head_chunk_size
                    q_h_end = min((hc_id + 1) * head_chunk_size, h_q)
                    q_s_start = q_sc_id * seq_chunk_size
                    q_s_end = min((q_sc_id + 1) * seq_chunk_size, local_s)
                    q_d_start = dc_id * dim_chunk_size
                    q_d_end = min((dc_id + 1) * dim_chunk_size, d)

                    k_h_start = (hc_id * head_chunk_size) // group_size
                    k_h_end = min(((hc_id + 1) * head_chunk_size) // group_size, h_kv)
                    k_s_start = k_sc_id * seq_chunk_size
                    k_s_end = min((k_sc_id + 1) * seq_chunk_size, local_s)
                    k_d_start = dc_id * dim_chunk_size
                    k_d_end = min((dc_id + 1) * dim_chunk_size, d)

                    q_hsd_range = [[q_h_start, q_h_end], [q_s_start, q_s_end], [q_d_start, q_d_end]]
                    k_hsd_range = [[k_h_start, k_h_end], [k_s_start, k_s_end], [k_d_start, k_d_end]]
                    v_hsd_range = [[k_h_start, k_h_end], [k_s_start, k_s_end], [k_d_start, k_d_end]]

                    # for phase in ('grad', 'grad_acc'):
                    for phase in ('grad',):
                        backward_compute_order.append(
                            {
                                'phase': phase,
                                'hc_id': hc_id,
                                'qkv_sc_id': f'q{q_sc_id}k{k_sc_id}',
                                'q_hsd_range': q_hsd_range,
                                'k_hsd_range': k_hsd_range,
                                'v_hsd_range': v_hsd_range,
                            }
                        )

                global_step += 1

    return backward_compute_order


def _demo():
    # Exercise the make_compute_step_order function.
    b, h, s, d = 1, 64, 2000, 64
    h_kv = 8  # GQA
    # q = torch.randn(b, h, s, d)
    # k = torch.randn(b, h_kv, s, d)
    # v = torch.randn(b, h_kv, s, d)

    CP = None  # Placeholder; the planner does not use CP for local ranges.
    head_chunk_size = 32
    seq_chunk_size = 512
    # dim_chunk_size = 16
    dim_chunk_size = d

    print("Test 1: generate the complete forward order (start_q_seq_chunk_id=None)")
    compute_order_full = make_compute_step_order(
        b,
        h,
        h_kv,
        s,
        d,
        CP,
        start_q_seq_chunk_id=None,
        head_chunk_size=head_chunk_size,
        seq_chunk_size=seq_chunk_size,
        dim_chunk_size=dim_chunk_size,
        q_d2d_mode='alltoall',
        kv_d2d_mode='alltoall',
    )
    print(len(compute_order_full))
    for step in compute_order_full:
        print(step)

    print("Test 2: generate the complete backward order (KV-major Z-shaped traversal)")
    backward_compute_order = make_backward_compute_step_order(
        b,
        h,
        h_kv,
        s,
        d,
        CP,
        is_k0_first=False,
        start_q_seq_chunk_id=None,
        start_head_chunk_id=0,
        head_chunk_size=head_chunk_size,
        seq_chunk_size=seq_chunk_size,
        dim_chunk_size=dim_chunk_size,
        q_d2d_mode='alltoall',
        kv_d2d_mode='alltoall',
    )
    print(len(backward_compute_order))
    for step in backward_compute_order:
        print(step)


if __name__ == "__main__":
    _demo()
