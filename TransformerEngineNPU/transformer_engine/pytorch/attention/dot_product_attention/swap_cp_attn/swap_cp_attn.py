# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch.distributed as dist

from .swap_cp_attn_utils import (
    make_chunked_grad_cpu,
    materialize_chunked_grad_cpu,
    run_full_npu_backward_saved_qkv,
    select_head_chunk_size,
    to_pinned_cpu,
    all_to_all,
    assign_kv_chunk_cache_slots,
    assign_q_chunk_cache_slots,
    convert_compute_step_order,
    make_compute_step_order,
    make_backward_compute_step_order,
    KVChunkCache,
    QChunkCache,
    prefetch_q,
    prefetch_kv,
    run_fa,
    run_fa_backward,
    online_update_o_chunk,
)


class SwapCPAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_layer,  # sbhd
        key_layer,  # sbhd
        value_layer,  # sbhd
        attention_mask,
        qkv_format,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_mask_type,
        attention_dropout,
        softmax_scale,
        deterministic,
        cp_group,
        cp_stream,
        ulysses_comm_para,
        # -- SwapAttn-specific parameters
        # Chunk parameters
        seq_chunk_size,  # seq_chunk_size before D2D; [sc*CP] after all-to-all
        head_chunk_size,  # head_chunk_size before D2D; [hc/CP] after all-to-all
        kv_chunk_cache_slots: int,  # forward only: number of resident KV chunk pairs
        seq_layout_mode: str,  # "striped" "contiguous"
        # Stream parameters
        qk_prefetch_stream,  # QK chunk prefetch
        v_prefetch_stream,  # V chunk prefetch
        general_offload_stream,  # offload each computed chunk output
        final_hdc_stream,  # H2D stream when the final output stays on NPU
        # Device parameters
        compute_device,
        is_cpu_output: bool,  # forward: output on CPU; backward: grad_query_layer, grad_key_layer, and grad_value_layer on CPU
        # Forward/backward parameters
        is_forward_only: bool,  # if True, do not save tensors for backward
        # Backward-specific parameters
        q_chunk_cache_slots: int,  # backward only: 0 disables cross-step reuse; positive values keep Q chunks resident
        q_prefetch_stream,  # Q-side staging and serialized submission of all backward A2A operations
        kv_prefetch_stream,  # K/V H2D staging
        backward_mode: str,  # "high_accuracy": FA processes all causal Q chunks for each KV chunk; "low_memory": FA processes one causal Q chunk at a time
    ):
        """Sequence layout contract for SwapCPAttn.

        seq_layout_mode describes how sequence tokens are owned by CP ranks:

        - "striped": each rank stores interleaved sequence chunks. For example,
        with CP=4, seq_chunk_size=sc, and local_s=2*sc (global sequence
        length 8*sc), the local sequence order is:

            rank 0: [0:sc],     [4*sc:5*sc]
            rank 1: [sc:2*sc],   [5*sc:6*sc]
            rank 2: [2*sc:3*sc], [6*sc:7*sc]
            rank 3: [3*sc:4*sc], [7*sc:8*sc]

        In general, rank r owns chunks r, r+CP, r+2*CP, ... in global
        sequence-chunk order.

        - "contiguous": rank r owns one contiguous global interval,
        [r*local_s:(r+1)*local_s]. Conversion between contiguous and striped
        layouts is not implemented yet.

        For CP > 1, the current executor only validates and accepts "striped";
        it does not reorder the input tensors.
        """

        scatter_idx = ulysses_comm_para.get('scatter_idx')
        gather_idx = ulysses_comm_para.get('gather_idx')
        CP = torch.distributed.get_world_size(cp_group)

        if qkv_format.lower() != 'sbhd':
            raise ValueError("Only sbhd input is supported")
        if scatter_idx != 2 or gather_idx != 0:
            raise ValueError("Only scatter H and gather S are supported")
        if attn_mask_type != 'causal':
            raise ValueError("Only causal attention is supported")
        if attention_dropout != 0:
            raise ValueError("The current implementation supports only attention_dropout=0")
        if query_layer.dtype != torch.bfloat16:
            raise ValueError("The current implementation supports only BF16")
        if query_layer.dtype != key_layer.dtype or query_layer.dtype != value_layer.dtype:
            raise ValueError("Q/K/V dtypes must match")
        if CP > 1 and (seq_layout_mode not in ['striped']):
            raise ValueError("seq_layout_mode must be 'striped' when CP > 1")

        local_s, b, h_q, d_qk = query_layer.shape
        local_s_k, b_k, h_kv, d_k = key_layer.shape
        local_s_v, b_v, h_v, d_v = value_layer.shape
        if (local_s_k, b_k) != (local_s, b) or (local_s_v, b_v) != (local_s, b):
            raise ValueError("Q/K/V local_s and batch dimensions must match")
        if d_k != d_qk:
            raise ValueError("Q/K head dimensions must match")
        if b != 1:
            raise ValueError("The current implementation supports only batch=1")
        if h_kv != h_v:
            raise ValueError("K/V head counts must match")
        if h_q % h_kv != 0:
            raise ValueError("The Q head count must be divisible by the KV head count")
        group_size = h_q // h_kv
        if local_s % seq_chunk_size != 0:
            raise ValueError("seq_chunk_size must divide local_s")
        if head_chunk_size <= 0 or h_q % head_chunk_size != 0:
            raise ValueError("head_chunk_size must be positive and divide h_q")
        requested_head_chunk_size = head_chunk_size
        # Reduce the head chunk when needed so each rank keeps complete GQA groups.
        head_chunk_size = select_head_chunk_size(
            h_q,
            h_kv,
            requested_head_chunk_size,
            CP,
        )
        kv_head_chunk_size = head_chunk_size // group_size
        if kv_chunk_cache_slots < 1:
            raise ValueError("kv_chunk_cache_slots must be greater than 0")
        if q_chunk_cache_slots < 0:
            raise ValueError("q_chunk_cache_slots cannot be negative")
        if backward_mode not in ("high_accuracy", "low_memory"):
            raise ValueError("backward_mode must be 'high_accuracy' or 'low_memory'")

        compute_device = torch.device(compute_device)
        compute_stream = cp_stream
        current_stream = torch.npu.current_stream()
        qk_prefetch_stream.wait_stream(current_stream)
        v_prefetch_stream.wait_stream(current_stream)
        softmax_scale = d_qk**-0.5 if softmax_scale is None else softmax_scale
        fa_heads = head_chunk_size // CP
        npu_io = (
            not is_forward_only
            and not is_cpu_output
            and query_layer.device.type == 'npu'
            and key_layer.device.type == 'npu'
            and value_layer.device.type == 'npu'
        )
        retain_full_qkv = (
            npu_io and backward_mode == "high_accuracy" and head_chunk_size == h_q and d_qk == d_v
        )
        save_inputs_on_npu = retain_full_qkv

        compute_step_order = make_compute_step_order(
            b,
            h_q=h_q,
            h_kv=h_kv,
            local_s=local_s,
            d=d_qk,
            CP=CP,
            head_chunk_size=head_chunk_size,
            seq_chunk_size=seq_chunk_size,
            dim_chunk_size=d_qk,
            q_d2d_mode="alltoall",
            kv_d2d_mode="alltoall",
        )
        compute_step_order = convert_compute_step_order(compute_step_order, d_v)
        kv_chunk_cache_slots = assign_kv_chunk_cache_slots(
            compute_step_order,
            kv_chunk_cache_slots,
        )
        kv_chunk_cache = KVChunkCache(
            key_layer,
            value_layer,
            compute_step_order[0],
            kv_chunk_cache_slots,
            qk_prefetch_stream,
            v_prefetch_stream,
            compute_device,
            cp_group,
            q_head_chunk_size=head_chunk_size,
        )

        output_cpu = torch.empty(
            (local_s, b, h_q, d_v),
            dtype=value_layer.dtype,
            device='cpu',
            pin_memory=True,
        )
        saved_o = {} if not is_forward_only else None
        saved_softmax_max = {} if not is_forward_only else None
        saved_softmax_sum = {} if not is_forward_only else None
        if retain_full_qkv:
            with torch.npu.stream(compute_stream):
                saved_full_q = torch.empty(
                    local_s * CP,
                    b,
                    fa_heads,
                    d_qk,
                    dtype=query_layer.dtype,
                    device=compute_device,
                )
                saved_full_k = torch.empty(
                    local_s * CP,
                    b,
                    kv_chunk_cache.gathered_heads,
                    d_qk,
                    dtype=key_layer.dtype,
                    device=compute_device,
                )
                saved_full_v = torch.empty(
                    local_s * CP,
                    b,
                    kv_chunk_cache.gathered_heads,
                    d_v,
                    dtype=value_layer.dtype,
                    device=compute_device,
                )
            saved_kv_chunk_ids = set()
        else:
            saved_full_q = saved_full_k = saved_full_v = None
            saved_kv_chunk_ids = None

        group_start_index = 0
        pending_q = prefetch_q(
            query_layer,
            compute_step_order[0],
            qk_prefetch_stream,
            compute_device,
            cp_group,
        )
        while group_start_index < len(compute_step_order):
            first_compute_step = compute_step_order[group_start_index]
            group_key = (first_compute_step['hc_id'], first_compute_step['q_sc_id'])
            group_end_index = group_start_index + 1
            while group_end_index < len(compute_step_order):
                compute_step = compute_step_order[group_end_index]
                if (compute_step['hc_id'], compute_step['q_sc_id']) != group_key:
                    break
                group_end_index += 1

            q_chunk, q_ready_stream = pending_q
            pending_q = None
            compute_stream.wait_stream(q_ready_stream)
            q_chunk.record_stream(compute_stream)
            if retain_full_qkv:
                q_sc_id = first_compute_step['q_sc_id']
                q_start = q_sc_id * seq_chunk_size * CP
                q_end = q_start + seq_chunk_size * CP
                with torch.npu.stream(compute_stream):
                    saved_full_q[q_start:q_end].copy_(q_chunk, non_blocking=True)
            pending_kv = prefetch_kv(kv_chunk_cache, first_compute_step)

            o_chunk_this_q = None
            prev_softmax_max = None
            prev_softmax_sum = None

            for compute_step_index in range(group_start_index, group_end_index):
                compute_step = compute_step_order[compute_step_index]
                k_chunk, v_chunk, kv_ready_event, kv_slot_id = pending_kv
                compute_stream.wait_event(kv_ready_event)
                if retain_full_qkv and compute_step['k_sc_id'] not in saved_kv_chunk_ids:
                    k_sc_id = compute_step['k_sc_id']
                    k_start = k_sc_id * seq_chunk_size * CP
                    k_end = k_start + seq_chunk_size * CP
                    with torch.npu.stream(compute_stream):
                        saved_full_k[k_start:k_end].copy_(k_chunk, non_blocking=True)
                        saved_full_v[k_start:k_end].copy_(v_chunk, non_blocking=True)
                    saved_kv_chunk_ids.add(k_sc_id)

                with torch.npu.stream(compute_stream):
                    o_chunk, softmax_max, softmax_sum = run_fa(
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        fa_heads,
                        compute_step['q_sc_id'] == compute_step['k_sc_id'],
                        attention_mask,
                        softmax_scale,
                    )
                kv_chunk_cache.record_use(kv_slot_id, compute_stream)

                # Prefetch the next KV compute step after submitting FA, overlapping it with the current FA/online update.
                if compute_step_index + 1 < group_end_index:
                    pending_kv = prefetch_kv(
                        kv_chunk_cache,
                        compute_step_order[compute_step_index + 1],
                    )
                elif group_end_index < len(compute_step_order):
                    pending_q = prefetch_q(
                        query_layer,
                        compute_step_order[group_end_index],
                        qk_prefetch_stream,
                        compute_device,
                        cp_group,
                    )

                with torch.npu.stream(compute_stream):
                    o_chunk_this_q, prev_softmax_max, prev_softmax_sum = online_update_o_chunk(
                        o_chunk_this_q,
                        o_chunk,
                        softmax_max,
                        softmax_sum,
                        prev_softmax_max,
                        prev_softmax_sum,
                    )

            if not is_forward_only:
                # Move the final O/LSE needed by backward to CPU first, avoiding long-term NPU residency.
                compute_stream.synchronize()
                saved_o[group_key] = to_pinned_cpu(o_chunk_this_q)
                saved_softmax_max[group_key] = to_pinned_cpu(prev_softmax_max)
                saved_softmax_sum[group_key] = to_pinned_cpu(prev_softmax_sum)

            # The current o_chunk is complete; transform it back to the local layout.
            with torch.npu.stream(compute_stream):
                if pending_q is not None:
                    # Within the same HCCL group, submit the next Q forward A2A before the current O reverse A2A.
                    compute_stream.wait_stream(pending_q[1])
                o_chunk_this_q = o_chunk_this_q.view(
                    seq_chunk_size * CP,
                    b,
                    fa_heads,
                    d_v,
                )
                o_chunk_local = all_to_all(
                    input_=o_chunk_this_q,
                    process_group=cp_group,
                    scatter_dim=0,
                    gather_dim=2,
                    gather_size=head_chunk_size,
                )
            qk_prefetch_stream.wait_stream(compute_stream)

            h_range, s_range, _ = first_compute_step['q_hsd_range']
            general_offload_stream.wait_stream(compute_stream)
            with torch.npu.stream(general_offload_stream):
                output_cpu[
                    s_range[0] : s_range[1],
                    :,
                    h_range[0] : h_range[1],
                    :,
                ].copy_(o_chunk_local, non_blocking=True)
                o_chunk_local.record_stream(general_offload_stream)

            group_start_index = group_end_index

        ctx.is_forward_only = is_forward_only
        if not is_forward_only:
            if save_inputs_on_npu:
                saved_query_layer = query_layer
                saved_key_layer = key_layer
                saved_value_layer = value_layer
            else:
                saved_query_layer = to_pinned_cpu(query_layer)
                saved_key_layer = to_pinned_cpu(key_layer)
                saved_value_layer = to_pinned_cpu(value_layer)
            saved_group_keys = tuple(saved_o)
            pack_full_state = save_inputs_on_npu
            if pack_full_state:
                q_keys = [(0, q_sc_id) for q_sc_id in range(local_s // seq_chunk_size)]
                saved_group_tensors = [
                    to_pinned_cpu(torch.cat([saved_o[key] for key in q_keys], dim=0)),
                    to_pinned_cpu(torch.cat([saved_softmax_max[key] for key in q_keys], dim=2)),
                    to_pinned_cpu(torch.cat([saved_softmax_sum[key] for key in q_keys], dim=2)),
                ]
            else:
                saved_group_tensors = []
                for group_key in saved_group_keys:
                    saved_group_tensors.extend(
                        (
                            saved_o[group_key],
                            saved_softmax_max[group_key],
                            saved_softmax_sum[group_key],
                        )
                    )
            ctx.save_for_backward(
                saved_query_layer,
                saved_key_layer,
                saved_value_layer,
                *saved_group_tensors,
            )
            ctx.compute_step_order = compute_step_order
            ctx.saved_group_index = {
                group_key: index for index, group_key in enumerate(saved_group_keys)
            }
            ctx.full_state_is_packed = pack_full_state
            ctx.saved_full_qkv = (
                (saved_full_q, saved_full_k, saved_full_v) if retain_full_qkv else None
            )
            ctx.cp_group = cp_group
            ctx.cp_stream = cp_stream
            ctx.qk_prefetch_stream = qk_prefetch_stream
            ctx.v_prefetch_stream = v_prefetch_stream
            ctx.general_offload_stream = general_offload_stream
            ctx.final_hdc_stream = final_hdc_stream
            ctx.compute_device = compute_device
            ctx.attention_mask = attention_mask
            ctx.softmax_scale = softmax_scale
            ctx.fa_heads = fa_heads
            ctx.group_size = group_size
            ctx.requested_head_chunk_size = requested_head_chunk_size
            ctx.kv_head_chunk_size = kv_head_chunk_size
            ctx.kv_is_routed = kv_chunk_cache.is_routed
            ctx.kv_head_index = kv_chunk_cache.kv_head_index
            ctx.seq_chunk_size = seq_chunk_size
            ctx.head_chunk_size = head_chunk_size
            ctx.kv_chunk_cache_slots = kv_chunk_cache_slots
            ctx.q_chunk_cache_slots = q_chunk_cache_slots
            ctx.q_prefetch_stream = q_prefetch_stream
            ctx.kv_prefetch_stream = kv_prefetch_stream
            ctx.is_cpu_output = is_cpu_output
            ctx.backward_mode = backward_mode
            ctx.save_inputs_on_npu = save_inputs_on_npu
            ctx.input_devices = (
                query_layer.device,
                key_layer.device,
                value_layer.device,
            )

        if is_cpu_output:
            general_offload_stream.synchronize()
            return output_cpu

        final_hdc_stream.wait_stream(general_offload_stream)
        with torch.npu.stream(final_hdc_stream):
            output = torch.empty(
                output_cpu.shape,
                dtype=output_cpu.dtype,
                device=compute_device,
            )
            output.copy_(output_cpu, non_blocking=True)
        final_hdc_stream.synchronize()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_forward_only:
            raise RuntimeError("SwapAttn backward requested while is_forward_only=True")
        saved_tensors = ctx.saved_tensors
        save_inputs_on_npu = getattr(ctx, 'save_inputs_on_npu', False)
        if save_inputs_on_npu:
            query_layer = saved_tensors[0]
            key_layer = saved_tensors[1]
            value_layer = saved_tensors[2]
        else:
            query_layer = to_pinned_cpu(saved_tensors[0])
            key_layer = to_pinned_cpu(saved_tensors[1])
            value_layer = to_pinned_cpu(saved_tensors[2])
        saved_group_tensors = saved_tensors[3:]
        compute_device = ctx.compute_device

        grad_output_device = grad_output.device.type
        expected_grad_output_device = "cpu" if ctx.is_cpu_output else "npu"
        if grad_output_device != expected_grad_output_device:
            raise ValueError(
                f"grad_output must be on {expected_grad_output_device}, "
                f"but is on {grad_output.device}"
            )
        if grad_output_device == "cpu":
            grad_output = to_pinned_cpu(grad_output)
        elif grad_output.device.type != compute_device.type:
            raise ValueError(
                f"grad_output must be on {compute_device}, but is on {grad_output.device}"
            )
        else:
            grad_output = grad_output.contiguous()

        local_s, b, h_q, dim_qk = query_layer.shape
        _, _, h_kv, _ = key_layer.shape
        _, _, _, dim_v = value_layer.shape
        CP = dist.get_world_size(ctx.cp_group)
        compute_stream = ctx.cp_stream
        q_prefetch_stream = ctx.q_prefetch_stream
        kv_prefetch_stream = ctx.kv_prefetch_stream
        backward_comm_stream = ctx.qk_prefetch_stream
        offload_stream = ctx.general_offload_stream
        final_hdc_stream = ctx.final_hdc_stream
        current_stream = torch.npu.current_stream()
        compute_stream.wait_stream(current_stream)
        q_prefetch_stream.wait_stream(current_stream)
        kv_prefetch_stream.wait_stream(current_stream)
        backward_comm_stream.wait_stream(current_stream)
        high_accuracy = ctx.backward_mode == "high_accuracy"

        if high_accuracy and save_inputs_on_npu:
            # For NPU-to-NPU, run full-sequence FA gradients directly to avoid CPU round trips for Q/K/V and gradients.
            gradients = run_full_npu_backward_saved_qkv(
                ctx,
                query_layer,
                key_layer,
                value_layer,
                grad_output,
                saved_group_tensors,
                compute_device,
                current_stream,
            )
            return (
                *gradients,
                *([None] * 26),
            )

        backward_compute_order = make_backward_compute_step_order(
            b,
            h_q=h_q,
            h_kv=h_kv,
            local_s=local_s,
            d=dim_qk,
            CP=CP,
            is_k0_first=high_accuracy,
            head_chunk_size=ctx.head_chunk_size,
            seq_chunk_size=ctx.seq_chunk_size,
            dim_chunk_size=dim_qk,
            q_d2d_mode="alltoall",
            kv_d2d_mode="alltoall",
        )
        backward_compute_step_order = convert_compute_step_order(
            backward_compute_order,
            dim_v,
        )
        assign_kv_chunk_cache_slots(backward_compute_step_order, 1)
        q_chunk_cache_slots = assign_q_chunk_cache_slots(
            backward_compute_step_order,
            ctx.q_chunk_cache_slots,
        )
        backward_kv_chunk_cache = KVChunkCache(
            key_layer,
            value_layer,
            backward_compute_step_order[0],
            1,
            kv_prefetch_stream,
            kv_prefetch_stream,
            compute_device,
            ctx.cp_group,
            comm_stream=backward_comm_stream,
            q_head_chunk_size=ctx.head_chunk_size,
        )
        backward_q_chunk_cache = QChunkCache(
            query_layer,
            grad_output,
            saved_group_tensors,
            ctx.saved_group_index,
            backward_compute_step_order[0],
            q_chunk_cache_slots,
            q_prefetch_stream,
            backward_comm_stream,
            compute_device,
            ctx.cp_group,
        )

        head_chunk_num = h_q // ctx.head_chunk_size
        seq_chunk_num = local_s // ctx.seq_chunk_size
        grad_q_slot_num = max(seq_chunk_num - 1, 1)
        gathered_s = ctx.seq_chunk_size * CP
        gathered_q_h = ctx.head_chunk_size // CP
        gathered_kv_h = backward_kv_chunk_cache.gathered_heads
        grad_q_accum_shape = (
            head_chunk_num,
            grad_q_slot_num,
            gathered_s,
            b,
            gathered_q_h,
            dim_qk,
        )
        grad_k_accum_shape = (1, gathered_s, b, gathered_kv_h, dim_qk)
        grad_v_accum_shape = (1, gathered_s, b, gathered_kv_h, dim_v)
        grad_q_accum_numel = torch.Size(grad_q_accum_shape).numel()
        grad_k_accum_numel = 0 if high_accuracy else torch.Size(grad_k_accum_shape).numel()
        grad_v_accum_numel = 0 if high_accuracy else torch.Size(grad_v_accum_shape).numel()
        with torch.npu.stream(compute_stream):
            # low_memory additionally uses a fixed FP32 dKV arena to accumulate across Q chunks.
            grad_accum_workspace = torch.empty(
                grad_q_accum_numel + grad_k_accum_numel + grad_v_accum_numel,
                dtype=torch.float32,
                device=compute_device,
            )
            grad_q_accum = grad_accum_workspace[:grad_q_accum_numel].view(grad_q_accum_shape)
            if high_accuracy:
                grad_k_accum_buffers = None
                grad_v_accum_buffers = None
            else:
                grad_k_accum_buffers = grad_accum_workspace[
                    grad_q_accum_numel : grad_q_accum_numel + grad_k_accum_numel
                ].view(grad_k_accum_shape)
                grad_v_accum_buffers = grad_accum_workspace[
                    grad_q_accum_numel + grad_k_accum_numel :
                ].view(grad_v_accum_shape)

            backward_q_group_chunks = seq_chunk_num if high_accuracy else 1
            max_group_s = backward_q_group_chunks * gathered_s
            grouped_q = torch.empty(
                max_group_s,
                b,
                gathered_q_h,
                dim_qk,
                dtype=query_layer.dtype,
                device=compute_device,
            )
            grouped_grad_output = torch.empty(
                max_group_s,
                b,
                gathered_q_h,
                dim_v,
                dtype=grad_output.dtype,
                device=compute_device,
            )
            saved_o_shape = saved_group_tensors[0].shape
            grouped_o = torch.empty(
                (max_group_s, *saved_o_shape[1:]),
                dtype=saved_group_tensors[0].dtype,
                device=compute_device,
            )
            saved_softmax_shape = saved_group_tensors[1].shape
            grouped_softmax_max = torch.empty(
                (*saved_softmax_shape[:2], max_group_s, *saved_softmax_shape[3:]),
                dtype=saved_group_tensors[1].dtype,
                device=compute_device,
            )
            grouped_softmax_sum = torch.empty_like(grouped_softmax_max)

        # Store CPU gradients in chunk-major layout so every D2H destination is contiguous.
        grad_query_chunks_cpu = make_chunked_grad_cpu(
            head_chunk_num,
            seq_chunk_num,
            ctx.seq_chunk_size,
            b,
            ctx.head_chunk_size,
            dim_qk,
            query_layer.dtype,
        )
        grad_key_chunks_cpu = make_chunked_grad_cpu(
            head_chunk_num,
            seq_chunk_num,
            ctx.seq_chunk_size,
            b,
            ctx.kv_head_chunk_size,
            dim_qk,
            key_layer.dtype,
        )
        grad_value_chunks_cpu = make_chunked_grad_cpu(
            head_chunk_num,
            seq_chunk_num,
            ctx.seq_chunk_size,
            b,
            ctx.kv_head_chunk_size,
            dim_v,
            value_layer.dtype,
        )
        local_chunk_s = ctx.seq_chunk_size
        q_reverse_numel = CP * local_chunk_s * b * gathered_q_h * dim_qk
        kv_reverse_numel = (
            CP
            * local_chunk_s
            * b
            * (ctx.kv_head_chunk_size if ctx.kv_is_routed else gathered_kv_h)
            * (dim_qk + dim_v)
        )
        reverse_buffer_numel = max(q_reverse_numel, kv_reverse_numel)
        reverse_send = torch.empty(
            reverse_buffer_numel,
            dtype=query_layer.dtype,
            device=compute_device,
        )
        reverse_recv = torch.empty_like(reverse_send)
        grad_q_slot_reusable = [torch.npu.Event() for _ in range(head_chunk_num)]
        grad_kv_accum_reusable = torch.npu.Event() if not high_accuracy else None
        grad_kv_accum_in_use = False
        reverse_offloaded = torch.npu.Event()
        reverse_offload_in_use = False

        group_start_index = 0
        prefetched_q = {}
        pending_kv = prefetch_kv(
            backward_kv_chunk_cache,
            backward_compute_step_order[0],
        )
        while group_start_index < len(backward_compute_step_order):
            first_compute_step = backward_compute_step_order[group_start_index]
            group_key = (
                first_compute_step['hc_id'],
                first_compute_step['k_sc_id'],
            )
            group_end_index = group_start_index + 1
            while group_end_index < len(backward_compute_step_order):
                compute_step = backward_compute_step_order[group_end_index]
                if (
                    compute_step['hc_id'],
                    compute_step['k_sc_id'],
                ) != group_key:
                    break
                group_end_index += 1

            pending_q = prefetched_q.pop(group_start_index, None)
            if pending_q is None:
                pending_q = backward_q_chunk_cache.load(first_compute_step)
            k_chunk, v_chunk, kv_ready_event, kv_slot_id = pending_kv
            compute_stream.wait_event(kv_ready_event)
            if high_accuracy:
                grad_k_chunk_accum = None
                grad_v_chunk_accum = None
            else:
                if grad_kv_accum_in_use:
                    compute_stream.wait_event(grad_kv_accum_reusable)
                grad_k_chunk_accum = grad_k_accum_buffers[0]
                grad_v_chunk_accum = grad_v_accum_buffers[0]
            is_first_grad_kv = True
            for tile_start_index in range(
                group_start_index,
                group_end_index,
                backward_q_group_chunks,
            ):
                tile_end_index = min(
                    tile_start_index + backward_q_group_chunks,
                    group_end_index,
                )
                tile_q_ids = [
                    backward_compute_step_order[index]['q_sc_id']
                    for index in range(tile_start_index, tile_end_index)
                ]
                diagonal_q_id = first_compute_step['k_sc_id']
                tile_has_diagonal = diagonal_q_id in tile_q_ids
                ordered_q_ids = (
                    [diagonal_q_id]
                    + [q_sc_id for q_sc_id in tile_q_ids if q_sc_id != diagonal_q_id]
                    if tile_has_diagonal
                    else tile_q_ids
                )
                q_offsets = {
                    q_sc_id: index * gathered_s for index, q_sc_id in enumerate(ordered_q_ids)
                }
                tile_s = len(tile_q_ids) * gathered_s

                for compute_step_index in range(tile_start_index, tile_end_index):
                    compute_step = backward_compute_step_order[compute_step_index]
                    (
                        q_chunk,
                        grad_output_chunk,
                        o_chunk_this_q,
                        softmax_max,
                        softmax_sum,
                        q_ready_event,
                        q_slot_id,
                    ) = pending_q
                    q_sc_id = compute_step['q_sc_id']
                    if q_ready_event is not None:
                        compute_stream.wait_event(q_ready_event)
                    q_offset = q_offsets[q_sc_id]
                    with torch.npu.stream(compute_stream):
                        # Keep the Z-shaped load order; place the diagonal Q first for left-up causal masking.
                        q_end = q_offset + gathered_s
                        grouped_q[q_offset:q_end].copy_(q_chunk)
                        grouped_grad_output[q_offset:q_end].copy_(grad_output_chunk)
                        grouped_o[q_offset:q_end].copy_(o_chunk_this_q)
                        grouped_softmax_max[:, :, q_offset:q_end].copy_(softmax_max)
                        grouped_softmax_sum[:, :, q_offset:q_end].copy_(softmax_sum)
                    if q_chunk_cache_slots == 0 or compute_step.get('q_slot_last_use', False):
                        backward_q_chunk_cache.record_use(q_slot_id, compute_stream)

                    q_prefetch_step_index = compute_step.get('q_prefetch_step_index')
                    if q_prefetch_step_index is not None:
                        prefetched_q[q_prefetch_step_index] = backward_q_chunk_cache.load(
                            backward_compute_step_order[q_prefetch_step_index]
                        )

                    if compute_step_index + 1 < group_end_index:
                        next_compute_step_index = compute_step_index + 1
                        pending_q = prefetched_q.pop(
                            next_compute_step_index,
                            None,
                        )
                        if pending_q is None:
                            pending_q = backward_q_chunk_cache.load(
                                backward_compute_step_order[next_compute_step_index]
                            )

                with torch.npu.stream(compute_stream):
                    # Reduce dKV inside the FA kernel for each tile, then continue FP32 accumulation across tiles.
                    grad_q_group, grad_k_chunk, grad_v_chunk = run_fa_backward(
                        grouped_q[:tile_s],
                        k_chunk,
                        v_chunk,
                        grouped_grad_output[:tile_s],
                        ctx.fa_heads,
                        tile_has_diagonal,
                        ctx.attention_mask,
                        ctx.softmax_scale,
                        grouped_softmax_max[:, :, :tile_s],
                        grouped_softmax_sum[:, :, :tile_s],
                        grouped_o[:tile_s],
                    )
                    if high_accuracy:
                        grad_k_chunk_accum = grad_k_chunk
                        grad_v_chunk_accum = grad_v_chunk
                    elif is_first_grad_kv:
                        grad_k_chunk_accum.copy_(grad_k_chunk)
                        grad_v_chunk_accum.copy_(grad_v_chunk)
                        is_first_grad_kv = False
                    else:
                        grad_k_chunk_accum.add_(grad_k_chunk)
                        grad_v_chunk_accum.add_(grad_v_chunk)

                d_q_step_indices = list(range(tile_start_index, tile_end_index))
                if high_accuracy and first_compute_step['k_sc_id'] == 0:
                    # Release the shared q0 slot first so q1 can reuse it.
                    d_q_step_indices.sort(
                        key=lambda index: backward_compute_step_order[index]['q_sc_id']
                    )
                for compute_step_index in d_q_step_indices:
                    compute_step = backward_compute_step_order[compute_step_index]
                    q_sc_id = compute_step['q_sc_id']
                    q_offset = q_offsets[q_sc_id]
                    grad_q_chunk_step = grad_q_group[q_offset : q_offset + gathered_s]
                    grad_q_slot_id = q_sc_id - 1 if q_sc_id else 0
                    grad_q_chunk_accum = grad_q_accum[compute_step['hc_id'], grad_q_slot_id]
                    wait_for_shared_slot = (
                        high_accuracy and q_sc_id == 1 and compute_step['k_sc_id'] == 0
                    ) or (not high_accuracy and q_sc_id == 0)
                    if wait_for_shared_slot and seq_chunk_num > 1:
                        compute_stream.wait_event(grad_q_slot_reusable[compute_step['hc_id']])
                    is_first_grad_q = (
                        compute_step['k_sc_id'] == 0
                        if high_accuracy
                        else q_sc_id == compute_step['k_sc_id']
                    )
                    with torch.npu.stream(compute_stream):
                        if is_first_grad_q:
                            grad_q_chunk_accum.copy_(grad_q_chunk_step)
                        else:
                            grad_q_chunk_accum.add_(grad_q_chunk_step)

                    is_grad_q_complete = (
                        q_sc_id == compute_step['k_sc_id']
                        if high_accuracy
                        else compute_step['k_sc_id'] == 0
                    )
                    if is_grad_q_complete:
                        # All K chunks for the current Q are complete; use fixed buffers for reverse A2A and offload.
                        gathered_s, chunk_b, gathered_h, grad_q_dim = grad_q_chunk_accum.shape
                        local_chunk_s = gathered_s // CP
                        grad_q_buffer_shape = (
                            CP,
                            local_chunk_s,
                            chunk_b,
                            gathered_h,
                            grad_q_dim,
                        )
                        grad_q_buffer_numel = grad_q_chunk_accum.numel()
                        grad_q_send = reverse_send[:grad_q_buffer_numel].view(grad_q_buffer_shape)
                        grad_q_recv = reverse_recv[:grad_q_buffer_numel].view(grad_q_buffer_shape)

                        backward_comm_stream.wait_stream(compute_stream)
                        if reverse_offload_in_use:
                            backward_comm_stream.wait_event(reverse_offloaded)
                        with torch.npu.stream(backward_comm_stream):
                            grad_q_send.copy_(grad_q_chunk_accum.view(grad_q_send.shape))
                            release_shared_slot = q_sc_id == 0 if high_accuracy else q_sc_id == 1
                            if release_shared_slot and seq_chunk_num > 1:
                                grad_q_slot_reusable[compute_step['hc_id']].record(
                                    backward_comm_stream
                                )
                            dist.all_to_all_single(
                                grad_q_recv,
                                grad_q_send,
                                group=ctx.cp_group,
                            )
                            grad_q_local_offload = reverse_send[:grad_q_buffer_numel].view(
                                local_chunk_s,
                                chunk_b,
                                CP * gathered_h,
                                grad_q_dim,
                            )
                            grad_q_local_offload.view(
                                local_chunk_s,
                                chunk_b,
                                CP,
                                gathered_h,
                                grad_q_dim,
                            ).copy_(grad_q_recv.permute(1, 2, 0, 3, 4))
                        offload_stream.wait_stream(backward_comm_stream)
                        with torch.npu.stream(offload_stream):
                            grad_query_chunks_cpu[
                                compute_step['hc_id'], compute_step['q_sc_id']
                            ].copy_(grad_q_local_offload, non_blocking=True)
                            reverse_offloaded.record(offload_stream)
                        reverse_offload_in_use = True

                del grad_q_group, grad_k_chunk, grad_v_chunk

            backward_kv_chunk_cache.record_use(kv_slot_id, compute_stream)
            if group_end_index < len(backward_compute_step_order):
                # The current K has been consumed; transfer the next K early and overlap it with this group's dKV return.
                pending_kv = prefetch_kv(
                    backward_kv_chunk_cache,
                    backward_compute_step_order[group_end_index],
                )

            gathered_s, chunk_b, gathered_kv_h, grad_k_dim = grad_k_chunk_accum.shape
            grad_v_dim = grad_v_chunk_accum.shape[-1]
            local_chunk_s = gathered_s // CP

            # Once dKV is complete, return it to the source sequence shard using the KV routing mode, then offload by chunk.
            backward_comm_stream.wait_stream(compute_stream)
            if reverse_offload_in_use:
                backward_comm_stream.wait_event(reverse_offloaded)
            with torch.npu.stream(backward_comm_stream):
                if ctx.kv_is_routed:
                    send_shape = (
                        CP,
                        local_chunk_s,
                        chunk_b,
                        ctx.kv_head_chunk_size,
                        grad_k_dim + grad_v_dim,
                    )
                    recv_shape = (
                        local_chunk_s,
                        chunk_b,
                        ctx.kv_head_chunk_size,
                        grad_k_dim + grad_v_dim,
                    )
                    send_numel = torch.Size(send_shape).numel()
                    recv_numel = torch.Size(recv_shape).numel()
                    grad_kv_send = reverse_send[:send_numel].view(send_shape)
                    grad_kv_recv = reverse_recv[:recv_numel].view(recv_shape)
                    grad_kv_send.zero_()
                    grad_kv_send[..., ctx.kv_head_index : ctx.kv_head_index + 1, :grad_k_dim].copy_(
                        grad_k_chunk_accum.view(
                            CP,
                            local_chunk_s,
                            chunk_b,
                            1,
                            grad_k_dim,
                        )
                    )
                    grad_kv_send[..., ctx.kv_head_index : ctx.kv_head_index + 1, grad_k_dim:].copy_(
                        grad_v_chunk_accum.view(
                            CP,
                            local_chunk_s,
                            chunk_b,
                            1,
                            grad_v_dim,
                        )
                    )
                else:
                    buffer_shape = (
                        CP,
                        local_chunk_s,
                        chunk_b,
                        gathered_kv_h,
                        grad_k_dim + grad_v_dim,
                    )
                    buffer_numel = torch.Size(buffer_shape).numel()
                    grad_kv_send = reverse_send[:buffer_numel].view(buffer_shape)
                    grad_kv_recv = reverse_recv[:buffer_numel].view(buffer_shape)
                    grad_kv_send[..., :grad_k_dim].copy_(
                        grad_k_chunk_accum.view(
                            CP,
                            local_chunk_s,
                            chunk_b,
                            gathered_kv_h,
                            grad_k_dim,
                        )
                    )
                    grad_kv_send[..., grad_k_dim:].copy_(
                        grad_v_chunk_accum.view(
                            CP,
                            local_chunk_s,
                            chunk_b,
                            gathered_kv_h,
                            grad_v_dim,
                        )
                    )
                if high_accuracy:
                    grad_k_chunk_accum.record_stream(backward_comm_stream)
                    grad_v_chunk_accum.record_stream(backward_comm_stream)
                else:
                    grad_kv_accum_reusable.record(backward_comm_stream)
                    grad_kv_accum_in_use = True
                if ctx.kv_is_routed:
                    dist.reduce_scatter_tensor(
                        grad_kv_recv,
                        grad_kv_send,
                        group=ctx.cp_group,
                    )
                    grad_k_local_offload = grad_kv_recv[..., :grad_k_dim]
                    grad_v_local_offload = grad_kv_recv[..., grad_k_dim:]
                else:
                    dist.all_to_all_single(
                        grad_kv_recv,
                        grad_kv_send,
                        group=ctx.cp_group,
                    )
                    local_kv_h = CP * gathered_kv_h
                    grad_k_local_numel = local_chunk_s * chunk_b * local_kv_h * grad_k_dim
                    grad_v_local_numel = local_chunk_s * chunk_b * local_kv_h * grad_v_dim
                    grad_k_local_offload = reverse_send[:grad_k_local_numel].view(
                        local_chunk_s, chunk_b, local_kv_h, grad_k_dim
                    )
                    grad_v_local_offload = reverse_send[
                        grad_k_local_numel : grad_k_local_numel + grad_v_local_numel
                    ].view(local_chunk_s, chunk_b, local_kv_h, grad_v_dim)
                    grad_k_local_offload.view(
                        local_chunk_s,
                        chunk_b,
                        CP,
                        gathered_kv_h,
                        grad_k_dim,
                    ).copy_(grad_kv_recv[..., :grad_k_dim].permute(1, 2, 0, 3, 4))
                    grad_v_local_offload.view(
                        local_chunk_s,
                        chunk_b,
                        CP,
                        gathered_kv_h,
                        grad_v_dim,
                    ).copy_(grad_kv_recv[..., grad_k_dim:].permute(1, 2, 0, 3, 4))

            offload_stream.wait_stream(backward_comm_stream)
            with torch.npu.stream(offload_stream):
                grad_key_chunks_cpu[
                    first_compute_step['hc_id'], first_compute_step['k_sc_id']
                ].copy_(grad_k_local_offload, non_blocking=True)
                grad_value_chunks_cpu[
                    first_compute_step['hc_id'], first_compute_step['k_sc_id']
                ].copy_(grad_v_local_offload, non_blocking=True)
                reverse_offloaded.record(offload_stream)
            reverse_offload_in_use = True
            del grad_k_chunk_accum, grad_v_chunk_accum
            group_start_index = group_end_index

        offload_stream.synchronize()
        grad_cpu = (
            materialize_chunked_grad_cpu(grad_query_chunks_cpu),
            materialize_chunked_grad_cpu(grad_key_chunks_cpu),
            materialize_chunked_grad_cpu(grad_value_chunks_cpu),
        )
        if all(input_device.type == 'cpu' for input_device in ctx.input_devices):
            gradients = grad_cpu
        else:
            # Release the compute workspace before allocating final NPU gradients to avoid overlapping peaks.
            del grad_q_accum
            if not high_accuracy:
                del grad_k_accum_buffers, grad_v_accum_buffers
            del grad_accum_workspace, reverse_send, reverse_recv
            del grouped_q, grouped_grad_output, grouped_o
            del grouped_softmax_max, grouped_softmax_sum
            del grad_q_local_offload, grad_k_local_offload, grad_v_local_offload
            del backward_q_chunk_cache, backward_kv_chunk_cache
            del q_chunk, k_chunk, v_chunk, grad_output_chunk, o_chunk_this_q
            del softmax_max, softmax_sum, grad_q_chunk_accum
            del grad_q_send, grad_q_recv, grad_kv_send, grad_kv_recv
            del pending_q, pending_kv, prefetched_q
            gradients = []
            with torch.npu.stream(final_hdc_stream):
                for grad, input_device in zip(grad_cpu, ctx.input_devices):
                    if input_device.type == 'cpu':
                        gradients.append(grad)
                    else:
                        grad_device = torch.empty_like(grad, device=input_device)
                        grad_device.copy_(grad, non_blocking=True)
                        gradients.append(grad_device)
            final_hdc_stream.synchronize()
            current_stream.wait_stream(final_hdc_stream)
        return (
            *gradients,
            # Forward has 26 non-tensor parameters in addition to Q/K/V.
            *([None] * 26),
        )
