# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch


_CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None

_CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW = None
_CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = None
_CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = None
_CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW = None
_CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP = None


def initialize_context_parallel_group_for_send_recv_overlap_with_te(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        use_cp_send_recv_overlap: bool = True,
):
    if not use_cp_send_recv_overlap:
        return

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    global _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP

    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group_send_recv_overlap = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = group_send_recv_overlap


def initialize_context_parallel_group_for_double_ring_with_te(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        use_2d_tp: bool = False,
        context_parallel_algo: str = "megatron_cp_algo",
        ulysses_degree_in_cp: int = 1,
        cp_window_size: int = 1,
        use_cp_send_recv_overlap: bool = False,
):
    if use_2d_tp:
        return
    if context_parallel_size == 1 or context_parallel_algo not in ['megatron_cp_algo', 'p2p', 'hybrid_cp_algo', 'a2a+p2p']:
        return

    use_hybrid_cp = context_parallel_algo == 'hybrid_cp_algo' and ulysses_degree_in_cp > 1

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    def _initialize_helper_te(
            rank,
            ring_global_ranks,
            window_size
    ):
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV
        global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
        global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW
        global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP

        ring_size = len(ring_global_ranks)
        inter_size = ring_size // window_size
        for wid in range(inter_size):
            intra_ranks = [ring_global_ranks[idx] for idx in range(wid * window_size, (wid + 1) * window_size)]
            intra_group = torch.distributed.new_group(intra_ranks)
            intra_group_for_send_recv_overlap = None
            if use_cp_send_recv_overlap:
                intra_group_for_send_recv_overlap = torch.distributed.new_group(intra_ranks)

            if rank in intra_ranks:
                _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW = intra_ranks
                _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW = intra_group
                _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP = intra_group_for_send_recv_overlap

        for inner_id in range(window_size):
            inter_ranks = [ring_global_ranks[idx] for idx in range(inner_id, ring_size, window_size)]
            if rank in inter_ranks:
                _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = inter_ranks
                break

        for inner_id in range(window_size):
            inter_dkv_ranks = []
            cur_rank = ring_global_ranks[inner_id]
            cur_idx = inner_id
            cur_window = 0
            while cur_rank not in inter_dkv_ranks:
                inter_dkv_ranks.append(cur_rank)
                cur_window = (cur_window + 1) % inter_size
                window_start = cur_window * window_size
                cur_idx = window_start + (cur_idx + 1) % window_size
                cur_rank = ring_global_ranks[cur_idx]

            if rank in inter_dkv_ranks:
                _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = inter_dkv_ranks
                break

    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                cp_ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)

                if use_hybrid_cp:
                    ulysses_degree = ulysses_degree_in_cp
                    if not (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0):
                        raise AssertionError('Invalid ulysses degree configuration')
                    for m in range(ulysses_degree):
                        ring_ranks = [cp_ranks[idx] for idx in range(m, len(cp_ranks), ulysses_degree)]
                        _initialize_helper_te(rank, ring_ranks, cp_window_size)
                else:
                    _initialize_helper_te(rank, cp_ranks, cp_window_size)


def get_context_parallel_group_for_send_recv_overlap(check_initialized=True):
    if check_initialized:
        if _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP is None:
            raise AssertionError('context parallel group for send-recv overlap is not initialized')
    return _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP


def get_ring_group_for_intra_window():
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW
    return _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW


def get_ring_group_for_intra_window_send_recv_overlap():
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP
    return _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP


def get_ring_ranks_for_intra_window():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW
    if _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW is None:
        raise AssertionError('Context parallel ranks for ring intra window not initialized')
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW


def get_ring_ranks_for_inter_window_kv():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV
    if _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV is None:
        raise AssertionError('Context parallel ranks for ring inter window KV not initialized')
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV


def get_ring_ranks_for_inter_window_dkv():
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
    if _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV is None:
        raise AssertionError('Context parallel ranks for ring inter window DKV not initialized')
    return _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV


def destroy_ring_context_parallel_groups():
    global _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV
    global _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW
    global _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP

    _CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = None
    _CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW = None
    _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = None
    _CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = None
    _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW = None
    _CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP = None
