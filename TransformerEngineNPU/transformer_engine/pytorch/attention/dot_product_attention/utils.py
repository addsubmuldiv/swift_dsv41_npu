# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from functools import lru_cache
from typing import Callable, Optional

import torch
from packaging.version import Version as PkgVersion
from torch.distributed import ProcessGroup as dist_group_type
from einops import rearrange

from .ring_parallel_utils import (
    initialize_context_parallel_group_for_send_recv_overlap_with_te,
    initialize_context_parallel_group_for_double_ring_with_te,
    get_context_parallel_group_for_send_recv_overlap,
    get_ring_ranks_for_intra_window,
    get_ring_ranks_for_inter_window_kv,
    get_ring_ranks_for_inter_window_dkv,
    get_ring_group_for_intra_window,
    get_ring_group_for_intra_window_send_recv_overlap,
)


class FlashAttentionUtils:
    """Manage Flash Attention versioning information.

    TransformerEngineNPU does not depend on the ``flash-attn`` package: fused
    attention on Ascend is provided by ``torch_npu`` instead.  This class is
    kept as an API-compatible stub for downstream frameworks that import it,
    and every capability flag is permanently ``False`` so that they never
    select the upstream FlashAttention backend.
    """

    is_installed = False
    version = PkgVersion("0")
    version_required = PkgVersion("2.1.1")
    version_required_blackwell = PkgVersion("2.7.3")
    max_version = PkgVersion("2.8.4")
    v2_plus = False
    v2_1_plus = False
    v2_3_plus = False
    v2_4_plus = False
    v2_4_1_plus = False
    v2_5_plus = False
    v2_5_7_plus = False
    v2_6_0_plus = False
    v2_7_0_plus = False
    warning_printed = False

    v3_is_installed = False
    fa3_version = PkgVersion("0")
    v3_0_0_beta = False
    use_v3 = False
    v3_installation_steps = """\
(1) git clone https://github.com/Dao-AILab/flash-attention.git
(2) cd flash-attention/hopper && python setup.py install"""
    v3_warning_printed = False

    v4_is_installed = False
    fa4_version = PkgVersion("0")
    use_v4 = False
    v4_installation_steps = """\
pip install flash-attn-4==4.0.0b11 nvidia-cutlass-dsl[cu13]"""
    v4_warning_printed = False
    # Set by upstream backends.py when FA4 is installed; never used on NPU.
    v4_validate_head_dims: Callable = None

    @staticmethod
    def is_version_supported(version: PkgVersion, minimum_version: PkgVersion) -> bool:
        """Check whether a Flash Attention v2 version is supported."""
        return minimum_version <= version < FlashAttentionUtils.max_version

    @staticmethod
    def set_flash_attention_version():
        """No-op: the ``flash-attn`` package is not available on NPU."""

    @staticmethod
    def set_flash_attention_3_params():
        """No-op: the ``flash-attn`` package is not available on NPU."""

    @staticmethod
    def set_flash_attention_4_params():
        """No-op: the ``flash-attn`` package is not available on NPU."""


@lru_cache
def get_distributed_world_size(group: Optional[dist_group_type] = None) -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


@lru_cache
def get_distributed_rank(group: Optional[dist_group_type] = None) -> int:
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    return torch.distributed.get_rank(group=group)


def prepare_sbhd_format(query_layer, key_layer, value_layer):
    _, _, n_head, _ = query_layer.shape

    query_layer, key_layer, value_layer = [
        rearrange(x, 's b h d -> s b (h d)') for x in [query_layer, key_layer, value_layer]
    ]

    return query_layer, key_layer, value_layer, n_head


def prepare_thd_format(query_layer, cu_seqlens_q, cu_seqlens_kv):
    _, n_head, _ = query_layer.shape

    if isinstance(cu_seqlens_q, torch.Tensor):
        cu_seqlens_q = cu_seqlens_q.tolist()
    if isinstance(cu_seqlens_kv, torch.Tensor):
        cu_seqlens_kv = cu_seqlens_kv.tolist()

    return n_head, cu_seqlens_q, cu_seqlens_kv


def ring_prepare(self, cu_seqlens_q_padded, cu_seqlens_kv_padded, attn_mask_type=None):
    cp_group = self.cp_group
    cp_size = self.cp_size
    rank = self.cp_rank
    cp_global_ranks = self.cp_global_ranks

    if get_context_parallel_group_for_send_recv_overlap(check_initialized=False) is None:
        initialize_context_parallel_group_for_send_recv_overlap_with_te(
            self.tp_size, 1, self.cp_size, use_cp_send_recv_overlap=True
        )
    if get_ring_group_for_intra_window() is None:
        initialize_context_parallel_group_for_double_ring_with_te(
            self.tp_size,
            1,
            self.cp_size,
            use_cp_send_recv_overlap=True,
            context_parallel_algo=self.cp_comm_type,
            ulysses_degree_in_cp=getattr(self, 'ulysses_degree_in_cp', 1),
            cp_window_size=getattr(self, 'cp_window_size', 1),
        )

    cp_para = dict()
    cp_para['cp_in_bnsd'] = getattr(self, 'megatron_cp_in_bnsd', False)
    cp_para['use_fused_ring_attention_update'] = getattr(
        self, 'use_fused_ring_attention_update', False
    )
    if attn_mask_type is None:
        attn_mask_type = self.attn_mask_type
    cp_para['causal'] = "causal" in attn_mask_type
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    cp_para['cu_seqlens_q_padded'] = cu_seqlens_q_padded
    cp_para['cu_seqlens_kv_padded'] = cu_seqlens_kv_padded

    if self.cp_comm_type in ['megatron_cp_algo', 'p2p', 'hybrid_cp_algo', 'a2a+p2p']:
        cp_para['cp_global_ranks'] = cp_global_ranks
        if getattr(self, 'use_cp_send_recv_overlap', False):
            cp_para['cp_group_for_send_recv_overlap'] = (
                get_context_parallel_group_for_send_recv_overlap()
            )
        else:
            cp_para['cp_group_for_send_recv_overlap'] = None
        cp_para['pse'] = self.pse
        cp_para['pse_type'] = self.pse_type

        if self.cp_size > 1:
            cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
            cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
            cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
            cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
            cp_para['cp_group_for_intra_window_send_recv_overlap'] = (
                get_ring_group_for_intra_window_send_recv_overlap()
            )
            cp_para['cache_policy'] = get_cache_policy(
                self.layer_number,
                getattr(self, 'context_parallel_kv_cache_policy', None),
                getattr(self, 'context_parallel_cache_interval', 0),
            )

    return cp_para


def get_cache_policy(layer_number, cache_policy_init, cache_interval):
    cache_policy = cache_policy_init
    if cache_interval != 0:
        if layer_number % (cache_interval + 1) == 1:
            cache_policy = cache_policy_init
        else:
            cache_policy = None

    return cache_policy
