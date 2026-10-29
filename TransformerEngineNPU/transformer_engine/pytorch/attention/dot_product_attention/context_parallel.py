# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

from abc import ABC, abstractmethod
from typing import List

import torch

from .kvallgather_context_parallel import (
    AttnFuncWithCPAndKVAllGatherForSBHD,
    AttnFuncWithCPAndKVAllGatherForTHD,
)

from .ring_context_parallel import (
    AttnFuncWithCPAndRing,
)

from .ulysses_context_parallel import AttnFuncWithCPAndQKVOA2A
from .swap_cp_attn.swap_cp_attn import SwapCPAttnFunc
from .utils import prepare_sbhd_format, prepare_thd_format


class BaseCPStrategy(torch.nn.Module, ABC):
    """Base class for all Context Parallelism strategies"""

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_type: str = "self",
        deterministic: bool = False,
    ):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.deterministic = deterministic

    @abstractmethod
    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        qkv_format,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_mask_type,
        max_seqlen_q,
        max_seqlen_kv,
        cp_group,
        cp_global_ranks,
        cp_stream,
        **kwargs,
    ) -> torch.Tensor:
        pass


class UlyssesCPStrategy(BaseCPStrategy):
    """Ulysses-based Context Parallelism Strategy"""

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_type: str = "self",
        deterministic: bool = False,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        gather_size_calculator=None,
    ):
        super().__init__(softmax_scale, attention_dropout, attention_type, deterministic)
        self.ulysses_comm_para = {
            'scatter_idx': scatter_idx,
            'gather_idx': gather_idx,
            'gather_size_calculator': gather_size_calculator,
        }

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        qkv_format,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_mask_type,
        max_seqlen_q,
        max_seqlen_kv,
        cp_group,
        cp_global_ranks,
        cp_stream,
        **kwargs,
    ):
        return AttnFuncWithCPAndQKVOA2A(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            cu_seqlens_q,
            cu_seqlens_kv,
            attn_mask_type,
            self.attention_dropout,
            self.softmax_scale,
            self.deterministic,
            cp_group,
            cp_stream,
            self.ulysses_comm_para,
        )


class SwapCPAttnStrategy(BaseCPStrategy):
    """Swap-based Context Parallelism Strategy"""

    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_type: str = "self",
        deterministic: bool = False,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        gather_size_calculator=None,
    ):
        super().__init__(softmax_scale, attention_dropout, attention_type, deterministic)
        self.ulysses_comm_para = {
            'scatter_idx': scatter_idx,
            'gather_idx': gather_idx,
            'gather_size_calculator': gather_size_calculator,
        }
        # streams for swapattnCP
        self.qk_prefetch_stream = torch.npu.Stream()
        self.v_prefetch_stream = torch.npu.Stream()
        self.general_offload_stream = torch.npu.Stream()
        self.final_hdc_stream = torch.npu.Stream()
        self.q_prefetch_stream = torch.npu.Stream()
        self.kv_prefetch_stream = torch.npu.Stream()

    def forward(  # pylint: disable=arguments-differ
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        qkv_format,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_mask_type,
        max_seqlen_q,
        max_seqlen_kv,
        cp_group,
        cp_global_ranks,
        cp_stream,
        # -- SwapAttn-specific parameters
        seq_chunk_size=None,  # seq_chunk_size before D2D; [sc*CP] after all-to-all
        head_chunk_size=None,  # head_chunk_size before D2D; [hc/CP] after all-to-all
        q_chunk_cache_slots: int = 2,  # backward only: 0 disables cross-step reuse; positive values keep Q chunks resident
        kv_chunk_cache_slots: int = 2,  # forward only: number of resident KV chunk pairs
        seq_layout_mode: str = 'contiguous',
        is_cpu_output: bool = False,  # forward: output on CPU; backward: grad_query_layer, grad_key_layer, and grad_value_layer on CPU
        is_forward_only: bool = False,
        backward_mode: str = "high_accuracy",
        **kwargs,
    ):
        if seq_chunk_size is None:
            seq_chunk_size = query_layer.shape[0]
        if head_chunk_size is None:
            head_chunk_size = query_layer.shape[2]

        # Use the stream supplied by the common CP framework, or fall back to the current stream.
        compute_stream = cp_stream or torch.npu.current_stream()

        # Swap computation always runs on NPU; CPU inputs use the current NPU by default.
        compute_device = "npu"

        return SwapCPAttnFunc.apply(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            cu_seqlens_q,
            cu_seqlens_kv,
            attn_mask_type,
            self.attention_dropout,
            self.softmax_scale,
            self.deterministic,
            cp_group,
            compute_stream,
            self.ulysses_comm_para,
            # -- SwapAttn-specific parameters
            # Chunk parameters
            seq_chunk_size,  # seq_chunk_size before D2D; [sc*CP] after all-to-all
            head_chunk_size,  # head_chunk_size before D2D; [hc/CP] after all-to-all
            kv_chunk_cache_slots,  # forward only: number of resident KV chunk pairs
            seq_layout_mode,
            # Stream parameters
            self.qk_prefetch_stream,  # QK chunk prefetch
            self.v_prefetch_stream,  # V chunk prefetch
            self.general_offload_stream,  # offload each computed chunk output
            self.final_hdc_stream,  # H2D stream when the final output stays on NPU
            # Device parameters
            compute_device,
            is_cpu_output,  # forward: output on CPU; backward: gradients on CPU
            # Forward/backward parameters
            is_forward_only,
            q_chunk_cache_slots,  # backward only: 0 disables cross-step reuse; positive values keep Q chunks resident
            self.q_prefetch_stream,  # Q-side staging and serialized submission of all backward A2A operations
            self.kv_prefetch_stream,  # K/V H2D staging
            backward_mode,  # "high_accuracy": FA processes all causal Q chunks for each KV chunk
        )


class KVAllGatherCPStrategy(BaseCPStrategy):
    """AllGather-based Context Parallelism Strategy"""

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        qkv_format,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_mask_type,
        max_seqlen_q,
        max_seqlen_kv,
        cp_group,
        cp_global_ranks,
        cp_stream,
        **kwargs,
    ):
        if qkv_format == 'sbhd':
            query_layer, key_layer, value_layer, n_head = prepare_sbhd_format(
                query_layer, key_layer, value_layer
            )

            return AttnFuncWithCPAndKVAllGatherForSBHD.apply(
                query_layer,
                key_layer,
                value_layer,
                n_head,
                attention_mask,
                qkv_format,
                attn_mask_type,
                self.attention_dropout,
                self.softmax_scale,
                self.deterministic,
                cp_group,
                cp_stream,
            )

        elif qkv_format == 'thd':
            n_head, cu_seqlens_q, cu_seqlens_kv = prepare_thd_format(
                query_layer, cu_seqlens_q, cu_seqlens_kv
            )

            return AttnFuncWithCPAndKVAllGatherForTHD.apply(
                query_layer,
                key_layer,
                value_layer,
                n_head,
                attention_mask,
                qkv_format,
                attn_mask_type,
                self.attention_dropout,
                self.softmax_scale,
                self.deterministic,
                cp_group,
                cu_seqlens_q,
                cu_seqlens_kv,
            )

        else:
            raise ValueError(f"Unsupported qkv_format: {qkv_format}")


class RingCPStrategy(BaseCPStrategy):
    """P2P-based Context Parallelism Strategy (Ring Attention)"""

    def forward(  # pylint: disable=arguments-differ
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        qkv_format,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_mask_type,
        cp_group,
        cp_global_ranks,
        cp_stream,
        max_seqlen_q,
        max_seqlen_kv,
        cp_para,
        ring_config,
        **kwargs,
    ):
        if qkv_format == 'sbhd':
            query_layer, key_layer, value_layer, n_head = prepare_sbhd_format(
                query_layer, key_layer, value_layer
            )

        elif qkv_format == 'thd':
            n_head, cu_seqlens_q, cu_seqlens_kv = prepare_thd_format(
                query_layer, cu_seqlens_q, cu_seqlens_kv
            )
        else:
            raise ValueError(f"Unsupported qkv_format: {qkv_format}")

        AttnFuncWithCPAndRing.block_size = query_layer.shape[0]
        AttnFuncWithCPAndRing.batch_size = query_layer.shape[1]

        return AttnFuncWithCPAndRing.apply(
            query_layer,
            key_layer,
            value_layer,
            n_head,
            attention_mask,
            qkv_format,
            attn_mask_type,
            self.attention_dropout,
            self.softmax_scale,
            self.deterministic,
            cp_group,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            cp_para,
            ring_config,
            None,
        )


class CPStrategyFactory:
    """Factory for creating Context Parallelism strategies"""

    _STRATEGY_MAP = {
        "kvallgather_cp_algo": KVAllGatherCPStrategy,
        "megatron_cp_algo": RingCPStrategy,
        "ulysses_cp_algo": UlyssesCPStrategy,
        "swapcpattn_algo": SwapCPAttnStrategy,
    }

    _MEGATRON_ALIAS_MAP = {
        "all_gather": "kvallgather_cp_algo",
        "allgather": "kvallgather_cp_algo",
        "a2a": "ulysses_cp_algo",
        "p2p": "megatron_cp_algo",
    }

    @staticmethod
    def _resolve_strategy_type(strategy_type: str) -> str:
        if strategy_type in CPStrategyFactory._STRATEGY_MAP:
            return strategy_type
        if strategy_type in CPStrategyFactory._MEGATRON_ALIAS_MAP:
            return CPStrategyFactory._MEGATRON_ALIAS_MAP[strategy_type]
        available = CPStrategyFactory.get_available_strategies() + list(
            CPStrategyFactory._MEGATRON_ALIAS_MAP.keys()
        )
        raise ValueError(
            f"Unsupported CP strategy: {strategy_type}. Available strategies: {available}"
        )

    @staticmethod
    def create_strategy(strategy_type: str, **strategy_params) -> BaseCPStrategy:
        resolved_type = CPStrategyFactory._resolve_strategy_type(strategy_type)
        strategy_class = CPStrategyFactory._STRATEGY_MAP[resolved_type]
        return strategy_class(**strategy_params)

    @staticmethod
    def get_available_strategies() -> List[str]:
        return list(CPStrategyFactory._STRATEGY_MAP.keys())
