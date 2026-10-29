# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from contextlib import nullcontext
from typing import Callable, Optional, Tuple, Union, Dict

import torch
import torch_npu
from einops import rearrange


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


def get_fa_config(attn_mask_type: str) -> Dict[str, int]:
    fa_config = {'pre_tokens': 65536, 'next_tokens': 0, 'sparse_mode': 0}

    if attn_mask_type == 'no_mask':
        return fa_config
    elif attn_mask_type in ('causal', 'padding_causal', 'padding,causal', 'causal,padding'):
        fa_config['sparse_mode'] = 2
        return fa_config
    elif attn_mask_type == 'general':
        fa_config['sparse_mode'] = 1
        return fa_config

    return fa_config


class FlashAttention(torch.nn.Module):
    def __init__(
        self,
        softmax_scale: float,
        attention_dropout: float = 0.0,
        attention_dropout_ctx: Optional[Callable] = nullcontext,
        attention_type: str = "self",
        layer_number: Optional[int] = None,
        deterministic: bool = False,
    ):
        super().__init__()

        self.softmax_scale = softmax_scale
        self.attention_dropout_ctx = attention_dropout_ctx
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.layer_number = 1 if layer_number is None else layer_number
        self.deterministic = deterministic

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        qkv_format: str = "sbhd",
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        attn_mask_type: str = "causal",
    ):
        if qkv_format == 'sbhd':
            seq_length, bsz, n_head, head_dim = query_layer.shape

            actual_seq_qlen = None
            actual_seq_kvlen = None
            query_layer, key_layer, value_layer = [
                rearrange(x, 's b h d -> s b (h d)') for x in [query_layer, key_layer, value_layer]
            ]
            shape_order = 'SBH'

        elif qkv_format == 'thd':
            seq_length, n_head, head_dim = query_layer.shape

            if isinstance(cu_seqlens_q, list):
                actual_seq_qlen = cu_seqlens_q
                actual_seq_kvlen = cu_seqlens_kv
            else:
                actual_seq_qlen = cu_seqlens_q.tolist()
                actual_seq_kvlen = cu_seqlens_kv.tolist()

            shape_order = 'TND'

        else:
            raise ValueError(
                f"Unsupported qkv_format: {qkv_format}. Only 'sbhd' and 'thd' are supported."
            )

        fa_config = get_fa_config(attn_mask_type)

        output = torch_npu.npu_fusion_attention(  # pylint: disable=no-member
            query_layer,
            key_layer,
            value_layer,
            n_head,
            shape_order,
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=self.softmax_scale,
            pre_tockens=fa_config['pre_tokens'],
            next_tockens=fa_config['next_tokens'],
            keep_prob=1 - self.attention_dropout,
            inner_precise=0,
            sparse_mode=fa_config['sparse_mode'],
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )[0]

        return output
