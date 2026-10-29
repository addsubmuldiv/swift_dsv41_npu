# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.


from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import nn

from .swap_cp_attn import SwapCPAttnFunc
from .swap_cp_attn import run_fa


__all__ = ["SwapFlashAttention"]


class SwapFlashAttention(nn.Module):
    """Single-rank FA-like wrapper around the v3.2 swap-attention executor.

    ``world_size`` must be one.  The constructor only owns the stream set;
    all attention, chunk, device, output, and backward options are per-forward
    arguments.  The constructor must run after ``torch.npu.set_device``.
    """

    def __init__(
        self,
        cp_stream=None,
        qk_prefetch_stream=None,
        v_prefetch_stream=None,
        general_offload_stream=None,
        final_hdc_stream=None,
        q_prefetch_stream=None,
        kv_prefetch_stream=None,
    ):
        super().__init__()

        # Keep one compute stream and one stream for each independent staging
        # or copy path.  Callers may inject streams when integrating with TE.
        self.cp_stream = cp_stream if cp_stream is not None else torch.npu.Stream()
        self.qk_prefetch_stream = (
            qk_prefetch_stream if qk_prefetch_stream is not None else torch.npu.Stream()
        )
        self.v_prefetch_stream = (
            v_prefetch_stream if v_prefetch_stream is not None else torch.npu.Stream()
        )
        self.general_offload_stream = (
            general_offload_stream if general_offload_stream is not None else torch.npu.Stream()
        )
        self.final_hdc_stream = (
            final_hdc_stream if final_hdc_stream is not None else torch.npu.Stream()
        )
        self.q_prefetch_stream = (
            q_prefetch_stream if q_prefetch_stream is not None else torch.npu.Stream()
        )
        self.kv_prefetch_stream = (
            kv_prefetch_stream if kv_prefetch_stream is not None else torch.npu.Stream()
        )

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask=None,
        qkv_format: str = "sbhd",
        attn_mask_type: str = "causal",
        softmax_scale: Optional[float] = None,
        seq_chunk_size: Optional[int] = None,
        head_chunk_size: Optional[int] = None,
        kv_chunk_cache_slots: int = 2,
        compute_device: Optional[Union[str, torch.device]] = None,
        is_cpu_output: bool = False,
        is_forward_only: bool = False,
        q_chunk_cache_slots: int = 2,
        backward_mode: str = "high_accuracy",
    ):
        """Run swap attention with the module-owned stream set."""
        if not dist.is_initialized():
            raise RuntimeError("SwapAttentionFlash requires torch.distributed to be initialized")
        cp_group = dist.group.WORLD
        world_size = dist.get_world_size(cp_group)
        if world_size != 1:
            raise RuntimeError(f"SwapAttentionFlash requires CP=1, but world_size={world_size}")

        if qkv_format.lower() != "sbhd":
            raise ValueError("SwapAttentionFlash currently supports only qkv_format='sbhd'")
        if attn_mask_type != "causal":
            raise ValueError("SwapAttentionFlash currently supports only attn_mask_type='causal'")
        if kv_chunk_cache_slots < 1:
            raise ValueError("kv_chunk_cache_slots must be greater than 0")
        if q_chunk_cache_slots < 0:
            raise ValueError("q_chunk_cache_slots cannot be negative")
        if backward_mode not in ("high_accuracy", "low_memory"):
            raise ValueError("backward_mode must be 'high_accuracy' or 'low_memory'")
        if seq_chunk_size is None:
            seq_chunk_size = query_layer.shape[0]
        if head_chunk_size is None:
            head_chunk_size = query_layer.shape[2]
        if seq_chunk_size <= 0:
            raise ValueError("seq_chunk_size must be positive")
        if head_chunk_size <= 0:
            raise ValueError("head_chunk_size must be positive")

        if compute_device is None:
            compute_device = torch.device(f"npu:{torch.npu.current_device()}")
        else:
            compute_device = torch.device(compute_device)
        if compute_device.type != "npu":
            raise ValueError("compute_device must be an NPU device")

        softmax_scale = query_layer.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        if seq_chunk_size == query_layer.shape[0]:
            # With one full sequence chunk, skip cache/A2A setup and use FA.
            current_stream = torch.npu.current_stream()
            self.cp_stream.wait_stream(current_stream)
            with torch.npu.stream(self.cp_stream):
                # Keep the device copies differentiable so CPU inputs receive grads.
                query_compute = query_layer.to(
                    compute_device,
                    non_blocking=query_layer.device.type == "cpu",
                ).contiguous()
                key_compute = key_layer.to(
                    compute_device,
                    non_blocking=key_layer.device.type == "cpu",
                ).contiguous()
                value_compute = value_layer.to(
                    compute_device,
                    non_blocking=value_layer.device.type == "cpu",
                ).contiguous()
                if is_forward_only:
                    with torch.no_grad():
                        output_compute = run_fa(
                            query_compute,
                            key_compute,
                            value_compute,
                            query_layer.shape[2],
                            True,
                            attention_mask,
                            softmax_scale,
                        )[0]
                else:
                    output_compute = run_fa(
                        query_compute,
                        key_compute,
                        value_compute,
                        query_layer.shape[2],
                        True,
                        attention_mask,
                        softmax_scale,
                    )[0]
                output_compute = output_compute.view(
                    query_layer.shape[0],
                    query_layer.shape[1],
                    query_layer.shape[2],
                    value_layer.shape[3],
                )

            if not is_cpu_output:
                current_stream.wait_stream(self.cp_stream)
                return output_compute

            self.final_hdc_stream.wait_stream(self.cp_stream)
            with torch.npu.stream(self.final_hdc_stream):
                output_cpu = output_compute.to("cpu", non_blocking=True)
            self.final_hdc_stream.synchronize()
            current_stream.wait_stream(self.final_hdc_stream)
            return output_cpu

        return SwapCPAttnFunc.apply(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            None,
            None,
            attn_mask_type,
            0.0,
            softmax_scale,
            False,
            cp_group,
            self.cp_stream,
            {"scatter_idx": 2, "gather_idx": 0},
            seq_chunk_size,
            head_chunk_size,
            kv_chunk_cache_slots,
            "striped",
            self.qk_prefetch_stream,
            self.v_prefetch_stream,
            self.general_offload_stream,
            self.final_hdc_stream,
            compute_device,
            is_cpu_output,
            is_forward_only,
            q_chunk_cache_slots,
            self.q_prefetch_stream,
            self.kv_prefetch_stream,
            backward_mode,
        )
