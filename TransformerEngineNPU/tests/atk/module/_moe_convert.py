# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional

import torch


def _convert_row_id_map_to_npu(
    row_id_map: torch.Tensor,
    num_tokens: Optional[int] = None,
    num_experts: Optional[int] = None,
    num_out_tokens: Optional[int] = None,
    drop_and_pad: bool = True,
) -> torch.Tensor:
    """Convert a TE dense ``row_id_map`` to the NPU ``row_id_map`` layout.

    For ``drop_and_pad=False`` the result is the 1-D ``sorted_indices_out``:
    ``result[flat_input_index] = dst``, where the flat input index is
    token-major (``token * top_k + occurrence``).

    For ``drop_and_pad=True`` the result is the direct ``dst -> src`` mapping
    used by the NPU unpermute operator.

    For ``drop_and_pad=False`` the returned token-major map is additionally
    rearranged to match the NPU layout: each token's ``top_k`` destinations
    (a contiguous group) are stored in reversed order, which is independent of
    per-expert loads.
    """
    device = row_id_map.device

    if row_id_map.numel() == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)
    if row_id_map.dim() != 2:
        raise ValueError("TE row_id_map must be a 2-D tensor.")

    width = row_id_map.shape[1]
    if width < 3 or (width - 1) % 2 != 0:
        raise ValueError("TE row_id_map width must be 2 * num_experts + 1.")

    num_experts = (width - 1) // 2 if num_experts is None else num_experts
    if num_experts * 2 + 1 != width:
        raise ValueError("num_experts does not match the TE row_id_map width.")

    num_tokens = row_id_map.shape[0] if num_tokens is None else num_tokens
    if num_tokens != row_id_map.shape[0]:
        raise ValueError("num_tokens does not match the TE row_id_map height.")
    if num_tokens == 0 or num_experts == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)

    token_counts = row_id_map[:, -1].to(torch.int64)
    if (token_counts < 0).any() or (token_counts > num_experts).any():
        raise ValueError("TE row_id_map contains an invalid routed token_count.")

    if drop_and_pad:
        if num_out_tokens is None:
            raise ValueError("num_out_tokens is required to reconstruct the NPU padded row_id_map.")

        capacity = num_out_tokens // num_experts
        if num_out_tokens % num_experts != 0:
            raise ValueError("num_out_tokens must be divisible by num_experts.")

        dest = row_id_map[:, :num_experts].flip(1).to(torch.int64)
        dest_experts = row_id_map[:, num_experts:-1].flip(1).to(torch.int64)
        valid_slot_fwd = torch.arange(num_experts, dtype=torch.int64, device=device).unsqueeze(
            0
        ) < token_counts.unsqueeze(1)
        valid_slot = valid_slot_fwd.flip(1)
        valid_dest = dest[valid_slot]
        valid_dest_experts = dest_experts[valid_slot]
        counts = torch.bincount(valid_dest_experts, minlength=num_experts)

        out = torch.full((num_out_tokens,), -1, dtype=torch.int32, device=device)
        if valid_dest.numel() > 0:
            token_ids = (
                torch.arange(num_tokens, dtype=torch.int64, device=device)
                .unsqueeze(1)
                .expand(num_tokens, num_experts)[valid_slot]
            )
            if (valid_dest < 0).any() or (valid_dest >= num_out_tokens).any():
                raise ValueError("TE row_id_map contains a destination outside the padded output.")

            # 先生成原始映射（按 token 顺序，每个 token 映射到对应的 dest）
            # 按 valid_dest_experts 分组，但暂时不插入 -1
            temp_out = []
            for e in range(num_experts):
                expert_mask = valid_dest_experts == e
                if expert_mask.any():
                    expert_token_ids = token_ids[expert_mask]
                    expert_dest = valid_dest[expert_mask]
                    # 按 dest 从小到大排序，确保映射顺序正确
                    sorted_indices = torch.argsort(expert_dest)
                    expert_token_ids_sorted = expert_token_ids[sorted_indices]
                    # 添加到临时列表
                    temp_out.extend(expert_token_ids_sorted.tolist())

            # 现在根据 counts 在每组间填充未使用的 token ID
            result = []
            start_idx = 0
            for e in range(num_experts):
                num_tokens_for_expert = counts[e].item()
                # 取出该专家的 token IDs
                expert_tokens_all = temp_out[start_idx : start_idx + num_tokens_for_expert]
                expert_tokens = expert_tokens_all[:capacity]
                result.extend(expert_tokens)

                # 填充缺失的位置 - 使用该组未使用的 token ID
                missing = capacity - num_tokens_for_expert
                if missing > 0:
                    # 获取该组已使用的 token IDs
                    used_tokens = set(expert_tokens)
                    # 找出所有未使用的 token IDs
                    unused_tokens = [t for t in range(num_tokens) if t not in used_tokens]
                    # 取前 missing 个
                    fill_tokens = unused_tokens[:missing]
                    result.extend(fill_tokens)

                start_idx += num_tokens_for_expert

            out = torch.tensor(result, dtype=torch.int32, device=device)

        return out

    if num_tokens > 0:
        top_k = int(token_counts[0].item())
        if (token_counts != top_k).any():
            raise ValueError("drop_and_pad=False requires the same top_k for every token.")
        if top_k <= 0:
            return torch.empty((0,), dtype=torch.int32, device=device)
        if top_k > num_experts:
            raise ValueError("drop_and_pad=False top_k cannot exceed num_experts.")
    else:
        top_k = 0

    num_out_tokens = num_tokens * top_k
    dest = row_id_map[:, :top_k].flip(1).reshape(-1).to(torch.int32)
    if dest.numel() > 0:
        min_dest = torch.min(dest)
        max_dest = torch.max(dest)
        if min_dest < 0 or max_dest >= num_out_tokens:
            raise ValueError("drop_and_pad=False TE row_id_map contains an invalid destination.")
    return dest


def _mask_unused_row_id_map(row_id_map: torch.Tensor) -> torch.Tensor:
    """Zero the unused destination/expert slots of a TE mask-map row_id_map.

    ``row_id_map`` has shape ``[num_tokens, 2 * num_experts + 1]``.  The TE
    Triton pass-3 kernel only writes the first ``n_routed`` destination and
    expert slots, leaving the remaining slots uninitialized.  This helper
    keeps those valid slots unchanged and forces all unused slots to -1 so
    that NPU and NVTE golden comparisons are deterministic.
    """
    if row_id_map.dim() != 2:
        return row_id_map

    width = row_id_map.shape[1]
    if width < 3 or (width - 1) % 2 != 0:
        return row_id_map

    num_experts = (width - 1) // 2
    n_routed = row_id_map[:, -1].to(torch.int64)
    slot_ids = torch.arange(num_experts, device=row_id_map.device).unsqueeze(0)
    valid_slots = slot_ids < n_routed.unsqueeze(1)

    out = row_id_map.clone()
    out[:, :num_experts] = torch.where(
        valid_slots,
        out[:, :num_experts],
        torch.full_like(out[:, :num_experts], -1),
    )
    out[:, num_experts : num_experts * 2] = torch.where(
        valid_slots,
        out[:, num_experts : num_experts * 2],
        torch.full_like(out[:, num_experts : num_experts * 2], -1),
    )
    return out


def _suppress_padded_permuted_to_npu(
    padded: torch.Tensor,
    capacity: int,
    align_size: int,
    tokens_per_expert: torch.Tensor,
    *,
    nvte_layout: Optional[bool] = None,
) -> torch.Tensor:
    """Rebuild a backend-specific padded ``permuted`` tensor in the NPU layout.

    The canonical layout gives every expert a contiguous block of exactly
    ``capacity`` rows: routed tokens first (up to ``capacity``) and zeros in
    the remaining slots. This matches the per-expert layout of the padded
    ``row_id_map``, so CPU, TENPU, and NVTE outputs become directly
    comparable. The conversion is out-of-place and differentiable: cropped
    tokens and padding receive zero gradient, and retained rows receive the
    gradient of their canonical destination. Apply it BEFORE forming a loss.

    CPU and TENPU outputs already contain one ``capacity``-row block per
    expert, so only slots past the routed count are zeroed. NVTE outputs are
    grouped by the per-expert aligned sizes instead; each group is cropped or
    zero-padded to ``capacity``, applying the same drop-and-pad semantics
    as the CPU and TENPU baselines. ``nvte_layout`` can explicitly select the
    source layout; by default it is inferred from the tensor device.
    """
    num_experts = tokens_per_expert.size(0)
    if padded.size(0) != capacity * num_experts:
        raise ValueError(
            f"num_out_tokens [{padded.size(0)}] must be the same as required [{capacity * num_experts}]"
        )

    capacity = int(capacity)
    counts = tokens_per_expert.tolist()
    if nvte_layout is None:
        nvte_layout = padded.device.type not in ("npu", "cpu")
    if nvte_layout:
        source_sizes = [(count + align_size - 1) // align_size * align_size for count in counts]
    else:
        source_sizes = [capacity] * num_experts

    suppressed = []
    for group, count in zip(padded.split(source_sizes, dim=0), counts):
        kept = min(count, capacity)
        # Ignore the backend's pad contents, which can contain duplicated real
        # tokens on NPU/CPU. Never write through a view of a custom-op output:
        # autograd may need the original output for its backward implementation.
        suppressed.append(
            torch.cat([group[:kept], padded.new_zeros((capacity - kept, padded.size(1)))], dim=0)
        )
    return torch.cat(suppressed, dim=0)


def _suppress_padded_probs_to_npu(
    prob_pad: torch.Tensor,
    capacity: int,
    align_size: int,
    tokens_per_expert: torch.Tensor,
    *,
    nvte_layout: Optional[bool] = None,
) -> torch.Tensor:
    """Rebuild a backend-specific padded probabilities tensor in the NPU layout.

    One-dimensional counterpart of ``_suppress_padded_permuted_to_npu`` for
    the padded per-token probabilities returned by the API: every expert keeps
    exactly ``capacity`` contiguous entries, with routed probabilities first
    and zeros in the remaining slots. This matches the per-expert layout of
    the padded ``row_id_map``, so CPU, TENPU, and NVTE outputs become directly
    comparable.

    CPU and TENPU outputs already contain one ``capacity``-entry block per
    expert, so only slots past the routed count are zeroed. NVTE outputs are
    grouped by the per-expert aligned sizes instead; each group is cropped or
    zero-padded to ``capacity``, applying the same drop-and-pad semantics
    as the CPU and TENPU baselines.
    """
    return _suppress_padded_permuted_to_npu(
        prob_pad.unsqueeze(1),
        capacity,
        align_size,
        tokens_per_expert,
        nvte_layout=nvte_layout,
    ).squeeze(1)
