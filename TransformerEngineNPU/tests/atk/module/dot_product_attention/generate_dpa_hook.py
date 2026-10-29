# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Hook-route ATK case generator for ``transformer_engine.pytorch.DotProductAttention``.

ITERATION STATUS: this file is the hook-route candidate, coexisting with
the default full-build generator (``generate_dot_product_attention.py``,
registry key ``generate_dot_product_attention_p0`` — curated two-tier
matrix, golden-compatible). Once this route is validated through iteration,
the full-build file gets deleted and this one becomes the only generator.
Until then, switch routes by pointing the yaml ``generate:`` field AND
the ``atk case -p`` argument at this file.

Route: the native ATK engine drives sampling; this subclass patches the
cross-input coupling via the official extension points.

Dual-format contract (slot 8 selects the mode, mirroring lnl's
normalization enum). The format value set is {sbhd, thd} — **not** bshd:
TENPU's DotProductAttention asserts qkv_format ∈ {sbhd, thd}
(``dot_product_attention.py:645``); bshd is documented in NVTE and
accepted there, but unimplemented in TENPU. That implementation gap is
documented here instead of being turned into a case that can only fail on
the device side.

  sbhd (4-D, golden-validated): q/k/v = [4-D], dy = 3-D
    slot 0  query  = [d0, d1, H, D]   (alignment: d0≥2, d1%16, D%8)
    slot 1  key    = [d0, d1, H_kv, D]
    slot 2  value  = [d0, d1, H_kv, D]
    slot 3  dy     = [d0, d1, H × D]  (3-D flattened output)
    slots 4-7      = attrs (num_heads, kv_channels, gqa, mask_type)
    slot 8         = "sbhd"

  thd (3-D, probe verified):
    slot 0  query  = [T, H, D]      (T=total tokens, NO alignment constraints)
    slot 1  key    = [T, H_kv, D]
    slot 2  value  = [T, H_kv, D]
    slot 3  dy     = [T, H, D]      (semantic output shape. TENPU returns
                                     this natively; NVTE flattens its thd
                                     output to [T, H*D] ("thd -> t(hd)"),
                                     and the adapter feeds the loss in
                                     that form and canonicalizes the
                                     returned output back to 3-D)
    slots 4-7      = same attrs
    slot 8         = "thd"

The thd shape is derived by flattening the natively sampled 4-D shape
(d0 × d1 → T), so the same pools, element caps and alignment guarantees
carry over; the alignment constraints are kept as strictly-tighter-than-
necessary bounds (thd has NO alignment constraints on NPU — probe: odd
tokens, odd heads, odd head_dim all pass). The adapter then derives
cu_seqlens from T with a fixed 3-sequence ragged split.

mask_type × format coupling (reconciled here, see _FORMAT_MASK_TYPES):
the two attrs are sampled independently by the native engine, so it can
emit combinations that are illegal on the golden side — NVTE asserts
``"padding" in attn_mask_type`` for thd
(``dot_product_attention.py:1368``), and an sbhd padding mask would
require a mask tensor this adapter never passes. The reconcile preserves
the causal bit (causal → *_causal, otherwise the plain variant).

Determinism: identical yaml + identical ``atk case -s <seed>`` (CLI default
1234) + identical ATK version ⇒ identical cases.
"""

from typing import List, Union

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig, InputCaseConfig

# Fail-closed layout check: 5 tensors (q,k,v,dy + qkv_format attr is slot 8)
# 4 attrs (num_heads, kv_channels, gqa, mask_type) + 1 format attr.
_EXPECTED_TYPES = ("tensor", "tensor", "tensor", "tensor", "attr", "attr", "attr", "attr", "attr")

_KNOWN_MASK_TYPES = ("causal", "no_mask", "padding", "padding_causal")
_KNOWN_FORMATS = ("sbhd", "thd")

# Which attn_mask_type values are legal for each qkv_format. The native
# engine samples the two attrs independently, so this table drives the
# reconcile in after_case_config (see the module docstring for the
# NVTE-side asserts behind it). Order within a tuple is irrelevant — the
# reconcile keys off the causal bit, not on membership order.
_FORMAT_MASK_TYPES = {
    "sbhd": ("causal", "no_mask"),
    "thd": ("padding_causal", "padding"),
}


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# aclnnFlashAttentionScore rejects larger head dims outright (EZ0026:
# "Parameter D dim of FlashAttentionScore ... can only be in the range of
# (0, 768]"). Applies to BOTH sbhd and thd (same kernel). The GPU side is
# unverified above 512, so oversized draws are clamped back to 512 rather
# than rejected: ATK's generator has no per-case skip, and raising here
# aborts the whole `atk case` run.
_MAX_HEAD_DIM = 768
_VERIFIED_HEAD_DIM_CAP = 512

# thd 上界一：上游 ops-transformer 的 FA TND 路径缺 16 位 stride 守卫，一旦
# **num_heads * align(head_dim, 16) > 65535** 就静默算错（fp32 只见 dq/dk/dv 错，
# fp16/bf16 连前向一起退化；sbhd 同尺寸正常）。所以本文件对 thd 把 head_dim
# 按 8 步下钳，并纳入 fail-closed 断言。
# 详见 README「已知算子限制」。
_MAX_THD_HEADS_TIMES_DIM = 65535
_HEAD_DIM_STEP = 8
_MIN_HEAD_DIM = 8
_HEAD_DIM_ALIGN = 16  # the kernel's effective head dim is aligned to 16 elements


# thd 上界二（golden 侧）：CUDA 融合 fp16 后向的 dq 一旦达到 2 GiB 就大面积没写进去
# （前向 out/dk/dv 正常、不报错），1 GiB 实测干净、中间未测 —— 故按 dq 的**字节数**
# 把 batch 维下钳到 1 GiB 以内，使该区间的用例根本不生成。
# 详见 README「已知算子限制」。
_VERIFIED_DQ_BYTES = 1 << 30  # 2**30 bytes = 1 GiB: measured clean on CUDA
_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


def _clamp_batch_for_dq_bytes(batch, seqlen, heads, head_dim, dtype, floor):
    """Halve the batch dim until dq fits the measured-clean byte budget (1 GiB)."""
    item = _DTYPE_BYTES.get(str(dtype), 2)
    clamped = max(int(floor), int(batch))
    while clamped > floor and clamped * seqlen * heads * head_dim * item > _VERIFIED_DQ_BYTES:
        clamped = max(int(floor), clamped // 2)
    return clamped


def _clamp_thd_head_dim(heads: int, head_dim: int) -> int:
    """Step head_dim down (8 at a time) until heads*align(head_dim, 16) <= 65535.

    The stride field is 16-bit and the head dim the kernel sees is the
    16-element-aligned one (raw D=120 with 512 heads = 61440, but 512*128
    overflows). head_dim is clamped rather than num_heads: the head count is
    mirrored by the num_attention_heads attr and GQA divisibility. Raising is
    reserved for "no legal head_dim exists".
    """
    clamped = head_dim
    while (
        heads * _round_up(clamped, _HEAD_DIM_ALIGN) > _MAX_THD_HEADS_TIMES_DIM
        and clamped > _MIN_HEAD_DIM
    ):
        clamped = max(_MIN_HEAD_DIM, clamped - _HEAD_DIM_STEP)
    if heads * _round_up(clamped, _HEAD_DIM_ALIGN) > _MAX_THD_HEADS_TIMES_DIM:
        raise ValueError(
            "generate_dpa_hook: thd num_heads={} cannot fit the upstream 16-bit "
            "stride limit (num_heads*align(head_dim, {}) <= {}); the sampled query "
            "shape is unrepresentable for thd".format(
                heads, _HEAD_DIM_ALIGN, _MAX_THD_HEADS_TIMES_DIM
            )
        )
    return clamped


@GENERATOR_REGISTRY.register("generate_dpa_hook")
class GenerateDPAHook(CaseGenerator):
    """Native sampling + coupling patch for the DotProductAttention matrix."""

    def __init__(self, config):
        got = tuple(i.type for i in config.inputs)
        if got != _EXPECTED_TYPES:
            raise ValueError(
                "generate_dpa_hook expects the input layout "
                "(query, key, value, dy, num_heads, kv_channels, "
                "num_gqa_groups, attn_mask_type, qkv_format) with types "
                "tensor x4 + attr x5, got {}".format(got)
            )
        super().__init__(config)
        self._q_shape = None
        self._q_dtype = None
        self._format = "sbhd"

    def after_input_config(
        self,
        index: int,
        input_case: Union[InputCaseConfig, List[InputCaseConfig]],
    ) -> Union[InputCaseConfig, List[InputCaseConfig]]:
        # Slot 8 (qkv_format) fires LAST — we need it before patching
        # shapes, so capture it here but do all shape work in
        # after_case_config (which fires after all slots).
        if index == 8:
            fmt = str(input_case.range_values)
            if fmt not in _KNOWN_FORMATS:
                raise ValueError(
                    "generate_dpa_hook: unknown qkv_format {!r} — expected "
                    "one of {} (check the yaml qkv_format attr "
                    "values)".format(fmt, list(_KNOWN_FORMATS))
                )
            self._format = fmt
        elif index == 0:
            shape = list(input_case.shape)
            # Fail-closed guard: at least 3-D with all dims >= 1.
            if len(shape) < 3:
                raise ValueError(
                    "generate_dpa_hook: degenerate query shape {} — DPA "
                    "requires 3-D (thd) or 4-D (sbhd). Keep the boundary "
                    "track off: extra_numbers: 0 and has_empty/has_infnan/"
                    "has_scalar: false on the tensor inputs".format(input_case.shape)
                )
            if any(d == 0 for d in shape):
                raise ValueError(
                    "generate_dpa_hook: empty query shape {} — the empty "
                    "track is not usable for DPA in this harness: thd needs "
                    ">= 3 tokens (the adapter derives 3 ragged cu_seqlens "
                    "sequences), ATK zeroes a RANDOM dim (H=0/D=0 break "
                    "module init; only sbhd batch/seq=0 was NPU-probed), "
                    "and the GPU side is unverified. Keep has_empty: "
                    "false".format(input_case.shape)
                )
            # sbhd-only alignment reconcile (thd has NO constraints).
            # Applied in-place before anything mirrors off this shape.
            if len(shape) == 4:
                shape[0] = max(shape[0], 2)  # batch >= 2
                shape[1] = _round_up(shape[1], 16)  # seq % 16
                shape[3] = _round_up(shape[3], 8)  # head_dim % 8
                # The head_dim cap binds BOTH formats (the same kernel
                # runs thd), and slot 0 is the 4-D sample either way.
                # Draws past the NPU hard limit are clamped to the
                # both-backends-verified cap (512) instead of raising —
                # ATK's generator has no per-case skip, so a raise kills
                # the entire run. Values in (512, 768] pass through:
                # NPU-legal, GPU side unverified by design.
                if shape[3] > _MAX_HEAD_DIM:
                    shape[3] = _VERIFIED_HEAD_DIM_CAP
                input_case.shape = shape
            self._q_shape = shape
            self._q_dtype = input_case.dtype
        return input_case

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        fmt = self._format
        shape = self._q_shape
        mask_type = str(case_config.inputs[7].range_values)
        if mask_type not in _KNOWN_MASK_TYPES:
            raise ValueError(
                "generate_dpa_hook: unknown attn_mask_type {!r} — expected one of {}".format(
                    mask_type, list(_KNOWN_MASK_TYPES)
                )
            )

        # mask × format reconcile. The engine sampled the two attrs
        # independently, so an illegal pair is the common case, not an
        # exception: map it onto the legal set for this format, keeping
        # the causal bit (causal → the *_causal variant, else the plain
        # one). Both pairs are legal on both backends afterwards.
        allowed_masks = _FORMAT_MASK_TYPES[fmt]
        if mask_type not in allowed_masks:
            wants_causal = "causal" in mask_type
            mask_type = next(
                (m for m in allowed_masks if ("causal" in m) == wants_causal),
                allowed_masks[0],
            )
            case_config.inputs[7].range_values = mask_type

        # Format conversion: the native engine samples 4-D (yaml
        # dim_numbers=[4]); thd needs 3-D [T, H, D]. Flatten
        # d0 × d1 → T (total tokens). The alignment constraints applied
        # in slot 0 carry over as conservative bounds on thd shapes
        # (probe: thd has no NPU constraints, so this is strictly tighter
        # than necessary).
        if fmt == "thd":
            if len(shape) == 4:
                d0, d1, H, D = shape
                # dq bytes: keep the case inside the CUDA-verified zone. The
                # floor keeps T = d0*d1 >= 3 so the thd contract still holds.
                d0 = _clamp_batch_for_dq_bytes(
                    d0, d1, H, D, self._q_dtype, floor=max(1, -(-3 // d1))
                )
                shape = [d0 * d1, H, D]
                self._q_shape = shape
                case_config.inputs[0].shape = shape
            T, H, D = shape[0], shape[1], shape[2]
            # cannot be judged above the 16-bit stride limit -> clamp head_dim
            # (keep num_heads; ATK has no per-case skip, so no raise)
            if H * _round_up(D, _HEAD_DIM_ALIGN) > _MAX_THD_HEADS_TIMES_DIM:
                D = _clamp_thd_head_dim(H, D)
                shape = [T, H, D]
                self._q_shape = shape
                case_config.inputs[0].shape = shape
        else:
            d0, d1, H, D = shape[0], shape[1], shape[2], shape[3]
            # same dq-byte budget; floor 2 = the sbhd batch>=2 constraint
            d0 = _clamp_batch_for_dq_bytes(d0, d1, H, D, self._q_dtype, floor=2)
            shape = [d0, d1, H, D]
            case_config.inputs[0].shape = shape
            self._q_shape = shape

        # GQA reconcile: gqa must be 0 (MHA) or a divisor of H.
        raw_gqa = case_config.inputs[6].range_values
        if isinstance(raw_gqa, (list, tuple)):
            gqa = 0
        else:
            gqa = int(raw_gqa)
        if gqa < 0:
            gqa = 0
        elif gqa > 0 and H % gqa != 0:
            # O(√H) pair-factor search.
            best = 0
            d = 1
            while d * d <= H:
                if H % d == 0:
                    for cand in (d, H // d):
                        if best == 0 or abs(cand - gqa) < abs(best - gqa):
                            best = cand
                d += 1
            gqa = best
        case_config.inputs[6].range_values = gqa
        kv_heads = gqa if gqa > 0 else H

        # Patch shapes per format.
        if fmt == "thd":
            kv_shape = [T, kv_heads, D]
            dy_shape = [T, H, D]  # thd output is 3-D, same as input
        else:
            kv_shape = [d0, d1, kv_heads, D]
            dy_shape = [d0, d1, H * D]  # sbhd output is 3-D flattened

        for slot in (1, 2):
            case_config.inputs[slot].shape = kv_shape
            case_config.inputs[slot].dtype = self._q_dtype
        case_config.inputs[3].shape = dy_shape
        case_config.inputs[3].dtype = self._q_dtype

        # Sync attrs.
        case_config.inputs[4].range_values = H  # num_heads
        case_config.inputs[5].range_values = D  # kv_channels

        # Fail-closed consistency check.
        if not (
            case_config.inputs[1].shape == kv_shape
            and case_config.inputs[2].shape == kv_shape
            and case_config.inputs[3].shape == dy_shape
            and len({c.dtype for c in case_config.inputs[:4]}) == 1
            and (gqa == 0 or H % gqa == 0)
            and mask_type in allowed_masks
            and (fmt == "thd" or (d1 % 16 == 0 and D % 8 == 0 and d0 >= 2))
            and (fmt != "thd" or T >= 3)
            and (fmt != "thd" or H * _round_up(D, _HEAD_DIM_ALIGN) <= _MAX_THD_HEADS_TIMES_DIM)
        ):
            raise ValueError(
                "generate_dpa_hook: assembled case violates the DPA coupling "
                "contract (q={}, k={}, dy={}, gqa={}, mask={}, fmt={})".format(
                    shape,
                    case_config.inputs[1].shape,
                    case_config.inputs[3].shape,
                    gqa,
                    mask_type,
                    fmt,
                )
            )
        return case_config
