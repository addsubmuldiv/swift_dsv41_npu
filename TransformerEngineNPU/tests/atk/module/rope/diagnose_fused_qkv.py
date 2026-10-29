# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Channel-level discriminator for the fused-QKV RoPE forward divergence.

The 2026-08-29 golden run (post 3#19 layout fix) shows the fused-QKV forward
q/k slots missing the accuracy gate with mean_rel_err 0.0018..0.019 and
max_rel_err 0.25..5.07 - small but systematic, and amplified by GQA (case 9:
0.019 vs MHA ~0.002). Shape/contract slots are all green, so this is a
numerical channel-mapping difference, prime suspect being the freqs (angles)
interpretation: whether the kernel builds rotary pairs from the [s,1,1,hn]
angle tensor the same way TENPU's _apply_fused_rope_4d does.

This script runs on the intranet NPU host with the real ATK input.bin and the
NVTE golden q output. It recomputes the TENPU forward and reports, per rotary
PAIR index, the abs error between the two sides - a half-shift or pair-swap
in the freqs mapping shows up as error concentrated on every other pair or on
the first/second half of the head.

Usage:
  python diagnose_fused_qkv.py \
    --input-bin ./rope_inputs/6/input.bin \
    --golden-q ./rope_goldens/cpu_nvte/all_rope/6/output_0.pt \
    [--variant-args 4 5 6 7]   # positions of variant/fmt/interleaved/split
"""

import argparse
from pathlib import Path

import torch

RTOL = 0.0156
ATOL = 0.0156


def _load(input_bin: str, golden_q: str):
    saved = torch.load(input_bin, map_location="cpu", weights_only=True)
    # fused-qkv layout: qkv, q_freqs, k_freqs, dy, variant, fmt, interleaved, split
    qkv, q_freqs, k_freqs, dy, variant, fmt, interleaved, split_str = saved
    golden = torch.load(golden_q, map_location="cpu", weights_only=True)
    return (
        qkv,
        q_freqs,
        k_freqs,
        str(variant),
        str(fmt),
        bool(interleaved),
        [int(x) for x in str(split_str).split(",")],
        golden,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--input-bin", required=True)
    parser.add_argument("--golden-q", required=True, help="NVTE golden output_0.pt (q)")
    parser.add_argument("--device", default="npu")
    cli = parser.parse_args()

    for label, path in (("input", cli.input_bin), ("golden", cli.golden_q)):
        if not Path(path).exists():
            raise FileNotFoundError("{} not found: {}".format(label, path))

    qkv, q_freqs, k_freqs, variant, fmt, interleaved, split, golden = _load(
        cli.input_bin, cli.golden_q
    )
    if variant != "rope_fused_qkv":
        raise ValueError("not a fused-QKV case (variant={})".format(variant))

    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    q, _, _ = apply_fused_qkv_rotary_pos_emb(
        qkv.to(cli.device),
        q_freqs.to(cli.device),
        k_freqs.to(cli.device),
        split,
        tensor_format=fmt,
        interleaved=interleaved,
    )
    got = q.cpu().float()
    exp = golden.float()
    print(
        "q shape: {} golden: {} dtype {}".format(tuple(got.shape), tuple(exp.shape), golden.dtype)
    )
    if got.shape != exp.shape:
        print("SHAPE MISMATCH — layout divergence (REPORT atk 3#19 territory)")
        return

    err = (got - exp).abs()
    ok = err <= ATOL + RTOL * exp.abs()
    print(
        "overall: mean_abs={:.6f} max_abs={:.6f} failing={}/{}".format(
            err.mean().item(), err.max().item(), int((~ok).sum().item()), ok.numel()
        )
    )

    # per rotary-pair analysis: pair j covers channels (2j, 2j+1) under
    # interleaved and (j, j + hn/2) under half
    hn = split[1]
    print()
    print("per-pair mean abs error (first 12 pairs of head 0, group 0):")
    if interleaved:
        # pairs are adjacent channels inside each head
        heads = err.view(err.shape[0], -1, hn)  # [s, heads, hn]
        h0 = heads[0, 0]
        for j in range(min(12, hn // 2)):
            e = h0[2 * j : 2 * j + 2].mean().item()
            print("  pair {:>2} (ch {:>2},{:>3}): mean_abs={:.6f}".format(j, 2 * j, 2 * j + 1, e))
    else:
        # half mode: pair j = (j, j + hn/2) inside each head
        heads = err.view(err.shape[0], -1, hn)
        h0 = heads[0, 0]
        for j in range(min(12, hn // 2)):
            e = (h0[j] + h0[j + hn // 2]).item() / 2
            print(
                "  pair {j:>2} (ch {j:>2},{k:>3}): mean_abs={e:.6f}".format(j=j, k=j + hn // 2, e=e)
            )

    # half-vs-half summary: is the error concentrated in the first or second
    # half of each head (classic half-shift signature)?
    heads = err.view(err.shape[0], -1, hn)
    first_half = heads[..., : hn // 2].mean().item()
    second_half = heads[..., hn // 2 :].mean().item()
    print()
    print(
        "half summary: first-half mean_abs={:.6f}  second-half mean_abs={:.6f}".format(
            first_half, second_half
        )
    )
    even_ch = heads[..., 0::2].mean().item()
    odd_ch = heads[..., 1::2].mean().item()
    print("parity summary: even-ch mean_abs={:.6f}  odd-ch mean_abs={:.6f}".format(even_ch, odd_ch))

    # diagnostic verdicts
    print()
    if err.max().item() < 1e-7:
        print("verdict: bitwise identical to golden — no divergence at all")
        return
    if first_half < 1e-6 and second_half > 1e-4:
        print("verdict: error concentrated in SECOND half of each head ->")
        print("  freqs half-width mismatch (TENPU rotates [0:hn/2], NVTE rotates")
        print("  [hn/2:hn] or vice versa). Check rot_dim/rotary-width handling.")
        return
    ratio = second_half / max(first_half, 1e-12)
    if ratio > 5 or ratio < 0.2:
        print(
            "verdict: asymmetric halves ({:.1f}x) -> partial-rotary boundary difference".format(
                ratio
            )
        )
        return
    if abs(even_ch - odd_ch) / max(even_ch, odd_ch, 1e-12) > 0.5:
        print("verdict: parity asymmetry -> interleaved/half pairing mismatch")
        return
    if err.max().item() < 0.05:
        print("verdict: uniform small error (~bf16 noise + few near-zero elements)")
        print("  -> likely tolerance-domain; mixed_tolerance should absorb it")
        return
    print("verdict: mixed signature — inspect the per-pair table above")


if __name__ == "__main__":
    main()
