# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Case / shape grids for Parallel CE mass validation (>=1500)."""

import itertools
from typing import Tuple

Shape = Tuple[int, int, int]  # B, S, V
# B,S,V,dtype,reduce_loss,label_smoothing,use_ignore,swap_dim,noncontiguous
AccCase = Tuple[int, int, int, str, bool, float, bool, bool, bool]


def _unique_shapes(items):
    # type: (List[Shape]) -> List[Shape]
    seen = set()
    out = []
    for b, s, v in items:
        key = (int(b), int(s), int(v))
        if key in seen:
            continue
        if key[0] < 1 or key[1] < 1 or key[2] < 2:
            continue
        seen.add(key)
        out.append(key)
    return out


def generate_perf_shapes(min_count=1500, max_elements=64 * 1024 * 1024):
    # type: (int, int) -> List[Shape]
    batches = [1, 2, 3, 4, 6, 8]
    seqs = list(range(1, 33)) + [
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        320,
        384,
        448,
        512,
        768,
        1024,
    ]
    vocabs = [
        32,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        20480,
        24576,
        32000,
        40960,
        48000,
        65536,
        81920,
        100000,
        128000,
    ]
    raw = []
    for b in batches:
        for s in seqs:
            for v in vocabs:
                if b * s * v <= max_elements:
                    raw.append((b, s, v))
    extras = [
        (1, 1, 128000),
        (1, 2048, 4096),
        (2, 2048, 8192),
        (4, 1024, 32000),
        (8, 512, 65536),
        (1, 4096, 2048),
        (2, 1, 128000),
        (8, 1, 128000),
        (1, 8, 100000),
        (3, 127, 32000),
        (5, 63, 65536),
        (7, 17, 16384),
    ]
    for b, s, v in extras:
        if b * s * v <= max_elements:
            raw.append((b, s, v))
    shapes = _unique_shapes(raw)
    v = 64
    while len(shapes) < min_count and v <= 200000:
        for b, s in [(1, 16), (2, 32), (4, 64), (2, 128), (1, 256)]:
            if b * s * v <= max_elements:
                shapes = _unique_shapes(shapes + [(b, s, v)])
            if len(shapes) >= min_count:
                break
        v += 64
    # Order: small V, mid V, large V interleaved for coverage in first N
    small = [x for x in shapes if x[2] < 1000]
    mid = [x for x in shapes if 1000 <= x[2] < 16384]
    large = [x for x in shapes if x[2] >= 16384]
    ordered = []
    for i in range(max(len(small), len(mid), len(large))):
        if i < len(small):
            ordered.append(small[i])
        if i < len(mid):
            ordered.append(mid[i])
        if i < len(large):
            ordered.append(large[i])
    ordered = _unique_shapes(ordered)
    if len(ordered) < min_count:
        raise RuntimeError("perf grid too small: %d < %d" % (len(ordered), min_count))
    # Return at least min_count unique shapes (keep extras only up to 2x for coverage).
    n = max(min_count, min(len(ordered), max(min_count, 1500)))
    if len(ordered) > n * 2:
        return ordered[:n]
    return ordered[: max(min_count, min(len(ordered), n))]


def generate_accuracy_cases(min_count=1500, max_elements=8 * 1024 * 1024):
    # type: (int, int) -> List[AccCase]
    batches = [1, 2, 4]
    seqs = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 127, 128, 256]
    vocabs = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000]
    dtypes = ["float32", "bfloat16"]
    reduces = [True, False]
    smoothings = [0.0, 0.1]
    ignores = [False, True]
    swaps = [False, True]
    nonconts = [False, True]
    cases = []
    for b, s, v, dtype, reduce_loss, ls, use_ignore, swap, noncontig in itertools.product(
        batches,
        seqs,
        vocabs,
        dtypes,
        reduces,
        smoothings,
        ignores,
        swaps,
        nonconts,
    ):
        if b * s * v > max_elements:
            continue
        if noncontig and not swap:
            continue
        cases.append(
            (
                b,
                s,
                v,
                dtype,
                reduce_loss,
                ls,
                use_ignore,
                swap,
                noncontig,
            )
        )
    # unique
    seen = set()
    out = []
    for c in cases:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    if len(out) < min_count:
        raise RuntimeError("accuracy grid too small: %d < %d" % (len(out), min_count))
    # At least min_count; keep full grid when moderate, else cap for runtime.
    if len(out) <= max(min_count, 3000):
        return out
    # Stratified sample to >= min_count while covering ignore/swap/dtype/V bands
    import random

    rng = random.Random(0)
    buckets = {"small": [], "mid": [], "large": []}
    for c in out:
        v = c[2]
        if v < 1000:
            buckets["small"].append(c)
        elif v < 16384:
            buckets["mid"].append(c)
        else:
            buckets["large"].append(c)
    picked = []
    target_n = max(min_count, 1500)
    per = target_n // 3 + 1
    for key in ("small", "mid", "large"):
        bucket = buckets[key]
        rng.shuffle(bucket)
        picked.extend(bucket[:per])
    rng.shuffle(picked)
    seen = set()
    final = []
    for c in picked + out:
        if c in seen:
            continue
        seen.add(c)
        final.append(c)
        if len(final) >= target_n:
            break
    return final


def summarize_shape_coverage(shapes):
    # type: (Sequence[Shape]) -> dict
    vs = [v for _, _, v in shapes]
    ss = [s for _, s, _ in shapes]
    bs = [b for b, _, _ in shapes]
    return {
        "count": len(shapes),
        "B_range": [min(bs), max(bs)],
        "S_range": [min(ss), max(ss)],
        "V_range": [min(vs), max(vs)],
        "small_V_lt_1k": sum(1 for v in vs if v < 1000),
        "mid_V_1k_16k": sum(1 for v in vs if 1000 <= v < 16384),
        "large_V_ge_16k": sum(1 for v in vs if v >= 16384),
        "max_elements": max(b * s * v for b, s, v in shapes),
    }
