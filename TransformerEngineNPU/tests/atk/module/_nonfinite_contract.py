# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.
"""Exact special-value position signatures for nonfinite-input boundary cases.

ATK mixed_tolerance_bm accepts an invalid GPU reference without checking NPU
values. Add signatures to the existing exact-compared API contract so such a
case cannot silently pass with a different NaN/+Inf/-Inf pattern. Numeric
outputs remain untouched and keep their original comparison standard.
"""

import hashlib
import torch


def special_value_signatures(outputs):
    signatures = []
    for output in outputs:
        tags = torch.isnan(output).to(torch.uint8)
        tags += torch.isposinf(output).to(torch.uint8) * 2
        tags += torch.isneginf(output).to(torch.uint8) * 3
        signatures.append(hashlib.sha256(tags.contiguous().cpu().numpy().tobytes()).hexdigest())
    return signatures
