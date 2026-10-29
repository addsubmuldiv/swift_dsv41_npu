# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Debug utility helpers."""


def next_iter_when_debug_should_be_run(quantizers):
    """Return the next iteration where any quantizer needs debug checks."""
    out = None
    for quantizer in quantizers:
        next_iter = quantizer.get_next_debug_iter()
        if next_iter is not None:
            out = next_iter if out is None else min(out, next_iter)
    return out


def any_feature_enabled(quantizers):
    """Return whether any debug quantizer has an active feature."""
    return any(quantizer.any_feature_enabled() for quantizer in quantizers)
