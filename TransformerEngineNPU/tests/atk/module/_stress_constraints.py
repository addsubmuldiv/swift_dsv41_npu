# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.
"""Deterministic case constraints; YAML remains the sampling specification.

Two GiB is a per-input cap, NOT a peak-memory guarantee. Both x and dy,
clones, CPU FP64 statistics, results and device workspaces need extra memory.
The FP32 normalization row cap bounds the test reduction size; it does not
guarantee that every generated case meets the numerical tolerance.
"""

import math

MAX_INPUT_BYTES = 2 * 1024**3
MAX_FP32_NORM_ROWS = 2**21
MAX_NONFINITE_ELEMENTS = 2**20
NARROW_NORM_FEATURES = 128
MAX_NARROW_NORM_ROWS = 8192
_ITEMSIZE = {"bf16": 2, "fp16": 2, "fp32": 4}


def _contains_nonfinite_range(value):
    if isinstance(value, str):
        return value.lower() in ("nan", "inf", "-inf")
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite_range(x) for x in value)
    return False


def constrain_input_shape(spec, *, normalization=False):
    """Shrink leading dimensions, preserving rank, hidden size and input values.

    Called after native sampling, before paired inputs/bundles are finalized.
    No tensors are materialized and no numerical pass/fail results are used.
    """
    original = list(spec.shape)
    shape = list(original)
    if any(dim <= 0 for dim in original):
        return original, shape  # Empty-input semantics belong to the execution API.
    # Narrow parameter gradients have few elements for the matched-ratio
    # budget. Limit their reduction length for all supported input dtypes.
    # Preserve the normalized dimension and use the LayerNormLinear policy.
    narrow_limit = math.inf
    if normalization and shape[-1] < NARROW_NORM_FEATURES:
        narrow_limit = MAX_NARROW_NORM_ROWS
        for axis in range(len(shape) - 1):
            while shape[axis] > 1 and math.prod(shape[:-1]) > narrow_limit:
                shape[axis] = max(1, shape[axis] // 2)
    itemsize = _ITEMSIZE[spec.dtype]
    max_elements = MAX_INPUT_BYTES // itemsize
    if _contains_nonfinite_range(spec.range_values):
        max_elements = min(max_elements, MAX_NONFINITE_ELEMENTS)
    row_limit = MAX_FP32_NORM_ROWS if normalization and spec.dtype == "fp32" else math.inf
    axes = range(len(shape) - 1) if normalization else range(len(shape))
    while math.prod(shape) > max_elements or (normalization and math.prod(shape[:-1]) > row_limit):
        candidates = [i for i in axes if shape[i] > 1]
        if not candidates:
            raise ValueError(f"Cannot constrain shape {original} while preserving hidden size")
        axis = max(candidates, key=lambda i: (shape[i], -i))
        shape[axis] = max(1, shape[axis] // 2)
    if shape != original:
        spec.shape = shape
    return original, shape
