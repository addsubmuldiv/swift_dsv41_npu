# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""TE dtype enum for the ``transformer_engine_torch`` compatibility surface.

On GPU, ``tex.DType`` is the pybind11 binding of the C++ ``DType`` enum in
``transformer_engine/common/include/transformer_engine/transformer_engine.h``.
This module mirrors it value-for-value as a pure-Python ``IntEnum``, so that
``import transformer_engine_torch as tex; tex.DType.kFloat8E4M3`` works on NPU
and downstream code (e.g. Megatron-LM) can pass it where the CUDA build expects
the pybind enum. The ``kNumTypes`` sentinel is intentionally omitted, matching
the canonical Python enum in ``transformer_engine/pytorch/constants.py``.
"""

from __future__ import annotations

import enum

import torch

__all__ = ["DType"]

_TORCH_DTYPES: dict[int, torch.dtype] = {
    0: torch.uint8,
    1: torch.int16,
    2: torch.int32,
    3: torch.int64,
    4: torch.float32,
    5: torch.float16,
    6: torch.bfloat16,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}


class DType(enum.IntEnum):
    """TE datatype tags, mirroring the C++ ``transformer_engine::DType`` enum.

    Like the pybind enum on GPU, members compare equal to their integer values
    (``DType.kFloat8E4M3 == 7``). The :attr:`torch_dtype` property is an NPU
    addition that maps each member to the ``torch.dtype`` used by TENPU, which
    has no separate dtype enum of its own.
    """

    kByte = 0
    kInt16 = 1
    kInt32 = 2
    kInt64 = 3
    kFloat32 = 4
    kFloat16 = 5
    kBFloat16 = 6
    kFloat8E4M3 = 7
    kFloat8E5M2 = 8
    kFloat8E8M0 = 9
    kFloat4E2M1 = 10

    @property
    def torch_dtype(self) -> torch.dtype:
        """The ``torch.dtype`` TENPU uses for this datatype tag."""
        try:
            return _TORCH_DTYPES[self.value]
        except KeyError:
            raise NotImplementedError(
                f"{self.name} has no plain torch.dtype equivalent on this backend."
            ) from None
