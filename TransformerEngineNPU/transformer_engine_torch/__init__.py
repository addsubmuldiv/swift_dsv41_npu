# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Pure-torch stand-in for the ``transformer_engine_torch`` CUDA extension.

On GPU, ``transformer_engine_torch`` is the pybind module built from
``transformer_engine/pytorch/csrc`` and imported by downstream projects as
``tex``. The NPU stack ships no C++ extension, so this top-level package
provides the same import path with pure-torch implementations:

* :func:`thd_get_partitioned_indices` -- load-balanced THD context-parallel
  token partitioning, mirroring the CUDA kernel in
  ``transformer_engine/common/fused_attn/context_parallel.cu``.
* :class:`DType` -- TE datatype tags, mirroring the pybind enum value-for-value
  (with a ``torch_dtype`` property mapping to TENPU's torch dtypes).
"""

from .tex_dtype import DType
from .thd_get_partitioned_indices import thd_get_partitioned_indices

__all__ = ["DType", "thd_get_partitioned_indices"]
