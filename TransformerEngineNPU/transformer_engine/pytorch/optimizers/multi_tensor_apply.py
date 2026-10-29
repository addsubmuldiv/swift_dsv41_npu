# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Multi-tensor apply entry."""

try:
    from torch.distributed._tensor import DTensor
except ImportError:
    DTensor = None


class MultiTensorApply:
    """Multi-tensor apply entry.

    Applies an operation to a set of tensor lists, handling
    DTensor unwrapping and automatic chunk_size injection.
    """

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        if DTensor is not None:
            for i, ts in enumerate(tensor_lists):
                for j, t in enumerate(ts):
                    if isinstance(t, DTensor):
                        tensor_lists[i][j] = t._local_tensor

        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)