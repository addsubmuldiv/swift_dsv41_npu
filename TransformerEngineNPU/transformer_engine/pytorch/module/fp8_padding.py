# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.


class Fp8Padding:
    def __init__(self, num_local_experts, align_size=None):
        pass

    def __call__(self, inp, m_split):
        return inp, m_split


__all__ = ["Fp8Padding"]
