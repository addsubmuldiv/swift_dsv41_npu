# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps
from logging import getLogger

import torch_npu

LOG = getLogger(__name__)


def get_nccl_options_add_group_info_wrapper(get_nccl_options):
    """Add parallel group name to hccl_config for profiler group identification.

    Megatron-LM builds pg_options for every parallel group via
    ``megatron.core.parallel_state.get_nccl_options``. On NPU the HCCL
    communicator is created from ``options.hccl_config``; injecting
    ``group_name`` lets msprof traces show which parallel domain
    (dp/tp/pp/ep/...) each communication group belongs to.
    """

    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        options = get_nccl_options(pg_name, nccl_comm_cfgs)
        if hasattr(torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options, 'hccl_config'):
            options = options if options is not None else torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            try:
                # torch_npu not support inplace update
                hccl_config = options.hccl_config
                hccl_config.update({'group_name': pg_name})
                options.hccl_config = hccl_config
            except (TypeError, AttributeError) as e:
                # Old torch_npu may reject the update; degrade to a tracer
                # without group_name, but keep the failure observable.
                LOG.warning("Failed to inject group_name %r into hccl_config: %s", pg_name, e)
        return options

    return wrapper
