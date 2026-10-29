# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

import torch
import torch_npu

from megatron_adaptor.utils.args_utils import get_full_args

PROFILE_RECORD = None


def train_wrapper(train):
    @wraps(train)
    def wrapper(*args, **kwargs):
        args_ = get_full_args()
        if args_.profile:
            args_.profile_npu = True
            args_.profile = False
        else:
            args_.profile_npu = False

        is_profile = (
            hasattr(args_, 'profile_npu')
            and args_.profile_npu
            and ((torch.distributed.get_rank() in args_.profile_ranks) or (-1 in args_.profile_ranks))
        )
        if is_profile:
            global PROFILE_RECORD
            active = args_.profile_step_end - args_.profile_step_start
            skip_first = args_.profile_step_start

            if args_.profile_with_cpu:
                activities = [torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU]
            else:
                activities = [torch_npu.profiler.ProfilerActivity.NPU]

            level2level = {
                'level0': torch_npu.profiler.ProfilerLevel.Level0,
                'level1': torch_npu.profiler.ProfilerLevel.Level1,
                'level2': torch_npu.profiler.ProfilerLevel.Level2,
            }
            profiler_level = level2level[args_.profile_level]

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=profiler_level, l2_cache=False
            )

            with torch_npu.profiler.profile(
                activities=activities,
                record_shapes=args_.profile_record_shapes,
                profile_memory=args_.profile_with_memory,
                with_stack=args_.profile_with_stack,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1, skip_first=skip_first),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args_.profile_save_path),
            ) as prof:
                PROFILE_RECORD = prof
                return train(*args, **kwargs)
        return train(*args, **kwargs)

    return wrapper


def profile_step():
    """Advance the NPU profiler by one training step. No-op unless active.

    This is the stable entry point that lets other components drive the
    profiling window without coupling to any internal state. Any training
    implementation that fully replaces ``megatron.training.training.train_step``
    (e.g. MindSpeed dualpipev / async-log-allreduce) must call this once per
    step so the profile window keeps advancing. It is safe to call
    unconditionally: profiling state is lazily set by ``train_wrapper`` only on
    the ranks inside the profile window, so this is a zero-cost no-op otherwise.
    """
    # PROFILE_RECORD is only assigned while the profiler context is active on a
    # profiled rank, so checking it (rather than full args) is the cheapest gate.
    record = PROFILE_RECORD
    if record is None:
        return
    # profile_npu is a side-effect flag set by train_wrapper before the training
    # loop; when train_step is called directly without going through train()
    # (e.g. a custom main loop) the attribute is absent, so treat it as off.
    args_ = get_full_args()
    if not hasattr(args_, 'profile_npu') or not args_.profile_npu:
        return
    if (torch.distributed.get_rank() in args_.profile_ranks) or (-1 in args_.profile_ranks):
        record.step()


def train_step_wrapper(train_step):
    @wraps(train_step)
    def wrapper(*args, **kwargs):
        ret = train_step(*args, **kwargs)
        profile_step()
        return ret

    return wrapper
