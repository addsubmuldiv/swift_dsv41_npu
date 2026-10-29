# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
MSTT_DEBUGGER = None


def init_debugger():
    """Instantiate the msprobe PrecisionDebugger (idempotent).

    Must be called at the patch-registration stage, the only timing where:

    1. pre-stage patches are already applied: construction imports megatron,
       so torch.compile must already be dummy_compile, otherwise
       megatron.core.jit captures the real one and JIT warmup breaks;
    2. the model is not built yet: construction wraps F.gelu/F.silu, and
       MLP.forward's identity checks require the model to hold the wrapped
       objects.

    Stale activation_func defaults (captured pre-wrap, since msprobe imports
    megatron before wrapping) are repaired by _normalize_stale_activation_func
    in the transformer_engine patches.
    """
    global MSTT_DEBUGGER
    if MSTT_DEBUGGER is None:
        try:
            from msprobe.pytorch import PrecisionDebugger
        except ImportError as e:
            raise AssertionError('Mstt not found. You can install it with `pip install mindstudio-probe`.') from e
        MSTT_DEBUGGER = PrecisionDebugger(config_path=str(current_dir / 'config.json'))
    return MSTT_DEBUGGER


def dump_start_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        init_debugger().start(model=args[2])
        return fn(*args, **kwargs)

    return wrapper


def dump_end_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        debugger = init_debugger()
        debugger.stop()
        debugger.step()

    return wrapper
