# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Control import-time adaptor execution without importing the adaptor package."""

from collections.abc import Callable

# Process-wide switch. Decorated functions run at import time, which is
# serialized by Python's import lock, so a plain dict is race-free.
_STATE = {"auto_execute": True}


class AutoExecuteFunction:
    """Execute a decorated function only while automatic execution is enabled."""

    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, *args, **kwargs):
        if not _STATE["auto_execute"]:
            return None
        return self._func(*args, **kwargs)


class NoExecuteFunction:
    """Temporarily suppress functions decorated with ``AutoExecuteFunction``.

    The flag is process-wide; ``__exit__`` restores the value saved on
    ``__enter__``, so nested contexts and user presets are preserved.
    """

    def __init__(self):
        self._prev = True

    def __enter__(self):
        self._prev = _STATE["auto_execute"]
        _STATE["auto_execute"] = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _STATE["auto_execute"] = self._prev


__all__ = ["AutoExecuteFunction", "NoExecuteFunction"]
