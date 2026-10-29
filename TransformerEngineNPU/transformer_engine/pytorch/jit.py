# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import os
from functools import wraps

import torch

from .torch_version import torch_version

if torch.__version__ >= "2":
    import torch._dynamo

    def no_torch_dynamo(recursive=True):
        """Decorator to disable Torch Dynamo, except during ONNX export."""

        def decorator(f):
            # no "recursive" option in pyTorch 2.0 - it acts as if recursive was True
            disabled_f = (
                torch._dynamo.disable(f, recursive=recursive)
                if torch.__version__ >= "2.1"
                else torch._dynamo.disable(f)
            )

            @wraps(f)
            def wrapper(*args, **kwargs):
                # if is_in_onnx_export_mode():
                #     return f(*args, **kwargs)
                return disabled_f(*args, **kwargs)

            return wrapper

        return decorator

else:
    # Fallback for PyTorch < 2.0: no-op decorator
    def no_torch_dynamo(recursive=True):  # pylint: disable=unused-argument
        """No-op decorator for PyTorch < 2.0."""
        return lambda func: func


def lazy_compile(func):
    """Lazy compile a function with torch.compile

    This decorator defers the compilation of a function until the first call, speeding up the
    overall module's import time if these functions are not used.
    """
    compiled_func = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal compiled_func
        if compiled_func is None:
            compiled_func = torch.compile(func)
        return compiled_func(*args, **kwargs)

    return wrapper


jit_fuser = lambda func: func
if torch_version() >= (2, 0, 0) and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
    jit_fuser = lazy_compile


# See: https://github.com/NVIDIA/TransformerEngine/issues/597
dropout_fuser = torch.jit.script
if torch_version() >= (2, 2, 0) and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
    dropout_fuser = lazy_compile
