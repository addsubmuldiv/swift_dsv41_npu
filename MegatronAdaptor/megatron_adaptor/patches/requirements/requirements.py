import math
from functools import wraps

import torch


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

    return wrapper


def ensure_contiguous_wrapper(fn):
    @wraps(fn)
    def wrapper(tensor, *args, **kwargs):
        tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
        return fn(tensor, *args, **kwargs)

    return wrapper


def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def dummy_function(*args, **kwargs):
    pass


def torch_all_reduce_double_dtype_bypass_wrapper(fn):
    """Reduce double tensors on a float32 copy, then copy the result back.

    HCCL lacks fp64 collectives. The handle is returned unchanged so the async
    contract holds; it is already completed, so a later ``wait()`` is a no-op.
    """

    @wraps(fn)
    def wrapper(tensor, *args, **kwargs):
        if torch.is_tensor(tensor) and tensor.dtype == torch.double:
            float_tensor = tensor.float()
            handle = fn(float_tensor, *args, **kwargs)
            if handle is not None:
                handle.wait()
            tensor.copy_(float_tensor)
            return handle

        return fn(tensor, *args, **kwargs)

    return wrapper


def dummy_compile(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):

        def wrapper(*fn_args, **fn_kwargs):
            return args[0](*fn_args, **fn_kwargs)

        return wrapper
    else:

        def compile_wrapper(fn):
            def wrapper(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)

            return wrapper

        return compile_wrapper


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        return '2.2.0' if name == 'transformer-engine' else fn(name, *args, **kwargs)

    return wrapper
