# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

import torch


def prepare_existing_model_chunks_for_distributed_training_wrapper(fn):
    """Wrapper that ensures all model params are moved to device.

    Background
    ----------
    Commit 223e244f5 (Finish ModelBuilder integration) changed the code path
    routing in ``setup_model_and_optimizer``: standard GPT pretraining now
    goes through ``builder.build_distributed_models()`` →
    ``prepare_existing_model_chunks_for_distributed_training()`` instead of
    the old ``get_model()`` path.

    The two code paths use **different** GPU allocation conditions:

    - Old ``get_model()`` (correct):
      ``not (use_torch_fsdp2 and use_cpu_initialization)``
      → skips .cuda() only when BOTH flags are True.

    - New path in dist_utils.py (buggy):
      ``not use_torch_fsdp2 and not use_cpu_initialization``
      → skips .cuda() when EITHER flag is True.

    With ``--use-cpu-initialization=True`` and ``--use-torch-fsdp2=False``,
    the old path calls .cuda(); the new path skips it, leaving params on CPU.
    ``Float16Module`` then wraps them as CPU bf16 params, and the optimizer
    rejects them with::

        TypeError: Wrapped parameters must be one of
        torch.cuda.FloatTensor, torch.cuda.HalfTensor, or
        torch.cuda.BFloat16Tensor. Received torch.BFloat16Tensor

    Fix
    ---
    This wrapper runs after the original function and ensures all model params
    are moved to the NPU device by calling .cuda() directly (which is idempotent
    and skips params already on the device). It does NOT modify the upstream
    condition — other callers (MiMo etc.) may depend on the current semantics.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Move params to CUDA *before* calling the original function.
        #
        # The original ``prepare_existing_model_chunks_for_distributed_training``
        # has a De Morgan bug in its CUDA-allocation condition (it skips .cuda()
        # when --use-cpu-initialization is True, leaving params on CPU).  The old
        # ``get_model()`` path correctly calls .cuda() before Float16Module wrap
        # and DDP wrap.  If we call .cuda() *after* the original function (as the
        # previous version of this wrapper did), DDP's gradient accumulation hooks
        # are attached to the CPU params' grad_fn.  The subsequent .cuda() creates
        # *new* CUDA tensors with different grad_fn objects, so the hooks never
        # fire → gradients stay None → grad norm = 0.000.
        #
        # Fix: restore the correct ordering — .cuda() first, then let the original
        # function wrap with Float16Module and DDP (which then see CUDA params and
        # attach hooks to the correct grad_fn).
        model_list = args[0]
        use_torch_fsdp2 = kwargs.get('use_torch_fsdp2', False)

        if not use_torch_fsdp2:
            for model_module in model_list:
                model_module.cuda(torch.cuda.current_device())

        return fn(*args, **kwargs)

    return wrapper
