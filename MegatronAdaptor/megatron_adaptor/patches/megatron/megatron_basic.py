import logging
from functools import wraps
import time
import threading

import torch
from torch import _C
import torch_npu
from torch_npu.npu import _lazy_call, device as device_ctx_manager

from megatron.core.transformer.enums import AttnMaskType

logger = logging.getLogger(__name__)

# Thread-local storage for range counter and stack
_range_local = threading.local()


def _get_thread_range_counter():
    if not hasattr(_range_local, 'counter'):
        _range_local.counter = 0
    _range_local.counter += 1
    return _range_local.counter


def _get_range_stack():
    if not hasattr(_range_local, 'stack'):
        _range_local.stack = []
    return _range_local.stack


def _range_push_patch(msg):
    range_id = _get_thread_range_counter()
    torch_npu.npu.mstx.range_start(msg, range_id)
    _get_range_stack().append(range_id)


def _range_pop_patch():
    stack = _get_range_stack()
    if stack:
        range_id = stack.pop()
        torch_npu.npu.mstx.range_end(range_id)
    else:
        import traceback

        logger.warning(
            "Attempted to pop NVTX range from empty stack. "
            "This indicates a mismatch between range_push and range_pop calls. "
            "Call stack:\n%s",
            traceback.format_stack(),
        )


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):

        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            '>>> done with dataset index builder. Compilation time: {:.3f} seconds'.format(time.time() - start_time),
            flush=True,
        )


def get_device_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        backend = torch.distributed.get_backend()
        local_rank = args[0]
        if backend == 'hccl':
            if local_rank is None:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{local_rank}')
        else:
            device = func(*args, **kwargs)
        return device

    return wrapper


def get_device_arch_version():
    return 8


class ScaledUpperTriangMaskedSoftmax:
    @staticmethod
    def apply(input_, scale):
        size = input_.size()
        input_ = input_.view((1,) + tuple(size)).contiguous()
        dummy_mask = torch.zeros(input_.size()).bool().npu()
        output = torch_npu.npu_scaled_masked_softmax(input_, dummy_mask, scale, True)
        return output.view(size).contiguous()


class ScaledMaskedSoftmax:
    @staticmethod
    def apply(input_, mask, scale):
        return torch_npu.npu_scaled_masked_softmax(input_, mask, scale, False)


class ScaledSoftmax:
    @staticmethod
    def apply(input_, scale):
        dummy_mask = torch.zeros(input_.size()).bool().npu()
        return torch_npu.npu_scaled_masked_softmax(input_, dummy_mask, scale, False)


def is_kernel_available(self, mask, b, np, sq, sk):
    return (
        self.scaled_masked_softmax_fusion  # user want to fuse
        and self.input_in_float16  # input must be fp16
        and 32 < sk <= 4096  # sk must be 32 ~ 4096
        and sq % 16 == 0  # sq must be divisor of 16
        and sk % 16 == 0  # sk must be divisor of 16
    )


def forward_fused_softmax(self, input_, mask):
    b, np, sq, sk = input_.size()
    scale = self.scale if self.scale is not None else 1.0

    if self.attn_mask_type == AttnMaskType.causal:
        assert sq == sk, "causal mask is only for self attention"
        # if use upper triangular mask
        return torch_npu.npu_scaled_masked_softmax(input_, mask, scale, True)
    else:
        if mask is not None:
            return torch_npu.npu_scaled_masked_softmax(input_, mask, scale, False)
        else:
            return ScaledSoftmax.apply(input_, scale)


@staticmethod
def preload_tensors(write_buckets, non_blocking=True):
    """Stage to-be-written tensors into host memory during async checkpoint save.

    To prevent the staged data from being corrupted by subsequent training steps:
    1. Device tensors are copied D2H synchronously (non_blocking=False) instead
       of honoring the caller's non_blocking, so a tensor will not be updated
       in place by the next step before the D2H copy completes.
    2. Tensors already on CPU are clone()d, returning an independent snapshot
       rather than a reference to the original object.
    """
    result = []
    for bucket in write_buckets:
        file_name, storage_key, (bytes_data, tensor_data) = bucket
        tensor_data = [
            (item, tensor.to("cpu", non_blocking=False) if not tensor.is_cpu else tensor.clone())
            for item, tensor in tensor_data
        ]
        result.append((file_name, storage_key, (bytes_data, tensor_data)))
    if non_blocking:
        torch.cuda.synchronize()
    return result
