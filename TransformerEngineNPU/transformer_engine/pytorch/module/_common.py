# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Common utilities for TransformerEngine NPU PyTorch modules"""

import dataclasses
import queue
from typing import Any, Callable, List, Optional, Tuple

import torch

from transformer_engine.pytorch.utils import get_default_init_method


class _NoopCatFunc(torch.autograd.Function):
    """Concatenate tensors, doing a no-op if possible

    See _noop_cat.

    """

    @staticmethod
    def forward(
        ctx: Any,
        dim: int,
        *tensors: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # Check first tensor
        if not tensors:
            raise ValueError("Attempted to concatenate 0 tensors")

        # Check concat dim
        num_dims = tensors[0].dim()
        if not -num_dims <= dim < num_dims:
            raise ValueError(
                "Attempted to concatenate tensor "
                f"with shape {list(tensors[0].size())} along dim {dim}"
            )
        dim %= num_dims

        # Check remaining tensors
        out_shape = list(tensors[0].size())
        split_ranges = [(0, tensors[0].size(dim))]
        for tensor in tensors[1:]:
            in_shape = list(tensor.size())
            if (
                len(in_shape) != num_dims
                or in_shape[:dim] != out_shape[:dim]
                or in_shape[dim + 1 :] != out_shape[dim + 1 :]
            ):
                raise ValueError(
                    "Attempted to concatenate tensors with shapes "
                    f"{[list(tensor.size()) for tensor in tensors]} "
                    f"along dim {dim}"
                )
            split_start = out_shape[dim]
            split_end = split_start + in_shape[dim]
            out_shape[dim] = split_end
            split_ranges.append((split_start, split_end))

        # Save state for backward
        ctx.dim = dim
        ctx.split_ranges = split_ranges

        # Tensor properties from first tensor
        dtype = tensors[0].dtype
        device = tensors[0].device
        strides = tensors[0].stride()
        data_ptr_stride = strides[dim] * tensors[0].element_size()

        # Out-of-place concatenation when view tensors have different storage
        # Note: This works around an edge case with the split_quantize
        # function, which might allocate a buffer and construct
        # subviews. However, in order to reduce CPU overheads, these
        # views are configured manually outside of PyTorch. PyTorch
        # doesn't know these views share the same memory, and it
        # blocks us from reconstructing the full tensor because it
        # thinks we are accessing out-of-bounds memory.
        if tensors[0].untyped_storage().nbytes() < out_shape[dim] * data_ptr_stride:
            return torch.cat(tensors, dim=dim)

        # Out-of-place concatenation if tensor properties do not match
        data_ptr = tensors[0].data_ptr() + tensors[0].size(dim) * data_ptr_stride
        for tensor in tensors[1:]:
            if (
                tensor.dtype != dtype
                or tensor.device != device
                or tensor.stride() != strides
                or tensor.data_ptr() != data_ptr
            ):
                return torch.cat(tensors, dim=dim)
            data_ptr += tensor.size(dim) * data_ptr_stride

        # No-op concatenation
        out = tensors[0].as_strided(out_shape, strides)
        out.requires_grad = any(tensor.requires_grad for tensor in tensors)
        return out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        grad_inputs = []
        for split_start, split_end in ctx.split_ranges:
            slices = [slice(None)] * grad_output.dim()
            slices[ctx.dim] = slice(split_start, split_end)
            grad_inputs.append(grad_output[tuple(slices)])
        return None, *grad_inputs


def noop_cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate tensors, doing a no-op if possible

    If tensors are already concatenated in memory, a tensor view of
    that memory region will be returned. Otherwise the tensors will be
    concatenated out-of-place, as usual.

    """
    if not tensors:
        raise ValueError("Attempted to concatenate 0 tensors")
    if len(tensors) == 1:
        return tensors[0]
    # if is_in_onnx_export_mode():
    #     return torch.cat(tensors, dim=dim)
    return _NoopCatFunc.apply(dim, *tensors)


@dataclasses.dataclass
class _ParameterInitMeta:
    """
    Stores essential metadata needed to support deferred parameter initialization.
    """

    init_fn: Optional[Callable] = get_default_init_method()
    get_rng_state_tracker: Optional[Callable] = None
    fp8_meta_index: Optional[int] = None

    def __post_init__(self):
        """Safeguard reference to the parameter's parent module and initialization function."""
        if self.init_fn is None:
            self.init_fn = get_default_init_method()


class WeightGradStore:
    """
    A class to manage weight gradient storage and computation in Transformer modules.
    This class enables split backward propagation for better memory efficiency.
    """

    def __init__(self, delay_wgrad_compute=False, ub_bulk_wgrad=False):
        """
        Initialize the WeightGradStore.

        Args:
            delay_wgrad_compute (bool): Whether to delay weight gradient computation
            ub_bulk_wgrad (bool): Whether to enable bulk weight gradient computation
        """
        if delay_wgrad_compute:
            self.context = queue.Queue()
            self.recomputed_inputs = queue.Queue()
            assert ub_bulk_wgrad is False, (
                "ub_bulk_wgrad is not supported when enabling delay_wgrad_compute"
            )
            self.enabled = delay_wgrad_compute
        else:
            self.context = None
            self.recomputed_inputs = None
            self.enabled = False

    def delay_wgrad_compute(self):
        """
        Get the current split backward propagation status.

        Returns:
            bool: True if split backward is enabled, False otherwise
        """
        return self.enabled

    def enable_delay_wgrad_compute(self):
        """Enable split backward propagation."""
        self.enabled = True

    def disable_delay_wgrad_compute(self):
        """Disable split backward propagation."""
        self.enabled = False

    def put(self, tensor_list, func):
        """
        Store tensors and computation function for later execution.

        Args:
            tensor_list (list): List of tensors needed for computation
            func (callable): Function to be executed with the tensors
        """
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        self.context.put([tensor_list, func])

    def put_recomputed_input(self, tensor):
        """Provide a recomputed input to the next delayed wgrad computation."""
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        self.recomputed_inputs.put(tensor)

    def pop_recomputed_input(self):
        """Consume a recomputed delayed-wgrad input, if one was provided."""
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        if self.recomputed_inputs.empty():
            return None
        return self.recomputed_inputs.get()

    def pop(self):
        """
        Execute the stored computation with the stored tensors.
        Raises an exception if the queue is empty.
        """
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        if self.context.qsize() > 0:
            tensor_list, func = self.context.get()
            # 规避小算子的in-place检查，对齐原生行为
            with torch.no_grad():
                return func(*tensor_list), tensor_list
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            raise RuntimeError(f"Pop empty queue. rank {rank}")
        raise RuntimeError("Pop empty queue. No distributed environment detected.")

    def assert_empty(self):
        """
        Assert that the queue is empty.
        Used for debugging and ensuring proper cleanup.
        """
        assert self.enabled is True, "delay_wgrad_compute is not enabled"
        rank = torch.distributed.get_rank()
        assert self.context.empty(), f"Queue is not empty. rank {rank}"
        assert self.recomputed_inputs.empty(), f"Recomputed input queue is not empty. rank {rank}"
