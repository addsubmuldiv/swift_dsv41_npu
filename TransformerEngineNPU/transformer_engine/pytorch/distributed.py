# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Distributed utilities for TransformerEngine Ascend PyTorch API

This module provides distributed utilities for NPU-optimized operations,
integrating MindSpeed's distributed utilities with TransformerEngine's interface.
"""

import math
import warnings
from contextlib import AbstractContextManager, ContextDecorator, contextmanager
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import _C
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._traversal_utils import _get_fsdp_states_with_modules
from torch.utils.checkpoint import detach_variable, noop_context_fn
from torch_npu.npu import _lazy_call, _lazy_init
from torch_npu.npu import device as device_ctx_manager

from .constants import dist_group_type
from .quantization.manager import (
    FP8GlobalStateManager,
    autocast,
)
from .quantized_tensor import QuantizedTensor
from .utils import (
    safely_set_viewless_tensor_data,
)

_USE_REENTRANT_ACTIVATION_RECOMPUTE = True

_FP8_ACTIVATION_RECOMPUTE_ENABLED = False
_FP8_ACTIVATION_RECOMPUTE_PHASE = False


_ALL_ACTIVE_RNG_STATES = {}


def get_all_rng_states() -> bool:
    """Returns all generator states used by `CudaRNGStatesTracker`."""
    return _ALL_ACTIVE_RNG_STATES


def set_all_rng_states(states: List) -> None:
    """Updates all generator states used by `CudaRNGStatesTracker`."""
    global _ALL_ACTIVE_RNG_STATES
    _ALL_ACTIVE_RNG_STATES = states


def graph_safe_rng_available() -> bool:
    """Returns whether cuda graph safe RNG state manipulation is supported."""
    return (
        hasattr(torch.cuda.CUDAGraph, "register_generator_state")
        and hasattr(torch.Generator, "graphsafe_set_state")
        and hasattr(torch.Generator, "graphsafe_get_state")
        and hasattr(torch.Generator, "clone_state")
    )


def _set_cuda_rng_state(new_state, device=-1, graph_safe: bool = False):
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("npu")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("npu", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.npu.current_device()
            default_generator = torch.npu.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "npu",
    clone: bool = False,
    graph_safe: bool = True,
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU."""

    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("npu", device)
    idx = device.index
    if idx is None:
        idx = torch.npu.current_device()
    default_generator = torch.npu.default_generators[idx]
    if graph_safe_rng_available() and graph_safe:
        if clone:
            # Reference to the cloned generator state
            return default_generator.clone_state()
        # Reference to the current generator state
        return default_generator.graphsafe_get_state()
    return default_generator.get_state()


# region recompute checkpoint


class activation_recompute_forward(AbstractContextManager, ContextDecorator):
    _is_first_fp8_module: List = []

    def __init__(self, activation_recompute: bool = False, recompute_phase: bool = False):
        super().__init__()
        self.activation_recompute = activation_recompute
        self.recompute_phase = recompute_phase

    def __enter__(self):
        global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = self.activation_recompute
        _FP8_ACTIVATION_RECOMPUTE_PHASE = self.recompute_phase

        qstate = FP8GlobalStateManager.quantization_state
        if self.activation_recompute and not self.recompute_phase:
            activation_recompute_forward._is_first_fp8_module.append(qstate.is_first_fp8_module)
        if self.activation_recompute and self.recompute_phase:
            qstate.is_first_fp8_module = activation_recompute_forward._is_first_fp8_module.pop(0)

    def __exit__(self, *exc_details):
        global _FP8_ACTIVATION_RECOMPUTE_ENABLED, _FP8_ACTIVATION_RECOMPUTE_PHASE
        _FP8_ACTIVATION_RECOMPUTE_ENABLED = False
        _FP8_ACTIVATION_RECOMPUTE_PHASE = False


def get_activation_recompute_contexts():
    """Returns context objects for the checkpointed forward pass and the forward recompute phase."""
    forward_ctx = activation_recompute_forward(
        activation_recompute=True,
        recompute_phase=False,
    )
    recompute_ctx = activation_recompute_forward(
        activation_recompute=True,
        recompute_phase=True,
    )
    return forward_ctx, recompute_ctx


def is_fp8_activation_recompute_enabled() -> bool:
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_ENABLED


def in_fp8_activation_recompute_phase() -> bool:
    """Return global boolean"""
    return _FP8_ACTIVATION_RECOMPUTE_PHASE


@lru_cache
def get_te_classes():
    """
    Return all Transformer Engine modules.
    """
    from .module import LayerNorm, RMSNorm
    from .module.base import TransformerEngineBaseModule
    from .attention.dot_product_attention.dot_product_attention import (
        DotProductAttention,
    )

    return (
        LayerNorm,
        RMSNorm,
        TransformerEngineBaseModule,
        DotProductAttention,
    )


def has_te_modules(network):
    """
    Check if there are any Transformer Engine modules in the network.
    """
    te_classes = get_te_classes()
    if isinstance(network, torch.nn.Module):
        return any(isinstance(module, te_classes) for module in network.modules())

    # Cannot check for TE modules inside a custom class/callable that's not a torch.nn.Module,
    # so just assume that it has TE modules just to be safe.
    return True


def _is_te_module(instance):
    # pylint: disable=cyclic-import
    """
    Check if given module is a Transformer Engine module that requires the TE checkpoint
    implementation for activation recompute.
    """
    return isinstance(
        instance,
        get_te_classes(),
    )


@torch._disable_dynamo
def checkpoint(
    function: Callable,
    *args: Tuple[torch.Tensor, ...],
    **kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, ...]:
    """
    Checkpoint a part of the model by trading compute for memory. This function is based on
    `torch.utils.checkpoint.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`_.

    .. warning::

        It is the user's responsibility to ensure identical behavior when calling
        :attr:`function` from the forward and backward pass. If different output is
        produced (e.g. due to global state), then the checkpointed version won't
        be numerically equivalent.

    .. warning::
        `use_reentrant=False` does not support early stopping, and will execute the entire forward
        pass for the checkpointed module when recomputing activations in the backward pass.

    Parameters
    ----------
    function : Callable
            pytorch module used to run the forward and backward passes using
            the specified :attr:`args` and :attr:`kwargs`.
    distribute_saved_activations : bool, default = False
            if set to ``True`` and ``use_reentrant=True``, first tensor argument is distributed
            across the specified tensor parallel group (``tp_group``) before saving it for the
            backward pass. This has no effect when ``use_reentrant=False``.
    get_rng_state_tracker : Callable, default = None
            python callable which returns an instance of :class:`CudaRNGStatesTracker`.
    tp_group : ProcessGroup, default = None
            tensor parallel process group. Used only when ``distribute_saved_activations=True``
            and ``use_reentrant=True``. If ``None``, it falls back to the default group.
    use_reentrant : bool, default = True
            perform checkpointing in reentrant mode.
    args : tuple
            tuple of torch tensors for inputs to :attr:`function`.
    kwargs : dict
            dictionary of string keys for keyword arguments to :attr:`function`.
    """
    # Pop out te.distributed.checkpoint() arguments
    global _USE_REENTRANT_ACTIVATION_RECOMPUTE
    _USE_REENTRANT_ACTIVATION_RECOMPUTE = kwargs.pop("use_reentrant", True)
    distribute_saved_activations = kwargs.pop("distribute_saved_activations", False)
    tp_group = kwargs.pop("tp_group", None)
    get_rng_state_tracker = kwargs.pop("get_rng_state_tracker", None)

    # Ensure backward compatibility.
    if (
        len(args) > 3
        and isinstance(args[0], bool)
        and callable(args[1])
        and isinstance(args[2], None | dist_group_type)
    ):
        warnings.warn(
            "Passing non-tensor non-keyword arguments is deprecated and support will be removed in "
            "future releases of TransformerEngine. `distribute_saved_activations`, `tp_group`, and "
            "`get_rng_state_tracker` must be passed as keyword arguments to `checkpoint`.",
            DeprecationWarning,
            stacklevel=2,
        )
        distribute_saved_activations = args[0]
        get_rng_state_tracker = args[1]
        tp_group = args[2]
        args = args[3:]

    # Trigger the native PyTorch checkpoint if the function is not or does not contain a
    # Transformer Engine module.
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    determinism_check = kwargs.pop("determinism_check", "default")
    debug = kwargs.pop("debug", False)
    if not has_te_modules(function):
        return torch.utils.checkpoint.checkpoint(
            function,
            *args,
            use_reentrant=_USE_REENTRANT_ACTIVATION_RECOMPUTE,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            **kwargs,
        )

    from .module.base import TransformerEngineBaseModule

    if isinstance(function, TransformerEngineBaseModule):
        # If this TE module is FSDP-wrapped, clear its FSDP group information because there's no need
        # to scatter/gather activations that we will recompute anyway.
        function.fast_setattr("fsdp_wrapped", False)
        function.fast_setattr("fsdp_group", None)

    # Otherwise discard unused te.utils.checkpoint.checkpoint() arguments
    # and execute TE's own checkpointing
    # NOTE: This logic uses the TE checkpoint on all custom callable `function` handles because we
    #       cannot be sure there are no TE modules inside the function. It also means we might run
    #       the TE checkpoint for non-TE modules, so the TE checkpoint has to support a potential
    #       user context function.
    del determinism_check, debug
    if _USE_REENTRANT_ACTIVATION_RECOMPUTE:
        # If saved activations need to be distributed but there is no process group,
        # default to the world group.
        if distribute_saved_activations:
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "torch.distributed is not initialized. Call "
                    "torch.distributed.init_process_group() before using "
                    "distribute_saved_activations=True."
                )
            tp_group = torch.distributed.GroupMember.WORLD if tp_group is None else tp_group

        return _CheckpointFunction.apply(
            function,
            distribute_saved_activations,
            get_rng_state_tracker,
            tp_group,
            context_fn,
            kwargs,
            *args,
        )

    if distribute_saved_activations:
        warnings.warn(
            "`distribute_saved_activations=True` has no effect when `use_reentrant=False`. "
            "The non-reentrant checkpoint implementation does not manually store forward "
            "inputs for the activation recompute in the backward pass, and instead leverages "
            "the autograd engine's pack/unpack hooks."
        )

    user_forward_ctx, user_recompute_ctx = context_fn()
    te_forward_ctx, te_recompute_ctx = get_activation_recompute_contexts()

    # TODO Preserve the torch autocast contexts from the forward pass during recompute phase.

    fp8 = FP8GlobalStateManager.is_fp8_enabled()
    fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None

    def recompute_fn(*args, **kwargs):
        with (
            torch.autograd.enable_grad(),
            te_recompute_ctx,
            user_recompute_ctx,
            autocast(enabled=fp8, recipe=fp8_recipe),
        ):
            function(*args, **kwargs)

    # Initialize a new checkpoint frame for each new forward pass.
    new_frame = _CheckpointFrame(
        recompute_fn,
        get_rng_state_tracker,
    )
    new_frame.cache_rng_states(forward=True)

    with _checkpoint_hook(new_frame, args, kwargs), te_forward_ctx, user_forward_ctx:
        out = function(*args, **kwargs)

    return out


if hasattr(torch, "_disable_dynamo"):
    checkpoint = torch._disable_dynamo(checkpoint)


class _CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        distribute_saved_activations: bool,
        get_rng_state_tracker: Union[Callable, None],
        tp_group: Union[torch.distributed.ProcessGroup, None],
        context_fn: Union[Callable, None],
        kwargs: Dict[str, Any],
        *args: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Call forward function while saving state to be able to
        redo the computation later.
        """
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        if context_fn is not None:
            forward_ctx, recompute_ctx = context_fn()
        else:
            forward_ctx, recompute_ctx = noop_context_fn()

        # Preserve torch autocast context for the backward pass

        with torch.no_grad(), forward_ctx:
            with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
                outputs = run_function(*args, **kwargs)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0],
                split_tensor_into_1d_equal_chunks(args[0].data, tp_group=tp_group, new_buffer=True),
            )

        # Store everything.
        ctx.fwd = [arg if not torch.is_tensor(arg) else None for arg in args]
        tensor_inputs = [arg if torch.is_tensor(arg) else None for arg in args]
        ctx.save_for_backward(*tensor_inputs)

        fp8 = FP8GlobalStateManager.is_fp8_enabled()
        ctx.get_rng_state_tracker = get_rng_state_tracker
        ctx.tp_group = tp_group
        ctx.recompute_ctx = recompute_ctx
        ctx.fp8 = fp8
        ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        ctx.kwargs = kwargs

        return outputs

    @staticmethod
    def backward(
        ctx, *args: Tuple[Union[torch.Tensor, None], ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Call backward function with activation recomputation."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        inputs = tuple(t if t is not None else arg for (t, arg) in zip(ctx.saved_tensors, ctx.fwd))

        get_rng_state_tracker = ctx.get_rng_state_tracker

        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0],
                gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(ctx.input_0_shape),
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            bwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with (
            torch.enable_grad(),
            ctx.recompute_ctx,
            activation_recompute_forward(activation_recompute=True, recompute_phase=True),
            autocast(enabled=ctx.fp8, recipe=ctx.fp8_recipe),
        ):
            outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True, this checkpoint() is not necessary"
            )

        # backward does not require entering autocast context because
        # backward implementations already retrieve fp8 recipe and
        # enablement from stored ctx.
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs
        )
        return (None, None, None, None, None, None) + grads


class _CheckpointFrame:
    """
    Storage frame for forward RNG states and detached activations from the forward recompute.
    """

    def __init__(self, recompute_fn: Callable, get_rng_state_tracker: Callable):
        self.recompute_fn = recompute_fn
        self.recomputed = []
        self.count = 0
        self.get_rng_state_tracker = get_rng_state_tracker
        self.fwd_rng_states = None
        self.bwd_rng_states = None

    def cache_rng_states(self, forward=True):
        """Cache fwd/bwd RNG states in the frame to restore later."""
        rng_states = (
            torch.get_rng_state(),
            _get_cuda_rng_state(graph_safe=False),
        )
        if self.get_rng_state_tracker is not None:
            rng_states += (self.get_rng_state_tracker().get_states(),)

        if forward:
            self.fwd_rng_states = rng_states
        else:
            self.bwd_rng_states = rng_states

    def restore_rng_states(self, forward=True):
        """Restore fwd/bwd RNG states that were previously cached into the frame."""

        if forward:
            rng_states = self.fwd_rng_states
        else:
            rng_states = self.bwd_rng_states

        torch.set_rng_state(rng_states[0])
        _set_cuda_rng_state(rng_states[1], graph_safe=False)
        if self.get_rng_state_tracker is not None:
            self.get_rng_state_tracker().set_states(rng_states[2])


class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the activation recompute phase."""

    def __init__(self, frame):
        def pack_hook(x):
            """
            Packing hook for each recomputed activation passed into the `ctx.save_for_backward()`
            call in the forward recomputation.
            """
            frame.recomputed.append(x.detach())
            return x.detach()

        def unpack_hook(x):
            """
            No-op unpack hook that will never be called because the backward pass for the
            forward recomputation is never triggered.
            """
            return x

        super().__init__(pack_hook, unpack_hook)


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the checkpointed forward pass."""

    def __init__(self, frame, args, kwargs):
        def pack_hook(x):
            """
            Packing hook for each tensor passed into `ctx.save_for_backward()` call in the
            forward pass. Since this is the first forward pass, we discard the tensor and instead
            pack a placeholder tensor index into the autograd engine context.
            """
            del x
            idx = frame.count
            frame.count += 1
            return idx

        def unpack_hook(idx):
            """
            Unpacking hook for each tensor that comes out of the `ctx.saved_tensors` call in the
            backward pass. The first time this is called, the _recomputation_hook will save all the
            activation tensors from `ctx.save_for_backward()` in the forward recomputation into the
            _CheckpointFrame. Subsequent calls will simply return the already recomputed activation
            tensor at the given index of the _CheckpointFrame storage.
            """

            if not frame.recomputed:
                # Store current RNG states in the backward pass
                frame.cache_rng_states(forward=False)

                # Set RNG states to what we saved before the forward pass
                frame.restore_rng_states(forward=True)

                # Recompute the forward pass
                with _recomputation_hook(frame):
                    frame.recompute_fn(*args, **kwargs)

                # Restore RNG states back to the backward pass
                frame.restore_rng_states(forward=False)

            # Return the already recomputed activation tensor at the given index
            activation = frame.recomputed[idx]
            frame.recomputed[idx] = None
            return activation

        super().__init__(pack_hook, unpack_hook)


# endregion recompute checkpoint


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor,
    is_parallel: bool,
    dim: int,
    stride: int,
) -> None:
    """Set tensor model parallel attributes."""
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


@lru_cache
def get_distributed_world_size(group: Optional[dist_group_type] = None) -> int:
    """Return world size for the distributed group."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


@lru_cache
def get_distributed_rank(group: Optional[dist_group_type] = None) -> int:
    """Return my rank for the distributed group."""
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. Call torch.distributed.init_process_group() "
            "before calling get_distributed_rank()."
        )
    return torch.distributed.get_rank(group=group)


def get_tensor_parallel_group() -> Optional[Any]:
    """Get tensor parallel group."""
    return None


def allreduce(
    tensor: torch.Tensor,
    group: Optional[Any] = None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """All-reduce operation."""
    if group is None or not dist.is_initialized():
        return tensor, None

    if async_op:
        handle = dist.all_reduce(tensor, group=group, async_op=True)
        return tensor, handle
    else:
        dist.all_reduce(tensor, group=group)
        return tensor, None


def symmetric_all_reduce(
    tensor: torch.Tensor,
    group: Optional[Any] = None,
    all_reduce_type: Optional[str] = None,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """Symmetric all-reduce operation."""
    return allreduce(tensor, group=group, async_op=False)


def reduce_scatter_along_dim(
    tensor: torch.Tensor,
    group: Optional[Any] = None,
    async_op: bool = False,
    dim_size=None,
    dim=0,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """Reduce-scatter along first dimension."""
    if group is None or not dist.is_initialized():
        return tensor, None

    world_size = get_distributed_world_size(group)
    if world_size == 1:
        return tensor, None
    if dim_size is None:
        dim_size = list(tensor.size())
    dim_size[dim] //= world_size
    output = torch.empty(dim_size, dtype=tensor.dtype, device=tensor.device)

    handle = dist._reduce_scatter_base(output, tensor.contiguous(), group=group, async_op=async_op)
    return output, handle


def gather_along_dim(
    tensor: torch.Tensor,
    group: Optional[Any] = None,
    dim=0,
    async_op: bool = False,
    quantizer: Optional[Any] = None,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """Gather along first dimension."""
    from .quantized_tensor import QuantizedTensorStorage

    if dim != 0:
        raise NotImplementedError("gather_along_dim currently supports dim=0 only")

    if quantizer is None and not isinstance(tensor, QuantizedTensorStorage):
        return _all_gather_dense_along_dim(tensor, group, dim=dim)

    if quantizer is None:
        quantizer = tensor._get_quantizer()

    world_size = get_distributed_world_size(group)
    if group is None or not dist.is_initialized() or world_size == 1:
        if not isinstance(tensor, QuantizedTensorStorage):
            tensor = quantizer(tensor)
        return tensor, None

    if quantizer is not None:
        from .tensor import (
            Float8CurrentScalingQuantizer,
            Float8Quantizer,
            MXFP8Quantizer,
        )
        from .tensor.storage.float8_tensor_storage import Float8TensorStorage
        from .tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

        if isinstance(tensor, Float8TensorStorage) or isinstance(
            quantizer,
            (Float8Quantizer, Float8CurrentScalingQuantizer),
        ):
            return _all_gather_fp8_along_dim(
                tensor,
                group,
                dim=dim,
                async_op=async_op,
                quantizer=quantizer,
            )
        if isinstance(tensor, MXFP8TensorStorage) or isinstance(
            quantizer,
            MXFP8Quantizer,
        ):
            if not isinstance(tensor, MXFP8TensorStorage) and not quantizer.is_quantizable(tensor):
                warnings.warn(
                    "Cannot quantize input tensor for MXFP8 all-gather. "
                    "Falling back to high-precision all-gather."
                )
            else:
                warnings.warn(
                    "MXFP8 all-gather on NPU falls back to high-precision all-gather before "
                    "requantization because HCCL does not support packed MXFP8 buffers.",
                    UserWarning,
                )
            dense = (
                tensor.dequantize(dtype=tensor._dtype)
                if isinstance(tensor, MXFP8TensorStorage)
                else tensor
            )
            return _dense_all_gather_then_quantize(
                dense,
                group,
                dim=dim,
                async_op=async_op,
                quantizer=quantizer,
            )
        if isinstance(tensor, QuantizedTensorStorage):
            warnings.warn(
                "Attempting to all-gather an unsupported quantized tensor. "
                "Falling back to high-precision all-gather."
            )
            return _dense_all_gather_then_quantize(
                tensor.dequantize(),
                group,
                dim=dim,
                async_op=async_op,
                quantizer=quantizer,
            )

    warnings.warn(
        "Attempting to all-gather with an unsupported quantizer. "
        "Falling back to high-precision all-gather."
    )
    return _dense_all_gather_then_quantize(
        tensor,
        group,
        dim=dim,
        async_op=async_op,
        quantizer=quantizer,
    )


reduce_scatter_along_first_dim = reduce_scatter_along_dim


def gather_along_first_dim(
    tensor: torch.Tensor,
    group: Optional[Any] = None,
    async_op: bool = False,
    quantizer: Optional[Any] = None,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """Gather along first dimension with NVIDIA-compatible argument order."""
    return gather_along_dim(
        tensor,
        group,
        dim=0,
        async_op=async_op,
        quantizer=quantizer,
    )


class _MultiHandle:
    """Small wrapper that waits on multiple async collective handles."""

    def __init__(self, handles: List[Optional[Any]]) -> None:
        self._handles = [handle for handle in handles if handle is not None]

    def wait(self) -> None:
        for handle in self._handles:
            handle.wait()


def _all_gather_dense_along_dim(
    tensor: torch.Tensor,
    group: Optional[Any],
    *,
    dim: int = 0,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """Original high-precision all-gather path for dense tensors."""
    if group is None or not dist.is_initialized():
        return tensor, None

    world_size = get_distributed_world_size(group)
    if world_size == 1:
        return tensor, None

    dim_size = list(tensor.size())
    dim_size[dim] = dim_size[dim] * world_size
    output = torch.empty(dim_size, dtype=tensor.dtype, device=tensor.device)
    handle = dist._all_gather_base(output, tensor.contiguous(), group=group, async_op=False)
    return output, handle


def _all_gather_raw_along_dim(
    tensor: torch.Tensor,
    group: Optional[Any],
    *,
    dim: int = 0,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """All-gather a regular tensor along ``dim``."""
    world_size = get_distributed_world_size(group)
    if group is None or not dist.is_initialized() or world_size == 1:
        return tensor, None

    dim_size = list(tensor.size())
    dim_size[dim] *= world_size
    output = torch.empty(dim_size, dtype=tensor.dtype, device=tensor.device)
    handle = dist._all_gather_base(
        output,
        tensor.contiguous(),
        group=group,
        async_op=async_op,
    )
    return output, handle


def _dense_all_gather_then_quantize(
    tensor: torch.Tensor,
    group: Optional[Any],
    *,
    dim: int,
    async_op: bool,
    quantizer: Any,
) -> Tuple[Any, Optional[Any]]:
    """Fallback path when low-precision layout cannot represent the gather."""
    out, handle = _all_gather_raw_along_dim(tensor, group, dim=dim, async_op=async_op)
    if handle is not None:
        handle.wait()
    return quantizer(out), None


def _all_gather_fp8_along_dim(
    tensor: torch.Tensor,
    group: Optional[Any],
    *,
    dim: int = 0,
    async_op: bool = False,
    quantizer: Optional[Any] = None,
) -> Tuple[Any, Optional[Any]]:
    """All-gather Float8 tensors through high precision before requantization."""
    from .tensor import Float8CurrentScalingQuantizer, Float8Quantizer
    from .tensor.storage.float8_tensor_storage import Float8TensorStorage

    if quantizer is not None and not isinstance(
        quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)
    ):
        raise TypeError(f"Expected Float8 quantizer, got {type(quantizer).__name__}")

    if isinstance(tensor, Float8TensorStorage):
        if quantizer is None:
            quantizer = tensor._get_quantizer()
        dense = tensor.dequantize(dtype=tensor._dtype)
    else:
        if quantizer is None:
            raise TypeError("Float8 all-gather requires a Float8 tensor or quantizer")
        dense = tensor

    warnings.warn(
        "Float8 all-gather on NPU falls back to high-precision all-gather before "
        "requantization because scale_inv may be shard-local.",
        UserWarning,
    )
    return _dense_all_gather_then_quantize(
        dense,
        group,
        dim=dim,
        async_op=async_op,
        quantizer=quantizer,
    )


def _all_gather_mxfp8_along_dim(
    tensor: torch.Tensor,
    group: Optional[Any],
    *,
    dim: int = 0,
    async_op: bool = False,
    quantizer: Any,
) -> Tuple[Any, Optional[Any]]:
    """All-gather MXFP8 data and scale buffers without dequantizing."""
    from .tensor import MXFP8Quantizer, MXFP8Tensor
    from .tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

    if not isinstance(quantizer, MXFP8Quantizer):
        raise TypeError(f"Expected MXFP8Quantizer, got {type(quantizer).__name__}")

    if dim != 0:
        warnings.warn(
            "Low-precision MXFP8 all-gather currently supports dim=0 only. "
            "Falling back to high-precision all-gather."
        )
        dense = tensor.dequantize() if isinstance(tensor, MXFP8TensorStorage) else tensor
        return _dense_all_gather_then_quantize(
            dense,
            group,
            dim=dim,
            async_op=async_op,
            quantizer=quantizer,
        )

    if not isinstance(tensor, MXFP8TensorStorage):
        if not quantizer.is_quantizable(tensor):
            warnings.warn(
                "Cannot quantize input tensor for MXFP8 all-gather. "
                "Falling back to high-precision all-gather."
            )
            return _dense_all_gather_then_quantize(
                tensor,
                group,
                dim=dim,
                async_op=async_op,
                quantizer=quantizer,
            )
        tensor = quantizer(tensor)
    elif (quantizer.rowwise_usage and tensor._rowwise_data is None) or (
        quantizer.columnwise_usage and tensor._columnwise_data is None
    ):
        warnings.warn(
            "Input and quantizer do not have matching MXFP8 usages. "
            "Dequantizing and requantizing before all-gather."
        )
        tensor = quantizer(tensor.dequantize(dtype=tensor._dtype))

    handles: List[Optional[Any]] = []
    local_shape = tuple(tensor.size())
    local_m_dim = math.prod(local_shape[:-1]) if len(local_shape) > 1 else 1
    k_dim = local_shape[-1] if len(local_shape) > 0 else 1
    from .constants import MXFP8_BLOCK_SCALING_SIZE

    local_m_blocks = math.ceil(local_m_dim / MXFP8_BLOCK_SCALING_SIZE)
    world_size = get_distributed_world_size(group)
    columnwise_scale_inv = tensor._columnwise_scale_inv
    if (
        columnwise_scale_inv is not None
        and columnwise_scale_inv.ndim == 2
        and columnwise_scale_inv.shape[0] == k_dim
        and columnwise_scale_inv.shape[1] >= local_m_blocks
    ):
        warnings.warn(
            "MXFP8 columnwise scale layout requires nonzero-dim scale all-gather. "
            "Falling back to high-precision all-gather."
        )
        return _dense_all_gather_then_quantize(
            tensor.dequantize(dtype=tensor._dtype),
            group,
            dim=0,
            async_op=async_op,
            quantizer=quantizer,
        )

    def _gather_optional(src: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if src is None:
            return None
        dst, handle = _all_gather_raw_along_dim(
            src,
            group,
            dim=0,
            async_op=async_op,
        )
        handles.append(handle)
        return dst

    def _gather_columnwise_scale(src: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if src is None:
            return None

        # NPU MXFP8 columnwise scales are packed as [M_block_pack, K, pack_dim].
        # The logical M-block dimension may live in the final packing lane, so
        # all-gathering raw dim0 would over-expand shapes such as
        # local (1, 32, 2) -> gathered (2, 32, 2), while full quantize expects
        # (1, 32, 2). Gather logical M blocks, then repack to the raw layout.
        if (
            src.ndim >= 3
            and src.shape[1] == k_dim
            and src.shape[-1] > 0
            and src.shape[0] * src.shape[-1] >= local_m_blocks
        ):
            pack_dim = src.shape[-1]
            logical_scale = src.permute(0, 2, 1).reshape(-1, k_dim)[:local_m_blocks].contiguous()
            gathered_logical, handle = _all_gather_raw_along_dim(
                logical_scale,
                group,
                dim=0,
                async_op=async_op,
            )
            if handle is not None:
                handle.wait()

            full_m_blocks = local_m_blocks * world_size
            packed_m_blocks = math.ceil(full_m_blocks / pack_dim)
            padded_m_blocks = packed_m_blocks * pack_dim
            if gathered_logical.shape[0] < padded_m_blocks:
                padding = torch.zeros(
                    (padded_m_blocks - gathered_logical.shape[0], k_dim),
                    dtype=gathered_logical.dtype,
                    device=gathered_logical.device,
                )
                gathered_logical = torch.cat((gathered_logical, padding), dim=0)
            return (
                gathered_logical[:padded_m_blocks]
                .reshape(packed_m_blocks, pack_dim, k_dim)
                .permute(0, 2, 1)
                .contiguous()
            )

        return _gather_optional(src)

    rowwise_data = _gather_optional(tensor._rowwise_data)
    rowwise_scale_inv = _gather_optional(tensor._rowwise_scale_inv)
    columnwise_data = _gather_optional(tensor._columnwise_data)
    columnwise_scale_inv = _gather_columnwise_scale(tensor._columnwise_scale_inv)

    out_shape = list(tensor.size())
    out_shape[0] *= world_size
    out = MXFP8Tensor(
        out_shape,
        tensor._dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=tensor._fp8_dtype,
        quantizer=quantizer,
        with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
        device=tensor.device,
    )
    return out, (_MultiHandle(handles) if async_op else None)


def split_tensor_into_1d_equal_chunks(
    tensor: torch.Tensor,
    tp_group: dist_group_type,
    new_buffer: bool = False,
) -> torch.Tensor:
    """Break a tensor into equal 1D chunks across tensor parallel ranks.

    Args:
        tensor: The tensor to split
        new_buffer: If True, returns a new Tensor. If False, returns a view.
        tp_group: Tensor parallel group (optional)

    Returns:
        Tensor or view with this rank's portion of the data
    """

    partition_size = torch.numel(tensor) // get_distributed_world_size(tp_group)
    start_index = partition_size * get_distributed_rank(tp_group)
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.npu.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor: torch.Tensor, tp_group: dist_group_type) -> torch.Tensor:
    """Opposite of above function, gather values from model parallel ranks."""
    numel_gathered = torch.numel(tensor) * get_distributed_world_size(tp_group)
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.npu.current_device(),
        requires_grad=False,
    )
    torch.distributed.all_gather_into_tensor(gathered, tensor, group=tp_group)
    return gathered


def _get_module_fsdp_state(module):
    """
    If module is an FSDP module, return its _FSDPState.
    Otherwise, return the _FSDPState of the closest parent FSDP module
    in the module hierarchy the module belongs to.
    """

    if hasattr(module, "_get_fsdp_state"):
        # this will return correct fsdp state if module itself is an fsdp module
        fsdp_state = module._get_fsdp_state()
    elif getattr(module, "_te_cached_parent_fsdp_state", None) is not None:
        # See if we have cached the parent fsdp state of the module
        fsdp_state = module._te_cached_parent_fsdp_state
    else:
        from torch.distributed._composable_state import _module_state_mapping

        # Otherwise get the fsdp state of lca of module in the module hierarchy
        min_nodes_in_parent = float("inf")
        closest_parent_fsdp_mod = None
        for fsdp_mod in _module_state_mapping.keys():
            all_submodules = list(fsdp_mod.modules())
            for submodule in all_submodules:
                if submodule is module:
                    if min_nodes_in_parent > len(all_submodules):
                        closest_parent_fsdp_mod = fsdp_mod
                        min_nodes_in_parent = len(all_submodules)
        if closest_parent_fsdp_mod is None:
            raise RuntimeError(
                "Module is not FSDP-wrapped and does not have any FSDP-wrapped parent modules."
            )
        fsdp_state = closest_parent_fsdp_mod._get_fsdp_state()
        # Cache the parent fsdp state of the module to avoid recomputing
        # the closest parent fsdp module.
        module._te_cached_parent_fsdp_state = fsdp_state
    return fsdp_state


def _fsdp_scatter_tensors(
    fsdp_group: dist_group_type,
    *tensors: torch.Tensor,
):
    shapes = []
    if fsdp_group is not None:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                targets = t.get_data_tensors() if isinstance(t, QuantizedTensor) else [t]
                for target in targets:
                    shapes.append(target.data.shape)
                    safely_set_viewless_tensor_data(
                        target,
                        split_tensor_into_1d_equal_chunks(
                            target.data, tp_group=fsdp_group, new_buffer=True
                        ),
                    )
            else:
                shapes.append(None)
    return shapes


def _fsdp_gather_tensors(
    fsdp_group: dist_group_type,
    shapes: List[Tuple[int, ...]],
    *tensors: torch.Tensor,
):
    if fsdp_group is not None:
        if len(shapes) != len(tensors):
            raise ValueError(
                "Number of tensors and tensor shapes must be equal, "
                f"but got {len(shapes)} shapes and {len(tensors)} tensors."
            )
        for s, t in zip(shapes, tensors):
            if isinstance(t, torch.Tensor):
                if s is None:
                    raise RuntimeError(
                        "Internal TE error: shape is None for a non-None tensor in "
                        "post_optimizer_step_fwd_amax_reduction. "
                        f"Tensor type: {type(t).__name__}, tensor shape: {t.shape}."
                    )
                targets = t.get_data_tensors() if isinstance(t, QuantizedTensor) else [t]
                for target in targets:
                    safely_set_viewless_tensor_data(
                        target,
                        gather_split_1d_tensor(target.data, fsdp_group).view(s),
                    )


def prepare_te_modules_for_fsdp(fsdp_root: torch.nn.Module) -> None:
    """
    Inject FSDP process gorup references into FSDP-wrapped TE modules in an FSDP-wrapped root
    module in order to scatter/gather the Fp8 weight copies at the same time FSDP scatters/gathers
    its `FlatParameters`.

    Parameters
    ----------
    fsdp_root : torch.nn.Module
               FSDP-wrapped root module that may contain FSDP-wrapped TE modules.
    """
    if not isinstance(fsdp_root, FSDP):
        raise TypeError(f"Root module must be FSDP-wrapped, but got {type(fsdp_root).__name__}.")

    # If the root module is a TE module, inject FSDP information into it
    if _is_te_module(fsdp_root.module):
        if hasattr(fsdp_root, "primary_weights_in_fp8"):
            if fsdp_root.primary_weights_in_fp8:
                raise RuntimeError(
                    "TE modules with primary weights in FP8 cannot be FSDP-wrapped. "
                    "Please initialize your model without the te.quantized_model_init(...) context."
                )
        root_state = _get_module_fsdp_state(fsdp_root)
        if root_state is None:
            raise RuntimeError(
                f"Root module ({type(fsdp_root.module).__name__}) does not have a valid "
                "_FSDPState. Ensure the module is properly wrapped with FSDP."
            )
        fsdp_root.module.fast_setattr("fsdp_group", root_state.process_group)

    # Iterate through all FSDP-wrapped submodules and inject FSDP information into TE modules
    fsdp_states, fsdp_modules = _get_fsdp_states_with_modules(fsdp_root)
    for state, fsdp_module in zip(fsdp_states, fsdp_modules):
        if _is_te_module(fsdp_module.module):
            if hasattr(fsdp_module.module, "primary_weights_in_fp8"):
                if fsdp_module.module.primary_weights_in_fp8:
                    raise RuntimeError(
                        f"TE module '{type(fsdp_module.module).__name__}' with primary weights "
                        "in FP8 cannot be FSDP-wrapped. Please initialize your model without "
                        "the te.quantized_model_init(...) context."
                    )
            fsdp_module.module.fast_setattr("fsdp_group", state.process_group)


class FullyShardedDataParallel(FSDP):
    """
    Transformer Engine wrapper around `torch.distributed.fsdp.FullyShardedDataParallel` that
    extracts necessary information out of the FSDP wrap for TE modules to scatter their
    activation tensors after each forward pass and gather them before the backward pass.
    """

    def __init__(self, module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        prepare_te_modules_for_fsdp(self)


class CudaRNGStatesTracker:
    """
    For model parallelism, multiple RNG states need to simultaneously exist in order
    to execute operations in or out of the model parallel region. This class keeps
    track of the various RNG states and provides utility methods to maintain them and
    execute parts of the model under a given RNG setting. Using the :meth:`add` method, a
    cuda rng state is initialized based on the input ``seed`` and is assigned to ``name``.
    Later, by forking the rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """
        Set to the initial state (no tracker).
        """
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self) -> Dict[str, torch.Tensor]:
        """
        Get rng states. Copy the dictionary so we have direct pointers
        to the states, not just a pointer to the dictionary.
        """
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states: Dict[str, torch.Tensor]) -> None:
        """
        Set the rng states. For efficiency purposes, we do not
        check the size of seed for compatibility.

        Parameters
        ----------
        states : Dict[str, torch.Tensor]
               A mapping from string names to RNG states.
        """
        self.states_ = states
        # Update global states.
        set_all_rng_states(self.states_)

    def add(self, name: str, seed: int) -> None:
        """
        Adds a new RNG state.

        Parameters
        ----------
        name : str
             string identifier for the RNG state.
        seed : int
             PyTorch seed for the RNG state.
        """
        # Check seed is not already used.
        if seed in self.seeds_:
            raise RuntimeError(f"seed {seed} already exists")
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise RuntimeError(f"cuda rng state {name} already exists")

        if graph_safe_rng_available():
            new_state = _get_cuda_rng_state(clone=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
            # Update global states.
            set_all_rng_states(self.states_)
        else:
            # Get the current rng state.
            orig_rng_state = _get_cuda_rng_state()
            # Set the new state and store it.
            torch.cuda.manual_seed(seed)
            self.states_[name] = _get_cuda_rng_state(clone=True)
            # Reset rng state to what it was.
            _set_cuda_rng_state(orig_rng_state)
            # Update global states.
            set_all_rng_states(self.states_)

    @contextmanager
    def fork(self, name: str = "model-parallel-rng"):
        """
        Fork the cuda rng state, perform operations, and exit with
        the original state.

        Parameters
        ----------
        name : str
             string identifier for the RNG state.
        """
        # Check if we have added the state
        if name not in self.states_:
            raise KeyError(f"cuda rng state {name} is not added")
        # Get the reference to current rng state.
        orig_cuda_rng_state = _get_cuda_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # this is redundant with graph-safe API
            if not graph_safe_rng_available():
                self.states_[name] = _get_cuda_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


NpuRNGStatesTracker = CudaRNGStatesTracker


def get_hccl_comm_name(group):
    rank = get_distributed_rank(group)
    if torch.__version__ > "2.0":
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
    else:
        hcomm_name = group.get_hccl_comm_name(rank)

    return hcomm_name


class DummyHandle:
    @classmethod
    def wait(cls, *args, **kwargs):
        pass
