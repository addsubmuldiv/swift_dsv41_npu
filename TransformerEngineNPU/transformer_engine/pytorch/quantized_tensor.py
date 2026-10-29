# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

import abc
import math
import warnings
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    MutableSequence,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch.utils._pytree import tree_map

from .constants import TensorUsage

if TYPE_CHECKING:
    from transformer_engine.common.recipe import Recipe
    from .tensor.grouped_tensor import GroupedTensor


T = TypeVar("T")


def _stride_from_shape(shape: Iterable[int]) -> list[int]:
    """Calculate contiguous stride from shape."""
    dims = list(shape)
    if len(dims) == 0:
        return []
    rstride = [1]
    for d in reversed(dims[1:]):
        rstride.append(rstride[-1] * d)
    return list(reversed(rstride))


def transpose_quantized_tensor(data, scale):
    if data is None and scale is None:
        return None, None
    if data is None or scale is None:
        raise RuntimeError("Cannot transpose quantized storage with missing data or scale")
    data = data.transpose(-1, -2) if data.ndim > 2 else data.t()
    scale = scale.transpose(-1, -2) if scale.ndim > 2 else scale.t()
    return data, scale


def _dequantize_grouped_by_per_group_scale(
    grouped: "GroupedTensor",
    dtype: torch.dtype,
    *,
    columnwise: bool = False,
) -> torch.Tensor:
    """Dequantize grouped buffers carrying one scalar scale per group."""
    data = grouped.columnwise_data if columnwise else grouped.rowwise_data
    scale_inv = grouped.columnwise_scale_inv if columnwise else grouped.rowwise_scale_inv
    num_tensors = grouped.num_tensors

    float_data = data.to(torch.float32)
    float_scale = scale_inv.to(torch.float32)
    if float_scale.numel() == 1:
        result = float_data * float_scale.reshape(())
    elif float_scale.numel() == num_tensors:
        if float_data.numel() % num_tensors != 0:
            raise ValueError(
                "GroupedTensor packed data cannot be divided into per-group Float8 data"
            )
        result = (
            float_data.reshape(num_tensors, -1) * float_scale.reshape(num_tensors, 1)
        ).reshape(grouped.logical_shape)
    else:
        raise ValueError(
            "GroupedTensor Float8 scale_inv must be scalar or contain one value per group"
        )
    return result.to(dtype=dtype)


def transpose_mx_data(
    data: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transpose MXFP8 data and its E8M0 scale layout together."""
    if data is None or scale is None:
        raise RuntimeError("Cannot transpose MX format storage with missing data or scale")
    data = data.transpose(-1, -2) if data.ndim > 2 else data.t()
    scale = scale.transpose(-3, -2) if scale.ndim > 2 else scale.t()
    return data, scale


class _QuantizeFunc(torch.autograd.Function):
    """Quantize tensor"""

    @staticmethod
    def forward(
        _ctx: Optional[torch.autograd.function.FunctionCtx],  # unused
        tensor: torch.Tensor,
        quantize_impl: Callable,
    ) -> "QuantizedTensor":
        # pylint: disable=missing-function-docstring
        return quantize_impl(tensor)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        # Assume that we want gradients in full precision
        return grad, None


class QuantizedTensor(torch.Tensor):
    _requires_grad: bool

    def __new__(
        cls,
        shape: Iterable[int],
        dtype: torch.dtype,
        *,
        fake_dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        device: Optional[torch.device] = None,
        stride: Optional[Iterable[int]] = None,
    ):
        shape = tuple(shape)
        stride = _stride_from_shape(shape) if stride is None else tuple(stride)
        if device is None:
            if hasattr(torch, "npu"):
                device = torch.device("npu", torch.npu.current_device())
            else:
                device = torch.device("cpu")
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=stride,
            storage_offset=0,
            dtype=dtype,
            layout=torch.strided,
            requires_grad=requires_grad,
            device=device,
        )
        instance._dtype = dtype
        instance._shape = shape
        instance._requires_grad = requires_grad
        return instance

    @property
    def dtype(self) -> torch.dtype:
        """
        Return the high precision data type of the tensor
        Attribute access of custom tensors goes through an
        expensive Pyobject lookup. Since dtype for a tensor is never
        change after creation, we cache it in a member variable and return
        """
        # Lazy initialization for tensors created via alternate paths
        if not hasattr(self, "_dtype"):
            # pylint: disable=unnecessary-dunder-call
            self._dtype = torch._C.TensorBase.dtype.__get__(self, type(self))
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        """Set dtype property"""
        self._dtype = value

    @property
    def origin_shape(self) -> "MutableSequence[int]":
        if not hasattr(self, "_shape"):
            self._shape = tuple(self.shape)
        return self._shape

    @origin_shape.setter
    def origin_shape(self, value: "MutableSequence[int]") -> None:
        """Set dtype property"""
        self._shape = value

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA device."""
        return False

    @property
    def requires_grad(self) -> bool:
        """Return whether or not the tensor requires gradient."""
        # pylint: disable=unnecessary-dunder-call
        base_requires_grad = torch._C.TensorBase.requires_grad.__get__(self, type(self))
        if (not hasattr(self, "_requires_grad")) or self._requires_grad != base_requires_grad:
            self._requires_grad = base_requires_grad
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.requires_grad_(value)

    def requires_grad_(self, requires_grad: bool = True) -> "QuantizedTensor":
        """Cache requires_grad and update the wrapper subclass state."""
        self._requires_grad = requires_grad
        super().requires_grad_(requires_grad)
        return self

    def _get_data(self) -> torch.Tensor:
        """Get tensor data property."""
        return super().data

    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data and keep cached dtype in sync."""
        # pylint: disable=unnecessary-dunder-call
        super(QuantizedTensor, type(self)).data.__set__(self, tensor)
        self._dtype = tensor.dtype

    data = property(_get_data, _set_data)

    def can_reuse_quantized(self, quantizer: "Quantizer") -> bool:
        """Whether an input can retain its representation under this quantizer.

        Quantized model weights retain their existing format by default. Dense
        adapters may require processing again; cached workspace updates are
        handled separately through quantize_().
        """
        return True

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert quantized data to standard PyTorch tensor"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement dequantize function"
        )

    def quantize_(self, tensor: torch.Tensor) -> "QuantizedTensor":
        """Update quantized data in-place"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement quantize_ function"
        )

    def expand_as(self, other: torch.Tensor) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if other is self:
            from .tensor._quantization_helpers import _IdentityFunc

            return _IdentityFunc.apply(self)
        return super().expand_as(other)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """Dispatch support for wrapper-subclass quantized tensors."""
        if kwargs is None:
            kwargs = {}

        if func == torch.ops.aten.detach.default:
            return args[0].detach()

        if func == torch.ops.aten.clone.default:
            return args[0].clone()

        if func == torch.ops.aten._to_copy.default:
            tensor = args[0]
            target_dtype = kwargs.get("dtype", None)
            if target_dtype is None or target_dtype == tensor.dtype:
                target_device = kwargs.get("device", tensor.device) or tensor.device
                target_device = torch.device(target_device)
                non_blocking = kwargs.get("non_blocking", False)
                pin_memory = kwargs.get("pin_memory", False)
                new_metadata = {}
                for key, value in tensor.get_metadata().items():
                    if isinstance(value, torch.Tensor):
                        value = value.to(device=target_device, non_blocking=non_blocking)
                        if pin_memory and target_device.type == "cpu":
                            value = value.pin_memory()
                    new_metadata[key] = value
                new_metadata.update(
                    {
                        "device": target_device,
                        "dtype": tensor.dtype,
                        "shape": tensor.shape,
                        "requires_grad": tensor.requires_grad,
                    }
                )
                return type(tensor)(**new_metadata)

        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            if (
                isinstance(dst, QuantizedTensor)
                and isinstance(src, QuantizedTensor)
                and type(getattr(dst, "_quantizer", None)) is type(getattr(src, "_quantizer", None))
                and hasattr(dst, "get_usages")
                and hasattr(src, "get_usages")
                and dst.get_usages() == src.get_usages()
            ):
                dst.copy_from_storage(src)
                dst.origin_shape = src.origin_shape
                dst.dtype = src.dtype
                return dst

            if isinstance(dst, QuantizedTensor):
                dst.quantize_(src)
                return dst

            if isinstance(src, QuantizedTensor):
                dtype = dst.dtype
                if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    dtype = torch.float32
                src = src.dequantize(dtype=dtype)
            dst.copy_(src, *args[2:], **kwargs)
            return dst

        if func == torch.ops.aten.view.default:
            tensor = args[0]
            if isinstance(tensor, QuantizedTensor):
                return tensor.view(*args[1])

        if func == torch.ops.aten.new_empty.default:
            tensor = args[0]
            size = args[1]
            dtype = kwargs.get("dtype", tensor.dtype)
            device = kwargs.get("device", tensor.device)
            pin_memory = kwargs.get("pin_memory", False)
            if getattr(tensor, "_quantizer", None) is None:
                raise RuntimeError(
                    f"{type(tensor).__name__} does not have a quantizer; cannot create new_empty"
                )
            return tensor._quantizer.make_empty(
                shape=torch.Size(size),
                dtype=dtype,
                device=device,
                requires_grad=tensor.requires_grad,
                pin_memory=pin_memory,
            )

        if func == torch.ops.aten.empty_like.default:
            tensor = args[0]
            device = kwargs.get("device", tensor.device)
            requires_grad = kwargs.get("requires_grad", tensor.requires_grad)
            pin_memory = kwargs.get("pin_memory", False)
            if getattr(tensor, "_quantizer", None) is None:
                raise RuntimeError(
                    f"{type(tensor).__name__} does not have a quantizer; cannot create empty_like"
                )
            usage = tensor.get_usages() if hasattr(tensor, "get_usages") else None
            quantizer_usage = (
                tensor._quantizer.get_usages() if hasattr(tensor._quantizer, "get_usages") else None
            )
            if usage is not None and hasattr(tensor._quantizer, "set_usage"):
                tensor._quantizer.set_usage(**usage)
            try:
                return tensor._quantizer.make_empty(
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    device=device,
                    requires_grad=requires_grad,
                    pin_memory=pin_memory,
                )
            finally:
                if quantizer_usage is not None and hasattr(tensor._quantizer, "set_usage"):
                    tensor._quantizer.set_usage(**quantizer_usage)

        if func == torch.ops.aten.numel.default:
            tensor = args[0]
            return math.prod(tensor.size())

        if func == torch.ops.aten.is_pinned.default:
            tensor = args[0]
            if hasattr(tensor, "get_data_tensors"):
                data_tensors = tensor.get_data_tensors()
                if not isinstance(data_tensors, tuple):
                    data_tensors = (data_tensors,)
                for item in data_tensors:
                    if item is not None:
                        return item.is_pinned()
            return False

        def maybe_unwrap(arg):
            if isinstance(arg, QuantizedTensor):
                return arg.dequantize()
            return arg

        def maybe_update_inplace(arg, new_arg, schema_arg):
            if (
                isinstance(arg, QuantizedTensor)
                and isinstance(new_arg, torch.Tensor)
                and hasattr(schema_arg, "alias_info")
                and hasattr(schema_arg.alias_info, "is_write")
                and schema_arg.alias_info.is_write
            ):
                arg.quantize_(new_arg)
            elif isinstance(arg, list) and isinstance(new_arg, list):
                for a, na in zip(arg, new_arg):
                    maybe_update_inplace(a, na, schema_arg)

        if func._schema.is_mutable:
            unwrapped_args = tree_map(maybe_unwrap, args)
            unwrapped_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            super().__torch_dispatch__(func, types, unwrapped_args, unwrapped_kwargs)
            for arg, new_arg, schema_arg in zip(args, unwrapped_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(
                kwargs, unwrapped_kwargs, schema_args[args_len:]
            ):
                assert kwarg == new_kwarg == schema_arg.name
                maybe_update_inplace(kwargs[kwarg], unwrapped_kwargs[new_kwarg], schema_arg)
            return None

        unwrapped_args = tree_map(maybe_unwrap, args)
        unwrapped_kwargs = tree_map(maybe_unwrap, kwargs)
        return super().__torch_dispatch__(func, types, unwrapped_args, unwrapped_kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """Get keyword arguments for quantized tensor constructor

        Contains metadata so that the new quantized tensor has the
        same underlying quantized data.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement get_metadata function"
        )

    @classmethod
    def make_like(
        cls,
        tensor: "QuantizedTensor",
        *,
        shape: Optional[Iterable[int]] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
    ) -> "QuantizedTensor":
        """Create new quantized tensor

        By default, new tensor has the same attributes and underlying
        data. This function is intended to create view of tensors.

        """
        shape = shape if shape is not None else tensor.shape
        dtype = dtype if dtype is not None else tensor.dtype
        kwargs = tensor.get_metadata()
        kwargs["fake_dtype"] = dtype
        return cls(shape=shape, dtype=dtype, requires_grad=requires_grad, **kwargs)

    def allgather_matmul(
        self,
        B: "QuantizedTensor",
        bias,
        world_size,
        group,
        usage: TensorUsage,
        usage_b: TensorUsage,
        out_dtype: torch.dtype,
        return_gathered: bool = True,
    ): ...

    def matmul_reduce_scatter(
        self,
        B: "QuantizedTensor",
        bias,
        world_size,
        group,
        usage: TensorUsage,
        usage_b: TensorUsage,
        out_dtype: torch.dtype,
    ): ...

    def matmul(
        self,
        B: "QuantizedTensor",
        usage: TensorUsage,
        usage_b: TensorUsage,
        out_dtype: torch.dtype,
    ): ...

    def matmul_add(
        self,
        main_grad: torch.Tensor,
        B: "QuantizedTensor",
        usage: TensorUsage,
        usage_b: TensorUsage,
        out_dtype: torch.dtype,
    ): ...

    def float(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        return self.dequantize(dtype=torch.float16)

    def cpu(self, memory_format=torch.preserve_format) -> "QuantizedTensor":
        """Move tensor to CPU while preserving the QuantizedTensor type."""
        return self.to(device=torch.device("cpu"), memory_format=memory_format)

    def untyped_storage(self) -> torch.UntypedStorage:
        """Return an empty storage because wrapper subclasses have no backing storage."""
        return torch.UntypedStorage(0, device=self.device)

    def cuda(self, *args, **kwargs) -> torch.Tensor:
        """Dequantize and move to CUDA."""
        return self.dequantize().cuda(*args, **kwargs)


class QuantizedTensorStorage:
    r"""Base class for all TensorStorage classes.

    This class (and its subclasses) are optimization for when
    the full QuantizedTensor is not needed (when it is fully
    contained inside torch.autograd function and not visible to
    PyTorch's autograd).

    When creating a new tensor type X one should create both
    XTensorStorage class inheriting from QuantizedTensorStorage and
    XTensor inheriting from XTensorStorage and QuantizedTensor.
    XTensorStorage should contain all data members needed to
    implement the functionality of the tensor, while
    XTensor should only implement the functionality needed
    to behave like regular torch.Tensor (like __torch_dispatch__).
    """

    _quantizer: Optional["Quantizer"]
    _rowwise_usage = True
    _columnwise_usage = True

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        r"""
        Generate or remove quantized data based on provided usage.

        Parameters
        ----------
        rowwise_usage : Optional[bool[, default = None
                        Whether to create or keep the data needed for using the tensor
                        in rowwise fashion (e.g. as B argument in TN GEMM). Leaving it as `None`
                        preserves the original value in the tensor.
        columnwise_usage : Optional[bool], default = None
                           Whether to create or keep the data needed for using the tensor
                           in columnwise fashion (e.g. as A argument in TN GEMM). Leaving it as
                           `None` preserves the original value in the tensor.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_usage function"
        )

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], "QuantizedTensorStorage"]:
        """Prepare the tensor base for saving for backward"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement prepare_for_saving function"
        )

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement restore_from_saved function"
        )

    def _get_quantizer(self) -> "Quantizer":
        """Get builder for quantized tensor

        Quantizer can be used for in-place operations.

        """
        if self._quantizer is not None:
            return self._quantizer
        return self._build_default_quantizer()

    def _build_default_quantizer(self) -> "Quantizer":
        """Build default quantizer for the tensor"""
        raise ValueError(
            f"{self.__class__.__name__} has no quantizer "
            "and no default quantizer is available defined in the subclass."
        )

    def quantize_(
        self, tensor: torch.Tensor, *, noop_flag: Optional[torch.Tensor] = None
    ) -> QuantizedTensor:
        """Quantize tensor in-place"""
        self._get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def grouped_quantize_(
        self, tensor, group_list, *, noop_flag: Optional[torch.Tensor] = None
    ) -> QuantizedTensor:
        self._get_quantizer().update_grouped_quantized(
            tensor, self, group_list, noop_flag=noop_flag
        )
        return self

    def update_quantizer(self, quantizer: "Quantizer"):
        """Update quantizer for the tensor"""
        if self._quantizer is None:
            raise RuntimeError("To be updated, quantizer must be set")
        if self._quantizer is not quantizer:
            warnings.warn("Quantizer is being updated, this may affect model behavior")
            self._quantizer = quantizer

    def copy_from_storage(self, src: T) -> T:
        """Copy data from another QuantizedTensorStorage."""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement copy_from_storage function"
        )

    def get_data(self, usage: TensorUsage):
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement get_data function"
        )

    def clear_wise(self, rowwise=False, colwise=False):
        pass


def prepare_for_saving(
    *tensors: Union[torch.Tensor, QuantizedTensorStorage],
) -> Tuple[
    list[Optional[Union[torch.Tensor, torch.nn.Parameter]]],
    list[Optional[QuantizedTensorStorage]],
]:
    """Prepare tensors for saving. Needed because save_for_backward accepts only
    torch.Tensor/torch.nn.Parameter types, while we want to be able to save
    the internal TensorStorage types too.
    """

    tensor_list, tensor_objects_list = [], []
    for tensor in tensors:
        if tensor is None or isinstance(tensor, torch.Tensor):
            tensor_list.append(tensor)
            tensor_objects_list.append(None)
        else:
            t, t_obj = tensor.prepare_for_saving()
            tensor_list.extend(t)
            tensor_objects_list.append(t_obj)

    return tensor_list, tensor_objects_list


def restore_from_saved(
    tensors: list[Optional[Union[torch.Tensor, QuantizedTensorStorage]]],
    saved_tensors: list[Optional[Union[torch.Tensor, torch.nn.Parameter]]],
    return_saved_tensors: bool = False,
) -> Union[
    list[Union[torch.Tensor, QuantizedTensorStorage, None]],
    tuple[
        list[Union[torch.Tensor, QuantizedTensorStorage, None]],
        list[Optional[torch.Tensor]],
    ],
]:
    """Recombine the tensor data and metadata during backward pass. Note: please use
    `restore_from_func_ctx` instead if you are restoring tensors from a function context to make
    sure tensor_objects is detached and its memory can be freed
    """
    tensor_objects = []
    for tensor in tensors:
        if tensor is None or isinstance(tensor, torch.Tensor):
            tensor_objects.append(saved_tensors[0])
            saved_tensors = saved_tensors[1:]
        else:
            saved_tensors = tensor.restore_from_saved(saved_tensors)
            tensor_objects.append(tensor)

    if return_saved_tensors:
        return tensor_objects, saved_tensors
    return tensor_objects


def restore_from_func_ctx(
    ctx: torch.autograd.function.FunctionCtx, return_saved_tensors=False
) -> (
    list[Optional[torch.Tensor | QuantizedTensorStorage]]
    | tuple[
        list[Optional[torch.Tensor | QuantizedTensorStorage]],
        list[Optional[torch.Tensor]],
    ]
):
    """Recombine the tensor data and metadata during backward pass and delete tensor objects
    attached to function context.
    """
    if not hasattr(ctx, "tensor_objects") or ctx.tensor_objects is None:
        raise AttributeError("ctx must have .tensor_objects to restore saved tensors")
    out = restore_from_saved(
        ctx.tensor_objects, ctx.saved_tensors, return_saved_tensors=return_saved_tensors
    )
    # Delete the references to tensor objects once they've been consumed by the `restore_from_saved` method to construct back the actual tensors.
    ctx.tensor_objects = None
    return out


class Quantizer(abc.ABC):
    """Builder class for quantized tensors.

    This class is typically used to convert a high-precision tensor
    (e.g. in FP32 or BF16) into a quantized tensor (e.g. in FP8).

    """

    """Whether to construct quantized tensors with "row-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A *
    B^T (used in linear forward). Tensor Cores prefer "TN GEMMs" (in
    Fortran-style column-major order), so A and B should be in
    row-major order.

    """
    rowwise_usage: bool

    """Whether to construct quantized tensors with "column-wise usage"

    Hand-wave explanation: Consider the matrix multiplication C = A^T
    * B (used in linear backward wgrad). Tensor Cores prefer "TN
    GEMMs" (in Fortran-style column-major order), so A and B should be
    in column-major order.

    """
    columnwise_usage: bool

    """Whether to instantiates tensor for purely internal usage

    Internal tensors are storage classes with minimal logic. They have
    less overhead than PyTorch tensor sub-classes, but are not
    compatible with PyTorch's autograd infrastructure nor PyTorch
    operations.

    """
    internal: bool

    """Whether to solely optimize for matrix multiplication

    The resulting quantized tensors are not guaranteed to support any
    operation other than matrix multiplication. Use with care since
    this is likely to break communication, checkpointing, and many
    other features.

    """
    optimize_for_gemm: bool

    dtype: torch.dtype

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        self.rowwise_usage = rowwise
        self.columnwise_usage = columnwise
        self.internal = False
        self.optimize_for_gemm = False
        self.columnwise_use_group_quant: bool = False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rowwise_usage={self.rowwise_usage}, "
            f"columnwise_usage={self.columnwise_usage}, "
            f"internal={self.internal}, "
            f"columnwise_use_group_quant={self.columnwise_use_group_quant}, "
            ")"
        )

    def can_reuse(self, tensor: torch.Tensor) -> bool:
        """Return whether an input can bypass quantization."""
        return isinstance(tensor, QuantizedTensor) and tensor.can_reuse_quantized(self)

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Quantize tensor in-place"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_quantized"
        )

    def update_grouped_quantized(self, src, dst, group_list, *, noop_flag=None):
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_grouped_quantized"
        )

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        out: Optional[QuantizedTensor] = None,
        dtype: Optional[torch.dtype] = None,  # pylint: disable=unused-argument # used by override
    ) -> QuantizedTensor:
        """Quantize tensor"""
        if out is not None:
            return self.update_quantized(tensor, out)
        if (not self.internal) and torch.is_grad_enabled():
            return _QuantizeFunc.apply(tensor, self.quantize_impl)
        return _QuantizeFunc.forward(None, tensor, self.quantize_impl)

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement quantize_impl function"
        )

    def multi_quantize(self, list_of_tensors):
        """Quantize multiple tensors"""
        list_of_output_tensors = []
        for tensor in list_of_tensors:
            list_of_output_tensors.append(self.quantize(tensor))
        return list_of_output_tensors

    def grouped_quantize(
        self,
        tensor: torch.Tensor,
        group_list: torch.Tensor,
        *,
        out: Optional[QuantizedTensor] = None,
        dtype: Optional[torch.dtype] = None,  # pylint: disable=unused-argument # used by override
    ) -> QuantizedTensor:
        """Quantize tensor"""
        if out is not None:
            return self.update_grouped_quantized(tensor, out, group_list)
        grouped_quantize_impl = partial(self.grouped_quantize_impl, group_list)
        if (not self.internal) and torch.is_grad_enabled():
            return _QuantizeFunc.apply(tensor, grouped_quantize_impl)
        return _QuantizeFunc.forward(None, tensor, grouped_quantize_impl)

    def grouped_quantize_impl(self, group_list, tensor):
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement grouped_quantize_impl function"
        )

    def dequantize_grouped_tensor(
        self,
        tensor: "GroupedTensor",
        *,
        dtype: Optional[torch.dtype] = None,
        columnwise: bool = False,
    ) -> torch.Tensor:
        """Dequantize a packed GroupedTensor produced by this quantizer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} class cannot dequantize GroupedTensor"
        )

    def __call__(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor"""
        return self.quantize(tensor)

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> QuantizedTensor:
        """Construct quantized tensor with uninitialized data"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement make_empty function, "
            "required for construction of unintialized quantized tensor"
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        """Calibrate quantizer state

        Updates quantization state as if quantizing a tensor, but
        without actually performing the quantization.

        """

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        """Set how the quantized tensor is expected to be used

        See documentation for `rowwise_usage` and `columnwise_usage`
        variables.

        """
        if rowwise is not None:
            self.rowwise_usage = rowwise
        if columnwise is not None:
            self.columnwise_usage = columnwise

    def _get_compatible_recipe(self) -> "Recipe":
        """Returns recipe class that is compatible with this quantizer"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement _get_compatible_recipe"
        )

    def supports_only_rowwise_all_gather(self) -> bool:
        """Returns True if the quantizer supports only rowwise all-gather"""
        return False

    def is_quantizable(self, inp: torch.Tensor) -> bool:  # pylint: disable=unused-argument
        """Whether tensor supports quantized all-gather

        Consider a less misleading function name.

        """
        return True

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the quantizer"""
        return {
            "rowwise": self.rowwise_usage,
            "columnwise": self.columnwise_usage,
        }

    def transpose(self, data, scale):
        return transpose_quantized_tensor(data, scale)


def _make_module_cast_func(dtype):
    """Make module cast function that can handle QuantizedTensor"""
    cast_func_name = {
        torch.float32: "float",
        torch.float16: "half",
        torch.bfloat16: "bfloat16",
    }[dtype]

    def tensor_cast_func(tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor dtype"""
        if isinstance(tensor, QuantizedTensor):
            return tensor.__class__.make_like(tensor, dtype=dtype)
        if tensor.is_floating_point():
            return getattr(tensor, cast_func_name)()
        return tensor

    def module_cast_func(self: torch.nn.Module) -> torch.nn.Module:
        """Cast module dtype"""
        return self._apply(tensor_cast_func)

    return module_cast_func


# Monkey-patch module cast functions to handle QuantizedTensor
torch.nn.Module.float = _make_module_cast_func(torch.float32)
torch.nn.Module.half = _make_module_cast_func(torch.float16)
torch.nn.Module.bfloat16 = _make_module_cast_func(torch.bfloat16)
