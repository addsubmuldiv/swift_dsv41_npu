# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
from typing import List, Optional, Tuple, Union

import torch

from transformer_engine.pytorch.constants import NPUVersion, TensorUsage, USAGE_WITH_TRANS
from transformer_engine.pytorch.quantized_tensor import Quantizer
from transformer_engine.pytorch.utils import check_npu_version


def _stride_from_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate contiguous strides for a wrapper shape."""
    if len(shape) == 0:
        return ()
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return tuple(stride)


class _GroupedIdentityFunc(torch.autograd.Function):
    """Identity node used by DDP/MCore hook registration paths."""

    @staticmethod
    def forward(ctx, tensor: "GroupedTensor") -> "GroupedTensor":
        ctx.input_dtype = tensor.dtype
        return tensor.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if grad_output.dtype != ctx.input_dtype:
            grad_output = grad_output.to(ctx.input_dtype)
        return grad_output


BANNED_SHAPE_OPS = {
    # A GroupedTensor is a metadata wrapper around packed per-group buffers.
    # Shape/view ops on the wrapper could desynchronize the logical wrapper
    # shape from the underlying packed storage, so keep them explicit.
    torch.ops.aten.reshape.default,
    torch.ops.aten._reshape_alias.default,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.unflatten.int,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.movedim.int,
    torch.ops.aten.t.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.narrow.default,
    torch.ops.aten.select.int,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.chunk.default,
    torch.ops.aten.cat.default,
    torch.ops.aten.stack.default,
}


ROWWISE_IN_PLACE_OPS = {
    # Initializers operate on the dense packed rowwise buffer and keep the
    # GroupedTensor wrapper/metadata unchanged.
    torch.ops.aten.fill_.Scalar,
    torch.ops.aten.fill_.Tensor,
    torch.ops.aten.normal_.default,
    torch.ops.aten.uniform_.default,
    torch.ops.aten.zero_.default,
}

ROWWISE_OPTIMIZER_IN_PLACE_OPS = {
    # Optimizers may see GroupedTensor parameters as normal tensors. These
    # in-place ops are forwarded to rowwise_data so parameter updates hit the
    # actual storage used by grouped GEMM.
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.add_.Scalar,
    torch.ops.aten.addcdiv_.default,
    torch.ops.aten.addcmul_.default,
    torch.ops.aten.lerp_.Scalar,
    torch.ops.aten.lerp_.Tensor,
    torch.ops.aten.mul_.Tensor,
    torch.ops.aten.mul_.Scalar,
}

ROWWISE_OPTIMIZER_OUT_OF_PLACE_OPS = {
    # Adam-style update math can produce temporary tensors. Return another
    # GroupedTensor wrapper so subsequent optimizer ops still preserve grouped
    # metadata.
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
    torch.ops.aten.sqrt.default,
}


class GroupedTensor(torch.Tensor):
    """Minimal tensor wrapper for grouped tensor metadata on NPU grouped GEMM paths."""

    logical_shape: Tuple[int, ...]
    fake_dtype: torch.dtype
    num_tensors: int
    quantizer: Quantizer
    tensor_shapes: Optional[List[Tuple[int, ...]]]
    quantized_tensors: Optional[List[object]]
    columnwise_data: Optional[torch.Tensor]
    columnwise_scale_inv: Optional[torch.Tensor]
    rowwise_data: Optional[torch.Tensor]
    rowwise_scale_inv: Optional[torch.Tensor]
    scale_inv: Optional[torch.Tensor]

    def __new__(
        cls,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_tensors: int,
        quantizer,
        data: Optional[torch.Tensor],
        scale_inv: Optional[torch.Tensor],
        *,
        columnwise_data: Optional[torch.Tensor] = None,
        columnwise_scale_inv: Optional[torch.Tensor] = None,
        shapes: Optional[List[Tuple[int, ...]]] = None,
        requires_grad: bool = False,
        stride: Optional[Tuple[int, ...]] = None,
    ):
        wrapper_shape = cls._wrapper_shape(shape, num_tensors, shapes)
        device = cls._infer_device(
            data,
            columnwise_data,
            scale_inv,
            columnwise_scale_inv,
        )
        strides = _stride_from_shape(tuple(wrapper_shape)) if stride is None else tuple(stride)
        instance = torch.Tensor._make_wrapper_subclass(
            cls,
            tuple(wrapper_shape),
            strides=strides,
            storage_offset=0,
            dtype=dtype,
            layout=torch.strided,
            requires_grad=requires_grad,
            device=device,
        )
        cls._initialize_grouped_fields(
            instance,
            logical_shape=tuple(shape),
            dtype=dtype,
            num_tensors=num_tensors,
            quantizer=quantizer,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            shapes=shapes,
        )
        return instance

    @staticmethod
    def make_grouped_tensor_with_shapes(
        num_tensors: int,
        shapes: List[Tuple[int, ...]],
        quantizer=None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "GroupedTensor":
        """Allocate grouped rowwise storage for uniform member shapes."""
        if num_tensors <= 0:
            raise ValueError(f"num_tensors must be positive, got {num_tensors}")
        if len(shapes) != num_tensors:
            raise ValueError(f"Expected {num_tensors} grouped tensor shapes, got {len(shapes)}")
        if not shapes:
            raise ValueError("shapes must not be empty")
        member_shape = tuple(shapes[0])
        if len(member_shape) not in (1, 2):
            raise ValueError(
                "GroupedTensor member shapes must be 1D or 2D, "
                f"got {len(member_shape)}-D shape {member_shape!r}"
            )
        if any(tuple(shape) != member_shape for shape in shapes):
            raise NotImplementedError("NPU GroupedTensor currently supports uniform member shapes")
        if dtype is None:
            raise ValueError("dtype must be specified when allocating grouped tensor storage")
        if device is None:
            device = torch.device("npu")

        rowwise_data = torch.empty(
            (num_tensors, *member_shape),
            dtype=dtype,
            device=device,
        )
        return GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=num_tensors,
            tensor_shape=member_shape,
            rowwise_data=rowwise_data,
            dtype=dtype,
            quantizer=quantizer,
        )

    @staticmethod
    def make_grouped_tensor_from_rowwise_data(
        *,
        num_tensors: int,
        tensor_shape: Tuple[int, ...],
        rowwise_data: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        quantizer=None,
    ) -> "GroupedTensor":
        """Wrap existing packed rowwise storage as a grouped tensor."""
        if num_tensors <= 0:
            raise ValueError(f"num_tensors must be positive, got {num_tensors}")
        if rowwise_data is None:
            raise ValueError("rowwise_data must not be None")
        tensor_shape = tuple(tensor_shape)
        if len(tensor_shape) not in (1, 2):
            raise ValueError(
                "GroupedTensor member shapes must be 1D or 2D, "
                f"got {len(tensor_shape)}-D shape {tensor_shape!r}"
            )
        expected_shape = (num_tensors, *tensor_shape)
        expected_numel = 1
        for dim in expected_shape:
            expected_numel *= dim
        if rowwise_data.numel() != expected_numel:
            raise RuntimeError(
                "GroupedTensor rowwise_data size mismatch: expected "
                f"{expected_numel} elements, got {rowwise_data.numel()}"
            )
        if tuple(rowwise_data.shape) != expected_shape:
            rowwise_data = rowwise_data.contiguous().view(expected_shape)
        elif not rowwise_data.is_contiguous():
            rowwise_data = rowwise_data.contiguous()
        if dtype is None:
            dtype = rowwise_data.dtype

        return GroupedTensor(
            shape=tuple(rowwise_data.shape),
            dtype=dtype,
            num_tensors=num_tensors,
            quantizer=quantizer,
            columnwise_data=None,
            columnwise_scale_inv=None,
            data=rowwise_data,
            scale_inv=None,
            shapes=[tensor_shape] * num_tensors,
            requires_grad=rowwise_data.requires_grad,
        )

    @staticmethod
    def make_grouped_tensor_from_quantized_tensors(
        tensors: List[object],
        *,
        tensor_shape: Tuple[int, ...],
        dtype: torch.dtype,
        quantizer=None,
    ) -> "GroupedTensor":
        """Pack per-group quantized tensor buffers into one grouped tensor wrapper."""
        if not tensors:
            raise ValueError("tensors must not be empty")

        def _stack_optional_buffer(name: str) -> Optional[torch.Tensor]:
            buffers = [getattr(tensor, name, None) for tensor in tensors]
            if all(buffer is None for buffer in buffers):
                return None
            if any(buffer is None for buffer in buffers):
                raise RuntimeError(f"Cannot pack partially initialized quantized buffer {name}")
            return torch.stack([buffer.contiguous() for buffer in buffers], dim=0).contiguous()

        rowwise_data = _stack_optional_buffer("_rowwise_data")
        rowwise_scale_inv = _stack_optional_buffer("_rowwise_scale_inv")
        columnwise_data = _stack_optional_buffer("_columnwise_data")
        columnwise_scale_inv = _stack_optional_buffer("_columnwise_scale_inv")
        if rowwise_data is None and columnwise_data is None:
            raise RuntimeError("Cannot pack quantized tensors without rowwise or columnwise data")

        grouped = GroupedTensor(
            shape=tuple(rowwise_data.shape if rowwise_data is not None else columnwise_data.shape),
            dtype=dtype,
            num_tensors=len(tensors),
            quantizer=quantizer,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            data=rowwise_data,
            scale_inv=rowwise_scale_inv,
            shapes=[tuple(tensor_shape)] * len(tensors),
        )
        grouped.quantized_tensors = tensors
        return grouped

    @classmethod
    def as_dense_grouped_tensor(
        cls,
        tensor: Union["GroupedTensor", torch.Tensor],
        num_tensors: int,
        *,
        shapes: Optional[List[Tuple[int, ...]]] = None,
    ) -> "GroupedTensor":
        """Wrap dense grouped GEMM storage in a GroupedTensor."""
        if isinstance(tensor, GroupedTensor):
            if tensor.quantizer is None:
                return tensor
            if tensor.rowwise_data is None:
                raise RuntimeError("Dense GroupedTensor wrapper requires rowwise_data")
            return cls(
                shape=tuple(tensor.rowwise_data.shape),
                dtype=tensor.rowwise_data.dtype,
                num_tensors=tensor.num_tensors,
                quantizer=None,
                columnwise_data=None,
                columnwise_scale_inv=None,
                data=tensor.rowwise_data,
                scale_inv=None,
                shapes=tensor.tensor_shapes,
                requires_grad=tensor.requires_grad,
            )
        return cls(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            num_tensors=num_tensors,
            quantizer=None,
            columnwise_data=None,
            columnwise_scale_inv=None,
            data=tensor,
            scale_inv=None,
            shapes=shapes,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def _wrapper_shape(
        shape: Tuple[int, ...],
        num_tensors: int,
        shapes: Optional[List[Tuple[int, ...]]],
    ) -> Tuple[int, ...]:
        # For the fast NPU single grouped weight/bias path, uniform members are
        # represented as one packed tensor: (num_groups, *member_shape).
        # Non-uniform metadata falls back to the caller-provided logical shape.
        if (
            shapes is not None
            and len(shapes) == num_tensors
            and num_tensors > 0
            and all(shapes[0] == member_shape for member_shape in shapes)
        ):
            member_shape = tuple(shapes[0])
            if len(member_shape) not in (1, 2):
                raise ValueError(
                    "GroupedTensor member shapes must be 1D or 2D, "
                    f"got {len(member_shape)}-D shape {member_shape!r}"
                )
            return (num_tensors, *member_shape)
        return tuple(shape)

    @staticmethod
    def _infer_device(*tensors: Optional[torch.Tensor]) -> torch.device:
        # Wrapper subclasses still need a device even when buffers are created
        # lazily. The NPU grouped GEMM path is the only intended default here.
        for tensor in tensors:
            if tensor is not None:
                return tensor.device
        return torch.device("npu")

    @staticmethod
    def _initialize_grouped_fields(
        instance: "GroupedTensor",
        *,
        logical_shape: Tuple[int, ...],
        dtype: torch.dtype,
        num_tensors: int,
        quantizer: Optional[Quantizer],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        shapes: Optional[List[Tuple[int, ...]]],
    ) -> None:
        # Keep rowwise/columnwise fields separate because FP8/MXFP8 kernels may
        # consume different packed buffers depending on TensorUsage.
        instance.logical_shape = logical_shape
        instance.fake_dtype = dtype
        instance.num_tensors = num_tensors
        instance.quantizer = quantizer
        instance.tensor_shapes = shapes
        instance.quantized_tensors = None
        instance.columnwise_data = columnwise_data
        instance.columnwise_scale_inv = columnwise_scale_inv
        instance.rowwise_data = rowwise_data
        instance.rowwise_scale_inv = rowwise_scale_inv
        instance.scale_inv = rowwise_scale_inv

    @classmethod
    def _make_wrapper_like(cls, src: "GroupedTensor", requires_grad: bool) -> "GroupedTensor":
        # detach/alias should return a wrapper with identical grouped metadata,
        # not a view of rowwise_data that loses the grouped tensor type.
        out = torch.Tensor._make_wrapper_subclass(
            type(src),
            tuple(src.shape),
            strides=tuple(src.stride()),
            storage_offset=src.storage_offset(),
            dtype=src.dtype,
            layout=src.layout,
            requires_grad=requires_grad,
            device=src.device,
        )
        cls._copy_grouped_metadata(out, src)
        return out

    @classmethod
    def _make_wrapper_from_rowwise_data(
        cls,
        src: "GroupedTensor",
        rowwise_data: torch.Tensor,
        *,
        requires_grad: bool = False,
    ) -> "GroupedTensor":
        return GroupedTensor(
            shape=tuple(rowwise_data.shape),
            dtype=rowwise_data.dtype,
            num_tensors=src.num_tensors,
            quantizer=None,
            columnwise_data=None,
            columnwise_scale_inv=None,
            data=rowwise_data,
            scale_inv=None,
            shapes=src.tensor_shapes,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _copy_grouped_metadata(dst: "GroupedTensor", src: "GroupedTensor") -> None:
        dst.logical_shape = src.logical_shape
        dst.fake_dtype = src.fake_dtype
        dst.num_tensors = src.num_tensors
        dst.quantizer = src.quantizer
        dst.tensor_shapes = src.tensor_shapes
        dst.quantized_tensors = src.quantized_tensors
        dst.columnwise_data = src.columnwise_data
        dst.columnwise_scale_inv = src.columnwise_scale_inv
        dst.rowwise_data = src.rowwise_data
        dst.rowwise_scale_inv = src.rowwise_scale_inv
        dst.scale_inv = src.scale_inv

    def prepare_for_saving(
        self,
    ) -> Tuple[List[Optional[torch.Tensor]], "GroupedTensor"]:
        """Prepare the grouped tensor for saving for backward.

        Returns the plain payload tensors so that CPU offload's
        saved_tensors_hooks can intercept them, while this shell keeps only
        lightweight metadata. Mirrors QuantizedTensorStorage.prepare_for_saving.
        """
        tensors = [
            self.rowwise_data,
            self.columnwise_data,
            self.rowwise_scale_inv,
            self.columnwise_scale_inv,
        ]
        self.rowwise_data = None
        self.columnwise_data = None
        self.rowwise_scale_inv = None
        self.columnwise_scale_inv = None
        self.scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: List[Optional[torch.Tensor]]
    ) -> List[Optional[torch.Tensor]]:
        """Restore payload tensors offloaded/reloaded by CPU offload."""
        self.rowwise_data = tensors[0]
        self.columnwise_data = tensors[1]
        self.rowwise_scale_inv = tensors[2]
        self.columnwise_scale_inv = tensors[3]
        self.scale_inv = tensors[2]
        return tensors[4:]

    @staticmethod
    def _rowwise_arg(arg):
        if isinstance(arg, GroupedTensor):
            if arg.rowwise_data is None:
                raise RuntimeError("GroupedTensor optimizer op requires rowwise_data")
            return arg.rowwise_data
        return arg

    @classmethod
    def _apply_rowwise_in_place(cls, func, args, kwargs):
        dst = args[0]
        cls._check_grouped_tensor(dst)
        if dst.rowwise_data is None:
            raise RuntimeError(f"{cls.__name__} in-place update requires rowwise_data")
        # Forward the actual aten call to the packed rowwise storage while
        # preserving dst as the optimizer-visible parameter object.
        rowwise_args = (dst.rowwise_data,) + tuple(cls._rowwise_arg(arg) for arg in args[1:])
        func(*rowwise_args, **kwargs)
        return dst

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            src = args[0]
            cls._check_grouped_tensor(src)
            return cls._make_wrapper_like(
                src,
                requires_grad=False if func == torch.ops.aten.detach.default else src.requires_grad,
            )

        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            cls._check_grouped_tensor(dst)
            if dst.rowwise_data is None:
                raise RuntimeError(f"{cls.__name__} copy_ requires rowwise_data")
            src_data = src.rowwise_data if isinstance(src, GroupedTensor) else src
            if src_data.numel() != dst.rowwise_data.numel():
                raise RuntimeError(
                    "GroupedTensor copy_ size mismatch: expected "
                    f"{dst.rowwise_data.numel()} elements, got {src_data.numel()}"
                )
            # Parameter loading may provide either a packed GroupedTensor source
            # or a plain tensor with the same flattened element count.
            dst.rowwise_data.copy_(
                src_data.contiguous().view_as(dst.rowwise_data),
                *args[2:],
                **kwargs,
            )
            return dst

        if func in ROWWISE_IN_PLACE_OPS:
            return cls._apply_rowwise_in_place(func, args, kwargs)

        if func in ROWWISE_OPTIMIZER_IN_PLACE_OPS:
            return cls._apply_rowwise_in_place(func, args, kwargs)

        if func == torch.ops.aten.zeros_like.default:
            src = args[0]
            cls._check_grouped_tensor(src)
            if src.rowwise_data is None:
                raise RuntimeError(f"{cls.__name__} zeros_like requires rowwise_data")
            rowwise_data = torch.zeros_like(src.rowwise_data, **kwargs)
            return cls._make_wrapper_from_rowwise_data(src, rowwise_data)

        if func in ROWWISE_OPTIMIZER_OUT_OF_PLACE_OPS:
            src = args[0]
            cls._check_grouped_tensor(src)
            if src.rowwise_data is None:
                raise RuntimeError(f"{cls.__name__} optimizer op requires rowwise_data")
            rowwise_args = (src.rowwise_data,) + tuple(cls._rowwise_arg(arg) for arg in args[1:])
            rowwise_data = func(*rowwise_args, **kwargs)
            return cls._make_wrapper_from_rowwise_data(
                src,
                rowwise_data,
                requires_grad=rowwise_data.requires_grad,
            )

        if func == torch.ops.aten.expand.default:
            src = args[0]
            cls._check_grouped_tensor(src)
            expanded_shape = tuple(args[1])
            src_shape = tuple(src.shape)
            if len(expanded_shape) == len(src_shape):
                normalized_shape = tuple(
                    src_shape[i] if dim == -1 else dim for i, dim in enumerate(expanded_shape)
                )
                if normalized_shape == src_shape:
                    # Some distributed/optimizer code expands parameters by
                    # their existing shape. Treat that as a metadata-preserving
                    # no-op, but reject real broadcasting.
                    return cls._make_wrapper_like(src, requires_grad=src.requires_grad)
            raise RuntimeError(f"{cls.__name__} only supports expand with unchanged wrapper shape")

        if func == torch.ops.aten.expand_as.default:
            src = args[0]
            other = args[1]
            cls._check_grouped_tensor(src)
            if other is src:
                # DDP/MCore hook registration can call expand_as(x, x). The
                # identity autograd node keeps hooks attachable without touching
                # packed storage.
                return _GroupedIdentityFunc.apply(src)
            if tuple(other.shape) == tuple(src.shape):
                return cls._make_wrapper_like(src, requires_grad=src.requires_grad)
            raise RuntimeError(
                f"{cls.__name__} only supports expand_as with unchanged wrapper shape"
            )

        if func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
            src = args[0]
            cls._check_grouped_tensor(src)
            target_shape = tuple(args[1])
            if target_shape in ((-1,), (src.numel(),)):
                if src.rowwise_data is None:
                    raise RuntimeError(
                        f"{cls.__name__} view(-1) requires rowwise_data to be initialized"
                    )
                # Optimizers flatten parameters before applying updates. Only
                # expose the dense storage for that narrow case.
                return src.rowwise_data.view(-1)
            raise RuntimeError(f"{cls.__name__} only supports view(-1) for optimizer flattening")

        if func in BANNED_SHAPE_OPS:
            raise RuntimeError(f"{cls.__name__} forbids shape-manipulation op: {func}")

        raise RuntimeError(
            f"{cls.__name__} only supports detach/alias/expand/expand_as/view(-1) "
            "and grouped GEMM metadata access"
        )

    @staticmethod
    def _check_grouped_tensor(tensor) -> None:
        if not isinstance(tensor, GroupedTensor):
            raise TypeError(f"Expected GroupedTensor, got {type(tensor).__name__}")

    def expand_as(self, other: torch.Tensor) -> torch.Tensor:
        if other is self:
            return _GroupedIdentityFunc.apply(self)
        return super().expand_as(other)

    def all_same_shape(self) -> bool:
        return (
            self.tensor_shapes is not None
            and len(self.tensor_shapes) == self.num_tensors
            and self.num_tensors > 0
            and all(tuple(self.tensor_shapes[0]) == tuple(shape) for shape in self.tensor_shapes)
        )

    def get_dtype(self) -> torch.dtype:
        return self.fake_dtype

    def split_into_quantized_tensors(self) -> List[torch.Tensor]:
        if self.quantizer is not None:
            raise NotImplementedError(
                "GroupedTensor only supports splitting unquantized uniform rowwise data"
            )
        if self.rowwise_data is None:
            raise RuntimeError("GroupedTensor has no rowwise_data to split")
        if not self.all_same_shape():
            raise NotImplementedError("GroupedTensor only supports splitting uniform member shapes")

        tensor_shape = tuple(self.tensor_shapes[0])
        member_numel = 1
        for dim in tensor_shape:
            member_numel *= dim
        expected_numel = self.num_tensors * member_numel
        if self.rowwise_data.numel() != expected_numel:
            raise RuntimeError(
                "GroupedTensor rowwise_data size mismatch: expected "
                f"{expected_numel} elements, got {self.rowwise_data.numel()}"
            )

        flat_data = self.rowwise_data.view(-1)
        # Return views into the packed rowwise buffer so callers that still
        # expect per-GEMM tensors can consume the grouped parameter.
        return [
            flat_data[i * member_numel : (i + 1) * member_numel].view(tensor_shape)
            for i in range(self.num_tensors)
        ]

    def quantize_(
        self, tensor: torch.Tensor, *, noop_flag: Optional[torch.Tensor] = None
    ) -> 'GroupedTensor':
        """Quantize tensor in-place"""
        self.quantizer.update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def grouped_quantize_(
        self, tensor, group_list, *, noop_flag: Optional[torch.Tensor] = None
    ) -> 'GroupedTensor':
        self.quantizer.update_grouped_quantized(tensor, self, group_list, noop_flag=noop_flag)
        return self

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """
        For MXFP8, columnwise scaled output is only produced by x2
        scaling kernels, so this function only disables usages.
        """
        if not self.quantizer:
            return

        rowwise_data = self.rowwise_data
        rowwise_scale_inv = self.rowwise_scale_inv
        columnwise_data = self.columnwise_data
        columnwise_scale_inv = self.columnwise_scale_inv

        if rowwise_usage is None:
            rowwise_usage = rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = columnwise_data is not None

        if rowwise_usage:
            if rowwise_data is None:
                raise RuntimeError(
                    "Requested row-wise usage, but GroupedTensor is missing row-scaled FP8 data"
                )
            if rowwise_scale_inv is None:
                raise RuntimeError(
                    "Requested row-wise usage, "
                    "but GroupedTensor is missing row-scaled scale-inverses"
                )
        else:
            self.rowwise_data = None
            self.rowwise_scale_inv = None
            self.scale_inv = None

        if columnwise_usage:
            if columnwise_data is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but GroupedTensor is missing column-scaled FP8 data"
                )
            if columnwise_scale_inv is None:
                raise RuntimeError(
                    "Requested column-wise usage, "
                    "but GroupedTensor is missing column-scaled scale-inverses"
                )
        else:
            self.columnwise_data = None
            self.columnwise_scale_inv = None

    def get_data(self, usage: Union[TensorUsage, str]):
        if self.quantizer is None:
            if self.rowwise_data is None:
                raise RuntimeError("Dense GroupedTensor requires rowwise_data")
            data = self.rowwise_data
            if usage in USAGE_WITH_TRANS:
                # Dense grouped GEMM stores packed weights as
                # (num_groups, out_features, in_features). Transposed usages
                # are materialized lazily for the corresponding GEMM layout.
                if not data.is_contiguous() and check_npu_version(max_version=NPUVersion.A3):
                    data = data.contiguous()
                data = data.transpose(-1, -2)
            return data, None

        if getattr(self.quantizer, "use_2d_block_scaling", False):
            if usage == TensorUsage.LHS:
                return self.rowwise_data, self.rowwise_scale_inv
            if usage == TensorUsage.RHS_TRANS:
                if self.columnwise_data is None:
                    return self.quantizer.transpose(self.rowwise_data, self.rowwise_scale_inv)
                return self.columnwise_data, self.columnwise_scale_inv
            if usage == TensorUsage.RHS:
                if self.rowwise_data is None:
                    return self.columnwise_data, self.columnwise_scale_inv
                return self.rowwise_data, self.rowwise_scale_inv
            assert usage == TensorUsage.LHS_TRANS
            return self.columnwise_data, self.columnwise_scale_inv

        if usage == TensorUsage.LHS:
            return self.rowwise_data, self.rowwise_scale_inv
        if usage == TensorUsage.RHS_TRANS:
            return self.quantizer.transpose(self.rowwise_data, self.rowwise_scale_inv)
        if usage == TensorUsage.RHS:
            return self.columnwise_data, self.columnwise_scale_inv
        assert usage == TensorUsage.LHS_TRANS
        return self.quantizer.transpose(self.columnwise_data, self.columnwise_scale_inv)

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Return packed data buffers for the requested orientations."""
        if rowwise_data and columnwise_data:
            return self.rowwise_data, self.columnwise_data
        if rowwise_data:
            return self.rowwise_data
        if columnwise_data:
            return self.columnwise_data
        raise ValueError("No data to get, both rowwise_data and columnwise_data are False")

    def dequantize(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        columnwise: bool = False,
    ) -> torch.Tensor:
        """Dequantize one orientation of packed grouped storage."""
        if dtype is None:
            dtype = self.fake_dtype

        if self.quantizer is None:
            if columnwise:
                raise ValueError("Dense GroupedTensor does not contain columnwise data")
            if self.rowwise_data is None:
                raise RuntimeError("GroupedTensor has no data to dequantize")
            return self.rowwise_data.to(dtype=dtype)

        orientation = "columnwise" if columnwise else "rowwise"
        data = self.columnwise_data if columnwise else self.rowwise_data
        scale_inv = self.columnwise_scale_inv if columnwise else self.rowwise_scale_inv
        if data is None:
            raise ValueError(f"GroupedTensor does not contain {orientation} data")
        if scale_inv is None:
            raise ValueError(f"GroupedTensor does not contain {orientation} scale_inv")

        return self.quantizer.dequantize_grouped_tensor(self, dtype=dtype, columnwise=columnwise)
