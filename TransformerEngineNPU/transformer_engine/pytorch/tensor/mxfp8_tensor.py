# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import math
import warnings
from typing import Iterable, Optional, TYPE_CHECKING, Tuple

import torch
import torch_npu

from transformer_engine.common.recipe import MXFP832x32BlockScaling, MXFP8BlockScaling, Recipe
from .utils import noop_requested, to_torch_dtype

from ..constants import MXFP8_BLOCK_SCALING_SIZE, TensorUsage
from ..distributed import gather_along_dim, get_hccl_comm_name
from ..quantized_tensor import QuantizedTensor, Quantizer, transpose_mx_data
from ..utils import divide, view_as_n_dim
from ._quantization_helpers import (
    _IdentityFunc,
    _pack_fp8_tensor_for_pickle,
    _pack_tensor_for_pickle,
    _restore_fp8_tensor_from_pickle,
)
from .storage.mxfp8_tensor_storage import (
    MXFP8TensorStorage,
    _FromMXFP8Func,
)

aten = torch.ops.aten

if TYPE_CHECKING:
    from .grouped_tensor import GroupedTensor


class MXFP8Quantizer(Quantizer):
    """Builder class for FP8 tensors with MX block scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized by
    dividing them into groups of 32 elements, each scaled and cast
    separately using current data.

    ``fp8_dtype`` also accepts the ``transformer_engine_torch.DType``
    compatibility enum (GPU TE convention), which is mapped to the
    matching ``torch.dtype``.
    """

    dtype: torch.dtype

    def __init__(
        self,
        fp8_dtype: torch.dtype,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = to_torch_dtype(fp8_dtype)

    def copy(self) -> MXFP8Quantizer:
        """Create shallow copy"""
        quantizer = MXFP8Quantizer(
            fp8_dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        return quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: MXFP8Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MXFP8Tensor:
        if src is None:
            return src
        if noop_requested(noop_flag):
            return dst

        if not self.rowwise_usage and not self.columnwise_usage:
            return src
        coly, col_scale, rowy, row_scale = None, None, None, None
        tensor_2d = view_as_n_dim(src)
        if self.columnwise_usage and self.rowwise_usage:
            rowy, row_scale, coly, col_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                tensor_2d, dst_type=self.dtype
            )
        elif self.columnwise_usage:
            coly, col_scale = torch_npu.npu_dynamic_mx_quant(
                tensor_2d, axis=-2, dst_type=self.dtype
            )
        elif self.rowwise_usage:
            rowy, row_scale = torch_npu.npu_dynamic_mx_quant(
                tensor_2d, axis=-1, dst_type=self.dtype
            )
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        # copy_ may target a detached param.data wrapper, so preserve the shared storage.
        for name, value in (
            ("_columnwise_data", coly),
            ("_columnwise_scale_inv", col_scale),
            ("_rowwise_data", rowy),
            ("_rowwise_scale_inv", row_scale),
        ):
            dst_value = getattr(dst, name)
            if (dst_value is None) != (value is None):
                raise RuntimeError(
                    f"MXFP8 destination storage does not match quantizer usage: {name}"
                )
            if dst_value is not None:
                dst_value.copy_(value)
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> MXFP8Tensor:
        """Quantize tensor implementation"""
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor
        coly, col_scale, rowy, row_scale = None, None, None, None
        tensor_2d = view_as_n_dim(tensor)

        if self.columnwise_usage and self.rowwise_usage:
            rowy, row_scale, coly, col_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                tensor_2d, dst_type=self.dtype
            )
        elif self.columnwise_usage:
            coly, col_scale = torch_npu.npu_dynamic_mx_quant(
                tensor_2d, axis=-2, dst_type=self.dtype
            )
        elif self.rowwise_usage:
            rowy, row_scale = torch_npu.npu_dynamic_mx_quant(
                tensor_2d, axis=-1, dst_type=self.dtype
            )
        return MXFP8Tensor(
            tensor.shape,
            tensor.dtype,
            rowwise_data=rowy,
            rowwise_scale_inv=row_scale,
            columnwise_data=coly,
            columnwise_scale_inv=col_scale,
            fp8_dtype=self.dtype,
            quantizer=self,
            with_gemm_swizzled_scales=False,
            device=tensor.device,
        )

    def update_grouped_quantized(self, src, dst: "GroupedTensor", group_list, *, noop_flag=None):
        if src is None:
            return src
        if noop_requested(noop_flag):
            return dst
        if not self.rowwise_usage and not self.columnwise_usage:
            return src

        coly, col_scale, rowy, row_scale = None, None, None, None

        if not self.columnwise_use_group_quant and self.columnwise_usage and self.rowwise_usage:
            rowy, row_scale, coly, col_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                src, dst_type=self.dtype
            )
        else:
            if self.columnwise_usage:
                if self.columnwise_use_group_quant:
                    coly, col_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                        src,
                        group_list,
                        round_mode="rint",
                        dst_type=self.dtype,
                        blocksize=32,
                    )
                else:
                    coly, col_scale = torch_npu.npu_dynamic_mx_quant(
                        src, axis=-2, dst_type=self.dtype
                    )
            if self.rowwise_usage:
                rowy, row_scale = torch_npu.npu_dynamic_mx_quant(src, axis=-1, dst_type=self.dtype)

        dst.origin_shape = src.shape
        dst.columnwise_data = coly
        dst.columnwise_scale_inv = col_scale
        dst.rowwise_data = rowy
        dst.rowwise_scale_inv = row_scale
        return dst

    def grouped_quantize_impl(self, group_list, tensor: torch.Tensor):
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor
        coly, col_scale, rowy, row_scale = None, None, None, None

        if not self.columnwise_use_group_quant and self.columnwise_usage and self.rowwise_usage:
            rowy, row_scale, coly, col_scale = torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                tensor, dst_type=self.dtype
            )
        else:
            if self.columnwise_usage:
                if self.columnwise_use_group_quant:
                    coly, col_scale = torch_npu.npu_grouped_dynamic_mx_quant(
                        tensor,
                        group_list.to(torch.int32),
                        round_mode="rint",
                        dst_type=self.dtype,
                        blocksize=32,
                    )
                else:
                    coly, col_scale = torch_npu.npu_dynamic_mx_quant(
                        tensor, axis=-2, dst_type=self.dtype
                    )
            if self.rowwise_usage:
                rowy, row_scale = torch_npu.npu_dynamic_mx_quant(
                    tensor, axis=-1, dst_type=self.dtype
                )
        # pylint: disable=cyclic-import
        from .grouped_tensor import GroupedTensor

        return GroupedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            num_tensors=len(group_list),
            quantizer=self,
            columnwise_data=coly,
            columnwise_scale_inv=col_scale,
            data=rowy,
            scale_inv=row_scale,
        )

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % MXFP8_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % MXFP8_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "MXFP8Tensor":
        if device is None:
            device = torch.device("npu" if hasattr(torch, "npu") else "cpu")
        shape = tuple(shape)
        rowwise_data = None
        rowwise_scale_inv = None
        columnwise_data = None
        columnwise_scale_inv = None
        m_dim = math.prod(shape[:-1]) if len(shape) > 1 else 1
        k_dim = shape[-1] if len(shape) > 0 else 1
        if self.rowwise_usage:
            rowwise_data = torch.empty(
                shape, dtype=self.dtype, device=device, pin_memory=pin_memory
            )
            rowwise_scale_inv = torch.zeros(
                (m_dim, math.ceil(k_dim / (2 * MXFP8_BLOCK_SCALING_SIZE)), 2),
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
        if self.columnwise_usage:
            columnwise_data = torch.empty(
                shape, dtype=self.dtype, device=device, pin_memory=pin_memory
            )
            columnwise_scale_inv = torch.zeros(
                (math.ceil(m_dim / (2 * MXFP8_BLOCK_SCALING_SIZE)), k_dim, 2),
                dtype=torch.uint8,
                device=device,
                pin_memory=pin_memory,
            )
        return MXFP8Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=self.dtype,
            quantizer=self,
            with_gemm_swizzled_scales=False,
            requires_grad=requires_grad,
            device=device,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # TODO(ksivamani): No calibration needed for mxfp8?
        pass

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return MXFP8BlockScaling

    def transpose(self, data, scale):
        return transpose_mx_data(data, scale)

    def dequantize_grouped_tensor(self, tensor, *, dtype=None, columnwise=False):
        """Dequantize one orientation of packed MXFP8 grouped buffers."""
        storage = MXFP8TensorStorage(
            rowwise_data=None if columnwise else tensor.rowwise_data,
            rowwise_scale_inv=None if columnwise else tensor.rowwise_scale_inv,
            columnwise_data=tensor.columnwise_data if columnwise else None,
            columnwise_scale_inv=tensor.columnwise_scale_inv if columnwise else None,
            fp8_dtype=self.dtype,
            quantizer=self,
            with_gemm_swizzled_scales=False,
            fake_dtype=tensor.fake_dtype,
        )
        return storage.dequantize(dtype=dtype if dtype is not None else tensor.fake_dtype)


class MXFP832x32Quantizer(MXFP8Quantizer):
    """Quantize weights with one FP8 payload shared by two GEMM orientations."""

    use_2d_block_scaling = True

    def copy(self) -> "MXFP832x32Quantizer":
        """Create shallow copy."""
        quantizer = MXFP832x32Quantizer(
            fp8_dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        return quantizer

    def _quantize_weight(self, tensor: torch.Tensor):
        data, forward_scale, backward_scale = torch_npu.npu_dynamic_block_mx_quant(
            tensor,
            dst_type=self.dtype,
        )
        columnwise_data, columnwise_scale = transpose_mx_data(data, forward_scale)
        return data, backward_scale, columnwise_data, columnwise_scale

    def quantize_impl(self, tensor: torch.Tensor) -> "MXFP8Tensor":
        """Quantize a weight with the native 32x32 block operator."""
        if tensor is None:
            return tensor
        if not self.rowwise_usage and not self.columnwise_usage:
            return tensor
        # In inference only the forward orientation is needed. Keep the
        # standard one-axis path used by MindSpeed master for that case.
        if not (self.rowwise_usage and self.columnwise_usage):
            return super().quantize_impl(tensor)

        tensor_2d = view_as_n_dim(tensor)
        rowy, row_scale, coly, col_scale = self._quantize_weight(tensor_2d)
        return MXFP8Tensor(
            tensor.shape,
            tensor.dtype,
            rowwise_data=rowy,
            rowwise_scale_inv=row_scale,
            columnwise_data=coly,
            columnwise_scale_inv=col_scale,
            fp8_dtype=self.dtype,
            quantizer=self,
            with_gemm_swizzled_scales=False,
            device=tensor.device,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: "MXFP8Tensor",
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> "MXFP8Tensor":
        if src is None:
            return src
        if noop_requested(noop_flag):
            return dst
        if not (self.rowwise_usage and self.columnwise_usage):
            return super().update_quantized(src, dst, noop_flag=noop_flag)

        rowy, row_scale, coly, col_scale = self._quantize_weight(view_as_n_dim(src))
        dst.origin_shape = src.shape
        dst.dtype = src.dtype
        for name, value in (
            ("_rowwise_data", rowy),
            ("_rowwise_scale_inv", row_scale),
            ("_columnwise_scale_inv", col_scale),
        ):
            dst_value = getattr(dst, name)
            if dst_value is None:
                raise RuntimeError(
                    f"MXFP8-32x32 destination storage does not match quantizer usage: {name}"
                )
            dst_value.copy_(value)
        # The columnwise payload must remain a view of the single rowwise
        # allocation. Validate its layout but do not copy the same bytes twice.
        if dst._columnwise_data is None or dst._columnwise_data.shape != coly.shape:
            raise RuntimeError("MXFP8-32x32 destination has an invalid shared weight layout")
        return dst

    def grouped_quantize_impl(self, group_list, tensor: torch.Tensor):
        if tensor is None:
            return tensor
        if not (self.rowwise_usage and self.columnwise_usage):
            return super().grouped_quantize_impl(group_list, tensor)

        rowy, row_scale, coly, col_scale = self._quantize_weight(tensor)
        # pylint: disable=cyclic-import
        from .grouped_tensor import GroupedTensor

        return GroupedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            num_tensors=len(group_list),
            quantizer=self,
            columnwise_data=coly,
            columnwise_scale_inv=col_scale,
            data=rowy,
            scale_inv=row_scale,
        )

    def update_grouped_quantized(self, src, dst, group_list, *, noop_flag=None):
        if src is None:
            return src
        if noop_requested(noop_flag):
            return dst
        if not (self.rowwise_usage and self.columnwise_usage):
            return super().update_grouped_quantized(src, dst, group_list, noop_flag=noop_flag)

        rowy, row_scale, coly, col_scale = self._quantize_weight(src)
        dst.origin_shape = src.shape
        dst.rowwise_data = rowy
        dst.rowwise_scale_inv = row_scale
        dst.scale_inv = row_scale
        dst.columnwise_data = coly
        dst.columnwise_scale_inv = col_scale
        return dst

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> "MXFP8Tensor":
        if not (self.rowwise_usage and self.columnwise_usage):
            return super().make_empty(
                shape,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
                pin_memory=pin_memory,
            )
        if device is None:
            device = torch.device("npu" if hasattr(torch, "npu") else "cpu")
        shape = tuple(shape)
        m_dim = math.prod(shape[:-1]) if len(shape) > 1 else 1
        k_dim = shape[-1] if shape else 1
        rowwise_data = torch.empty(shape, dtype=self.dtype, device=device, pin_memory=pin_memory)
        columnwise_data = rowwise_data.transpose(-1, -2)
        # DynamicBlockMxQuant emits packed E8M0 scale1 [M, ceil(K/64), 2]
        # and scale2 [ceil(M/64), K, 2]. Store scale1 transposed for fprop.
        rowwise_scale_inv = torch.zeros(
            (math.ceil(m_dim / (2 * MXFP8_BLOCK_SCALING_SIZE)), k_dim, 2),
            dtype=torch.uint8,
            device=device,
            pin_memory=pin_memory,
        )
        columnwise_scale_inv = torch.zeros(
            (math.ceil(k_dim / (2 * MXFP8_BLOCK_SCALING_SIZE)), m_dim, 2),
            dtype=torch.uint8,
            device=device,
            pin_memory=pin_memory,
        )
        return MXFP8Tensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=self.dtype,
            quantizer=self,
            with_gemm_swizzled_scales=False,
            requires_grad=requires_grad,
            device=device,
        )

    def _get_compatible_recipe(self) -> Optional[type[Recipe]]:
        return MXFP832x32BlockScaling


class MXFP8Tensor(MXFP8TensorStorage, QuantizedTensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.
    """

    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: torch.dtype,
        quantizer: Optional[Quantizer],
        with_gemm_swizzled_scales: bool,
        **kwargs,
    ):
        return super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp8_dtype,
            quantizer,
            with_gemm_swizzled_scales,
            *args,
            **kwargs,
        )

    def _build_default_quantizer(self) -> Optional[Quantizer]:
        """Build default quantizer for the tensor"""
        return MXFP8Quantizer(fp8_dtype=self.dtype)

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("MXFP8Tensor has no data!")

    @property
    def shape(self):
        """Return the shape of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.shape
        if self._columnwise_data is not None:
            return self._columnwise_data.shape
        return torch.Tensor.size(self)

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA/NPU-backed device."""
        if self._rowwise_data is not None:
            return self._rowwise_data.is_npu
        if self._columnwise_data is not None:
            return self._columnwise_data.is_npu
        raise RuntimeError("MXFP8Tensor has no data!")

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Construct plain PyTorch tensor from MXFP8Tensor."""
        if dtype is None:
            dtype = self.dtype
        if torch.is_grad_enabled():
            result = _FromMXFP8Func.apply(self, dtype)
        else:
            result = _FromMXFP8Func.forward(None, self, dtype)
        if tuple(result.shape) != tuple(self.origin_shape):
            result = result.reshape(self.origin_shape)
        return result

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MXFP8Tensor:
        """Update MXFP8 data in-place from a dense tensor."""
        if isinstance(tensor, QuantizedTensor):
            tensor = tensor.dequantize()
        if tensor.device != self.device:
            tensor = tensor.to(device=self.device)
        return super().quantize_(tensor, noop_flag=noop_flag)

    def detach(self) -> MXFP8Tensor:
        """Return a detached copy that preserves the logical shape.

        ``make_like`` defaults to the physical storage shape (``_rowwise_data``
        is the quantizer's flattened 2D view); dropping the logical shape here
        corrupts ``dequantize()`` of detached/offloaded copies (e.g. CPU
        offload roundtrips).
        """
        return MXFP8Tensor.make_like(self, shape=self.origin_shape)

    def clone(self) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        rowwise_data = None
        columnwise_data = None
        rowwise_scale_inv = None
        columnwise_scale_inv = None
        if self._rowwise_data is not None:
            rowwise_data = self._rowwise_data.detach().clone()
        if (
            rowwise_data is not None
            and self._columnwise_data is not None
            and self._quantizer is not None
            and getattr(self._quantizer, "use_2d_block_scaling", False)
        ):
            columnwise_data = rowwise_data.transpose(-1, -2)
        elif self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        if self._rowwise_scale_inv is not None:
            rowwise_scale_inv = self._rowwise_scale_inv.detach().clone()
        if self._columnwise_scale_inv is not None:
            columnwise_scale_inv = self._columnwise_scale_inv.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
                "rowwise_scale_inv": rowwise_scale_inv,
                "columnwise_scale_inv": columnwise_scale_inv,
            },
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Serialize MXFP8 payload and scale metadata."""
        columnwise_data = self._columnwise_data
        if (
            self._rowwise_data is not None
            and columnwise_data is not None
            and self._quantizer is not None
            and getattr(self._quantizer, "use_2d_block_scaling", False)
            and self._rowwise_data.untyped_storage().data_ptr()
            == columnwise_data.untyped_storage().data_ptr()
        ):
            columnwise_data = None
        return (
            _make_mxfp8_tensor_in_reduce_ex,
            (
                _pack_fp8_tensor_for_pickle(self._rowwise_data),
                _pack_tensor_for_pickle(self._rowwise_scale_inv),
                _pack_fp8_tensor_for_pickle(columnwise_data),
                _pack_tensor_for_pickle(self._columnwise_scale_inv),
                self._fp8_dtype,
                self.dtype,
                torch.Size(self.origin_shape),
                self._quantizer,
                self._with_gemm_swizzled_scales,
            ),
        )

    @classmethod
    def _make_in_reduce_ex(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: torch.dtype,
        dtype: torch.dtype,
        shape: torch.Size,
        quantizer: Optional[Quantizer] = None,
        with_gemm_swizzled_scales: bool = False,
    ) -> MXFP8Tensor:
        """Reconstruct an MXFP8 tensor from an older pickle payload."""
        return _make_mxfp8_tensor_in_reduce_ex(
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp8_dtype,
            dtype,
            shape,
            quantizer,
            with_gemm_swizzled_scales,
        )

    @staticmethod
    def _split_optional_tensor(
        tensor: Optional[torch.Tensor],
        split_func,
        split_size: int,
        dim: int,
    ) -> Optional[list[torch.Tensor]]:
        """Split a backing tensor if it is present."""
        if tensor is None:
            return None
        return list(split_func(tensor, split_size, dim))

    @staticmethod
    def _split_packed_rowwise_scale(
        scale: torch.Tensor,
        *,
        full_m_blocks: int,
        split_m_blocks: int,
        k_dim: int,
        split_func,
    ) -> Optional[list[torch.Tensor]]:
        """Split packed rowwise scale layouts along logical M blocks."""
        if scale.ndim != 2 or scale.shape[1] <= 0:
            return None

        packed_dim = scale.shape[1]
        packed_k = math.ceil(k_dim / packed_dim)
        if scale.shape[0] == full_m_blocks * packed_k:
            return list(split_func(scale, split_m_blocks * packed_k, 0))

        packed_m = math.ceil(full_m_blocks / packed_dim)
        if scale.shape[0] != k_dim * packed_m:
            return None

        scale_view = scale.reshape(k_dim, packed_m, packed_dim)
        outputs = []
        for start in range(0, full_m_blocks, split_m_blocks):
            start_pack = start // packed_dim
            end_pack = math.ceil((start + split_m_blocks) / packed_dim)
            outputs.append(
                scale_view[:, start_pack:end_pack, :].contiguous().reshape(-1, packed_dim)
            )
        return outputs

    @classmethod
    def _split_rowwise_scale(
        cls,
        scale: Optional[torch.Tensor],
        *,
        full_m_dim: int,
        split_m_dim: int,
        full_m_blocks: int,
        split_m_blocks: int,
        k_dim: int,
        split_func,
    ) -> Optional[list[torch.Tensor]]:
        """Split rowwise scales while preserving known NPU packed layouts."""
        if scale is None:
            return None
        if scale.ndim >= 2 and scale.shape[0] == full_m_dim:
            return list(split_func(scale, split_m_dim, 0))
        if scale.ndim >= 2 and scale.shape[0] == full_m_blocks:
            return list(split_func(scale, split_m_blocks, 0))
        if scale.ndim == 2 and scale.shape[0] == k_dim:
            return list(split_func(scale, split_m_blocks, 1))

        packed_splits = cls._split_packed_rowwise_scale(
            scale,
            full_m_blocks=full_m_blocks,
            split_m_blocks=split_m_blocks,
            k_dim=k_dim,
            split_func=split_func,
        )
        if packed_splits is not None:
            return packed_splits

        return list(split_func(scale, split_m_blocks, -1))

    @staticmethod
    def _split_columnwise_scale(
        scale: Optional[torch.Tensor],
        *,
        full_m_blocks: int,
        split_m_blocks: int,
        split_func,
    ) -> Optional[list[torch.Tensor]]:
        """Split columnwise scales along logical M blocks."""
        if scale is None:
            return None
        if scale.ndim >= 2 and scale.shape[0] == full_m_blocks:
            return list(split_func(scale, split_m_blocks, 0))
        if scale.ndim >= 3:
            pack_dim = scale.shape[-1]
            if (
                pack_dim > 0
                and scale.shape[0] == math.ceil(full_m_blocks / pack_dim)
                and split_m_blocks % pack_dim == 0
            ):
                return list(split_func(scale, split_m_blocks // pack_dim, 0))
        return list(split_func(scale, split_m_blocks, 0))

    @staticmethod
    def _requantized_split(
        tensor: "MXFP8Tensor",
        split_func,
        split_args,
        split_kwargs,
    ):
        """Fallback split path for unknown packed scale layouts."""
        dense_splits = split_func(
            tensor.dequantize(dtype=tensor.dtype),
            *split_args,
            **split_kwargs,
        )
        if getattr(tensor, "_quantizer", None) is None:
            return dense_splits
        return [tensor._quantizer(split) for split in dense_splits]

    @staticmethod
    def _make_split_tensor(
        tensor: "MXFP8Tensor",
        *,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
    ) -> "MXFP8Tensor":
        """Build one MXFP8Tensor shard from split buffers."""
        shape = rowwise_data.shape if rowwise_data is not None else columnwise_data.shape
        return MXFP8Tensor(
            shape=shape,
            dtype=tensor.dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            quantizer=tensor._quantizer,
            with_gemm_swizzled_scales=False,
            device=tensor.device,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == aten.view.default:
            tensor = args[0]
            return tensor.view(*args[1])

        if func == aten.split.Tensor:
            tensor = args[0]
            split_size = args[1]
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            if (
                dim != 0
                or tensor.size(0) % split_size != 0
                or split_size % MXFP8_BLOCK_SCALING_SIZE != 0
                or tensor._with_gemm_swizzled_scales
            ):
                return super().__torch_dispatch__(func, types, args, kwargs)

            split_args = args[1:]
            rowwise_data_splits = cls._split_optional_tensor(
                tensor._rowwise_data, func, split_size, dim
            )
            columnwise_data_splits = cls._split_optional_tensor(
                tensor._columnwise_data, func, split_size, dim
            )

            inner_multiplier = math.prod(tensor.shape[1:-1]) if len(tensor.shape) > 2 else 1
            split_m_dim = split_size * inner_multiplier
            full_m_dim = math.prod(tensor.shape[:-1]) if len(tensor.shape) > 1 else 1
            full_m_blocks = math.ceil(full_m_dim / MXFP8_BLOCK_SCALING_SIZE)
            split_m_blocks = math.ceil(split_m_dim / MXFP8_BLOCK_SCALING_SIZE)
            k_dim = tensor.shape[-1]

            rowwise_scale_splits = cls._split_rowwise_scale(
                tensor._rowwise_scale_inv,
                full_m_dim=full_m_dim,
                split_m_dim=split_m_dim,
                full_m_blocks=full_m_blocks,
                split_m_blocks=split_m_blocks,
                k_dim=k_dim,
                split_func=func,
            )
            columnwise_scale_splits = cls._split_columnwise_scale(
                tensor._columnwise_scale_inv,
                full_m_blocks=full_m_blocks,
                split_m_blocks=split_m_blocks,
                split_func=func,
            )

            num_outputs = (
                len(rowwise_data_splits)
                if rowwise_data_splits is not None
                else len(columnwise_data_splits)
            )
            if rowwise_scale_splits is not None and len(rowwise_scale_splits) != num_outputs:
                return cls._requantized_split(tensor, func, split_args, kwargs)
            if columnwise_scale_splits is not None and len(columnwise_scale_splits) != num_outputs:
                return cls._requantized_split(tensor, func, split_args, kwargs)
            outputs = []
            for idx in range(num_outputs):
                outputs.append(
                    cls._make_split_tensor(
                        tensor,
                        rowwise_data=(
                            rowwise_data_splits[idx] if rowwise_data_splits is not None else None
                        ),
                        rowwise_scale_inv=(
                            rowwise_scale_splits[idx] if rowwise_scale_splits is not None else None
                        ),
                        columnwise_data=(
                            columnwise_data_splits[idx]
                            if columnwise_data_splits is not None
                            else None
                        ),
                        columnwise_scale_inv=(
                            columnwise_scale_splits[idx]
                            if columnwise_scale_splits is not None
                            else None
                        ),
                    )
                )
            return outputs

        if func == aten.as_strided.default:
            tensor = args[0]
            shape = args[1]
            strides = args[2]
            if (
                len(shape) == len(strides) == 2
                and tuple(strides) == (shape[-1], 1)
                and tuple(shape) == tuple(tensor.size())
            ):
                return MXFP8Tensor.make_like(tensor)

        if func == aten.slice.Tensor:
            tensor = args[0]
            dim = args[1]
            start = args[2]
            end = args[3]
            step = args[4] if len(args) > 4 else 1
            if start == 0 and end == tensor.size(dim) and step == 1:
                return MXFP8Tensor.make_like(tensor)

        if func == aten.new_zeros.default:
            tensor = args[0]
            shape = torch.Size(args[1])
            if len(shape) < 2 or getattr(tensor, "_quantizer", None) is None:
                return super().__torch_dispatch__(func, types, args, kwargs)
            first_dim = math.prod(shape[:-1])
            last_dim = shape[-1]
            if (
                first_dim % MXFP8_BLOCK_SCALING_SIZE != 0
                or last_dim % MXFP8_BLOCK_SCALING_SIZE != 0
            ):
                return super().__torch_dispatch__(func, types, args, kwargs)
            out = tensor._quantizer.make_empty(
                shape=shape,
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=False,
            )
            for internal in (
                out._rowwise_data,
                out._rowwise_scale_inv,
                out._columnwise_data,
                out._columnwise_scale_inv,
            ):
                if internal is not None:
                    internal.zero_()
            return out

        return super().__torch_dispatch__(func, types, args, kwargs)

    def view(self, *shape) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> MXFP8Tensor:
        """Return MXFP8Tensor with contiguous buffers in the requested format."""
        rowwise_data = self._rowwise_data
        columnwise_data = self._columnwise_data
        rowwise_scale_inv = self._rowwise_scale_inv
        columnwise_scale_inv = self._columnwise_scale_inv

        if rowwise_data is not None and not rowwise_data.is_contiguous(memory_format=memory_format):
            pass
        elif columnwise_data is not None and not columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            pass
        elif rowwise_scale_inv is not None and not rowwise_scale_inv.is_contiguous():
            pass
        elif columnwise_scale_inv is not None and not columnwise_scale_inv.is_contiguous():
            pass
        else:
            return self

        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": (
                    rowwise_data.contiguous(memory_format=memory_format)
                    if rowwise_data is not None
                    else None
                ),
                "columnwise_data": (
                    columnwise_data.contiguous(memory_format=memory_format)
                    if columnwise_data is not None
                    else None
                ),
                "rowwise_scale_inv": (
                    rowwise_scale_inv.contiguous() if rowwise_scale_inv is not None else None
                ),
                "columnwise_scale_inv": (
                    columnwise_scale_inv.contiguous() if columnwise_scale_inv is not None else None
                ),
            },
        )

    def _view_or_reshape(self, shape, *, reshape: bool) -> MXFP8Tensor:
        """Return an MXFP8Tensor with updated logical shape."""
        cur_shape = self.size()
        shape = _canonicalize_shape(cur_shape, shape)
        if tuple(shape) == tuple(cur_shape):
            return self

        if shape[-1] != cur_shape[-1]:
            if not reshape:
                warnings.warn(
                    "MXFP8Tensor does not support reshaping inner dimension. "
                    f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)}) "
                    "Falling back to dequantize().view(...) in higher precision.",
                    stacklevel=2,
                )
                return self.dequantize().view(*shape)
            raise RuntimeError(
                "MXFP8Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(cur_shape)} to {tuple(shape)})"
            )

        cur_rowwise_data = self._rowwise_data
        cur_columnwise_data = self._columnwise_data
        new_rowwise_data = None
        new_columnwise_data = None
        if cur_rowwise_data is not None:
            new_rowwise_data = (
                cur_rowwise_data.reshape(*shape) if reshape else cur_rowwise_data.view(*shape)
            )
        if cur_columnwise_data is not None:
            new_columnwise_data = (
                cur_columnwise_data.reshape(*shape) if reshape else cur_columnwise_data.view(*shape)
            )

        return MXFP8Tensor(
            shape,
            self.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=self._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=self._columnwise_scale_inv,
            fp8_dtype=self._fp8_dtype,
            quantizer=self._quantizer,
            with_gemm_swizzled_scales=self._with_gemm_swizzled_scales,
            device=self.device,
        )

    def matmul(self, B: MXFP8Tensor, usage, usage_b, out_dtype):
        from .storage.w4a8_tensor_storage import W4A8WeightTensorStorage

        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)

        extra_kwargs = {}
        if isinstance(B, W4A8WeightTensorStorage):
            extra_kwargs["x2_dtype"] = (
                torch_npu.float4_e2m1fn_x2 if b.dtype == torch.uint8 else b.dtype
            )

        out = torch_npu.npu_quant_matmul(
            a,
            b,
            b_scale,
            pertoken_scale=a_scale,
            output_dtype=out_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
            **extra_kwargs,
        )
        return out

    def matmul_add(self, main_grad, B: MXFP8Tensor, usage, usage_b, out_dtype):
        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)

        # aclnnQuantBatchMatmulInplaceAdd requires yRef (main_grad) in
        # DT_FLOAT. Recipes using bf16 main grads (e.g. MB deepseek_v3
        # main_grads_dtype=bfloat16) fail with EZ0020. Fallback: separate
        # quant matmul then add_ accumulation (equivalent).
        if main_grad.dtype != torch.float32:
            wgrad = self.matmul(B, usage, usage_b, out_dtype=main_grad.dtype)
            main_grad.add_(wgrad)
            return

        torch_npu.npu_add_quant_matmul_(
            main_grad,
            a,
            b,
            b_scale,
            x1_scale=a_scale,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )

    def allgather_matmul(
        self,
        B: MXFP8Tensor,
        bias,
        world_size,
        group,
        usage,
        usage_b,
        out_dtype,
        return_gathered: bool = True,
    ):
        from .storage.w4a8_tensor_storage import W4A8WeightTensorStorage

        if isinstance(B, W4A8WeightTensorStorage):
            raise NotImplementedError("W4A8 does not support overlap allgather_matmul yet")

        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)
        # TODO @Muu 这里需要校验一下是否正确
        gather_out = None
        if return_gathered:
            gather_out = self.gather_along_data(group, transpose=usage_b == TensorUsage.LHS)
        out, _, _ = torch_npu.npu_all_gather_quant_mm(
            a,
            b,
            get_hccl_comm_name(group),
            world_size,
            bias=bias,
            x1_scale=a_scale,
            x2_scale=b_scale,
            quant_scale=None,
            block_size=0,
            comm_turn=0,
            group_sizes=[1, 1, 32],
            amax_output=False,
            y_dtype=out_dtype,
            gather_output=False,
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        return out, gather_out

    def gather_along_data(self, group, col=True, transpose=False):
        data, scale = self.col if col else self.row
        if transpose:
            data, scale = transpose_mx_data(data, scale)
        data, _ = gather_along_dim(data, group)
        scale, _ = gather_along_dim(scale, group)
        if transpose:
            data, scale = transpose_mx_data(data, scale)
        rowwise_data, rowwise_scale_inv = (data, scale) if not col else (None, None)
        columnwise_data, columnwise_scale_inv = (data, scale) if col else (None, None)
        return MXFP8Tensor(
            self.origin_shape,
            self.dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            fp8_dtype=self._fp8_dtype,
            quantizer=self._quantizer,
            with_gemm_swizzled_scales=self._with_gemm_swizzled_scales,
            device=self.device,
        )

    def matmul_reduce_scatter(
        self, B: MXFP8Tensor, bias, world_size, group, usage, usage_b, out_dtype
    ):
        from .storage.w4a8_tensor_storage import W4A8WeightTensorStorage

        if isinstance(B, W4A8WeightTensorStorage):
            raise NotImplementedError("W4A8 does not support overlap matmul_reduce_scatter yet")

        a, a_scale = self.get_data(usage)
        b, b_scale = B.get_data(usage_b)
        out, _ = torch_npu.npu_quant_mm_reduce_scatter(
            a,
            b,
            get_hccl_comm_name(group),
            world_size,
            bias=bias,
            reduce_op="sum",
            x1_scale=a_scale,
            x2_scale=b_scale,
            quant_scale=None,
            block_size=0,
            comm_turn=0,
            group_sizes=[1, 1, 32],
            amax_output=False,
            y_dtype=out_dtype,
            x1_dtype=None,
            x2_dtype=None,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        return out

    @classmethod
    def quant_reduce_scatter(cls, tensor, quantizer, world_size, group):
        quantizer.set_usage(rowwise=False)
        data, scale = quantizer.quantize(tensor).col

        total_out = torch_npu.npu_quant_reduce_scatter(
            data,
            scale,
            get_hccl_comm_name(group),
            world_size,
            scales_dtype=torch_npu.float8_e8m0fnu,
        )
        return total_out.view(divide(tensor.shape[0], world_size), *tensor.shape[1:])


def _make_mxfp8_tensor_in_reduce_ex(
    rowwise_data: Optional[torch.Tensor],
    rowwise_scale_inv: Optional[torch.Tensor],
    columnwise_data: Optional[torch.Tensor],
    columnwise_scale_inv: Optional[torch.Tensor],
    fp8_dtype: torch.dtype,
    dtype: torch.dtype,
    shape: torch.Size,
    quantizer: Optional[Quantizer] = None,
    with_gemm_swizzled_scales: bool = False,
) -> MXFP8Tensor:
    """Reconstruct an ``MXFP8Tensor`` from its pickle payload."""
    rowwise_data = _restore_fp8_tensor_from_pickle(rowwise_data, fp8_dtype)
    columnwise_data = _restore_fp8_tensor_from_pickle(columnwise_data, fp8_dtype)
    if (
        rowwise_data is not None
        and columnwise_data is None
        and columnwise_scale_inv is not None
        and quantizer is not None
        and getattr(quantizer, "use_2d_block_scaling", False)
    ):
        columnwise_data = rowwise_data.transpose(-1, -2)
    device = None
    if rowwise_data is not None:
        device = rowwise_data.device
    elif columnwise_data is not None:
        device = columnwise_data.device
    return MXFP8Tensor(
        shape,
        dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        fp8_dtype=fp8_dtype,
        quantizer=quantizer,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
        device=device,
    )


def _reshape_grad_to_input(grad: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reshape view/reshape gradients back to the input tensor shape."""
    if isinstance(grad, QuantizedTensor) and hasattr(grad, "_view_or_reshape"):
        return grad._view_or_reshape(shape, reshape=True)
    return grad.reshape(shape)


class _ViewFunc(torch.autograd.Function):
    """Autograd-aware view for MXFP8Tensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP8Tensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        ctx.shape = torch.Size(tensor.origin_shape)
        if shape is None:
            return tensor.detach()
        return tensor._view_or_reshape(shape, reshape=False)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        return _reshape_grad_to_input(grad, ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Autograd-aware reshape for MXFP8Tensor."""

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP8Tensor,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        ctx.shape = torch.Size(tensor.origin_shape)
        if shape is None:
            return tensor.detach()
        return tensor._view_or_reshape(shape, reshape=True)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        return _reshape_grad_to_input(grad, ctx.shape), None


def _canonicalize_shape(cur_shape, shape) -> torch.Size:
    """Normalize view/reshape arguments and resolve a single inferred dim."""
    if not isinstance(shape, Iterable):
        shape = [shape]
    elif len(shape) == 1 and isinstance(shape[0], Iterable):
        shape = list(shape[0])
    else:
        shape = list(shape)

    infer_index = None
    known_product = 1
    for i, dim in enumerate(shape):
        if dim == -1:
            if infer_index is not None:
                raise RuntimeError("Only one dimension can be inferred")
            infer_index = i
            continue
        known_product *= dim

    if infer_index is not None:
        inferred = math.prod(cur_shape) // known_product
        shape[infer_index] = inferred

    return torch.Size(shape)


__all__ = ["MXFP8Quantizer", "MXFP8Tensor"]
