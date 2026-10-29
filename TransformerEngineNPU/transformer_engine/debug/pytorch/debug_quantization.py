# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Debug quantizer support for TransformerEngine NPU."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch

from transformer_engine.pytorch.constants import TensorUsage
from transformer_engine.pytorch.quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.utils import view_as_n_dim
from transformer_engine.debug.pytorch.debug_state import TEDebugState


API_CALL_MODIFY = "modify_tensor()"
STANDARD_QUANTIZE = "Quantize"
HIGH_PRECISION = "High Precision"

_tensor_to_gemm_names_map = {
    "weight": ["fprop", "dgrad"],
    "activation": ["fprop", "wgrad"],
    "output": ["fprop", None],
    "gradient": ["dgrad", "wgrad"],
    "wgrad": ["wgrad", None],
    "dgrad": ["dgrad", None],
}


class DebugQuantizer(Quantizer):
    """Wrap a TENPU quantizer and insert nvdlfw-inspect debug hooks."""

    def __init__(
        self,
        layer_name: str,
        tensor_name: str,
        parent_quantizer: Optional[Quantizer],
        tp_group: Optional[torch.distributed.ProcessGroup],
        tp_size: int,
    ):
        super().__init__(rowwise=True, columnwise=True)
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.parent_quantizer = parent_quantizer
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.iteration = TEDebugState.get_iteration()
        self.rowwise_gemm_name, self.columnwise_gemm_name = _tensor_to_gemm_names_map[tensor_name]
        self.next_debug_iter = None
        self.output_tensor = tensor_name in {"output", "wgrad", "dgrad"}

        if parent_quantizer is not None:
            parent_quantizer.internal = False
            parent_quantizer.optimize_for_gemm = False

        if self.output_tensor:
            self.inspect_tensor_enabled, self.rowwise_tensor_plan = (
                self.get_plans_for_output_tensors()
            )
            self.columnwise_tensor_plan = HIGH_PRECISION
            self.inspect_tensor_postquantize_enabled_rowwise = False
            self.inspect_tensor_postquantize_enabled_columnwise = False
        else:
            (
                self.inspect_tensor_enabled,
                self.inspect_tensor_postquantize_enabled_rowwise,
                self.inspect_tensor_postquantize_enabled_columnwise,
            ) = self.get_enabled_look_at_tensors()
            self.rowwise_tensor_plan, self.columnwise_tensor_plan = self.get_tensors_plan()
            self.log_messages_about_plans()

    def get_plans_for_output_tensors(self) -> Tuple[bool, str]:
        """Return inspect enablement and output-tensor plan."""
        import nvdlfw_inspect.api as debug_api

        inspect_tensor_enabled = self.process_enabled_api_call(
            debug_api.transformer_engine.inspect_tensor_enabled(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                iteration=self.iteration,
            )
        )
        modify_enabled = self.process_enabled_api_call(
            debug_api.transformer_engine.modify_tensor_enabled(
                layer_name=self.layer_name,
                gemm=self.rowwise_gemm_name,
                tensor_name=self.tensor_name,
                iteration=self.iteration,
            )
        )
        plan = API_CALL_MODIFY if modify_enabled else HIGH_PRECISION
        return inspect_tensor_enabled, plan

    def get_enabled_look_at_tensors(self):
        """Return inspect pre/post-quantize enablement."""
        import nvdlfw_inspect.api as debug_api

        inspect_tensor_enabled = self.process_enabled_api_call(
            debug_api.transformer_engine.inspect_tensor_enabled(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                iteration=self.iteration,
            )
        )
        inspect_tensor_postquantize_enabled_rowwise = self.process_enabled_api_call(
            debug_api.transformer_engine.inspect_tensor_postquantize_enabled(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                iteration=self.iteration,
                gemm=self.rowwise_gemm_name,
            )
        )
        inspect_tensor_postquantize_enabled_columnwise = self.process_enabled_api_call(
            debug_api.transformer_engine.inspect_tensor_postquantize_enabled(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                iteration=self.iteration,
                gemm=self.columnwise_gemm_name,
            )
        )
        return (
            inspect_tensor_enabled,
            inspect_tensor_postquantize_enabled_rowwise,
            inspect_tensor_postquantize_enabled_columnwise,
        )

    def get_tensors_plan(self):
        """
        Returns (rowwise_plan, columnwise_plan). Each element of the tuple is one of
        API_CALL_MODIFY, STANDARD_QUANTIZE, or HIGH_PRECISION, indicating the behavior
        of this quantizer with respect to these tensors.
        """
        import nvdlfw_inspect.api as debug_api

        rowwise_plan = None
        columnwise_plan = None

        modify_rowwise = self.process_enabled_api_call(
            debug_api.transformer_engine.modify_tensor_enabled(
                layer_name=self.layer_name,
                gemm=self.rowwise_gemm_name,
                tensor_name=self.tensor_name,
                iteration=self.iteration,
            )
        )

        if modify_rowwise:
            rowwise_plan = API_CALL_MODIFY
        else:
            if self.parent_quantizer is not None:
                quantize_enabled = self.process_enabled_api_call(
                    debug_api.transformer_engine.fp8_gemm_enabled(
                        layer_name=self.layer_name,
                        gemm=self.rowwise_gemm_name,
                        iteration=self.iteration,
                    )
                )

                if quantize_enabled:
                    rowwise_plan = STANDARD_QUANTIZE
        if rowwise_plan is None:
            rowwise_plan = HIGH_PRECISION

        if self.columnwise_gemm_name is not None:
            modify_columnwise = self.process_enabled_api_call(
                debug_api.transformer_engine.modify_tensor_enabled(
                    layer_name=self.layer_name,
                    gemm=self.columnwise_gemm_name,
                    tensor_name=self.tensor_name,
                    iteration=self.iteration,
                )
            )

            if modify_columnwise:
                columnwise_plan = API_CALL_MODIFY
            else:
                if self.parent_quantizer is not None:
                    quantize_enabled = self.process_enabled_api_call(
                        debug_api.transformer_engine.fp8_gemm_enabled(
                            layer_name=self.layer_name,
                            gemm=self.columnwise_gemm_name,
                            iteration=self.iteration,
                        )
                    )

                    if quantize_enabled:
                        columnwise_plan = STANDARD_QUANTIZE
        if columnwise_plan is None:
            columnwise_plan = HIGH_PRECISION

        return rowwise_plan, columnwise_plan

    def log_messages_about_plans(self) -> None:
        """Log rowwise/columnwise debug plans."""
        import nvdlfw_inspect.api as debug_api

        debug_api.log_message(
            f"Tensor: {self.tensor_name}, gemm {self.rowwise_gemm_name} -"
            f" {self.rowwise_tensor_plan}",
            layer_name=self.layer_name,
            extra_cachable_args=(self.rowwise_gemm_name, self.tensor_name),
        )
        debug_api.log_message(
            f"Tensor: {self.tensor_name}, gemm {self.columnwise_gemm_name} -"
            f" {self.columnwise_tensor_plan}",
            layer_name=self.layer_name,
            extra_cachable_args=(self.columnwise_gemm_name, self.tensor_name),
        )

    def process_enabled_api_call(self, enabled_call_output) -> bool:
        """Update next-debug iteration from nvdlfw-inspect API output."""
        if isinstance(enabled_call_output, tuple):
            assert len(enabled_call_output) == 2, "Expected a tuple of length 2"
            enabled, next_iter = enabled_call_output
        else:
            enabled = enabled_call_output
            next_iter = self.iteration + 1

        if self.next_debug_iter is None:
            self.next_debug_iter = next_iter
        elif next_iter is not None:
            self.next_debug_iter = min(self.next_debug_iter, next_iter)
        return enabled

    def any_feature_enabled(self) -> bool:
        """Return whether this quantizer changes normal behavior."""
        if self.output_tensor:
            return self.inspect_tensor_enabled or self.rowwise_tensor_plan == API_CALL_MODIFY

        if (
            self.inspect_tensor_enabled
            or self.inspect_tensor_postquantize_enabled_rowwise
            or self.inspect_tensor_postquantize_enabled_columnwise
            or self.rowwise_tensor_plan == API_CALL_MODIFY
            or self.columnwise_tensor_plan == API_CALL_MODIFY
        ):
            return True

        if self.parent_quantizer is not None:
            return (
                self.rowwise_tensor_plan != STANDARD_QUANTIZE
                or self.columnwise_tensor_plan != STANDARD_QUANTIZE
            )
        return False

    def get_next_debug_iter(self):
        """Return the next iteration this quantizer needs debug checks."""
        return self.next_debug_iter

    def _update_parent_quantizer_usage(self) -> None:
        if self.parent_quantizer is None:
            return
        rowwise_gemm_quantize = self.rowwise_usage and self.rowwise_tensor_plan == STANDARD_QUANTIZE
        columnwise_gemm_quantize = (
            self.columnwise_usage and self.columnwise_tensor_plan == STANDARD_QUANTIZE
        )
        if STANDARD_QUANTIZE in (self.rowwise_tensor_plan, self.columnwise_tensor_plan):
            self.parent_quantizer.set_usage(
                rowwise=rowwise_gemm_quantize,
                columnwise=columnwise_gemm_quantize,
            )

    def set_usage(
        self, *, rowwise: Optional[bool] = None, columnwise: Optional[bool] = None
    ) -> None:
        super().set_usage(rowwise=rowwise, columnwise=columnwise)
        if not self.output_tensor:
            self._update_parent_quantizer_usage()

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        out: Optional[QuantizedTensorStorage] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Quantize or route a tensor according to the debug plans."""
        assert not self.output_tensor
        if out is not None:
            return self.update_quantized(tensor, out)

        import nvdlfw_inspect.api as debug_api

        self._update_parent_quantizer_usage()
        if self.parent_quantizer is not None:
            if not self.parent_quantizer.rowwise_usage and self.parent_quantizer.columnwise_usage:
                self.parent_quantizer.set_usage(rowwise=True)

        rowwise_gemm_tensor = None
        columnwise_gemm_tensor = None
        if STANDARD_QUANTIZE in (self.rowwise_tensor_plan, self.columnwise_tensor_plan):
            quantized_tensor = self.parent_quantizer(tensor)
            if self.rowwise_tensor_plan == STANDARD_QUANTIZE:
                rowwise_gemm_tensor = quantized_tensor
            if self.columnwise_tensor_plan == STANDARD_QUANTIZE:
                columnwise_gemm_tensor = quantized_tensor

        if self.columnwise_tensor_plan == API_CALL_MODIFY:
            columnwise_gemm_tensor = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.columnwise_gemm_name,
                tensor=tensor,
                default_quantizer=self.parent_quantizer,
                iteration=self.iteration,
                dtype=dtype,
            )
            self._check_modified_dtype(columnwise_gemm_tensor, dtype)
        if self.rowwise_tensor_plan == API_CALL_MODIFY:
            rowwise_gemm_tensor = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.rowwise_gemm_name,
                tensor=tensor,
                default_quantizer=self.parent_quantizer,
                iteration=self.iteration,
                dtype=dtype,
            )
            self._check_modified_dtype(rowwise_gemm_tensor, dtype)

        high_precision_tensor = tensor if dtype is None else tensor.to(dtype=dtype)
        if high_precision_tensor.dim() > 2:
            high_precision_tensor = view_as_n_dim(high_precision_tensor)
        if self.rowwise_tensor_plan == HIGH_PRECISION:
            rowwise_gemm_tensor = high_precision_tensor
        if self.columnwise_tensor_plan == HIGH_PRECISION:
            columnwise_gemm_tensor = high_precision_tensor

        self._call_inspect_tensor_api(tensor, rowwise_gemm_tensor, columnwise_gemm_tensor)
        return DebugQuantizedTensor(
            rowwise_gemm_tensor=rowwise_gemm_tensor,
            columnwise_gemm_tensor=columnwise_gemm_tensor,
            quantizer=self,
            layer_name=self.layer_name,
            tensor_name=self.tensor_name,
        )

    @staticmethod
    def _check_modified_dtype(tensor, dtype: Optional[torch.dtype]) -> None:
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError("Dtype does not match the output of the modify_tensor call")

    def quantize_impl(self, tensor: torch.Tensor):
        return self.quantize(tensor)

    def grouped_quantize_impl(self, group_list, tensor: torch.Tensor):
        """Quantize packed grouped data while preserving both GEMM usages."""
        from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

        self._update_parent_quantizer_usage()
        if self.parent_quantizer is not None:
            self.parent_quantizer.columnwise_use_group_quant = self.columnwise_use_group_quant
        rowwise_gemm_tensor = None
        columnwise_gemm_tensor = None
        if STANDARD_QUANTIZE in (self.rowwise_tensor_plan, self.columnwise_tensor_plan):
            quantized_tensor = self.parent_quantizer.grouped_quantize(tensor, group_list)
            if self.rowwise_tensor_plan == STANDARD_QUANTIZE:
                rowwise_gemm_tensor = quantized_tensor
            if self.columnwise_tensor_plan == STANDARD_QUANTIZE:
                columnwise_gemm_tensor = quantized_tensor

        num_tensors = len(group_list)
        high_precision_tensor = tensor
        if tensor.ndim > 2 and tensor.size(0) == num_tensors:
            shapes = [tuple(tensor.shape[1:])] * num_tensors
        else:
            shapes = None
        high_precision_tensor = GroupedTensor.as_dense_grouped_tensor(
            high_precision_tensor,
            num_tensors,
            shapes=shapes,
        )
        if self.rowwise_tensor_plan == HIGH_PRECISION:
            rowwise_gemm_tensor = high_precision_tensor
        if self.columnwise_tensor_plan == HIGH_PRECISION:
            columnwise_gemm_tensor = high_precision_tensor

        self._call_inspect_tensor_api(tensor, rowwise_gemm_tensor, columnwise_gemm_tensor)
        return DebugQuantizedTensor(
            rowwise_gemm_tensor=rowwise_gemm_tensor,
            columnwise_gemm_tensor=columnwise_gemm_tensor,
            quantizer=self,
            layer_name=self.layer_name,
            tensor_name=self.tensor_name,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensorStorage,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ):
        """Update a debug quantized tensor in-place."""
        assert noop_flag is None, "NPU Graphs are not supported with debug=True!"
        assert isinstance(dst, DebugQuantizedTensor)

        import nvdlfw_inspect.api as debug_api

        updated_rowwise_gemm = False
        if self.parent_quantizer is not None:
            if (
                dst.rowwise_gemm_tensor is not None
                and self.rowwise_tensor_plan == STANDARD_QUANTIZE
            ):
                self._update_one_tensor(src, dst.rowwise_gemm_tensor)
                updated_rowwise_gemm = True
            if (
                dst.columnwise_gemm_tensor is not None
                and self.columnwise_tensor_plan == STANDARD_QUANTIZE
                and not updated_rowwise_gemm
            ):
                self._update_one_tensor(src, dst.columnwise_gemm_tensor)

        if self.columnwise_tensor_plan == API_CALL_MODIFY:
            out = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.columnwise_gemm_name,
                tensor=src,
                default_quantizer=self.parent_quantizer,
                out=dst.columnwise_gemm_tensor,
                iteration=self.iteration,
            )
            assert out is None, "modify_tensor with out != None should return None"
        if self.rowwise_tensor_plan == API_CALL_MODIFY:
            out = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.rowwise_gemm_name,
                tensor=src,
                default_quantizer=self.parent_quantizer,
                out=dst.rowwise_gemm_tensor,
                iteration=self.iteration,
            )
            assert out is None, "modify_tensor with out != None should return None"

        if self.rowwise_tensor_plan == HIGH_PRECISION and dst.rowwise_gemm_tensor is not None:
            dst.rowwise_gemm_tensor.copy_(src)
        if self.columnwise_tensor_plan == HIGH_PRECISION and dst.columnwise_gemm_tensor is not None:
            if dst.columnwise_gemm_tensor is not dst.rowwise_gemm_tensor:
                dst.columnwise_gemm_tensor.copy_(src)

        self._call_inspect_tensor_api(src, dst.rowwise_gemm_tensor, dst.columnwise_gemm_tensor)
        return dst

    def update_grouped_quantized(self, src, dst, group_list, *, noop_flag=None):
        """Update packed grouped debug storage in-place."""
        assert noop_flag is None, "NPU Graphs are not supported with debug=True!"
        assert isinstance(dst, DebugQuantizedTensor)

        updated_rowwise_gemm = False
        if self.parent_quantizer is not None:
            if (
                dst.rowwise_gemm_tensor is not None
                and self.rowwise_tensor_plan == STANDARD_QUANTIZE
            ):
                self.parent_quantizer.update_grouped_quantized(
                    src,
                    dst.rowwise_gemm_tensor,
                    group_list,
                )
                updated_rowwise_gemm = True
            if (
                dst.columnwise_gemm_tensor is not None
                and self.columnwise_tensor_plan == STANDARD_QUANTIZE
                and not updated_rowwise_gemm
            ):
                self.parent_quantizer.update_grouped_quantized(
                    src,
                    dst.columnwise_gemm_tensor,
                    group_list,
                )

        if self.rowwise_tensor_plan == HIGH_PRECISION and dst.rowwise_gemm_tensor is not None:
            dst.rowwise_gemm_tensor.rowwise_data.copy_(src)
        if (
            self.columnwise_tensor_plan == HIGH_PRECISION
            and dst.columnwise_gemm_tensor is not None
            and dst.columnwise_gemm_tensor is not dst.rowwise_gemm_tensor
        ):
            dst.columnwise_gemm_tensor.rowwise_data.copy_(src)

        self._call_inspect_tensor_api(src, dst.rowwise_gemm_tensor, dst.columnwise_gemm_tensor)
        return dst

    def _update_one_tensor(self, src: torch.Tensor, dst) -> None:
        if hasattr(dst, "quantize_"):
            dst.quantize_(src)
        else:
            self.parent_quantizer.update_quantized(src, dst, noop_flag=None)

    def process_gemm_output(self, tensor: torch.Tensor):
        """Inspect and optionally modify a GEMM output tensor."""
        assert self.parent_quantizer is None, "Quantized output is not supported for debug=True."
        assert self.output_tensor

        import nvdlfw_inspect.api as debug_api

        tensor_to_gemm = {"output": "fprop", "wgrad": "wgrad", "dgrad": "dgrad"}
        if self.rowwise_tensor_plan == API_CALL_MODIFY:
            tensor = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                gemm=tensor_to_gemm[self.tensor_name],
                tensor_name=self.tensor_name,
                tensor=tensor,
                iteration=self.iteration,
                default_quantizer=self.parent_quantizer,
            )
        self._call_inspect_tensor_api(tensor)
        return tensor

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        if self.parent_quantizer is not None:
            return self.parent_quantizer.make_empty(shape, dtype=dtype, device=device)
        return torch.empty(shape, dtype=dtype, device=device)

    def calibrate(self, tensor: torch.Tensor):
        """Calibration override; debug quantization does not support it."""
        raise RuntimeError("[TPUTORCH-INSPECT ERROR] Calibration with debug is not supported")

    def supports_only_rowwise_all_gather(self) -> bool:
        if self.parent_quantizer is not None:
            return self.parent_quantizer.supports_only_rowwise_all_gather()
        return False

    def _get_compatible_recipe(self):
        return None

    def _call_inspect_tensor_api(
        self,
        tensor,
        rowwise_gemm_tensor=None,
        columnwise_gemm_tensor=None,
    ) -> None:
        import nvdlfw_inspect.api as debug_api

        args = {
            "layer_name": self.layer_name,
            "tensor_name": self.tensor_name,
            "tensor": tensor,
            "iteration": TEDebugState.get_iteration(),
            "tp_group": self.tp_group,
            "tp_size": self.tp_size,
            "rowwise_quantized_tensor": rowwise_gemm_tensor,
            "columnwise_quantized_tensor": columnwise_gemm_tensor,
            "quantizer": self.parent_quantizer,
        }
        if self.inspect_tensor_enabled:
            debug_api.transformer_engine.inspect_tensor(**args)

        if self.output_tensor:
            return

        del args["rowwise_quantized_tensor"]
        del args["columnwise_quantized_tensor"]
        del args["quantizer"]

        if (
            self.rowwise_tensor_plan in {API_CALL_MODIFY, STANDARD_QUANTIZE}
            and self.inspect_tensor_postquantize_enabled_rowwise
        ):
            args["tensor"] = rowwise_gemm_tensor
            args["rowwise"] = True
            debug_api.transformer_engine.inspect_tensor_postquantize(**args)

        if (
            self.columnwise_tensor_plan in {API_CALL_MODIFY, STANDARD_QUANTIZE}
            and self.inspect_tensor_postquantize_enabled_columnwise
        ):
            args["tensor"] = columnwise_gemm_tensor
            args["rowwise"] = False
            debug_api.transformer_engine.inspect_tensor_postquantize(**args)

    def wrap_quantized_tensor(self, tensor: QuantizedTensor):
        """Wrap an existing quantized tensor, used for primary FP8 weights."""
        assert (
            self.rowwise_tensor_plan == STANDARD_QUANTIZE
            and self.columnwise_tensor_plan == STANDARD_QUANTIZE
        ), (
            "[TPUTORCH INSPECT ERROR] Weight tensor with fp8 model parameters enabled cannot be "
            "modified by any feature."
        )
        self._call_inspect_tensor_api(None, tensor, tensor)
        return DebugQuantizedTensor(
            rowwise_gemm_tensor=tensor,
            columnwise_gemm_tensor=tensor,
            quantizer=self,
            layer_name=self.layer_name,
            tensor_name=self.tensor_name,
        )

    @classmethod
    def multi_tensor_quantize(
        cls,
        tensor: torch.Tensor,
        quantizers: list[Quantizer],
        m_splits: list[int],
        activation_dtype: torch.dtype,
    ) -> list["DebugQuantizedTensor"]:
        """Split a tensor and quantize each split with its matching quantizer."""
        tensors = torch.split(tensor, m_splits)
        return [
            quantizer.quantize(split, dtype=activation_dtype)
            for split, quantizer in zip(tensors, quantizers)
        ]


class DebugQuantizedTensor(QuantizedTensorStorage):
    """Container with separate tensors for rowwise and columnwise GEMM use."""

    def __init__(
        self,
        rowwise_gemm_tensor,
        columnwise_gemm_tensor,
        quantizer: DebugQuantizer,
        layer_name=None,
        tensor_name=None,
    ):
        self.rowwise_gemm_tensor = rowwise_gemm_tensor
        self.columnwise_gemm_tensor = columnwise_gemm_tensor
        self.quantizer = quantizer
        self._layer_name = layer_name
        self._tensor_name = tensor_name
        self.tensors_to_save = None

    def prepare_for_saving(self):
        self.tensors_to_save = (
            [self.rowwise_gemm_tensor, self.columnwise_gemm_tensor]
            if self.rowwise_gemm_tensor is not self.columnwise_gemm_tensor
            else [self.rowwise_gemm_tensor]
        )
        tensor_list, tensor_objects_list = prepare_for_saving(*self.tensors_to_save)
        self.tensors_to_save = tensor_objects_list
        return tensor_list, self

    def restore_from_saved(self, tensors):
        tensor_objects_list, saved_tensors = restore_from_saved(
            self.tensors_to_save,
            tensors,
            return_saved_tensors=True,
        )
        if len(tensor_objects_list) == 2:
            self.rowwise_gemm_tensor, self.columnwise_gemm_tensor = tensor_objects_list  # pylint: disable=unbalanced-tuple-unpacking
        else:
            self.rowwise_gemm_tensor = tensor_objects_list[0]
            self.columnwise_gemm_tensor = self.rowwise_gemm_tensor
        return saved_tensors

    def quantize_(self, tensor, *, noop_flag=None):
        assert noop_flag is None, "NPU Graphs are not supported with debug=True!"
        return self.quantizer.update_quantized(tensor, self)

    def dequantize(self, *, dtype=None):
        tensor = self.rowwise_gemm_tensor
        if dtype is None:
            dtype = tensor.dtype
        return tensor.dequantize().to(dtype)

    def get_tensor(self, transpose: bool):
        return self.columnwise_gemm_tensor if transpose else self.rowwise_gemm_tensor

    def get_tensor_for_usage(self, usage):
        usage = TensorUsage(usage)
        tensor_name = self._tensor_name
        use_columnwise = False
        if tensor_name == "activation":
            use_columnwise = usage == TensorUsage.RHS
        elif tensor_name == "gradient":
            use_columnwise = usage == TensorUsage.LHS_TRANS
        elif tensor_name == "weight":
            use_columnwise = usage == TensorUsage.RHS
        else:
            use_columnwise = usage in {TensorUsage.LHS_TRANS, TensorUsage.RHS_TRANS}
        return self.columnwise_gemm_tensor if use_columnwise else self.rowwise_gemm_tensor

    def get_data(self, usage):
        return self.get_tensor_for_usage(usage)

    def size(self, *args):
        tensor = self.rowwise_gemm_tensor
        if tensor is None:
            tensor = self.columnwise_gemm_tensor
        return tensor.size(*args)

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        if not isinstance(src, DebugQuantizedTensor):
            raise TypeError("copy_from_storage expects DebugQuantizedTensor")
        if self.rowwise_gemm_tensor is not None and src.rowwise_gemm_tensor is not None:
            self.rowwise_gemm_tensor.copy_(src.rowwise_gemm_tensor)
        if (
            self.columnwise_gemm_tensor is not None
            and src.columnwise_gemm_tensor is not None
            and self.columnwise_gemm_tensor is not self.rowwise_gemm_tensor
        ):
            self.columnwise_gemm_tensor.copy_(src.columnwise_gemm_tensor)

    def update_usage(self, rowwise_usage: bool = None, columnwise_usage: bool = None):
        if self.rowwise_gemm_tensor is not self.columnwise_gemm_tensor:
            if rowwise_usage is False:
                self.rowwise_gemm_tensor = None
            if columnwise_usage is False:
                self.columnwise_gemm_tensor = None

        if isinstance(self.rowwise_gemm_tensor, QuantizedTensor):
            self.rowwise_gemm_tensor.update_usage(rowwise_usage, columnwise_usage)
        if isinstance(self.columnwise_gemm_tensor, QuantizedTensor):
            self.columnwise_gemm_tensor.update_usage(rowwise_usage, columnwise_usage)

        if rowwise_usage and self.rowwise_gemm_tensor is None:
            raise RuntimeError(
                "Cannot recreate rowwise tensor from columnwise tensor in debug mode."
            )
        if columnwise_usage and self.columnwise_gemm_tensor is None:
            raise RuntimeError(
                "Cannot recreate columnwise tensor from rowwise tensor in debug mode."
            )

    @property
    def device(self):
        if self.rowwise_gemm_tensor is not None:
            return self.rowwise_gemm_tensor.device
        if self.columnwise_gemm_tensor is not None:
            return self.columnwise_gemm_tensor.device
        raise RuntimeError("DebugQuantizedTensor has no data!")


class GroupedDebugQuantizer(Quantizer):
    """Bundle per-GEMM debug quantizers behind the grouped quantizer interface."""

    def __init__(self, quantizers: list[DebugQuantizer], m_splits: list[int]):
        super().__init__(rowwise=True, columnwise=True)
        self.quantizers = quantizers
        self.m_splits = m_splits

    def set_usage(self, *, rowwise=None, columnwise=None) -> None:
        super().set_usage(rowwise=rowwise, columnwise=columnwise)
        for quantizer in self.quantizers:
            quantizer.set_usage(rowwise=rowwise, columnwise=columnwise)

    def quantize(self, tensor, *, out=None, dtype=None):
        if out is not None:
            return out.quantize_(tensor)
        if tensor.size(0) != len(self.quantizers):
            raise ValueError("Grouped debug weight must have one tensor per GEMM")
        return GroupedDebugQuantizedTensor(
            [
                quantizer.quantize(weight, dtype=dtype)
                for weight, quantizer in zip(tensor.unbind(0), self.quantizers)
            ],
            self,
        )

    def grouped_quantize(self, tensor, _group_list, *, out=None, dtype=None):
        if out is not None:
            return out.grouped_quantize_(tensor, _group_list)
        if tensor.dim() == 3 and tensor.size(0) == len(self.quantizers):
            return self.quantize(tensor, dtype=dtype)
        return GroupedDebugQuantizedTensor(
            DebugQuantizer.multi_tensor_quantize(
                tensor,
                self.quantizers,
                self.m_splits,
                tensor.dtype if dtype is None else dtype,
            ),
            self,
        )

    def get_next_debug_iter(self):
        iterations = [
            quantizer.get_next_debug_iter()
            for quantizer in self.quantizers
            if quantizer.get_next_debug_iter() is not None
        ]
        return min(iterations) if iterations else None

    def any_feature_enabled(self) -> bool:
        return any(quantizer.any_feature_enabled() for quantizer in self.quantizers)

    def calibrate(self, _tensor):
        raise RuntimeError("[TPUTORCH-INSPECT ERROR] Calibration with debug is not supported")


class GroupedDebugQuantizedTensor(QuantizedTensorStorage):
    """Grouped storage for per-GEMM debug tensors."""

    def __init__(self, tensors: list[DebugQuantizedTensor], quantizer: GroupedDebugQuantizer):
        self.tensors = tensors
        self.quantizer = quantizer
        self.tensor_objects = None

    def get_data(self, usage):
        return [tensor.get_data(usage) for tensor in self.tensors]

    def update_usage(self, rowwise_usage=None, columnwise_usage=None):
        for tensor in self.tensors:
            tensor.update_usage(rowwise_usage, columnwise_usage)

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        if not isinstance(src, GroupedDebugQuantizedTensor):
            raise TypeError("copy_from_storage expects GroupedDebugQuantizedTensor")
        if len(self.tensors) != len(src.tensors):
            raise ValueError("Grouped debug tensors must have the same number of GEMMs")
        for dst_tensor, src_tensor in zip(self.tensors, src.tensors):
            dst_tensor.copy_from_storage(src_tensor)

    def prepare_for_saving(self):
        tensors, self.tensor_objects = prepare_for_saving(*self.tensors)
        return tensors, self

    def restore_from_saved(self, tensors):
        self.tensors, tensors = restore_from_saved(
            self.tensor_objects,
            tensors,
            return_saved_tensors=True,
        )
        self.tensor_objects = None
        return tensors

    def quantize_(self, tensor, *, noop_flag=None):
        assert noop_flag is None, "NPU Graphs are not supported with debug=True!"
        self.tensors = self.quantizer.quantize(tensor).tensors
        return self

    def grouped_quantize_(self, tensor, group_list, *, noop_flag=None):
        assert noop_flag is None, "NPU Graphs are not supported with debug=True!"
        self.tensors = self.quantizer.grouped_quantize(tensor, group_list).tensors
        return self
