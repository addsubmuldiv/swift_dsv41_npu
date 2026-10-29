# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.


"""FusedAdam implementation for Ascend NPU."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from typing import Optional

import torch
import torch_npu
from torch.distributed._tensor import DTensor

from ..quantized_tensor import QuantizedTensor
from ..tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer, Float8Tensor


class FusedAdam(torch.optim.Optimizer):
    """Adam/AdamW optimizer with the Transformer Engine FusedAdam interface.

    This NPU implementation intentionally does not depend on NVIDIA's
    ``transformer_engine_torch`` CUDA extension. It uses NPU fused updates and
    a Triton-Ascend delayed-FP8 path while keeping the public constructor and
    ``step`` signatures compatible with TE/Apex FusedAdam.

    DTensor parameters use TE-style local updates and distributed state storage
    for MindSpeed FSDP. Parameters and gradients must have the same reduced
    layout. Floating-point states and master weights are supported; DTensor
    FP8 storage and INT16 parameter remainders are not supported.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter | dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        *,
        bias_correction=True,
        adam_w_mode=True,
        capturable=False,
        master_weights=False,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        use_decoupled_grad=False,
        store_param_remainders=False,
        set_grad_none: Optional[bool] = None,
    ):
        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        lr_value = self._scalar_to_float(lr, "lr")
        eps_value = self._scalar_to_float(eps, "eps")
        weight_decay_value = self._scalar_to_float(weight_decay, "weight_decay")
        beta1_value = self._scalar_to_float(betas[0], "beta1")
        beta2_value = self._scalar_to_float(betas[1], "beta2")
        if lr_value < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps_value < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1_value < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= beta2_value < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay_value < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not bias_correction:
            raise NotImplementedError(
                "NPU FusedAdam requires bias_correction=True because "
                "torch._fused_adamw_ does not support disabling bias correction."
            )

        if capturable:
            raise NotImplementedError(
                "NPU FusedAdam implementation does not support capturable=True yet."
            )
        if master_weights and master_weight_dtype not in (torch.float32, torch.float16):
            raise NotImplementedError(
                "NPU FusedAdam implementation currently supports only "
                "master_weight_dtype=torch.float32 or torch.float16."
            )
        self._check_state_dtype(exp_avg_dtype, "exp_avg_dtype")
        self._check_state_dtype(exp_avg_sq_dtype, "exp_avg_sq_dtype")
        if torch.uint8 in (exp_avg_dtype, exp_avg_sq_dtype) and not self._float8_state_available():
            raise NotImplementedError(
                "NPU FusedAdam implementation requires TransformerEngineNPU Float8Tensor, "
                "torch_npu.npu_dynamic_quant, and torch.float8_e4m3fn support when "
                "exp_avg_dtype or exp_avg_sq_dtype is torch.uint8."
            )
        store_param_remainders = store_param_remainders and master_weights
        if store_param_remainders and master_weight_dtype != torch.float32:
            raise RuntimeError(
                "store_param_remainders=True requires master_weight_dtype=torch.float32."
            )
        defaults = {
            "lr": lr_value,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self.adam_w_mode = 1 if adam_w_mode else 0
        self.capturable = capturable
        self.master_weights = master_weights
        self.master_weight_dtype = master_weight_dtype
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_sq_dtype
        self.use_decoupled_grad = use_decoupled_grad
        self.store_param_remainders = store_param_remainders
        self.dtype_to_range_map = {
            torch.float16: torch.finfo(torch.float16).max / 2.0,
            torch.uint8: 448.0,
        }
        self._scales = {}
        self._state_quantizers = {}
        self._step_tensors = {}
        self._fp8_adam_metadata_cache = {}
        self._remainder_metadata_cache = {}
        for group in self.param_groups:
            if self.store_param_remainders and any(
                not isinstance(param, QuantizedTensor) and param.dtype != torch.bfloat16
                for param in group["params"]
            ):
                raise RuntimeError(
                    "store_param_remainders=True supports only BF16 model parameters."
                )
            group["step"] = self._step_to_int(group.get("step", 0))
            if group["params"]:
                self._get_group_step_tensor(group)
        self.fp8_exp_avg_sq_storage = os.getenv(
            "TE_NPU_FUSED_ADAM_FP8_EXP_AVG_SQ_STORAGE",
            "sqrt",
        ).lower()

        self.set_grad_none = set_grad_none
        if self.set_grad_none is not None:
            warnings.warn(
                "set_grad_none kwarg in FusedAdam constructor is deprecated. "
                "Use set_to_none kwarg in zero_grad instead.",
                DeprecationWarning,
            )

    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        """Reset parameter gradients.

        If ``set_to_none`` is omitted, use the deprecated constructor option
        ``set_grad_none`` when provided; otherwise match TE FusedAdam's default
        of setting gradients to ``None``.
        """

        if self.set_grad_none is not None:
            if set_to_none is not None and set_to_none != self.set_grad_none:
                raise ValueError(
                    f"Called zero_grad with set_to_none={set_to_none}, "
                    "but FusedAdam was initialized with "
                    f"set_grad_none={self.set_grad_none}"
                )
            set_to_none = self.set_grad_none
        if set_to_none is None:
            set_to_none = True

        if not self.use_decoupled_grad:
            super().zero_grad(set_to_none=set_to_none)
            return

        for group in self.param_groups:
            for param in group["params"]:
                if set_to_none:
                    param.decoupled_grad = None
                else:
                    decoupled_grad = getattr(param, "decoupled_grad", None)
                    if decoupled_grad is not None:
                        decoupled_grad.zero_()

    def initialize_state(self, param, store_param_remainders=None) -> None:
        """Initialize optimizer states for Megatron precision-aware optimizers."""

        # Keep Megatron 0.18's single-argument call compatible; TE also accepts
        # the explicit store_param_remainders argument.

        use_remainder = self._stores_param_remainder(param)
        if store_param_remainders is not None and bool(store_param_remainders) != use_remainder:
            raise RuntimeError(
                "initialize_state store_param_remainders must match the optimizer configuration."
            )
        self._check_unsupported_param_type(param)
        self._check_param_dtype(param)
        self._init_or_fix_state(param)

    def state_dict(self):
        """Return FP32 optimizer states, except persistent INT16 parameter remainders."""

        state_dict = super().state_dict()

        for group, packed_group in zip(self.param_groups, state_dict["param_groups"]):
            packed_group["step"] = self._step_to_int(group.get("step", 0))
            for param, param_id in zip(group["params"], packed_group["params"]):
                packed_state = state_dict["state"].get(param_id)
                param_state = self.state.get(param)
                if not isinstance(packed_state, dict) or not isinstance(
                    param_state,
                    dict,
                ):
                    continue
                packed_state = dict(packed_state)
                state_dict["state"][param_id] = packed_state
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in param_state:
                        packed_state[name] = self._wrap_state(
                            param, self.get_unscaled_state(param, name)
                        )
                if "master_param" in param_state:
                    packed_state["master_param"] = self._wrap_state(
                        param, self.get_unscaled_state(param, "master_param")
                    )

        return state_dict

    def load_state_dict(self, state_dict):
        saved_to_current_params = {}
        for saved_group, current_group in zip(
            state_dict["param_groups"],
            self.param_groups,
        ):
            for saved_param_id, param in zip(
                saved_group["params"],
                current_group["params"],
            ):
                saved_to_current_params[saved_param_id] = param

        saved_state_tensors = {}
        plain_uint8_placeholders = {}
        for param_id, param in saved_to_current_params.items():
            saved_state = state_dict["state"].get(param_id)
            if not isinstance(saved_state, dict):
                continue
            for name in ("exp_avg", "exp_avg_sq", "master_param"):
                value = saved_state.get(name)
                if self._is_float8_state(value):
                    saved_state_tensors[(param, name)] = value.dequantize(
                        dtype=torch.float32
                    ).detach()
                elif torch.is_tensor(value):
                    if (
                        name in ("exp_avg", "exp_avg_sq")
                        and value.dtype == torch.uint8
                        and self._state_dtype_for_name(name) == torch.uint8
                    ):
                        # Megatron DistributedOptimizer first loads empty state tensors
                        # in the configured dtype, then restores the real parameter state.
                        # Keep a plain uint8 placeholder for that second loading stage.
                        # The base Optimizer.load_state_dict casts state tensors to the
                        # param dtype, so the placeholder must be restored afterwards.
                        plain_uint8_placeholders[(param, name)] = value.detach().clone()
                        continue
                    if name == "master_param" and not value.is_floating_point():
                        if not (self._stores_param_remainder(param) and value.dtype == torch.int16):
                            raise RuntimeError(
                                "An INT16 master_param checkpoint requires "
                                "store_param_remainders=True."
                            )
                    saved_state_tensors[(param, name)] = value.detach().clone()

        result = super().load_state_dict(state_dict)
        self._scales = {}
        self._state_quantizers = {}
        self._step_tensors = {}
        self._fp8_adam_metadata_cache = {}
        for group in self.param_groups:
            group["step"] = self._step_to_int(group.get("step", 0))
            if group["params"]:
                self._get_group_step_tensor(group)

        for param in saved_to_current_params.values():
            state = self.state.get(param)
            if not isinstance(state, dict):
                continue

            for name in ("exp_avg", "exp_avg_sq"):
                placeholder = plain_uint8_placeholders.get((param, name))
                if placeholder is not None:
                    state[name] = placeholder.to(device=param.device)
                    continue
                value = saved_state_tensors.get((param, name))
                if value is not None:
                    self.set_scaled_state(param, name, value)

            master_param = saved_state_tensors.get((param, "master_param"))
            if self.master_weights and (
                self._is_quantized_model_param(param)
                or param.dtype in (torch.float16, torch.bfloat16)
            ):
                if master_param is None:
                    master_param = (
                        torch.zeros_like(param, dtype=torch.int16)
                        if self._stores_param_remainder(param)
                        else (
                            param.dequantize(dtype=torch.float32).detach()
                            if self._is_quantized_model_param(param)
                            else param.detach().float()
                        )
                    )
                self.set_scaled_state(param, "master_param", master_param)
            else:
                state.pop("master_param", None)
                self._scale_dict(param).pop("master_param", None)
        return result

    @staticmethod
    def _scalar_to_float(value, name: str) -> float:
        """Convert a scalar optimizer hyperparameter to a Python float."""

        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Optimizer parameter {name} must be a scalar.")
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _step_to_int(step) -> int:
        """Convert a scalar optimizer step to a Python int."""

        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError("Optimizer step must be a scalar.")
            return int(step.detach().item())
        return int(step)

    @staticmethod
    def _has_sparse_grad(grad: torch.Tensor) -> bool:
        """Return True for COO or compressed sparse gradients."""

        if grad.is_sparse:
            return True
        is_sparse_csr = getattr(grad, "is_sparse_csr", False)
        if callable(is_sparse_csr):
            is_sparse_csr = is_sparse_csr()
        return bool(is_sparse_csr)

    @staticmethod
    def _check_param_dtype(param: torch.Tensor) -> None:
        """Validate parameter dtype supported by this fallback."""

        supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if param.dtype not in supported_dtypes:
            raise RuntimeError(
                "FusedAdam fallback supports parameters in torch.float32, "
                "torch.float16, and torch.bfloat16 only, but got "
                f"{param.dtype}."
            )

    @staticmethod
    def _check_state_dtype(dtype: torch.dtype, name: str) -> None:
        """Validate optimizer state dtype supported by this fallback."""

        supported_dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.uint8)
        if dtype not in supported_dtypes:
            raise NotImplementedError(
                "NPU FusedAdam fallback supports "
                f"{name}=torch.float32, torch.float16, torch.bfloat16, "
                f"or torch.uint8 only; "
                f"got {dtype}."
            )

    @staticmethod
    def _is_dtensor_param(param) -> bool:
        """Return whether a parameter is a DTensor model weight."""

        return isinstance(param, DTensor)

    @staticmethod
    def _local_tensor(tensor):
        """Expose local storage to optimizer kernels, as in TE FusedAdam."""
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    @staticmethod
    def _wrap_state(param, value):
        """Preserve the model shard layout for distributed checkpointing."""
        if not isinstance(param, DTensor):
            return value
        return DTensor.from_local(
            value,
            device_mesh=param.device_mesh,
            placements=param.placements,
            shape=param.shape,
            stride=param.stride(),
            run_check=False,
        )

    @staticmethod
    def _is_quantized_model_param(param) -> bool:
        """Return whether a parameter is a TE QuantizedTensor model weight."""

        return isinstance(param, QuantizedTensor)

    def _stores_param_remainder(self, param: torch.Tensor) -> bool:
        """Return whether a parameter stores an INT16 master-weight remainder."""

        return (
            self.store_param_remainders
            and not self._is_quantized_model_param(param)
            and param.dtype == torch.bfloat16
        )

    def _check_unsupported_param_type(self, param: torch.Tensor) -> None:
        """Reject advanced model weight types that the fallback cannot update."""

        if self._is_dtensor_param(param):
            if (
                isinstance(self._local_tensor(param), QuantizedTensor)
                or torch.uint8
                in (
                    self.exp_avg_dtype,
                    self.exp_avg_sq_dtype,
                )
                or self._stores_param_remainder(param)
            ):
                raise NotImplementedError(
                    "DTensor FusedAdam supports unquantized parameters and floating-point "
                    "optimizer states; FP8 and parameter remainders are not supported."
                )
            if any(placement.is_partial() for placement in param.placements):
                raise RuntimeError("FusedAdam requires reduced DTensor parameters.")
        if self._is_quantized_model_param(param):
            if not self.master_weights:
                raise RuntimeError(
                    f"{type(param).__name__} model parameters require master_weights=True."
                )
            if self.master_weight_dtype != torch.float32:
                raise RuntimeError(
                    "QuantizedTensor model parameters require master_weight_dtype=torch.float32."
                )

    def _check_param_group_supported(self, group: dict) -> None:
        """Reject unsupported parameter combinations in a param group."""

        has_fp16 = False
        has_bf16 = False
        for param in group["params"]:
            self._check_unsupported_param_type(param)
            if self._is_quantized_model_param(param):
                continue
            has_fp16 = has_fp16 or param.dtype == torch.float16
            has_bf16 = has_bf16 or param.dtype == torch.bfloat16
        if has_fp16 and has_bf16:
            raise RuntimeError(
                "FusedAdam fallback does not support mixing torch.float16 and "
                "torch.bfloat16 parameters in the same param group."
            )

    @staticmethod
    def _float8_state_available() -> bool:
        """Return whether the NPU Float8Tensor state path can be used."""

        if not hasattr(torch_npu, "npu_dynamic_quant"):
            return False
        return hasattr(torch, "float8_e4m3fn") and torch.float8_e4m3fn != torch.bfloat16

    @staticmethod
    def _is_float8_state(value) -> bool:
        """Return whether a state value is a TransformerEngineNPU Float8Tensor."""

        return isinstance(value, Float8Tensor)

    @staticmethod
    def _float8_state_device(value) -> Optional[torch.device]:
        """Return the device backing a TransformerEngineNPU Float8Tensor state."""

        if not FusedAdam._is_float8_state(value):
            return None
        try:
            device = torch.device(value.device)
        except (AttributeError, TypeError, RuntimeError):
            device = None
        else:
            return device
        for attr_name in ("_data", "_transpose"):
            tensor = getattr(value, attr_name, None)
            if torch.is_tensor(tensor):
                return tensor.device
        return None

    @staticmethod
    def _float8_quantizer_device(quantizer) -> Optional[torch.device]:
        """Return the device backing a Float8 optimizer-state quantizer."""

        for attr_name in ("scale", "amax"):
            tensor = getattr(quantizer, attr_name, None)
            if torch.is_tensor(tensor):
                return tensor.device
        return None

    @staticmethod
    def _devices_match(left: torch.device, right: torch.device) -> bool:
        """Compare devices while treating implicit device index as compatible."""

        left = torch.device(left)
        right = torch.device(right)
        if left == right:
            return True
        return left.type == right.type and (left.index is None or right.index is None)

    def _state_dtype_for_name(self, name: str) -> torch.dtype:
        """Return configured storage dtype for an optimizer state tensor."""

        if name == "exp_avg":
            return self.exp_avg_dtype
        if name == "exp_avg_sq":
            return self.exp_avg_sq_dtype
        if name == "master_param":
            return self.master_weight_dtype
        raise KeyError(f"Unknown optimizer state name: {name}")

    def _scale_dict(self, param: torch.Tensor) -> dict:
        """Return per-state scale dictionary for a parameter."""

        return self._scales.setdefault(param, {})

    def _quantizer_dict(self, param: torch.Tensor) -> dict:
        """Return per-state FP8 quantizer dictionary for a parameter."""

        return self._state_quantizers.setdefault(param, {})

    def _state_quantizer(self, param: torch.Tensor, state_name: str):
        """Return the Float8CurrentScalingQuantizer for an FP8 optimizer state."""

        if not self._float8_state_available():
            raise NotImplementedError(
                "FP8 optimizer state requires TransformerEngineNPU Float8Tensor, "
                "torch_npu.npu_dynamic_quant, and torch.float8_e4m3fn support."
            )

        quantizers = self._quantizer_dict(param)
        quantizer = quantizers.get(state_name)
        if quantizer is not None:
            quantizer_device = self._float8_quantizer_device(quantizer)
            if quantizer_device is None or not self._devices_match(
                quantizer_device,
                param.device,
            ):
                quantizer = None

        if quantizer is None:
            quantizer = Float8CurrentScalingQuantizer(
                fp8_dtype=torch.float8_e4m3fn,
                rowwise=True,
                columnwise=False,
                device=param.device,
            )
            quantizers[state_name] = quantizer
        return quantizer

    @staticmethod
    def _fp8_quant_input_dtype(param: torch.Tensor) -> torch.dtype:
        """Return dense input dtype accepted by NPU FP8 dynamic quantization."""

        if param.dtype in (torch.float16, torch.bfloat16):
            return param.dtype
        return torch.bfloat16

    def _to_fp8_quant_input(
        self,
        param: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Cast FP32 optimizer state to the dtype accepted by the FP8 quantizer."""

        return value.to(
            device=param.device,
            dtype=self._fp8_quant_input_dtype(param),
            non_blocking=True,
        )

    def _state_to_fp8_storage_value(
        self,
        state_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Map an fp32 optimizer state to the value stored in FP8."""

        if (
            state_name == "exp_avg_sq"
            and self.exp_avg_sq_dtype == torch.uint8
            and self.fp8_exp_avg_sq_storage == "sqrt"
        ):
            # NPU-specific internal representation: storing sqrt(exp_avg_sq)
            # reduces denominator drift while preserving the public fp32
            # exp_avg_sq semantics through get_unscaled_state/state_dict.
            return value.clamp_min(0.0).sqrt()
        return value

    def _state_from_fp8_storage_value(
        self,
        state_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Map a dequantized FP8 storage value back to optimizer-state semantics."""

        if (
            state_name == "exp_avg_sq"
            and self.exp_avg_sq_dtype == torch.uint8
            and self.fp8_exp_avg_sq_storage == "sqrt"
        ):
            value = value.clamp_min(0.0)
            return value.mul(value).clamp_min(torch.finfo(torch.float32).tiny)
        return value

    def _set_unit_state_scale(self, param: torch.Tensor, state_name: str) -> None:
        """Record a no-op fp32 scale for states that are stored directly."""

        scales = self._scale_dict(param)
        scale = scales.get(state_name)
        if scale is None or scale.dtype != torch.float32 or scale.device != param.device:
            scales[state_name] = torch.ones(
                (),
                dtype=torch.float32,
                device=param.device,
            )

    def _ensure_state_scale(self, param: torch.Tensor, state_name: str) -> torch.Tensor:
        """Return the fp32 scalar scale for a state tensor."""

        scales = self._scale_dict(param)
        scale = scales.get(state_name)
        if scale is None:
            scale = torch.ones((), dtype=torch.float32, device=param.device)
            scales[state_name] = scale
        elif scale.device != param.device:
            scale = scale.detach().to(device=param.device, dtype=torch.float32)
            scales[state_name] = scale
        return scale

    def _require_state_scale(
        self,
        param: torch.Tensor,
        state_name: str,
    ) -> torch.Tensor:
        """Return an existing state scale, preserving it across device moves."""

        scales = self._scale_dict(param)
        scale = scales.get(state_name)
        if scale is None:
            raise RuntimeError(
                f"Missing scale for low-precision optimizer state {state_name}. "
                "Cannot safely rebuild scaled optimizer state."
            )
        if scale.device != param.device:
            scale = scale.detach().to(device=param.device, dtype=torch.float32)
            scales[state_name] = scale
        return scale

    def _get_scaled_state_as_unscaled(
        self,
        param: torch.Tensor,
        state_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Return a stored low-precision state as unscaled fp32 data."""

        value = self._local_tensor(value)

        if self._is_float8_state(value):
            scale = self._require_state_scale(param, state_name)
            storage_value = (
                value.dequantize(dtype=torch.float32)
                .detach()
                .float()
                .to(
                    device=param.device,
                )
                .mul(scale)
            )
            return self._state_from_fp8_storage_value(state_name, storage_value)

        if value.dtype == torch.float32:
            return value.detach().to(device=param.device, dtype=torch.float32)
        if value.dtype == torch.bfloat16:
            return value.detach().to(device=param.device, dtype=torch.float32)

        scale = self._require_state_scale(param, state_name)
        return value.detach().to(device=param.device, dtype=torch.float32).mul(scale)

    def _compute_state_scale(
        self,
        unscaled_state: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute a scalar fp32 scale for low-precision state storage."""

        if dtype in (torch.float32, torch.bfloat16) or unscaled_state.numel() == 0:
            return torch.ones((), dtype=torch.float32, device=unscaled_state.device)

        max_abs = unscaled_state.detach().abs().max()
        if (not torch.isfinite(max_abs)) or max_abs.item() == 0.0:
            return torch.ones((), dtype=torch.float32, device=unscaled_state.device)

        storage_amax = self.dtype_to_range_map.get(dtype)
        if storage_amax is None:
            storage_amax = torch.finfo(dtype).max * 0.5
        scale = (max_abs / storage_amax).to(dtype=torch.float32)
        return scale.clamp_min(torch.finfo(torch.float32).tiny)

    def get_unscaled_state(
        self, param: torch.nn.Parameter, state_name: str, skip_unscale: bool = False
    ) -> torch.Tensor:
        """Return unscaled state, preserving an INT16 parameter remainder."""

        state = self.state[param]
        value = self._local_tensor(state[state_name])
        if skip_unscale and value.dtype == torch.bfloat16:
            return value
        if self._stores_param_remainder(param) and state_name == "master_param":
            if value.dtype != torch.int16:
                raise RuntimeError("Parameter remainder state must use torch.int16.")
            return value.detach().clone()
        if self._is_float8_state(value):
            return self._get_scaled_state_as_unscaled(param, state_name, value)
        if value.dtype == torch.float32:
            return value.detach()
        if value.dtype == torch.bfloat16:
            self._set_unit_state_scale(param, state_name)
            return value.detach().float()

        return self._get_scaled_state_as_unscaled(param, state_name, value)

    def set_scaled_state(self, param, state_name, unscaled_state):
        """Store fp32 state in the configured dtype with TE-style scaling."""

        unscaled_state = self._local_tensor(unscaled_state)

        if state_name in ("exp_avg", "exp_avg_sq") and unscaled_state.dtype == torch.uint8:
            raise RuntimeError(
                "legacy plain uint8 FP8 optimizer state is not supported. "
                "Please load from an FP32 unscaled state_dict, or regenerate the checkpoint."
            )

        if self._stores_param_remainder(param) and state_name == "master_param":
            if unscaled_state.dtype != torch.int16:
                # Megatron DP-zero transports all coalesced optimizer state through
                # FP32 CPU buffers, including an otherwise-INT16 remainder.
                if unscaled_state.dtype != torch.float32 or unscaled_state.device.type != "cpu":
                    raise RuntimeError("Parameter remainder state must use torch.int16.")
                remainder = unscaled_state.to(dtype=torch.int16)
                if not torch.equal(unscaled_state, remainder.to(dtype=torch.float32)):
                    raise RuntimeError(
                        "FP32 master_param cannot be losslessly restored as an INT16 "
                        "parameter remainder. Check store_param_remainders configuration."
                    )
                unscaled_state = remainder
            self.state[param][state_name] = unscaled_state.detach().to(
                device=param.device,
                dtype=torch.int16,
            )
            return

        state_dtype = self._state_dtype_for_name(state_name)
        value = unscaled_state.detach().to(
            device=param.device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if state_dtype in (torch.float32, torch.bfloat16):
            self.state[param][state_name] = self._wrap_state(param, value.to(dtype=state_dtype))
            self._set_unit_state_scale(param, state_name)
            return
        if state_dtype == torch.uint8:
            quantizer = self._state_quantizer(param, state_name)
            old_state = self.state[param].get(state_name)
            storage_value = self._state_to_fp8_storage_value(state_name, value)
            scale = self._compute_state_scale(storage_value, state_dtype)
            self._scale_dict(param)[state_name] = scale
            scaled_value = storage_value.div(scale)
            quant_input = self._to_fp8_quant_input(param, scaled_value)
            with torch.no_grad():
                if self._is_float8_state(old_state):
                    fp8_state = quantizer.update_quantized(
                        quant_input,
                        old_state,
                    )
                else:
                    fp8_state = quantizer(quant_input)
                if fp8_state._data is not None:
                    fp8_state._data = fp8_state._data.reshape(quant_input.shape)
                fp8_state.origin_shape = quant_input.shape
                self.state[param][state_name] = fp8_state
            return

        scale = self._compute_state_scale(value, state_dtype)
        self._scale_dict(param)[state_name] = scale
        scaled_value = value.div(scale)
        self.state[param][state_name] = self._wrap_state(param, scaled_value.to(dtype=state_dtype))

    def _init_or_fix_state(self, param: torch.Tensor) -> dict:
        """Create missing state and keep buffers in configured storage dtypes."""

        state = self.state[param]
        param_for_empty = None

        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            state_dtype = self._state_dtype_for_name(name)
            if value is None:
                if self._is_quantized_model_param(param):
                    if param_for_empty is None:
                        param_for_empty = param.dequantize(dtype=torch.float32)
                else:
                    param_for_empty = self._local_tensor(param)
                self.set_scaled_state(
                    param,
                    name,
                    torch.zeros_like(
                        param_for_empty,
                        dtype=torch.float32,
                        memory_format=torch.preserve_format,
                    ),
                )
            elif state_dtype == torch.uint8:
                if self._is_float8_state(value):
                    state_device = self._float8_state_device(value)
                    if state_device is None or not self._devices_match(
                        state_device,
                        param.device,
                    ):
                        fp32_value = self._get_scaled_state_as_unscaled(
                            param,
                            name,
                            value,
                        )
                        state.pop(name, None)
                        self.set_scaled_state(param, name, fp32_value)
                    else:
                        self._require_state_scale(param, name)
                        self._state_quantizer(param, name)
                elif torch.is_tensor(value) and value.dtype == torch.uint8:
                    raise RuntimeError(
                        "legacy plain uint8 FP8 optimizer state is not supported. "
                        "Please load from an FP32 unscaled state_dict, or regenerate "
                        "the checkpoint."
                    )
                else:
                    unscaled_value = self._get_scaled_state_as_unscaled(
                        param,
                        name,
                        value,
                    )
                    self.set_scaled_state(param, name, unscaled_value)
            elif value.dtype != state_dtype or value.device != param.device:
                unscaled_value = self._get_scaled_state_as_unscaled(param, name, value)
                self.set_scaled_state(param, name, unscaled_value)
            elif state_dtype in (torch.float32, torch.bfloat16):
                self._set_unit_state_scale(param, name)
            else:
                self._require_state_scale(param, name)
        if self.master_weights and (
            self._is_quantized_model_param(param) or param.dtype in (torch.float16, torch.bfloat16)
        ):
            store_param_remainder = self._stores_param_remainder(param)
            master_param = state.get("master_param")
            if master_param is None:
                if store_param_remainder:
                    state["master_param"] = torch.zeros_like(param, dtype=torch.int16)
                else:
                    if self._is_quantized_model_param(param):
                        if param_for_empty is None:
                            param_for_empty = param.dequantize(dtype=torch.float32)
                        master = param_for_empty.detach()
                    else:
                        master = self._local_tensor(param).detach().float()
                    self.set_scaled_state(param, "master_param", master)
            else:
                if store_param_remainder:
                    if master_param.dtype != torch.int16 or master_param.device != param.device:
                        state["master_param"] = master_param.to(
                            device=param.device,
                            dtype=torch.int16,
                        )
                    return state
                master_dtype = self._state_dtype_for_name("master_param")
                if master_param.dtype != master_dtype or master_param.device != param.device:
                    master_fp32 = self._get_scaled_state_as_unscaled(
                        param,
                        "master_param",
                        master_param,
                    )
                    self.set_scaled_state(param, "master_param", master_fp32)
                elif master_dtype == torch.bfloat16:
                    self._set_unit_state_scale(param, "master_param")
                elif master_dtype == torch.float16:
                    self._require_state_scale(param, "master_param")
                else:
                    self._ensure_state_scale(param, "master_param")
        return state

    def _max_param_state_step(self, group: dict) -> int:
        """Return the maximum legacy per-parameter step in a group."""

        max_step = 0
        for param in group["params"]:
            state = self.state.get(param)
            if state is not None and "step" in state:
                max_step = max(max_step, self._step_to_int(state["step"]))
        return max_step

    def _get_group_step_tensor(self, group: dict) -> torch.Tensor:
        """Return the non-persistent NPU tensor mirror of a group's integer step."""

        device = group["params"][0].device
        key = (id(group), device.type, device.index)
        step_tensor = self._step_tensors.get(key)
        if step_tensor is None or step_tensor.device != device:
            step_tensor = torch.tensor(
                self._step_to_int(group.get("step", 0)),
                dtype=torch.int64,
                device=device,
            )
            self._step_tensors[key] = step_tensor
        return step_tensor

    def _advance_group_step(self, group: dict) -> torch.Tensor:
        """Advance the public integer step and its private NPU tensor mirror."""

        step = self._step_to_int(group.get("step", 0))
        if step == 0:
            step = self._max_param_state_step(group)
        group["step"] = step + 1

        step_tensor = self._get_group_step_tensor(group)
        step_tensor.fill_(group["step"])
        return step_tensor

    def _apply_fused_adamw(
        self,
        step: torch.Tensor,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        exp_avgs: list[torch.Tensor],
        exp_avg_sqs: list[torch.Tensor],
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        """Apply an NPU fused update to a homogeneous tensor group."""

        if not params:
            return
        if weight_decay != 0.0 and not self.adam_w_mode:
            for grad, param in zip(grads, params):
                grad.add_(param, alpha=weight_decay)
            fused_weight_decay = 0.0
        else:
            fused_weight_decay = weight_decay

        torch._fused_adamw_(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            [step] * len(params),
            amsgrad=False,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=fused_weight_decay,
            eps=eps,
            maximize=False,
        )

    @staticmethod
    def _new_tensor_group() -> dict[str, list]:
        """Create the tensor lists consumed by fused AdamW."""

        return {
            "params": [],
            "grads": [],
            "exp_avgs": [],
            "exp_avg_sqs": [],
            "entries": [],
        }

    @staticmethod
    def _append_tensor_group(
        groups: dict[tuple, dict[str, list]],
        key: tuple,
        param: torch.Tensor,
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        entry=None,
    ) -> None:
        """Append tensors with compatible execution properties to one group."""

        tensor_group = groups.setdefault(key, FusedAdam._new_tensor_group())
        tensor_group["params"].append(param)
        tensor_group["grads"].append(grad)
        tensor_group["exp_avgs"].append(exp_avg)
        tensor_group["exp_avg_sqs"].append(exp_avg_sq)
        if entry is not None:
            tensor_group["entries"].append(entry)

    def _unscale_low_precision_groups(
        self,
        groups: dict[tuple, dict[str, list]],
        weight_decay: float,
    ) -> None:
        """Prepare FP32 working tensors for indirect fused AdamW updates."""

        for tensor_group in groups.values():
            source_entries = tensor_group["entries"]
            tensor_group["entries"] = []
            for param, grad, state in source_entries:
                direct_exp_avg = self.exp_avg_dtype == torch.float32
                direct_exp_avg_sq = self.exp_avg_sq_dtype == torch.float32
                exp_avg = (
                    self._local_tensor(state["exp_avg"])
                    if direct_exp_avg
                    else self.get_unscaled_state(param, "exp_avg")
                )
                exp_avg_sq = (
                    self._local_tensor(state["exp_avg_sq"])
                    if direct_exp_avg_sq
                    else self.get_unscaled_state(param, "exp_avg_sq")
                )
                grad_detached = grad.detach()
                grad_fp32 = grad_detached.float()
                # float() on a same-dtype fp32 grad returns it unchanged, so only
                # clone then: the fused update adds weight decay into grads in place.
                if weight_decay != 0.0 and not self.adam_w_mode and grad_fp32 is grad_detached:
                    grad_fp32 = grad_fp32.clone()

                using_master_param = self.master_weights and (
                    self._is_quantized_model_param(param)
                    or param.dtype in (torch.float16, torch.bfloat16)
                )
                direct_master_param = (
                    using_master_param and self.master_weight_dtype == torch.float32
                )
                if direct_master_param:
                    param_fp32 = self._local_tensor(state["master_param"])
                elif using_master_param:
                    param_fp32 = self.get_unscaled_state(param, "master_param")
                else:
                    param_fp32 = self._local_tensor(param).detach().float()

                tensor_group["params"].append(param_fp32)
                tensor_group["grads"].append(grad_fp32)
                tensor_group["exp_avgs"].append(exp_avg)
                tensor_group["exp_avg_sqs"].append(exp_avg_sq)
                tensor_group["entries"].append(
                    (
                        param,
                        param_fp32,
                        exp_avg,
                        exp_avg_sq,
                        using_master_param,
                        direct_exp_avg,
                        direct_exp_avg_sq,
                        direct_master_param,
                    )
                )

    def _scale_low_precision_groups(
        self,
        groups: dict[tuple, dict[str, list]],
    ) -> None:
        """Commit indirect fused AdamW updates to configured state storage."""

        for tensor_group in groups.values():
            for entry in tensor_group["entries"]:
                (
                    param,
                    param_fp32,
                    exp_avg,
                    exp_avg_sq,
                    using_master_param,
                    direct_exp_avg,
                    direct_exp_avg_sq,
                    direct_master_param,
                ) = entry
                if not direct_exp_avg:
                    self.set_scaled_state(param, "exp_avg", exp_avg)
                if not direct_exp_avg_sq:
                    self.set_scaled_state(param, "exp_avg_sq", exp_avg_sq)
                if using_master_param and not direct_master_param:
                    self.set_scaled_state(param, "master_param", param_fp32)
                if self._is_quantized_model_param(param):
                    old_param = param.detach()
                    param.quantize_(param_fp32.to(dtype=param.dtype))
                    new_tensors, _ = param.prepare_for_saving()
                    old_tensors, _ = old_param.prepare_for_saving()
                    for old_tensor, new_tensor in zip(old_tensors, new_tensors):
                        if (old_tensor is None) != (new_tensor is None):
                            param.restore_from_saved(old_tensors)
                            raise RuntimeError(
                                "Quantized parameter storage layout changed during writeback."
                            )
                        if old_tensor is None:
                            continue
                        if old_tensor.numel() != new_tensor.numel():
                            param.restore_from_saved(old_tensors)
                            raise RuntimeError(
                                "Quantized parameter storage size changed during writeback."
                            )
                        old_tensor.reshape(-1).copy_(new_tensor.reshape(-1))
                    param.restore_from_saved(old_tensors)
                else:
                    self._local_tensor(param).copy_(param_fp32.to(dtype=param.dtype))

    def _apply_fused_adamw_groups(
        self,
        groups: dict[tuple, dict[str, list]],
        step: torch.Tensor,
        **update_kwargs,
    ) -> None:
        """Apply fused AdamW to each homogeneous tensor group."""

        for tensor_group in groups.values():
            self._apply_fused_adamw(
                step,
                tensor_group["params"],
                tensor_group["grads"],
                tensor_group["exp_avgs"],
                tensor_group["exp_avg_sqs"],
                **update_kwargs,
            )

    @torch.no_grad()
    def step(self, closure=None, grad_scaler=None):
        """Perform a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grad_scaler (optional): Accepted for TE FusedAdam API compatibility.
                Gradients must already be unscaled by the caller, matching the
                non-capturable upstream path.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if len(group["params"]) == 0:
                group["step"] = self._step_to_int(group.get("step", 0)) + 1
                continue
            if group.get("amsgrad", False):
                raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
            self._check_param_group_supported(group)

            step = self._advance_group_step(group)
            lr = self._scalar_to_float(group["lr"], "lr")
            eps = self._scalar_to_float(group["eps"], "eps")
            weight_decay = self._scalar_to_float(group["weight_decay"], "weight_decay")
            beta1, beta2 = group["betas"]
            beta1 = self._scalar_to_float(beta1, "beta1")
            beta2 = self._scalar_to_float(beta2, "beta2")
            direct_groups = {}
            delayed_fp8_groups = {}
            low_precision_groups = {}
            remainder_groups = {}

            for param in group["params"]:
                self._check_param_dtype(param)
                if self.use_decoupled_grad:
                    grad = getattr(param, "decoupled_grad", None)
                else:
                    grad = param.grad
                if grad is None:
                    continue
                if isinstance(param, DTensor):
                    if not isinstance(grad, DTensor) or (
                        grad.device_mesh != param.device_mesh
                        or grad.placements != param.placements
                        or grad.shape != param.shape
                    ):
                        raise RuntimeError(
                            "DTensor gradient must match the parameter shard layout."
                        )
                    grad = self._local_tensor(grad)
                if self._has_sparse_grad(grad):
                    raise RuntimeError("FusedAdam does not support sparse gradients.")

                self._init_or_fix_state(param)
                state = self.state[param]
                if isinstance(param, DTensor) and self._local_tensor(param).numel() == 0:
                    continue
                if self._stores_param_remainder(param):
                    # The remainder kernel requires matching FP32 or BF16 moments.
                    # Match upstream by using FP32 work tensors for every other combination.
                    direct_moments = (
                        self.exp_avg_dtype == self.exp_avg_sq_dtype
                        and self.exp_avg_dtype in (torch.float32, torch.bfloat16)
                    )
                    direct_exp_avg = direct_moments or self.exp_avg_dtype == torch.float32
                    direct_exp_avg_sq = direct_moments or self.exp_avg_sq_dtype == torch.float32
                    exp_avg = (
                        state["exp_avg"]
                        if direct_exp_avg
                        else self.get_unscaled_state(param, "exp_avg")
                    )
                    exp_avg_sq = (
                        state["exp_avg_sq"]
                        if direct_exp_avg_sq
                        else self.get_unscaled_state(param, "exp_avg_sq")
                    )
                    remainder_group = remainder_groups.setdefault(
                        (param.device, grad.dtype, exp_avg.dtype),
                        {
                            "grads": [],
                            "params": [],
                            "exp_avgs": [],
                            "exp_avg_sqs": [],
                            "remainders": [],
                            "entries": [],
                        },
                    )
                    remainder_group["grads"].append(grad.detach().contiguous())
                    remainder_group["params"].append(param)
                    remainder_group["exp_avgs"].append(exp_avg)
                    remainder_group["exp_avg_sqs"].append(exp_avg_sq)
                    remainder_group["remainders"].append(state["master_param"])
                    remainder_group["entries"].append(
                        (param, exp_avg, exp_avg_sq, direct_exp_avg, direct_exp_avg_sq)
                    )
                    continue
                quantizer = param._get_quantizer() if isinstance(param, Float8Tensor) else None
                is_fused_fp8_param = (
                    isinstance(quantizer, Float8Quantizer)
                    and param._data is not None
                    and quantizer.scale is not None
                    and quantizer.amax is not None
                    and param._scale_inv is not None
                )
                use_fused_fp8 = is_fused_fp8_param and (
                    state["master_param"].dtype == torch.float32
                    and state["exp_avg"].dtype == state["exp_avg_sq"].dtype
                    and state["exp_avg"].dtype in (torch.float32, torch.bfloat16)
                )
                if use_fused_fp8:
                    fp8_group = delayed_fp8_groups.setdefault(
                        (param.device, grad.dtype, param._fp8_dtype, state["exp_avg"].dtype),
                        {
                            "grads": [],
                            "params": [],
                            "exp_avgs": [],
                            "exp_avg_sqs": [],
                            "masters": [],
                            "scales": [],
                            "amaxes": [],
                            "scale_invs": [],
                            "model_params": [],
                        },
                    )
                    fp8_group["grads"].append(grad.detach().contiguous())
                    fp8_group["params"].append(param._data)
                    fp8_group["exp_avgs"].append(state["exp_avg"])
                    fp8_group["exp_avg_sqs"].append(state["exp_avg_sq"])
                    fp8_group["masters"].append(state["master_param"])
                    fp8_group["scales"].append(quantizer.scale)
                    fp8_group["amaxes"].append(quantizer.amax)
                    fp8_group["scale_invs"].append(param._scale_inv)
                    fp8_group["model_params"].append(param)
                    continue
                direct_state = (
                    not self.master_weights
                    and param.dtype in (torch.float32, torch.float16, torch.bfloat16)
                    and grad.dtype == param.dtype
                    and self.exp_avg_dtype == torch.float32
                    and self.exp_avg_sq_dtype == torch.float32
                )
                if direct_state:
                    direct_grad = (
                        grad.detach().clone()
                        if weight_decay != 0.0 and not self.adam_w_mode
                        else grad.detach()
                    )
                    direct_key = (param.device, param.dtype, direct_grad.dtype)
                    self._append_tensor_group(
                        direct_groups,
                        direct_key,
                        self._local_tensor(param),
                        direct_grad,
                        self._local_tensor(state["exp_avg"]),
                        self._local_tensor(state["exp_avg_sq"]),
                    )
                    continue

                low_precision_key = (param.device, torch.float32)
                low_precision_group = low_precision_groups.setdefault(
                    low_precision_key,
                    self._new_tensor_group(),
                )
                low_precision_group["entries"].append((param, grad, state))

            update_kwargs = {
                "lr": lr,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "weight_decay": weight_decay,
            }
            self._unscale_low_precision_groups(low_precision_groups, weight_decay)

            self._apply_fused_adamw_groups(direct_groups, step, **update_kwargs)
            self._apply_fused_adamw_groups(low_precision_groups, step, **update_kwargs)

            if delayed_fp8_groups:
                from ..triton.multi_tensor_adam_fp8 import multi_tensor_adam_fp8

                for fp8_key, fp8_group in delayed_fp8_groups.items():
                    multi_tensor_adam_fp8(
                        [
                            fp8_group["grads"],
                            fp8_group["params"],
                            fp8_group["exp_avgs"],
                            fp8_group["exp_avg_sqs"],
                            fp8_group["masters"],
                            fp8_group["scales"],
                            fp8_group["amaxes"],
                            fp8_group["scale_invs"],
                        ],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        step=group["step"],
                        adam_w_mode=bool(self.adam_w_mode),
                        bias_correction=bool(group["bias_correction"]),
                        weight_decay=weight_decay,
                        fp8_dtype=fp8_key[2],
                        metadata_cache=self._fp8_adam_metadata_cache,
                        cache_key=(id(group), fp8_key),
                    )
                    for param in fp8_group["model_params"]:
                        param._reset_caches()

            if remainder_groups:
                from ..triton.multi_tensor_adam_param_remainder import (
                    multi_tensor_adam_param_remainder,
                )

                for remainder_key, remainder_group in remainder_groups.items():
                    multi_tensor_adam_param_remainder(
                        [
                            remainder_group["grads"],
                            remainder_group["params"],
                            remainder_group["exp_avgs"],
                            remainder_group["exp_avg_sqs"],
                            remainder_group["remainders"],
                        ],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        step=group["step"],
                        adam_w_mode=bool(self.adam_w_mode),
                        weight_decay=weight_decay,
                        metadata_cache=self._remainder_metadata_cache,
                        cache_key=(id(group), remainder_key),
                    )
                    for (
                        param,
                        exp_avg,
                        exp_avg_sq,
                        direct_exp_avg,
                        direct_exp_avg_sq,
                    ) in remainder_group["entries"]:
                        if not direct_exp_avg:
                            self.set_scaled_state(param, "exp_avg", exp_avg)
                        if not direct_exp_avg_sq:
                            self.set_scaled_state(param, "exp_avg_sq", exp_avg_sq)

            self._scale_low_precision_groups(low_precision_groups)

        return loss
