# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.
"""Python fallback implementation of FusedSGD for Ascend NPU."""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch.optim.optimizer import Optimizer, required


class FusedSGD(Optimizer):
    """SGD optimizer with the Transformer Engine FusedSGD interface.

    This NPU implementation is a correctness-first Python fallback.  It keeps
    the public TE/Apex FusedSGD API, but intentionally avoids NVIDIA CUDA
    extensions such as ``transformer_engine_torch.multi_tensor_sgd``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter | dict],
        lr: float | Any = required,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        *,
        wd_after_momentum=False,
        materialize_master_grads=True,
        set_grad_none: Optional[bool] = None,  # deprecated
    ):
        if lr is not required and self._scalar_to_float(lr, "lr") < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if self._scalar_to_float(momentum, "momentum") < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if self._scalar_to_float(weight_decay, "weight_decay") < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        momentum_value = self._scalar_to_float(momentum, "momentum")
        dampening_value = self._scalar_to_float(dampening, "dampening")
        if nesterov and (momentum_value <= 0.0 or dampening_value != 0.0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        self.wd_after_momentum = wd_after_momentum
        self.materialize_master_grads = materialize_master_grads
        self.most_recent_scale = 1.0
        self.scale_set_by_backward = False

        self.set_grad_none = set_grad_none
        if self.set_grad_none is not None:
            warnings.warn(
                "set_grad_none kwarg in FusedSGD constructor is deprecated. "
                "Use set_to_none kwarg in zero_grad instead.",
                DeprecationWarning,
            )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
        self.wd_after_momentum = getattr(self, "wd_after_momentum", False)
        self.materialize_master_grads = getattr(self, "materialize_master_grads", True)
        self.most_recent_scale = getattr(self, "most_recent_scale", 1.0)
        self.scale_set_by_backward = getattr(self, "scale_set_by_backward", False)
        self.set_grad_none = getattr(self, "set_grad_none", None)

    def zero_grad(self, set_to_none: Optional[bool] = None) -> None:
        """Reset parameter gradients.

        If ``set_to_none`` is omitted, use the deprecated constructor option
        ``set_grad_none`` when provided; otherwise match TE FusedSGD's default
        of setting gradients to ``None``.
        """

        if self.set_grad_none is not None:
            if set_to_none is not None and set_to_none != self.set_grad_none:
                raise ValueError(
                    f"Called zero_grad with set_to_none={set_to_none}, "
                    f"but FusedSGD was initialized with set_grad_none={self.set_grad_none}"
                )
            set_to_none = self.set_grad_none
        if set_to_none is None:
            set_to_none = True

        super().zero_grad(set_to_none=set_to_none)

    @staticmethod
    def _scalar_to_float(value, name: str) -> float:
        """Convert a scalar optimizer hyperparameter to a Python float."""

        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Optimizer parameter {name} must be a scalar.")
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _has_sparse_grad(grad: torch.Tensor) -> bool:
        """Return True for COO or compressed sparse gradients."""

        if grad.is_sparse:
            return True
        if grad.layout != torch.strided:
            return True
        is_sparse_csr = getattr(grad, "is_sparse_csr", False)
        if callable(is_sparse_csr):
            is_sparse_csr = is_sparse_csr()
        return bool(is_sparse_csr)

    def _check_master_grad_path(self) -> None:
        """Reject Apex AMP master-parameter paths that are not implemented yet."""

        if hasattr(self, "_amp_stash") and hasattr(self._amp_stash, "fp32_from_fp16_groups"):
            raise NotImplementedError(
                "NPU FusedSGD fallback does not support Apex AMP explicit "
                "master-parameter gradients/materialize_master_grads yet."
            )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._check_master_grad_path()
        grad_scale = 1.0 / self._scalar_to_float(self.most_recent_scale, "most_recent_scale")

        for group in self.param_groups:
            lr = self._scalar_to_float(group["lr"], "lr")
            momentum = self._scalar_to_float(group["momentum"], "momentum")
            dampening = self._scalar_to_float(group["dampening"], "dampening")
            weight_decay = self._scalar_to_float(group["weight_decay"], "weight_decay")
            nesterov = group["nesterov"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if self._has_sparse_grad(grad):
                    raise RuntimeError("FusedSGD does not support sparse gradients.")
                if not param.is_floating_point():
                    raise RuntimeError(
                        "FusedSGD only supports floating point model parameters."
                    )

                param_fp32 = param.detach().to(dtype=torch.float32)
                d_p = grad.detach().to(dtype=torch.float32).mul(grad_scale)

                if weight_decay != 0.0 and not self.wd_after_momentum:
                    d_p.add_(param_fp32, alpha=weight_decay)

                if momentum != 0.0:
                    param_state = self.state[param]
                    momentum_buffer = param_state.get("momentum_buffer")
                    if momentum_buffer is None:
                        momentum_buffer = torch.clone(d_p).detach()
                        param_state["momentum_buffer"] = momentum_buffer
                    else:
                        if (
                            momentum_buffer.dtype != torch.float32
                            or momentum_buffer.device != param.device
                        ):
                            momentum_buffer = momentum_buffer.to(
                                device=param.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                            param_state["momentum_buffer"] = momentum_buffer
                        momentum_buffer.mul_(momentum).add_(d_p, alpha=1.0 - dampening)

                    if nesterov:
                        d_p = d_p.add(momentum_buffer, alpha=momentum)
                    else:
                        d_p = momentum_buffer

                if weight_decay != 0.0 and self.wd_after_momentum:
                    d_p = d_p.add(param_fp32, alpha=weight_decay)

                param_fp32.add_(d_p, alpha=-lr)
                param.copy_(param_fp32.to(dtype=param.dtype))

        self.most_recent_scale = 1.0
        self.scale_set_by_backward = False

        return loss

