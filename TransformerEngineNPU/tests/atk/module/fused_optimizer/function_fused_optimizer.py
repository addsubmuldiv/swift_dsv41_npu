# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""ATK execution and comparison plugins for TE fused optimizers (FusedAdam/FusedSGD).

Follows the **no-CPU-reference** template (see ``../rmsnorm/README.md``): the
adapter only runs the device (NPU) implementation of
``transformer_engine.pytorch.optimizers.FusedAdam`` / ``FusedSGD``. NVTE
consistency comes from the offline golden loop (``collect_nvte_golden.py`` + an
ATK ``accuracy_load`` node), never from a local torch reference.

Unlike op guards there is no forward/backward: the optimizer updates parameters
**in place** over N steps. The guard compares the post-step parameters and the
optimizer state (``exp_avg`` / ``exp_avg_sq`` for Adam, ``momentum_buffer`` for
SGD) against the NVTE golden collected on CUDA under the same canonical inputs,
hyperparams and step count.

Coverage: single param group, fp32 master params + fp32 grads, N steps
(small=7 matching NVTE's own ``test_fused_optimizer.py`` default, real=3),
Adam (bias_correction / adam_w_mode) and SGD (momentum / nesterov) variants.
Multi-param groups, ``grad_scaler`` (loss scaling) and ``master_weights`` are
follow-ups.
"""

import json
from typing import Dict, Tuple

import torch

from atk.tasks.api_execute import register

# ATK's API plugin loader resolves both BaseApi symbols from this namespace.
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi  # noqa: F401
from atk.tasks.api_execute.base_api import BaseApi


# Shared accuracy comparators (allclose / mare / mixed_tolerance) live in
# ../_comparators.py and are registered here under this module's prefix so
# the yaml standard.acc block can select any of the three standards.
import os
import sys

_comparators_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _comparators_dir not in sys.path:
    sys.path.insert(0, _comparators_dir)
from _comparators import register_module_comparators  # noqa: E402

register_module_comparators("fused_optimizer")

RESULT_SCHEMA = [
    "params_final",  # post-step fp32 params — tolerance compare
    "first_moment",  # Adam exp_avg / SGD momentum_buffer (fp32)
    "second_moment",  # Adam exp_avg_sq / SGD zeros (fp32)
    "step_count",  # int optimizer step — exact compare
    "api_contract_json_utf8",
]


def _encode_contract(contract: Dict, device: torch.device) -> torch.Tensor:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8, device=device)


def run_fused_optimizer_case(input_data, device: str) -> Tuple[torch.Tensor, ...]:
    """Run N fused-optimizer steps on canonical ATK inputs and return post-step
    params + optimizer state.

    Only the device implementation is provided; numerical consistency with NVTE
    is established by the offline golden loop, never by a local torch reference.

    Input layout: ``params, grads, optim_type(str), lr, weight_decay, steps,
    b0, b1, eps, momentum, nesterov``. Adam uses (b0, b1, eps); SGD uses
    (momentum, nesterov); the unused attrs are 0/False placeholders so the case
    layout is fixed.
    """
    values = list(input_data.args)
    if len(values) != 11:
        raise ValueError(
            "fused_optimizer ATK case expects 11 positional inputs "
            "(params, grads, optim_type, lr, weight_decay, steps, b0, b1, eps, "
            "momentum, nesterov), got {}".format(len(values))
        )
    (params, grads, optim_type, lr, weight_decay, steps, b0, b1, eps, momentum, nesterov) = values
    optim_type = str(optim_type)
    lr = float(lr)
    weight_decay = float(weight_decay)
    steps = int(steps)
    nesterov = bool(nesterov)

    if optim_type == "adam":
        betas = (float(b0), float(b1))
        eps_value = float(eps)
        momentum_value = 0.0
    elif optim_type == "sgd":
        betas = (0.9, 0.999)
        eps_value = 1e-8
        momentum_value = float(momentum)
    else:
        raise ValueError("unsupported optim_type: {} (want adam/sgd)".format(optim_type))

    # FusedAdam/FusedSGD are exported from transformer_engine.pytorch.optimizers
    # (NOT the top-level transformer_engine.pytorch namespace).
    from transformer_engine.pytorch.optimizers import FusedAdam, FusedSGD

    param = torch.nn.Parameter(params.detach().clone().requires_grad_(True))
    grads = grads.detach().clone()

    if optim_type == "adam":
        opt = FusedAdam(
            [param],
            lr=lr,
            betas=betas,
            eps=eps_value,
            weight_decay=weight_decay,
        )
    else:
        opt = FusedSGD(
            [param],
            lr=lr,
            momentum=momentum_value,
            dampening=0.0,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    for _ in range(steps):
        param.grad = grads
        opt.step()

    state = opt.state[param]
    first = state.get("exp_avg", state.get("momentum_buffer"))
    if first is None:
        first = torch.zeros_like(param).detach()
    # Adam tracks a second moment; SGD does not (fill zeros so both sides emit
    # the same second_moment slot).
    second = state.get("exp_avg_sq", torch.zeros_like(param).detach())
    step_count = int(state.get("step", steps))

    contract = {
        "optim_type": optim_type,
        "steps": steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "b0": float(b0) if optim_type == "adam" else 0.0,
        "b1": float(b1) if optim_type == "adam" else 0.0,
        "eps": eps_value,
        "momentum": momentum_value,
        "nesterov": nesterov,
        "param_shape": list(params.shape),
        "param_dtype": str(params.dtype),
        "grad_dtype": str(grads.dtype),
    }
    return (
        param.detach(),
        first.detach(),
        second.detach(),
        torch.tensor([step_count], dtype=torch.int64),
        _encode_contract(contract, param.device),
    )


@register("function_fused_optimizer")
class FunctionFusedOptimizer(BaseApi):
    """ATK module adapter with a device-only implementation.

    The ``cpu`` backend is deliberately unsupported: this template ships no CPU
    reference. The ``cpu_nvte`` accuracy-load node feeds NVTE golden artifacts and
    never instantiates this class.
    """

    def __call__(self, input_data, with_output: bool = False):
        if self.device == "npu":
            result = run_fused_optimizer_case(input_data, "npu")
        elif self.device in ("gpu", "cuda"):
            # Online GPU-side execution (ATK server / new gpu backend): run the
            # same case function against NVTE on CUDA. 'gpu' is the ATK backend
            # label; torch needs 'cuda'.
            result = run_fused_optimizer_case(input_data, "cuda")
        else:
            raise RuntimeError(
                "fused_optimizer template has no CPU reference; only 'npu' "
                "execution is supported. NVTE consistency comes from the "
                "accuracy_load golden node, not from a local torch implementation. "
                "Got backend: {}".format(self.device)
            )
        return result if with_output else None
