# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Activation function implementations for NPU.

These are used both by LayerNormMLP (via dispatch tables) and by the
FusibleOperation activation classes in activation.py.
"""

import torch
import torch.nn.functional as F
import torch_npu


# =============================================================================
# Element-wise activation functions
# =============================================================================


def gelu_fwd(x, approximate="tanh"):
    return torch_npu.npu_gelu(x, approximate=approximate)


def gelu_bwd(x, dy, approximate="tanh"):
    return torch_npu.npu_gelu_backward(dy, x, approximate=approximate)


def silu_fwd(x):
    return F.silu(x)


def silu_bwd(x, dy):
    # aclnn fused backward
    return torch.ops.aten.silu_backward(dy, x)


def relu_fwd(x):
    return F.relu(x)


def relu_bwd(x, dy):
    return dy * (x > 0).to(dy.dtype)


def qgelu_fwd(x):
    return torch_npu.npu_fast_gelu(x)


def qgelu_bwd(x, dy):
    orig_dtype = x.dtype
    return torch_npu.npu_fast_gelu_backward(dy.float(), x.float()).to(orig_dtype)


def srelu_fwd(x):
    relu_x = F.relu(x)
    return relu_x * relu_x


def srelu_bwd(x, dy):
    return dy * 2.0 * F.relu(x)


# =============================================================================
# Gated (GLU) activation functions
# =============================================================================


def glu_fwd(x):
    gate, linear = x.chunk(2, dim=-1)
    return torch.sigmoid(gate) * linear


def glu_bwd(x, dy):
    orig_dtype = x.dtype
    gate, linear = x.chunk(2, dim=-1)
    gate_f = gate.float()
    linear_f = linear.float()
    dy_f = dy.float()
    sig = torch.sigmoid(gate_f)
    dgate = dy_f * linear_f * sig * (1.0 - sig)
    dlinear = dy_f * sig
    return torch.cat([dgate.to(orig_dtype), dlinear.to(orig_dtype)], dim=-1)


def swiglu_fwd(x):
    return torch_npu.npu_swiglu(x)


def swiglu_bwd(x, dy):
    return torch_npu.npu_swiglu_backward(dy, x)


def geglu_fwd(x, approximate="tanh"):
    gate, linear = x.chunk(2, dim=-1)
    return torch_npu.npu_gelu(gate, approximate=approximate) * linear


def geglu_bwd(x, dy, approximate="tanh"):
    gate, linear = x.chunk(2, dim=-1)
    gelu_gate = torch_npu.npu_gelu(gate, approximate=approximate)
    dgate = torch_npu.npu_gelu_backward(dy * linear, gate, approximate=approximate)
    dlinear = dy * gelu_gate
    return torch.cat([dgate, dlinear], dim=-1)


def reglu_fwd(x):
    gate, linear = x.chunk(2, dim=-1)
    return F.relu(gate) * linear


def reglu_bwd(x, dy):
    gate, linear = x.chunk(2, dim=-1)
    dgate = dy * linear * (gate > 0).to(dy.dtype)
    dlinear = dy * F.relu(gate)
    return torch.cat([dgate, dlinear], dim=-1)


def sreglu_fwd(x):
    gate, linear = x.chunk(2, dim=-1)
    relu_gate = F.relu(gate)
    return relu_gate * relu_gate * linear


def sreglu_bwd(x, dy):
    gate, linear = x.chunk(2, dim=-1)
    dgate = dy * linear * 2.0 * F.relu(gate)
    relu_gate = F.relu(gate)
    dlinear = dy * relu_gate * relu_gate
    return torch.cat([dgate, dlinear], dim=-1)


def qgeglu_fwd(x):
    orig_dtype = x.dtype
    gate, linear = x.chunk(2, dim=-1)
    gate_f = gate.float()
    linear_f = linear.float()
    return (torch_npu.npu_fast_gelu(gate_f) * linear_f).to(orig_dtype)


def qgeglu_bwd(x, dy):
    orig_dtype = x.dtype
    gate, linear = x.chunk(2, dim=-1)
    gate_f = gate.float()
    linear_f = linear.float()
    dy_f = dy.float()
    qgelu_gate = torch_npu.npu_fast_gelu(gate_f)
    dgate = torch_npu.npu_fast_gelu_backward(dy_f * linear_f, gate_f)
    dlinear = dy_f * qgelu_gate
    return torch.cat([dgate.to(orig_dtype), dlinear.to(orig_dtype)], dim=-1)


def clamped_swiglu_fwd(x, limit=7.0, alpha=1.702, glu_linear_offset=1.0):
    """NVTE-compatible clamped SwiGLU forward."""
    glu, linear = x.chunk(2, dim=-1)
    glu = torch.clamp(glu, max=limit)
    linear = torch.clamp(linear, min=-limit, max=limit) + glu_linear_offset
    return glu * torch.sigmoid(alpha * glu) * linear


def clamped_swiglu_bwd(x, dy, limit=7.0, alpha=1.702, glu_linear_offset=1.0):
    """NVTE-compatible clamped SwiGLU backward."""
    glu, linear = x.chunk(2, dim=-1)
    clamped_glu = torch.clamp(glu, max=limit)
    clamped_linear = torch.clamp(linear, min=-limit, max=limit) + glu_linear_offset
    sigmoid = torch.sigmoid(alpha * clamped_glu)
    activated_glu = clamped_glu * sigmoid

    dactivated_glu = sigmoid + alpha * clamped_glu * sigmoid * (1.0 - sigmoid)
    dactivated_glu = dactivated_glu * (glu <= limit).to(dactivated_glu.dtype)
    dglu = dy * clamped_linear * dactivated_glu

    dlinear = dy * activated_glu
    linear_in_range = (linear >= -limit) & (linear <= limit)
    dlinear = dlinear * linear_in_range.to(dlinear.dtype)
    return torch.cat([dglu, dlinear], dim=-1)


# =============================================================================
# Dispatch tables
# =============================================================================

ACTIVATION_FWD = {
    "gelu": gelu_fwd,
    "silu": silu_fwd,
    "swiglu": swiglu_fwd,
    "geglu": geglu_fwd,
    "reglu": reglu_fwd,
    "sreglu": sreglu_fwd,
    "relu": relu_fwd,
    "glu": glu_fwd,
    "qgelu": qgelu_fwd,
    "qgeglu": qgeglu_fwd,
    "srelu": srelu_fwd,
    "clamped_swiglu": clamped_swiglu_fwd,
}

ACTIVATION_BWD = {
    "gelu": gelu_bwd,
    "silu": silu_bwd,
    "swiglu": swiglu_bwd,
    "geglu": geglu_bwd,
    "reglu": reglu_bwd,
    "sreglu": sreglu_bwd,
    "relu": relu_bwd,
    "glu": glu_bwd,
    "qgelu": qgelu_bwd,
    "qgeglu": qgeglu_bwd,
    "srelu": srelu_bwd,
    "clamped_swiglu": clamped_swiglu_bwd,
}

GLU_VARIANTS = {"swiglu", "geglu", "reglu", "sreglu", "glu", "qgeglu", "clamped_swiglu"}
