# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""CPU regressions for pad adapter layout/gradient semantics, without ATK.

The CUDA stand-in models NVTE's aligned expert blocks with differentiable
gathers. It exercises the real adapter and conversion code, not CUDA kernels.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


MODULE_DIR = Path(__file__).parent / "module"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="adapter")
def adapter_fixture(monkeypatch):
    stubs = {
        name: ModuleType(name)
        for name in (
            "atk",
            "atk.tasks",
            "atk.tasks.api_execute",
            "atk.tasks.api_execute.base_api",
            "atk.tasks.api_execute.aclnn_base_api",
            "_comparators",
            "transformer_engine",
            "transformer_engine.pytorch",
        )
    }
    stubs["atk.tasks.api_execute"].register = lambda name: lambda cls: cls
    stubs["atk.tasks.api_execute.base_api"].BaseApi = object
    stubs["atk.tasks.api_execute.aclnn_base_api"].AclnnBaseApi = object
    stubs["_comparators"].register_module_comparators = lambda name: None
    stubs["transformer_engine"].pytorch = stubs["transformer_engine.pytorch"]
    stubs["_moe_convert"] = _load_module("pad_convert", MODULE_DIR / "_moe_convert.py")
    stubs["_cpu_baselines"] = _load_module("pad_cpu", MODULE_DIR / "_cpu_baselines.py")
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    return _load_module(
        "pad_adapter",
        MODULE_DIR / "moe_permute_and_pad_with_probs" / "function_mp_pad_probs.py",
    )


def _nvte_pad_reference(inp, probs, routes, counts, align):
    """Model native CUDA output layout; record routes in TE's dense map."""
    tokens, experts = routes.shape
    targets = ((counts + align - 1) // align * align).long()
    pad_lengths = targets - counts
    offsets = pad_lengths.cumsum(0) - pad_lengths
    row_map = torch.full((tokens, 2 * experts + 1), -1, dtype=torch.int32)
    row_map[:, -1] = 0
    outputs, probabilities = [], []
    start = 0
    for expert in range(experts):
        ids = torch.where(routes[:, expert].bool())[0]
        padding = int(targets[expert]) - len(ids)
        outputs.append(torch.cat([inp[ids], inp.new_zeros((padding, inp.size(1)))]))
        probabilities.append(torch.cat([probs[ids, expert], probs.new_zeros(padding)]))
        for rank, token in enumerate(ids.tolist()):
            slot = int(row_map[token, -1])
            row_map[token, slot] = start + rank
            row_map[token, experts + slot] = expert
            row_map[token, -1] += 1
        start += int(targets[expert])
    return torch.cat(outputs), torch.cat(probabilities), row_map, offsets, targets


def _expected(inp, probs, routes, capacity, dy, dy_probs):
    """Independent oracle: accumulate each retained route's canonical seed."""
    experts = routes.size(1)
    out = inp.new_zeros((experts * capacity, inp.size(1)))
    out_probs = probs.new_zeros(experts * capacity)
    dx, dp = torch.zeros_like(inp), torch.zeros_like(probs)
    for expert in range(experts):
        slot = 0
        for token in range(inp.size(0)):
            if routes[token, expert] and slot < capacity:
                row = expert * capacity + slot
                out[row] = inp[token]
                out_probs[row] = probs[token, expert]
                dx[token] += dy[row]
                dp[token, expert] += dy_probs[row]
                slot += 1
    return out, out_probs, dx, dp


# Balanced control, cropping without alignment padding, unequal padded blocks,
# an empty expert, and top-k accumulation with cropping and padding together.
CASES = [
    ([[1, 0], [0, 1], [1, 0], [0, 1]], 1),
    ([[1, 0], [1, 0], [1, 0], [0, 1]], 1),
    ([[1, 0], [0, 1], [0, 1], [0, 1]], 2),
    ([[1, 0], [0, 1], [1, 0], [0, 1]], 4),
    (
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        4,
    ),
    (
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ],
        4,
    ),
]


@pytest.mark.parametrize("rows,align", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_adapter_forward_and_backward_share_canonical_layout(
    adapter, monkeypatch, rows, align, dtype, backend
):
    routes = torch.tensor(rows, dtype=torch.bool)
    tokens, experts = routes.shape
    hidden = 8
    counts = routes.sum(0).to(torch.int32)
    padded = int(((counts + align - 1) // align * align).sum())
    assert padded % experts == 0
    inp = (torch.arange(tokens * hidden).reshape(tokens, hidden) / 16).to(dtype)
    probs = (torch.arange(tokens * experts).reshape(tokens, experts) / 16).to(dtype)
    # Nonzero, row-dependent seeds catch both misplaced rows and pad leakage.
    dy = ((torch.arange(padded * hidden).reshape(padded, hidden) + 1) / 16).to(dtype)
    dy_probs = ((torch.arange(padded) + 1) / 16).to(dtype)
    expected = _expected(inp, probs, routes, padded // experts, dy, dy_probs)
    args = [
        inp,
        probs,
        dy,
        dy_probs,
        ";".join(",".join(str(e) for e, routed in enumerate(row) if routed) for row in rows),
        experts,
        sum(rows[0]),
        "bool",
        ",".join(str(c) for c in counts.tolist()),
        align,
    ]
    snapshots = [arg.clone() for arg in args[:4]]
    build_routes = adapter._build_routing_map
    build_counts = adapter._build_tokens_per_expert
    monkeypatch.setattr(adapter, "_build_routing_map", lambda *a: build_routes(*a[:-1], "cpu"))
    monkeypatch.setattr(
        adapter, "_build_tokens_per_expert", lambda *a: build_counts(*a[:-1], "cpu")
    )
    captured = {}

    def call_native(module, x, p, routing, received, alignment):
        captured["probs"] = p
        fn = _nvte_pad_reference if backend == "cuda" else adapter.mp_pad_probs_cpu
        result = fn(x, p, routing, received, alignment)
        captured["native"] = result[:2]
        captured["snapshots"] = [tensor.detach().clone() for tensor in result[:2]]
        return result

    monkeypatch.setattr(adapter, "_call_mp_pad_probs", call_native)
    actual = adapter.run_mp_pad_probs_case(SimpleNamespace(args=args), backend)
    assert len(actual) == 7
    for result, reference in zip((*actual[:3], captured["probs"].grad), expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for arg, snapshot in zip(args[:4], snapshots):
        torch.testing.assert_close(arg, snapshot, rtol=0, atol=0)
    for native, snapshot in zip(captured["native"], captured["snapshots"]):
        torch.testing.assert_close(native, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("nvte_layout", [False, True])
def test_conversion_does_not_modify_leaf_outputs(adapter, nvte_layout):
    # A direct leaf also catches in-place writes through views, independently
    # of whether a particular backend's custom autograd happens to allow them.
    counts = torch.tensor([3, 1], dtype=torch.int32)
    raw = torch.arange(8, dtype=torch.float32).reshape(4, 2).requires_grad_()
    before = raw.detach().clone()
    result = adapter.mp_convert._suppress_padded_permuted_to_npu(
        raw, 2, 1, counts, nvte_layout=nvte_layout
    )
    result.sum().backward()
    torch.testing.assert_close(raw, before)
    expected_grad = torch.ones_like(raw)
    expected_grad[2 if nvte_layout else 3] = 0
    torch.testing.assert_close(raw.grad, expected_grad)
