"""CPU regressions for the unpermute pad adapter, including its GPU fallback.

The NPU-labelled path uses CPU stand-ins for the native operators; these tests
check adapter semantics, not NPU kernel execution.
"""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


MODULE_DIR = Path(__file__).parent / "module"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="adapter")
def adapter_fixture(monkeypatch):
    names = (
        "atk",
        "atk.tasks",
        "atk.tasks.api_execute",
        "atk.tasks.api_execute.base_api",
        "atk.tasks.api_execute.aclnn_base_api",
        "_comparators",
        "transformer_engine",
        "transformer_engine.pytorch",
    )
    stubs = {name: ModuleType(name) for name in names}
    stubs["atk.tasks.api_execute"].register = lambda name: lambda cls: cls
    stubs["atk.tasks.api_execute.base_api"].BaseApi = object
    stubs["atk.tasks.api_execute.aclnn_base_api"].AclnnBaseApi = object
    stubs["_comparators"].register_module_comparators = lambda name: None
    stubs["transformer_engine"].pytorch = stubs["transformer_engine.pytorch"]
    stubs["_cpu_baselines"] = _load("unpermute_cpu", MODULE_DIR / "_cpu_baselines.py")
    stubs["_moe_convert"] = _load("unpermute_convert", MODULE_DIR / "_moe_convert.py")
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    return _load("unpermute_adapter", MODULE_DIR / "moe_unpermute" / "function_moe_unpermute.py")


CASES = [
    ([[1, 0], [0, 1], [1, 0], [0, 1]], 1),  # balanced control
    ([[1, 0]] * 3 + [[0, 1]], 1),  # capacity cropping without alignment padding
    ([[1, 0], [0, 1], [1, 0], [0, 1]], 4),  # duplicated pad tokens
    ([[1, 0], [0, 1], [0, 1], [0, 1]], 2),  # unequal aligned counts
    (
        [[1, 0, 0, 0]] * 2 + [[0, 1, 0, 0]] * 2 + [[0, 0, 1, 0]] * 2 + [[0, 0, 0, 1]],
        2,
    ),  # only last expert pads
    ([[1, 0, 0, 0]] * 4 + [[0, 1, 0, 0]] * 3 + [[0, 0, 1, 0]], 4),
    ([[1, 1, 0, 0]] * 3 + [[1, 0, 1, 0]] * 2 + [[0, 1, 1, 0]], 4),
]


@pytest.mark.parametrize("rows,align", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("routing_dtype", ["bool", "int32"])
@pytest.mark.parametrize("backend", ["cuda", "npu"])
def test_pad_round_trip_matches_retained_routes(
    adapter, monkeypatch, rows, align, dtype, routing_dtype, backend
):
    routes = torch.tensor(rows, dtype=torch.bool)
    tokens, experts = routes.shape
    counts = routes.sum(0).int()
    padded = int(((counts + align - 1) // align * align).sum())
    assert padded % experts == 0
    capacity = padded // experts
    inp = ((torch.arange(tokens * 4).reshape(tokens, 4) % 8 + 1) / 8).to(dtype)
    probs = ((torch.arange(tokens * experts).reshape(tokens, experts) % 4 + 1) / 8).to(dtype)
    dy = ((torch.arange(tokens * 4).reshape(tokens, 4) % 4 + 1) / 8).to(dtype)
    expected, expected_grad = torch.zeros_like(inp), torch.zeros_like(inp)
    # Independent oracle: only true routes retained by capacity contribute.
    for expert in range(experts):
        selected = [t for t in range(tokens) if rows[t][expert]][:capacity]
        for token in selected:
            expected[token] += inp[token] * probs[token, expert]
            expected_grad[token] += dy[token] * probs[token, expert]
    args = [
        inp,
        probs,
        dy,
        "pad_with_probs",
        ";".join(",".join(str(e) for e, routed in enumerate(row) if routed) for row in rows),
        experts,
        sum(rows[0]),
        routing_dtype,
        ",".join(str(c) for c in counts.tolist()),
        align,
    ]
    snapshots = [tensor.clone() for tensor in args[:3]]
    build_routes, build_counts = adapter._build_routing_map, adapter._build_tokens_per_expert
    monkeypatch.setattr(adapter, "_build_routing_map", lambda *a: build_routes(*a[:-1], "cpu"))
    monkeypatch.setattr(
        adapter, "_build_tokens_per_expert", lambda *a: build_counts(*a[:-1], "cpu")
    )
    captured = {}
    call_permute = adapter._call_mp_pad_probs

    def permute(*a):
        result = call_permute(*a)
        captured["raw"] = result[0]
        captured["snapshot"] = result[0].detach().clone()
        return result

    monkeypatch.setattr(adapter, "_call_mp_pad_probs", permute)
    output, gradient, contract = adapter.run_moe_unpermute_case(SimpleNamespace(args=args), backend)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(captured["raw"], captured["snapshot"], rtol=0, atol=0)
    for tensor, snapshot in zip(args[:3], snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    metadata = json.loads(bytes(contract.tolist()).decode("utf-8"))
    assert metadata["pad_offsets_shape"] == [experts]
