# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-card vocabulary-parallel CE smoke test (2 ranks via torchrun).

Ascend HCCL is unreliable under pytest in-process ``spawn``/``forkserver``
(parent already imported ``torch_npu``). Launch workers with ``torchrun`` instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("TE_PARALLEL_CE_FORCE_TRITON", "1")


def _npu_ok(min_devices: int = 2) -> bool:
    try:
        import torch_npu  # noqa: F401

        return (
            hasattr(torch, "npu")
            and torch.npu.is_available()
            and torch.npu.device_count() >= min_devices
        )
    except Exception:
        return False


def _worker(label_smoothing: float) -> None:
    """Rank body — must be launched by torchrun (RANK/LOCAL_RANK set)."""
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from transformer_engine.pytorch import parallel_cross_entropy

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    device = torch.device(f"npu:{local_rank}")
    torch.manual_seed(0)
    batch, seq, global_v = 2, 16, 512
    assert global_v % world_size == 0
    local_v = global_v // world_size

    full = torch.randn(batch, seq, global_v, device=device, dtype=torch.float32)
    target = torch.randint(0, global_v, (batch, seq), device=device)
    shard = full[:, :, rank * local_v : (rank + 1) * local_v].contiguous().requires_grad_(True)

    loss = parallel_cross_entropy(
        shard, target, label_smoothing, True, dist.group.WORLD, ignore_idx=-100
    )
    loss.backward()

    ok = True
    if rank == 0:
        full_t = full.clone().detach().requires_grad_(True)
        ref = torch.nn.functional.cross_entropy(
            full_t.reshape(-1, global_v),
            target.reshape(-1),
            label_smoothing=label_smoothing,
            reduction="mean",
        )
        ref.backward()
        ok = bool(torch.allclose(loss.detach(), ref.detach(), atol=1e-4, rtol=1e-4))
        g_full = full_t.grad[:, :, 0:local_v]
        ok = ok and bool(torch.allclose(shard.grad, g_full, atol=1e-3, rtol=1e-3))

    # All ranks must agree before exit (rank0 holds numeric check).
    flag = torch.tensor([1 if ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()
    if int(flag.item()) != 1:
        raise AssertionError(f"vocab-parallel CE mismatch (label_smoothing={label_smoothing})")


def _run_vp(label_smoothing: float) -> None:
    """Launch 2-rank torchrun on this file."""
    this = Path(__file__).resolve()
    env = os.environ.copy()
    env["TE_PARALLEL_CE_FORCE_TRITON"] = "1"
    env["TE_CE_DIST_LABEL_SMOOTHING"] = str(label_smoothing)
    # Drop parent VISIBLE remaps that confuse child logical ids.
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)

    port = "29881" if label_smoothing == 0.0 else "29882"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={port}",
        str(this),
        "--worker",
    ]

    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(this.parent.parent.parent),  # repo root
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"torchrun dist CE failed rc={proc.returncode}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\n"
            f"STDERR:\n{proc.stderr[-4000:]}"
        )


@pytest.mark.skipif(not _npu_ok(2), reason="need >=2 NPUs")
def test_vocab_parallel_two_ranks():
    _run_vp(0.0)


@pytest.mark.skipif(not _npu_ok(2), reason="need >=2 NPUs")
def test_vocab_parallel_label_smoothing():
    _run_vp(0.1)


if __name__ == "__main__":
    # Worker entry for torchrun.
    if "--worker" in sys.argv:
        ls = float(os.environ.get("TE_CE_DIST_LABEL_SMOOTHING", "0.0"))
        _worker(ls)
    else:
        raise SystemExit("use pytest or: torchrun ... this_file.py --worker")
