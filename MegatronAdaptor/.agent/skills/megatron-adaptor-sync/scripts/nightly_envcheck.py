#!/usr/bin/env python3
"""Stage ①.5 ENVCHECK — Pre-flight NPU occupancy check.

Detects whether the shared NPUs are occupied by another training job BEFORE
the expensive fetch/setup/launch cycle runs.  When occupied, the orchestrator
skips the branch instead of burning ~20 min and generating false failures,
then bogus Issues & PRs for an environment problem that no patch can fix.

Root case that motivated this (2026-08-21): a co-tenant mindspeed_mm training
job started 03:47 held the default HCCL NPU socket port 16666, so every
nightly test crashed at hcclCommInitRootInfoConfig() with error code 7
("HCCL function error") + EI0020 "port ... already been bound" — 45/45 tests
failed across all branches, none of which was a Megatron regression
(core branches had zero code delta).

Decisive signals (any hit -> occupied=True):
  1. HCCL NPU socket port already bound (default 16666; honours
     HCCL_NPU_SOCKET_PORT_RANGE when set).
  2. Training processes owned by a user OTHER than the nightly's identity
     (patterns: torchrun / torch.distributed.run / pretrain* / trainer.py /
     mindspeed / fsdp / megatron).
Context (non-decisive, reported only):
  3. npu-smi excerpt so a human sees AICore / HBM usage.

Exit codes:
  0  NPU free (or check inconclusive — no decisive signal)
  2  NPU occupied (at least one decisive signal hit)

Usage:
  nightly_envcheck.py --output <envcheck_report.json>
"""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Nightly's own identity — processes owned by these users are never treated
# as an external occupation (e.g. the nightly's own leftover between branches).
SELF_USERS = {"ma_daily_ci", getpass.getuser()}

# Default HCCL NPU socket port (also the default for the Ascend HCCL NIC).
DEFAULT_HCCL_PORT = 16666

# Command-line patterns that identify a training job.
# ⚠️ 必须是脚本/可执行名级别，不能用裸 "megatron"/"fsdp"/"pretrain" 等宽泛子串——
# skill 路径含 "megatron-adaptor-sync"、MLM 克隆含 "Megatron-LM/megatron/..."，
# 宽模式会让 root 包装进程（cron 守卫的 su 等）cmdline 命中而误报 occupied。
# 真实训练进程（torchrun / dist.run / trainer.py / mindspeed / pretrain_gpt.py）
# 由这些具体入口名命中；admin123 案例 = "mindspeed"+"trainer.py"+"torch.distributed.run"。
TRAINING_PATTERNS = (
    "torchrun",
    "torch.distributed.run",
    "pretrain_gpt",
    "pretrain_llama",
    "pretrain_mistral",
    "trainer.py",
    "mindspeed",
    "run_pretraining",
    "run_clm",
    "megatron/pretraining",  # 精确到 MLM 训练脚本路径段，避免匹配 skill/克隆路径
)

_NPU_SMI_CANDIDATES = [
    "npu-smi",  # if on PATH
    "/usr/local/Ascend/driver/tools/npu-smi",
    "/usr/local/Ascend/driver/bin/npu-smi",
    "/usr/local/Ascend/ascend-toolkit/latest/bin/npu-smi",
]


def _collect_hccl_ports() -> set:
    """Default HCCL socket port + any port range from HCCL_NPU_SOCKET_PORT_RANGE."""
    ports = {DEFAULT_HCCL_PORT}
    env_range = os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", "").strip()
    for part in env_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
                if 0 < lo <= hi <= 65535 and hi - lo <= 1000:
                    ports.update(range(lo, hi + 1))
            except ValueError:
                continue
        elif part.isdigit() and 0 < int(part) <= 65535:
            ports.add(int(part))
    return ports


def detect_port_occupied(ports: set) -> list:
    """Return the HCCL socket ports currently LISTEN-bound by any process."""
    try:
        out = subprocess.run(  # nosec B607 — ss on PATH, trusted
            ["ss", "-ltn"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except Exception:
        out = ""
    if not out:
        return []  # inconclusive — treated as not occupied (no false abort)
    # ss line:  LISTEN  0  4096  192.168.2.192:16666  *:*  ...
    return sorted(p for p in sorted(ports) if re.search(rf":{p}\s", out))


def detect_other_training_processes() -> list:
    """Return training processes owned by a user outside the nightly's identity."""
    try:
        out = subprocess.run(  # nosec B607 — ps on PATH, trusted
            ["ps", "-eo", "user:40,args"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except Exception:
        out = ""
    hits = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            user, args = line.split(None, 1)
        except ValueError:
            continue
        if not user or user in SELF_USERS:
            continue
        if any(p in args for p in TRAINING_PATTERNS):
            hits.append({"user": user, "cmd": args[:200]})
    return hits


def npu_smi_excerpt() -> str:
    """Best-effort npu-smi excerpt for human context. Empty string if unavailable."""
    for cand in _NPU_SMI_CANDIDATES:
        try:
            r = subprocess.run([cand, "info"], capture_output=True, text=True, timeout=15, check=False)
            if r.returncode == 0 and r.stdout.strip():
                lines = r.stdout.splitlines()
                # Summary table header + per-card rows; cap to keep the report small.
                head = [ln for ln in lines[:24] if "----" in ln or "|" in ln]
                return "\n".join(head[-16:]) if head else ""
        except Exception:
            continue
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight NPU occupancy check")
    parser.add_argument("--output", required=True, help="Path to envcheck_report.json")
    args = parser.parse_args()

    ports = _collect_hccl_ports()
    ports_bound = detect_port_occupied(ports)
    procs = detect_other_training_processes()
    npu_smi = npu_smi_excerpt()

    occupied = bool(ports_bound or procs)
    reasons = []
    if ports_bound:
        reasons.append("hccL_socket_port_bound:" + ",".join(str(p) for p in ports_bound))
    if procs:
        owners = sorted({p["user"] for p in procs})
        reasons.append("other_user_training_process:" + ",".join(owners))

    report = {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "occupied": occupied,
        "reason": ";".join(reasons) if reasons else "ok",
        "signals": {
            "hccl_ports_bound": ports_bound,
            "other_user_training_processes": procs[:10],  # cap report size
            "other_user_process_count": len(procs),
        },
        "npu_smi": npu_smi,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    status = "OCCUPIED" if occupied else "FREE"
    print(f"[envcheck] {status} -> {out}")
    print(f"[envcheck]   ports_bound={ports_bound}")
    print(f"[envcheck]   other_user_training_procs={len(procs)}")
    for p in procs[:5]:
        print(f"[envcheck]     - {p['user']}: {p['cmd'][:100]}")
    if npu_smi:
        print(f"[envcheck]   npu-smi excerpt ({len(npu_smi)} chars)")

    return 2 if occupied else 0


if __name__ == "__main__":
    sys.exit(main())
