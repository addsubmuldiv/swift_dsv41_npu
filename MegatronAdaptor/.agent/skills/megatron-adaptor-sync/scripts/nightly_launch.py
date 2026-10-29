#!/usr/bin/env python3
"""Stage ③ LAUNCH — Deterministic test case runner. Zero LLM involvement.

Reads ``test/system_tests/test_cases.txt`` from the MegatronAdaptor
checkout — the current branch's case registry — and runs each test
script as a subprocess with timeout control.  Test scripts live in the
MA repo (e.g. ``test/system_tests/feature_tests/test_basic_training.sh``).

Test cases are registered in the MA repo — add or remove tests by
editing ``MegatronAdaptor/test/system_tests/test_cases.txt``.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent

# ── Test case defaults ─
DEFAULT_TIMEOUT_SECONDS = 600


def load_testcase_list(adaptor_path: str, branch: str) -> list[dict]:
    """Load test cases from the MA branch's ``test/system_tests/test_cases.txt``.

    The branch-owned registry evolves with the repo: each branch ships its own
    copy listing the enabled system tests (one script name per line; ``#``
    comments disable a case).  LAUNCH runs after FETCH has checked the adaptor
    out to the target branch, so reading from the adaptor root yields that
    branch's authoritative list.

    ``branch`` is kept for CLI compatibility but not used — the case list no
    longer distinguishes ``[common]`` vs ``[branch]`` sections.
    """
    config_path = Path(adaptor_path) / "test/system_tests/test_cases.txt"
    if not config_path.exists():
        print(f"WARNING: {config_path} not found — no tests to run")
        return []

    test_cases = []
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Bare script name → resolve inside feature_tests/; a line containing
        # '/' is treated as an adaptor-root-relative path already (future
        # sub-directory entries).
        script_rel = f"test/system_tests/feature_tests/{line}" if "/" not in line else line
        test_id = Path(script_rel).stem  # e.g. "test_basic_training"
        test_cases.append(
            {
                "test_id": test_id,
                "name": Path(script_rel).name,
                "script": script_rel,
                "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            }
        )

    return test_cases


def resolve_script_path(adaptor_path: str, script_rel: str) -> Path:
    """Resolve a test script path relative to the adaptor root."""
    return (Path(adaptor_path) / script_rel).resolve()


def extract_metrics_from_log(log_path: str) -> dict:
    """Extract training metrics from stdout log file.

    Handles multi-config scripts: sub-runs are split at ``iteration 1`` reset
    points, and each sub-run's initial/final loss plus grad-norm health are
    recorded separately. ``initial_loss``/``last_loss`` remain the first
    sub-run's initial and last sub-run's final for backward compatibility.
    """
    metrics = {
        "last_loss": None,
        "initial_loss": None,
        "throughput_tps": None,
        "peak_memory_gb": None,
        "training_iterations_completed": 0,
        "sub_runs": [],
        "grad_norm_min": None,
        "grad_norm_max": None,
        "grad_norm_zero_count": None,
        "grad_norm_count": None,
        "grad_norm_all_zero": None,
    }

    log_file = Path(log_path)
    if not log_file.exists():
        return metrics

    content = log_file.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    # Megatron training lines carry both ``lm loss`` and ``grad norm``.
    # \d[\d.]* ensures the captured value starts with a digit — not dots/progress bars.
    loss_pattern = re.compile(r'iteration\s+(\d+).*?\|?\s*lm loss[:\s]+(\d[\d.]*(?:[eE][+-]?\d+)?)')
    grad_norm_pattern = re.compile(r'grad norm[:\s]+(\d[\d.]*(?:[eE][+-]?\d+)?)')
    simple_loss = re.compile(r'loss[:\s]+(\d[\d.]*(?:[eE][+-]?\d+)?)', re.IGNORECASE)

    # Partition losses into sub-runs at ``iteration 1`` reset points.
    sub_runs = []  # list of {"losses": [...], "grad_norms": [...]}
    current: dict | None = None
    for line in lines:
        lm = loss_pattern.search(line)
        gm = grad_norm_pattern.search(line)
        if lm:
            it = int(lm.group(1))
            loss = float(lm.group(2))
            if it == 1 and current is not None and current["losses"]:  # pylint: disable=unsubscriptable-object
                # iteration reset → a new independent training run begins
                sub_runs.append(current)
                current = {"losses": [], "grad_norms": []}
            elif current is None:
                current = {"losses": [], "grad_norms": []}
            current["losses"].append(loss)
        elif gm is None:
            # Not a Megatron training line (no grad norm); only then fall back
            # to the bare ``loss:`` pattern so we don't double-count.
            m = simple_loss.search(line)
            if m:
                if current is None:
                    current = {"losses": [], "grad_norms": []}
                current["losses"].append(float(m.group(1)))
        if gm:
            g = float(gm.group(1))
            if current is None:
                current = {"losses": [], "grad_norms": []}
            current["grad_norms"].append(g)
    if current is not None and current["losses"]:
        sub_runs.append(current)

    all_losses = [loss for r in sub_runs for loss in r["losses"]]
    all_grad_norms = [g for r in sub_runs for g in r["grad_norms"]]

    if all_losses:
        metrics["initial_loss"] = all_losses[0]
        metrics["last_loss"] = all_losses[-1]

    # Per sub-run records for verify-stage per-config convergence checks.
    for r in sub_runs:
        gns = r["grad_norms"]
        metrics["sub_runs"].append(
            {
                "initial_loss": r["losses"][0] if r["losses"] else None,
                "last_loss": r["losses"][-1] if r["losses"] else None,
                "num_losses": len(r["losses"]),
                "grad_norm_min": min(gns) if gns else None,
                "grad_norm_max": max(gns) if gns else None,
                "grad_norm_zero_count": sum(1 for g in gns if g == 0) if gns else None,
                "grad_norm_count": len(gns),
            }
        )

    # Global grad-norm health (True only when data exists AND all zero).
    if all_grad_norms:
        metrics["grad_norm_min"] = min(all_grad_norms)
        metrics["grad_norm_max"] = max(all_grad_norms)
        metrics["grad_norm_zero_count"] = sum(1 for g in all_grad_norms if g == 0)
        metrics["grad_norm_count"] = len(all_grad_norms)
        metrics["grad_norm_all_zero"] = metrics["grad_norm_zero_count"] == len(all_grad_norms)

    # Pattern: throughput (tokens/sec)
    # \d[\d,.]* ensures the value starts with a digit — not dots/progress bars.
    tput_pattern = re.compile(r'(?:throughput|tokens?/s(?:ec)?)[:\s]+(\d[\d,.]*)', re.IGNORECASE)
    for line in lines:
        m = tput_pattern.search(line)
        if m:
            metrics["throughput_tps"] = float(m.group(1).replace(",", ""))
            break

    # Pattern: iteration count
    iter_pattern = re.compile(r'iteration\s+(\d+)')
    max_iter = 0
    for line in lines:
        for m in iter_pattern.finditer(line):
            it = int(m.group(1))
            max_iter = max(max_iter, it)
    metrics["training_iterations_completed"] = max_iter

    # Pattern: memory usage
    mem_pattern = re.compile(
        r'(?:peak memory|memory used|allocated)[:\s]+([\d,.]+)\s*(?:GB|GiB|MB|MiB)?', re.IGNORECASE
    )
    for line in lines:
        m = mem_pattern.search(line)
        if m:
            val = float(m.group(1).replace(",", ""))
            if "MB" in line.upper() or "MiB" in line:
                val = val / 1024.0
            metrics["peak_memory_gb"] = val
            break

    return metrics


def run_test_case(tc: dict, adaptor_path: str, state_dir: str, log_dir: str) -> dict:
    """Run a single test case and return structured result."""
    test_id = tc["test_id"]
    test_log_dir = Path(log_dir) / test_id
    test_log_dir.mkdir(parents=True, exist_ok=True)

    script_path = resolve_script_path(adaptor_path, tc["script"])
    if not script_path.exists():
        return {
            "test_id": test_id,
            "name": tc["name"],
            "script": tc["script"],
            "status": "skipped",
            "errors": [f"Script not found: {script_path}"],
        }

    # Build environment — only pass the log dir; test scripts own their config
    env = os.environ.copy()
    env["MEGATRON_NIGHTLY_LOG_DIR"] = str(test_log_dir)

    timeout = tc.get("timeout_seconds", 3600)

    start_time = datetime.now(timezone.utc)
    stdout_path = test_log_dir / "stdout.log"
    stderr_path = test_log_dir / "stderr.log"

    print(f"  [{test_id}] Starting: {tc['name']}")
    print(f"  [{test_id}] Script: {script_path}")
    print(f"  [{test_id}] Log dir: {test_log_dir}")

    try:
        with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
            # start_new_session puts the child in its own process group so
            # os.killpg can reap the whole tree on timeout (same as preexec_fn=setsid).
            with subprocess.Popen(  # nosec B607 — bash on PATH, trusted
                ["bash", "-e", str(script_path)],
                stdout=out_f,
                stderr=err_f,
                env=env,
                cwd="/home/ma_daily_ci/workspace/Megatron-LM",
                start_new_session=True,
            ) as proc:
                try:
                    proc.wait(timeout=timeout)
                    exit_code = proc.returncode
                    status = "pass" if exit_code == 0 else "fail"
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait()
                    exit_code = -1
                    status = "timeout"
    except Exception as e:
        exit_code = -2
        status = "error"
        with open(stderr_path, "a", encoding="utf-8") as err_f:
            err_f.write(f"\n[nightly_launch] Exception: {e}\n")

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    # Extract metrics
    metrics = extract_metrics_from_log(str(stdout_path))

    errors = []
    if status in ("fail", "timeout", "error"):
        errors = [f"Exit code: {exit_code}"]

    print(
        f"  [{test_id}] Result: {status} (exit={exit_code}, {duration:.1f}s, loss: {metrics.get('last_loss', 'N/A')})"
    )

    return {
        "test_id": test_id,
        "name": tc["name"],
        "script": tc["script"],
        "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": duration,
        "exit_code": exit_code,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "extracted_metrics": metrics,
        "status": status,
        "errors": errors,
    }


def run_envcheck(state_dir: str) -> dict:
    """Pre-flight NPU occupancy check (first action inside LAUNCH).

    共享 NPU 被其他用户训练任务占用（如 2026-08-21 admin123 的 mindspeed_mm
    训练占默认 HCCL socket 端口 16666 → 45/45 用例在 hcclCommInitRootInfoConfig()
    报 error 7 + EI0020）时，所有分布式训练测试都会假失败——环境问题非代码回归。
    冒烟测试只做单进程 device init、不初始化 HCCL 集合通信/不绑定端口，不受影响；
    因此在这里（LAUNCH 第一步、首个分布式用例前）拦截最合适：省掉整轮假失败，
    也不触发 ⑤⑥⑦⑧ 诊断和 Issue/PR。

    返回 envcheck_report dict；探针本身失败时返回 {}（视为 free，绝不阻断夜航）。
    """
    out = Path(state_dir) / "envcheck_report.json"
    try:
        subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "nightly_envcheck.py"), "--output", str(out)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        try:
            with open(out, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    except Exception as e:
        print(f"  [envcheck] WARNING: pre-flight NPU check failed ({e}) — treating as free")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run nightly training test cases on NPU")
    parser.add_argument(
        "--adaptor-path",
        required=True,
        help="Path to MegatronAdaptor checkout (contains test/system_tests/test_cases.txt)",
    )
    parser.add_argument("--branch", required=True, help="Megatron branch being tested (e.g. main, core_r0.18.0)")
    parser.add_argument(
        "--state-dir", default="/home/ma_daily_ci/workspace/output/state", help="State directory for pipeline artifacts"
    )
    parser.add_argument(
        "--log-dir", default="/home/ma_daily_ci/workspace/output/logs", help="Log output directory for test cases"
    )
    parser.add_argument("--output", required=True, help="Output training_log.json path")
    args = parser.parse_args()

    # ── ①.5 ENVCHECK（LAUNCH 第一步）：NPU 被其他训练任务占用 → 整体 skip ──
    # 命中时写 env_busy 标记的 training_log.json 并返回 0，编排器据此把分支标为
    # skip（不跑用例、不进 ⑤⑥⑦⑧ 诊断、不建 Issue/PR）。
    envcheck = run_envcheck(args.state_dir)
    if envcheck.get("occupied"):
        print("\n=== NPU OCCUPIED — SKIPPING LAUNCH ===")
        print(f"  Reason: {envcheck.get('reason', 'npu_occupied')}")
        for p in envcheck.get("signals", {}).get("other_user_training_processes", [])[:5]:
            print(f"    - {p.get('user')}: {p.get('cmd', '')[:110]}")
        print("  No tests run. Environment issue (not a code regression).")
        report = {
            "launch_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "env_busy": True,
            "env_busy_reason": envcheck.get("reason", "npu_occupied"),
            "env_busy_signals": envcheck.get("signals", {}),
            "test_cases": [],
            "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"OK: env_busy recorded in {args.output}")
        return

    test_cases = load_testcase_list(args.adaptor_path, args.branch)
    if not test_cases:
        print("INFO: No test cases found. Nothing to run.")
        report = {
            "launch_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "test_cases": [],
            "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return

    # ── Symlink MA test/ into Megatron-LM cwd ────────────────────────
    # Test scripts source "test/system_tests/env_npu.sh" relative to CWD
    # (Megatron-LM), but the file lives in MegatronAdaptor.  A directory
    # symlink keeps the single source of truth while satisfying relative paths.
    _test_src = (Path(args.adaptor_path) / "test").resolve()
    _test_link = Path("/home/ma_daily_ci/workspace/Megatron-LM/test")
    if _test_src.is_dir():
        if _test_link.is_symlink():
            _test_link.unlink()
        if not _test_link.exists():
            _test_link.symlink_to(_test_src)
            print(f"Symlink: {_test_link} -> {_test_src}")
    else:
        print(f"WARNING: test/ directory not found at {_test_src}")

    launch_time = datetime.now(timezone.utc)
    results = []

    print(f"=== Nightly Launch: {len(test_cases)} test case(s) ===")
    start = time.time()

    for tc in test_cases:
        result = run_test_case(tc, args.adaptor_path, args.state_dir, args.log_dir)
        results.append(result)

    elapsed = time.time() - start

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errored = sum(1 for r in results if r["status"] == "error")

    # ── Retry failed tests ──────────────────────────────────────────
    first_failures = [r for r in results if r["status"] in ("fail", "timeout", "error")]
    retry_passed = 0
    retry_failed = 0

    if first_failures:
        print(f"\n=== Retrying {len(first_failures)} failed test(s) after 10s cooldown ===")
        time.sleep(10)

        for idx, orig in enumerate(first_failures):
            test_id = orig["test_id"]
            tc = next((t for t in test_cases if t["test_id"] == test_id), None)
            if tc is None:
                continue

            print(f"  [{test_id}] Retry {idx + 1}/{len(first_failures)}...")
            retry_result = run_test_case(tc, args.adaptor_path, args.state_dir, args.log_dir)
            retry_result["_retry_of"] = test_id

            if retry_result["status"] == "pass":
                retry_passed += 1
                for i, r in enumerate(results):
                    if r["test_id"] == test_id:
                        results[i] = retry_result
                        break
                print(f"  [{test_id}] Retry PASSED ✅")
            else:
                retry_failed += 1
                for i, r in enumerate(results):
                    if r["test_id"] == test_id:
                        results[i] = retry_result
                        break
                print(f"  [{test_id}] Retry FAILED ❌")

    # Recompute summary after retries
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    errored = sum(1 for r in results if r["status"] == "error")

    report = {
        "launch_time": launch_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_cases": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "timeout": timeout_count,
            "skipped": skipped,
            "error": errored,
            "total_duration_seconds": elapsed,
            "retry_passed": retry_passed,
            "retry_failed": retry_failed,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Launch Summary ===")
    print(
        f"  Total: {total} | Passed: {passed} | Failed: {failed} | "
        f"Timeout: {timeout_count} | Skipped: {skipped} | Error: {errored}"
    )
    if retry_passed or retry_failed:
        print(f"  Retry: {retry_passed} passed, {retry_failed} still failed")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"OK: training_log.json written to {args.output}")


if __name__ == "__main__":
    main()
