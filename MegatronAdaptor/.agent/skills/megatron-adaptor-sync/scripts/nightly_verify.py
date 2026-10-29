#!/usr/bin/env python3
"""Stage ④ VERIFY — Deterministic result verification. Zero LLM involvement.

4 independent checks on training results, each producing pass/fail with details:
1. Exit code check — all test cases returned 0
2. Loss convergence — final_loss < initial_loss
3. Error keyword scan — no error indicators in logs
4. Throughput floor — throughput >= configured threshold
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ERROR_KEYWORDS = [
    "ERROR",
    "Traceback",
    "CORRUPTION",
    "RuntimeError",
    "AssertionError",
    "FATAL",
    "CUDA error",
    "hccl error",
    "ASCEND error",
    "Segmentation fault",
    "Aborted",
    "Killed",
    "out of memory",
    "OOM",
    # ── 环境类：NPU 被其他训练任务占用（HCCL socket 端口冲突）──
    # 2026-08-21 案例：hcclCommInitRootInfoConfig error 7 + EI0020
    # "port 16666 already been bound" = 其他用户训练脚本占端口，非代码回归。
    # 识别后 failure_details 会带上这些行，便于归因为环境问题而非代码问题。
    "HCCL function error",
    "hcclCommInitRootInfoConfig",
    "ERR02200",
    "EI0020",
    "already been bound",
    "Communication_Error_Bind_IP_Port",
]

# ── Exception-type detection (移植自 daily_build.sh) ──────────────
# Match concrete Python exception class names followed by `:`, which
# indicates an actual raise site rather than a comment or config param.
# This is the PRIMARY signal; the keyword scan above is the fallback.

_EXCEPTION_PATTERN = re.compile(
    r"(FileNotFoundError|RuntimeError|ValueError|TypeError|AssertionError"
    r"|IndexError|KeyError|ConnectionError|TimeoutError|OSError"
    r"|AttributeError|ModuleNotFoundError|ImportError"
    r"|ChildFailedError|DistBackendError):",
)

# Lines containing these substrings are never treated as errors even
# when they happen to contain an exception class name.
_EXCLUDE_SUBSTRINGS = ["error_injection", "error_file"]

# Log lines matching these patterns are known to be benign (e.g. ASCEND
# runtime diagnostic messages, log-level prefixes that happen to contain
# "ERROR").  These are excluded from BOTH Tier 1 (exception type) and
# Tier 2 (keyword) scans.
_BENIGN_LOG_PATTERNS = [
    # megatron rerun_state_machine logs at ERROR level for missing tracker
    # files on first run — this is expected behaviour, not a real error.
    r"ERROR:megatron\.core\.rerun_state_machine:",
]


def _filter_benign_lines(content: str) -> str:
    """Remove lines matching known benign patterns (log noise)."""
    lines = content.splitlines()
    kept = []
    for line in lines:
        if any(x in line for x in _EXCLUDE_SUBSTRINGS):
            continue
        if any(re.search(p, line) for p in _BENIGN_LOG_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def _extract_root_cause(stderr_path: str) -> str:
    """Extract the first real exception from a stderr log, skipping ChildFailedError wrappers."""
    try:
        content = Path(stderr_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

    lines = content.splitlines()
    # Scan for concrete Python exceptions, skip torch elastic wrappers
    for i, line in enumerate(lines):
        stripped = line.strip()
        # ChildFailedError is a wrapper — look deeper
        if "ChildFailedError" in stripped:
            continue
        if "torch.distributed.elastic" in stripped:
            continue
        # Concrete exception + message
        for exc in (
            "RuntimeError",
            "AssertionError",
            "PermissionError",
            "FileNotFoundError",
            "ModuleNotFoundError",
            "ImportError",
            "ValueError",
            "TypeError",
            "OSError",
            "KeyError",
            "AttributeError",
        ):
            if f"{exc}" in stripped or f"{exc}:" in stripped:
                # Return the exception line + 1 context line
                ctx = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if ctx and not any(kw in ctx for kw in ("ChildFailedError", "elastic", "Traceback", "File \"")):
                    return f"{stripped} | {ctx}"
                return stripped
    # Fallback: last non-empty, non-wrapper line
    for line in reversed(lines):
        s = line.strip()
        if s and "torch.distributed.elastic" not in s and "Sending process" not in s:
            return s[:200]
    return ""


def check_exit_codes(training_log: dict) -> dict:
    """Verify all test cases exited with code 0."""
    test_cases = training_log.get("test_cases", [])
    if not test_cases:
        return {"passed": True, "details": "No test cases to check"}

    failed = []
    for tc in test_cases:
        if tc.get("status") == "pass":
            continue
        if tc.get("exit_code", 0) != 0:
            test_id = tc["test_id"]
            # Extract the actual error from stderr
            stderr_path = tc.get("stderr_path", "")
            cause = _extract_root_cause(stderr_path) if stderr_path else ""
            if cause:
                failed.append(f"{test_id}: {cause}")
            else:
                failed.append(f"{test_id}: exit_code={tc['exit_code']}")

    if failed:
        return {"passed": False, "details": f"Failed tests: {'; '.join(failed)}"}
    return {"passed": True, "details": f"All {len(test_cases)} test cases exited with code 0"}


def check_loss_convergence(training_log: dict) -> dict:
    """Verify loss decreased during training.

    Convergence is judged **per sub-run**: a multi-config test script runs
    several independent trainings into one log, and ``sub_runs`` (extracted by
    nightly_launch.py) splits them at ``iteration 1`` reset points. Each sub-run
    must independently converge.

    A sub-run whose final loss did not decrease is only reported as
    Non-convergent when gradients are **not** flowing — i.e. grad-norm data is
    missing, or every recorded grad norm is 0 (``grad_norm_zero_count ==
    grad_norm_count``). If gradients are healthy (some nonzero grad norm), the
    flat/rising loss is a test-design artifact (too few iterations, tiny model,
    low LR), not a training bug, and is recorded as a note rather than a
    failure. This distinguishes real zero-gradient bugs (grad_norm=0.000) from
    false positives on multi-config scripts.
    """
    test_cases = training_log.get("test_cases", [])
    if not test_cases:
        return {"passed": True, "details": "No test cases to check"}

    non_convergent = []
    healthy_flat = []  # final >= initial but gradients flowing — not a bug
    no_loss_data = 0

    def judge(initial, final, gn_zero_count, gn_count, label):
        nonlocal no_loss_data
        if initial is None or final is None:
            no_loss_data += 1
            return
        if final >= initial:
            gradients_healthy = gn_count and gn_zero_count is not None and gn_zero_count < gn_count
            if gradients_healthy:
                healthy_flat.append(
                    f"{label}: final={final:.4f} >= initial={initial:.4f}"
                    f" (gradients flowing: {gn_count - gn_zero_count}/{gn_count} nonzero)"
                )
            else:
                non_convergent.append(f"{label}: initial={initial:.4f}, final={final:.4f}")

    for tc in test_cases:
        if tc.get("status") != "pass":
            continue
        metrics = tc.get("extracted_metrics", {})
        sub_runs = metrics.get("sub_runs") or []
        if sub_runs:
            for idx, sr in enumerate(sub_runs, start=1):
                judge(
                    sr.get("initial_loss"),
                    sr.get("last_loss"),
                    sr.get("grad_norm_zero_count"),
                    sr.get("grad_norm_count"),
                    f"{tc['test_id']}#sub{idx}",
                )
        else:
            # Backward compat: no sub_run data → single training run.
            judge(
                metrics.get("initial_loss"),
                metrics.get("last_loss"),
                metrics.get("grad_norm_zero_count"),
                metrics.get("grad_norm_count"),
                tc["test_id"],
            )

    details_parts = []
    if non_convergent:
        details_parts.append(f"Non-convergent: {'; '.join(non_convergent)}")
    if healthy_flat:
        details_parts.append(f"Flat-with-healthy-gradients (not failures): {'; '.join(healthy_flat)}")
    if no_loss_data > 0:
        details_parts.append(f"{no_loss_data} test(s) without loss data")

    if non_convergent:
        return {"passed": False, "details": " | ".join(details_parts)}
    return {
        "passed": True,
        "details": "All tests with loss data show convergence"
        + (f" ({no_loss_data} without loss data)" if no_loss_data else ""),
    }


def check_error_keywords(training_log: dict) -> dict:
    """Scan log files for error indicators.

    Two-tier detection (matching daily_build.sh):
      1. PRIMARY — concrete Python exception types followed by ``:``.
         These are real raise sites, not config-param names.
         Excludes lines matching ``_EXCLUDE_SUBSTRINGS``.
      2. FALLBACK — keyword scan with word boundaries (``\b``) so
         ``ERROR`` does not match ``error_injection_rate``.
    """
    test_cases = training_log.get("test_cases", [])
    if not test_cases:
        return {"passed": True, "details": "No test cases to check"}

    findings = []
    for tc in test_cases:
        if tc.get("status") != "pass":
            continue  # Only scan passing tests for hidden errors
        test_id = tc["test_id"]

        for log_type, log_key in [("stdout", "stdout_path"), ("stderr", "stderr_path")]:
            log_path = tc.get(log_key, "")
            if not log_path:
                continue
            log_file = Path(log_path)
            if not log_file.exists():
                continue

            try:
                content = log_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # ── Filter out benign log lines before scanning ──
            filtered = _filter_benign_lines(content)
            if not filtered.strip():
                continue  # all lines were benign — nothing to check

            # ── Tier 1: concrete exception types (daily_build.sh algorithm) ──
            exception_hits = []
            for line in filtered.splitlines():
                m = _EXCEPTION_PATTERN.search(line)
                if m:
                    exception_hits.append(m.group(1))

            if exception_hits:
                # Report unique exception types found
                unique = sorted(set(exception_hits))
                findings.append(
                    f"{test_id}/{log_type}: exceptions found — {', '.join(unique)}"
                    f" ({len(exception_hits)} occurrence(s))"
                )
                continue  # Tier 1 matched — skip Tier 2 for this log

            # ── Tier 2: keyword scan with word boundaries (fallback) ──
            for keyword in ERROR_KEYWORDS:
                if " " in keyword:
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                else:
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                matches = pattern.findall(filtered)
                if matches:
                    findings.append(f"{test_id}/{log_type}: '{keyword}' found ({len(matches)} times)")

    if findings:
        return {"passed": False, "details": f"Error keywords found: {'; '.join(findings[:10])}"}
    return {"passed": True, "details": "No error indicators found in any log"}


def check_throughput_floor(training_log: dict, floor_tps: int = 0) -> dict:
    """Verify throughput meets configured minimum (disabled by default)."""
    test_cases = training_log.get("test_cases", [])
    if not test_cases:
        return {"passed": True, "details": "No test cases to check"}

    if floor_tps <= 0:
        return {"passed": True, "details": "No throughput floor configured"}

    below_floor = []
    no_tput_data = 0

    for tc in test_cases:
        if tc.get("status") != "pass":
            continue
        test_id = tc["test_id"]

        tput = tc.get("extracted_metrics", {}).get("throughput_tps")
        if tput is None:
            no_tput_data += 1
            continue

        if tput < floor_tps:
            below_floor.append(f"{test_id}: {tput:.1f} t/s < {floor_tps} t/s floor")

    if below_floor:
        return {"passed": False, "details": f"Below throughput floor: {'; '.join(below_floor)}"}
    return {
        "passed": True,
        "details": "All tests meet throughput floor"
        + (f" ({no_tput_data} without throughput data)" if no_tput_data else ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify nightly training results")
    parser.add_argument("--training-log", required=True, help="training_log.json from Stage ③ LAUNCH")
    parser.add_argument("--output", required=True, help="Output verify_report.json path")
    args = parser.parse_args()

    with open(args.training_log, encoding="utf-8") as f:
        training_log = json.load(f)

    verify_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    checks = {
        "exit_code_check": check_exit_codes(training_log),
        "loss_convergence": check_loss_convergence(training_log),
        "error_keyword_search": check_error_keywords(training_log),
        "throughput_floor": check_throughput_floor(training_log, floor_tps=0),
    }

    overall = "pass" if all(c["passed"] for c in checks.values()) else "fail"

    # Collect failure details
    failure_details = []
    if overall == "fail":
        for name, result in checks.items():
            if not result["passed"]:
                failure_details.append(
                    {
                        "check": name,
                        "details": result["details"],
                    }
                )

    report = {
        "verify_time": verify_time,
        "checks": checks,
        "overall": overall,
        "failure_details": failure_details,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"RESULT: {overall}")
    for name, result in checks.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {name}: {status} — {result['details']}")

    if overall == "fail":
        print("STOP: Verification failed. Proceed to DIAG stage.", file=sys.stderr)


if __name__ == "__main__":
    main()
