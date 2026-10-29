#!/usr/bin/env python3
"""Stage ⑤ REPORT — Deterministic report synthesis and flywheel logging.

Reads all pipeline artifacts (Stages ①-④), synthesizes final_report.json,
computes eval metrics, and appends to nightly_case_library.jsonl.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
CASE_LIBRARY = SCRIPTS_DIR / "nightly_case_library.jsonl"

ERROR_KEYWORDS = [
    "ERROR",
    "Traceback",
    "CRITICAL",
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
]

_EXCEPTION_PATTERN = re.compile(
    r"(FileNotFoundError|RuntimeError|ValueError|TypeError|AssertionError"
    r"|IndexError|KeyError|ConnectionError|TimeoutError|OSError"
    r"|AttributeError|ModuleNotFoundError|ImportError"
    r"|ChildFailedError|DistBackendError):",
)
_EXCLUDE_SUBSTRINGS = ["error_injection", "error_file"]


def _scan_log_for_errors(log_path: str) -> bool:
    """Return True if any real error indicators found in the log file.

    Same two-tier algorithm as nightly_verify.py and daily_build.sh:
      1. Concrete Python exception types (raise sites).
      2. Keyword scan with word boundaries (fallback).
    """
    log_file = Path(log_path)
    if not log_file.exists():
        return False
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    # Tier 1: concrete exception types
    for line in content.splitlines():
        if any(x in line for x in _EXCLUDE_SUBSTRINGS):
            continue
        if _EXCEPTION_PATTERN.search(line):
            return True

    # Tier 2: keyword scan with word boundaries
    for kw in ERROR_KEYWORDS:
        if " " in kw:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
        else:
            pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        if re.search(pattern, content):
            return True

    return False


def _stage_status(run: dict, stage_name: str) -> str:
    """Extract stage status from either string or dict format.

    Flywheel entries store stages as strings (e.g. ``"setup": "ok"``);
    in-memory final_report stores them as dicts (``"setup": {"status": "ok"}``).
    This helper accepts both shapes.
    """
    stage = run.get("stages", {}).get(stage_name, {})
    if isinstance(stage, str):
        return stage
    return stage.get("status", "unknown")


def _has_false_pass(final_report: dict) -> bool:
    """Scan all passing test stdout/stderr for residual error indicators."""
    training_data = final_report.get("_training_data", {})
    for tc in training_data.get("test_cases", []):
        if tc.get("status") != "pass":
            continue
        for log_key in ("stdout_path", "stderr_path"):
            log_path = tc.get(log_key, "")
            if log_path and _scan_log_for_errors(log_path):
                return True
    return False


def compute_evals(final_report: dict, eval_set: dict, historical_runs: list) -> dict:
    """Compute eval metrics against thresholds."""
    results = {}
    all_runs = historical_runs + [final_report]

    for ev in eval_set.get("evals", []):
        eid = ev["eval_id"]
        threshold = ev["threshold"]

        if eid == "N001":  # nightly_run_success_rate
            passes = sum(1 for r in all_runs if r.get("overall_status") == "pass")
            value = passes / len(all_runs) if all_runs else 0
        elif eid == "N002":  # setup_success_rate
            setups_ok = sum(1 for r in all_runs if _stage_status(r, "setup") == "ok")
            value = setups_ok / len(all_runs) if all_runs else 0
        elif eid == "N003":  # test_pass_rate
            # From current run only
            summary = final_report.get("stages", {}).get("launch", {}).get("tests", {})
            total = summary.get("total", 0)
            passed = summary.get("passed", 0)
            value = passed / total if total > 0 else 1.0
        elif eid == "N004":  # no_false_pass
            # Real check: scan all "pass" test case stdout/stderr for residual errors
            overall = final_report.get("overall_status") == "pass"
            value = (not _has_false_pass(final_report)) if overall else True
        elif eid == "N005":  # report_latency
            value = final_report.get("total_duration_seconds", 0)
        elif eid == "N006":  # diagnosis_accuracy (human eval)
            value = None  # Not auto-computed
        else:
            continue

        if value is not None:
            if ev["metric"] == "latency":
                passed = value <= threshold
            else:
                passed = value >= threshold
        else:
            passed = True  # Human-evaluated metrics pass by default

        results[eid] = {
            "name": ev["name"],
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }

    return results


def load_historical_runs() -> list:
    """Load all previous runs from the case library."""
    runs = []
    if CASE_LIBRARY.exists():
        for line in CASE_LIBRARY.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return runs


def append_to_flywheel(entry: dict):
    """Append a run entry to the nightly case library."""
    with open(CASE_LIBRARY, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Synthesize final report and log to flywheel")
    parser.add_argument("--env-report", required=True, help="env_report.json from Stage ①")
    parser.add_argument("--setup-report", required=True, help="setup_report.json from Stage ②")
    parser.add_argument("--training-log", required=True, help="training_log.json from Stage ③")
    parser.add_argument("--verify-report", required=True, help="verify_report.json from Stage ④")
    parser.add_argument("--sigcheck-report", default="", help="sigcheck_report.json from Stage ② (optional)")
    parser.add_argument("--eval-set", required=True, help="nightly_eval_set.json path")
    parser.add_argument("--output", required=True, help="Output final_report.json path")
    args = parser.parse_args()

    # Load all artifacts
    with open(args.env_report, encoding="utf-8") as f:
        env_data = json.load(f)
    with open(args.setup_report, encoding="utf-8") as f:
        setup_data = json.load(f)
    with open(args.training_log, encoding="utf-8") as f:
        training_data = json.load(f)
    with open(args.verify_report, encoding="utf-8") as f:
        verify_data = json.load(f)
    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)

    sigcheck_data = {}
    if args.sigcheck_report and Path(args.sigcheck_report).exists():
        with open(args.sigcheck_report, encoding="utf-8") as f:
            sigcheck_data = json.load(f)

    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Determine overall status
    fetch_ok = env_data.get("status") in ("ok", "partial")
    setup_ok = setup_data.get("status") == "ok"
    summary = training_data.get("summary", {})
    launch_ok = all(summary.get(k, 0) == 0 for k in ("failed", "timeout", "error"))
    verify_ok = verify_data.get("overall") == "pass"

    overall = "pass" if all([fetch_ok, setup_ok, launch_ok, verify_ok]) else "fail"

    final_report = {
        "run_date": report_date,
        "megatron_commit": env_data.get("megatron", {}).get("commit_sha", "unknown"),
        "megatron_branch": env_data.get("megatron", {}).get("branch", "unknown"),
        "last_known_good_commit": env_data.get("last_known_good_commit", ""),
        "te_path": "required",
        "overall_status": overall,
        "_training_data": training_data,
        "stages": {
            "fetch": {
                "status": env_data.get("status", "unknown"),
                "errors": env_data.get("errors", []),
            },
            "setup": {
                "status": setup_data.get("status", "unknown"),
                "errors": setup_data.get("errors", []),
            },
            "sigcheck": {
                "status": sigcheck_data.get("overall", "unknown"),
            },
            "launch": {
                "status": "pass" if launch_ok else "fail",
                "tests": training_data.get("summary", {}),
            },
            "verify": {
                "status": verify_data.get("overall", "unknown"),
                "checks": {k: v["passed"] for k, v in verify_data.get("checks", {}).items()},
                "failure_details": verify_data.get("failure_details", []),
            },
            "fetch_mrs": {
                "status": "skipped",
            },
            "diag": {
                "status": "skipped",
            },
            "patch": {
                "status": "skipped",
            },
        },
        "total_duration_seconds": training_data.get("summary", {}).get("total_duration_seconds", 0),
        "eval_results": {},
    }

    # Compute evals
    historical = load_historical_runs()
    eval_results = compute_evals(final_report, eval_set, historical)
    final_report["eval_results"] = eval_results

    # Write final report (strip internal _ prefixed keys)
    persist_report = {k: v for k, v in final_report.items() if not k.startswith("_")}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(persist_report, f, indent=2)

    # Append to flywheel
    flywheel_entry = {
        "type": "nightly_run",
        "run_date": report_date,
        "megatron_commit": final_report["megatron_commit"],
        "megatron_branch": final_report["megatron_branch"],
        "overall_status": overall,
        "stages": {
            "fetch": final_report["stages"]["fetch"]["status"],
            "setup": final_report["stages"]["setup"]["status"],
            "launch": final_report["stages"]["launch"]["status"],
            "verify": final_report["stages"]["verify"]["status"],
        },
        "total_duration_seconds": final_report["total_duration_seconds"],
        "failures": final_report["stages"]["verify"].get("failure_details", []),
        "eval_results": {k: v.get("value") for k, v in eval_results.items()},
    }
    append_to_flywheel(flywheel_entry)

    print(f"OK: final_report.json written to {args.output}")
    print(f"  Overall: {overall}")
    print("  Eval results:")
    for eid, ev in eval_results.items():
        status = "PASS" if ev["passed"] else "FAIL"
        val_str = f"{ev['value']:.3f}" if isinstance(ev["value"], float) else str(ev["value"])
        print(f"    {eid} {ev['name']}: {val_str} (threshold: {ev['threshold']}) — {status}")

    if overall == "pass":
        print("INFO: All stages passed. No human intervention needed.")
    else:
        print("ACTION: Verification failed. Proceed to DIAG stage for LLM diagnosis.")
        print(f"  Failure details: {final_report['stages']['verify'].get('failure_details', [])}")


if __name__ == "__main__":
    main()
