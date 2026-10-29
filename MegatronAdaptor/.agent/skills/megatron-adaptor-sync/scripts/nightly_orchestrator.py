#!/usr/bin/env python3
"""Multi-branch nightly pipeline orchestrator.

Per-branch full cycle model:
  Each branch combo runs the complete ①→⑧ flywheel independently before
  moving to the next branch.  Most of the time ①→④ passes directly and
  the diagnosis path (⑤→⑥→⑦→③) is never triggered.  When a branch fails,
  ⑤ FETCH_MRS and ⑥ DIAG prompt build run immediately within the same
  context — no context loss from cross-phase serialization.

  The LLM interaction for ⑥ DIAG analysis, ⑦ PATCH generation, AST fix
  loop, and ③ LAUNCH retry is handled by the Claude agent following
  SKILL.md instructions.  The orchestrator builds the prompts and halts.

Branch timeout: 60 minutes wall-clock per branch.  Exceeded → terminated,
recorded as failure, next branch proceeds.

Modeled after run_daily_build.py in the MA daily-build skill.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPTS_DIR.parent

# Branch combos（唯一来源 scripts/branch_matrix.json，不再硬编码）
BRANCH_MATRIX_PATH = SCRIPTS_DIR / "branch_matrix.json"


def _load_branch_combos() -> list:
    """从 branch_matrix.json 读取分支组合；缺失/损坏即报错退出（关键配置不静默降级）。"""
    try:
        with open(BRANCH_MATRIX_PATH, encoding="utf-8") as f:
            combos = json.load(f)["branch_combos"]
        if not isinstance(combos, list) or not combos:
            raise ValueError("branch_combos 为空或格式错误")
        return combos
    except Exception as e:
        raise SystemExit(f"[nightly_orchestrator] 无法读取 {BRANCH_MATRIX_PATH}: {e}")


BRANCH_COMBOS = _load_branch_combos()

# Shared paths
WORK_DIR = Path("/home/ma_daily_ci/workspace")
SETUP_ENV_SH = str(SCRIPTS_DIR / "setup_env.sh")

# RUN_DATE / OUTPUT_DIR are set in main() after parsing --date.
OUTPUT_DIR: Path = None  # type: ignore[assignment]


def _resolve_run_date(date_arg: str | None) -> str:
    """Resolve RUN_DATE: explicit --date > UTC fallback."""
    if date_arg:
        return date_arg
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# Per-branch wall-clock timeout (seconds)
BRANCH_TIMEOUT = 3600  # 60 minutes


def run_stage(
    branch: str,
    conda_env: str,
    script: str,
    args: list,
    timeout: int = 600,
    deadline: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a single pipeline stage with the correct conda + CANN environment.

    ``deadline`` is the branch's monotonic end time.  The subprocess timeout
    is capped at the remaining branch budget, so a hung stage (e.g. LAUNCH
    training) is killed at the branch limit instead of running past it; a
    stage that only exceeds its own ``timeout`` ceiling still raises
    TimeoutExpired unchanged.
    """
    state_dir = OUTPUT_DIR / branch / "state"
    log_dir = OUTPUT_DIR / branch / "logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    script_path = SCRIPTS_DIR / script
    cmd_parts = [str(script_path)] + args

    # Substitute placeholders in args
    resolved = []
    for a in cmd_parts:
        a = a.replace("__STATE_DIR__", str(state_dir))
        a = a.replace("__LOG_DIR__", str(log_dir))
        a = a.replace("__BRANCH__", branch)
        resolved.append(a)

    shell_cmd = (
        f"source {SETUP_ENV_SH} {conda_env} && "
        f"export PYTHONPATH={WORK_DIR}/Megatron-LM:$PYTHONPATH && "
        "python3 " + " ".join(resolved)
    )

    print(f"  [{branch}] $ {script} {' '.join(args)}")
    timeout_used = timeout
    budget_capped = False
    if deadline is not None:
        remaining = int(deadline - time.monotonic())
        if remaining < timeout:
            # Cap at the branch's remaining budget (min 60s so a stage near
            # the deadline isn't killed instantly).  Budget cap firing means
            # the per-branch limit was hit → BranchTimeout.
            timeout_used = max(60, remaining)
            budget_capped = True
    try:
        result = subprocess.run(
            ["/bin/bash", "-c", shell_cmd],
            capture_output=True,
            text=True,
            timeout=timeout_used,
            cwd=str(SKILL_DIR),
            check=False,
        )
    except subprocess.TimeoutExpired:
        if budget_capped:
            raise BranchTimeout(
                f"branch timeout: {script} hit the remaining budget (per-branch limit: {BRANCH_TIMEOUT}s)"
            )
        raise
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()[-5:] if result.stderr else []
        print(
            f"  [{branch}] !!! exit={result.returncode} "
            + ("; ".join(stderr_tail)[:200] if stderr_tail else "(no stderr)")
        )
    return result


def cleanup_processes():
    """Kill leftover torchrun / pretrain_gpt processes between branches."""
    subprocess.run(  # nosec B607 — intentional multi-process kill via shell
        'pkill -9 -f torchrun 2>/dev/null; '
        'pkill -9 -f pretrain_gpt 2>/dev/null; '
        'pkill -9 -f "python.*gpt" 2>/dev/null; '
        'echo done',
        shell=True,  # nosec B602
        timeout=10,
        check=False,
    )
    time.sleep(2)


def read_json(path: str) -> dict:
    """Safely read a JSON artifact. Returns {} on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run_diagnostics(
    branch: str,
    result: dict,
    state_dir: Path,
    conda_env: str,
    env_report: dict,
    adaptor_clone_dir: str,
    check_timeout=None,
    deadline: float | None = None,
):
    """⑤ FETCH_MRS + ⑥ DIAG prompt build for a failed branch.

    Shared by the normal failure path (SIGCHECK/LAUNCH/VERIFY failure) and
    the SETUP-failure path: an import-smoke crash is a code-level breakage
    (e.g. an upstream CUDA-only import path) that the LLM can diagnose and
    patch, so it must not stay silent.  Sets ``needs_llm_diag`` so the cron
    triggers Claude diag + patch.
    """
    result["needs_llm_diag"] = True
    print(f"\n  [{branch}] --- FAIL detected, entering diagnosis ---")

    last_good = env_report.get("last_known_good_commit", "")
    target_commit = env_report.get("megatron", {}).get("commit_sha", "")
    repo_path = str(WORK_DIR / "Megatron-LM")

    print(f"  [{branch}]   last_good: {last_good[:12] if last_good else '?'}")
    print(f"  [{branch}]   target:    {target_commit[:12] if target_commit else '?'}")

    if not last_good or not target_commit:
        result["errors"].append("Cannot run FETCH_MRS: missing last_good_commit or target_commit in env_report.json")
        return
    if check_timeout is not None:
        check_timeout()

    # ⑤ FETCH_MRS
    print(f"\n  [{branch}] (5) FETCH_MRS - scanning git diff ...")
    r = run_stage(
        branch,
        conda_env,
        "fetch_mrs.py",
        [
            "--repo-path",
            repo_path,
            "--last-good",
            last_good,
            "--target",
            target_commit,
            "--adaptor-path",
            adaptor_clone_dir,
            "--output",
            "__STATE_DIR__/diff_report.json",
        ],
        timeout=300,
        deadline=deadline,
    )
    diff_report = read_json(str(state_dir / "diff_report.json"))
    if r.returncode != 0:
        result["stages"]["fetch_mrs"] = "fail"
        result["errors"].append(f"FETCH_MRS failed: exit={r.returncode}")
        print(f"  [{branch}] (5) FETCH_MRS FAIL (exit={r.returncode})")
    else:
        commits = len(diff_report.get("commits", []))
        hits_high = diff_report.get("summary", {}).get("high", 0)
        hits_med = diff_report.get("summary", {}).get("medium", 0)
        result["stages"]["fetch_mrs"] = "ok"
        print(f"  [{branch}] (5) FETCH_MRS OK - {commits} commits, hits: {hits_high} high / {hits_med} medium")

    # ⑥ DIAG (prompt build - LLM call is external)
    print(f"\n  [{branch}] (6) DIAG - building LLM diagnosis prompt ...")
    if check_timeout is not None:
        check_timeout()
    r = run_stage(
        branch,
        conda_env,
        "nightly_diag.py",
        [
            "--env-report",
            "__STATE_DIR__/env_report.json",
            "--setup-report",
            "__STATE_DIR__/setup_report.json",
            "--training-log",
            "__STATE_DIR__/training_log.json",
            "--verify-report",
            "__STATE_DIR__/verify_report.json",
            "--sigcheck-report",
            "__STATE_DIR__/sigcheck_report.json",
            "--diff-report",
            "__STATE_DIR__/diff_report.json",
            "--adaptor-path",
            adaptor_clone_dir,
            "--output",
            "__STATE_DIR__/diag_report.json",
            "--output-prompt",
            "__STATE_DIR__/diag_prompt.txt",
        ],
        timeout=120,
        deadline=deadline,
    )
    if r.returncode != 0:
        result["stages"]["diag"] = "fail"
        result["errors"].append(f"DIAG prompt build failed: exit={r.returncode}")
        print(f"  [{branch}] (6) DIAG prompt build FAIL")
    else:
        result["stages"]["diag"] = "ok"
        diag_report = read_json(str(state_dir / "diag_report.json"))
        print(f"  [{branch}] (6) DIAG prompt ready -> {state_dir}/diag_prompt.txt")
        print(f"  [{branch}]   symptom: {diag_report.get('symptom', '?')[:100]}")
        print(f"  [{branch}]   ** LLM diagnosis + patch generation needed (Claude agent) **")


# =====================================================================
# Branch cycle: ①→⑧ for one branch combo
# =====================================================================


class BranchTimeout(Exception):
    """Raised when a branch exceeds its wall-clock time limit."""

    pass


def run_branch_cycle(combo: dict) -> dict:
    """Execute the full ①→⑧ flywheel for one branch combo.

    ① FETCH + SETUP  →  ② SIGCHECK  →  ③ LAUNCH  →  ④ VERIFY
         (FATAL)         (NON-FATAL)     (NON-FATAL)    (always)

    If any step fails → ⑤ FETCH_MRS → ⑥ DIAG (prompt build)
    Then → ⑧ REPORT (always, if env_report exists).

    Returns a result dict.  The LLM interaction (DIAG analysis, PATCH
    generation, AST fix loop, LAUNCH retry) is handled by the Claude
    agent per SKILL.md — this function builds the prompts and halts.
    """
    branch = combo["megatron_branch"]
    ma_branch = combo["ma_branch"]
    te_branch = combo["te_branch"]
    conda_env = combo["conda_env"]
    state_dir = OUTPUT_DIR / branch / "state"

    result = {
        "branch": branch,
        "ma_branch": ma_branch,
        "te_branch": te_branch,
        "conda_env": conda_env,
        "overall": "pending",
        "stages": {},
        "errors": [],
        "needs_llm_diag": False,
        "artifacts": {
            "env_report": str(state_dir / "env_report.json"),
            "setup_report": str(state_dir / "setup_report.json"),
            "sigcheck_report": str(state_dir / "sigcheck_report.json"),
            "training_log": str(state_dir / "training_log.json"),
            "verify_report": str(state_dir / "verify_report.json"),
            "diff_report": str(state_dir / "diff_report.json"),
            "diag_report": str(state_dir / "diag_report.json"),
            "diag_prompt": str(state_dir / "diag_prompt.txt"),
            "final_report": str(state_dir / "final_report.json"),
        },
    }

    # Set up per-branch wall-clock timeout (monotonic — immune to NTP jumps)
    branch_start = time.monotonic()
    branch_deadline = branch_start + BRANCH_TIMEOUT

    def _check_timeout():
        elapsed = time.monotonic() - branch_start
        if elapsed > BRANCH_TIMEOUT:
            raise BranchTimeout(f"branch timeout after {elapsed:.0f}s (limit: {BRANCH_TIMEOUT}s)")

    try:
        # ---- ① FETCH (FATAL) ----
        print(f"\n  [{branch}] (1) FETCH - cloning / updating repos ...")
        _check_timeout()
        r = run_stage(
            branch,
            conda_env,
            "nightly_fetch.py",
            [
                "--work-dir",
                str(WORK_DIR),
                "--megatron-target",
                f"origin/{branch}",
                "--ma-target",
                f"origin/{ma_branch}",
                "--te-target",
                f"origin/{te_branch}",
                "--output",
                "__STATE_DIR__/env_report.json",
            ],
            timeout=600,
            deadline=branch_deadline,
        )
        env_report = read_json(str(state_dir / "env_report.json"))
        if env_report.get("status") not in ("ok", "partial"):
            result["stages"]["fetch"] = "fail"
            result["errors"].append(f"FETCH failed: {env_report.get('errors', r.stderr[:200])}")
            result["overall"] = "fail"
            return result
        result["stages"]["fetch"] = "ok"
        megatron_sha = env_report.get("megatron", {}).get("commit_sha", "?")[:12]
        ma_sha = env_report.get("adaptor", {}).get("commit_sha", "?")[:12]
        te_sha = env_report.get("transformer_engine_npu", {}).get("commit_sha", "?")[:12]
        print(f"  [{branch}] (1) FETCH OK - MLM:{megatron_sha} MA:{ma_sha} TE:{te_sha}")

        # ── Safety gate: import injection MUST succeed before LAUNCH ──
        # Without the adaptor import in pretrain_gpt.py, the CUDA check
        # patch never fires → all tests crash with "Megatron requires CUDA".
        # This can happen when git clean -fd fails due to root-owned files
        # leaving the clone on a stale branch.
        if not env_report.get("import_patched_in_pretrain"):
            result["stages"]["fetch"] = "fail"
            result["errors"].append(
                "FETCH incomplete: import megatron_adaptor NOT injected "
                "into pretrain_gpt.py. The CUDA assertion in "
                "megatron/training/initialize.py:61 will not be patched. "
                "Check for root-owned files blocking git clean -fd."
            )
            result["overall"] = "fail"
            print(f"  [{branch}] (1) FETCH ABORT: import_patched_in_pretrain=false")
            return result

        adaptor_clone_dir = env_report.get("adaptor", {}).get("clone_dir", "")

        # ---- ② SETUP (FATAL, but import-smoke failure routes to ⑤⑥ DIAG) ----
        print(f"\n  [{branch}] (2) SETUP - installing packages, checking NPU ...")
        _check_timeout()
        r = run_stage(
            branch,
            conda_env,
            "nightly_setup.py",
            [
                "--env-report",
                "__STATE_DIR__/env_report.json",
                "--output",
                "__STATE_DIR__/setup_report.json",
            ],
            timeout=600,
            deadline=branch_deadline,
        )
        setup_report = read_json(str(state_dir / "setup_report.json"))
        if setup_report.get("status") != "ok":
            result["stages"]["setup"] = "fail"
            result["errors"].append(f"SETUP failed: {setup_report.get('errors', r.stderr[:200])}")
            result["overall"] = "fail"
            # Import-smoke failure is a code-level breakage (e.g. an upstream
            # CUDA-only import path) that the LLM can diagnose + patch → build
            # the diag prompt so the cron's LLM step fires.  Env-class setup
            # failures (pip install / NPU / dataset / TE missing) stay FATAL —
            # no patch can fix them, and routing them to diag would waste LLM
            # tokens on every infra hiccup.
            smoke_ok = setup_report.get("packages", {}).get("adaptor_import_smoke", True)
            if smoke_ok is False:
                result["stages"]["launch"] = "skipped"
                result["stages"]["verify"] = "skipped"
                _run_diagnostics(
                    branch,
                    result,
                    state_dir,
                    conda_env,
                    env_report,
                    adaptor_clone_dir,
                    _check_timeout,
                    deadline=branch_deadline,
                )
            return result
        result["stages"]["setup"] = "ok"
        te_avail = setup_report.get("packages", {}).get("te_available")
        npu_count = setup_report.get("npu", {}).get("device_count", 0)
        print(f"  [{branch}] (2) SETUP OK - TE: {te_avail}, NPU: {npu_count} cards")

        # ---- ②.5 SIGCHECK (NON-FATAL) ----
        print(f"\n  [{branch}] (2.5) SIGCHECK - cross-checking patch signatures ...")
        _check_timeout()
        r = run_stage(
            branch,
            conda_env,
            "nightly_sigcheck.py",
            [
                "--adaptor-path",
                adaptor_clone_dir,
                "--env-report",
                "__STATE_DIR__/env_report.json",
                "--output",
                "__STATE_DIR__/sigcheck_report.json",
            ],
            timeout=120,
            deadline=branch_deadline,
        )
        sigcheck_report = read_json(str(state_dir / "sigcheck_report.json"))
        sigcheck_ok = sigcheck_report.get("overall") == "pass"
        if sigcheck_ok:
            result["stages"]["sigcheck"] = "pass"
            print(
                f"  [{branch}] (2.5) SIGCHECK PASS - "
                f"{sigcheck_report.get('targets_ok', 0)}/"
                f"{sigcheck_report.get('targets_total', 0)} ok"
            )
        else:
            result["stages"]["sigcheck"] = "fail"
            result["overall"] = "fail"
            result["errors"].append(f"SIGCHECK: {sigcheck_report.get('changed_targets', [])}")
            print(
                f"  [{branch}] (2.5) SIGCHECK FAIL - "
                f"{sigcheck_report.get('targets_changed', 0)} signatures "
                f"changed (continuing for richer diagnostics)"
            )

        # ---- ③ LAUNCH (NON-FATAL) ----
        print(f"\n  [{branch}] (3) LAUNCH - running training tests ...")
        _check_timeout()
        r = run_stage(
            branch,
            conda_env,
            "nightly_launch.py",
            [
                "--adaptor-path",
                adaptor_clone_dir,
                "--branch",
                branch,
                "--state-dir",
                "__STATE_DIR__",
                "--log-dir",
                "__LOG_DIR__",
                "--output",
                "__STATE_DIR__/training_log.json",
            ],
            timeout=7200,
            deadline=branch_deadline,
        )

        training_log_path = state_dir / "training_log.json"
        if r.returncode != 0 or not training_log_path.exists():
            result["stages"]["launch"] = "fail"
            result["overall"] = "fail"
            stderr_tail = r.stderr.strip().splitlines()[-3:] if r.stderr else ["(no output)"]
            result["errors"].append(f"LAUNCH crashed (exit={r.returncode}): {'; '.join(stderr_tail)[:300]}")
            print(f"  [{branch}] (3) LAUNCH FAIL (exit={r.returncode}, continuing to VERIFY)")
        else:
            training_log = read_json(str(training_log_path))
            if training_log.get("env_busy"):
                # ①.5 ENVCHECK 在 nightly_launch.py 第一步执行：NPU 被其他训练任务
                # 占用（如 admin123 mindspeed_mm 占 HCCL 端口 16666）→ 分支 skip。
                # 环境问题非代码回归 → 不跑 VERIFY、不触发 ⑤⑥⑦⑧、不建 Issue/PR。
                result["overall"] = "skip"
                result["stages"]["launch"] = "env_busy"
                result["stages"]["verify"] = "skipped"
                result["errors"].append(
                    f"ENV_BUSY: {training_log.get('env_busy_reason', 'NPU occupied by another training job')} "
                    f"— branch skipped, no diagnosis"
                )
                print(f"\n  [{branch}] (3) LAUNCH SKIP - {training_log.get('env_busy_reason', 'NPU occupied')}")
                print(f"  [{branch}]     branch skipped (environment issue, not a code regression)")
                print(f"\n  [{branch}] branch cycle complete (env-busy skip) -> overall: skip")
                return result
            summary = training_log.get("summary", {})
            launch_failed = (
                summary.get("total", 0) == 0
                or summary.get("failed", 0) > 0
                or summary.get("timeout", 0) > 0
                or summary.get("error", 0) > 0
            )
            if launch_failed:
                result["stages"]["launch"] = "fail"
                result["overall"] = "fail"
                result["errors"].append(
                    f"LAUNCH: {summary.get('failed', 0)} failed, "
                    f"{summary.get('timeout', 0)} timeout, "
                    f"{summary.get('error', 0)} error"
                )
                print(f"  [{branch}] (3) LAUNCH FAIL - {summary}")
            else:
                result["stages"]["launch"] = "pass"
                retry_note = ""
                if summary.get('retry_passed'):
                    retry_note = f", retry: {summary.get('retry_passed')} saved"
                print(
                    f"  [{branch}] (3) LAUNCH OK - "
                    f"{summary.get('passed', 0)}/{summary.get('total', 0)} "
                    f"passed ({summary.get('total_duration_seconds', 0):.0f}s)"
                    f"{retry_note}"
                )

        # ---- ④ VERIFY (always runs) ----
        print(f"\n  [{branch}] (4) VERIFY - checking training results ...")
        _check_timeout()
        r = run_stage(
            branch,
            conda_env,
            "nightly_verify.py",
            [
                "--training-log",
                "__STATE_DIR__/training_log.json",
                "--output",
                "__STATE_DIR__/verify_report.json",
            ],
            timeout=120,
            deadline=branch_deadline,
        )
        verify_report = read_json(str(state_dir / "verify_report.json"))
        verify_ok = verify_report.get("overall") == "pass"
        if verify_ok:
            result["stages"]["verify"] = "pass"
            print(f"  [{branch}] (4) VERIFY PASS")
            if result["overall"] != "fail":
                result["overall"] = "pass"
        else:
            result["stages"]["verify"] = "fail"
            result["overall"] = "fail"
            failed_checks = [k for k, v in verify_report.get("checks", {}).items() if not v.get("passed")]
            result["errors"].append(f"VERIFY: {verify_report.get('failure_details', [])}")
            print(f"  [{branch}] (4) VERIFY FAIL - {failed_checks}")

        # ---- ⑤⑥ DIAGNOSE (only if failed) ----
        if result["overall"] in ("fail",):
            _run_diagnostics(
                branch,
                result,
                state_dir,
                conda_env,
                env_report,
                adaptor_clone_dir,
                _check_timeout,
                deadline=branch_deadline,
            )

        # ---- ⑧ REPORT (always, if env_report exists) ----
        print(f"\n  [{branch}] (8) REPORT - generating final report ...")
        _check_timeout()
        if (state_dir / "env_report.json").exists():
            run_stage(
                branch,
                conda_env,
                "nightly_report.py",
                [
                    "--env-report",
                    "__STATE_DIR__/env_report.json",
                    "--setup-report",
                    "__STATE_DIR__/setup_report.json",
                    "--training-log",
                    "__STATE_DIR__/training_log.json",
                    "--verify-report",
                    "__STATE_DIR__/verify_report.json",
                    "--sigcheck-report",
                    "__STATE_DIR__/sigcheck_report.json",
                    "--eval-set",
                    str(SCRIPTS_DIR / "nightly_eval_set.json"),
                    "--output",
                    "__STATE_DIR__/final_report.json",
                ],
                timeout=60,
                deadline=branch_deadline,
            )
            final_report = read_json(str(state_dir / "final_report.json"))
            evals = final_report.get("eval_results", {})
            passed_evals = sum(1 for v in evals.values() if v.get("passed"))
            result["stages"]["report"] = "ok"
            print(
                f"  [{branch}] (8) REPORT - "
                f"overall: {final_report.get('overall_status', '?')}, "
                f"evals: {passed_evals}/{len(evals)} passed"
            )
        else:
            result["stages"]["report"] = "skipped"
            print(f"  [{branch}] (8) REPORT skipped - env_report.json missing")

    except BranchTimeout as e:
        result["overall"] = "fail"
        result["stages"]["timeout"] = "fail"
        result["errors"].append(str(e))
        print(f"\n  [{branch}] !!! TIMEOUT - {e}")

    except Exception as e:
        result["overall"] = "error"
        result["errors"].append(f"Orchestrator exception: {e}")
        import traceback

        traceback.print_exc()

    elapsed = time.monotonic() - branch_start
    print(
        f"\n  [{branch}] branch cycle complete ({elapsed:.0f}s / "
        f"{elapsed / 60:.1f} min) -> overall: {result['overall']}"
    )

    return result


# =====================================================================
# Main
# =====================================================================


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_summary(results: list):
    """Print a formatted summary of all branch results."""
    total = len(results)
    passed = sum(1 for r in results if r["overall"] == "pass")
    failed = sum(1 for r in results if r["overall"] == "fail")
    errored = sum(1 for r in results if r["overall"] == "error")
    skipped = sum(1 for r in results if r["overall"] == "skip")
    needs_diag = sum(1 for r in results if r.get("needs_llm_diag"))

    emoji = "PASS" if failed == 0 and errored == 0 else "FAIL"
    print(f"\n  {emoji}: {passed}/{total} passed | {failed} failed | {errored} error | {skipped} skipped")
    if needs_diag:
        print(f"  {needs_diag} branch(es) need LLM diagnosis + patch")
    if skipped:
        print(f"  {skipped} branch(es) skipped (NPU occupied by another training job)")
    print()

    for r in results:
        branch = r["branch"]
        overall = r["overall"]
        icon = {"pass": "OK", "fail": "XX", "error": "!!", "pending": "??", "skip": "--"}.get(overall, "??")  # nosec B105
        stages_str = " -> ".join(f"{k}:{v}" for k, v in r.get("stages", {}).items())
        print(f"  [{icon}] {branch:<20} {overall:<6} [{stages_str}]")
        if r.get("errors"):
            for e in r["errors"][:2]:
                print(f"       L- {str(e)[:120]}")
        artifacts = r.get("artifacts", {})
        diag_prompt = artifacts.get("diag_prompt", "")
        if diag_prompt and Path(diag_prompt).exists():
            print(f"       L- ** LLM diagnosis needed -> {diag_prompt}")


def write_combined_summary(results: list, elapsed: float = 0.0):
    """Write the combined_summary.json artifact."""
    total = len(results)
    passed = sum(1 for r in results if r["overall"] == "pass")
    failed = sum(1 for r in results if r["overall"] in ("fail", "error"))
    errored = sum(1 for r in results if r["overall"] == "error")
    skipped = sum(1 for r in results if r["overall"] == "skip")
    needs_diag = sum(1 for r in results if r.get("needs_llm_diag"))

    # Capture shared clone state for PATCH verification
    clone_state = {}
    for repo_name in ["Megatron-LM", "MegatronAdaptor", "TransformerEngineNPU"]:
        repo_path = WORK_DIR / repo_name
        if repo_path.exists():
            try:
                r = subprocess.run(
                    [shutil.which("git") or "git", "branch", "--show-current"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(repo_path),
                    check=False,
                )
                branch = r.stdout.strip()
                r2 = subprocess.run(
                    [shutil.which("git") or "git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(repo_path),
                    check=False,
                )
                sha = r2.stdout.strip()
                clone_state[repo_name] = {"branch": branch, "commit": sha}
            except Exception:
                clone_state[repo_name] = {"branch": "?", "commit": "?"}

    combined = {
        # output/YYYY-MM/MM-DD → 完整日期 YYYY-MM-DD
        "run_date": f"{OUTPUT_DIR.parent.name}-{OUTPUT_DIR.name.split('-')[-1]}",
        "total_branches": total,
        "passed": passed,
        "failed": failed,
        "error": errored,
        "skipped": skipped,
        "needs_llm_diagnosis": needs_diag,
        "total_duration_seconds": elapsed,
        "shared_clone_state": clone_state,
        "results": results,
    }
    combined_path = OUTPUT_DIR / "combined_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined summary: {combined_path}")
    return combined


def main():
    global OUTPUT_DIR  # set after parsing --date

    parser = argparse.ArgumentParser(description="MegatronAdaptor Nightly Pipeline Orchestrator")
    parser.add_argument("--date", default=None, help="Run date in YYYY-MM-DD format (default: UTC today)")
    args = parser.parse_args()

    RUN_DATE = _resolve_run_date(args.date)
    # 按月分层：output/YYYY-MM/MM-DD（与 CI 仓 sync_build_results 一致），
    # 避免 output/ 下日期目录无限增长；同一天重跑由 sync 脚本归档 _rN。
    OUTPUT_DIR = WORK_DIR / "output" / RUN_DATE[:7] / RUN_DATE[5:]

    print("=" * 60)
    print("MegatronAdaptor Nightly Pipeline (per-branch full cycle)")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Run date: {RUN_DATE}")
    print("=" * 60)
    print(f"Branches: {len(BRANCH_COMBOS)}")
    for c in BRANCH_COMBOS:
        print(
            f"  * MLM:{c['megatron_branch']:<18} "
            f"MA:{c['ma_branch']:<18} "
            f"TE:{c['te_branch']:<10} -> conda {c['conda_env']}"
        )
    print(f"Per-branch timeout: {BRANCH_TIMEOUT}s ({BRANCH_TIMEOUT / 60:.0f} min)")

    start_time = time.time()
    results = []

    for idx, combo in enumerate(BRANCH_COMBOS, 1):
        branch = combo["megatron_branch"]
        print_header(f"[{idx}/{len(BRANCH_COMBOS)}] Megatron {branch} (conda: {combo['conda_env']})")

        if idx > 1:
            print(f"\n  [{branch}] Cleaning up leftover processes ...")
            cleanup_processes()

        result = run_branch_cycle(combo)
        results.append(result)

    # ---- Restore shared clone for PATCH verification ----
    # After processing all branches, the shared clones may be on any branch.
    # Restore Megatron-LM + MegatronAdaptor to the FIRST failed branch so
    # the Claude agent can immediately verify patches on correct code.
    failed_branches = [r for r in results if r.get("needs_llm_diag")]
    if failed_branches:
        first_failed = failed_branches[0]
        fb_branch = first_failed["branch"]
        fb_ma_branch = first_failed.get("ma_branch", fb_branch)
        print_header("PATCH PREP: Restoring shared clones for verification")
        print(f"\n  Restoring to failed branch: MLM={fb_branch}, MA={fb_ma_branch}")
        _restore_clone("Megatron-LM", fb_branch)
        _restore_clone("MegatronAdaptor", fb_ma_branch)
        print(f"\n  Shared clones ready for PATCH verification on [{fb_branch}].")
        if len(failed_branches) > 1:
            others = [r["branch"] for r in failed_branches[1:]]
            print(f"  ⚠️  Other failed branches ({', '.join(others)}) — restore manually before verifying those.")

    # ---- Final combined summary ----
    total_elapsed = time.time() - start_time

    print_header("COMBINED SUMMARY")
    total = len(results)
    passed = sum(1 for r in results if r["overall"] == "pass")
    failed = sum(1 for r in results if r["overall"] in ("fail", "error"))
    skipped = sum(1 for r in results if r["overall"] == "skip")
    needs_diag = sum(1 for r in results if r.get("needs_llm_diag"))

    print(f"\n  Total:     {total} branches")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {failed}")
    if skipped:
        print(f"  Skipped:   {skipped} (NPU occupied by another training job)")
    if needs_diag:
        print(f"  Need LLM:  {needs_diag} (Claude agent to diagnose + patch)")
    print(f"  Duration:  {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print()

    print_summary(results)

    print()
    print("=" * 60)
    print(f"Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    write_combined_summary(results, total_elapsed)

    # Exit codes: 0 = at least one branch passed (or mixed pass/skip),
    # 1 = code-level failure (failed already includes overall=="error"),
    # 2 = NPU occupied → all branches skipped this round.
    if failed > 0:
        return 1
    if total > 0 and passed == 0 and skipped == total:
        return 2
    return 0


def _restore_clone(repo_name: str, target_branch: str):
    """Restore a shared clone to origin/<target_branch>."""
    repo_path = WORK_DIR / repo_name
    if not repo_path.exists():
        print(f"  WARNING: {repo_name} clone not found at {repo_path}")
        return
    try:
        subprocess.run(
            [shutil.which("git") or "git", "fetch", "origin"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_path),
            check=False,
        )
        subprocess.run(
            [shutil.which("git") or "git", "checkout", "--force", f"origin/{target_branch}"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_path),
            check=False,
        )
        subprocess.run(
            [shutil.which("git") or "git", "reset", "--hard", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_path),
            check=False,
        )
        subprocess.run(
            [shutil.which("git") or "git", "clean", "-fd"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_path),
            check=False,
        )
        r = subprocess.run(
            [shutil.which("git") or "git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_path),
            check=False,
        )
        sha = r.stdout.strip()
        print(f"  {repo_name}: checkout origin/{target_branch} -> {sha}")
    except Exception as e:
        print(f"  ERROR restoring {repo_name}: {e}")


if __name__ == "__main__":
    sys.exit(main())
