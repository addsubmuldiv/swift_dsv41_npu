#!/usr/bin/env python3
"""Stage ⑥ DIAG — Structured LLM prompt preparer. Triggered only on failure.

Collects all pipeline artifacts and failure context, builds a constrained
prompt for LLM diagnosis (with adapter domain knowledge injected as
deterministic context), and validates LLM output against schema.

This script does NOT invoke the LLM itself — it prepares the prompt and
validates the response. The actual LLM call is done by the orchestrator.
"""

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
# ── Error extraction (3-tier algorithm, shared with nightly_verify.py) ──
# Curated list of Python exception types.  We grep for `<ExceptionName>:` so
# we catch the actual raise, not parameter names or comments.

_EXCEPTION_PATTERN = re.compile(
    r"(FileNotFoundError|RuntimeError|ValueError|TypeError|AssertionError"
    r"|IndexError|KeyError|ConnectionError|TimeoutError|OSError"
    r"|AttributeError|ModuleNotFoundError|ImportError"
    r"|ChildFailedError|DistBackendError):",
)

# Lines containing these substrings are never treated as errors even when
# they happen to contain an exception class name.
_EXCLUDE_SUBSTRINGS = ["error_injection", "error_file"]


def extract_error_from_log(log_paths: dict) -> dict:
    """Extract structured error info from test stdout / stderr logs.

    Implements the 3-tier algorithm:
      1. Traceback-first — locate the FIRST traceback (root cause, not cascade).
      2. Exception-first — locate the FIRST concrete Python exception.
      3. Fallback — grep for ERROR / FATAL keywords as last resort.

    Returns ``{reason, detail_block}`` — both human-readable strings suitable
    for LLM diagnosis prompts.
    """
    combined = ""
    for key in ("stderr_path", "stdout_path"):
        p = log_paths.get(key, "")
        if p:
            try:
                combined += Path(p).read_text(encoding="utf-8", errors="replace") + "\n"
            except Exception:
                pass  # nosec B110

    if not combined:
        return {"reason": "", "detail_block": ""}

    lines = combined.splitlines()

    # Tier 1: locate first Traceback line
    first_tb = None
    for i, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            first_tb = i
            break

    # Tier 2: locate first concrete exception (exclude known non-errors)
    first_err = None
    for i, line in enumerate(lines):
        if any(x in line for x in _EXCLUDE_SUBSTRINGS):
            continue
        if _EXCEPTION_PATTERN.search(line):
            first_err = i
            break

    reason = ""
    detail_block = ""

    if first_tb is not None and first_err is not None:
        # Perfect case: extract from Traceback → exception + 10 lines.
        end = min(first_err + 10, len(lines))
        block_lines = lines[first_tb:end]
        block_lines = [re.sub(r"^\[rank\d*\]:?", "", line) for line in block_lines if "frame #" not in line]
        detail_block = "\n".join(block_lines)
        reason = lines[first_err].strip()
        reason = re.sub(r"^\[rank\d*\]:?\s*", "", reason)
    elif first_err is not None:
        # Exception found but no traceback — grab surrounding context.
        start = max(first_err - 5, 0)
        end = min(first_err + 25, len(lines))
        block_lines = lines[start:end]
        block_lines = [re.sub(r"^\[rank\d*\]:?", "", line) for line in block_lines if "frame #" not in line]
        detail_block = "\n".join(block_lines)
        reason = lines[first_err].strip()
        reason = re.sub(r"^\[rank\d*\]:?\s*", "", reason)
    else:
        # Fallback: grep for ERROR / FATAL / exitcode indicators.
        error_lines = [
            line for line in lines if re.search(r"(ERROR|FATAL|child.*failed|exitcode|ERR\d)", line, re.IGNORECASE)
        ]
        if error_lines:
            reason = error_lines[-1].strip()
            reason = re.sub(r"^\[rank\d*\]:?\s*", "", reason)
            tail_start = max(0, len(lines) - 40)
            detail_block = "\n".join(line for line in lines[tail_start:] if line.strip())

    # Truncate for embedding size limits.
    reason = " ".join(reason.split())[:500]
    if len(detail_block) > 6000:
        detail_block = "\n".join(detail_block.splitlines()[:45])

    return {"reason": reason, "detail_block": detail_block}


DIAG_OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "diagnosis_time",
        "failed_stage",
        "symptom",
        "root_cause",
        "confidence",
        "needs_human_action",
    ],
    "properties": {
        "diagnosis_time": {"type": "string"},
        "failed_stage": {
            "type": "string",
            "enum": ["fetch", "setup", "sigcheck", "launch", "verify"],
        },
        "failed_test_id": {"type": "string"},
        "failed_checks": {"type": "array", "items": {"type": "string"}},
        "symptom": {"type": "string"},
        "root_cause": {"type": "string"},
        "affected_patch_file": {"type": "string"},
        "affected_megatron_file": {"type": "string"},
        "fix_summary": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "needs_human_action": {"type": "boolean"},
        "rollback_commit": {"type": "string"},
        "breaking_mrs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "mr_number": {"type": "integer"},
                    "title": {"type": "string"},
                    "breakage_reason": {"type": "string"},
                    "patch_strategy": {"type": "string"},
                },
            },
        },
        "similar_past_cases": {"type": "array", "items": {"type": "object"}},
    },
}


def collect_failure_context(artifacts: dict) -> dict:
    """Collect structured failure context from all artifacts."""
    context = {
        "diagnosis_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "failed_stage": "unknown",
        "failed_test_id": None,
        "failed_checks": [],
        "symptom": "",
    }

    env_data = artifacts.get("env_report", {})
    setup_data = artifacts.get("setup_report", {})
    training_data = artifacts.get("training_log", {})
    verify_data = artifacts.get("verify_report", {})
    sigcheck_data = artifacts.get("sigcheck_report", {})

    if env_data.get("status") not in ("ok", "partial"):
        context["failed_stage"] = "fetch"
        context["symptom"] = f"FETCH errors: {env_data.get('errors', [])}"
    elif setup_data.get("status") != "ok":
        context["failed_stage"] = "setup"
        context["symptom"] = f"SETUP errors: {setup_data.get('errors', [])}"
    elif sigcheck_data.get("overall") == "fail":
        context["failed_stage"] = "sigcheck"
        changed = sigcheck_data.get("changed_targets", [])
        context["symptom"] = f"SIGCHECK: {len(changed)} targets have signature changes: {changed}"
    elif verify_data.get("overall") == "fail":
        context["failed_stage"] = "verify"
        for name, check in verify_data.get("checks", {}).items():
            if not check.get("passed", True):
                context["failed_checks"].append(name)
        for tc in training_data.get("test_cases", []):
            if tc.get("status") != "pass":
                context["failed_test_id"] = tc.get("test_id")
                break
        context["symptom"] = json.dumps(verify_data.get("failure_details", []), indent=2)
    elif any(training_data.get("summary", {}).get(k, 0) > 0 for k in ("failed", "timeout", "error")):
        context["failed_stage"] = "launch"
        for tc in training_data.get("test_cases", []):
            if tc.get("status") in ("fail", "timeout", "error"):
                context["failed_test_id"] = tc.get("test_id")
                error_text = " ".join(tc.get("errors", [])[:3])
                context["symptom"] = error_text[:500] if error_text else f"status={tc.get('status')}"
                break

    return context


def extract_errors_from_training_log(training_log: dict) -> list[dict]:
    """Run error extraction on every failed test case.

    Returns a list of ``{test_id, reason, detail_block}`` for downstream
    prompt building.
    """
    errors = []
    for tc in training_log.get("test_cases", []):
        if tc.get("status") == "pass":
            continue
        err_info = extract_error_from_log(
            {
                "stdout_path": tc.get("stdout_path", ""),
                "stderr_path": tc.get("stderr_path", ""),
            }
        )
        errors.append(
            {
                "test_id": tc["test_id"],
                "reason": err_info.get("reason", ""),
                "detail_block": err_info.get("detail_block", ""),
            }
        )
    return errors


def collect_failed_test_scripts(training_log: dict, adaptor_path: str) -> list[dict]:
    """Read the content of failed test scripts for prompt context.

    Returns ``[{test_id, script, content}]``.
    """
    scripts = []
    adaptor_root = Path(adaptor_path)
    for tc in training_log.get("test_cases", []):
        if tc.get("status") == "pass":
            continue
        script_rel = tc.get("script", "")
        if not script_rel:
            continue
        script_path = adaptor_root / script_rel
        if not script_path.exists():
            continue
        try:
            content = script_path.read_text()
        except Exception:
            continue
        scripts.append(
            {
                "test_id": tc["test_id"],
                "script": script_rel,
                "content": content,
            }
        )
    return scripts


def get_adaptor_context(adaptor_path: str) -> dict:
    """Extract deterministic context about the adaptor: registered patch targets."""
    if not adaptor_path:
        return {"targets": [], "target_count": 0}

    targets = []
    patches_dir = Path(adaptor_path) / "megatron_adaptor"
    if not patches_dir.exists():
        return {"targets": [], "target_count": 0}

    for py_file in patches_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_register = False
            if isinstance(func, ast.Attribute) and func.attr == "register_patch":
                is_register = True
            elif isinstance(func, ast.Name) and func.id == "register_patch":
                is_register = True
            if not is_register or not node.args:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                targets.append(first_arg.value)

    targets = sorted(set(targets))
    return {"targets": targets, "target_count": len(targets)}


def search_last_success() -> dict:
    """Find the most recent successful run in the case library for rollback hint."""
    case_library = SCRIPTS_DIR / "nightly_case_library.jsonl"
    if not case_library.exists():
        return {}

    try:
        # Parse all lines, find last success (reverse order)
        lines = case_library.read_text().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            if case.get("overall_status") == "pass":
                return {
                    "run_date": case.get("run_date", ""),
                    "megatron_commit": case.get("megatron_commit", ""),
                }
    except Exception:
        pass  # nosec B110
    return {}


def search_similar_cases(failure_context: dict) -> list:
    """Search the case library for similar past failures."""
    case_library = SCRIPTS_DIR / "nightly_case_library.jsonl"
    if not case_library.exists():
        return []

    similar = []
    failed_stage = failure_context.get("failed_stage", "")

    try:
        for line in case_library.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue

            if case.get("overall_status") == "pass":
                continue

            case_stages = case.get("stages", {})
            if case_stages.get(failed_stage) in ("fail", "error"):
                similar.append(
                    {
                        "run_date": case.get("run_date", ""),
                        "megatron_commit": case.get("megatron_commit", "")[:12],
                        "failures": case.get("failures", []),
                    }
                )

            if len(similar) >= 3:
                break
    except Exception:
        pass  # nosec B110

    return similar


def build_diag_prompt(
    failure_context: dict,
    extracted_errors: list,
    artifacts: dict,
    similar_cases: list,
    adaptor_ctx: dict = None,
    last_success: dict = None,
    mr_list: list = None,
    diff_report: dict = None,
    failed_scripts: list = None,
) -> str:
    """Build a structured LLM prompt for MR-diff-driven failure diagnosis.

    The core logic: yesterday's commit was verified working. Today's commit
    fails. The delta = MRs merged in the last 24 hours. The LLM's job is to
    identify which MRs introduced TE-related breaking changes.
    """
    parts = []

    parts.append(
        "You are diagnosing a MegatronAdaptor nightly validation failure on Ascend NPU with TransformerEngine."
    )
    parts.append("")

    # ── Delta logic: yesterday OK, today broken, MRs are the delta ──
    env_data = artifacts.get("env_report", {})
    setup_data = artifacts.get("setup_report", {})
    today_commit = env_data.get("megatron", {}).get("commit_sha", "N/A")[:12]
    yesterday_commit = env_data.get("last_known_good_commit", "N/A")[:12]

    parts.append("## Core Logic: Delta Diagnosis")
    parts.append("")
    parts.append(f"- **Yesterday's verified-working commit**: {yesterday_commit} (ran successfully on NPU with TE)")
    parts.append(f"- **Today's failing commit**: {today_commit}")
    parts.append(
        "- **The delta**: Commits between the last known-good commit and "
        "today's failing commit. These are the ONLY changes that could have "
        "caused today's failure."
    )
    parts.append("")

    # ── MR/commit list (the delta — deterministic, from fetch_mrs.py) ──
    if mr_list:
        parts.append("## Megatron MRs (last 24 hours — the delta)")
        parts.append("")
        for mr in mr_list:
            parts.append(f"### MR #{mr.get('mr_number', '?')}: {mr.get('title', 'N/A')}")
            parts.append(f"- Merged at: {mr.get('merged_at', 'N/A')}")
            parts.append(f"- Changed files: {', '.join(mr.get('changed_files', [])[:10])}")
            parts.append("")
    elif diff_report and diff_report.get("commits"):
        parts.append("## Megatron Commits (delta: last_good → target)")
        parts.append("")
        for c in diff_report["commits"]:
            parts.append(f"- `{c['sha'][:12]}` {c['subject']}")
        parts.append("")
    else:
        parts.append("## Megatron Commits (delta)")
        parts.append("")
        parts.append("(No commit delta data provided — cannot perform delta diagnosis)")
        parts.append("")

    # ── Diff report (commit-range scan — adaptor-relevant hits) ──
    if diff_report:
        hits = diff_report.get("hits", [])
        high_hits = [h for h in hits if h.get("severity") == "high"]
        med_hits = [h for h in hits if h.get("severity") == "medium"]
        if high_hits or med_hits:
            commit_list = diff_report.get("commits", [])
            commit_summary = (
                f"{commit_list[0]['sha'][:12]}..{commit_list[-1]['sha'][:12]}"
                if len(commit_list) >= 2
                else (commit_list[0]['sha'][:12] if commit_list else "?")
            )
            parts.append("## Diff Scanner Results (commits hitting adaptor patch targets)")
            parts.append("")
            parts.append(f"Commit range: {commit_summary} ({len(commit_list)} commits)")
            parts.append(f"High severity: {len(high_hits)} | Medium: {len(med_hits)}")
            parts.append("")
            for h in high_hits + med_hits:
                parts.append(f"  Hit targets: {h.get('hit_targets', [])}")
                parts.append(f"  Severity: {h.get('severity', 'N/A')}")
                parts.append("")

    parts.append("## Failure Context")
    parts.append("")
    parts.append(f"- Failed stage: {failure_context.get('failed_stage', 'unknown')}")
    parts.append(f"- Failed test: {failure_context.get('failed_test_id', 'N/A')}")
    parts.append(f"- Failed checks: {failure_context.get('failed_checks', [])}")
    parts.append(f"- Symptom: {failure_context.get('symptom', 'N/A')}")
    parts.append("")

    parts.append("## Environment Info")
    parts.append("")
    parts.append(f"- Megatron commit: {today_commit}")
    parts.append(f"- Megatron version: {env_data.get('megatron', {}).get('version', 'N/A')}")
    parts.append(f"- NPU available: {setup_data.get('npu', {}).get('npu_available', 'N/A')}")
    parts.append(
        f"- TE available: "
        f"{setup_data.get('packages', {}).get('te_available', 'N/A')} "
        f"(REQUIRED — TransformerEngine path)"
    )
    parts.append(f"- Adaptor import smoke: {setup_data.get('packages', {}).get('adaptor_import_smoke', 'N/A')}")
    parts.append("")

    # ── Failed test scripts (config context — what triggered the failure) ──
    if failed_scripts:
        parts.append("## Failed Test Scripts")
        parts.append("")
        for fs in failed_scripts:
            parts.append(f"### {fs['test_id']} ({fs['script']})")
            parts.append("```bash")
            parts.append(fs["content"])
            parts.append("```")
            parts.append("")
    elif "training_log" in artifacts:
        # Fallback: list script paths without content
        training_data = artifacts["training_log"]
        failed = [tc for tc in training_data.get("test_cases", []) if tc.get("status") != "pass"]
        if failed:
            parts.append("## Failed Test Scripts")
            parts.append("")
            for tc in failed:
                parts.append(f"- `{tc['test_id']}`: {tc.get('script', 'N/A')}")
            parts.append("")

    # ── Adapter domain knowledge (deterministic injection, not from LLM memory) ──
    if adaptor_ctx and adaptor_ctx.get("targets"):
        parts.append("## MegatronAdaptor Context (deterministic)")
        parts.append("")
        parts.append(
            f"This adaptor monkey-patches {adaptor_ctx['target_count']} "
            f"Megatron/PyTorch functions. If the failure involves any of "
            f"these targets, the patch likely needs updating:"
        )
        parts.append("")
        for t in adaptor_ctx["targets"]:
            parts.append(f"  - {t}")
        parts.append("")
    else:
        parts.append("## MegatronAdaptor Context")
        parts.append("")
        parts.append("(No adaptor path provided — cannot extract patch target list)")
        parts.append("")

    # ── Rollback hint (飞轮: last known-good commit) ──
    if last_success and last_success.get("megatron_commit"):
        parts.append("## Rollback Reference (from flywheel)")
        parts.append("")
        parts.append(
            f"Last known-good Megatron commit: "
            f"{last_success['megatron_commit'][:12]} "
            f"(date: {last_success.get('run_date', 'N/A')})"
        )
        parts.append(
            "If the failure is caused by an upstream Megatron change, "
            "consider rolling back to this commit as a temporary workaround."
        )
        parts.append("")
    else:
        parts.append("## Rollback Reference")
        parts.append("")
        parts.append("(No previous successful run found in flywheel)")
        parts.append("")

    if extracted_errors:
        parts.append("## Error Details (from failed tests)")
        parts.append("")
        for err in extracted_errors:
            parts.append(f"### {err['test_id']}")
            if err.get("reason"):
                parts.append(f"- Root cause: {err['reason']}")
            if err.get("detail_block"):
                parts.append("```")
                parts.append(err["detail_block"])
                parts.append("```")
            parts.append("")

    if similar_cases:
        parts.append("## Similar Past Failures (from case library)")
        parts.append("")
        for i, case in enumerate(similar_cases):
            parts.append(f"### Case {i + 1}: {case.get('run_date', 'N/A')}")
            parts.append(f"- Commit: {case.get('megatron_commit', 'N/A')}")
            parts.append(f"- Failures: {json.dumps(case.get('failures', []))}")
            parts.append("")

    parts.append("## Task")
    parts.append("")
    parts.append("Diagnose why the nightly validation failed. Follow this process:")
    parts.append("")
    parts.append("1. **Read the error details** — identify the concrete exception type,")
    parts.append("   the exact file:line where it was raised, and the call chain leading to it.")
    parts.append("")
    parts.append("2. **Check the delta commits** — if any of them touch the files or")
    parts.append("   functions in the traceback, that commit introduced the breakage.")
    parts.append("   If the delta commits are unrelated, the failure is a latent bug")
    parts.append("   in MegatronAdaptor that was exposed by a specific test configuration")
    parts.append("   (look at the test script for unusual args like --use-cpu-initialization,")
    parts.append("   --no-pin-cpu-params, new fp8 formats, etc.).")
    parts.append("")
    parts.append("3. **Cross-reference with MA patch targets** — if the traceback passes")
    parts.append("   through a function that MA monkey-patches (listed above), the adaptor")
    parts.append("   patch is the likely root cause. Check whether the patch handles all")
    parts.append("   edge cases (device types, dtype combinations, new code paths).")
    parts.append("")
    parts.append("4. **Check similar past failures** for recurring patterns.")
    parts.append("")
    parts.append("5. **Produce a root_cause** explaining what went wrong and why.")
    parts.append("   Include the affected file (MA patch or Megatron source), the exact")
    parts.append("   mechanism, and a concrete fix strategy.")
    parts.append("")
    parts.append("## Output Format (strict JSON schema)")
    parts.append("")
    parts.append("Output ONLY valid JSON matching this schema:")
    parts.append("```json")
    parts.append(json.dumps(DIAG_OUTPUT_SCHEMA, indent=2))
    parts.append("```")
    parts.append("")
    parts.append("- confidence=low: uncertain → needs_human_action MUST be true")

    return "\n".join(parts)


def validate_diag_output(diag_output: dict) -> dict:
    """Validate LLM diagnosis output against schema."""
    required = DIAG_OUTPUT_SCHEMA["required"]
    errors = []

    for field in required:
        if field not in diag_output:
            errors.append(f"Missing required field: {field}")

    if "confidence" in diag_output:
        valid = DIAG_OUTPUT_SCHEMA["properties"]["confidence"]["enum"]
        if diag_output["confidence"] not in valid:
            errors.append(f"Invalid confidence: {diag_output['confidence']}")

    if diag_output.get("confidence") == "low" and not diag_output.get("needs_human_action"):
        errors.append("confidence=low requires needs_human_action=true")

    return {"valid": len(errors) == 0, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Prepare structured LLM prompt for failure diagnosis")
    parser.add_argument("--env-report", required=True, help="env_report.json from Stage ①")
    parser.add_argument("--setup-report", required=True, help="setup_report.json from Stage ②")
    parser.add_argument("--training-log", required=True, help="training_log.json from Stage ③")
    parser.add_argument("--verify-report", required=True, help="verify_report.json from Stage ④")
    parser.add_argument(
        "--sigcheck-report",
        default="",
        help="sigcheck_report.json from Stage ②.5 (optional — only when SIGCHECK failed)",
    )
    parser.add_argument("--output", required=True, help="Output diag_report.json path")
    parser.add_argument(
        "--adaptor-path", default="", help="Path to MegatronAdaptor for extracting patch target context"
    )
    parser.add_argument("--mr-list", default="", help="mr_list.json from FETCH_MRS (yesterday→today Megatron MRs)")
    parser.add_argument(
        "--diff-report", default="", help="diff_report.json from fetch_mrs (commits hitting patch targets)"
    )
    parser.add_argument(
        "--diag-output", default="", help="LLM response JSON to validate against schema (post-generation check)"
    )
    parser.add_argument(
        "--output-prompt",
        help="Output the prepared LLM prompt to this file",
    )
    args = parser.parse_args()

    with open(args.env_report, encoding="utf-8") as f:
        env_data = json.load(f)
    with open(args.setup_report, encoding="utf-8") as f:
        setup_data = json.load(f)

    # ③/④ artifacts are absent when SETUP fails on the smoke test (routed to
    # diag): read them only if present so the diag prompt can still be built
    # from the setup-report errors.
    training_data = {}
    verify_data = {}
    sigcheck_data = {}
    if args.training_log and Path(args.training_log).exists():
        with open(args.training_log, encoding="utf-8") as f:
            training_data = json.load(f)
    if args.verify_report and Path(args.verify_report).exists():
        with open(args.verify_report, encoding="utf-8") as f:
            verify_data = json.load(f)
    if args.sigcheck_report and Path(args.sigcheck_report).exists():
        with open(args.sigcheck_report, encoding="utf-8") as f:
            sigcheck_data = json.load(f)

    artifacts = {
        "env_report": env_data,
        "setup_report": setup_data,
        "training_log": training_data,
        "verify_report": verify_data,
        "sigcheck_report": sigcheck_data,
    }

    # ── Gather deterministic context ──
    adaptor_ctx = get_adaptor_context(args.adaptor_path) if args.adaptor_path else {}
    last_success = search_last_success()

    # ── Load MR delta (yesterday→today) ──
    mr_list = None
    if args.mr_list and Path(args.mr_list).exists():
        with open(args.mr_list, encoding="utf-8") as f:
            mr_list = json.load(f).get("mrs", [])
    diff_report = None
    if args.diff_report and Path(args.diff_report).exists():
        with open(args.diff_report, encoding="utf-8") as f:
            diff_report = json.load(f)

    failure_context = collect_failure_context(artifacts)
    extracted_errors = extract_errors_from_training_log(training_data)
    failed_scripts = collect_failed_test_scripts(training_data, args.adaptor_path) if args.adaptor_path else []
    similar_cases = search_similar_cases(failure_context)
    diag_prompt = build_diag_prompt(
        failure_context,
        extracted_errors,
        artifacts,
        similar_cases,
        adaptor_ctx=adaptor_ctx,
        last_success=last_success,
        mr_list=mr_list,
        diff_report=diff_report,
        failed_scripts=failed_scripts,
    )

    diag_input = {
        "diagnosis_time": failure_context["diagnosis_time"],
        "failed_stage": failure_context["failed_stage"],
        "failed_test_id": failure_context.get("failed_test_id"),
        "failed_checks": failure_context.get("failed_checks", []),
        "symptom": failure_context.get("symptom", ""),
        "extracted_errors": [{"test_id": e["test_id"], "reason": e["reason"]} for e in extracted_errors],
        "similar_past_cases": similar_cases,
        "last_success_commit": last_success.get("megatron_commit", ""),
        "mr_count": len(mr_list) if mr_list else 0,
        "mr_diff_hits": diff_report.get("summary", {}) if diff_report else {},
        "adapter_patch_targets": adaptor_ctx.get("targets", []),
        "adapter_patch_target_count": adaptor_ctx.get("target_count", 0),
        "megatron_commit": env_data.get("megatron", {}).get("commit_sha", "")[:12],
        "llm_prompt": diag_prompt,
        "output_schema": DIAG_OUTPUT_SCHEMA,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diag_input, f, indent=2)

    # ── Validate LLM output if provided ──
    if args.diag_output:
        diag_output_path = Path(args.diag_output)
        if diag_output_path.exists():
            with open(diag_output_path, encoding="utf-8") as f:
                llm_response = json.load(f)
            validation = validate_diag_output(llm_response)
            diag_input["validation"] = validation
            if validation["valid"]:
                print("  LLM output validation: PASS")
            else:
                print("  LLM output validation: FAIL")
                for e in validation["errors"]:
                    print(f"    - {e}", file=sys.stderr)

    print(f"OK: diag_report.json written to {args.output}")
    print(f"  Failed stage: {failure_context['failed_stage']}")
    print(f"  Failed test: {failure_context.get('failed_test_id', 'N/A')}")
    print(f"  Similar past cases: {len(similar_cases)}")
    print(f"  Errors extracted: {len(extracted_errors)}")
    print(f"  LLM prompt length: {len(diag_prompt)} chars")

    if args.output_prompt:
        with open(args.output_prompt, "w", encoding="utf-8") as f:
            f.write(diag_prompt)
        print(f"  LLM prompt written to: {args.output_prompt}")


if __name__ == "__main__":
    main()
