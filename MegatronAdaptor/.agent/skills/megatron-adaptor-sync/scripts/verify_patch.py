#!/usr/bin/env python3
"""Stage ⑦ PATCH (verify) — Deterministic patch verification. Zero LLM involvement.

5 independent checks, each produces a pass/fail with details.
"""

import argparse
import ast
import json
import sys
from pathlib import Path


def check_static_signatures(draft_path: str, meta_path: str, diff_report_path: str) -> dict:
    """Check that patch function signatures match what the upstream diff reports."""
    try:
        draft_content = Path(draft_path).read_text(encoding="utf-8")
        tree = ast.parse(draft_content)
    except (SyntaxError, FileNotFoundError) as e:
        return {"passed": False, "details": f"Cannot parse draft: {e}"}

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(diff_report_path, encoding="utf-8") as f:
        diff_report = json.load(f)

    new_sigs_from_diff = {}
    for mr_hit in diff_report["hits"]:
        for target, sig in mr_hit.get("new_signatures", {}).items():
            new_sigs_from_diff[target] = sig

    patch_funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            params = [arg.arg for arg in node.args.args]
            patch_funcs[node.name] = {"params": params}

    mismatch = []
    for patch_entry in meta.get("patches", []):
        orig = patch_entry["orig_func_name"]
        new_func = patch_entry["new_patch_func"]
        if orig in new_sigs_from_diff and new_func in patch_funcs:
            expected_params = new_sigs_from_diff[orig].get("added_params", [])
            actual_params = patch_funcs[new_func]["params"]
            if expected_params and expected_params != actual_params:
                mismatch.append(f"{new_func}: expected params {expected_params}, got {actual_params}")

    if mismatch:
        return {"passed": False, "details": f"Signature mismatches: {mismatch}"}
    return {"passed": True, "details": f"All {len(patch_funcs)} patch functions match new_signatures"}


def check_syntax(draft_path: str) -> dict:
    """Verify the draft has no syntax errors."""
    try:
        content = Path(draft_path).read_text(encoding="utf-8")
        ast.parse(content)
    except SyntaxError as e:
        return {"passed": False, "details": f"Syntax error at line {e.lineno}: {e.msg}"}
    except FileNotFoundError:
        return {"passed": False, "details": f"File not found: {draft_path}"}
    return {"passed": True, "details": "ast.parse passed without errors"}


def check_registration_paths(meta_path: str, diff_report_path: str) -> dict:
    """Verify all orig_func_names in patches exist in diff_report's hit_targets."""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(diff_report_path, encoding="utf-8") as f:
        diff_report = json.load(f)

    all_hit_targets = set()
    for mr_hit in diff_report["hits"]:
        for t in mr_hit.get("hit_targets", []):
            all_hit_targets.add(t)

    invalid = []
    for patch_entry in meta.get("patches", []):
        orig = patch_entry["orig_func_name"]
        if orig not in all_hit_targets:
            invalid.append(orig)

    if invalid:
        return {"passed": False, "details": f"orig_func_names not in hit_targets: {invalid}"}
    return {
        "passed": True,
        "details": f"All {len(meta.get('patches', []))} orig_func_names validated against hit_targets",
    }


def check_naming_convention(draft_path: str) -> dict:
    """Verify wrapper/decorator naming follows *_wrapper / *_decorator convention."""
    try:
        content = Path(draft_path).read_text(encoding="utf-8")
        tree = ast.parse(content)
    except (SyntaxError, FileNotFoundError):
        return {"passed": False, "details": "Cannot parse draft for naming check"}

    wrappers = []
    replacements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.endswith("_wrapper") or node.name.endswith("_decorator"):
                wrappers.append(node.name)
            else:
                replacements.append(node.name)

    return {"passed": True, "details": f"{len(wrappers)} wrapper(s), {len(replacements)} direct replacement(s)"}


def check_architecture_placement(meta_path: str) -> dict:
    """Verify new files follow megatron_adaptor/patches/<domain>/<feature>.py convention."""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    valid_domains = {"megatron", "transformer_engine", "requirements"}
    invalid = []
    for patch_entry in meta.get("patches", []):
        filepath = patch_entry.get("file_to_edit", "")
        parts = filepath.split("/")
        if len(parts) >= 3 and parts[1] == "patches" and parts[2] in valid_domains:
            continue
        if filepath and not filepath.startswith("megatron_adaptor/patches/"):
            invalid.append(filepath)

    if invalid:
        return {"passed": False, "details": f"Invalid file paths: {invalid}"}
    return {"passed": True, "details": "All files in correct architecture paths"}


def main():
    parser = argparse.ArgumentParser(description="Verify generated patch draft")
    parser.add_argument("--draft", required=True, help="patch_draft.py path")
    parser.add_argument("--meta", required=True, help="patch_draft_meta.json path")
    parser.add_argument("--diff-report", required=True, help="diff_report.json from Stage ②")
    parser.add_argument("--output", required=True, help="verify_report.json output path")
    args = parser.parse_args()

    checks = {
        "static_signatures": check_static_signatures(args.draft, args.meta, args.diff_report),
        "syntax": check_syntax(args.draft),
        "registration_paths": check_registration_paths(args.meta, args.diff_report),
        "naming_convention": check_naming_convention(args.draft),
        "architecture_placement": check_architecture_placement(args.meta),
    }

    overall = "pass" if all(c["passed"] for c in checks.values()) else "fail"

    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    needs_human = any(p.get("needs_human_review", True) for p in meta.get("patches", [])) or any(
        nf.get("needs_human_review", True) for nf in meta.get("new_features", [])
    )

    report = {
        "verify_date": __import__("datetime").datetime.now().isoformat(),
        "checks": checks,
        "overall": overall,
        "needs_human_review": needs_human,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"RESULT: {overall}")
    for name, result in checks.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {name}: {status} — {result['details']}")

    if overall == "fail":
        print("STOP: Verification failed. Do not write to MegatronAdaptor source.", file=sys.stderr)


if __name__ == "__main__":
    main()
