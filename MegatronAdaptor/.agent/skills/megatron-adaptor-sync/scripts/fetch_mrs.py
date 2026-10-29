#!/usr/bin/env python3
"""Stage ⑤ FETCH_MRS — Git-based commit and diff scanner. Zero external API.

Compares two commits in a Megatron-LM clone, extracts changed symbols, and
cross-references against MegatronAdaptor's registered patch targets.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent


def get_registered_targets(adaptor_path: str) -> list:
    """Extract target function names from ``register_patch`` calls in MA source."""
    targets = set()
    root = Path(adaptor_path) / "megatron_adaptor"
    patches_dir = root / "patches"
    features_dir = root / "features_manager"

    for py_file in list(patches_dir.rglob("*.py")) + list(features_dir.rglob("*.py")):
        try:
            content = py_file.read_text()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Attribute) and func.attr == "register_patch":
                    func_name = "register_patch"
                elif isinstance(func, ast.Name) and func.id == "register_patch":
                    func_name = "register_patch"
                if func_name and node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        targets.add(first_arg.value)
    return sorted(targets)


def git_log(repo_path: str, last_good: str, target: str) -> list[dict]:
    """Return commit list between two refs."""
    cmd = ["git", "-C", repo_path, "log", "--format=%H %s", f"{last_good}..{target}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"ERROR: git log failed: {e}", file=sys.stderr)
        return []

    commits = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        sha, _, subject = line.partition(" ")
        commits.append({"sha": sha, "subject": subject})
    return commits


def git_diff(repo_path: str, last_good: str, target: str) -> str:
    """Return unified diff text between two refs."""
    cmd = ["git", "-C", repo_path, "diff", last_good, target]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"ERROR: git diff failed: {e}", file=sys.stderr)
        return ""
    return result.stdout


def extract_changed_symbols(diff_text: str) -> dict:
    """From diff text, extract class/function definitions that were added,
    removed, or modified.  Returns ``{qualname: {added_params, removed_params, changes}}``.
    """
    symbols = {}
    current_file = None
    current_module_path = None

    for line in diff_text.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            filepath = line.split("/", 2)[-1] if "/" in line else ""
            current_file = filepath
            parts = filepath.replace("/", ".").replace(".py", "")
            current_module_path = parts
            continue

        if current_module_path and current_file and current_file.endswith(".py"):
            # function definitions
            m = re.match(r'^[+-]\s*(async\s+)?def\s+(\w+)\s*\(([^)]*)\)', line)
            if m:
                func_name = m.group(2)
                params = m.group(3).strip()
                prefix = "+" if line.startswith("+") else "-"
                qualname = f"{current_module_path}.{func_name}"
                if qualname not in symbols:
                    symbols[qualname] = {"added_params": [], "removed_params": [], "changes": []}
                param_list = [
                    p.strip().split(":")[0].split("=")[0]
                    for p in params.split(",")
                    if p.strip() and p.strip() != "self"
                ]
                if prefix == "+":
                    symbols[qualname]["added_params"] = param_list
                    symbols[qualname]["changes"].append("signature_added")
                else:
                    symbols[qualname]["removed_params"] = param_list
                    symbols[qualname]["changes"].append("signature_removed")

            # class definitions
            m = re.match(r'^[+-]\s*class\s+(\w+)\s*([:(].*)?', line)
            if m:
                class_name = m.group(1)
                qualname = f"{current_module_path}.{class_name}"
                if qualname not in symbols:
                    symbols[qualname] = {"added_params": [], "removed_params": [], "changes": []}
                symbols[qualname]["changes"].append("class_modified")

    return symbols


def compute_severity(hit_targets: list, symbols: dict) -> str:
    """Rate severity: high (signature change), medium (logic change), none."""
    has_signature_change = False
    has_other_change = False
    for target in hit_targets:
        if target in symbols:
            changes = symbols[target].get("changes", [])
            if any("signature" in c for c in changes):
                has_signature_change = True
            else:
                has_other_change = True
    if has_signature_change:
        return "high"
    if has_other_change:
        return "medium"
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="Scan git diff between two Megatron-LM commits against MA patch targets"
    )
    parser.add_argument("--repo-path", required=True, help="Path to Megatron-LM clone")
    parser.add_argument("--last-good", required=True, help="Last known-good commit SHA")
    parser.add_argument("--target", required=True, help="Target commit SHA being tested")
    parser.add_argument(
        "--adaptor-path", required=True, help="Path to MegatronAdaptor clone (for scanning registered targets)"
    )
    parser.add_argument("--output", required=True, help="Output diff_report.json path")
    args = parser.parse_args()

    # Same-commit shortcut
    if args.last_good == args.target:
        print(f"INFO: last-good == target ({args.last_good[:12]}) — delta is empty")
        report = {
            "scan_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo": args.repo_path,
            "last_good": args.last_good,
            "target": args.target,
            "commits": [],
            "registered_targets": [],
            "registered_target_count": 0,
            "hits": [],
            "summary": {"high": 0, "medium": 0, "none": 0},
        }
    else:
        commits = git_log(args.repo_path, args.last_good, args.target)
        diff_text = git_diff(args.repo_path, args.last_good, args.target)

        if not diff_text:
            print("WARNING: empty diff — no changes to scan")

        registered_targets = get_registered_targets(args.adaptor_path)
        symbols = extract_changed_symbols(diff_text)
        changed_symbol_paths = list(symbols.keys())

        # Cross-reference once for the entire delta range.
        # Per-commit attribution is NOT reliable from a range diff
        # (``git diff A..B`` cannot tell which commit changed which
        # symbol).  We report the full commit list separately and
        # compute hits against the whole range.
        hit_targets = [
            t
            for t in registered_targets
            if any(t == cs or cs.startswith(t + ".") or t.startswith(cs + ".") for cs in changed_symbol_paths)
        ]
        severity = compute_severity(hit_targets, symbols)
        new_signatures = {t: symbols[t] for t in hit_targets if t in symbols}

        hits = (
            [
                {
                    "hit_targets": hit_targets,
                    "changed_symbols": changed_symbol_paths,
                    "new_signatures": new_signatures,
                    "severity": severity,
                }
            ]
            if hit_targets
            else []
        )

        report = {
            "scan_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "repo": args.repo_path,
            "last_good": args.last_good,
            "target": args.target,
            "commits": commits,
            "registered_targets": registered_targets,
            "registered_target_count": len(registered_targets),
            "hits": hits,
            "summary": {
                "high": sum(1 for h in hits if h["severity"] == "high"),
                "medium": sum(1 for h in hits if h["severity"] == "medium"),
                "none": sum(1 for h in hits if h["severity"] == "none"),
            },
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    print(
        f"OK: {len(report['commits'])} commits, {len(report['hits'])} hits. "
        f"severity: high={s['high']} medium={s['medium']} none={s['none']}"
    )
    if s["high"] == 0 and s["medium"] == 0:
        print("INFO: No breaking changes detected against registered patch targets.")


if __name__ == "__main__":
    main()
