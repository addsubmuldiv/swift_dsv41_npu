#!/usr/bin/env python3
"""Stage ① FETCH — Deterministic multi-repo fetcher. Zero LLM involvement.

Clones or updates three repositories to the working directory:
  - Megatron-LM       (code reference only, NOT installed)
  - MegatronAdaptor    (pip install -e in stage ②)
  - TransformerEngineNPU (pip install -e in stage ②)

Each repo is checked out to its target branch, and commit info is
recorded in env_report.json for downstream stages.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


MEGATRON_REPO = "https://gitcode.com/GitHub_Trending/me/Megatron-LM.git"
MA_REPO = "https://gitcode.com/Ascend/MegatronAdaptor.git"
TE_REPO = "https://gitcode.com/Ascend/TransformerEngineNPU.git"

ADAPTOR_IMPORT_LINE = "import megatron_adaptor\n"
SCRIPTS_DIR = Path(__file__).resolve().parent


def get_last_good_commit(megatron_branch: str = "") -> dict:
    """Read the most recent successful run's commit from the flywheel.

    This is the anchor point: yesterday's version was verified working.
    Today's MRs (from this commit to HEAD) are the delta that matters.

    If ``megatron_branch`` is provided, only considers entries for that branch.
    """
    case_library = SCRIPTS_DIR / "nightly_case_library.jsonl"
    if not case_library.exists():
        return {}

    try:
        lines = case_library.read_text().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            if case.get("overall_status") != "pass":
                continue
            if megatron_branch:
                if case.get("megatron_branch") != megatron_branch:
                    continue
            return {
                "commit": case.get("megatron_commit", ""),
                "run_date": case.get("run_date", ""),
            }
    except Exception:  # nosec B110 — case library read is best-effort
        pass
    return {}


def resolve_yesterday_commit(clone_dir: str, branch: str = "main") -> str:
    """Resolve the first commit after yesterday 00:00 UTC from the cloned repo.

    This is the fallback anchor when neither --last-good nor flywheel
    provides a known-good commit.  Uses git log rather than a placeholder
    string so downstream scripts (fetch_mrs.py, etc.) receive a real SHA.

    ``branch`` is the Megatron-LM branch being built (e.g. ``main``,
    ``core_r0.18.0``).  The function queries ``origin/<branch>`` so that
    non-main branches get a last_good commit from their own history.
    """
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    remote_ref = f"origin/{branch}"
    try:
        # First merge commit after yesterday 00:00 (most likely the
        # "sync point" in a busy repo).
        result = run_cmd(
            ["git", "log", "--merges", "--reverse", "--since", yesterday, "--format=%H", "-1", remote_ref],
            cwd=clone_dir,
            timeout=30,
        )
        sha = result.stdout.strip()
        if sha:
            print(f"INFO: Resolved yesterday's first merge commit (branch={branch}): {sha[:12]}")
            return sha
    except Exception:  # nosec B110 — first resolution attempt failed; try fallback below
        pass

    # Fallback: any commit after yesterday 00:00.
    try:
        result = run_cmd(
            ["git", "log", "--reverse", "--since", yesterday, "--format=%H", "-1", remote_ref],
            cwd=clone_dir,
            timeout=30,
        )
        sha = result.stdout.strip()
        if sha:
            print(f"INFO: Resolved yesterday's first commit (branch={branch}): {sha[:12]}")
            return sha
    except Exception:  # nosec B110 — fallback resolution failed; return empty anchor
        pass

    return ""


def run_cmd(cmd: list, cwd: str = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a subprocess command with error handling."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd, timeout=timeout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"  stderr: {e.stderr}", file=sys.stderr)
        raise
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out: {' '.join(cmd)}", file=sys.stderr)
        raise


def ensure_clone(work_dir: str, repo_name: str, repo_url: str, target: str = "origin/main") -> dict:
    """Clone a repo if not exists, or fetch + checkout target ref.

    Shared by Megatron-LM, MegatronAdaptor, and TransformerEngineNPU.

    ``target`` is a remote ref like ``origin/main``.  The function extracts
    the local branch name (e.g. ``main``) and checks it out as a named
    branch (not detached HEAD) so that subsequent runs can ``git pull``.
    """
    clone_dir = Path(work_dir) / repo_name
    branch_name = target.replace("origin/", "")
    action = None

    if clone_dir.exists() and (clone_dir / ".git").exists():
        print(f"INFO: {repo_name} clone exists at {clone_dir}. Fetching latest...")
        # Ensure origin points to the configured repo (handles --repo-url changes).
        run_cmd(["git", "remote", "set-url", "origin", repo_url], cwd=str(clone_dir))
        try:
            run_cmd(["git", "fetch", "origin", "--prune"], cwd=str(clone_dir))
        except Exception as e:
            print(
                f"WARNING: git fetch failed for {repo_name} ({e}) — using current clone state. Results may be stale.",
                file=sys.stderr,
            )

        # Discard any dirty tracked files BEFORE checkout, otherwise checkout
        # fails on modified tracked files when branches share one clone directory.
        run_cmd(["git", "reset", "--hard", "HEAD"], cwd=str(clone_dir))

        # Pre-clean known disposable training artifacts that may be owned by
        # root (NPU driver / Ascend runtime can create files as root).
        # Without this, git clean -fd fails → FETCH fails → import injection
        # skipped → tests crash with "Megatron requires CUDA".
        for pattern in ["ckpt_*", "*.pt", "*.bin", "output"]:
            for p in clone_dir.glob(pattern):
                try:
                    if p.is_dir():
                        subprocess.run(  # nosec B607 — sudo/rm on PATH, trusted
                            ["sudo", "rm", "-rf", str(p)],
                            timeout=30,
                            capture_output=True,
                            check=False,
                        )
                    elif p.is_file():
                        subprocess.run(  # nosec B607 — sudo/rm on PATH, trusted
                            ["sudo", "rm", "-f", str(p)],
                            timeout=10,
                            capture_output=True,
                            check=False,
                        )
                except Exception:  # nosec B110 — pre-clean is best-effort
                    pass

        try:
            run_cmd(["git", "clean", "-fd"], cwd=str(clone_dir))
        except Exception:
            # git clean -fd can fail if untracked files are owned by another
            # user (e.g. root-owned ckpt dirs from prior training runs).
            # This is non-fatal — checkout should still succeed as long as
            # the untracked files don't shadow tracked paths.  List what
            # remains so the operator can investigate.
            print(
                f"WARNING: git clean -fd failed for {repo_name} — untracked files may remain (check permissions).",
                file=sys.stderr,
            )
            try:
                dry = subprocess.run(  # nosec B607 — git on PATH, trusted
                    ["git", "clean", "-fd", "--dry-run"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(clone_dir),
                    check=False,
                )
                if dry.stdout.strip():
                    print(f"  Untracked files that could not be removed:\n{dry.stdout.strip()}", file=sys.stderr)
            except Exception:  # nosec B110 — dry-run diagnostic is best-effort
                pass

        try:
            # -B: create or reset the local branch to track the remote ref.
            # This avoids detached HEAD state and ensures the branch is
            # at the latest remote commit.
            run_cmd(["git", "checkout", "-B", branch_name, target], cwd=str(clone_dir))
            # Pull latest after checkout
            try:
                run_cmd(["git", "pull", "origin", branch_name], cwd=str(clone_dir), timeout=120)
            except Exception:  # nosec B110 — already on target, pull is best-effort
                pass
        except Exception as e:
            print(
                f"WARNING: git checkout {target} failed for {repo_name} ({e}) — will use current HEAD.", file=sys.stderr
            )
        action = "updated"
    else:
        # Directory exists but .git missing → corrupted clone → clean up first
        if clone_dir.exists():
            print(f"INFO: Removing broken clone at {clone_dir} (missing .git)")
            shutil.rmtree(str(clone_dir))
        print(f"INFO: Cloning {repo_name} to {clone_dir}...")
        clone_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", repo_url, str(clone_dir)])
        # git clone leaves us on the default branch; explicitly checkout
        # the target ref as a named branch (not detached HEAD).
        try:
            run_cmd(["git", "checkout", "-B", branch_name, target], cwd=str(clone_dir))
        except Exception as e:
            print(
                f"WARNING: git checkout {target} after clone failed for {repo_name} ({e}) — will use default branch.",
                file=sys.stderr,
            )
        action = "cloned"

    # Extract git info
    sha_result = run_cmd(["git", "rev-parse", "HEAD"], cwd=str(clone_dir))
    commit_sha = sha_result.stdout.strip()

    date_result = run_cmd(["git", "log", "-1", "--format=%cI", "HEAD"], cwd=str(clone_dir))
    commit_date = date_result.stdout.strip()

    branch_result = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(clone_dir))
    branch = branch_result.stdout.strip()

    subject_result = run_cmd(["git", "log", "-1", "--format=%s", "HEAD"], cwd=str(clone_dir))
    commit_subject = subject_result.stdout.strip()

    return {
        "action": action,
        "clone_dir": str(clone_dir),
        "repo_name": repo_name,
        "commit_sha": commit_sha,
        "commit_date": commit_date,
        "branch": branch,
        "repo_url": repo_url,
        "commit_subject": commit_subject,
    }


def inject_adaptor_import(clone_dir: str) -> bool:
    """Ensure 'import megatron_adaptor' is present in pretrain_gpt.py."""
    pretrain_path = Path(clone_dir) / "pretrain_gpt.py"
    if not pretrain_path.exists():
        print(f"WARNING: {pretrain_path} not found — cannot inject adaptor import")
        return False

    content = pretrain_path.read_text()
    lines = content.splitlines()

    # Check if already present
    if ADAPTOR_IMPORT_LINE.strip() in content:
        print("INFO: adaptor import already present in pretrain_gpt.py")
        return True

    # Inject after 'import torch' or at the top of the file
    injected = False
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if not injected and re.match(r'^\s*import\s+torch\s*$', line):
            new_lines.append(ADAPTOR_IMPORT_LINE.rstrip())
            injected = True

    if not injected:
        # Insert at the top, after the shebang/docstring if present
        insert_pos = 0
        if lines and lines[0].startswith("#!"):
            insert_pos = 1
        if len(lines) > 1 and lines[1].startswith('"""'):
            # Skip docstring
            for j in range(1, len(lines)):
                if '"""' in lines[j] and j > 1:
                    insert_pos = j + 1
                    break
        new_lines = lines[:insert_pos] + [ADAPTOR_IMPORT_LINE.rstrip()] + lines[insert_pos:]

    pretrain_path.write_text("\n".join(new_lines) + "\n")
    print("INFO: Injected 'import megatron_adaptor' into pretrain_gpt.py")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch latest repos for MegatronAdaptor nightly validation")
    parser.add_argument(
        "--work-dir", default="/home/ma_daily_ci/workspace", help="Root working directory where all repos are cloned"
    )
    parser.add_argument("--output", required=True, help="Output env_report.json path")
    # ── Megatron-LM ──
    parser.add_argument("--megatron-repo-url", default=MEGATRON_REPO, help="Megatron-LM repository URL")
    parser.add_argument("--megatron-target", default="origin/main", help="Megatron-LM target branch/ref")
    # ── MegatronAdaptor ──
    parser.add_argument("--ma-repo-url", default=MA_REPO, help="MegatronAdaptor repository URL")
    parser.add_argument("--ma-target", default="origin/main", help="MegatronAdaptor target branch/ref")
    # ── TransformerEngineNPU ──
    parser.add_argument("--te-repo-url", default=TE_REPO, help="TransformerEngineNPU repository URL")
    parser.add_argument("--te-target", default="origin/main", help="TransformerEngineNPU target branch/ref")
    # ── Legacy (kept for backward compat) ──
    parser.add_argument(
        "--adaptor-path", default="", help="[deprecated] Path to MegatronAdaptor — now derived from --work-dir"
    )
    parser.add_argument("--last-good", default="", help="Last known-good Megatron commit (default: auto-resolve)")
    parser.add_argument("--target", default="origin/main", help="[deprecated] Use --megatron-target instead")
    args = parser.parse_args()

    # Support legacy --target as fallback for --megatron-target
    megatron_target = args.megatron_target
    if args.target != "origin/main" and args.megatron_target == "origin/main":
        megatron_target = args.target

    errors = []
    megatron_info = {}
    adaptor_info = {}
    te_info = {}
    import_patched = False
    fetch_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Resolve last-good from static sources (user arg or flywheel) ──
    last_good = args.last_good
    last_good_source = "user" if last_good else ""
    if not last_good:
        branch_name = megatron_target.replace("origin/", "")
        flywheel_good = get_last_good_commit(branch_name)
        if flywheel_good.get("commit"):
            last_good = flywheel_good["commit"]
            last_good_source = "flywheel"
            print(f"INFO: Using flywheel last-good for {branch_name}: {last_good[:12]}")

    # ── Fetch Megatron-LM ──
    try:
        megatron_info = ensure_clone(args.work_dir, "Megatron-LM", args.megatron_repo_url, megatron_target)

        # ── Resolve last-good from git log if still unknown (post-clone) ──
        if not last_good and megatron_info.get("clone_dir"):
            last_good = resolve_yesterday_commit(megatron_info["clone_dir"], branch_name)
            last_good_source = "git-log"
            if last_good:
                print(f"INFO: Using git-log last-good: {last_good[:12]}")
            else:
                print("WARNING: Could not resolve yesterday's commit from git log")
        pass  # commit info (sha, subject, date) already set by ensure_clone
    except Exception as e:
        errors.append(f"Megatron-LM fetch failed: {e}")
        megatron_info = {"error": str(e)}

    # ── Fetch MegatronAdaptor ──
    try:
        adaptor_info = ensure_clone(args.work_dir, "MegatronAdaptor", args.ma_repo_url, args.ma_target)
        adaptor_info["exists"] = True
    except Exception as e:
        errors.append(f"MegatronAdaptor fetch failed: {e}")
        adaptor_info = {"error": str(e), "exists": False}

    # ── Fetch TransformerEngineNPU ──
    try:
        te_info = ensure_clone(args.work_dir, "TransformerEngineNPU", args.te_repo_url, args.te_target)
        te_info["exists"] = True
    except Exception as e:
        errors.append(f"TransformerEngineNPU fetch failed: {e}")
        te_info = {"error": str(e), "exists": False}

    # ── Inject adaptor import ──
    if megatron_info.get("clone_dir"):
        try:
            import_patched = inject_adaptor_import(megatron_info["clone_dir"])
        except Exception as e:
            errors.append(f"Import injection failed: {e}")

    # ── Resolve last_known_good_date ──
    last_known_good_date = ""
    if last_good and megatron_info.get("clone_dir"):
        try:
            date_result = subprocess.run(  # nosec B607 — git on PATH, trusted
                ["git", "log", "-1", "--format=%cI", last_good],
                capture_output=True,
                text=True,
                check=True,
                cwd=megatron_info["clone_dir"],
                timeout=30,
            )
            last_known_good_date = date_result.stdout.strip()
        except Exception:  # nosec B110 — best-effort; leave date empty
            pass

    # ── Build output ──
    # "fail" when Megatron-LM has no usable clone_dir (fetch completely failed).
    # "partial" when MLM clone exists but there are non-fatal errors (e.g.
    # import injection failed, or adaptor/TE fetch failed).
    megatron_has_clone = bool(megatron_info.get("clone_dir"))
    if not errors:
        status = "ok"
    elif megatron_has_clone:
        status = "partial"
    else:
        status = "fail"

    report = {
        "fetch_time": fetch_time,
        "work_dir": str(Path(args.work_dir).resolve()),
        "last_good_commit": last_good,
        "last_good_source": last_good_source,
        "target_commit": megatron_info.get("commit_sha", ""),
        "last_known_good_commit": last_good,
        "last_known_good_date": last_known_good_date,
        "megatron": megatron_info,
        "adaptor": adaptor_info,
        "transformer_engine_npu": te_info,
        "import_patched_in_pretrain": import_patched,
        "status": status,
        "errors": errors,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    def _short(repo_info):
        if not repo_info or "error" in repo_info:
            return f"ERROR: {repo_info.get('error', 'unknown')[:80]}"
        sha = repo_info.get("commit_sha", "N/A")[:12]
        subject = repo_info.get("commit_subject", "")[:80]
        branch = repo_info.get("branch", "N/A")
        return f"{sha} [{branch}] {subject}"

    print(f"OK: env_report.json written to {args.output}")
    print(f"  Status: {status}")
    print(f"  Megatron-LM:       {_short(megatron_info)}")
    print(f"  MegatronAdaptor:   {_short(adaptor_info)}")
    print(f"  TransformerEngine: {_short(te_info)}")
    print(f"  Adaptor import patched: {import_patched}")

    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
