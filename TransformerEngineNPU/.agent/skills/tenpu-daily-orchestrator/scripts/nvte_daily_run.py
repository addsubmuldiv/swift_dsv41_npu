#!/usr/bin/env python3
"""NVTE daily PR 数据抓取器和报告仓库同步器。

抓取职责：抓取 NVIDIA/TransformerEngine 在目标北京自然日合入的 PR，写入
`output/<date>/prs.json`。需求分析由 agent 使用 `tenpu-upstream-value-judgment`
skill 综合 PR 与 commit 证据完成（本脚本不拉起 agent）。

同步职责：`--push` 时将 `output/<date>/report.md` 和 `prs.json` 提交到
datasets 仓库的 `nvte-daily-reports` 分支，形成开发侧可追溯的报告链。

日边界 = 北京自然日：报告日期 D 覆盖 `[UTC D-1 16:00, UTC D 16:00)` 合入的 PR，
与 GitHub 网页按 commit 日期分组一致。

Usage:
  python3 .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py [--date YYYY-MM-DD]
  python3 .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py --date YYYY-MM-DD --push
  python3 .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py --date YYYY-MM-DD --push --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BEIJING_TZ = dt.timezone(dt.timedelta(hours=8))

REPO = "NVIDIA/TransformerEngine"
TENPU_REPO = Path(__file__).resolve().parents[4]
DATASETS_WORKTREE = (
    Path(os.environ["TENPU_DATASETS_WORKTREE"]).expanduser().resolve()
    if os.environ.get("TENPU_DATASETS_WORKTREE")
    else TENPU_REPO.parent / "datasets"
)
OUTPUT_DIR = TENPU_REPO / "output"
DATASETS_BRANCH = "nvte-daily-reports"


def compute_target_date(reference: dt.datetime) -> str:
    """返回 `reference` 的前一天北京自然日（YYYY-MM-DD）。reference 需带时区。"""
    day = reference.astimezone(BEIJING_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return (day - dt.timedelta(days=1)).strftime("%Y-%m-%d")


def compute_search_window(date_str: str) -> tuple[str, str]:
    """报告日期 D（北京自然日）→ [start, end) UTC ISO 时间戳。

    北京 D 00:00 = UTC D-1 16:00（北京减 8 小时）；北京 D+1 00:00 = UTC D 16:00。
    """
    day = dt.datetime.strptime(date_str, "%Y-%m-%d")
    start = day - dt.timedelta(hours=8)
    end = day + dt.timedelta(hours=16)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_binary(name: str) -> str:
    """把命令名解析为绝对路径，避免 subprocess 裸命令名依赖 PATH（bandit B607）。

    bandit 对 `subprocess.run(["git", ...])` 报 start_process_with_partial_path。
    这里用 shutil.which 解析；命令缺失时抛 FileNotFoundError，与原行为一致。
    """
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(name)
    return path


def load_token() -> str:
    """读取进程环境或 GitHub CLI 的 API token，不依赖仓库内凭据文件。"""
    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        return token

    try:
        result = subprocess.run(
            [resolve_binary("gh"), "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "缺少 GitHub API 凭据：设置 GH_TOKEN/GITHUB_TOKEN，或安装 gh 后执行 gh auth login"
        ) from exc

    token = result.stdout.strip()
    if result.returncode == 0 and token:
        return token
    raise RuntimeError("缺少 GitHub API 凭据：设置 GH_TOKEN/GITHUB_TOKEN，或执行 gh auth login")


def github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def fetch_json(url: str, token: str, params: dict | None = None) -> dict:
    import requests

    resp = requests.get(url, params=params, headers=github_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()


PR_RE = re.compile(r"\(#(\d+)\)")


def _committer_date_utc(commit: dict) -> dt.datetime:
    raw = commit["commit"]["committer"]["date"]
    return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))


def fetch_merged_prs(token: str, date_str: str) -> list[dict]:
    """返回北京自然日 `date_str` 合入的 PR 列表，每个 PR dict 增补 `files`。

    与 GitHub 网页 commit 列表一致：walk main 分支 commit 历史，按 committer
    日期过滤到 `[date-1 16:00, date 16:00) UTC`，从 commit message 提取 PR 号
    （`(#NNNN)`），再拉每个 PR 的明细 + 变更文件。
    """
    start, end = compute_search_window(date_str)
    start_dt = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))

    # walk main（newest first），收集窗口内 commit
    in_window: list[dict] = []
    page = 1
    while True:
        batch = fetch_json(
            f"https://api.github.com/repos/{REPO}/commits",
            token,
            {"sha": "main", "per_page": 100, "page": page},
        )
        if not batch:
            break
        for commit in batch:
            cd = _committer_date_utc(commit)
            if start_dt <= cd < end_dt:
                in_window.append(commit)
        if _committer_date_utc(batch[-1]) < start_dt:
            break
        page += 1

    numbers: list[int] = []
    for commit in in_window:
        match = PR_RE.search(commit["commit"]["message"])
        if match:
            numbers.append(int(match.group(1)))
            continue
        pulls = fetch_json(
            f"https://api.github.com/repos/{REPO}/commits/{commit['sha']}/pulls", token
        )
        numbers.extend(p["number"] for p in pulls)

    # 去重并保持时间顺序
    seen: set[int] = set()
    numbers = [n for n in numbers if not (n in seen or seen.add(n))]

    prs = []
    for n in numbers:
        pr = fetch_json(f"https://api.github.com/repos/{REPO}/pulls/{n}", token)
        files = fetch_json(
            f"https://api.github.com/repos/{REPO}/pulls/{n}/files",
            token,
            {"per_page": 100},
        )
        pr["files"] = files
        prs.append(pr)
    return prs


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_git(cwd: Path, args: list[str]) -> None:
    subprocess.run(
        [resolve_binary("git"), "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def has_staged_changes(cwd: Path) -> bool:
    result = subprocess.run(
        [resolve_binary("git"), "-C", str(cwd), "diff", "--cached", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.returncode == 1


def commit_and_push(date_str: str, report: Path, data_path: Path) -> None:
    worktree = DATASETS_WORKTREE
    if not (worktree / ".git").exists():
        raise RuntimeError(f"datasets worktree 不存在: {worktree}（先完成报告仓库初始化）")
    target_report = worktree / "nvte-daily" / f"{date_str}.md"
    target_data = worktree / "nvte-daily" / "data" / f"{date_str}_prs.json"
    target_report.parent.mkdir(parents=True, exist_ok=True)
    target_data.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report, target_report)
    shutil.copyfile(data_path, target_data)
    run_git(worktree, ["add", "nvte-daily/"])
    if not has_staged_changes(worktree):
        print(f"skip push: datasets already contains report revision for {date_str}")
        return
    run_git(worktree, ["commit", "-m", f"report: NVTE daily value judgment {date_str}"])
    run_git(worktree, ["push", "origin", DATASETS_BRANCH])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVTE daily PR fetch/report sync runner")
    parser.add_argument("--date", help="目标北京自然日 YYYY-MM-DD（默认: 前一天）")
    parser.add_argument("--push", action="store_true", help="将定稿报告和 PR 输入同步到 datasets")
    parser.add_argument("--dry-run", action="store_true", help="只显示同步动作，不执行 commit/push")
    args = parser.parse_args(argv)

    date_str = args.date or compute_target_date(dt.datetime.now(dt.timezone.utc))
    out_dir = OUTPUT_DIR / date_str
    data_path = out_dir / "prs.json"
    report_path = out_dir / "report.md"
    # 幂等抓取：prs.json 已存在则跳过，支持当日重跑。
    if data_path.exists():
        print(f"skip fetch: {data_path} already exists")
    else:
        try:
            token = load_token()
            prs = fetch_merged_prs(token, date_str)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR fetch: {exc}", file=sys.stderr)
            return 1
        save_json(data_path, prs)
        print(f"fetched {len(prs)} PRs for Beijing day {date_str} -> {data_path}")

    if not args.push:
        print(
            "input ready. 请使用 tenpu-upstream-value-judgment skill 综合 PR 与 commit 证据，"
            f"将报告保存到 {report_path}，定稿后使用 --push 同步。"
        )
        return 0

    if not report_path.exists():
        print(f"ERROR push: 报告不存在 {report_path}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"dry-run: would push {report_path} + {data_path} to {DATASETS_BRANCH}")
        return 0
    try:
        commit_and_push(date_str, report_path, data_path)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR commit/push: {exc}", file=sys.stderr)
        return 1

    print(f"done: {date_str} report pushed to {DATASETS_BRANCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
