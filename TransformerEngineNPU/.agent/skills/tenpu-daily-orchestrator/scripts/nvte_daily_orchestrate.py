#!/usr/bin/env python3
"""NVTE 每日研判自动化编排脚本。

由 cron 每日 08:15（Asia/Shanghai）调用，自动完成：
  fetch(幂等) -> [report.md 已存在则 skip] -> headless claude 跑 analysis-only 写 report.md
  -> 本地校验 -> push datasets。

默认目标日期 = 前一个北京自然日；也支持 --date YYYY-MM-DD 指定。
headless agent 只允许写 output/<D>/report.md，禁止改运行时代码、建分支、commit 或 push。
HIGH 风险项以 pending human-spec-review 保留并照常同步，供开发持续看护。

Usage:
  python3 .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_orchestrate.py [--date YYYY-MM-DD] [--force] [--dry-run] [--skip-agent]
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from nvte_daily_run import OUTPUT_DIR, TENPU_REPO, compute_target_date  # noqa: E402

LOGS_DIR = OUTPUT_DIR / "logs"
CLAUDE_BIN = "claude"
AGENT_TIMEOUT_SECONDS = 20 * 60  # 20 分钟
FINAL_STATUSES_OK = {"analysis-complete", "no-op", "blocked", "needs-review"}

PROMPT_TEMPLATE = """你是 NVTE 每日研判的 headless 执行器。请对北京自然日 {date} 完成 analysis-only 研判并写出报告。

严格按以下步骤执行：
1. 用 Read 读取并遵循以下文件（直接读文件，不要调用 skill 工具）：
   - .agent/skills/tenpu-daily-orchestrator/SKILL.md
   - .agent/skills/tenpu-daily-orchestrator/references/single-report-schema.md
   - .agent/skills/tenpu-upstream-value-judgment/SKILL.md
   - .agent/skills/tenpu-upstream-value-judgment/references/nvte-npu-migration-knowledge.md
   - .agent/skills/shared-references/nvte-tenpu-architecture-differences.md
2. 读取 output/{date}/prs.json（当日合入 PR 输入，含 files）。用 grep/read 对照仓库代码做 evidence-driven 研判；MEDIUM/HIGH 或涉及架构边界的行为必须读取共享架构知识中命中的领域并记录 architecture pattern id。对每个行为单元产出 value_case、adaptation_decision(ADAPT|TRANSLATE|REFERENCE|SKIP)、execution_action、tenpu_coverage、confidence、risk 分级和 Risk-Based Adaptation Spec。
3. 把报告写入 output/{date}/report.md（唯一产物）。要求：
   - schema_version 3，文末恰好一个 daily_te_npu YAML 块。
   - knowledge.triage、knowledge.architecture 均记录对应文件的 git blob SHA；requirement 记录命中的 architecture_pattern_ids 和当前代码证据。
   - run_mode: analysis-only；phase: analyzed 或 specified；final_status: analysis-complete 或 no-op。
   - upstream_range.mode: calendar-audit；old_sha/new_sha 为 null；双游标不推进；bootstrap.required: true。
   - 从 output/ 或 datasets 工作树中上一份同分支报告按稳定 task id 继承 open_adaptations（去重）。
   - knowledge_feedback 写候选（可为空数组）。
   - 人类可读章节：执行摘要、上游证据、需求与价值判断、Adaptation Spec、风险与 Gate 必须完整；后续章节写"待执行"。
   - HIGH 风险项（FP8/MXFP8/HIF8、Attention、MoE、Distributed、Kernel、Autograd、Checkpoint、Memory Layout、Collective 默认 HIGH）：spec.status: draft；next_action: human-spec-review；不得进入 design/实现。
   - created_at/updated_at 用当前 Asia/Shanghai 时间。
4. 硬性禁止：
   - 不得修改任何 TENPU 运行时代码（transformer_engine/ 下任何 .py/.cpp/.cu 等）。
   - Agent 不得创建/切换 git 分支，不得 git commit、git push 或直接修改 datasets；同步由 Runner 完成。
   - 不得运行 torch_npu 相关验证或 pytest。
   - 只允许写 output/{date}/report.md 这一个文件。
5. 完成后，最后一行输出：STATUS: done <final_status>"""


def log(msg: str, date_str: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_file = LOGS_DIR / f"nvte-daily-orchestrate-{date_str}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_runner(date_str: str, push: bool = False) -> int:
    """调用 nvte_daily_run.py；返回 exit code。"""
    cmd = [sys.executable, str(SCRIPT_DIR / "nvte_daily_run.py"), "--date", date_str]
    if push:
        cmd.append("--push")
    r = subprocess.run(cmd, cwd=str(TENPU_REPO), capture_output=True, text=True, check=False)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip(), file=sys.stderr)
    return r.returncode


def extract_final_status(date_str: str) -> str | None:
    """校验 report.md；通过则返回 final_status，否则返回 None。"""
    report = OUTPUT_DIR / date_str / "report.md"
    if not report.exists():
        return None
    txt = report.read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)\n```", txt, re.S)
    if len(blocks) != 1:
        print(f"  validate: YAML block count = {len(blocks)} (期望 1)", file=sys.stderr)
        return None
    try:
        import yaml

        data = yaml.safe_load(blocks[0])
        top = data.get("daily_te_npu") if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        print(f"  validate: YAML 解析失败: {exc}", file=sys.stderr)
        return None
    if not isinstance(top, dict):
        print("  validate: 缺少顶层键 daily_te_npu", file=sys.stderr)
        return None
    fs = top.get("final_status")
    if fs not in FINAL_STATUSES_OK:
        print(f"  validate: final_status={fs!r} 不在 {sorted(FINAL_STATUSES_OK)}", file=sys.stderr)
        return None
    return fs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVTE daily orchestration runner")
    parser.add_argument("--date", help="目标北京自然日 YYYY-MM-DD（默认: 前一天）")
    parser.add_argument(
        "--force", action="store_true", help="report.md 已存在也重新生成（手动恢复用）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将执行的步骤，不拉起 agent、不 push"
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="调试：跳过 headless claude，校验现有报告后同步（需报告已存在）",
    )
    args = parser.parse_args(argv)

    date_str = args.date or compute_target_date(dt.datetime.now(dt.timezone.utc))
    report_path = OUTPUT_DIR / date_str / "report.md"

    # Step 1: fetch（幂等）
    log(f"step fetch: nvte_daily_run.py --date {date_str}", date_str)
    if not args.dry_run:
        rc = run_runner(date_str, push=False)
        if rc != 0:
            log(f"ERROR fetch failed rc={rc}", date_str)
            return 1

    # Step 2: 复用已有报告，允许同步失败后重试；只有 --force 才重新生成。
    reuse_existing_report = report_path.exists() and not args.force
    if reuse_existing_report:
        log(f"reuse: report.md already exists ({report_path}); validate and retry sync", date_str)

    # Step 3: headless agent 生成报告
    if args.skip_agent or reuse_existing_report:
        log("skip-agent: 跳过 headless claude，直接校验并同步现有报告", date_str)
    elif args.dry_run:
        print(f"  dry-run: would run `{CLAUDE_BIN} -p` (cwd={TENPU_REPO}) for date {date_str}")
    else:
        prompt = PROMPT_TEMPLATE.format(date=date_str)
        cmd = [CLAUDE_BIN, "-p", "--output-format", "text", prompt]
        log(f"step agent: claude -p ... (cwd={TENPU_REPO})", date_str)
        try:
            r = subprocess.run(
                cmd,
                cwd=str(TENPU_REPO),
                capture_output=True,
                text=True,
                timeout=AGENT_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log(f"ERROR agent timeout after {AGENT_TIMEOUT_SECONDS}s", date_str)
            return 1
        if r.stdout.strip():
            print("--- agent stdout (tail) ---")
            print(r.stdout.strip()[-4000:])
        if r.stderr.strip():
            print("--- agent stderr (tail) ---", file=sys.stderr)
            print(r.stderr.strip()[-2000:], file=sys.stderr)
        if r.returncode != 0:
            log(f"ERROR agent exited rc={r.returncode}", date_str)
            return 1

    # Step 4: 校验报告
    final_status = extract_final_status(date_str)
    if args.dry_run:
        if final_status:
            print(f"  validate ok: final_status={final_status} (dry-run)")
        else:
            print("  dry-run: report.md 由 agent 生成，dry-run 不做存在性校验")
        print(f"  dry-run: would run `nvte_daily_run.py --date {date_str} --push`")
        return 0
    if final_status is None:
        log("ERROR validate failed: report.md 缺失或 YAML/final_status 不合法", date_str)
        return 1
    log(f"validate ok: final_status={final_status}", date_str)

    log(f"step push: sync {report_path} to datasets", date_str)
    rc = run_runner(date_str, push=True)
    if rc != 0:
        log(f"ERROR push failed rc={rc}", date_str)
        return 1
    log(f"done: {date_str} report pushed", date_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
