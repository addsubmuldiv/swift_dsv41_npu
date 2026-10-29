"""
MegatronAdaptor Sync CI 结果推送脚本

功能：
1. 读取 megatron-adaptor-sync 夜间验证产出 (combined_summary.json + state 文件)
2. 合成 Markdown 汇总报告（对齐原版 MegatronAdaptor CI 报告格式）
3. 推送到 MegatronAdaptorCI 代码仓的 sync_build_results/ 目录
"""
# pylint: disable=redefined-outer-name

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── 配置（全部在 /home/ma_daily_ci/workspace/ 下） ──────────────────────

WORKSPACE = "/home/ma_daily_ci/workspace"
OUTPUT_DIR = Path(WORKSPACE) / "output"


def _load_ci_config():
    """从 .push_config.json 读取 CI 代码仓配置 + token，失败则使用默认值。"""
    config_path = Path(__file__).resolve().parent / ".push_config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "url": data["ci_repo"]["remote_url"],
            "token": data["repo"]["api_token"],
            "branch": "main",
            "clone_dir": data["ci_repo"]["local_dir"],
        }
    except Exception:
        return {
            "url": "https://gitcode.com/<user>/MegatronAdaptorCI.git",
            "token": "",  # nosec B105
            "branch": "main",
            "clone_dir": f"{WORKSPACE}/MegatronAdaptorCI_repo",
        }


CI_REPO_CONFIG = _load_ci_config()

# 分支组合（唯一来源 scripts/branch_matrix.json）
BRANCH_COMBOS_PATH = Path(__file__).resolve().parent / "branch_matrix.json"


def _load_branch_combos() -> list:
    """从 branch_matrix.json 读取分支组合；缺失/损坏即报错退出（关键配置不静默降级）。"""
    try:
        with open(BRANCH_COMBOS_PATH, encoding="utf-8") as f:
            combos = json.load(f)["branch_combos"]
        if not combos:
            raise ValueError("branch_combos 为空")
        return combos
    except Exception as e:
        raise SystemExit(f"[push_sync_ci_results] 无法读取 {BRANCH_COMBOS_PATH}: {e}")


_BRANCH_COMBOS = _load_branch_combos()
BRANCH_COMBOS = [c["megatron_branch"] for c in _BRANCH_COMBOS]
CONDA_ENV_MAP = {c["megatron_branch"]: c["conda_env"] for c in _BRANCH_COMBOS}

# ── Git 认证 ────────────────────────────────────────────────────────────


def _clean_url():
    """返回 config 的干净 remote URL（不含 token/用户名；认证由 credential helper 提供）。"""
    return CI_REPO_CONFIG["url"]


def _credential_helper():
    """返回 git 的 credential.helper 参数。

    token 由 helper 脚本从 0600 的 .push_config.json 读取，不进 argv/env、
    不落盘 .git/config；remote URL 保持干净。helper 路径非敏感，可进 argv。
    """
    helper = Path(__file__).resolve().parent / "git-credential-helper.sh"
    return ["-c", f"credential.helper={helper}"]


def ensure_clean_url():
    """确保 remote URL 不含凭据（清洗可能残留的 tokenized URL）。"""
    subprocess.run(
        [shutil.which("git") or "git", "remote", "set-url", "origin", _clean_url()],
        check=True,
        capture_output=True,
        text=True,
    )


# ── 数据读取 ────────────────────────────────────────────────────────────


def date_to_archive_path(date_str: str) -> str:
    """将日期字符串 YYYY-MM-DD 转换为归档路径 YYYY-MM/MM-DD。"""
    parts = date_str.split("-")
    return f"{parts[0]}-{parts[1]}/{parts[1]}-{parts[2]}"


def get_latest_build_date() -> str | None:
    """获取 output/ 下最新的构建日期（按月分层 YYYY-MM/MM-DD，含 _rN 归档）。"""
    dates = []
    for month_dir in OUTPUT_DIR.iterdir():
        if not month_dir.is_dir():
            continue
        parts = month_dir.name.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            continue  # 不是 YYYY-MM 月份目录
        for day_dir in month_dir.iterdir():
            if not day_dir.is_dir():
                continue
            m = day_dir.name.split("_")[0]
            dm = m.split("-")
            if len(dm) != 2 or len(dm[0]) != 2 or len(dm[1]) != 2:
                continue  # 不是 MM-DD（或 MM-DD_rN）
            dates.append(f"{month_dir.name}-{dm[1]}")
    if not dates:
        return None
    return sorted(dates, reverse=True)[0]


def _is_date_dir(name: str) -> bool:
    """检查目录名是否为 YYYY-MM 月份目录（顶层按月分层）。"""
    try:
        parts = name.split("-")
        return len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 2
    except ValueError:
        return False


def load_json(path: Path) -> dict | None:
    """安全加载 JSON 文件。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"⚠️  解析 {path} 失败: {e}")
        return None


def _format_time(iso_str: str) -> str:
    """将 ISO 时间字符串转为可读格式。"""
    if not iso_str:
        return "N/A"
    try:
        # "2026-07-24T07:21:34Z" → "2026-07-24 07:21:34"
        return iso_str.replace("T", " ").replace("Z", "")
    except Exception:
        return iso_str


def read_build_results(build_date: str) -> dict:
    """读取指定日期的所有构建结果（含 auto_fix 数据）。

    本地 output 已按月分层：output/YYYY-MM/MM-DD（与 CI 仓归档路径一致）。
    """
    date_dir = OUTPUT_DIR / date_to_archive_path(build_date)

    # 1) 顶层汇总
    summary_path = date_dir / "combined_summary.json"
    summary = load_json(summary_path)
    if not summary:
        print(f"❌ 未找到 combined_summary.json: {summary_path}")
        return None

    # 2) 各分支详情
    branches = []
    env_info = None

    for branch_name in BRANCH_COMBOS:
        state_dir = date_dir / branch_name / "state"
        if not state_dir.exists():
            continue

        branch_data = {
            "name": branch_name,
            "conda_env": CONDA_ENV_MAP.get(branch_name, "unknown"),
            "env_report": load_json(state_dir / "env_report.json"),
            "training_log": load_json(state_dir / "training_log.json"),
            "verify_report": load_json(state_dir / "verify_report.json"),
            "final_report": load_json(state_dir / "final_report.json"),
            "diag_report": load_json(state_dir / "diag_report.json"),
        }

        # 从 summary 中获取该分支的 overall 状态
        for r in summary.get("results", []):
            if r.get("branch") == branch_name or r.get("ma_branch") == branch_name:
                branch_data["overall"] = r.get("overall", "unknown")
                branch_data["stages"] = r.get("stages", {})
                branch_data["errors"] = r.get("errors", [])
                branch_data["needs_llm_diag"] = r.get("needs_llm_diag", False)
                break

        # 取第一个有效分支的环境信息
        if env_info is None and branch_data["env_report"]:
            er = branch_data["env_report"]
            setup_report = load_json(state_dir / "setup_report.json")
            env_info = {"fetch_time": er.get("fetch_time", "")}
            if setup_report:
                pkgs = setup_report.get("packages", {})
                env_info["python"] = pkgs.get("python", "")
                env_info["torch"] = pkgs.get("torch", "")
                env_info["te_version"] = pkgs.get("te_version", "")
                env_info["npu"] = setup_report.get("npu", {})

        # 3) PR 信息（如有修复 PR）——先读，供 auto_fix 最终状态判定
        pr_info = load_json(state_dir / "pr_info.json")
        branch_data["pr_info"] = pr_info

        # 4) auto_fix 数据（如有）
        # 以最终效果为准：是否已修复取决于 PR 是否已创建（create_issue_pr.py 只在
        # daily_fix_<branch> 已推送 fork 时才建 PR），而非构建时 auto_fix/state 目录
        # 是否存在——auto-fix 流程不写该目录，会导致报告恒为"待修复"。
        auto_fix_dir = date_dir / "auto_fix" / branch_name
        auto_fix_state = auto_fix_dir / "state"
        branch_data["auto_fix"] = {
            "bisect": load_json(auto_fix_state / "bisect_report_corrected.json")
            or load_json(auto_fix_state / "bisect_report.json"),
            "fix_plan": load_json(auto_fix_state / "fix_plan.json"),
            "root_cause": load_json(auto_fix_state / "root_cause_analysis_v2.json")
            or load_json(auto_fix_state / "root_cause_analysis.json"),
            "available": auto_fix_state.exists() or bool(pr_info and pr_info.get("pr_number")),
        }

        branches.append(branch_data)

    # 收集 failure reasons（从 verify_report）
    for b in branches:
        verify = b.get("verify_report") or {}
        b["failure_reasons"] = []
        for fd in verify.get("failure_details", []):
            b["failure_reasons"].append(fd.get("details", str(fd)))

        # 也从 diag_report 提取 extracted_errors
        diag = b.get("diag_report") or {}
        for err in diag.get("extracted_errors", []):
            reason = f"{err.get('test_id', '?')}: {err.get('reason', 'unknown')}"
            if reason not in b["failure_reasons"]:
                b["failure_reasons"].append(reason)

    return {
        "date": build_date,
        "summary": summary,
        "branches": branches,
        "env_info": env_info,
    }


# ── Markdown 报告生成 ───────────────────────────────────────────────────


def generate_markdown_report(data: dict) -> str:
    """从构建数据合成 Markdown 报告（对齐原版格式）。"""
    summary = data["summary"]
    branches = data["branches"]
    env_info = data["env_info"]
    date = data["date"]

    failed = summary.get("failed", 0)
    errored = summary.get("error", 0)
    skipped = summary.get("skipped", 0)
    total_branches = summary.get("total_branches", 0)
    overall_ok = failed == 0 and errored == 0

    lines = []
    lines.append("# MegatronAdaptor Sync 每日构建报告")
    lines.append("")

    # ═══ 构建信息 ═══
    lines.append("## 构建信息")
    lines.append("")
    lines.append(f"- **构建时间**: {_format_time(env_info.get('fetch_time', ''))}")
    lines.append(f"- **工作目录**: {WORKSPACE}")
    lines.append(f"- **总耗时**: {_fmt_duration(summary.get('total_duration_seconds', 0))}")
    lines.append("")

    # ═══ 分支 Commit 汇总（速查） ═══
    lines.append("## 代码仓 Commit 速查")
    lines.append("")
    lines.append("| 分支 | Megatron-LM | MegatronAdaptor | TransformerEngineNPU |")
    lines.append("|------|-------------|-----------------|----------------------|")
    for b in branches:
        er_b = b.get("env_report") or {}
        if b.get("overall") == "skip":
            # 跳过分支未跑 fetch，无 commit 信息
            lines.append(f"| {b['name']} | - | - | - |")
            continue
        mg = (er_b.get("megatron", {}).get("commit_sha") or "")[:9]
        ma = (er_b.get("adaptor", {}).get("commit_sha") or "")[:9]
        te = (er_b.get("transformer_engine_npu", {}).get("commit_sha") or "")[:9]
        lines.append(f"| {b['name']} | `{mg}` | `{ma}` | `{te}` |")
    lines.append("")

    # ═══ 构建结果 ═══
    lines.append("## 构建结果")
    lines.append("")
    if skipped and skipped == total_branches:
        # 前置预检发现 NPU 被其他训练任务占用 → 全部跳过，非代码失败
        overall_icon = "⏭️"
        overall_text = f"NPU 被其他训练任务占用，{skipped}/{total_branches} 分支全部跳过（未运行验证）"
    else:
        overall_icon = "✅" if overall_ok else "❌"
        overall_text = "全部通过" if overall_ok else f"存在 {failed + errored} 个分支失败"
    if skipped and skipped != total_branches:
        overall_text += f"；{skipped} 分支跳过（NPU 被占用）"
    lines.append(f"**整体状态**: {overall_icon} {overall_text}")
    lines.append("")

    # 分支汇总表
    lines.append("### 分支汇总")
    lines.append("")
    lines.append("| 分支 | 结果 | 测试用例 | 失败原因 | 自动修复 |")
    lines.append("|------|------|----------|----------|----------|")
    for b in branches:
        training = b.get("training_log") or {}
        test_cases = training.get("test_cases", [])
        passed_tests = sum(1 for t in test_cases if t.get("exit_code") == 0)
        test_str = f"{passed_tests}/{len(test_cases)}" if test_cases else "N/A"

        overall = b.get("overall", "unknown")
        if overall == "pass":
            status = "✅ PASS"
            fail_reason = "-"
            auto_fix_str = "-"
        elif overall == "skip":
            status = "⏭️ SKIP"
            fail_reason = "; ".join(b.get("failure_reasons", [])) or "NPU 被其他训练任务占用"
            fail_reason = fail_reason[:120]
            auto_fix_str = "-"
        else:
            status = f"❌ {overall.upper()}"
            fail_reason = "; ".join(b.get("failure_reasons", [])) or "-"
            fail_reason = fail_reason[:120]  # 一句话截断
            af = b.get("auto_fix", {})
            auto_fix_str = "✅ 已修复" if af.get("available") else "❌ 待修复"

        lines.append(f"| {b['name']} | {status} | {test_str} | {fail_reason} | {auto_fix_str} |")
    lines.append("")

    # 修复 PR 链接（单独展示，不塞进表格）
    for b in branches:
        pr_info = b.get("pr_info") or {}
        if pr_info.get("pr_url"):
            lines.append(
                f"🔗 **{b['name']} 修复 PR**: [{pr_info['pr_url']}]({pr_info['pr_url']}) "
                f"（`{pr_info.get('push_branch', '?')}`）"
            )
    # 诊断报告链接（同目录归档的 DIAGNOSIS_AND_FIX，如存在）
    diag_path = OUTPUT_DIR / date_to_archive_path(date) / "DIAGNOSIS_AND_FIX.md"
    if diag_path.exists():
        lines.append(
            f"📄 **诊断报告**: [DIAGNOSIS_AND_FIX_{date}.md](DIAGNOSIS_AND_FIX_{date}.md)（根因分析 / 修复方案 / 独立重验证据）"
        )
    lines.append("")

    # ═══ 各分支详情 ═══
    lines.append("## 各分支详情")
    lines.append("")

    for b in branches:
        overall = b.get("overall", "unknown")
        icon = "✅" if overall == "pass" else ("⏭️" if overall == "skip" else "❌")
        lines.append(f"### {icon} {b['name']}（{b['conda_env']}）")
        lines.append("")

        # 代码仓信息（每分支独立列出）
        er_b = b.get("env_report") or {}
        if er_b:
            lines.append("| 代码仓 | 分支 | Commit | 描述 |")
            lines.append("|--------|------|--------|------|")
            for repo_name, repo_key in [
                ("Megatron-LM", "megatron"),
                ("MegatronAdaptor", "adaptor"),
                ("TransformerEngineNPU", "transformer_engine_npu"),
            ]:
                repo = er_b.get(repo_key, {})
                br = repo.get("branch", "?")
                sha = (repo.get("commit_sha") or "")[:9]
                desc = repo.get("commit_subject", "")
                lines.append(f"| {repo_name} | `{br}` | `{sha}` | {desc} |")
            lines.append("")

        # 各阶段状态
        stages = b.get("stages", {})
        if b.get("overall") == "skip":
            lines.append("**阶段状态**: ENVCHECK ⛔ busy → 跳过（NPU 被其他训练任务占用，未进入 fetch/setup/launch）")
            lines.append("")
        else:
            lines.append(
                f"**阶段状态**: ENVCHECK {_s(stages.get('envcheck'))} → FETCH {_s(stages.get('fetch'))} → "
                f"SETUP {_s(stages.get('setup'))} → SIGCHECK {_s(stages.get('sigcheck'))} → "
                f"LAUNCH {_s(stages.get('launch'))} → VERIFY {_s(stages.get('verify'))}"
            )
            lines.append("")

        # 测试用例
        training = b.get("training_log") or {}
        test_cases = training.get("test_cases", [])
        if b.get("overall") == "skip":
            # 前置预检跳过：未跑测试，展示 ENV_BUSY 原因，不渲染空测试表
            lines.append("**测试用例**: N/A（NPU 被其他训练任务占用，未运行）")
            skip_errs = b.get("errors", [])
            if skip_errs:
                lines.append("")
                lines.append("**跳过原因**:")
                for e in skip_errs:
                    lines.append(f"- ⏭️ {str(e)[:200]}")
            lines.append("")
            continue
        passed_tests = sum(1 for t in test_cases if t.get("exit_code") == 0)
        failed_tests = [t for t in test_cases if t.get("exit_code") != 0]
        lines.append(f"**测试用例**: {passed_tests}/{len(test_cases)} 通过")
        lines.append("")

        if test_cases:
            lines.append("| 用例 | 结果 | 耗时 |")
            lines.append("|------|------|------|")
            for tc in test_cases:
                exit_code = tc.get("exit_code", -1)
                if exit_code == 0:
                    result = "✅ PASS"
                else:
                    result = f"❌ FAIL (exit={exit_code})"
                dur_s = tc.get("duration_seconds", 0)
                dur_str = f"{dur_s:.0f}s" if dur_s else "N/A"
                lines.append(f"| {tc.get('test_id', '?')} | {result} | {dur_str} |")
            lines.append("")

        # 失败原因（一句话）
        if failed_tests:
            lines.append("**失败原因**:")
            lines.append("")
            reasons = b.get("failure_reasons", [])
            if reasons:
                for r in reasons:
                    lines.append(f"- ❌ {r}")
            else:
                for ft in failed_tests:
                    lines.append(f"- ❌ {ft.get('test_id', '?')}: exit_code={ft.get('exit_code')}")
            lines.append("")

        # 自动修复信息
        af = b.get("auto_fix", {})
        if af.get("available"):
            lines.append("### 🔧 自动定位修复")
            lines.append("")

            bisect = af.get("bisect") or {}
            if bisect.get("result"):
                r = bisect["result"]
                lines.append(
                    f"- **定位到的 Breaking Commit**: `{r.get('breaking_commit', '?')}` — {r.get('breaking_commit_subject', '?')}"
                )
                lines.append(
                    f"- **Bisect 方法**: {bisect.get('bisect_range', {}).get('total_commits', '?')} 个 commit，{bisect.get('bisect_range', {}).get('bisect_steps', '?')} 步二分"
                )
                if bisect.get("false_positive_note"):
                    lines.append(f"- **⚠️ 去误中**: {bisect['false_positive_note']}")
                lines.append("")

            root_cause = af.get("root_cause") or {}
            if root_cause.get("root_cause"):
                rc = root_cause["root_cause"]
                lines.append(f"- **根因类型**: {rc.get('type', '?')}")
                lines.append(f"- **涉及文件**: `{rc.get('file', '?')}` — `{rc.get('function', '?')}`")
                lines.append(f"- **问题描述**: {rc.get('bug', rc.get('description', '?'))}")
                lines.append("")

            fix_plan = af.get("fix_plan") or {}
            if fix_plan.get("strategy"):
                lines.append(f"- **修复策略**: {fix_plan['strategy']}")
                lines.append(
                    f"- **为何不直接改条件**: {fix_plan.get('why_not_change_condition', {}).get('reason', 'N/A')}"
                )
                for k, v in fix_plan.items():
                    if k.startswith("fix_"):
                        lines.append(
                            f"- **{k}**: {v.get('approach', v.get('description', str(v)[:200]))} [`{v.get('file', '?')}`]"
                        )
                lines.append("")

        # 修复 PR 链接
        pr_info = b.get("pr_info") or {}
        if pr_info.get("pr_url"):
            lines.append(
                f"🔗 **修复 PR**: [{pr_info['pr_url']}]({pr_info['pr_url']}) — `{pr_info.get('push_branch', '?')}`"
            )
            lines.append("")

        lines.append("---")
        lines.append("")

    # ═══ 环境信息 ═══
    if env_info:
        lines.append("## 环境信息")
        lines.append("")
        lines.append(f"- **Python**: {env_info.get('python', 'N/A')}")
        lines.append(f"- **Torch**: {env_info.get('torch', 'N/A')}")
        lines.append(f"- **TE-NPU**: {env_info.get('te_version', 'N/A')}")
        npu = env_info.get("npu", {})
        if npu:
            lines.append(f"- **NPU**: {npu.get('device_name', 'N/A')} × {npu.get('device_count', 'N/A')}")
        lines.append("")

    return "\n".join(lines)


def _fmt_duration(seconds: float) -> str:
    if not seconds:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def _s(status: str | None) -> str:
    """阶段状态图标。"""
    if not status:
        return "❓"
    return "✅" if status in ("ok", "pass") else "❌"


# ── Git 操作 ─────────────────────────────────────────────────────────────


def clone_or_update_repo():
    """克隆或更新 CI 代码仓。"""
    clone_dir = Path(CI_REPO_CONFIG["clone_dir"])

    if (clone_dir / ".git").exists():
        print("\n📦 更新现有 CI 代码仓...")
        os.chdir(clone_dir)

        result = subprocess.run(
            [shutil.which("git") or "git", "rev-parse", "--verify", f"origin/{CI_REPO_CONFIG['branch']}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                # 先清洗可能残留的 tokenized URL（防 token 落盘 .git/config），
                # 再联网 —— 认证经 credential helper 提供，不进 argv/env。
                ensure_clean_url()
                subprocess.run(
                    [shutil.which("git") or "git", *_credential_helper(), "fetch", "origin", CI_REPO_CONFIG["branch"]],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [shutil.which("git") or "git", "reset", "--hard", f"origin/{CI_REPO_CONFIG['branch']}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("✅ CI 代码仓更新成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ 代码仓更新失败：{e.stderr}")
                sys.exit(1)
        else:
            print("✅ 使用本地 CI 代码仓（尚未关联远程分支）")
    else:
        print("\n📦 克隆 CI 代码仓...")
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        result = subprocess.run(
            [
                shutil.which("git") or "git",
                *_credential_helper(),
                "clone",
                "-b",
                CI_REPO_CONFIG["branch"],
                _clean_url(),
                str(clone_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            os.chdir(clone_dir)
            print("✅ CI 代码仓克隆成功")
        else:
            print("   分支不存在，尝试克隆空仓库...")
            result2 = subprocess.run(
                [shutil.which("git") or "git", *_credential_helper(), "clone", _clean_url(), str(clone_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result2.returncode == 0:
                os.chdir(clone_dir)
                print("✅ 空 CI 代码仓克隆成功")
            else:
                print(f"❌ 代码仓克隆失败：{result2.stderr}")
                sys.exit(1)


def save_report_and_push(markdown_content: str, build_date: str) -> str | None:
    """保存报告并推送到代码仓。同时上传 DIAGNOSIS_AND_FIX.md（如存在）。"""
    clone_dir = Path(CI_REPO_CONFIG["clone_dir"])
    os.chdir(clone_dir)

    archive_subpath = date_to_archive_path(build_date)
    target_dir = clone_dir / "sync_build_results" / archive_subpath
    target_dir.mkdir(parents=True, exist_ok=True)

    report_filename = f"SYNC_SUMMARY_{build_date}.md"
    report_path = target_dir / report_filename
    report_path.write_text(markdown_content, encoding='utf-8')
    print(f"✅ 报告已保存：{report_path}")

    # 如果存在 DIAGNOSIS_AND_FIX.md，一并上传
    diag_fix_src = OUTPUT_DIR / date_to_archive_path(build_date) / "DIAGNOSIS_AND_FIX.md"
    if diag_fix_src.exists():
        diag_fix_dst = target_dir / f"DIAGNOSIS_AND_FIX_{build_date}.md"
        shutil.copy2(diag_fix_src, diag_fix_dst)
        print(f"✅ 诊断修复报告已复制：{diag_fix_dst}")
    else:
        print("ℹ️  未找到 DIAGNOSIS_AND_FIX.md（本次构建无需修复或尚未生成）")

    # 一并上传 nightly_case_library.jsonl（飞轮案例库：构建结果历史）。
    # 代码仓不再跟踪该文件（见 MA-agent .gitignore），这里推到 CI 仓稳定路径做
    # 备份与历史（覆盖写，git 历史即版本）。本地副本仍被 ① fetch / ⑥ diag 运行时读取。
    case_library_rel = "sync_build_results/nightly_case_library.jsonl"
    case_library_src = Path(__file__).resolve().parent / "nightly_case_library.jsonl"
    if case_library_src.exists():
        case_library_dst = clone_dir / case_library_rel
        case_library_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(case_library_src, case_library_dst)
        print(f"✅ 案例库已复制：{case_library_dst}")
    else:
        print("ℹ️  未找到 nightly_case_library.jsonl（跳过案例库推送）")

    try:
        git_add_paths = [str(target_dir.relative_to(clone_dir))]
        if (clone_dir / case_library_rel).exists():
            git_add_paths.append(case_library_rel)
        subprocess.run([shutil.which("git") or "git", "add"] + git_add_paths, check=True)

        status = subprocess.run(
            [shutil.which("git") or "git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        if not status.stdout.strip():
            print("⚠️  没有变更需要提交（报告内容与远程一致）")
            result = subprocess.run(
                [shutil.which("git") or "git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            )
            return result.stdout.strip()

        commit_msg = f"CI: Add sync build report {build_date}"
        subprocess.run([shutil.which("git") or "git", "commit", "-m", commit_msg], check=True)

        push_result = subprocess.run(
            [shutil.which("git") or "git", *_credential_helper(), "push", "-u", "origin", CI_REPO_CONFIG["branch"]],
            capture_output=True,
            text=True,
            check=False,
        )
        if push_result.returncode != 0:
            print(f"❌ 推送失败：{push_result.stderr}")
            return None
        print(f"   推送输出：{push_result.stdout.strip()}")

        result = subprocess.run(
            [shutil.which("git") or "git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        commit_hash = result.stdout.strip()
        print(f"✅ 推送成功！Commit: {commit_hash}")
        return commit_hash

    except subprocess.CalledProcessError as e:
        print(f"❌ Git 操作失败：{e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"   {e.stderr}")
        return None


# ── 主函数 ───────────────────────────────────────────────────────────────


def main(build_date: str | None = None):
    """主函数。build_date 为 None 时自动获取最新日期。"""
    print("=" * 60)
    print("MegatronAdaptor Sync CI 结果推送工具")
    print("=" * 60)

    try:
        if build_date is None:
            build_date = get_latest_build_date()
            if build_date is None:
                print("❌ output/ 下未找到构建结果目录")
                return {"success": False, "error": "未找到构建结果"}

        print(f"\n📅 构建日期：{build_date}")

        # 读取构建结果
        data = read_build_results(build_date)
        if data is None:
            return {"success": False, "error": "读取构建结果失败"}

        summary = data["summary"]
        print(f"📊 分支数：{summary.get('total_branches', '?')}")
        print(f"   通过：{summary.get('passed', '?')}")
        print(f"   失败：{summary.get('failed', '?')}")
        skipped = summary.get("skipped", 0)
        if skipped:
            print(f"   跳过：{skipped}（NPU 被其他训练任务占用）")

        # 生成 Markdown 报告
        print("\n📝 生成 Markdown 报告...")
        markdown = generate_markdown_report(data)
        print(f"   报告长度：{len(markdown)} 字符")

        # 克隆/更新代码仓
        clone_or_update_repo()

        # 保存并推送
        commit_hash = save_report_and_push(markdown, build_date)

        print("\n" + "=" * 60)
        print("📊 推送结果汇总")
        print("=" * 60)
        print(f"📅 日期：{build_date}")
        if commit_hash:
            print(f"🔗 Commit：{commit_hash}")
        print(f"🌐 代码仓：{CI_REPO_CONFIG['url']}")
        print(f"📁 路径：sync_build_results/{date_to_archive_path(build_date)}/")
        print("=" * 60)

        return {
            "success": commit_hash is not None,
            "commit_hash": commit_hash,
            "date": build_date,
            "branch_count": summary.get("total_branches", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "report_file": f"SYNC_SUMMARY_{build_date}.md",
        }

    except Exception as e:
        print(f"\n❌ 执行错误：{e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    build_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = main(build_date)
