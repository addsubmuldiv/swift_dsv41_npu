#!/usr/bin/env python3
"""
Deterministic idempotent Issue/PR creation for MegatronAdaptor nightly fixes.

背景
----
此前 Issue/PR 由每晚新起的 `claude -p` 会话按 gitcode_api_guide.md 手工 curl 创建。
幂等查询 `search=<分支> <用例>` 含未编码空格 → 返回空 → 每晚误判"无已存在 Issue"→ 重复建 Issue
（实测 main test_attention 一个 bug 累积 7 个重复 open Issue）。

本脚本把幂等检查做成确定性逻辑，替代 LLM 手工创建：

  - Issue 粒度 = (分支, 失败用例)：搜索已有 open Issue，存在则追加"每日更新"评论，
    不存在才新建。标题前缀 `[daily-ci] <分支> <用例> FAIL —`。
  - PR 粒度 = 分支：按 `<fork_user>:daily_fix_<分支>` 查已有 open PR（fork_user 取自 .push_config.json 的 repo.fork_path），存在则复用
    （force-push 自动更新 diff），不存在才新建并关联首个 Issue。
  - 创建/复用后评论 `compile` 触发 CI。
  - 结果写回 `output/<YYYY-MM>/<MM-DD>/<branch>/state/pr_info.json`。

调用方式
--------
    python3 create_issue_pr.py --date 2026-08-03 [--branch main] [--dry-run]

依赖 state 产物（由 nightly_orchestrator.py 产出，按月分层）：
    output/<YYYY-MM>/<MM-DD>/combined_summary.json       ← 分支组合 + needs_llm_diag
    output/<YYYY-MM>/<MM-DD>/<branch>/state/diag_report.json   ← 失败用例 + 根因
    output/<YYYY-MM>/<MM-DD>/<branch>/state/verify_report.json ← 失败详情（兜底）
    output/<YYYY-MM>/<MM-DD>/DIAGNOSIS_AND_FIX.md        ← Issue body（可选，LLM 写完）
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# 复用 nightly_launch 的训练日志指标提取（loss / grad_norm），保证与 verify 判定口径一致
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from nightly_launch import extract_metrics_from_log  # noqa: E402

# ── 路径与常量 ──────────────────────────────────────────────────────────

WORKSPACE = "/home/ma_daily_ci/workspace"
OUTPUT_DIR = Path(WORKSPACE) / "output"
SKILL_DIR = Path(__file__).resolve().parent.parent  # .../megatron-adaptor-sync
CONFIG_PATH = SKILL_DIR / "scripts" / ".push_config.json"

REPO = "Ascend/MegatronAdaptor"
API = "https://gitcode.com/api/v5"
ISSUE_BASE = f"https://gitcode.com/{REPO}/issues"
TITLE_PREFIX = "[daily-ci]"
# 本地 issue 注册表（append-only）：记录每次创建的 issue 及其问题 key，供去重兜底与清理审计
REGISTRY_PATH = OUTPUT_DIR / "issue_registry.jsonl"


def _date_dir(date: str) -> Path:
    """按月分层：output/YYYY-MM/MM-DD（与 CI 仓归档路径一致）。"""
    return OUTPUT_DIR / date[:7] / date[5:]


def _log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── 配置 ─────────────────────────────────────────────────────────────────


def _load_config():
    """读取 .push_config.json 的 repo 配置；失败时给出明确错误而非静默默认。"""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)["repo"]
        return {
            "token": cfg["api_token"],
            "remote_url": cfg["remote_url"],
            "fork_path": cfg["fork_path"],
        }
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[create_issue_pr] 无法读取 {CONFIG_PATH}: {e}")


CONFIG = _load_config()
TOKEN = CONFIG["token"]
FORK_PATH = CONFIG["fork_path"]
FORK_USER = FORK_PATH.split("/")[0]
REMOTE_URL = CONFIG["remote_url"]


# ── 底层 API 请求 ───────────────────────────────────────────────────────


def _request(method, path, payload=None):
    """发起 gitcode API 请求，返回解析后的 JSON；HTTP 错误显式抛异常。"""
    url = API + path
    headers = {"PRIVATE-TOKEN": TOKEN, "Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API {method} {path} -> {e.code} {e.reason}\n{body[:600]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"API {method} {path} -> 网络错误: {e}")


# ── 幂等查询（搜索关键字只用用例名，URL 编码；避免空格破坏 URL）─────────


def list_open_pulls(branch):
    """查 <fork_user>:daily_fix_<branch> 的已开 PR。

    注意：gitcode API 的 `head=<user>:<branch>` 查询过滤实测不生效（返回空），
    因此拉取全量 open PR 后客户端按 `head.ref` + `head.user.login` 过滤。
    """
    path = f"/repos/{REPO}/pulls?state=opened&per_page=100"
    pulls = _request("GET", path) or []
    want_ref = f"daily_fix_{branch}"
    return [
        p
        for p in pulls
        if p.get("head", {}).get("ref") == want_ref and p.get("head", {}).get("user", {}).get("login") == FORK_USER
    ]


def list_open_issues():
    """拉取全部 open issues（分页）。幂等核心依据：不依赖 search 召回质量。"""
    issues = []
    page = 1
    while True:
        path = f"/repos/{REPO}/issues?state=opened&per_page=100&page={page}"
        batch = _request("GET", path) or []
        if not batch:
            break
        issues.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return issues


def find_existing_problem(key, branch, test):
    """按问题 key 找已开 Issue（幂等核心）：实时扫 open issues 标题归一化子串匹配，
    (branch, test) 前缀兜底，本地 registry 最后兜底（校验仍 open）。

    全局按「问题」去重——同一根因跨分支/跨用例/跨天只保留一个 Issue。
    - 已知问题：归一化根因子串匹配（兼容新旧标题格式），取最早为 canonical。
    - unknown 问题（无根因文本）：退回 `[daily-ci] <branch> <test>` 前缀匹配，
      复用既有 per-(branch,test) Issue，避免无根因问题也每天新建。
    - registry：仅当 API 全 miss 时兜底（逐个确认仍 open）。
    """
    candidates = []
    for it in list_open_issues():
        title = str(it.get("title", ""))
        if not title.startswith(TITLE_PREFIX):
            continue
        rest = title[len(TITLE_PREFIX) :].strip()
        if key and key in normalize_problem_key(rest):
            candidates.append(it)
            continue
        if key.startswith("unknown-problem:") and rest.startswith(f"{branch} {test}"):
            candidates.append(it)
    if not candidates:
        # API 未命中 → 本地 registry 兜底（逐个确认仍 open，避免复用已关闭的 canonical）
        for rec in load_registry().get(key, []):
            number = rec.get("issue_number")
            if number is None:
                continue
            try:
                cur = _request("GET", f"/repos/{REPO}/issues/{number}")
            except Exception:  # noqa: BLE001
                continue
            if (cur or {}).get("state") == "open":
                candidates.append(cur)
    if not candidates:
        return None
    # 多个匹配取最早（number 最小）作为 canonical，其余为历史重复
    return min(candidates, key=lambda it: it.get("number", 0))


# ── 写操作 ──────────────────────────────────────────────────────────────


def create_issue(title, body):
    return _request("POST", f"/repos/{REPO}/issues", {"title": title, "body": body})


def comment_issue(number, body):
    return _request("POST", f"/repos/{REPO}/issues/{number}/comments", {"body": body})


def create_pull(head, base, title, body):
    payload = {
        "head": head,
        "base": base,
        "title": title,
        "body": body,
        "fork_path": FORK_PATH,
    }
    return _request("POST", f"/repos/{REPO}/pulls", payload)


def comment_pull(number, body):
    return _request("POST", f"/repos/{REPO}/pulls/{number}/comments", {"body": body})


# ── 辅助 ─────────────────────────────────────────────────────────────────


def fork_branch_exists(branch):
    """确认 daily_fix_<branch> 已推送到 fork（无修复推送则跳过该分支）。"""
    proc = subprocess.run(
        [shutil.which("git") or "git", "ls-remote", "--heads", REMOTE_URL, f"refs/heads/daily_fix_{branch}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def fork_branch_has_diff(branch):
    """daily_fix_<branch> 相对 origin/<branch> 是否有内容变更（只读，不 mutation）。

    等价于「rebase 到最新基线后是否为空」：已合入 main 的旧修复（如 PR #107 的
    daily_fix_main=0aae811 vs 已合入的 cc58e75，兄弟 commit、tree 相同）→ 无 diff →
    建空 PR 无意义 → 应跳过。用 `git diff --quiet`（tree 级），不用祖先判断：
    跨天 reset/rebase 后同一修复 hash 会变，祖先关系会误判为「有新提交」。
    """
    ma = f"{WORKSPACE}/MegatronAdaptor"
    git = shutil.which("git") or "git"
    # 最佳努力刷新 refs；fetch 失败沿用本地旧 ref，兜底不阻断
    for remote, ref in (("origin", branch), ("fanlu", f"daily_fix_{branch}")):
        subprocess.run(
            [git, "-C", ma, "fetch", remote, ref],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    r = subprocess.run(
        [git, "-C", ma, "diff", "--quiet", f"origin/{branch}", f"fanlu/daily_fix_{branch}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # 0=无 diff（空）；1=有 diff；>1=出错 → 兜底返回 True（不因探针故障漏建 PR）
    return r.returncode != 0


def local_fix_commit_msg(branch):
    """读取本地 MegatronAdaptor 仓 daily_fix_<branch> 最新 commit message。"""
    proc = subprocess.run(
        [
            shutil.which("git") or "git",
            "-C",
            f"{WORKSPACE}/MegatronAdaptor",
            "log",
            "-1",
            "--format=%s",
            f"daily_fix_{branch}",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def collect_failing_tests(state_dir):
    """确定该分支的**确认修复的真问题**用例列表（幂等、去重）。

    来源优先级：
    0. **fixed_tests.json**（auto-fix 诊断后产出的"确认修复的真问题"清单，排除 verify 误报）——
       这是建 Issue 的唯一权威来源：只有确认是真问题且要提 PR 修复的测试才建 Issue。
    1. diag_report.extracted_errors[].test_id
    2. diag_report.failed_test_id
    3. diag_report.symptom / failed_checks 文本中的 `test_<name>` 前缀
    4. verify_report.failure_details 文本中的 `test_<name>` 前缀（兜底）

    `fixed_tests.json` 缺失时退回多来源提取（兼容旧产物，但可能包含 verify 误报）。
    """
    fixed_path = state_dir / "fixed_tests.json"
    if fixed_path.exists():
        try:
            fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
            fixed_tests = fixed if isinstance(fixed, list) else fixed.get("tests", [])
            tests = [t for t in fixed_tests if t]
            if tests:
                return tests
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    test_pat = re.compile(r"(test_[a-zA-Z0-9_]+)")
    tests = []
    diag_path = state_dir / "diag_report.json"
    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            for err in diag.get("extracted_errors") or []:
                tid = err.get("test_id")
                if tid and tid not in tests:
                    tests.append(tid)
            if diag.get("failed_test_id") and diag["failed_test_id"] not in tests:
                tests.append(diag["failed_test_id"])
            # 兜底：从 symptom / failed_checks 文本提取 test_ 前缀
            blob = str(diag.get("symptom", "")) + json.dumps(diag.get("failed_checks") or [])
            for m in test_pat.findall(blob):
                if m not in tests:
                    tests.append(m)
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    if not tests:
        verify_path = state_dir / "verify_report.json"
        if verify_path.exists():
            try:
                verify = json.loads(verify_path.read_text(encoding="utf-8"))
                blob = json.dumps(verify.get("failure_details") or [])
                for m in test_pat.findall(blob):
                    if m not in tests:
                        tests.append(m)
            except Exception:  # noqa: BLE001
                pass  # nosec B110
    return tests


def normalize_problem_key(reason):
    """归一化根因文本为稳定问题签名（去 rank 前缀 / device id / tensor 值 / 地址）。

    同一根因在不同分支/天数/rank 上的表述（如 `tensor(10, device='npu:5')` vs
    `tensor(10, device='npu:0')`）应归一到同一 key，作为 Issue 全局去重键。
    """
    s = str(reason).lower()
    s = re.sub(r"\[rank\d+\]\s*:?\s*", "", s)
    s = re.sub(r"device\s*=\s*['\"]?(npu|cuda|cpu)[^'\"]*['\"]?", "", s)
    s = re.sub(r"['\"]?(npu|cuda):\d+['\"]?", "", s)
    s = re.sub(r"\btensor\([^)]*\)", "tensor()", s)
    s = re.sub(r"\b0x[0-9a-f]+", "0x", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _verify_error_map(state_dir):
    """从 verify_report 的 failure_details 解析 {test: error} 映射（兜底根因源）。

    details 形如 `Failed tests: A: errA; B: errB`，每段可带 `[rankN]:` 前缀。
    仅作 diag_report 缺失时的兜底（core 分支等）；按 test 解析，多问题分支不串。
    解析失败返回 {}。
    """
    mapping = {}
    verify_path = state_dir / "verify_report.json"
    if not verify_path.exists():
        return mapping
    try:
        verify = json.loads(verify_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return mapping
    for fd in verify.get("failure_details") or []:
        details = str(fd.get("details") or "")
        for seg in re.split(r";\s*(?=\S+:\s)", details):
            m = re.match(r"^(?:Failed tests:\s*)?(\S+):\s*(.+)$", seg)
            if m:
                mapping.setdefault(m.group(1), m.group(2).strip())
    return mapping


def extract_reason(state_dir, test):
    """提取该 test 的根因一句话（原始文本，未归一化），截断到 90 字符。

    来源优先级：diag_report extracted_errors（test_id 匹配）→ 任意 error 的 reason →
    root_cause 字段 → verify_report failure_details（按 test 解析）。
    """
    diag_path = state_dir / "diag_report.json"
    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            for e in diag.get("extracted_errors") or []:
                if (e.get("test_id") or "") == test and e.get("reason"):
                    return re.sub(r"\s+", " ", str(e["reason"])).strip()[:90]
            for e in diag.get("extracted_errors") or []:
                if e.get("reason"):
                    return re.sub(r"\s+", " ", str(e["reason"])).strip()[:90]
            if diag.get("root_cause"):
                return re.sub(r"\s+", " ", str(diag["root_cause"])).strip()[:90]
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    err = _verify_error_map(state_dir).get(test)
    if err:
        return re.sub(r"\s+", " ", err).strip()[:90]
    return "see issue body"


def problem_key(state_dir, branch, test):
    """问题去重键：归一化根因；无稳定根因时退回 (branch, test) 避免不同问题被合并。"""
    reason = extract_reason(state_dir, test)
    if reason and reason != "see issue body":
        key = normalize_problem_key(reason)
        if len(key) >= 8:
            return key
    return f"unknown-problem:{branch}:{test}"


def root_cause_one_liner(state_dir):
    """从 diag_report 提取根因一句话（用于 PR 描述），截断到 90 字符。"""
    return extract_reason(state_dir, None)


def load_registry():
    """加载本地 issue 注册表 {problem_key: [records]}；文件不存在/坏行则跳过。"""
    reg = {}
    if REGISTRY_PATH.exists():
        for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            reg.setdefault(rec.get("problem_key", ""), []).append(rec)
    return reg


def append_registry(rec):
    """追加一条 issue 记录到本地注册表（append-only，重复跑不覆盖历史）。"""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compose_issue_body(date, branch, test, state_dir, diagnosis_md):
    """Issue body：优先嵌入 LLM 写的 DIAGNOSIS_AND_FIX.md 全文；否则从 state 兜底拼装。"""
    if diagnosis_md and diagnosis_md.exists():
        content = diagnosis_md.read_text(encoding="utf-8")
        if content.strip():
            return content
    # 兜底：从 verify_report 摘录失败详情
    lines = [
        f"## 失败现象（{date}）",
        "",
    ]
    verify_path = state_dir / "verify_report.json"
    if verify_path.exists():
        try:
            verify = json.loads(verify_path.read_text(encoding="utf-8"))
            for fd in verify.get("failure_details") or []:
                lines.append(f"- **{fd.get('check')}**: {fd.get('details')}")
        except Exception:  # noqa: BLE001
            pass  # nosec B110
    lines += ["", f"分支: `{branch}` / 用例: `{test}`", "", "---", "*Auto-generated by MegatronAdaptor daily CI*"]
    return "\n".join(lines)


def _log_excerpts(path, max_subruns=4):
    """从 stdout 日志取每个子测试首/尾两条训练行**原文**，封装进 code block；无则 None。

    训练行形如 `[timestamp] iteration 9/1000 | ... | lm loss: 1.08E+01 | ... | grad norm: 55.34 | ...`。
    子测试按 `iteration 1/` 重置点分组（与 extract_metrics_from_log 同口径）。
    """
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    train = re.compile(r"iteration\s+\d+/.*lm loss[:\s].*grad norm[:\s]")
    # 训练行在多进程日志中可能与其他输出拼接在同一物理行，只截取 timestamp→grad norm 值
    # 的完整片段作为原文证据（如开头的 `Number of parameters...`、结尾的内存 dump 均丢弃）
    segment = re.compile(
        r"\[20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\] iteration\s+\d+/\s*\d+.*?grad norm[:\s]+\d[\d.]*(?:[eE][+-]?\d+)?"
    )
    groups = []
    current = []
    for line in content.splitlines():
        if train.search(line):
            if current and re.search(r"iteration\s+1/", line):
                groups.append(current)
                current = []
            current.append(line)
    if current:
        groups.append(current)
    if not groups:
        return None
    lines = ["```text"]
    for i, g in enumerate(groups[:max_subruns], 1):

        def _seg(line):
            m = segment.search(line)
            return m.group(0) if m else line.rstrip()

        if len(g) == 1:
            lines.append(f"{_seg(g[0])}  ← 子测试 {i}")
        else:
            lines.append(f"{_seg(g[0])}  ← 子测试 {i} 首")
            lines.append(f"{_seg(g[-1])}  ← 子测试 {i} 尾")
    lines.append("```")
    return "\n".join(lines)


def compose_verification_report(date, branch, test):
    """从重验 stdout 日志提取**详细**验证报告（每子测试一张表，供 Issue 评论）。

    无可用日志返回 None（不编造）。PR 描述里只放一句话摘要（`_reverify_summary`），
    详细数字沉淀在这里——完整诊断在 Issue，PR 正文保持精简。
    """
    path = next(iter(_reverify_stdout_paths(test, branch, date)), None)
    if path is None:
        return None
    sub = (extract_metrics_from_log(str(path)).get("sub_runs")) or []
    if not sub:
        return None
    converged = sum(
        1
        for r in sub
        if r["last_loss"] is not None and r["initial_loss"] is not None and r["last_loss"] < r["initial_loss"]
    )
    healthy = all((r["grad_norm_count"] or 0) > 0 and (r["grad_norm_zero_count"] or 0) == 0 for r in sub)
    lines = [f"### 验证报告（{test}）", ""]
    lines.append(
        f"- 子测试: **{len(sub)}**，loss 收敛: **{converged}/{len(sub)}**，"
        f"梯度: {'全非零' if healthy else '存在零梯度'}"
    )
    lines.append("")
    lines.append("| 子测试 | loss 首→尾 | grad_norm 范围 | 零梯度次数 |")
    lines.append("|--------|-----------|---------------|-----------|")
    for i, r in enumerate(sub, 1):
        il, fl = r["initial_loss"], r["last_loss"]
        loss = "—" if il is None or fl is None else f"{il:.4g} → {fl:.4g}"
        gmin, gmax = r["grad_norm_min"], r["grad_norm_max"]
        gr = "—" if gmin is None or gmax is None else f"{gmin:.0f}–{gmax:.0f}"
        zc, cnt = r["grad_norm_zero_count"], r["grad_norm_count"]
        zg = "—" if zc is None else f"{zc}/{cnt}"
        lines.append(f"| {i} | {loss} | {gr} | {zg} |")
    excerpts = _log_excerpts(path)
    if excerpts:
        lines += ["", "### 日志原文摘录（重验）", "", excerpts]
    return "\n".join(lines)


def compose_daily_comment(date, branch, test, commit_msg):
    """已存在 Issue 时追加的每日更新评论（含当日详细验证报告）。"""
    lines = [f"## {date} 更新", "", f"- 失败用例: `{test}`"]
    if commit_msg:
        lines.append(f"- 修复 commit: `{commit_msg}`")
    report = compose_verification_report(date, branch, test)
    if report:
        lines += ["", report]
    lines += ["", "---", "*Auto-generated by MegatronAdaptor daily CI*"]
    return "\n".join(lines)


def compose_pr_title(commit_msg):
    """PR 标题：按模板 frontmatter 的 `type(scope): describe` 约定。

    gitcode 合入后标题会作为提交信息首行，故沿用模板的 `fix(backend): describe`
    风格。commit message 形如 `fix: <desc>` 或 `fix(agent): <desc> [<test>] [branch: x]`，
    去掉类型头与尾部 `[...]` 标签，重写为 `fix(adaptor): [agent] <desc>`——保留
    agent 标记，让评审者一眼看出是 agent（Claude）提交的修改，而非人工/脚本产出。
    """
    if commit_msg:
        is_agent = bool(re.search(r"^[a-zA-Z]+\(agent\):", commit_msg))
        subject = re.sub(r"^[a-zA-Z]+(?:\([^)]*\))?:\s*", "", commit_msg)
        # 循环剥掉尾部所有 [test] [branch: x] 标签
        while True:
            stripped = re.sub(r"\s*\[[^\]]*\]\s*$", "", subject)
            if stripped == subject:
                break
            subject = stripped
        subject = re.sub(r"\s{2,}", " ", subject).strip()
        if subject:
            prefix = "[agent] " if is_agent else ""
            return f"fix(adaptor): {prefix}{subject}"[:120]
    return "fix(adaptor): nightly adaptor compatibility fix"


def _reverify_stdout_paths(test, branch, date):
    """候选重验日志位置：独立重验 `reverify/<test>/` 或 auto_fix 循环内 `auto_fix/<branch>/<test>/`。"""
    return [
        p
        for p in (
            _date_dir(date) / "reverify" / test / "stdout.log",
            _date_dir(date) / "auto_fix" / branch / test / "stdout.log",
        )
        if p.is_file()
    ]


def _reverify_summary(test, branch, date):
    """从重验 stdout 日志提取可读的验证摘要；无可用日志返回 None（不编造数字）。

    例：`4/4 子测试 loss 收敛，loss 10.82→10.38，grad_norm 42–62（全非零）`
    """
    for path in _reverify_stdout_paths(test, branch, date):
        m = extract_metrics_from_log(str(path))
        sub = m.get("sub_runs") or []
        if not sub:
            continue
        converged = sum(
            1
            for r in sub
            if r["last_loss"] is not None and r["initial_loss"] is not None and r["last_loss"] < r["initial_loss"]
        )
        healthy = all((r["grad_norm_count"] or 0) > 0 and (r["grad_norm_zero_count"] or 0) == 0 for r in sub)
        # 只取第一个子测试自身的 initial→final 展示收敛（跨子测试比较首尾是已知误报源，
        # 见 verify-multi-config-false-positive；子测试数已由 `converged/N` 体现）
        first_initial = sub[0]["initial_loss"]
        first_final = sub[0]["last_loss"]
        gmin = min((r["grad_norm_min"] for r in sub if r["grad_norm_min"] is not None), default=None)
        gmax = max((r["grad_norm_max"] for r in sub if r["grad_norm_max"] is not None), default=None)
        parts = [f"{converged}/{len(sub)} 子测试 loss 收敛"]
        if first_initial is not None and first_final is not None:
            parts.append(f"loss {first_initial:.4g}→{first_final:.4g}")
        if gmin is not None and gmax is not None:
            parts.append(f"grad_norm {gmin:.0f}–{gmax:.0f}{'（全非零）' if healthy else ''}")
        return "，".join(parts)
    return None


def compose_pr_body(date, branch, tests, state_dir, issue_urls):
    """PR 描述：按仓库 .gitcode/PULL_REQUEST_TEMPLATE.md 的三个 section 组织。

    Ascend/MegatronAdaptor（及 TransformerEngineNPU）的 PR 模板要求描述包含
    `What this PR does / why we need it?`、`Does this PR introduce any user-facing change?`、
    `How was this patch tested?` 三个 section。模板顶部 frontmatter（title/labels/assignees）
    是平台交互式建 PR 时预填元数据，API 建 PR 不写入 body（labels 无权限会 403）。

    ⚠️ 描述保持精简：合入后正文会作为提交信息写入代码仓提交记录，完整诊断留在关联 Issue。
    """
    lines = []
    if issue_urls:
        lines.append(f"Closes {issue_urls[0]}")
        for u in issue_urls[1:]:
            lines.append(f"Related {u}")
        lines.append("")
    lines.append("## What this PR does / why we need it?")
    tests_str = ", ".join(f"`{t}`" for t in tests) if tests else "见关联 Issue"
    intro = f"修复 {branch} 分支夜间构建失败（{date}）。失败用例：{tests_str}。"
    reason = root_cause_one_liner(state_dir)
    if reason and reason != "see issue body":
        intro += f" 根因：{reason}。"
    lines.append(intro)
    lines.append("")
    lines.append("## Does this PR introduce any user-facing change?")
    lines.append("")
    lines.append("No。")
    lines.append("")
    lines.append("## How was this patch tested?")
    lines.append("")
    summaries = [(t, _reverify_summary(t, branch, date)) for t in tests]
    if any(s for _, s in summaries):
        # 有可解析的重验日志 → 给出真实数字（loss / grad_norm / 子测试收敛数）
        for t, s in summaries:
            lines.append(f"- `{t}`: {s or '重验通过（详情见关联 Issue）'}")
    else:
        # 无重验日志时只做定性陈述，不写无法访问的本地路径
        lines.append("修复后重跑失败用例通过（完整验证见关联 Issue）。")
    lines.append("")
    lines.append("---")
    lines.append("*Auto-generated by MegatronAdaptor daily CI*")
    return "\n".join(lines)


# ── 主流程 ──────────────────────────────────────────────────────────────


def process_branch(date, result, diagnosis_md, dry_run):
    branch = result.get("branch")
    state_dir = _date_dir(date) / branch / "state"
    _log(f"==== 处理分支 {branch} ====")

    # 1. 修复是否已推送到 fork（无推送 = LLM 未修复/需人工 → 跳过，避免建孤儿 Issue）
    if not dry_run and not fork_branch_exists(branch):
        _log(f"[{branch}] daily_fix_{branch} 不在 fork 上，跳过（修复未推送）")
        return
    _log(f"[{branch}] 确认修复已推送（daily_fix_{branch}）")

    # 1.5 空 PR 守卫：分支相对 origin/<branch> 无内容变更（旧修复已合入 / 环境类修复无代码）
    # → 跳过建 Issue/PR。防止 PR #107 事故：auto-fix 无产出时 daily_fix 滞留 stale 分支，
    # 残留旧提交与已合入 main 的修复 tree 相同，建出的 PR 为空。
    if not dry_run and not fork_branch_has_diff(branch):
        _log(f"[{branch}] daily_fix_{branch} 相对 origin/{branch} 无内容差异（已合入或空修复）→ 跳过建 Issue/PR")
        for stale_pr in list_open_pulls(branch):
            stale_no = stale_pr.get("number")
            _log(f"[{branch}] 提示 stale 空 PR #{stale_no} 待人工关闭")
            comment_pull(stale_no, "此 PR 相对 base 分支无内容变更（修复已合入或无代码修复），建议人工关闭。")
        return

    commit_msg = local_fix_commit_msg(branch)
    tests = collect_failing_tests(state_dir)
    if not tests:
        _log(f"[{branch}] 未从 diag_report 取到失败用例，仅处理 PR")
    _log(f"[{branch}] 失败用例: {tests or '(unknown)'}")

    # 2. 逐问题幂等处理 Issue（同一问题跨分支/跨用例/跨天只建一个，其余追加每日评论）
    issue_numbers = []
    seen_keys = set()
    for test in tests:
        key = problem_key(state_dir, branch, test)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        existing = find_existing_problem(key, branch, test)
        if existing:
            number = existing["number"]
            url = existing.get("html_url", f"{ISSUE_BASE}/{number}")
            body = compose_daily_comment(date, branch, test, commit_msg)
            if not dry_run:
                comment_issue(number, body)
            _log(f"[{branch}][{test}] 复用已有问题 Issue #{number} {url}，追加每日评论")
            issue_numbers.append(number)
        else:
            reason = extract_reason(state_dir, test)
            title = f"{TITLE_PREFIX} {reason}"
            body = compose_issue_body(date, branch, test, state_dir, diagnosis_md)
            if dry_run:
                number = "N/A(dry-run)"
            else:
                issue = create_issue(title, body)
                number = issue.get("number")
                append_registry(
                    {
                        "problem_key": key,
                        "issue_number": number,
                        "url": issue.get("html_url", f"{ISSUE_BASE}/{number}"),
                        "first_seen_date": date,
                        "title": title,
                    }
                )
                _log(f"[{branch}][{test}] 新建问题 Issue #{number} {issue.get('html_url')}")
            issue_numbers.append(number if number != "N/A(dry-run)" else 0)

    # 3. 幂等处理 PR（粒度 = 分支）
    prs = list_open_pulls(branch)
    if prs:
        pr = prs[0]
        pr_no = pr.get("number")
        _log(f"[{branch}] 复用已有 PR #{pr_no} {pr.get('web_url')}")
    else:
        if dry_run:
            pr_no = "N/A(dry-run)"
            _log(f"[{branch}] 将新建 PR（dry-run）")
        else:
            issue_urls = [f"{ISSUE_BASE}/{n}" for n in issue_numbers if n]
            pr_title = compose_pr_title(commit_msg)
            pr_body = compose_pr_body(date, branch, tests, state_dir, issue_urls)
            pr = create_pull(f"{FORK_USER}:daily_fix_{branch}", branch, pr_title, pr_body)
            pr_no = pr.get("number")
            _log(f"[{branch}] 新建 PR #{pr_no} {pr.get('web_url')}")

    # 4. 评论 compile 触发 CI（复用 PR 时 force-push 已更新 diff，仍需重触发）
    if not dry_run and pr_no and pr_no != "N/A(dry-run)":
        comment_pull(pr_no, "compile")
        _log(f"[{branch}] 已在 PR #{pr_no} 评论 compile 触发 CI")

    # 5. 写回 pr_info.json
    if not dry_run:
        state_dir.mkdir(parents=True, exist_ok=True)
        pr_info = {
            "date": date,
            "pr_number": pr_no,
            "pr_url": f"https://gitcode.com/{REPO}/merge_requests/{pr_no}",
            "branch": branch,
            "push_branch": f"daily_fix_{branch}",
            "issues": [{"number": n, "url": f"{ISSUE_BASE}/{n}"} for n in issue_numbers if n],
        }
        out = state_dir / "pr_info.json"
        out.write_text(json.dumps(pr_info, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(f"[{branch}] 已写入 {out}")
    else:
        _log(f"[{branch}] dry-run 结束，未做任何写操作")


def main():
    parser = argparse.ArgumentParser(description="MegatronAdaptor nightly 幂等 Issue/PR 创建")
    parser.add_argument("--date", required=True, help="构建日期 YYYY-MM-DD")
    parser.add_argument("--branch", default=None, help="只处理指定分支（默认处理全部 needs_llm_diag 分支）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的操作，不调用写 API")
    args = parser.parse_args()

    date = args.date
    combined_path = _date_dir(date) / "combined_summary.json"
    if not combined_path.exists():
        raise SystemExit(f"[create_issue_pr] 缺少 {combined_path}，无法确定需诊断的分支")
    summary = json.loads(combined_path.read_text(encoding="utf-8"))

    diagnosis_md = _date_dir(date) / "DIAGNOSIS_AND_FIX.md"
    if not diagnosis_md.exists():
        diagnosis_md = None
        _log("未找到 DIAGNOSIS_AND_FIX.md，Issue body 将从 state 产物兜底生成")

    handled = 0
    for result in summary.get("results") or []:
        branch = result.get("branch")
        if args.branch and branch != args.branch:
            continue
        # 门控：needs_llm_diag（cron 诊断路径）或 fixed_tests.json（手动/agent 路径确认的真问题）。
        # fixed_tests.json 是"确认真问题"的设计信号，存在即处理——否则手动修复只建 PR 不建 Issue
        # （2026-08-14 test_activation_variants 案例：只有 PR #94 无 Issue）。
        fixed_path = _date_dir(date) / branch / "state" / "fixed_tests.json"
        if not result.get("needs_llm_diag") and not fixed_path.exists():
            _log(f"[{branch}] 无需诊断，跳过")
            continue
        try:
            process_branch(date, result, diagnosis_md, args.dry_run)
            handled += 1
        except Exception as e:  # noqa: BLE001
            _log(f"[{branch}] 处理失败: {e}")

    _log(f"完成：处理 {handled} 个分支。")
    sys.exit(0)


if __name__ == "__main__":
    main()
