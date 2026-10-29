#!/bin/bash
# 每日凌晨自动：确定性 pipeline → CI report 推送 → (可选) LLM 诊断修复
# 整个脚本完全以 ma_daily_ci 用户运行：crontab 属于 ma_daily_ci，仓库属主 =
# 运行用户，避免 git dubious ownership 与 root 产生的 .pyc/ckpt 属主问题。
# 脚本顶部设有守卫：即使 crontab 意外挂在 root 下触发，也会立即 su 重执行
# 为 ma_daily_ci，保证 ①→⑨ 全流程身份一致（git / pip / claude 均属 ma_daily_ci）。
#
# crontab (ma_daily_ci):
#   CRON_TZ=Asia/Shanghai
#   57 4 * * * bash /home/ma_daily_ci/workspace/.claude/skills/megatron-adaptor-sync/scripts/nightly_sync_and_report.sh
#   （脚本自管日志重定向到 output/logs/YYYY-MM/MM-DD/cron_nightly.log，crontab 不再重定向）

set -e

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="/home/ma_daily_ci/workspace"
BUILD_DATE="${1:-$(date +%Y-%m-%d)}"
# 按月分层：output/YYYY-MM/MM-DD 与 output/logs/YYYY-MM/MM-DD（与 CI 仓 sync_build_results 一致）
BUILD_MONTH="${BUILD_DATE:0:7}"   # 2026-08
BUILD_DAY="${BUILD_DATE:5}"        # 08-19
RUN_DIR="$WORKSPACE/output/$BUILD_MONTH/$BUILD_DAY"
LOG_DIR="$WORKSPACE/output/logs/$BUILD_MONTH/$BUILD_DAY"
DATASET_CACHE="/home/dataset/cache/megatron_st_cache"

# 日志统一重定向到按月分层的 cron_nightly.log（append；crontab 不再单独重定向，
# 避免 output/logs/ 下平铺文件。同一天多次运行追加到同一文件，不覆盖）。
mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/cron_nightly.log" 2>&1

# Conda / CANN 路径
export PATH="/home/miniconda3/bin:$PATH"

# ── 完全以 ma_daily_ci 运行守卫（crontab 挂在 root 下触发）──
# crontab 以 root 触发本脚本，恰好利用 root 权限先做防御性清理：把三个仓库 /
# 输出目录 / 数据集缓存中的 root 属主文件 chown 给 ma_daily_ci，再 su 重执行
# 为 ma_daily_ci（login shell，HOME=/home/ma_daily_ci），保证全部步骤（确定性
# pipeline、claude 诊断、Issue/PR、CI 推送）都以 ma_daily_ci 身份执行，避免
# root 产生 .pyc / ckpt 属主问题、git dubious ownership 与 CLA 身份混乱。
# 若不先清理：root 属主的 untracked 残留（例如曾以 root 手动 checkout core 分支
# 留下的 tests/unit_tests/distributed/megatron_fsdp/）会阻断 git clean -fd 与
# git checkout 分支切换——fetch 静默失败在错误分支上继续（2026-08-16/17 事故）。
# chown root 属主文件需要 root 权限，因此放在切换用户之前、仍为 root 时执行。
if [ "$(id -u)" = "0" ]; then
    echo ">>> Running as root (uid=0); cleaning root-owned artifacts, then re-executing as ma_daily_ci..."
    find "$WORKSPACE/Megatron-LM" \
         "$WORKSPACE/MegatronAdaptor" \
         "$WORKSPACE/TransformerEngineNPU" \
         "$WORKSPACE/output/$BUILD_MONTH" \
         "$WORKSPACE/output/logs" \
         "$DATASET_CACHE" \
         -user root -exec chown ma_daily_ci:ma_daily_ci {} + 2>/dev/null || true
    # GPT dataset 索引缓存：root 属主会阻断 ma_daily_ci 新建 cache 文件。
    # 2026-08-21~24 main 分支连续 4 天 PermissionError 根因：上游 #6622 tokenizer
    # 重构致 dataset cache hash 全变，main 需新建 cache 但目录 root:root 不可写。
    # chown -R 该 cache 目录树（仅派生构建产物，不动源数据），防复发。
    find /home/dataset -type d -name "*Dataset_indices" -exec chown -R ma_daily_ci:ma_daily_ci {} + 2>/dev/null || true
    echo "<<< Root-owned artifact cleanup done."
    exec su - ma_daily_ci -c "bash \"$0\" $*"
fi

# 同一天多次运行时，把已存在的运行目录改名归档（MM-DD_rN），避免覆盖历史
# 运行日志（含定位修复的 DIAGNOSIS_AND_FIX.md / 各分支 state 等）。
if [ -d "$RUN_DIR" ]; then
    n=1
    while [ -d "${RUN_DIR}_r${n}" ]; do n=$((n+1)); done
    mv "$RUN_DIR" "${RUN_DIR}_r${n}"
    echo ">>> Archived previous run to ${RUN_DIR}_r${n}"
fi
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "Nightly Sync + CI Report - $BUILD_DATE"
echo "Start: $(date)"
echo "Running as: $(id -un)"
echo "============================================================"

# ── Step 1: Deterministic pipeline (runs as current user) ──
echo ""
echo ">>> Step 1/3: Running deterministic pipeline (nightly_orchestrator.py)..."

PIPELINE_EXIT=0
python3 "$SKILL_DIR/scripts/nightly_orchestrator.py" --date "$BUILD_DATE" \
    2>&1 || PIPELINE_EXIT=$?

echo ""
echo "<<< Step 1/3 done (exit=$PIPELINE_EXIT)"

# ── Step 2: LLM diagnosis + patch (runs as ma_daily_ci, nvm/claude belong to them) ──
COMBINED_SUMMARY="$RUN_DIR/combined_summary.json"
NEEDS_DIAG=0

if [ -f "$COMBINED_SUMMARY" ]; then
    NEEDS_DIAG=$(python3 -c "
import json
with open('$COMBINED_SUMMARY') as f:
    data = json.load(f)
print(data.get('needs_llm_diagnosis', 0))
" 2>/dev/null || echo 0)
fi

if [ "$NEEDS_DIAG" -gt 0 ] 2>/dev/null; then
    echo ""
    echo ">>> Step 2/3: $NEEDS_DIAG branch(es) need LLM diagnosis + patch..."

    # 构建 claude prompt（写入临时文件避免 shell 转义问题）
    CLAUDE_PROMPT_FILE="$LOG_DIR/claude_prompt.txt"
    cat > "$CLAUDE_PROMPT_FILE" << 'CLAUDE_EOF'
/megatron-adaptor-sync 读取 combined_summary.json，对 needs_llm_diagnosis=true 的分支执行 ⑥ DIAG → ⑦ PATCH → ③ 重验流程。
严格按照 SKILL.md 中的诊断修复循环操作。不要询问任何问题，全自动执行。
每个失败分支：读取 diag_prompt.txt → 分析根因 → 生成 patch → AST 校验 → 重验失败用例。
全部修复完成后：本地 commit → force push 到个人仓（不要创建 Issue/PR，幂等创建由 Step 2.5 的 create_issue_pr.py 自动完成，切勿手工 curl 创建）。
⚠️ 分支卫生（无论本次是否产生新 fix 都必须做）：push 前先 `git checkout -B daily_fix_<branch> origin/<branch>` 从上游基线重建分支，再做新 commit。防止残留旧提交（已合入 main 的修复）滞留分支导致下次建出空 PR（2026-08-24 PR #107 事故）。若重建后无新提交，则无需推送。
提交身份硬性要求：先读取 scripts/.push_config.json 中的 repo.author_name 和 repo.author_email，commit 的 author 与 committer 必须同时用这两个值（`git -c user.name=<author_name> -c user.email=<author_email> commit`），禁止使用 MA Daily CI 或其它未签署 gitcode CLA 的身份，否则 PR 的 CLA 检查会失败（详见 references/operations.md ②③ 与检查清单）。commit 标题必须用 `fix(agent): <描述>` 前缀（标明 agent 提交），描述后追加 `[<测试名>] [branch: <megatron_branch>]` 标签。
代码注释风格：注释必须简洁明了，一两句说明"做什么 + 为什么"；函数 docstring 一句话说清目的，复杂"为什么"用一行行内注释。禁止写历史背景/跨文件考证/上游 commit 溯源进代码注释（这些放 Issue 评论或 DIAGNOSIS_AND_FIX.md）。
最后，将诊断分析和修复总结写入 output/<本月>/<本日>/DIAGNOSIS_AND_FIX.md（如 output/2026-08/08-19/，下同），按 SKILL.md ⑧.5 中规定的结构编写。报告正文必须用中文撰写（代码、diff、日志摘录保留原文），严格遵循 references/diag_report_writing_guide.md 的 12 条硬性要求。
另外，为每个失败分支写 output/<本月>/<本日>/<分支>/state/fixed_tests.json，内容是本次**确认修复的真问题**测试名列表（如 ["test_tokenizer_dataset"]）。只列入确认为训练代码 bug 且已修复、需要建 Issue/PR 的测试；verify 脚本误报（如跨子测试首尾比较的误报）等非训练 bug 一律不要列入。create_issue_pr.py 只据此建 Issue（幂等，绝不重复建）。
（CI 报告仓归档由 Step 3 的 push_sync_ci_results.py 自动完成，无需在此处理。）
当前项目即 /home/ma_daily_ci/workspace，所有路径均以此为基础。
CLAUDE_EOF

    # 全流程已由顶部守卫保证以 ma_daily_ci 身份运行（nvm/claude 属主）；
    # root 属主产物的防御性 chown 已在顶部守卫的 root 阶段完成。
    CLAUDE_EXIT=0
    # 统一的 claude 调用（已由顶部守卫保证以 ma_daily_ci 运行）
    run_claude() {
        export NVM_DIR="$HOME/.nvm" && \
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" && \
        cd "$WORKSPACE" && \
        claude -p \
            --add-dir "$WORKSPACE" \
            --output-format text \
            --max-turns 200 \
            --allow-dangerously-skip-permissions \
            "$(cat "$CLAUDE_PROMPT_FILE")" \
            2>&1
    }
    run_claude || CLAUDE_EXIT=$?

    rm -f "$CLAUDE_PROMPT_FILE"

    # ── Step 2.5: 确定性幂等 Issue/PR 创建（替代 LLM 手工 curl，避免每晚重复建 Issue）──
    echo ""
    echo ">>> Step 2.5: Creating/reusing Issues & PRs (create_issue_pr.py)..."
    python3 "$SKILL_DIR/scripts/create_issue_pr.py" --date "$BUILD_DATE" \
        2>&1
    ISSUE_PR_EXIT=$?
    echo ""
    echo "<<< Step 2.5 done (exit=$ISSUE_PR_EXIT)"

    echo ""
    echo "<<< Step 2/3 done (exit=$CLAUDE_EXIT)"
else
    echo ""
    echo ">>> Step 2/3: No LLM diagnosis needed (all branches passed or no diag required). Skipping."
    CLAUDE_EXIT=0
fi

# ── Step 3: CI report push (always, includes artifacts from Step 2) ──
echo ""
echo ">>> Step 3/3: Pushing CI report..."
python3 "$SKILL_DIR/scripts/push_sync_ci_results.py" "$BUILD_DATE" \
    2>&1
REPORT_EXIT=$?
echo ""
echo "<<< Step 3/3 done (exit=$REPORT_EXIT)"

echo ""
echo "============================================================"
echo "Done: $(date)"
echo "Pipeline exit: $PIPELINE_EXIT  |  Claude exit: $CLAUDE_EXIT  |  Report exit: $REPORT_EXIT"
echo "============================================================"
