---
name: tenpu-daily-orchestrator
description: 编排 TransformerEngine 到 TransformerEngineNPU 的每日需求价值判断，并将定稿报告同步到开发看护仓库。用户要求运行每日上游分析、生成价值判断报告或同步报告仓库时使用。
---

# TE NPU 每日需求研判编排

## 唯一事实源

所有阶段只读写 `output/<YYYY-MM-DD>/report.md`。完整契约见 `references/single-report-schema.md`。

- triage 创建报告，完成上游证据、价值判断、决策、风险和 Spec。
- daily-report 校验、递增 revision、推进游标并定稿。
- `scripts/nvte_daily_run.py` 将原始 PR 输入保存在同一日期目录。

禁止创建 `artifacts/`、独立 state 或阶段报告。`prs.json` 只是可重放输入。

## 运行模式

当前只执行需求研判（`run_mode: analysis-only`）：

```text
准备输入 -> value judgment -> daily-report 定稿 -> push datasets -> 停止
```

不得因报告中存在 `port-now` 就在需求研判阶段自动编码。

## 分支与仓库

自动识别 TE 当前原仓分支；TENPU 目标原仓分支与其同名。当前约定：NVTE `main` 对应 TENPU `main`，`agent_dev` 是基于 `main` 的流程研发分支。

需求研判不创建业务适配分支，也不修改 TENPU 运行时代码。

## 跨日恢复

按同一 `branch_pair_id` 查找上一份有效报告：

1. 本地 `output/` 中日期更早的最新报告。
2. datasets 工作树 `nvte-daily/` 中最新报告。
3. 人工写入当前报告的 bootstrap SHA。

分别恢复：

- `last_analyzed_upstream_sha`：下一次需求分析起点。
- `compatibility_baseline_sha`：已经完成兼容闭环的上游基线。

首次运行且没有可靠基线时仍生成 blocked 报告，不得构造 `null..HEAD` 范围。

## 标准流程

### 1. 准备

- 日期默认前一个北京自然日。
- fetch TE 远端，识别分支和 SHA；检查 TENPU 工作树及目标原仓分支。
- 执行 `python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py --date <D>` 获取 PR 输入。失败时记录 unavailable，若本地 git 证据完整可继续。
- 计算迁移价值和共享架构差异两个知识库的 git blob SHA。

### 2. 需求研判

读取 `tenpu-upstream-value-judgment/SKILL.md`。综合 PR、commit、diff、上游测试、TENPU 代码、价值知识库和共享架构差异知识，生成 Risk-Based Adaptation Spec。

完成后调用 daily-report 校验，将 `phase` 设为 `analyzed`/`specified`，`final_status` 设为 `analysis-complete`、`no-op` 或 `blocked`。推进分析游标，保留 compatibility baseline。

### 3. 定稿与同步

读取 daily-report skill，校验任务状态、Spec、人工 Gate 和双游标。每次实质更新递增 `report_revision`。

报告先定稿到 `output/<D>/report.md`，原始 PR 输入保存在同目录 `prs.json`，再由 Runner 同步到 datasets。Agent 不得直接操作报告仓库。

```bash
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py --date <D> --push
```

## 人工 Gate

人工意见只写报告的 `human_reviews`。Agent 不得编辑人工原文，只能追加 `agent_responses`。

- `changes-requested`/`rejected`：停止当前 task。
- `approved-with-conditions`：条件有可验证完成状态后才继续。
- Spec revision 超出批准范围：旧批准失效，重新请求评审。
- HIGH Spec 必须经过 spec-review；不得因缺少人工批准就进入后续阶段。

## 结束检查

- 只有一个日报和一个机器 YAML 块。
- 需求研判没有修改 TENPU 运行时代码。
- 四类工程决策与执行动作分开。
- 分析游标不因开放任务阻塞；兼容基线不越过未关闭任务。
- 每个已识别的 adaptation task 有稳定 task/requirement id。
- 人工批准绑定当前 Spec 版本。
- `output/` 未进入 TENPU git 状态。
- 报告同步只由 Runner 执行，目标路径和分支正确。

变更流程后按 `references/daily-run-checklist.md` dry-run。
