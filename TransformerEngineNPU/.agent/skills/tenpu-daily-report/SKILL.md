---
name: tenpu-daily-report
description: 校验并定稿 TENPU 每日需求价值判断的唯一日报，维护双游标与跨日开放任务。用户要求完成每日汇报、更新生命周期状态、同步开发看护仓库或生成下一日 handoff 时使用。
---

# TE NPU 单日报定稿

不创建新报告，只校验并更新 `output/<D>/report.md`。以代码、命令、人工原始意见和 `single-report-schema.md` 为事实依据，不根据对话记忆补结果。

## Revision

实质变更包括需求/Spec/Task、上游证据、人工评审、开放任务或游标变化。每次实质变更递增 `report_revision`，更新 `updated_at` 并在 `report_changes` 记录摘要和来源。

## analysis-only 定稿

检查 commit range、PR 关联、四类决策、风险、Spec、开放任务和知识 revision。

- 完整且有变更：`final_status: analysis-complete`。
- 无上游变更：`final_status: no-op`。
- 基线/分支/输入不足：`final_status: blocked`。

推进 `next_last_analyzed_upstream_sha`，保持 compatibility baseline。implementation/verification 的 pending 不是失败。

## 人工意见

原样保留 blocking findings、required changes、non-blocking suggestions、accepted risks 和 approval conditions。Agent 只能追加关联 response。

批准对象发生实质变化时保留原决定，将 review 的 `validity` 标记为 `stale`、记录失效原因并重新请求，不得静默沿用。人工接受风险不能替代失败的必需正确性测试。

## 双游标

- 分析完整即可推进 analysis cursor。
- compatibility baseline 只能沿上游顺序连续推进；更早的开放任务未闭环时不得越过它推进。
- 未闭环项以稳定 id 留在 open adaptations，下一日报继承。

## 人类摘要

保持紧凑，依次给出：日期/分支/范围/模式、价值和四类决策、风险/Gate、人工必改项和建议、开放任务、两个游标及下一步。

报告先保存在 TENPU 本地 `output/<D>/report.md`，定稿后由 Orchestrator Runner 同步到 datasets。同步失败不得删除或改写本地定稿报告。

## 完成检查

- 人类摘要和唯一 YAML 块一致。
- 双游标与开放任务能够让下一日恢复。
- 人工意见完整且批准仍有效。
- `output/` 只作为本地暂存，不得进入 TENPU 代码提交；运行环境可通过 `.git/info/exclude` 做本地忽略。
