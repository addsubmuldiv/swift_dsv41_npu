# TENPU 单日报 ASDD 契约

本文件是每日需求研判(triage)与日报定稿共用的唯一运行时契约。只写 `output/<YYYY-MM-DD>/report.md`，不创建阶段 artifact 或独立 state 文件。

## 目录

- 报告角色与修订
- 人类可读章节
- 机器可读模板
- Risk-Based ASDD
- 人工评审
- 双游标与跨日任务
- 一致性规则

## 报告角色与修订

- `analysis-only`：当前唯一运行模式。完成需求分析和 Spec 后定稿到 TENPU 本地 `output/<D>/report.md`，再同步到 datasets；不修改 TENPU 运行时代码。
- 每次实质更新递增 `report_revision` 并更新 `updated_at`。同日报告的修订摘要写入 `report_changes`，历史版本由 datasets Git 历史保存。
- 需求分析是事实和目标契约。后续发现错误可以修订，但必须递增对应 revision，并记录 `report_changes`。

## 人类可读章节

报告使用中文，至少包含：执行摘要、上游证据、需求与价值判断、Adaptation Spec、风险与 Gate、适配设计、任务与分支、实施、验证与 Spec Compliance、人工评审、开放任务、下一步、机器可读状态。

`analysis-only` 报告只要求前五项完整；后续章节明确写“待执行”，不因尚未实施而标记失败。

## 机器可读模板

文末只放一个 `yaml` 代码块，顶层键固定为 `daily_te_npu`。未知值用 `null`，空集合用 `[]`。

```yaml
daily_te_npu:
  schema_version: 3
  report_revision: 1
  run_mode: analysis-only
  date: YYYY-MM-DD
  timezone: Asia/Shanghai
  created_at: ""
  updated_at: ""
  phase: analyzing | analyzed | specifying | specified | designing | designed | implementing | implemented | verifying | verified | blocked | needs-review
  final_status: pending | analysis-complete | no-op | adapted | partial | blocked | needs-review
  branch:
    branch_pair_id: te-main__tenpu-main
    upstream_branch: main
    target_tenpu_branch: main
    target_tenpu_ref: origin/main
    workflow_branch: agent_dev
    workflow_branch_base: main
    workflow_branch_base_evidence: ""
  knowledge:
    triage:
      reference: .agent/skills/tenpu-upstream-value-judgment/references/nvte-npu-migration-knowledge.md
      revision: ""
    architecture:
      reference: .agent/skills/shared-references/nvte-tenpu-architecture-differences.md
      revision: ""
  cursors:
    previous_last_analyzed_upstream_sha: null
    next_last_analyzed_upstream_sha: null
    previous_compatibility_baseline_sha: null
    next_compatibility_baseline_sha: null
  upstream_range:
    mode: incremental | calendar-audit
    old_sha: null
    new_sha: null
    commits: []
    analysis_complete: false
  inputs:
    pr_data: output/YYYY-MM-DD/prs.json
    pr_input_status: available | unavailable | not-requested
    previous_report: null
    tenpu_evidence_ref: null
  bootstrap:
    required: false
    confirmed_sha: null
  requirements:
    - id: "<commit-sha>:<behavior-unit>"
      upstream_commit: ""
      upstream_pr: null
      title: ""
      upstream_behavior: ""
      functional_layers: []
      applicable_surfaces: []
      architecture_pattern_ids: []
      architecture_evidence: []
      value_case: correctness | api-parity | ecosystem-enabler | performance-enabler | operability | design-reference | none
      adaptation_value: none | low | medium | high | critical
      adaptation_decision: ADAPT | TRANSLATE | REFERENCE | SKIP
      execution_action: port-now | port-python-only | verify-existing | prepare-capability | defer-human-review | skip-not-applicable
      tenpu_coverage: implemented | partial | absent | npu-divergent | not-applicable | unknown
      capability_evidence: []
      dependency_group: null
      dependency_edges: []
      confidence: low | medium | high
      risk:
        score: 0
        level: LOW | MEDIUM | HIGH
        factors: []
        spec_depth: quick | standard | full
        required_gates: []
      spec:
        revision: 1
        status: draft | ready | approved | changes-requested | not-required
        goal: ""
        scope: []
        out_of_scope: []
        api_contract: []
        required_semantics: []
        npu_constraints: []
        acceptance_criteria: []
      design:
        revision: 0
        status: pending | draft | ready | approved | changes-requested | not-required
        upstream_to_npu_mapping: []
        semantic_differences: []
        fallback_and_failure_behavior: []
        performance_considerations: []
        test_strategy: []
      tasks: []
      human_review_required: false
      reasons: []
  adaptation_tasks:
    - task_id: "ADAPT-<pr-or-sha>-<slug>"
      requirement_ids: []
      status: discovered | specified | designing | ready | implementing | testing | review | merged | validated | skipped | reference-only | blocked
      work_branch:
        name: "adapt/nvte-<pr-or-sha>-<slug>"
        base_branch: main
        base_sha: null
        head_sha: null
        base_evidence: ""
      upstream_commits: []
      upstream_prs: []
  implementation:
    status: pending | not-required | implemented | partial | blocked | needs-review
    task_ids: []
    completed_task_item_ids: []
    deferred_task_item_ids: []
    changed_files: []
    test_files_changed: []
    intentional_divergences: []
    notes: []
  verification:
    status: pending | not-required | passed | partial | failed | not-run | needs-review
    independence: independent-agent | same-agent | unavailable
    required_levels: []
    execution_context:
      npu_available: unknown
      device_model: ""
      device_count: 0
      world_size: null
    tests: []
    spec_compliance: []
    autocorrect: []
    performance: {status: not-required, baseline: null, result: null}
    residual_risk: low | medium | high | critical | unknown
  human_reviews: []
  agent_responses: []
  open_adaptations: []
  blockers: []
  knowledge_feedback: []
  report_changes: []
  next_action: await-adaptation-selection | tenpu-daily-report | human-spec-review | human-design-review | human-code-review | bootstrap-required | none
```

## Risk-Based ASDD

- LOW（建议 0-3）：Quick Spec。只保留目标、范围、API/行为、NPU 约束和必要验收；可自动进入实现。
- MEDIUM（建议 4-8）：Standard Spec。必须有 Design 和 Tasks；实现后需要 Code Review。
- HIGH（建议 >=9）：Full Spec。Spec Review 和 Design Review 通过前不得编码，Code Review 通过前不得合并。

风险因子建议：Python API +2、数值行为 +3、FP8/MXFP8/HIF8 +5、Attention +5、Distributed +5、MoE +4、Kernel +5、Autograd/RNG +4、Checkpoint +3、CUDA/NPU 语义差异 +3、大 diff +2、correctness fix +4。分数用于一致分流，不替代专家判断；人工可以调整，但必须记录理由。

`adaptation_decision` 是工程结论；`execution_action` 是当前执行动作。两者必须分别填写。例如 GPU kernel 对 TENPU 有等价功能诉求时可为 `TRANSLATE + prepare-capability`，不能因当前缺 kernel 就写成 SKIP。

## 人工评审

每条 `human_reviews` 使用以下字段：

```yaml
- id: HR-001
  gate: spec-review | design-review | code-review | risk-acceptance
  status: pending | approved | approved-with-conditions | changes-requested | rejected
  validity: active | stale
  requirement_ids: []
  reviewed_subject:
    report_revision: 1
    spec_revisions: {}
    design_revisions: {}
    tenpu_commit: null
  reviewer: ""
  reviewed_at: ""
  invalidated_at: null
  invalidation_reason: null
  decision_summary: ""
  blocking_findings: []
  required_changes: []
  non_blocking_suggestions: []
  accepted_risks: []
  approval_conditions: []
  evidence: []
  re_review_required_when: []
```

- 人工可以在 `required_changes` 写必须修改项，在 `non_blocking_suggestions` 写优化建议。
- Agent 不得改写人工原文；处理结果写入 `agent_responses`，通过 review id 关联。
- 批准只对 `reviewed_subject` 有效。Spec、Design 或代码发生实质变化时，保留原决定，将 `validity` 设为 `stale`、记录失效原因并重新进入对应 Gate。
- `approved-with-conditions` 只有在所有 `approval_conditions` 有可验证状态时才能继续。

## 双游标与跨日任务

- `next_last_analyzed_upstream_sha`：完整分析范围内所有 commit 后推进，即使其中存在待实现或待人工任务。它用于避免次日重复分析。
- `next_compatibility_baseline_sha`：只能沿上游提交顺序连续推进；目标 SHA 之前的所有适用行为均已 ADAPT/TRANSLATE 并验证、REFERENCE/SKIP 有充分理由，且所有必需 Gate 已关闭。较早的开放任务会阻止越过它推进基线。
- 未完成任务进入 `open_adaptations`，下一日报按稳定 task/requirement id 继承，不得通过回退分析游标重复创建。
- schema v2 的 `next_last_adapted_upstream_sha` 只有在旧报告状态为 `no-op`、分支一致且无未决人工项时，才可同时初始化两个游标；否则要求 bootstrap。

## 一致性规则

- 一个 commit 可拆成多个行为单元；一个依赖组可组成一个 adaptation task 和一个工作分支。
- `open_adaptations` 保存未闭环的 `adaptation_tasks.task_id`，按稳定 id 跨日继承和去重。
- 实际适配分支必须从 TENPU `origin/<target_tenpu_branch>` 创建。`agent_dev` 是流程研发分支，不是业务适配分支。
- Tasks 至少包含 `id`、`requirement_ids`、`description`、`files`、`status` 和 `acceptance_refs`。Coding 只能执行已批准 Scope 内的 Tasks。
- 测试项至少包含 `requirement_ids`、`level`、`command`、`result`、`evidence` 和 `reason`。
- `spec_compliance` 逐条关联 acceptance criterion，状态为 `passed | failed | not-verified | not-applicable`。
- 需求研判的正确终态是 `analysis-complete` 或 `no-op`。
- 未运行、skip、static-only、缺全 rank 证据或缺硬件证据不能写成通过。
