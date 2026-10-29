---
name: tenpu-upstream-value-judgment
description: 分析 NVIDIA TransformerEngine 每日 PR 和提交，生成 TENPU 的上游证据、ADAPT/TRANSLATE/REFERENCE/SKIP 决策、风险分级和 Adaptation Spec。用户要求检查上游更新、验证迁移价值识别或生成每日需求分析报告时使用。
---

# TENPU 上游需求价值判断

本阶段只分析，不修改 TENPU 运行时代码。唯一输出是 `output/<D>/report.md`。

面向开发、测试和运维人员的部署、使用及流程验收说明见同目录 `README.md`。Agent 执行时仍以本文件、知识库和单日报 schema 为约束来源。

开始前读取：

1. `references/nvte-npu-migration-knowledge.md`：价值、适用性、指纹和风险知识。
2. `../shared-references/nvte-tenpu-architecture-differences.md`：NVTE/TENPU 稳定架构差异、跨层不变量和证据入口。
3. `../tenpu-daily-orchestrator/references/single-report-schema.md`：单日报 ASDD 契约。

快速筛选可只使用迁移知识；对 MEDIUM/HIGH、依赖组或涉及状态、布局、Autograd、运行时、Kernel、Collective 的候选，必须读取共享架构知识中命中的领域，并记录架构模式 ID。未命中模式时记录 `unknown-architecture-pattern`，不得凭同名路径补全结论。

其他分支可复用通用规则，但不得复用 `main` 的 SHA、完成状态或测试结论。

## 输入可靠性

确认 TE 分支、TENPU 同名原仓目标分支、知识 revision、分析游标和 TENPU evidence ref。没有可靠 `last_analyzed_upstream_sha` 时输出 `blocked/bootstrap-required`，不得虚构范围。

PR 元数据用于理解上下文；commit range 是完整性依据。标题、目录、代码量和同名符号都不是等价实现证据。

## 两阶段筛选

先按标题、路径、符号、测试和知识指纹快速分桶；再只对候选读取完整 diff、调用方、前后依赖、上游测试和 TENPU 对应生产实现。

必须覆盖：公共 API、module、tensor/quantization、FP8/MXFP8/HIF8、Attention/RoPE/THD/CP、Distributed/FSDP/EP、ops/fuser、optimizer、router/permutation/MoE、tests、build/docs 和后端专属代码。

同一 commit 含多种行为时拆成独立 requirement，并通过 dependency edges 关联。

## 四个独立判断

每个行为单元分别判断：

1. `value_case`：对用户的主要价值。
2. `adaptation_decision`：`ADAPT | TRANSLATE | REFERENCE | SKIP`。
3. `execution_action`：当前可执行动作。
4. `tenpu_coverage` 与 `confidence`：当前覆盖和证据可信度。

决策定义：

- ADAPT：API、通用逻辑、正确性或数学行为可直接适配。
- TRANSLATE：必须保留功能/语义，但 CUDA/NCCL/kernel 等实现需转换为 NPU/HCCL 方案。
- REFERENCE：当前不直接实现，仅作为未来 NPU 能力或优化设计参考。
- SKIP：确认对 TENPU 无公共契约、语义或产品能力影响。

缺当前 NPU 能力不等于 SKIP；高价值语义通常是 `TRANSLATE + prepare-capability/defer-human-review`。

## 风险评分与 Spec 深度

按 schema 的建议因子计算初始分数并列出 factors：

- LOW：Quick Spec，可自动进入设计/实现。
- MEDIUM：Standard Spec，需要 Design/Tasks 和 code review。
- HIGH：Full Spec，需要 spec、design、code 三类人工 Gate。

FP8/MXFP8/HIF8 数值、Attention、MoE、Distributed、Kernel、Autograd、RNG、Checkpoint、Memory Layout 和 Collective 默认 HIGH，除非有明确证据支持降低并记录理由。

## 生成 Adaptation Spec

Spec 描述目标行为，不描述复制哪些代码。每项至少写：

- Goal、Scope、Out of Scope。
- Python API contract 和调用方可观察行为。
- 必须保持的数值、状态、布局、异步和分布式语义。
- 禁止引入的 CUDA/NVIDIA 假设及已知 NPU 约束。
- 可执行的 acceptance criteria，并指定需要的 L1-L7 层级。

MEDIUM/HIGH Spec 还必须用共享架构知识核对 TENPU 抽象、状态所有者、跨模块影响、刻意差异和跨领域不变量。架构知识只用于确定 Scope 与契约；具体 NPU 实现方案留给 Design。

REFERENCE/SKIP 也要写证据和边界；`spec.status` 可为 `not-required`，但不能没有理由。

HIGH requirement 创建 pending `spec-review`，`next_action: human-spec-review`。在批准前不得进入 Design 或 Coding。

## 游标和开放任务

分析范围完整时，即使有待实现项，也将 `next_last_analyzed_upstream_sha` 设为范围 new SHA。为 ADAPT/TRANSLATE 创建稳定 adaptation task；REFERENCE/SKIP 形成正式终态。

不要推进 compatibility baseline。未完成任务写入 `open_adaptations`，从前日报继承并按 id 去重。

## analysis-only 定稿

默认写：

- `run_mode: analysis-only`
- `phase: analyzed` 或 `specified`
- `final_status: analysis-complete | no-op | blocked`
- implementation/verification 为 `pending` 或 `not-required`
- `next_action: await-adaptation-selection | human-spec-review | bootstrap-required | none`

将可复用的新模式写入 `knowledge_feedback`，不得直接修改知识库。

## 完成检查

- commit range 完整，PR 和依赖提交已关联。
- 每个有价值行为都检查 TENPU 生产路径和上游测试。
- 工程决策、执行动作、覆盖程度和价值没有混写。
- Risk、Spec 深度和 Gate 一致。
- 报告足以让后续 agent 设计，无需重新猜需求。
- 未修改 TENPU 运行时代码，未创建第二份产物。
