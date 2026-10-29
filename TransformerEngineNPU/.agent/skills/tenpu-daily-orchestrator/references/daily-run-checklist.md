# TENPU ASDD 单日报流程验证清单

## 结构

- [ ] 各 skill 的目录名与 frontmatter name 一致，description 同时说明用途和触发场景。
- [ ] 单日报 schema 为 v3，文末只有一个 `daily_te_npu` YAML 块。
- [ ] triage、architecture 两个知识库路径存在且 revision 可计算。
- [ ] 没有 `artifacts/`、独立 state 或阶段报告写入指令。

## Analysis-only

1. 无基线：只生成 blocked/bootstrap-required 报告，不构造 `null..HEAD`。
2. 无新提交：生成 no-op；两个游标各自保持前值，不强制彼此相等。
3. API-only：输出 ADAPT、LOW/Quick Spec、API contract 和 L1/L2 acceptance。
4. CUDA kernel 且有目标语义：输出 TRANSLATE/REFERENCE，不因缺 NPU kernel 直接 SKIP。
5. 纯 CUDA/JAX/build：证据充分时 SKIP，不创建 adaptation task。
6. 混合提交：拆行为单元，可同时包含 ADAPT、TRANSLATE、REFERENCE 或 SKIP。
7. 高风险：FP8/Attention/Distributed 等输出 HIGH/Full Spec 和 pending spec-review。
8. 分析完整但待实现：推进 analysis cursor，不推进 compatibility baseline，创建 open adaptation。
9. analysis-only 完成后 TENPU 运行时代码和业务分支均未改变。

## 流程验收

按 `../../tenpu-upstream-value-judgment/README.md` 的顺序执行，不得只检查 Agent 是否生成了 Markdown：

1. Skill 结构与依赖路径检查。
2. Orchestrator `--dry-run`，确认无代码和 Git 状态变化。
3. 抓取指定日期的 `prs.json`，核对 PR、commit、文件列表和本地上游 ref。
4. 使用固定 Gold Cases 执行价值研判。
5. 校验唯一报告、人类摘要、机器 YAML、判断证据和风险 Gate。
6. 人工抽检一个适配项、一个排除项和一个高风险项。
7. 在测试 datasets 工作树验证同步目标、幂等提交和失败恢复后，再接入定时任务。

Gold Cases 至少覆盖：

- 公共 Python API：`ADAPT`。
- JAX-only：`SKIP + skip-not-applicable`。
- CUDA Kernel 但目标语义有价值：`TRANSLATE` 或 `REFERENCE`。
- FP8/MXFP8 scale/amax：HIGH/Full Spec/人工 Gate。
- JAX 与公共 API 混合 PR：拆成独立行为分别判断。
- TENPU 已有等价生产实现：`verify-existing` 并提供代码证据。
- NVFP4-only：默认 SKIP，不得伪支持。
- Attention/HCCL/Distributed：HIGH 并记录架构证据。

流程通过条件：没有阻塞级漏判或误判；每条非 `none` 判断有价值证据；每条 SKIP 有排除理由；HIGH 项未降级；报告未修改 TENPU 运行时代码。

## 报告和 datasets

1. 人类摘要、机器块、开放任务和双游标一致。
2. 同一内容重复同步时幂等跳过 commit；报告修订后生成新的 datasets commit。
3. TENPU 项目 `.gitignore` 不增加通用 `output/` 规则；运行环境通过 `.git/info/exclude` 或任务后清理暂存，代码提交不得包含日报或 PR 原始输入。
4. 同步失败保留本地报告，且 Agent 本身不直接操作 datasets。

## 验收提示词

- “分析今天 NVTE main 的变化，只执行需求研判（analysis-only），不修改 TENPU 代码。”
