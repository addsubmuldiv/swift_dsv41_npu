# TE-Agent 需求价值识别 Skill 使用说明

## 1. 目标

本 Skill 用于分析 NVIDIA TransformerEngine（TE/NVTE）已经合入的 PR 和 commit，判断其中哪些行为需要在 TransformerEngineNPU（TENPU）适配，并输出中文价值判断报告。

当前交付范围只有“需求价值识别”，不修改 TENPU 运行时代码，不创建业务适配分支，不执行 NPU 测试。

主要能力：

- 识别公共 Python API、正确性、量化、Attention、分布式、MoE、构建和测试等变更。
- 将一个混合 PR 拆成多个独立行为单元，避免整项迁移或整项跳过。
- 输出 `ADAPT | TRANSLATE | REFERENCE | SKIP` 工程决策。
- 排除 JAX-only、CUDA-only、NCCL-only、NVIDIA CI 等对 TENPU 不适用的行为。
- 给出价值、TENPU 覆盖情况、风险、Spec 深度、人工 Gate 和建议行动。
- 将最终报告和原始 PR 输入同步到开发看护使用的 datasets 仓库。

## 2. 交付内容

需求价值识别不能只复制当前目录下的 `SKILL.md`。完整运行至少需要以下文件：

```text
.agent/skills/
├── shared-references/
│   └── nvte-tenpu-architecture-differences.md
├── tenpu-upstream-value-judgment/
│   ├── SKILL.md
│   ├── README.md
│   └── references/
│       └── nvte-npu-migration-knowledge.md
├── tenpu-daily-orchestrator/
│   ├── SKILL.md
│   ├── scripts/
│   │   ├── nvte_daily_run.py
│   │   └── nvte_daily_orchestrate.py
│   └── references/
│       ├── daily-run-checklist.md
│       └── single-report-schema.md
└── tenpu-daily-report/
    └── SKILL.md
```

各组件职责：

| 组件 | 职责 |
| --- | --- |
| `tenpu-upstream-value-judgment` | 读取上游变更和 TENPU 代码，完成价值判断。 |
| 迁移知识库 | 提供价值类型、迁移指纹、排除规则和风险规则。 |
| 架构差异知识库 | 提供 NVTE 与 TENPU 的稳定架构差异和检查面。 |
| 单日报 schema | 规定报告的机器可读格式和一致性要求。 |
| `tenpu-daily-report` | 校验并定稿唯一日报。 |
| Orchestrator | 串联输入抓取、Agent 研判、报告校验和 datasets 同步。 |

## 3. 环境要求

运行服务器需要：

1. TENPU 仓库，以及可读取的 NVTE 仓库或等价的完整上游 diff。
2. Python 3.10 或更高版本。
3. Git。
4. 已安装并完成认证的 Claude Code CLI，命令名为 `claude`。
5. Python 包 `requests` 和 `PyYAML`；运行单元测试时还需要 `pytest`。
6. 可读取 NVIDIA/TransformerEngine GitHub API 的 token。
7. 已初始化并可推送的 datasets 工作树。

默认路径由脚本定义：

| 内容 | 默认位置 |
| --- | --- |
| 本地报告目录 | `<TENPU>/output/<YYYY-MM-DD>/` |
| datasets 工作树 | `<TENPU>/../datasets` |
| datasets 分支 | `nvte-daily-reports` |

推荐目录结构：

```text
<workspace>/
├── TransformerEngineNPU/
└── datasets/
```

如果实际工作树名称或位置不同，运行前设置：

```bash
export TENPU_DATASETS_WORKTREE=/absolute/path/to/datasets
```

### 3.1 GitHub API 认证

GitHub API 凭据只用于读取 NVTE PR 和 commit，不是 datasets 的 Git push 凭据。Runner 按以下顺序获取：

1. 当前进程的 `GH_TOKEN`。
2. 当前进程的 `GITHUB_TOKEN`。
3. 当前系统用户执行 `gh auth login` 后的 `gh auth token`。

定时任务可通过任务环境或服务器密钥管理服务注入 `GH_TOKEN`：

```bash
export GH_TOKEN=<injected-by-secret-manager>
```

不要把 token 写入 TENPU、datasets、README、`.env` 模板或 Git 配置文件。

### 3.2 datasets Git 凭据

datasets 的 pull/push 使用该仓库自己的 Git remote 和凭据，不由 Runner 保存密码。先检查：

```bash
git -C ../datasets switch nvte-daily-reports
git -C ../datasets remote -v
```

多人服务器建议每位开发者使用独立系统账号、SSH key 和 datasets 工作树。Git remote、`credential.useHttpPath` 和经安全评审的 credential helper 可以通过 `git -C ../datasets config --local ...` 只写入该工作树的 `.git/config`；密钥或 token 本身仍应保存在 SSH agent、系统 credential helper 或密钥管理服务中，不能写入仓库。

## 4. 输入与输出

### 4.1 输入

每日流程使用：

- 北京自然日 `D`。
- 当日合入 TE 的 PR 和 commit。
- 上游完整 diff、调用方和测试。
- TENPU 当前目标原仓分支代码。
- 前一份同分支报告中的分析游标和开放任务。
- 迁移知识库和架构差异知识库的 Git blob SHA。

`output/<D>/prs.json` 是可重放的原始 PR 输入，不是最终结论。

### 4.2 唯一输出

```text
output/<D>/report.md
```

报告由两部分组成：

1. 人类可读的中文价值判断总结。
2. 文末唯一一个 `daily_te_npu` YAML 机器块。

报告至少回答：

- TE 合入了什么行为？
- 对 TENPU 用户是否有价值？
- TENPU 当前是否已经覆盖？
- 应当适配、转换、参考还是跳过？
- 判断依据是什么？
- 风险、优先级和下一步是什么？

`output/` 是同步前的本地暂存目录，不进入 TENPU 代码提交。定稿后的报告和 `prs.json` 由 Runner 同步到 datasets。

项目 `.gitignore` 不增加通用 `output/` 规则。服务器初始化时可以仅对当前工作树设置本地忽略：

```bash
echo /output/ >> .git/info/exclude
```

`.git/info/exclude` 不会被提交，也不会影响其他开发者；如果不配置，任务结束后应清理已成功同步的本地暂存，并在提交代码前检查 `git status`。

## 5. 标准使用流程

### 5.1 只检查将执行的步骤

```bash
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_orchestrate.py --date <YYYY-MM-DD> --dry-run
```

该命令不拉起 Agent，也不实际 push，只用于检查日期、路径和计划动作。

### 5.2 抓取每日 PR 输入

```bash
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py --date <YYYY-MM-DD>
```

成功后检查：

```text
output/<YYYY-MM-DD>/prs.json
```

不要只看 PR 标题。价值判断必须继续读取完整 diff、调用方、上游测试和 TENPU 对应生产路径。

### 5.3 手工触发 Skill

在 TENPU 仓库根目录向 Agent 输入：

```text
请分析北京自然日 <YYYY-MM-DD> 合入 NVTE 的变化，只执行需求价值识别（analysis-only）。

严格读取并遵循：
1. .agent/skills/tenpu-upstream-value-judgment/SKILL.md
2. .agent/skills/tenpu-upstream-value-judgment/references/nvte-npu-migration-knowledge.md
3. .agent/skills/shared-references/nvte-tenpu-architecture-differences.md
4. .agent/skills/tenpu-daily-orchestrator/references/single-report-schema.md

读取 output/<YYYY-MM-DD>/prs.json、上游完整 diff、调用方、上游测试和 TENPU 当前生产代码。
将混合 PR 拆成独立行为单元，分别输出价值、适用性、ADAPT/TRANSLATE/REFERENCE/SKIP、执行动作、TENPU 覆盖、风险和判断证据。
JAX-only、CUDA-only、NCCL-only 和 NVIDIA CI 等行为有充分证据时标记 SKIP；同一 PR 中独立的公共 Python/API 行为仍需单独判断。

唯一输出写入 output/<YYYY-MM-DD>/report.md。禁止修改 transformer_engine/，禁止创建分支、提交或推送。
```

### 5.4 自动运行

```bash
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_orchestrate.py --date <YYYY-MM-DD>
```

自动流程为：

```text
抓取输入 -> Headless Agent 价值研判 -> 校验 report.md -> push datasets
```

该命令在报告校验通过后，将本地报告和原始输入提交并推送到 datasets。

### 5.5 重跑

报告已存在时，Orchestrator 不会重新调用 Agent，而是复用、校验并重试同步，适合恢复上次 push 失败：

```bash
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_orchestrate.py --date <YYYY-MM-DD>
```

只有确认需要重新生成时才使用：

```bash
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_orchestrate.py --date <YYYY-MM-DD> --force
```

`--force` 可能覆盖本地报告。运行前应确认人工意见已进入 datasets Git 历史，或已另行备份。

## 6. 判断规则速查

### 6.1 四类工程决策

| 决策 | 含义 | 典型场景 |
| --- | --- | --- |
| `ADAPT` | 可直接保留上游 API、通用逻辑或正确性行为 | Python API、默认参数、DType、序列化、错误处理。 |
| `TRANSLATE` | 目标语义有价值，但 GPU 实现必须转换为 NPU/HCCL 方案 | CUDA Kernel、NCCL Collective、GPU Runtime。 |
| `REFERENCE` | 当前不直接实现，只作为未来能力设计参考 | 当前缺少 NPU 能力但存在潜在产品方向。 |
| `SKIP` | 已确认没有 TENPU 公共契约、计算语义或产品收益 | JAX-only、纯 NVIDIA CI、纯 CUDA wheel。 |

缺少 NPU Kernel 不是直接选择 `SKIP` 的理由。先判断用户可观察的目标语义是否有价值。

### 6.2 JAX 等排除项

以下变更默认按独立行为判断为不适用：

- JAX-only。
- CUDA、CUTLASS、cuBLASLt 或 cuDNN 后端专属实现。
- NCCL 后端细节。
- CUDA Graph 专属实现。
- NVIDIA wheel、CI 和发布元数据。
- NVFP4，而 TENPU 没有明确支持能力。

默认不适用不代表整个 PR 跳过。混合 PR 中的公共 Python API、数学语义、正确性或测试治理行为必须拆出后重新判断。

## 7. 按流程验收

测试必须按照生产流程逐层执行，不能只验证 Prompt 能生成 Markdown。

### 7.1 第一层：Skill 结构检查

检查：

- Skill 目录名与 frontmatter `name` 一致。
- `description` 同时说明做什么和何时使用。
- 所有引用的知识库和 schema 文件存在。
- Skill、知识库和 schema 能以 UTF-8 读取。
- 项目 `.gitignore` 与上游一致；运行工作树按需使用 `.git/info/exclude` 排除 `/output/`。

建议命令：

```bash
git status --short
pytest -q tests/agent/test_nvte_daily_run.py tests/agent/test_nvte_daily_orchestrate.py
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_orchestrate.py --date <YYYY-MM-DD> --dry-run
```

预期：命令返回 0，TENPU 运行时代码和 Git 状态不发生变化。

### 7.2 第二层：输入完整性检查

运行抓取：

```bash
python .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py --date <YYYY-MM-DD>
```

检查：

- `prs.json` 可以解析。
- PR 编号、merge commit、合入时间和文件列表可追溯。
- 当日无 PR 时输出空数组，而不是伪造变更。
- 大 PR 没有因分页或 patch 截断而遗漏文件。
- 本地 NVTE ref 能覆盖被分析的 commit。

### 7.3 第三层：价值判断 Gold Cases

至少准备以下固定案例，并保存人工期望结果：

| 案例 | 预期判断 |
| --- | --- |
| 公共 Python API 签名或默认值变化 | `ADAPT`，通常 LOW/Quick Spec。 |
| JAX-only 变化 | `SKIP + skip-not-applicable`。 |
| CUDA Kernel 但目标语义对 TENPU 有价值 | `TRANSLATE` 或 `REFERENCE`，不能因缺 NPU Kernel 直接 SKIP。 |
| FP8/MXFP8 scale 或 amax 变化 | 高价值、HIGH、Full Spec、人工 Gate。 |
| 混合 PR：JAX + 公共 Python API | 拆成至少两个行为单元，分别 SKIP 和 ADAPT。 |
| TENPU 已有等价实现 | `verify-existing`，必须提供生产路径证据。 |
| NVFP4-only | 默认 SKIP，不得伪支持。 |
| Attention/HCCL/Distributed | HIGH，并给出架构证据和人工审查要求。 |

每个案例检查：

- 行为单元是否拆分正确。
- `value_case` 是否准确。
- `ADAPT/TRANSLATE/REFERENCE/SKIP` 是否符合预期。
- `execution_action` 是否与能力证据一致。
- `tenpu_coverage` 是否基于生产代码，而非同名文件。
- 风险、Spec 深度和人工 Gate 是否一致。
- 判断是否给出正向证据、负向证据和理由。

### 7.4 第四层：完整报告检查

对固定日期完成一次 Agent 研判后，检查：

- 只生成 `output/<D>/report.md`，没有修改 `transformer_engine/`。
- 文末恰好有一个 `daily_te_npu` YAML 块。
- 人类摘要与机器块的计数、决策和状态一致。
- 每个非 `none` 价值判断都有证据。
- 每个 `SKIP` 都有明确的不适用理由。
- HIGH 项具有 Full Spec 和 pending 人工 Gate。
- 无新提交时输出 `no-op`。
- 基线或输入不可靠时输出 `blocked`，不得伪造 commit range。

### 7.5 第五层：人工抽检

测试人员从当日报告中至少抽检：

1. 一个需要适配的行为。
2. 一个被排除的行为。
3. 一个高风险或能力不明的行为；如果当日没有，则使用固定 Gold Case。

人工对照上游 diff、TENPU 生产代码和测试，记录：

- 是否漏掉高价值行为。
- 是否把不适用行为误判为需要适配。
- 是否错误降低高风险项。
- 是否存在没有证据的覆盖结论。

建议关注三个验收指标：

- 价值召回：应适配的行为是否被识别。
- 高风险召回：FP8、Attention、Distributed、Kernel 等是否进入正确 Gate。
- 决策准确：JAX/CUDA-only 等是否正确排除，混合 PR 是否正确拆分。

### 7.6 第六层：报告仓库同步测试

检查：

- `report.md` 和 `prs.json` 先保存在正确的 `output/<D>/` 目录。
- 两个文件被同步到 datasets 的正确日期和分支。
- 同一内容重复同步时不产生新 commit。
- 报告修订后能够形成新的 datasets commit。
- 同步失败不会删除已有本地报告。
- `git status` 不包含待提交的 `output/` 内容，Agent 不直接操作 datasets。

## 8. 验收通过条件

以下条件全部满足后，才认为需求价值识别阶段通过：

1. 能针对指定北京自然日获得完整上游输入。
2. 能输出一份中文价值判断报告。
3. 公共 API 案例被识别为需要适配。
4. JAX-only 案例被正确排除。
5. 混合 PR 能按行为单元拆分。
6. 高风险行为不会被自动降级或直接跳过。
7. 每条决策均有可追溯的上游和 TENPU 证据。
8. 报告未修改 TENPU 运行时代码。
9. 人工抽检没有发现阻塞级漏判或误判。
10. 报告能够同步到开发看护仓库，重复同步幂等且失败可恢复。

## 9. 常见问题

### 为什么不能只看 PR 标题？

标题通常只描述主目标，一个 PR 可能同时包含 Python API、CUDA Kernel、JAX 和测试变化。只看标题容易整项误判。

### 为什么 CUDA Kernel 不一定直接 SKIP？

GPU 实现本身不能直接迁移，但它解决的用户功能或数学语义可能对 TENPU 有价值。此时应选择 `TRANSLATE` 或 `REFERENCE`。

### 为什么同名文件不能证明 TENPU 已经适配？

同名文件可能使用不同后端、不同状态模型或不同调用链。必须证明 TENPU 生产路径实际调用了对应实现，并检查行为是否等价。

### 为什么当前阶段不运行 NPU 测试？

当前阶段只定义需求价值和验收目标。NPU 测试属于后续实施与验证阶段；价值判断只能提出需要的验证层级，不能声称测试已经通过。

### 报告为什么不提交到 TENPU？

日报属于运行数据和过程证据，不是产品代码。本地 `output/` 用于生成和失败恢复，长期历史保存在 datasets，避免污染 TENPU 产品代码提交。
