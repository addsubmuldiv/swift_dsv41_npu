# NVTE 到 TENPU 通用迁移知识库

本知识库用于分析 NVIDIA TransformerEngine 上游 PR 对 TransformerEngineNPU 的迁移价值、适用性、风险和验证要求。规则来自历史迁移实践，但不保存待迁移 SHA、分支完成状态或当日优先级；任何结论都必须重新对照当前 NVTE diff、TENPU 代码和运行环境。

## 目录

- 知识来源与生命周期
- 使用边界
- 证据驱动决策顺序
- 价值类型
- 适用性与行动门禁
- 四类工程决策与 Risk-Based ASDD
- 迁移指纹
- 跨领域不变量
- 依赖关系
- 默认不适用项
- 人工审查与验证
- 报告一致性
- 知识反馈与演进

## 知识来源与生命周期

- 当前规则由 2026 年 6 月迁移分析、2026 年 7 月 PR 研判及对应人工评论归纳而成；月度报告保留为历史证据，本文件是每日 agent 唯一使用的通用规则入口。
- 历史报告中的 PR 编号、SHA、覆盖状态、优先级和迁移完成情况不得进入通用规则；只提炼可跨提交、跨分支复用的行为指纹、门禁、不变量和验证方法。
- 每次规则变更都通过本文件的 git blob SHA 标识版本。日报保存知识版本和命中规则 ID，保证结论可回放。
- 日常运行只产出反馈候选；候选经证据聚合、历史回放和必要的人工审查后，才能修改稳定规则。

## 使用边界

- 所有 TE/TENPU 分支对可复用本文件的行为、可移植性、风险、依赖和验证规则。
- 禁止跨分支复用 commit SHA、覆盖状态、测试结论、优先级、分析游标和兼容基线。
- 知识规则只能辅助筛选，不能替代当前代码搜索、调用链检查和 NPU 能力证据。
- 每次研判记录本文件路径及 git blob SHA，确保结论可追溯到具体知识版本。
- 每日研判不得直接重写本文件，只能输出结构化知识反馈候选。

## 证据驱动决策顺序

1. 描述上游改变的调用方可观察行为，不以标题、目录或代码量代替行为。
2. 用 `functional_layers` 拆分 `python-api`、`python-behavior`、`runtime`、`kernel`、`communication`、`tests`、`docs-build`。
3. 在 TENPU 中检查对应符号、生产调用方和测试，判断 `implemented`、`partial`、`absent`、`npu-divergent` 或 `unknown`。
4. 分开记录 `applicability_evidence.positive` 与 `negative`；同名文件、类或函数不是等价实现证据。
5. 判断唯一主价值、风险因子、依赖边、可实施表面和验收不变量。
6. 选择 `ADAPT/TRANSLATE/REFERENCE/SKIP` 工程决策，再选择当前 `execution_action`；价值高不代表可以立即迁移。

## 价值类型

| `value_case` | 中文标签 | 判断问题 | 常见行动 |
| --- | --- | --- | --- |
| `correctness` | 正确性修复 | 是否修复错误结果、崩溃、状态损坏、死锁、静默精度回退？ | 能完整验证时 `port-now`；数值、通信例外。 |
| `api-parity` | API 兼容 | 是否改变公开签名、默认值、export、dtype、序列化或 Megatron 调用？ | `port-now` 或 `port-python-only`。 |
| `ecosystem-enabler` | 生态能力补齐 | 是否解除 checkpoint、compile、MoE、CP、FSDP 等已有工作流缺口？ | 依赖完整时实施，否则能力准备。 |
| `performance-enabler` | 性能能力前置 | 是否提供现有 NPU 能力可承接的融合、布局或减少同步路径？ | 有能力证据才实施，否则能力准备。 |
| `operability` | 可运维性提升 | 是否改善诊断、错误信息、测试可达性、兼容或构建维护？ | 条件迁移或不适用。 |
| `design-reference` | 方案参考 | 是否只提供未来 NPU/HCCL 方案的接口、算法或守卫模式？ | `prepare-capability` 或 `defer-human-review`。 |
| `none` | 无适配价值 | 是否对当前 TENPU 用户和能力没有可观察收益？ | `skip-not-applicable`。 |

每条非 `none` 记录都必须有 `value_evidence` 和 `decision_rationale`。只选一个主 `value_case`；其他收益写入辅助说明，不重复计数。

## 适用性与行动门禁

| 条件 | `actionability` | 允许的 `execution_action` |
| --- | --- | --- |
| 当前生产路径可达，所有必需功能层可实现，风险已关闭，聚焦测试已定义 | `ready` | `port-now` |
| 独立 Python/API 行为可实现，kernel/通信差异已明确隔离 | `conditional` | `port-python-only` |
| 价值明确，但缺 NPU kernel、HCCL、runtime 或完整依赖 | `blocked` | `prepare-capability` |
| 精度、时序、alias/lifetime、资源或语义需要人工决定 | `blocked` | `defer-human-review` |
| 没有对应产品能力、调用路径或只涉及不支持后端 | `not-applicable` | `skip-not-applicable` |

`execution_action` 只能是上述完整值之一。`design-reference` 只属于 `value_case` 或实现策略，禁止 `DR`、`DR(SNA)` 和括号修正。

## 四类工程决策与 Risk-Based ASDD

| `adaptation_decision` | 判断标准 | 常见 `execution_action` |
| --- | --- | --- |
| `ADAPT` | API、通用逻辑、正确性或数学行为可在 TENPU 直接适配 | `port-now`、`port-python-only`、`verify-existing` |
| `TRANSLATE` | 目标功能/语义必须保留，但 CUDA、NCCL、GPU kernel 或 runtime 实现需转换 | `port-now`、`prepare-capability`、`defer-human-review` |
| `REFERENCE` | 当前不直接实施，只作为 NPU 能力或优化设计参考 | `prepare-capability`、`defer-human-review` |
| `SKIP` | 无 TENPU 公共契约、计算语义、产品能力或验证收益 | `skip-not-applicable` |

当前缺少 NPU kernel/HCCL 能力不是选择 SKIP 的充分条件。先判断是否存在用户可观察的目标语义；存在时通常是 TRANSLATE 或 REFERENCE。

风险评分用于确定 Spec 深度和人工 Gate：Python API +2、数值行为 +3、FP8/MXFP8/HIF8 +5、Attention +5、Distributed +5、MoE +4、Kernel +5、Autograd/RNG +4、Checkpoint +3、CUDA/NPU 语义差异 +3、大 diff +2、correctness fix +4。建议 0-3 LOW、4-8 MEDIUM、>=9 HIGH。

- LOW 使用 Quick Spec，可在验收明确时自动实施。
- MEDIUM 使用 Standard Spec，必须有 Design/Tasks 和 code review。
- HIGH 使用 Full Spec，必须依次通过 spec review、design review 和 code review。

评分不能降低固有高风险门禁。FP8/MXFP8/HIF8、Attention、MoE、Distributed、Kernel、Autograd、RNG、Checkpoint、Memory Layout 和 Collective 默认 HIGH；调整风险必须记录证据与人工理由。

P0 必须同时满足：当前 TENPU 路径可达；用户影响有代码、调用方或复现证据；`actionability: ready`；能够在现有 NPU 能力上完整验证。上游标题含 bug、未来规划价值或只有同名辅助类都不足以成为 P0。

## 迁移指纹

| 规则 ID | 模式与信号 | 默认研判 | 核心检查 |
| --- | --- | --- | --- |
| `K-PY-API` | signature、默认值、export、DType、错误信息、序列化 | 优先检查 Python/API parity | Megatron 调用方、兼容默认值、稳定失败行为。 |
| `K-QT-META` | Quantized Tensor `shape`、`size`、view、storage、dequantize、autograd | 高价值 correctness | row/column storage、正负维、空 storage、前后向。 |
| `K-OFFLOAD-STATE` | CPU offload、extra state、checkpoint、FSDP hook | 高价值 Python/运行时迁移 | stream/event、状态恢复、storage 生命周期。 |
| `K-COMPILE` | `torch.compile`、graph break、opaque object、backend selection | 形成 compile 依赖组 | host sync、动态控制流、fallback shape/dtype。 |
| `K-FP8-STATE` | FP8/MXFP8/HIF8、scale、amax、recipe、quantizer、block scaling | 高价值、高风险 | 数值公式、布局、更新时机、精度基线。 |
| `K-NOOP-STATE` | noop、skip-update、overflow、空 tensor/group | correctness 状态机 | 未更新分支保持哪些 tensor、scale、amax 和 optimizer state。 |
| `K-SWIZZLE-GUARD` | preswizzle、fusion capability、specialized kernel | 先检查正向能力守卫 | 请求条件、后端能力探测、禁用路径。 |
| `K-GROUPED-QUANT` | grouped quant、splits、offsets、varying dims | 完整依赖组 | Python API、分组布局和真实 NPU grouped kernel。 |
| `K-GROUPED-DEGENERATE` | single group、unit probability、empty optional input | 高价值边界场景 | 快慢路径一致性、可选 tensor、autograd 和 alias。 |
| `K-ATTN-ROPE` | attention、RoPE、THD、CP、mask、packed QKV/KV | 条件迁移、高风险 | layout、mask、padding、前后向、collective。 |
| `K-HOST-SYNC` | `.item()`、`.tolist()`、CPU shape/control readback | NPU 性能/compile 候选 | 是否引入同步、是否能保持异步设备语义。 |
| `K-EP-HCCL` | EP、dispatch/combine、top-k、symm mem、all-to-all | API 与通信后端分层 | HCCL 等价能力、overflow policy、全 rank 证据。 |
| `K-ROUTER-PERMUTE` | routing map、probability、top-k、permutation | 先 API/数学行为 | 索引 dtype、概率、输出布局、空 token、autograd。 |
| `K-OPT-STATE` | optimizer state、multi-tensor、master grad、empty param | correctness | 状态更新、空参数、dtype、scale、checkpoint。 |
| `K-MATH-FALLBACK` | 新 fused activation/loss、clamp/scale、参考实现 | API、数学语义与 kernel 分层 | TENPU 路径可达性、参考公式、autograd、dtype/tolerance、fallback 性能契约。 |
| `K-BUFFER-ALIAS` | output/grad buffer、in-place、storage reuse | 性能收益与正确性风险并存 | shape/dtype/device、别名、所有权、生命周期、保存反向张量、空输入。 |
| `K-DIST-WEIGHT` | GTP、DistributedWeight、TP shard、fused distributed module | 架构依赖组，默认人工审查 | 分片规则、collective、checkpoint、autograd、量化状态和模块调用链。 |
| `K-GRAPH-RUNTIME` | graph capture、hook、lazy init、GIL/refcount | 设计参考或条件迁移 | NPU graph 能力、对象生命周期、线程/GIL 边界。 |
| `K-TEST-REACH` | 新测试、skip、QA/CI wiring、collection | operability | 测试是否被收集、执行及受何种硬件门禁。 |
| `K-BUILD-BACKEND` | CMake、wheel、CUDA/cuDNN/cuBLASLt/NCCL/JAX | 默认不适用 | 是否存在独立的公共 API、安全或可移植构建收益。 |
| `K-VERSION-CONTRACT` | 包版本、导出路径、环境变量、弃用和用户文档 | 先判断 TENPU 用户契约 | TENPU 独立版本策略、下游解析逻辑、公开配置和兼容窗口。 |

## 跨领域不变量

### 元数据与退化负载

- `shape == size()`，且 `size(dim) == shape[dim]`；覆盖 2D/3D、正负维、rowwise-only、columnwise-only 和双向 storage。
- 不支持的 view/reshape 必须稳定失败并给出有效错误。
- 单 group、零长度 tensor、空参数组、空 token、全空 optional tensor 和单位 probability 是一等测试场景。
- 只有当前生产调用链可达这些场景，才可宣称现存 correctness 债务；仅辅助类保留旧逻辑只能说明 API parity 候选。

### 状态控制与异步语义

- noop/skip/overflow 分支必须明确哪些输入、量化数据、scale、amax、optimizer state 和路由输出保持不变。
- 不得把设备 kernel 内 flag 检查机械替换为 Python `.item()`；这可能引入 NPU host sync 并破坏 compile/graph capture。
- 可选优化只有在“用户请求”且“运行时能力已证实”时启用；缺能力时走确定性禁用或 fallback。

### compile、attention 与分布式

- compile 检查对象可追踪性、backend 选择、packed 输入声明、fallback 以及设备到主机读回。
- attention/RoPE 检查 shape、dtype、layout、mask、padding、THD/CP 前后向和 tolerance 来源。
- HCCL/EP/FSDP 检查 process group、拓扑、`world_size`、collective 顺序、超时和每个 rank 的完成证据。

### 融合数学、缓冲区与架构能力

- 上游 fused kernel 修复只有在 TENPU 存在同一生产路径时才构成现存 correctness 债务；路径缺失时可评估独立 Python API 或数学 fallback，但不得选择完整 `port-now`。
- Python fallback 必须明确它是功能实现还是仅测试 oracle，并验证公式、边界、autograd、dtype 和 tolerance；不得用慢速 fallback 宣称达到 fused kernel 的性能收益。
- 新增 output/grad buffer、in-place 或 storage reuse 时，除了签名还要验证别名、所有权、生命周期和反向保存语义；这些证据不完整时默认人工审查。
- GTP/DistributedWeight 等架构变更必须先判断 TENPU 是否采用同一抽象，再按分片、通信、checkpoint、autograd、量化和融合模块组成依赖组；不能因为存在同名 TP 功能就直接迁移。
- C++/pybind 的 GIL、refcount 或资源生命周期修复只有在 TENPU 共享对应绑定架构且调用链可达时才可视为直接适用，否则属于设计参考或能力准备。
- 版本号、环境变量和文档不能仅因位于 build/docs 桶就跳过；若改变 TENPU 用户可观察的包契约、配置或兼容性，则按独立行为研判。

## 依赖关系

依赖组内每条边必须标记：

- `requires`：缺少前置就无法实现或验证。
- `unblocks`：提供后续能力，但不要求同一补丁合入。
- `follow-up`：当前可独立迁移，后续补性能或覆盖。
- `reference-only`：技术主题相关，但不是 TENPU 实施依赖。

常见依赖组包括 `quantized-tensor-metadata`、`grouped-quantization`、`attention-compile`、`thd-context-parallel`、`expert-parallelism`、`joint-fusion-grouped-mlp`、`router-moe`、`gtp-distributed-weight` 和 `graph-runtime`。不能仅凭目录、关键词或同属 MoE 就组成实施批次。

## 默认不适用项

纯 NVFP4、CUDA kernel、CUTLASS、cuBLASLt filter、cuDNN backend、NCCL backend、CUDA graph 细节、CUDA wheel、JAX-only 和 NVIDIA CI metadata 默认 `not-applicable`。若同一 diff 含独立且可验证的 Python API、正确性、安全或测试治理行为，只拆出该行为评估。

TENPU 缺失的 fused/extension 文件标为 `absent`；没有明确产品路线时使用 `skip-not-applicable`，有明确路线但缺能力时使用 `prepare-capability`。不得为追随上游目录结构而新建不成熟实现，也不得伪支持 NVFP4。

## 人工审查与验证

以下默认人工审查：FP8/MXFP8/HIF8 scale/amax/recipe/storage；NPU quantize/GEMM layout；attention/RoPE/THD/CP；HCCL/FSDP/EP；GTP/DistributedWeight；optimizer state、router probability、permutation；别名或生命周期不明确的输出/梯度 buffer；无法证明的 CUDA/C++/Triton 到 NPU kernel 映射。

| 层级 | 最低证据 |
| --- | --- |
| static | 导入、export、signature、符号、语法、unsupported guard 的命令与结果。 |
| behavioral | 确定性 API、shape、dtype、value、错误路径和不变量；记录 tolerance 来源。 |
| hardware-gated | NPU、HCCL、多卡、图、FP8、kernel；记录型号、卡数、命令和每个 rank 结果。 |
| performance | 相同 workload、shape、dtype 和测量方法的改前/改后数据；不能替代正确性。 |

上游测试文件存在不等于测试已执行。记录 `collected`、`present-not-collected`、`hardware-gated` 或 `unknown`；未接入 pytest/QA 的测试只可作为测试设计参考。

## 报告一致性

- 日报、周报、月报必须记录范围时区、cutoff、完整性、提交全集和排除原因。
- `upstream_old_sha: null` 时不得输出可推进的 `needs-port`；研究性分类保持 blocked 且禁止推进 state。
- 修改逐条工程决策或执行动作后，同步更新覆盖、风险、Spec、计数、摘要和机器状态。
- 覆盖结论绑定 `tenpu_evidence_ref`；知识版本绑定本文件 git blob SHA。
- `calendar-audit` 永不推进分析游标或兼容基线。
- `analysis-only` 完整分析后可以推进 `last_analyzed_upstream_sha`，但不得推进 `compatibility_baseline_sha`。

## 知识反馈与演进

每日研判必须输出 `knowledge_feedback.candidates`，即使为空。候选类型：

- `new-fingerprint`：现有规则无法覆盖的通用变更模式。
- `false-positive`：规则反复把不适用提交识别为可迁移。
- `false-negative`：规则漏掉当前 TENPU 可达的高价值行为。
- `rule-conflict`：多个规则给出矛盾决策或风险。
- `validation-gap`：有价值行为缺少可复用验收不变量或测试层级。

候选至少包含 `kind`、`rule_ids`、`fingerprint`、`upstream_evidence`、`tenpu_evidence`、`observed_outcome`、`proposed_change`、`confidence`、`risk` 和 `human_review_required`。

知识维护流程：

1. 每日 agent 只记录候选，不直接修改知识库。
2. 聚合重复候选，区分一次性 commit 细节与可复用规则。
3. 用至少一个上游 diff、一个 TENPU 证据和一个可执行验证场景证明候选；高风险数值/通信规则必须人工审查。
4. 用近期历史报告做回放，确认新规则减少误报或漏报且不破坏单值 AD、分支隔离和状态推进门禁。
5. 合入通用规则，更新或新增稳定 `规则 ID`；不写入每日 SHA、完成状态和临时优先级。
