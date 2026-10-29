# NVTE 与 TENPU 架构差异知识库

本知识库保存 NVIDIA TransformerEngine（NVTE）与 TransformerEngineNPU（TENPU）之间可跨任务复用的稳定架构事实，供需求研判(triage)和日报定稿使用。

它回答三个问题：

1. 上游行为在 NVTE 的哪一层产生；
2. TENPU 当前采用什么 NPU 架构承接该行为；
3. 这种架构差异会给 Spec 范围、实现设计和验证带来什么约束。

本文不保存每日待迁 SHA、任务状态和完成结论，也不代替当前代码搜索。

## 模式索引

| 模式 ID | 领域 | 主要搜索词 |
| --- | --- | --- |
| `A-GLOBAL-BACKEND-SHAPE` | 全局后端形态 | backend、CUDA extension、torch_npu、package |
| `A-API-FACADE` | 公共 API | export、signature、default、serialization、availability |
| `A-RECIPE-QUANT-STATE` | Recipe/量化状态 | recipe、Quantizer、scale、amax、state |
| `A-QUANTIZED-TENSOR-GEMM` | Tensor/GEMM | Storage、rowwise、columnwise、swizzle、grouped GEMM |
| `A-MODULE-FUSER` | Module/Fuser | Linear、RMSNorm、fuser、weight lifecycle |
| `A-ATTENTION-CP` | Attention/RoPE/CP | attention、RoPE、THD、mask、Ring、Ulysses |
| `A-DISTRIBUTED-HCCL-FSDP` | Distributed/FSDP | NCCL、HCCL、collective、overlap、FSDP、EP |
| `A-MOE-ROUTER-PERMUTE` | MoE | router、permutation、top-k、token、EP |
| `A-OPTIMIZER-STATE` | Optimizer | FusedAdam、FusedSGD、multi-tensor、master weight |
| `A-NPU-RUNTIME` | 设备运行时 | device、stream、event、RNG、host sync、compile |
| `A-DEBUG-BUILD-TEST` | Debug/Build/Test | debug、test collection、wheel、JAX、CI |
| `A-CAPABILITY-BOUNDARY` | 能力边界 | NVFP4、unsupported、guard、fallback |

Agent 应按行为单元搜索相关模式，不必为每个候选全文加载所有领域；跨领域 requirement 可以记录多个模式 ID。

## 1. 基线与使用边界

当前提炼基线：

- NVTE：`main@8260f49660cbadb78bc52c90449428c51625469d`；
- TENPU 产品代码：`origin/main@0a389a6775868821757cb37c7bdd254f48970ecb`；
- TENPU 工作流代码：`agent_dev@434337837354b2aa41881a01a0932668f71646f1`，仅用于 Skill/Agent 流程，不作为产品能力证据。

使用规则：

- 每次使用记录本文件 git blob SHA，并记录当前 NVTE/TENPU evidence ref。
- 架构模式用于定位和补全检查面，不能仅凭模式命中声明功能等价、已适配或测试通过。
- 当前代码与本文冲突时，以当前生产代码和可执行证据为准，并写入 `knowledge_feedback`。
- Spec 使用架构事实定义影响面、契约和不变量；Design 才决定当前任务如何实现。
- 数值、Kernel、Collective、Alias/Lifetime、性能和稳定性结论仍需对应验证与人工 Gate。
- 不同分支可以复用架构模式，但不得复用 SHA、覆盖状态、测试结论、优先级或游标。

## 2. 阶段使用方式

### 2.1 Triage

对 MEDIUM/HIGH 或涉及架构边界的 requirement，读取命中的架构模式，至少检查：

- 上游行为层和 TENPU 对应抽象是否相同；
- TENPU 生产调用链是否可达；
- 跨模块、状态、布局、Autograd 和通信影响；
- 刻意差异、能力缺口和 unsupported 边界；
- Full Spec 是否覆盖该模式要求的不变量与验证层级。

在 requirement 的证据或 Spec 中记录命中的 `architecture_pattern_id`。未命中已知模式不等于无价值，应标记 `unknown-architecture-pattern` 并提高人工审查。

### 2.2 Implementer

根据 Spec 中的模式 ID 和当前代码重新核对架构，再从实现知识库选择具体 NPU 策略。共享架构事实不能扩大 Spec Scope，也不能直接替代 Design。

### 2.3 Verifier

除逐条验证 Acceptance Criteria 外，按模式检查跨模块不变量。测试绿但遗漏模式要求的状态、布局、全 rank 或生命周期证据时，Spec Compliance 仍为 `not-verified`。

## 3. 全局架构结论

### `A-GLOBAL-BACKEND-SHAPE`

| 维度 | NVTE | TENPU |
| --- | --- | --- |
| 源码形态 | Python facade + C++/CUDA/pybind 扩展 | Python facade + `torch_npu`/CANN/HCCL/NPU 原生算子 |
| 后端选择 | CUDA、cuDNN、cuBLASLt、CUTLASS、NCCL 等多后端 | NPU 原生路径、`torch.distributed`/HCCL、PyTorch fallback 或明确 unsupported |
| 公共包名 | `transformer_engine` | 保持 `transformer_engine` 兼容，但公共导出是选择性子集并含 NPU 扩展 |
| 架构关系 | 上游产品架构 | 目标行为兼容、内部实现独立，不是目录级镜像 |

关键事实：

- NVTE 与 TENPU 源码规模、语言组成和内部层次明显不同。
- 同路径文件不表示实现等价；文件缺失也不自动表示待迁能力。
- GPU 后端专属改动只有在包含独立公共契约、数学语义或 TENPU 能力价值时才进入适配分析。

Spec 影响：

- 必须以调用方可观察行为定义 Scope。
- 必须区分 API/correctness 目标与 fused-performance 目标。
- 必须列出禁止引入的 CUDA、NCCL 和 NVIDIA 专属假设。

失效触发：TENPU 引入原生 C++/NPU 扩展层、公共包结构大幅调整或后端分派重构。

## 4. 公共 API 与包兼容

### `A-API-FACADE`

NVTE 架构：公共 API 由 `transformer_engine.pytorch` 及 module/tensor/recipe/distributed 等子包导出，部分 API 直接依赖 CUDA 扩展。

TENPU 架构：保持 `transformer_engine` 包名与主要调用路径，对支持能力保持兼容 facade；不支持能力通过 availability、guard、稳定异常或不导出表达。

关键差异：

- TENPU 的导出面是 NVTE 子集，不按上游目录机械补齐。
- 部分 CUDA 名称因生态兼容被保留，但内部必须使用 NPU 语义。
- API 存在不代表后端能力存在，不能创建空壳对象或伪支持。

Spec 必查：

- module path、export、signature、default、return、exception；
- pickle/checkpoint 与 `__module__`；
- availability API 和 unsupported reason；
- MegatronAdaptor 等真实调用方。

验证入口：AST/signature、导入与实例化、序列化、错误路径和调用方定向测试。

代码入口：

- NVTE/TENPU：`transformer_engine/pytorch/__init__.py`；
- TENPU：各 `module/`、`tensor/`、`quantization/` 子包 `__init__.py`。

失效触发：公共导出重构、包重命名、deprecation 策略或序列化路径改变。

## 5. Recipe 与量化状态

### `A-RECIPE-QUANT-STATE`

NVTE 架构：Recipe、RecipeState、Quantizer 和底层扩展共同管理 DelayedScaling、CurrentScaling、MXFP8、Block Scaling、NVFP4 等能力。

TENPU 架构：

- `common/recipe/` 将 recipe 拆分为 base/current/delayed/block/MXFP8/MXFP4/W4A8/custom；
- `pytorch/quantization/` 的 Manager、RecipeState、QuantizerRole/Request 构成独立 NPU 状态架构；
- Recipe 通过 state/role/request 生成 Quantizer，再接入 Tensor、Module 和 Distributed。

关键差异：

- 上游 recipe 新字段不能假定落回单文件结构。
- TENPU 存在 HIF8、MXFP4、W4A8 等 NPU 能力边界，不能由 NVFP4 逻辑覆盖。
- Delayed scaling 的 amax history、scale 更新、recompute、distributed reduction、skip/noop 和 checkpoint 是跨层状态契约。

Spec 必查：

- recipe dataclass、默认值、校验、repr 和序列化；
- RecipeState 创建、QuantizerRole/Request 和 Module 使用；
- scale/amax 所有者、更新时机、持久化与分布式归约；
- capability guard 和 unsupported 行为。

验证入口：Recipe/API 测试、scale/amax 状态序列、checkpoint restore、skip/noop、分布式 reduction。

代码入口：

- `transformer_engine/common/recipe/`；
- `transformer_engine/pytorch/quantization/manager.py`；
- `transformer_engine/pytorch/quantization/state/`。

失效触发：RecipeState/Manager 重构、Quantizer role 模型改变、格式能力守卫变化。

## 6. QuantizedTensor、Storage 与 GEMM

### `A-QUANTIZED-TENSOR-GEMM`

NVTE 架构：QuantizedTensor 与 CUDA/C++ 扩展管理 rowwise/columnwise data、scale inverse、swizzle 和 cuBLASLt/CUTLASS GEMM。

TENPU 架构：

- Quantizer、QuantizedTensorStorage、QuantizedTensor wrapper 和具体格式 Storage 分层；
- `general_gemm` 对 QuantizedTensor 委派格式特定 matmul；
- 普通 Tensor 使用 PyTorch matmul，Grouped 路径使用 NPU grouped matmul；
- MXFP8 使用 `npu_dynamic_mx_quant[_with_dual_axis]`、`npu_quant_matmul` 和 E8M0 scale。

关键差异：

- forward、dgrad、wgrad 的 axis、transpose、usage 和 scale layout 不同；
- 上游 preswizzle、cuBLASLt 参数和 GPU layout 不能机械迁移；
- shape/view/transpose/release/save/restore 与 Autograd helper 是跨层契约；
- Grouped GEMM 还涉及 group list device、split/offset、空 group 和 grad accumulation。

Spec 必查：

- rowwise/columnwise storage 和 scale layout；
- Tensor usage、transpose、view、alias 和生命周期；
- forward/backward、bias、gradient accumulation；
- grouped/empty/degenerate 输入；
- correctness 与 fused performance 的目标边界。

验证入口：格式 reference、Storage 元数据、前后向、Grouped 输入矩阵、checkpoint、distributed gather 和性能基线。

代码入口：

- `transformer_engine/pytorch/quantized_tensor.py`；
- `transformer_engine/pytorch/tensor/`；
- `transformer_engine/pytorch/ops/gemm.py`。

失效触发：Storage/wrapper 合并、GEMM dispatcher 重构、新 scale layout 或新量化格式进入生产路径。

## 7. Module 与 Fuser

### `A-MODULE-FUSER`

NVTE 架构：Module 与 fusible ops 通过 CUDA 扩展和多种融合窗口实现 Linear、Norm、MLP、Activation 等能力。

TENPU 架构：

- Linear、GroupedLinear、LayerNormLinear、LayerNormMLP、LayerNorm、RMSNorm 共享 base state、Quantizer、FSDP 和 weight lifecycle；
- Fuser/Sequential 保持组合 API，底层选择 NPU 原生 op、PyTorch fallback 或明确 unsupported；
- RMSNorm 等能力可由 `torch_npu.npu_rms_norm` 承接。

关键差异：

- 上游单模块变化可能影响 `_common.py`、base、多个 fused module、FSDP 和量化路径；
- NPU 原生算子的返回中间量和 backward 保存语义可能与 CUDA 扩展不同；
- Python fallback 可以实现功能，但不能据此宣称 fused 性能等价。

Spec 必查：

- 公共 Module 契约和所有融合调用方；
- weight、workspace、Quantizer 和 FSDP 生命周期；
- 中间输出、extra input/output、Autograd 保存；
- fallback、错误和性能声明边界。

验证入口：Module API、forward/backward、组合 Fuser、量化 recipe 组合、FSDP 和性能基线。

代码入口：

- `transformer_engine/pytorch/module/`；
- `transformer_engine/pytorch/ops/`。

失效触发：Module base state、Fuser contract、weight lifecycle 或 FSDP hook 重构。

## 8. Attention、RoPE 与 Context Parallel

### `A-ATTENTION-CP`

NVTE 架构：DotProductAttention 可在 Unfused、FlashAttention、cuDNN FusedAttention、FlexAttention 等后端间选择，并包含多种 CP helper 和 backend capability 逻辑。

TENPU 架构：

- 核心 Attention 映射到 `torch_npu.npu_fusion_attention`；
- RoPE 使用 `torch_npu.npu_rotary_mul` 等 NPU 能力；
- CP 采用 BaseCPStrategy 与 Ring、Ulysses、KVAllGather 策略，不复刻 NVTE 全部 backend/helper 结构。

关键差异：

- 上游 backend 选择不能逐个复制，必须提炼 mask、layout、dropout、scale 和 backward 等目标语义；
- SBHD/THD、packed QKV/KV、cu_seqlens、sparse mode 和 padding causal 需要 NPU 映射；
- CP 是 Attention、layout、stream 和 collective 的依赖组；
- `.item()`/`.tolist()` 读取长度或控制量可能引入 NPU host sync 并影响 compile/performance。

Spec 必查：

- Q/K/V shape、dtype、layout 和 packed 形式；
- mask、padding、causal、sparse mode、dropout 和 scale；
- RoPE mode、dtype 转换顺序与 Q/K 分离；
- THD、CP 策略、通信顺序、空/非均匀序列；
- forward/backward、全 rank 完成、超时和死锁；
- 不支持的 sliding window、bias 或 backend 行为。

验证入口：PyTorch/FP32 reference、NPU tolerance、THD/CP 多卡矩阵、每 rank 证据、compile 与性能检查。

代码入口：

- `transformer_engine/pytorch/attention/`；
- `transformer_engine/pytorch/attention/dot_product_attention/backends.py`；
- `transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py` 及 strategy 文件；
- `transformer_engine/pytorch/attention/rope.py`。

失效触发：NPU FusionAttention API、CP Strategy、Attention backend dispatcher 或 packed layout 重构。

## 9. Distributed、HCCL 与 FSDP

### `A-DISTRIBUTED-HCCL-FSDP`

NVTE 架构：使用 NCCL、CUDA stream、symmetric memory、NVLink 假设及 CUDA 融合通信/GEMM 路径，并持续演进 FSDP/FSDP2。

TENPU 架构：

- 通用 collective 优先使用 `torch.distributed`；
- NPU 融合算子需要 communicator name 时使用 HCCL comm；
- overlap 使用 `npu_all_gather_base_mm`、`npu_mm_reduce_scatter_base` 等 NPU 算子；
- quantized collective 同时处理 data、scale 和 row/column layout；
- FSDP 与 Module、Storage、weight re-quantize 和 checkpoint 共同构成跨层路径。

关键差异：

- NCCL 名称不能简单替换为 HCCL；拓扑、流、异步 handle 和资源语义不同；
- symmetric memory、NVLink 和 CUDA stream 假设可能没有直接等价能力；
- 当前部分量化 reduce-scatter 只支持 MXFP8，其他格式必须 guard；
- FSDP 变化不能只迁移 hook，需要检查多个 Module 与 Recipe 组合。

Spec 必查：

- process group、world size、rank、拓扑和 collective 顺序；
- async handle、stream wait、buffer 生命周期；
- data/scale/layout 的同步一致性；
- zero-token/empty rank 和不均匀输入；
- FSDP scatter/gather、re-quantize、cache cleanup 和 checkpoint；
- dense fallback、能力 guard 与稳定失败行为。

验证入口：1/2/4/8 卡适用拓扑、每 rank 日志、超时/死锁、量化数据与 scale、Module/FSDP 组合和模型级验证。

代码入口：

- `transformer_engine/pytorch/distributed.py`；
- `transformer_engine/pytorch/ops/fused/overlap.py`；
- `tests/pytorch/test_fsdp.py` 及 distributed tests。

失效触发：collective backend、overlap op、FSDP hook/state 或 HCCL communicator 获取方式重构。

## 10. MoE、Router 与 Permutation

### `A-MOE-ROUTER-PERMUTE`

NVTE 架构：包含 Python API、C++/CUDA/Triton fused router/permutation、EP dispatch/combine 和多种性能路径。

TENPU 架构：优先保持公共数学和索引语义，再选择 NPU token permute/unpermute、NPU fused op、PyTorch fallback 或能力准备；Autograd 契约由 NPU 路径显式承接。

关键差异：

- fused kernel 存在与公共语义存在是两个独立问题；
- routing map、probability、top-k、padding、empty token、inverse permutation 和 index dtype 共同定义行为；
- EP 与 HCCL、zero-token rank、overflow policy 和全 rank 时序组成依赖组；
- Triton 只能 lazy import，并保持无 Triton 环境可导入。

Spec 必查：

- routing 数学、索引和概率语义；
- padding、empty token、single expert/group 等退化输入；
- output layout、inverse permutation、alias 和 Autograd；
- EP collective、zero-token rank、overflow 和死锁；
- 功能 fallback 与 fused 性能目标分离。

验证入口：数学 reference、forward/backward、退化负载、全 rank EP、模型 MoE loss/gradient 和性能基线。

代码入口：

- `transformer_engine/pytorch/router.py`；
- `transformer_engine/pytorch/permutation.py`；
- 对应 MoE tests。

失效触发：routing map 格式、NPU fused op、EP dispatcher 或 Autograd wrapper 重构。

## 11. Optimizer

### `A-OPTIMIZER-STATE`

NVTE 架构：FusedAdam、FusedSGD 和 multi-tensor 更新依赖 CUDA fused kernel，并支持 master weights、capturable 和 overflow/noop 等状态。

TENPU 架构：保持主要 Optimizer API，以 NPU/PyTorch fallback 或 multi-tensor 实现优先保证 correctness；性能实现与功能兼容分开评价。

关键差异：

- Python/NPU fallback 不代表 NVIDIA fused kernel 性能等价；
- optimizer state、master weights、empty param group、mixed dtype、FP8 scale、overflow/noop 和 checkpoint 必须共同检查；
- inplace/out-of-place 更新、alias 和状态生命周期可能影响模型恢复。

Spec 必查：

- 参数和 state 更新公式；
- master weight、mixed dtype、scale 和 overflow；
- empty param group、noop 和异常；
- state_dict/checkpoint；
- correctness 与 performance 分离验收。

验证入口：参考 Optimizer 更新、状态序列、checkpoint restore、空参数和 mixed dtype、模型 loss、性能基线。

代码入口：

- `transformer_engine/pytorch/optimizers/`；
- `tests/pytorch/test_fused_optimizer.py`；
- `tests/pytorch/test_multi_tensor.py`。

失效触发：optimizer state schema、multi-tensor dispatcher、master weight 或 checkpoint 逻辑重构。

## 12. 设备运行时、Stream、Event 与 RNG

### `A-NPU-RUNTIME`

NVTE 架构：依赖 CUDA device、Stream、Event、Generator、memory、lazy init 和 graph/compile 语义。

TENPU 架构：通过 `torch.npu`/`torch_npu` 承接 NPU device、Stream、Event、Generator 和 memory 语义，部分兼容 API 仍保留 CUDA 命名。

关键差异：

- 设备迁移不是 `cuda` 到 `npu` 的文本替换；
- current device、wait/record、Event 生命周期、host sync、Generator state 和 lazy init 均可能不同；
- `.item()`/`.tolist()` 可引入 NPU host sync；
- 兼容 alias 不能掩盖内部 CUDA 假设。

Spec 必查：

- device selection、Stream/Event 创建与等待；
- 对象和资源生命周期；
- RNG state、重现性和 checkpoint；
- host sync、compile/graph capture 和 fallback；
- 保留兼容命名时的真实 NPU 行为。

验证入口：CPU offload、stream/event 顺序、RNG save/restore、compile、host sync 和内存生命周期。

代码入口：

- `transformer_engine/pytorch/cpu_offload.py`；
- `transformer_engine/pytorch/distributed.py`；
- `transformer_engine/pytorch/utils.py`；
- `tests/pytorch/test_cpu_offloading.py`。

失效触发：torch_npu runtime、graph/compile、RNG tracker 或 CPU offload 架构重构。

## 13. Debug、Build 与测试

### `A-DEBUG-BUILD-TEST`

NVTE 架构：Debug、测试和构建同时覆盖 Python、CUDA/C++、JAX、wheel、cuDNN/cuBLASLt/NCCL 等多后端。

TENPU 架构：

- Debug Python 控制面可迁移，但 quantized tensor stats 需适配 NPU Storage/GroupedTensor；
- 构建依赖 torch-npu、CANN、HCCL、triton-ascend 等 NPU 栈；
- `tests/pytorch/` 以 NPU 定向能力为主，不能按上游测试文件数量判断覆盖。

关键差异：

- 上游 CUDA/JAX/build 测试多数不直接适用，但其中独立 API、安全、版本或测试治理行为需拆分研判；
- 测试存在不表示被收集或执行，硬件 skip 不表示通过；
- Debug hook、统计 buffer 和生产 Module 接入需要共同验证。

Spec 必查：

- 上游测试意图和 TENPU 对应验证层级；
- collected、present-not-collected、hardware-gated 或 unknown；
- device/stream/reference/参数矩阵转换；
- 包版本、公开环境变量和用户可观察构建契约。

验证入口：collection、定向 API/行为测试、硬件条件证据、Debug 生产接入和安装/import。

代码入口：

- `transformer_engine/debug/`；
- `tests/pytorch/`；
- Python package/build 配置。

失效触发：测试入口、CI/QA wiring、Debug state 或打包策略重构。

## 14. Unsupported 与能力边界

### `A-CAPABILITY-BOUNDARY`

稳定边界：

- NVFP4 当前明确 unsupported，MXFP4/W4A8 不能冒充 NVFP4；
- CUDA Graph、cuDNN backend、cuBLASLt/CUTLASS、NCCL EP、JAX-only 默认不直接迁移；
- 图捕获、compile、HCCL 融合和特定 NPU op 必须做 runtime capability guard；
- `NotImplementedError` 可以是有效契约，但不能掩盖本应 TRANSLATE 的核心能力。

Spec 必查：

- capability 判断依据是芯片和运行时能力，而不是包安装成功；
- availability 返回值、reason 和异常稳定；
- fallback 是否保持功能、数值和状态语义；
- unsupported 是否有测试，是否会被调用方误判为可用；
- 能力不足时属于 REFERENCE、prepare-capability 还是稳定 SKIP。

失效触发：新增芯片能力、NPU Kernel、运行时 API 或产品路线明确支持此前 unsupported 能力。

## 15. 跨领域不变量

复杂任务应根据命中模式检查以下不变量：

| 类别 | 最低检查 |
| --- | --- |
| API | 路径、签名、默认值、return、exception、serialization、availability |
| 数值 | 公式、dtype、accumulation、scale/amax、NaN/Inf、tolerance 来源 |
| 状态 | 所有者、初始化、更新顺序、skip/noop、保存恢复、释放 |
| Layout | shape、stride/usage、row/column、transpose、packed、view |
| Autograd | 保存张量、forward/backward、gradient accumulation、alias |
| 异步 | stream/event、wait/record、host sync、handle 和资源生命周期 |
| 分布式 | group、rank、world size、拓扑、collective 顺序、全 rank 完成 |
| 能力边界 | device/version guard、fallback、unsupported、稳定失败 |
| 性能 | correctness 与性能目标分离，同 workload 基线，不用 UT 代替 |

不适用项应明确为 N/A 并给出理由，不能因为测试绿而静默遗漏。

## 16. 证据与更新模板

新增或修改架构模式时使用：

```yaml
architecture_pattern_update:
  pattern_id: A-...
  change_summary: ""
  upstream_architecture_evidence:
    ref: ""
    symbols: []
    tests: []
  tenpu_architecture_evidence:
    ref: ""
    symbols: []
    tests: []
  observed_in_tasks: []
  invariants_added_or_changed: []
  invalidation_triggers: []
  replay_reports: []
  confidence: low | medium | high
  human_review_required: false
```

合入稳定架构事实前至少需要：

1. 一个当前 NVTE 架构证据；
2. 一个当前 TENPU 生产代码证据；
3. 一个调用方或测试证据；
4. 对近期历史任务的回放结果；
5. 高风险数值、Kernel、通信和生命周期事实的人工审查。
