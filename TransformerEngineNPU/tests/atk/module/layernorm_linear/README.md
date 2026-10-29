# TENPU LayerNormLinear ATK 看护模块

`transformer_engine.pytorch.LayerNormLinear` 是支撑 Megatron
`TELayerNormColumnParallelLinear`（attention pre-norm+QKV 与 MLP
pre-norm+FC1）的融合 pre-norm + 线性投影。基线为 NVTE golden（CUDA 采集），
数值与 API 契约双重比对。

`layer_norm_bias` 仅在 `normalization == "LayerNorm"` 时存在（RMSNorm 没有），
`bias` 仅在线性 bias 开启时存在——参数缺失时梯度槽位为空 tensor，契约记录
哪些参数实际存在。线性 `weight`/`bias` 与 layer-norm `weight`/`bias` 作为
独立输出上报，失败时能精确定位是哪个梯度发散。

覆盖范围：BF16、2D/3D/4D、`normalization`（LayerNorm/RMSNorm）、线性
`bias` 开/关、`zero_centered_gamma`、`eps`；前向输出、输入梯度、layer-norm
与线性参数梯度，以及序列化 API 契约。不含 FP8。

## 精确梯度槽基线（仅 golden 侧）

LayerNormLinear 内的 Linear 与 `te.Linear` 走同一个 cuBLASLt 调用
（`"bias": bias, "grad": True` → `CUBLASLT_EPILOGUE_BGRADB`）：其
`weight_grad` / `bias_grad` 的 GEMM 内累加在大 token 的 fp16/bf16 用例上
粗于 NPU 的 fp32 累加（linear 模块的实测：NPU 对精确 fp64 1.0000、
NVTE 0.8802–0.9689——机制经共享 kernel 路径迁移）。归一化的**参数梯度**（`ln_weight_grad` / `ln_bias_grad`）经泛化
实测并入同一处理：它们是跨全 token 的批量求和，残差同样在
操作数一侧——把设备**自己**算出的 `dln` 喂给 fp64 精确归约，9 例每一槽
都 matched_ratio 1.000000、abs 零超限（小用例舍入到 dtype 后逐位相同）；
而拿掉 `dln` 的那一次 case dtype 舍入，参考立刻复现泛化实测的失败
（0.76–0.96，且只落在近零相消元素上）。`forward_output` 也在第二轮泛化并入（case 65：bf16、`[512,2048]`→96，对
NVTE 挂 0.9620，而设备对精确 fp64 是 **1.000000**）——与 `linear` 模块
all_linear 的 id73/id137（0.9616/0.9851，NPU 1.0000）同款：NVTE 在大 K/窄 N
形状挑的 `transB=T` forward kernel 粗，`forward_reference` 就是为它引入的。
golden 因此对这五个槽位携带精确 fp64 参考，比对变为"设备 vs 精确数学"
——**基线选择而非放宽容差**：

- `forward_output = round_dtype(norm_forward(x)) @ W^T [+ b]`——一次 dtype
  舍入（LN 输出以 case dtype 物化后进 GEMM），bias 按 `apply_bias =
  bias_enabled and not return_bias`（本适配器 `return_bias=False`，故
  bias 恒在输出里）；与槽 4 共用同一份 `ln_out`，不额外算一遍归一化
- `weight_grad = dy^T @ round_dtype(norm_forward(x))`——精确 fp64 的
  归一化前向输出**再舍一次到 case dtype**（镜像模块数据流：LN 输出以
  case dtype 物化后才进 GEMM。与 linear 的关键差异：linear 的 A 操作数
  是输入 x 本身、两侧同型无额外舍入；lnl 的 A 操作数是派生量，漏掉
  这次舍入会在基线里留下 ~√T·|dy|·eps_dtype 的 A 操作数量化偏移，
  近零 wgrad 元素破默认 atol——泛化实测槽 4 集体挂 0.94-0.99 即此）。
  LayerNorm 时 **beta 包含在内**：漏掉会让整个参考偏
  `outer(列和(dy), beta)` 一个量级，对应 linear 的 `apply_bias` 陷阱；
  RMSNorm 无 beta
- `ln_weight_grad = colsum(x̂ ⊙ dln)`、`ln_bias_grad = colsum(dln)`，其中
  `dln = round_dtype(dy @ W)`——LN 反向自身的输入，按模块数据流在 case
  dtype 里物化一次（与槽 4 的 A 操作数同款"一次舍入"契约；判别实验
  在项目目录 `atk_experiments/layernorm_linear/diagnose_slot23_vs_fp64.py`，仓库外）。两条要点：**`zero_centered_gamma`
  不进这两个槽**（仿射尺度与偏移都不进入这两个和，只有 `dX` 带
  `gamma_eff`）；RMSNorm 没有 LN bias 参数，该槽按构造留空
- `bias_grad` = 精确 fp64 列和（本模块 case 6
  [4096,512] bias=True 曾对 NVTE epilogue 输出挂 0.9648）；
  bias 关闭时空槽透传

策略单点在 `run_layernorm_linear_case(exact_baselines=True)`——离线采集器
与在线 GPU 适配器共用同一定义（此前 bgrad 列和只写在采集器里，在线模式
整体旁路——与 linear 前向槽收掉的缺口同款，本次一并收口）。归约定义本身
单点在与判别器共享的 `../_exact_reference.py::LngradAccumulator`（基线与
仲裁器不可漂移）。护栏三层：keyword-only 开关 + device 断言 + 产物 CUDA
驻留断言；`verify_exact_baselines.py` 本机可跑（独立 fp64 autograd 对照 +
四条负向对照 C10–C13），golden 侧端到端由 manifest 的
`device_vs_baseline` 字段留证。**槽位语义变更 ⇒ 离线 golden 必须重采**
（在线模式每跑现算，不受影响）。

**参考的 host 内存契约**（在线跑曾把 golden 侧 worker 打成 SIGKILL，
）：参考全部在 host 上算（`_to_host_fp64` 先搬后升），所以每个 fp64
工作缓冲受 `_CHUNK_BYTES_BUDGET`（256 MiB）约束——`chunk_rows_for` 按一条
路径**同时存活**的列数之和收 chunk，累加器的 scratch 每个累加器只分配一次
（不再每 chunk 一进一出，避免长用例列表下 RSS 抬升）。`norm_forward_reference`
同时支持 `out_dtype`：调用方不需要 fp64 输出时直接落到 case dtype，
**整张量 fp64 形式不再存在**。实测每例参考序列峰值（host 增量）：

| 形状 | 收敛前 | 收敛后 |
|---|---|---|
| T=524288 / 512→2048（case136 形状） | ~7.0 GiB | **1.26 GiB** |
| T=262144 / 2048→2048 | ~15.8 GiB | **1.59 GiB** |

设备侧不受影响：参考不住在 device 上，只有 `to_slot` 的槽位拷贝（与被替换
的槽同尺寸）。剩余项是 `chunked_wgrad` 需要的整张量 case-dtype LN 输出
（[T, in]×2B）——若要再省可把 LN 前向与 wgrad 合成一条融合分块通道。

## 已知的两个角落（泛化专属，基线用例集不受影响）

1. **fp32 + 超长归约的绝对门**（泛化实测 case 133，T=8.9M）。相对门轻松
   过（rtol 9.77e-4 允许 0.84），挂的是 fp32 的
   `max_abs_error_limit = 0.01`：设备自身 8.9M 项 fp32 归约离精确值
   2.7e-5 相对（|dγ|≈859 ⇒ 0.024 绝对），量级 ~`eps32·√T·|dγ|`
   （两点标定：T=4.2e5 预测 3.1e-3/实测 1.28e-3；T=8.9e6 预测=实测
   0.0236）。四参考变体（含用设备自己的 `dln`）**都是 abs 超限 10 个**，
   ⇒ 与实现无关，换任何基线都不动；只能给该输出槽放宽 abs 上限，或按
   GENERATION 口径收敛 `max_length`。**待决**。
2. **粗 dtype × eps 主导的 `dX`**（泛化实测 case 136：x 为
   `N(0,1e-3)` + 0.1%×1000 离群，eps=1e-5）。无离群行 var≈1e-6 ≪ eps ⇒
   `rsigma≈300`，`dX = rsigma×(相消括号)` 把 `dln` 的 ±1 ULP 放大 300 倍，
   绝对噪声底（实测 7–64）超过 bf16 的绝对上限 1.0；设备给定自己的 `dln`
   完全精确（1.000000），**精确基线反而更差**（7 个 abs 超限 vs NVTE 的
   53）⇒ 不覆盖槽 1。生成期已在钩子加约束（`rsigma ≤ 10`，即粗 dtype 下
   `std ≥ 0.1`，实测 0 超限且 4 倍余量；fp32 exempt），见
   `generate_layernorm_linear_hook.py::_MAX_RSIGMA` 的实测表。

两者的共同点：**都不是 TENPU 缺陷，也都不该用"非法形状"理由剔除**
（与 DPA 立的规矩一致）。

## 窄槽 × 归约长度耦合（生成器）

槽 2/3 是 `[in_features]` 一维向量，比较器的名额是 `floor(0.01×in_features)`
个**元素**（< 100 就是 0 个），而它们身上的绝对底噪由**归约长度（token 数）**
决定——两者解耦，于是窄槽的成败由"这一抽里有没有一个近零元素"决定。
实测（fp32、in=64、out=96，最紧 atol 的 dtype）：底噪 ∝ T
（T=1e4 → 3.2e-5，4.2e5 → 1.4e-3），而近零元素的门底就是 fp32 的 atol
1.53e-5 ⇒ 上限取 **8192 行**（≈2.7e-5，落回门内 2 倍内）。
`generate_layernorm_linear_hook.py` 里 `_NARROW_IN_FEATURES=128` /
`_NARROW_MAX_TOKENS=8192`：**`in_features < 128` 时把领先维从左往右折半**
到行数 ≤ 8192（末维不动，因此 dy/ln_w/ln_b/weight/bias 的形状契约不受影响），
并纳入 fail-closed 断言。粗 dtype 的底噪按 √T 收敛，同样受益（~10× 更少的
翻转）。这是**降低翻转概率**而非保证（近零元素总可能存在）；奇数宽度
（17/24/48…压 kernel 非对齐尾块）与大归约压力（宽 in_features 不夹）都保住。

## 文件构成

- `layernorm_linear.yaml`：ATK 设计文件与精度容差。
- `generate_layernorm_linear.py`：整建路线生成器（当前默认）。
- `generate_layernorm_linear_hook.py`：钩子路线生成器（候选；含粗 dtype 的
  eps/数据尺度耦合约束）。
- `function_layernorm_linear.py`：纯设备适配器、契约编码器、精确梯度槽
  基线（golden 侧）、比较器注册。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器（四个梯度槽
  写入精确 fp64 参考，manifest 登记 `device_vs_baseline`）。
- `verify_exact_baselines.py`：基线正确性自检（独立 fp64 autograd 对照 +
  负向对照 C8/C10–C13 + 槽位映射 C14，纯 CPU 可跑）。
- 判别器 `diagnose_slot23_vs_fp64.py` 已迁至项目目录
  `atk_experiments/layernorm_linear/`（仓库外）：槽 2/3（及可选槽 1）对精确 fp64 的四方对照，
  **NPU 本机可跑、无需 GPU/golden**（四方对照：精确舍入的 dln / 不舍入 /
  设备自己的 dln / case dtype 的 x̂，用来把残差归因到操作数）。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case \
  -f tests/atk/module/layernorm_linear/layernorm_linear.yaml \
  -p tests/atk/module/layernorm_linear/generate_layernorm_linear.py
```

预期用例文件：`result/layernorm_linear/json/all_layernorm_linear.json`
（12 例，small/real × bf16）。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/layernorm_linear/collect_nvte_golden.py \
  --cases result/layernorm_linear/json/all_layernorm_linear.json \
  --input-dir ./layernorm_linear_inputs \
  --golden-dir ./layernorm_linear_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/layernorm_linear/json/all_layernorm_linear.json`、`layernorm_linear_inputs` 与 `layernorm_linear_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/layernorm_linear/json/all_layernorm_linear.json \
  -n tests/atk/module/layernorm_linear/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/layernorm_linear/function_layernorm_linear.py \
  --input_data ./layernorm_linear_inputs \
  -tk accuracy
```
