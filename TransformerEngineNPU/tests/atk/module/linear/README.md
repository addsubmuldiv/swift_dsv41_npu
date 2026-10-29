# TENPU Linear ATK 看护模块

针对 `transformer_engine.pytorch.Linear`（TELinear / Column / RowParallel 背后的
GEMM 层）的 ATK 精度看护。基线为 NVTE golden（CUDA 采集），数值与 API 契约
双重比对。**其中 `output_0`（forward_output）、`output_3`（weight_grad）与
`output_4`（bias_grad）三个槽位的基线是精确 fp64 参考，不是 NVTE 的 kernel
输出**，理由见第 4 节。

覆盖范围：BF16、2D/3D/4D 输入、bias 开/关、`return_bias` /
`consume_returned_bias`、`parameters_split`（无 / tuple2 / dict2）；前向输出、
返回的 bias、输入梯度、拼接的 weight/bias 梯度，以及序列化 API 契约。

## 文件构成

- `linear.yaml`：ATK 设计文件与精度容差。
- `generate_linear.py`：整建路线生成器（当前默认）。
- `generate_linear_hook.py`：钩子路线生成器（候选）。
- `function_linear.py`：纯设备适配器、契约编码器、比较器注册（含 golden 侧
  精确基线策略与护栏）。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `diagnose_slot_vs_fp64.py`：逐槽位判「哪一侧是偏差方」的判别器（设备无关
  部分可离线跑）。
- `verify_exact_baselines.py`：精确基线的负向对照，防「看护变瞎」。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。
- `../_exact_reference.py`：三个槽位共用的 fp64 参考实现（本模块与
  `diagnose_slot_vs_fp64.py` 同源，见第 4 节）。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行（ATK 已安装，
安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case \
  -f tests/atk/module/linear/linear.yaml \
  -p tests/atk/module/linear/generate_linear.py
```

## 2. 采集 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/linear/collect_nvte_golden.py \
  --cases result/linear/json/all_linear.json \
  --input-dir ./linear_inputs \
  --golden-dir ./linear_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/linear/json/all_linear.json`、`linear_inputs` 与 `linear_goldens`
三样拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。
`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/linear/json/all_linear.json \
  -n tests/atk/module/linear/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/linear/function_linear.py \
  --input_data ./linear_inputs \
  -tk accuracy
```

## 4. 槽 0/3/4 的精确基线（为什么 golden 不是 NVTE kernel 输出）

NVTE 的 `weight_grad` 与 `bias_grad` 出自**同一个** cuBLASLt 调用
（`"bias": bias, "grad": True` → `CUBLASLT_EPILOGUE_BGRADB`，见 NVTE
`pytorch/module/linear.py` 的 `wgrad_gemm_kwargs`）。该 kernel 在 token 数很大时
累加精度低于 NPU 的 fp32 累加。`forward_output` 是后加进来的：泛化 150 例
（hook 路线 + 三 dtype）最后只剩两条红，都挂在它上面。拿精确 fp64 当裁判，
同一份脚本在两侧各跑一次：

| 槽位 | NPU 侧 | GPU/NVTE 侧 |
|---|---|---|
| input_grad | 1.0000 | 1.0000 |
| **weight_grad** | **1.0000** | **0.8802 ~ 0.9689** |
| **bias_grad** | **1.0000** | **0.9033 ~ 1.0000** |
| **forward_output**（仅 `id73` / `id137` 两条） | **1.0000** | **0.9616 / 0.9851** |

后两行的 0.9616/0.9851 与平台报告里 `output_0` 的 matched_ratio 0.9622/0.9852
基本逐位吻合。**偏差在 golden 侧，TENPU 是更准的一方**。这与 `REPORT atk 3#17`
是同族问题的扩大版（3#17 只覆盖了 db，后来证明 wgrad 主输出也受影响，且
`bias_enabled=False` 的用例同样挂，说明不只是 epilogue）。

**forward 的机制能直接看出来**：`id137` 的 forward 与 input_grad 是**同一个 GEMM
shape**（M=393216 N=2048 K=2048，同为 fp16），只有转置布局不同（`x @ W^T` 对
`dy @ W`），而两侧的 fp32 累加特征值 `med_rel` 也不同——forward 2.169e-04、
input_grad 1.692e-04（后者正是 NPU 也有的「全程 fp32 累加、仅末位舍一次」特征）。
所以是 cuBLASLt 为 `transB=T` 的 forward 在那些 shape 上选到的 kernel 偏粗，
不是归约长度本身；NVTE 干净的那些 shape 上，替换是实测 no-op。

**处理口径**：基线选择，不是放宽容差——真 bug（标度错、轴错、槽位互换、forward
漏了 bias）对精确参考仍然必红（`verify_exact_baselines.py` 的 C5/C5b/C6/C6b
就是这四条负向对照）。策略只有一处实现
（`function_linear.run_linear_case(..., exact_baselines=True)`），离线采集器与
在线 GPU 侧适配器共用，因此两条通路按构造同源。

**槽 0 特有的坑**：forward 参考必须按 `apply_bias = bias_enabled and not
return_bias`（NVTE `Linear.apply_bias`）决定加不加 bias。照抄 `x @ W^T + b` 会给
每一个 `return_bias=True` 的用例**把 golden 本身写错**一个 bias 量级（150 例那种
配置下约 0.5），然后把账算到设备头上——`id73` 正是这种用例。判别器里对应的真值
表见 `verify_exact_baselines.py` 的 C7c/C7d。

**两个必须知道的约束**：

- **替换只发生在 golden 侧**。`_golden_side_guard` 会拒绝在非 CUDA 环境合成基线
  ——如果被测侧也被替换，比较就变成「参考比参考」，永远全绿。判别器
  `diagnose_slot_vs_fp64.py` 因此显式传 `exact_baselines=False`（它判的是
  设备的**原始**输出），`verify_exact_baselines.py` 的 C1/C2 专门看护这一点。
- **改完必须重采 golden**：槽 0/3/4 的语义变了，旧 golden 不可复用。只重采而忘记
  同步拷贝/挂载 `linear_goldens`，离线任务会拿旧 golden 比新本地产物 → 这几个槽
  大面积红，**看起来像 NPU 回归**；先查是不是 golden 没同步。在线（ATK 平台
  GPU 侧）golden 每跑现算，不需要重采——重跑那两条红用例即可。

采集器会把基线的来龙去脉写进 `linear_manifest.json`（`schema_version: 3`，
`forward_slot_baseline` / `grad_slot_baseline` 两块 + 逐例 `slot_baseline` /
`device_vs_baseline`），其中 `device_vs_baseline` 记录「NVTE 自己的输出离基线有
多远」，是上表的自证。

跑一遍负向对照：

```bash
# 设备无关项（NPU 主机或任何装了 ATK 的机器）
python tests/atk/module/linear/verify_exact_baselines.py
# 加上 golden 侧的三项（CUDA 主机）
python tests/atk/module/linear/verify_exact_baselines.py --device cuda
```
