# TENPU GroupedLinear ATK 看护模块

看护 `transformer_engine.pytorch.GroupedLinear`——Megatron MoE 专家层
（`TEGroupedLinear`）背后的分组 GEMM。基线为 NVTE golden（CUDA 采集），
数值与 API 契约双重比对。

覆盖范围：BF16/FP16/FP32（各 50 例，共 150 例）、`num_gemms` 1/2/4/8/16（含 KIMI3 单专家/EP 场景）、
四类 token 切分（均匀 / 小幅不均 / 路由坍缩 / 欠载尾专家）、`bias` 开/关、
2D 输入，全部由 yaml 采样池驱动；前向输出、输入梯度、拼接后的逐 expert
weight/bias 梯度，以及序列化 API 契约。不含 FP8，不含 TP/sequence-parallel
模式。

**注意 arity 随用例变化**：每例的输入槽数 = `2 × num_gemms + 5`
（g=1 → 7 槽，g=8 → 21 槽，g=16 → 37 槽），yaml 声明的是最大 arity。
读 case json 或写下游工具时按尾部三个 attr 反推布局，不要假设固定槽数。

## 文件构成

- `grouped_linear.yaml`：ATK 设计文件与精度容差，当前选用**混合容差标准**
  （`grouped_linear_mixed_tolerance`）。
- `generate_grouped_linear_hook.py`：钩子路线生成器（唯一路线）——原生引擎
  驱动采样，生成器把声明的最大 arity 压缩成 2g+5 槽，并派生 input/weight/
  bias 形状、渲染 `m_splits`（由采样的切分风格 + token 数确定性生成）。
- `function_grouped_linear.py`：纯设备适配器、契约编码器、比较器注册，以及
  梯度槽的精确 fp64 基线（golden 侧，见第 4 节）。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器（含基线路径与
  manifest 记录）。
- `verify_exact_baselines.py`：基线负向对照（C1-C10）。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case \
  -f tests/atk/module/grouped_linear/grouped_linear.yaml \
  -p tests/atk/module/grouped_linear/generate_grouped_linear_hook.py
```

预期用例文件：`result/grouped_linear/json/all_grouped_linear.json`
（默认 `dtype_numbers: 12` × bf16 单 dtype ⇒ 12 例）。用例集由 yaml 采样池与
`-s <seed>`（CLI 默认 1234）决定：同 yaml + 同 seed + 同 ATK 版本 ⇒ 同用例。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/grouped_linear/collect_nvte_golden.py \
  --cases result/grouped_linear/json/all_grouped_linear.json \
  --input-dir ./grouped_linear_inputs \
  --golden-dir ./grouped_linear_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/grouped_linear/json/all_grouped_linear.json`、`grouped_linear_inputs` 与 `grouped_linear_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/grouped_linear/json/all_grouped_linear.json \
  -n tests/atk/module/grouped_linear/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/grouped_linear/function_grouped_linear.py \
  --input_data ./grouped_linear_inputs \
  -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并
重跑 NPU 任务。用例集本身的变更（如本轮从 14 例 curated 矩阵切到采样生成的
12 例）同理：旧 golden 与新 case 编号/形状/arity 不再对应，必须重采。过期
golden 表现为契约尺寸不匹配或无法解释的逐用例失败；不要调容差去掩盖它——
先重采集。

## 4. 梯度槽的精确 fp64 基线（摘要）

golden 侧对 output_2（及 bias 开时的 output_3）携带精确 fp64 逐 expert 参考
而非 NVTE kernel 输出——**基线选择，不是放宽容差**。口径细节在适配器
`function_grouped_linear.py` 的 `exact_baselines`/`_apply_exact_baselines`
docstring。判别器已迁至项目目录 `atk_experiments/grouped_linear/`。
