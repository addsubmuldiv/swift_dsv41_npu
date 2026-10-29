# TENPU fused-optimizer ATK 看护模块

看护 `transformer_engine.pytorch.optimizers.FusedAdam` / `FusedSGD`。基线为
NVTE golden（CUDA 采集），数值与 API 契约双重比对。

optimizer 看护比较的是 N 个 step 之后的最终参数与 optimizer state（Adam 的
`exp_avg`/`exp_avg_sq`、SGD 的 `momentum_buffer`，以及 step 计数与序列化
契约）。用例输入：单个 param group、fp32 master 参数 + fp32 梯度（跨 step
固定）、超参作为 attrs（`optim_type`、`lr`、`weight_decay`、`steps`、Adam
的 `b0`/`b1`/`eps`、SGD 的 `momentum`/`nesterov`）。

## 文件构成

- `fused_optimizer.yaml`：ATK 设计文件与精度容差。
- `generate_fused_optimizer.py`：整建路线生成器（当前默认）。
- `generate_fused_optimizer_hook.py`：钩子路线生成器（候选）。
- `function_fused_optimizer.py`：纯设备适配器、契约编码器、比较器注册。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与 canonical 输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case -f tests/atk/module/fused_optimizer/fused_optimizer.yaml \
  -p tests/atk/module/fused_optimizer/generate_fused_optimizer.py
```

## 2. 采集 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/fused_optimizer/collect_nvte_golden.py \
  --cases result/fused_optimizer/json/all_fused_optimizer.json \
  --input-dir ./fused_optimizer_inputs --golden-dir ./fused_optimizer_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/fused_optimizer/json/all_fused_optimizer.json`、`fused_optimizer_inputs` 与 `fused_optimizer_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/fused_optimizer/json/all_fused_optimizer.json \
  -n tests/atk/module/fused_optimizer/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/fused_optimizer/function_fused_optimizer.py \
  --input_data ./fused_optimizer_inputs -tk accuracy
```
