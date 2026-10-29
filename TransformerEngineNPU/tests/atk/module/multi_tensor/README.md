# TENPU multi-tensor ATK 看护模块

看护 `transformer_engine.pytorch.optimizers` 的 multi-tensor 辅助函数：
`multi_tensor_scale`、`multi_tensor_scale_tensor`、`multi_tensor_l2norm`。
基线为 NVTE golden（CUDA 采集），数值与 API 契约双重比对。

输出槽位：scale 用例为三个输出 tensor；l2norm 用例为 `total_norm`（fp32）
与 `per_tensor_norms`（`per_tensor=False` 时为空）；另有 `overflow_flag`
（int32，精确比对）与序列化契约。用例输入：3 个 fp32 tensor + attrs
（`op`、`scale`、`per_tensor`、`chunk_size`）。

## 文件构成

- `multi_tensor.yaml`：ATK 设计文件与精度容差。
- `generate_multi_tensor.py`：整建路线生成器（当前默认）。
- `generate_multi_tensor_hook.py`：钩子路线生成器（候选）。
- `function_multi_tensor.py`：纯设备适配器、契约编码器、比较器注册。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与 canonical 输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case -f tests/atk/module/multi_tensor/multi_tensor.yaml \
  -p tests/atk/module/multi_tensor/generate_multi_tensor.py
```

## 2. 采集 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/multi_tensor/collect_nvte_golden.py \
  --cases result/multi_tensor/json/all_multi_tensor.json \
  --input-dir ./multi_tensor_inputs --golden-dir ./multi_tensor_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/multi_tensor/json/all_multi_tensor.json`、`multi_tensor_inputs` 与 `multi_tensor_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/multi_tensor/json/all_multi_tensor.json \
  -n tests/atk/module/multi_tensor/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/multi_tensor/function_multi_tensor.py \
  --input_data ./multi_tensor_inputs -tk accuracy
```
