# TENPU 可融合激活 ATK 看护模块

看护 `transformer_engine.pytorch.ops` 中的可融合激活算子：**GELU、SiLU、
ReLU**（一元）、**GEGLU、ReGLU、SwiGLU**（gated）以及逐行后缩放的
fused-MoE 变体 **ScaledSwiGLU、ScaledClampedQGeGLU**。基线为 NVTE golden
（CUDA 采集），数值与 API 契约双重比对。

激活算子均为无参数算子，因此输出槽位为 `forward_output, input_grad,
api_contract`，契约记录算子名及输入/输出的 shape 与 dtype。gated 算子把
最后一维减半：GEGLU/ReGLU/SwiGLU（及 gated 缩放变体）沿最后一维把输入
切成 gate/linear 两半，因此这些用例要求最后一维为偶数，且生成器发出
shape 与算子输出一致的 `dy`。缩放变体接收逐行 scale 额外输入：
`ScaledSwiGLU`、`ScaledClampedQGeGLU`、`ScaledSReLU` 把 gated/一元输出乘以
shape 为 `x.shape[:-1]` 的 scale tensor（即 Megatron fused-MoE 的 "post-multiply
probs" 缩放）。它们不要求 FP8——scale 是普通 tensor，看护以 BF16 运行。
每个用例都发出统一的 4 输入布局 `(x, dy, scale, op)`；非缩放用例携带一个
被适配器忽略的占位 scale，以保持用例元数固定。

覆盖范围：BF16、一元（GELU/SiLU/ReLU）与 gated（GEGLU/ReGLU/SwiGLU）激活
外加缩放 MoE 变体（`ScaledSwiGLU`、`ScaledSReLU`、`ScaledClampedQGeGLU`）、
前向输出与输入梯度，以及序列化 API 契约。

## 文件构成

- `fused_activation.yaml`：ATK 设计文件与精度容差。
- `generate_fused_activation.py`：整建路线生成器（当前默认）。
- `generate_fused_activation_hook.py`：钩子路线生成器（候选）。
- `function_fused_activation.py`：纯设备适配器、契约编码器、allclose 比较器。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case \
  -f tests/atk/module/fused_activation/fused_activation.yaml \
  -p tests/atk/module/fused_activation/generate_fused_activation.py
```

预期用例文件：`result/fused_activation/json/all_fused_activation.json`
（20 例，small/real × bf16）。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/fused_activation/collect_nvte_golden.py \
  --cases result/fused_activation/json/all_fused_activation.json \
  --input-dir ./fused_activation_inputs \
  --golden-dir ./fused_activation_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/fused_activation/json/all_fused_activation.json`、`fused_activation_inputs` 与 `fused_activation_goldens` 拷贝或挂载进
NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对
路径。

```bash
atk task \
  -c result/fused_activation/json/all_fused_activation.json \
  -n tests/atk/module/fused_activation/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/fused_activation/function_fused_activation.py \
  --input_data ./fused_activation_inputs \
  -tk accuracy
```
