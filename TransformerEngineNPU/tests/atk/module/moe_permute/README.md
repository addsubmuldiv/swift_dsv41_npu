# TENPU moe_permute ATK 看护模块（no-CPU-reference 模板）

`transformer_engine.pytorch.permutation.moe_permute` 对 MoE token 执行置换，使路由到同一 expert 的 token 连续排列，并返回置换后的激活与供逆操作 `moe_unpermute` 使用的 `row_id_map`。本模块遵循 `../rmsnorm/` 确立的 **no-CPU-reference** 模板。

覆盖范围：BF16/FP16/FP32（各 50 例，共 150 例）、mask 与 index 两种 map 类型、top-1/top-2、2–8 个 expert、均匀/不均衡/空 expert 负载；mask 采用 `bool` routing map，另含一个遗留的 `int32`-attr 变体（NPU 侧构建为 `int8`，NVTE golden 侧构建为 `int32`）；index 采用 `int32` routing dtype。比对前向输出、输入梯度、`row_id_map` 与序列化 API 契约。不含 FP8，不含 dropped-token / output-buffer-padding / `max_token_num` 变体（已从 P0 矩阵排除）。

## 文件构成

- `moe_permute.yaml`：ATK 设计文件与固定精度容差。当前选用**混合容差标准**（layernorm_linear_mixed_tolerance，标准档 bf16 rtol/atol 0.0156 近零 abs/ULP 兜底）
- `generate_moe_permute_hook.py`：钩子路线生成器（唯一路线）——原生引擎驱动采样，生成器渲染 spec 并派生形状。
- `function_moe_permute.py`：纯设备 adapter、routing-spec 解析器、契约编码器。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行（ATK 已安装或位于 `PYTHONPATH` 中）：

```bash
atk case \
  -f tests/atk/module/moe_permute/moe_permute.yaml \
  -p tests/atk/module/moe_permute/generate_moe_permute_hook.py
```

预期用例文件：`result/moe_permute/json/all_moe_permute.json`（150 例（bf16 / fp16 / fp32 各 50））。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
PYTHONPATH=/path/to/ATK:/path/to/this/directory \
python tests/atk/module/moe_permute/collect_nvte_golden.py \
  --cases result/moe_permute/json/all_moe_permute.json \
  --input-dir ./mp_inputs \
  --golden-dir ./mp_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/moe_permute/json/all_moe_permute.json`、`mp_inputs` 与 `mp_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/moe_permute/json/all_moe_permute.json \
  -n tests/atk/module/moe_permute/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/moe_permute/function_moe_permute.py \
  --input_data ./mp_inputs \
  -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并
重跑 NPU 任务。用例集本身的变更（本轮从 curated 矩阵切到采样生成、且 dtype
扩到 bf16/fp16/fp32）同理：旧 golden 与新 case 编号/形状/槽位语义不再对应，
必须重采。过期 golden 表现为契约尺寸不匹配或无法解释的逐用例失败；不要调容差
去掩盖它——先重采集。
