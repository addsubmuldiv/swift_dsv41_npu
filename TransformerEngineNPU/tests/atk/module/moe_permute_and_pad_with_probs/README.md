# TENPU moe_permute_and_pad_with_probs ATK 看护模块

`transformer_engine.pytorch.permutation.moe_permute_and_pad_with_probs` 对 MoE token 及其路由概率执行置换，使路由到同一 expert 的 token 连续排列。NVTE CUDA 按各 expert 的 token 数向上对齐到 `align_size` 后分块；TENPU 则以 `drop_and_pad=True` 使用固定 `capacity = sum(target_tokens_per_expert) / num_experts`，逐 expert 填补或截断。adapter 支持在线 GPU/NPU 对比，离线采集也可使用 CPU reference `mp_pad_probs_cpu`。

两侧的激活与概率输出都先通过可微、非原地的转换统一成每个 expert 恰好 `capacity` 行的布局：保留前 `min(tokens_per_expert[e], capacity)` 个路由 token，其余位置置零。然后才用同一份 `dy` / `dy_probs` 计算 loss 和执行 backward。这样 `output_2.pt`（`inp.grad`）对应相同的路由及梯度种子，被截断的 token 和 padding 不贡献梯度。只在 backward 之后转换前向输出，或直接用固定 capacity 的 padding mask 遮罩 CUDA 原始布局，都不能保证这个条件。

覆盖范围：BF16/FP16/FP32（各 50 例，共 150 例）、mask routing-map 类型、top-1/top-2、2–8 个 expert、均匀/不均衡/空 expert 负载、bool mask routing map。比对前向输出、置换后的概率、输入梯度、`row_id_map`、`pad_offsets`、`target_tokens_per_expert` 与序列化 API 契约。P0 矩阵有意排除 output-buffer padding 覆写；**`int32` mask map 会被采样生成**（yaml 的 `routing_dtype` 池是 `{bool, int32}` 且钩子不覆写它，真实参数由 `tokens_per_expert_spec`/`align_size` 派生），NPU 侧需 `int8` 转换，该缺口即 §3#22 实测暴露（已修）。不含 FP8。

## 文件构成

- `mp_pad_probs.yaml`：ATK 设计文件与固定精度容差。当前选用**混合容差标准**（layernorm_linear_mixed_tolerance，标准档 bf16 rtol/atol 0.0156 近零 abs/ULP 兜底）
- `generate_mp_pad_probs_hook.py`：钩子路线生成器（唯一路线）——原生引擎驱动采样，生成器渲染 spec 并派生形状。
- `function_mp_pad_probs.py`：纯设备 adapter、routing-spec 解析器、契约编码器。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行（ATK 已安装或位于 `PYTHONPATH` 中）：

```bash
atk case \
  -f tests/atk/module/moe_permute_and_pad_with_probs/mp_pad_probs.yaml \
  -p tests/atk/module/moe_permute_and_pad_with_probs/generate_mp_pad_probs_hook.py
```

预期用例文件：`result/mp_pad_probs/json/all_mp_pad_probs.json`（150 例（bf16 / fp16 / fp32 各 50））。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
PYTHONPATH=/path/to/ATK:/path/to/this/directory \
python tests/atk/module/moe_permute_and_pad_with_probs/collect_nvte_golden.py \
  --cases result/mp_pad_probs/json/all_mp_pad_probs.json \
  --input-dir ./mp_pad_probs_inputs \
  --golden-dir ./mp_pad_probs_goldens \
  --device cpu
```

`--golden-dir` 必须与 `nodes_npu_vs_golden.yaml` 中 `accuracy_load` 节点的 `output_path`（`./mp_pad_probs_goldens`）一致。

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/mp_pad_probs/json/all_mp_pad_probs.json`、`mp_pad_probs_inputs` 与 `mp_pad_probs_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/mp_pad_probs/json/all_mp_pad_probs.json \
  -n tests/atk/module/moe_permute_and_pad_with_probs/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/moe_permute_and_pad_with_probs/function_mp_pad_probs.py \
  --input_data ./mp_pad_probs_inputs \
  -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并
重跑 NPU 任务。用例集本身的变更（本轮从 curated 矩阵切到采样生成、且 dtype
扩到 bf16/fp16/fp32）同理：旧 golden 与新 case 编号/形状/槽位语义不再对应，
必须重采。过期 golden 表现为契约尺寸不匹配或无法解释的逐用例失败；不要调容差
去掩盖它——先重采集。

本次反向布局修复也改变了梯度基线：在线 GPU/NPU 两侧需使用同一版本的 adapter
和 `_moe_convert.py`；离线比较需重采 golden。输入 schema 和用例生成规则未改变，
原有合法 case JSON 与规范输入可复用。

CPU 回归可在仓库根目录执行：

```bash
python -m pytest tests/atk/test_mp_pad_probs.py tests/atk/test_cpu_baselines.py -q
```

该回归用可微 gather 模拟 NVTE 的原始分块，并执行真实 adapter，检查三种 dtype
下的前向输出、输入/概率梯度和输入不被修改；覆盖不均衡负载、空专家、topK、
截断及 padding。它不依赖 ATK 或设备扩展，不能替代 GPU/NPU kernel 实机验证。
