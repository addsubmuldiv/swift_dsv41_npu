# TENPU moe_unpermute ATK 看护模块（先 permute）

`transformer_engine.pytorch.permutation.moe_unpermute` 是 permute 类 API 的逆操作：恢复原始 token 顺序（可选地把路由副本与其路由概率合并，并去除逐 expert padding）。每个 ATK 用例携带与之匹配的 permute API 的**输入**（`inp`、permute 变体接收的 `probs`（如适用）以及紧凑路由属性），adapter 先运行该 permute API，再按下述规则调用 `moe_unpermute`。

覆盖范围（由 yaml 采样池驱动）：BF16/FP16/FP32（各 50 例，共 150 例）、2–128 个
expert、topk 1/2/4/8（**`map_type == index` 的用例被钩子强制 `topk = 1`**——否则
unpermute 侧会拿 `row_id_map.size(0)` 当 token 数）、均匀/不均衡/坍缩/空 expert
负载；三种模式由采样分派。

按 permute API 形态划分，覆盖以下分支：

1. **`permute`**（`map_type="index"` 或 `"mask"`，不合并）：`moe_permute` -> `moe_unpermute`。
2. **`with_probs`**（mask，合并）：`moe_permute_with_probs` -> `moe_unpermute`，把原始的 `[num_tokens, num_experts]` 路由概率作为 `merging_probs` 送入。
3. **`pad_with_probs`**（mask，合并 + dropandpad）：`moe_permute_and_pad_with_probs` -> `moe_unpermute`，携带 `pad_offsets` 与 `merging_probs`。

`permute` 和 `with_probs` 模式直接传递 permute 输出。`pad_with_probs` 模式在 NPU 和 CPU 基线上均使用每个 expert 固定 `capacity` 行的布局：仅保留各 expert 前 `min(tokens_per_expert[e], capacity)` 个有效 token，其余槽位置零，再送入 `moe_unpermute`。置零采用非原地、可微转换，避免补位槽中的重复 token 污染前向合并和 `inp.grad`。

pad 模式的 permute 始终使用 `drop_and_pad=True`；因此当其返回 `pad_offsets=None` 时，adapter 补充 `[num_experts]` 的 int64 零张量，使 unpermute 仍选择同一模式。即使 `align_size=1`，不均衡专家负载也可能发生 capacity 截断和补位，不能据此切换为非 padding 模式。

此调整改变 pad 模式的输出、梯度和 `pad_offsets_shape` 契约；离线比较必须重新采集 golden，在线比较应在两侧部署同一版本 adapter。CPU 回归测试可用 `python -m pytest tests/atk/test_moe_unpermute_pad.py` 运行；其中 NPU 路径使用 CPU 替身验证 adapter，不代替 NPU 实机验证。

`merging_probs` 在**两个后端**都必须是原始的 2-D `[num_tokens, num_experts]` 概率（NVTE 读取 `merging_probs.stride(1)`；permute API 返回的 1-D `permuted_probs` 只是可选中间产物，不会送入 `moe_unpermute`）。

## 文件构成

- `moe_unpermute.yaml`：ATK 设计文件（三种模式的联合输入 schema）与固定精度容差。
- `generate_moe_unpermute_hook.py`：钩子路线生成器（唯一路线）——原生引擎驱动采样，生成器渲染 spec 并派生形状。
- `function_moe_unpermute.py`：纯设备 adapter、先 permute 的执行器、routing-spec 解析器、契约编码器。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器（pad_with_probs 模式自动运行在 CPU 基线上）。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行（ATK 已安装或位于 `PYTHONPATH` 中）：

```bash
atk case \
  -f tests/atk/module/moe_unpermute/moe_unpermute.yaml \
  -p tests/atk/module/moe_unpermute/generate_moe_unpermute_hook.py
```

预期用例文件：`result/moe_unpermute/json/all_moe_unpermute.json`（150 例（bf16 / fp16 / fp32 各 50））。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
PYTHONPATH=/path/to/ATK:/path/to/this/directory \
python tests/atk/module/moe_unpermute/collect_nvte_golden.py \
  --cases result/moe_unpermute/json/all_moe_unpermute.json \
  --input-dir ./moe_unpermute_inputs \
  --golden-dir ./moe_unpermute_goldens
```

`--golden-dir` 必须与 `nodes_npu_vs_golden.yaml` 中 `accuracy_load` 节点的 `output_path`（`./moe_unpermute_goldens`）一致。`pad_with_probs` 用例无论 `--device` 取值如何，都在 CPU 基线上采集。

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/moe_unpermute/json/all_moe_unpermute.json`、`moe_unpermute_inputs` 与 `moe_unpermute_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/moe_unpermute/json/all_moe_unpermute.json \
  -n tests/atk/module/moe_unpermute/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/moe_unpermute/function_moe_unpermute.py \
  --input_data ./moe_unpermute_inputs \
  -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并
重跑 NPU 任务。用例集本身的变更（本轮从 curated 矩阵切到采样生成、且 dtype
扩到 bf16/fp16/fp32）同理：旧 golden 与新 case 编号/形状/槽位语义不再对应，
必须重采。过期 golden 表现为契约尺寸不匹配或无法解释的逐用例失败；不要调容差
去掩盖它——先重采集。
