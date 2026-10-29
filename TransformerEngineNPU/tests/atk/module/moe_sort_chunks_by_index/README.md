# TENPU moe_sort_chunks_by_index ATK 看护模块（no-CPU-reference 模板）

`transformer_engine.pytorch.permutation.moe_sort_chunks_by_index` 沿 dim-0 把输入 tensor 按 `split_sizes` 切成多个 chunk，并按 `sorted_index` 重排这些 chunk（输出 chunk i = 输入 chunk `sorted_index[i]`），返回排序后的激活。本模块遵循 `../rmsnorm/` 确立的 **no-CPU-reference** 模板。

覆盖范围（由 yaml 采样池驱动）：BF16/FP16/FP32（各 50 例，共 150 例）、chunk 数 2–16、均匀/不均衡/坍缩（单 chunk 吃 ~60%）/空 chunk 负载、逆序/旋转/交错三种置换。比对前向输出、输入梯度与序列化 API 契约。不含 FP8。

**两条 spec 由生成器渲染、不是 yaml 里的字面量**：`split_spec`/`sorted_spec` 必须与采样到的 token 数一致（split 之和 == T、置换长度 == chunk 数且非恒等），yaml 只能采"风格"（`<pattern>:<num_chunks>` 与置换风格），钩子在 `after_case_config` 里渲染成真实字符串。下游工具读 case json 时照常按字符串解析即可（语义不变）。

## 文件构成

- `moe_sort_chunks_by_index.yaml`：ATK 设计文件与固定精度容差。
- `generate_moe_sort_chunks_by_index_hook.py`：钩子路线生成器（唯一路线）——原生引擎驱动采样，生成器渲染两条 spec 并镜像 dy 形状。
- `function_moe_sort_chunks_by_index.py`：纯设备 adapter、spec 解析器、契约编码器。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行（ATK 已安装或位于 `PYTHONPATH` 中）：

```bash
atk case \
  -f tests/atk/module/moe_sort_chunks_by_index/moe_sort_chunks_by_index.yaml \
  -p tests/atk/module/moe_sort_chunks_by_index/generate_moe_sort_chunks_by_index_hook.py
```

预期用例文件：`result/moe_sort_chunks_by_index/json/all_moe_sort_chunks_by_index.json`（默认 `dtype_numbers: 50` × 3 dtype ⇒ 150 例）。用例集由 yaml 采样池与 `-s <seed>`（CLI 默认 1234）决定：同 yaml + 同 seed + 同 ATK 版本 ⇒ 同用例。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
PYTHONPATH=/path/to/ATK:/path/to/this/directory \
python tests/atk/module/moe_sort_chunks_by_index/collect_nvte_golden.py \
  --cases result/moe_sort_chunks_by_index/json/all_moe_sort_chunks_by_index.json \
  --input-dir ./moe_sort_chunks_by_index_inputs \
  --golden-dir ./moe_sort_chunks_by_index_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/moe_sort_chunks_by_index/json/all_moe_sort_chunks_by_index.json`、`moe_sort_chunks_by_index_inputs` 与 `moe_sort_chunks_by_index_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/moe_sort_chunks_by_index/json/all_moe_sort_chunks_by_index.json \
  -n tests/atk/module/moe_sort_chunks_by_index/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/moe_sort_chunks_by_index/function_moe_sort_chunks_by_index.py \
  --input_data ./moe_sort_chunks_by_index_inputs \
  -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并
重跑 NPU 任务。用例集本身的变更（如本轮从 curated 矩阵切到采样生成的
150 例，即 `dtype_numbers: 50` × 3 dtype）同理：旧 golden 与新 case 编号/形状
不再对应，必须重采。过期 golden
表现为契约尺寸不匹配或无法解释的逐用例失败；不要调容差去掩盖它——先重采集。
