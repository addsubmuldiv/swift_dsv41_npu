# TENPU RoPE ATK 看护模块

看护 `transformer_engine.pytorch.attention.apply_rotary_pos_emb`（对 Q/K
tensor 施加 rotary position embedding，支撑 Megatron 融合与非融合 RoPE
路径）及其 fused-QKV 变体 `apply_fused_qkv_rotary_pos_emb`（Megatron 经
`qkv_split_arg_list` 切分融合 QKV tensor）。基线为 NVTE golden（CUDA 采集），
数值与 API 契约双重比对。

两种用例形态按第一个字符串 attr 分发：

- `rope`（3 个输出）：`forward_output, input_grad, api_contract`。输入为
  bf16 的 `t`/`dy` 加一个 **fp32 freqs tensor**。
- `rope_fused_qkv`（5 个输出）：`q_output, k_output, v_output_passthrough,
  qkv_input_grad, api_contract`。输入为融合的 `qkv` tensor 加独立的
  `q_freqs`/`k_freqs`（fp32）与 `qkv_split_arg_list`。覆盖 MHA 与 GQA 切分。
  反向以完整 `(q, k, v)` 元组播种。

覆盖范围（全部由 yaml 采样池驱动）：plain 形态支持 `sbhd` / `bshd` / `thd`
（thd 自动派生 `cu_seqlens` 并强制 `fused=True`）、fused-QKV 形态支持
`sbhd` / `bshd`（该 API 断言不支持 thd）与 GQA 比例（`np/ng` 由采样宽度
推导）；`interleaved` 开/关、`fused` 开/关、BF16/FP16/FP32 激活（各 50 例，共
150 例）；前向输出、输入梯度，
以及记录格式与 freqs 形状的序列化契约。

**注意槽位语义随变体变化**：两种形态都是 8 槽，但 plain 的变体 attr 在槽 3、
fused-QKV 在槽 4（适配器靠"从槽 3 起找第一个字符串"分派），槽 7 在 plain 下是
`fused`(bool)、在 fused-QKV 下是 `qkv_split_arg_list`(str)。读 case json 或写
下游工具时**必须按变体判布局**，不要按固定槽位解包。

## 文件构成

- `rope.yaml`：ATK 设计文件与精度容差，当前选用**混合容差标准**
  （`rope_mixed_tolerance`）。
- `generate_rope_hook.py`：钩子路线生成器（唯一路线）——原生引擎驱动采样，
  生成器按变体重建 8 槽布局并派生 freqs/dy/切分等形状与取值。
- `function_rope.py`：纯设备适配器（两种用例形态）、契约编码器、比较器注册。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义（plain 与
  fused-QKV 两套 schema）。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case \
  -f tests/atk/module/rope/rope.yaml \
  -p tests/atk/module/rope/generate_rope_hook.py
```

预期用例文件：`result/rope/json/all_rope.json`（默认 `dtype_numbers: 12` ×
bf16 单 dtype ⇒ 12 例）。用例集由 yaml 采样池与 `-s <seed>`（CLI 默认 1234）
决定：同 yaml + 同 seed + 同 ATK 版本 ⇒ 同用例。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/rope/collect_nvte_golden.py \
  --cases result/rope/json/all_rope.json \
  --input-dir ./rope_inputs \
  --golden-dir ./rope_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/rope/json/all_rope.json`、`rope_inputs` 与 `rope_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/rope/json/all_rope.json \
  -n tests/atk/module/rope/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/rope/function_rope.py \
  --input_data ./rope_inputs \
  -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并
重跑 NPU 任务。用例集本身的变更（如本轮从 21 例 curated 矩阵切到采样生成的
12 例）同理：旧 golden 与新 case 编号/形状/槽位语义不再对应，必须重采。过期
golden 表现为契约尺寸不匹配或无法解释的逐用例失败；不要调容差去掩盖它——
先重采集。
