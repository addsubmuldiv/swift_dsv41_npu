# TENPU LayerNormMLP ATK 看护模块

看护 `transformer_engine.pytorch.LayerNormMLP`——LayerNorm/RMSNorm 加 MLP
（两个线性层，中间夹激活）。基线为 NVTE golden（CUDA 采集），数值与 API
契约双重比对。

输出槽位：前向输出、输入梯度、ln/fc1/fc2 的 weight/bias 梯度共 8 个数值
槽位（RMSNorm / bias 关闭时对应槽位为空），以及序列化 API 契约。用例
输入：8 个 tensor（`x`、`dy`、`layer_norm_weight`、`layer_norm_bias`、
`fc1_weight`、`fc1_bias`、`fc2_weight`、`fc2_bias`）+ 属性（`eps`、
`normalization`、`activation`、`bias`、`zero_centered_gamma`）。

覆盖范围（由 yaml 采样池驱动）：BF16/FP16/FP32（各 50 例，共 150 例）、
2/3/4 维 x、LayerNorm/RMSNorm、gelu/silu、bias 与 zero_centered_gamma 开关、
ffn 从池中采样（受权重预算约束）。只有 `x` 真正被采样，其余 tensor 的形状全部
由钩子从 x 与 ffn 派生。

## 文件构成

- `layernorm_mlp.yaml`：ATK 设计文件与精度容差。
- `generate_layernorm_mlp_hook.py`：钩子路线生成器（唯一路线）——原生引擎驱动采样，生成器从 x 与 ffn 派生全部形状。
- `function_layernorm_mlp.py`：纯设备适配器、契约编码器、比较器注册。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器（可选
  `--baseline cpu-fp32 / cpu-bf16` 采集 CPU 参考作分析工具）。
- `cpu_reference.py`：CPU 精确参考实现（供 collector 的基线选项使用）。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。
- `diagnose_param_grad.py`：NPU 主机上诊断参数梯度失配的工具（三方比较）。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case -f tests/atk/module/layernorm_mlp/layernorm_mlp.yaml \
  -p tests/atk/module/layernorm_mlp/generate_layernorm_mlp_hook.py
```

## 2. 采集 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/layernorm_mlp/collect_nvte_golden.py \
  --cases result/layernorm_mlp/json/all_layernorm_mlp.json \
  --input-dir ./layernorm_mlp_inputs --golden-dir ./layernorm_mlp_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/layernorm_mlp/json/all_layernorm_mlp.json`、`layernorm_mlp_inputs` 与 `layernorm_mlp_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/layernorm_mlp/json/all_layernorm_mlp.json \
  -n tests/atk/module/layernorm_mlp/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/layernorm_mlp/function_layernorm_mlp.py \
  --input_data ./layernorm_mlp_inputs -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并重跑
NPU 任务。用例集本身的变更（本轮从 11 例 curated 矩阵切到采样生成、dtype 扩到
bf16/fp16/fp32）同理：旧 golden 与新 case 编号/形状不再对应，必须重采。不要调
容差去掩盖它——先重采集。

## 4. 前向槽的精确 fp64 基线（摘要）

golden 侧对 output_0 携带精确 fp64 物化链参考（ln_out / fc1 前激活 / 激活
输出逐级单次舍入）而非 NVTE kernel 输出——**基线选择，不是放宽容差**。口径
细节在适配器 `function_layernorm_mlp.py` 的 `exact_baselines`/
`_apply_exact_baselines` docstring。判别器已迁至项目目录
`atk_experiments/layernorm_mlp/`。output_1 不带基线（操作数为链派生互噪声
量），近零残留走救援键。
