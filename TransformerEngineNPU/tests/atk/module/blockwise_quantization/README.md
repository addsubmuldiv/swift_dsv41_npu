# TENPU Float8Blockwise quantization ATK 看护模块

看护 `transformer_engine.pytorch.tensor.float8_blockwise_tensor.Float8BlockwiseQTensor`
（`Float8BlockQuantizer` 128×128 块缩放 quantize/dequantize）。**基线为 CPU
计算的参考而非 NVTE golden**。

```
Float8BlockQuantizer(fp8_dtype, rowwise=True, columnwise=True, block_scaling_dim=2)
q = quantizer(x)                    # npu_dynamic_block_quant x2 (128x128 tiles)
q.dequantize()                      # Python tiled dequantize (rowwise payload)
q.update_usage(rowwise=False); q.dequantize()   # (columnwise payload)
```

| 输出 | 内容 | 参考 | 容差 |
|---|---|---|---|
| `output_0` `dequant_rowwise` | rowwise payload 的 dequantize / bound | 输入自身（CPU） | rtol 0.125, atol 0.0675 |
| `output_1` `dequant_columnwise` | columnwise payload 的 dequantize / bound | 输入自身（CPU） | rtol 0.125, atol 0.0675 |
| `output_2` `cross_consistency` | \|rowwise − columnwise\| / bound | 全零 | atol 0.26 |
| `output_3` `scale_rowwise_logical` | logical-grid [M/128, K/128] scale | CPU canonical `amax/448` | rtol 1.0, ratio 1.0 |
| `output_4` `scale_columnwise_logical` | logical-grid scale | CPU canonical `amax/448` | rtol 1.0, ratio 1.0 |
| `output_5` `api_contract_json_utf8` | 序列化契约 | 同一构建器 | 精确 |

## 文件构成

- `blockwise_quantization.yaml`：ATK 设计文件与精度容差。
- 无生成器文件：本模块**零跨输入耦合**，走**纯 yaml 路线**（`generate: default`，不传 `-p`）。
- `function_blockwise_quantization.py`：纯设备适配器、logical-grid scale 提取、
  CPU canonical scale 数学、契约编码器、比较器注册。
- `collect_cpu_reference.py`：CPU 参考采集器（任意主机）。
- `nodes_npu_vs_golden.yaml`：NPU + accuracy_load 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 TENPU 仓库根目录下执行（ATK 已安装，安装方法与 commit 点见
`../../README.md` 基线与环境节；用例生成不依赖 CUDA，任意机器均可）：

```bash
atk case -f tests/atk/module/blockwise_quantization/blockwise_quantization.yaml   # 纯 yaml 路线：不需要 -p
```

## 2. 采集 CPU 参考（任意主机，无需加速器）

仍在 TENPU 仓库根目录下执行：

```bash
  python tests/atk/module/blockwise_quantization/collect_cpu_reference.py \
  --cases result/blockwise_quantization/json/all_blockwise_quantization.json \
  --input-dir ./blockwise_inputs --reference-dir ./blockwise_quantization_reference
```

目录布局由收集器自动安排：input.bin 平铺在 `<input-dir>/<case_id>/`；
参考在 `<reference-dir>/<backend_label>/<用例文件 stem>/<case_id>/`（`atk
task` 运行时会把每个用例的 save_name 覆写为用例文件 stem，small/real 分层
在运行时不存在）。`--reference-dir` 必须与 `nodes_npu_vs_golden.yaml` 中
accuracy_load 节点的 `output_path` 一致——收集器收完会打印 ATK 将解析的确切
路径供核对。

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/blockwise_quantization/json/all_blockwise_quantization.json`、`blockwise_inputs` 与 `blockwise_quantization_reference` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/blockwise_quantization/json/all_blockwise_quantization.json \
  -n tests/atk/module/blockwise_quantization/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/blockwise_quantization/function_blockwise_quantization.py \
  --input_data ./blockwise_inputs -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本重采集基线**（golden 模块在 CUDA 主机，
golden-free 模块用 `collect_cpu_reference.py`）并重跑任务。用例集本身的变更（本轮从
curated 矩阵切到纯 yaml 采样、dtype 扩到 bf16/fp16/fp32）同理：旧基线与新 case
编号/形状不再对应，必须重采。不要调容差去掩盖它——先重采集。
