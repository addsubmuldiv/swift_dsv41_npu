# TENPU MXFP8 quantization ATK 看护模块

看护 `transformer_engine.pytorch.tensor.mxfp8_tensor.MXFP8Tensor`（`MXFP8Quantizer`
双 payload quantize/dequantize）。**基线为 CPU 计算的参考而非 NVTE golden**。

**输入 dtype 仅 bf16/fp16，不支持 fp32**

只运行 TENPU 设备路径，覆盖 dual-axis kernel 的两个独立 payload，输出槽位与参考：

```
MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=True)
q = quantizer(x)                    # npu_dynamic_mx_quant_with_dual_axis
q.dequantize()                      # Python dequantize (rowwise payload)
q.update_usage(rowwise=False); q.dequantize()   # (columnwise payload)
```

| 输出 | 内容 | 参考 | 容差 |
|---|---|---|---|
| `output_0` `dequant_rowwise` | rowwise payload 的 dequantize / bound | 输入自身（CPU） | rtol 0.125, atol 0.0675 |
| `output_1` `dequant_columnwise` | columnwise payload 的 dequantize / bound | 输入自身（CPU） | rtol 0.125, atol 0.0675 |
| `output_2` `cross_consistency` | \|rowwise − columnwise\| / bound | 全零 | atol 0.26 (= 2×(rtol·range + atol)) |
| `output_3` `scale_rowwise_decoded` | 解码后的 E8M0 scale，[M, K/32] | CPU canonical `2^ceil(log2(amax/448))` | rtol 1.0（一个指数），ratio 1.0 |
| `output_4` `scale_columnwise_decoded` | 解码后的 E8M0 scale，[M/32, K] | CPU canonical `2^ceil(log2(amax/448))` | rtol 1.0（一个指数），ratio 1.0 |
| `output_5` `api_contract_json_utf8` | 序列化契约 | 同一构建器 | 精确 |

## 文件构成

- `mxfp8_quantization.yaml`：ATK 设计文件与精度容差。
- 无生成器文件：本模块**零跨输入耦合**，走**纯 yaml 路线**（`generate: default`，不传 `-p`）。
- `function_mxfp8_quantization.py`：纯设备适配器、打包 scale 解码器、CPU
  canonical scale 数学、契约编码器、比较器注册。
- `collect_cpu_reference.py`：CPU 参考采集器（随处可跑，无需加速卡——910B3
  主机本身即可）。
- `nodes_npu_vs_golden.yaml`：NPU + accuracy_load 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 TENPU 仓库根目录下执行（ATK 已安装，安装方法与 commit 点见
`../../README.md` 基线与环境节；用例生成不依赖 CUDA，任意机器均可）：

```bash
atk case -f tests/atk/module/mxfp8_quantization/mxfp8_quantization.yaml   # 纯 yaml 路线：不需要 -p
```

## 2. 采集 CPU 参考（任意主机，无需加速器）

仍在 TENPU 仓库根目录下执行：

```bash
  python tests/atk/module/mxfp8_quantization/collect_cpu_reference.py \
  --cases result/mxfp8_quantization/json/all_mxfp8_quantization.json \
  --input-dir ./mxfp8_inputs --reference-dir ./mxfp8_quantization_reference
```

目录布局由收集器自动安排：input.bin 平铺在 `<input-dir>/<case_id>/`；
参考在 `<reference-dir>/<backend_label>/<用例文件 stem>/<case_id>/`（`atk
task` 运行时会把每个用例的 save_name 覆写为用例文件 stem，small/real 分层
在运行时不存在）。`--reference-dir` 必须与 `nodes_npu_vs_golden.yaml` 中
accuracy_load 节点的 `output_path` 一致——收集器收完会打印 ATK 将解析的确切
路径供核对。

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/mxfp8_quantization/json/all_mxfp8_quantization.json`、`mxfp8_inputs` 与 `mxfp8_quantization_reference` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/mxfp8_quantization/json/all_mxfp8_quantization.json \
  -n tests/atk/module/mxfp8_quantization/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/mxfp8_quantization/function_mxfp8_quantization.py \
  --input_data ./mxfp8_inputs -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本重采集基线**（golden 模块在 CUDA 主机，
golden-free 模块用 `collect_cpu_reference.py`）并重跑任务。用例集本身的变更（本轮从
curated 矩阵切到纯 yaml 采样、dtype 定为 bf16/fp16——fp32 因 kernel 限制排除，见
文首说明）同理：旧基线与新 case 编号/形状不再对应，必须重采。不要调容差去掩盖
它——先重采集。
