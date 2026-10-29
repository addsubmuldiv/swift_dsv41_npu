# FP8 量化 ATK 精度测试

本目录使用 ATK 对比 TENPU/NPU 与 NVIDIA Transformer Engine/GPU 的 FP8
量化、反量化结果。覆盖 E4M3、E5M2，以及 FP16、BF16、FP32 输入。

## 输入与执行流程

输入为 `x` 和格式字符串 `fp8_format`（`e4m3` 或 `e5m2`）。ATK 默认生成器
根据 YAML 采样。
YAML 的 shape、数值分布、离群点和边界配置直接参与 ATK 原生生成流程。

`function_float8_quantization.py` 在两端执行相同流程：

1. 计算整个 x 的最大绝对值 `amax`，并设置 `scale = fp8_max / amax`。
   E4M3 的最大有限值为 448，E5M2 为 57344；整个 tensor 共用一个 scale。
2. 创建 `Float8Quantizer(scale, amax, fp8_dtype)`，调用 `quantize(x)`。
3. 调用 `dequantize(dtype=torch.float32)`，返回反量化结果、原始 FP8 编码、
   `scale_inv` 和接口信息。

量化关系为 `q = cast_fp8(x * scale)`，反量化关系为
`dequant = float32(q) * scale_inv`，其中 `scale_inv = 1 / scale`。
全零或含 NaN/Inf 的输入使用 scale=1。非有限输入保持原值，用于检查传播行为。
空输入及无法表示为正有限 FP32 的 scale 会报错。

标量输入内部整理为 `[1, 1]` 执行，输出恢复原 shape。两端不同的内部存储布局
统一呈现为 x 的逻辑 shape；原始编码使用 uint8 字节视图，不做数值转换。

## 输出与比较

| ATK 输出 | 内容 | 比较方式 |
|---|---|---|
| output0 | 反量化后的 FP32 张量 | YAML 配置的浮点容差 |
| output1 | FP8 原始字节（uint8） | 精确比较 |
| output2 | scale_inv（FP32） | YAML 配置的浮点容差 |
| output3 | 格式、shape、dtype 等接口信息 | 精确比较 |

比较对象是 NPU 与 GPU 的结果，不是反量化结果与原始 x。
NaN/Inf 输入的接口信息还包含输出特殊值的位置签名，避免仅因 GPU 输出含
非有限值而跳过两端传播行为的检查。比较标准由 YAML 的 `standard.acc` 指定。

## TENPU 量化实现

`Float8Quantizer` 使用 FP32 输入和 FP32 scale 调用
`torch_npu.npu_quantize(..., div_mode=False)`，由量化算子完成缩放与 FP8 转换。
FP16/BF16 输入先转为 FP32；量化张量保存的反量化系数仍为 scale 的倒数，
amax 仍根据原始输入计算。

普通量化、更新量化及分组量化采用相同的量化路径。FP16/BF16 转 FP32
需要额外的临时张量；大输入测试需根据实际设备内存和任务并发安排执行。

## 文件说明

- `float8_quantization.yaml`：普通用例模板。
- `float8_quantization_extra.yaml`：边界用例模板。
- `function_float8_quantization.py`：ATK 注册、量化执行和输出整理。
- `result_schema.json`：输出编号与含义。
- `collect_nvte_golden.py`：已有离线基线采集入口；以下命令使用在线 GPU 对比。

## 环境与节点配置

NPU 环境安装 TENPU，GPU 环境安装对应的 NVIDIA TE（本分支对齐 2.17.1）。
两端安装 ATK，并使用同版本的测试插件及其共享依赖 `_comparators.py`、
`_nonfinite_contract.py`。GPU 不需要安装 TENPU，仓库仅用于提供 ATK 插件。

节点 YAML 由测试人员按实际环境编写：NPU 为主节点，GPU 为远端比较节点，
配置设备号、GPU 地址、server 端口和结果目录。下列 `/path/to/nodes.yaml`
必须替换成实际文件。

原 `nodes_npu_vs_golden.yaml` 保留，用于读取已采集的离线 GPU 基线
（`accuracy_load`），不用于下面的在线 GPU server 对比。

### GPU：启动 server

从 TENPU 仓库外启动；避免 `PYTHONPATH` 中的 TENPU 源码遮蔽 NVIDIA TE。

```bash
cd /tmp
python -c 'import transformer_engine as te; print(te.__file__)'
atk server --devices 0 --port 9090 \
  --plugin_path /path/to/TransformerEngineNPU/tests/atk/module/float8_quantization/function_float8_quantization.py \
  --output_path /tmp/atk_float8_gpu
```

### NPU：生成并执行普通用例

```bash
cd /path/to/TransformerEngineNPU

atk case -f tests/atk/module/float8_quantization/float8_quantization.yaml
atk task -c result/float8_quantization/json/all_float8_quantization.json \
  -n /path/to/nodes.yaml -p tests/atk/module/float8_quantization/function_float8_quantization.py \
  --syc_dataset --save_data input:bin --save_data output:bin -tk accuracy
```

### NPU：生成并执行 extra 用例

在仓库根目录执行，沿用上述 GPU server：

```bash
atk case -f tests/atk/module/float8_quantization/float8_quantization_extra.yaml
atk task -c result/float8_quantization_extra/json/all_float8_quantization_extra.json \
  -n /path/to/nodes.yaml -p tests/atk/module/float8_quantization/function_float8_quantization.py \
  --syc_dataset --save_data input:bin --save_data output:bin -tk accuracy
```

`atk case` 不带 `-p`，使用 ATK 默认生成器；`atk task -p` 仍必须指定执行插件。
`--syc_dataset` 让两端使用同一份实际输入；保存的 input.bin 可用于复现失败用例。
普通模板每种 dtype 50 例，共 150 例；当前 extra 模板生成 13 例。
比较标准为 `mixed_tolerance_bm`。
上述生成命令均使用 ATK 默认随机种子。

更新测试插件后重启 GPU server 和 NPU task。修改 YAML 或采样设置后重新生成 JSON；
保持相同模板、种子和 ATK 版本以复现配置。使用离线基线时，基线必须与当前插件、
输入和缩放策略相匹配。

## 离线 GPU 基线验证

也可沿用原离线采集方式，NPU 和 GPU 使用同一份输入及当前版本的插件。

NPU 在仓库根目录生成配置：

```bash
atk case -f tests/atk/module/float8_quantization/float8_quantization.yaml
```

GPU 在仓库外采集基线（先复制上述 JSON 到对应相对路径）：

```bash
python /path/to/TransformerEngineNPU/tests/atk/module/float8_quantization/collect_nvte_golden.py \
  --cases result/float8_quantization/json/all_float8_quantization.json \
  --input-dir ./float8_quantization_inputs \
  --golden-dir ./float8_quantization_goldens
```

将 `float8_quantization_inputs` 和 `float8_quantization_goldens` 复制到 NPU 仓库根目录后执行：

```bash
atk task \
  -c result/float8_quantization/json/all_float8_quantization.json \
  -n tests/atk/module/float8_quantization/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/float8_quantization/function_float8_quantization.py \
  --input_data ./float8_quantization_inputs \
  -tk accuracy
```
