# LayerNorm ATK 精度测试

本目录使用 ATK 对比 TENPU/NPU 与 NVIDIA Transformer Engine/GPU 的结果，
分别执行前向和反向测试，比较标准由 YAML 的 `standard.acc` 指定。

## 测试设计

前反向级联测试将每个设备前向产生的统计量继续传入本设备反向。前向统计量即使
满足容差，也可能在反向计算中造成更大的输出差异，使反向失败难以单独定位。

独立前向检查本设备计算的输出和统计量；独立反向使用两端共享的输入与统计量，
只执行反向，以隔离前向统计量差异对反向结果的影响。

## 输入生成与执行逻辑

1. `atk case` 通过 `generate_layernorm_hook.py` 调用 ATK 原生采样，生成
   x、dy、gamma、beta、eps、zero_centered_gamma 的配置。dy 的 shape/dtype 与 x 一致，权重参数
   的 shape 为 `[x.shape[-1]]`，dtype 与 x 一致。
2. hook 将配置包装为 `tenpu_layernorm_inputs`。此时只生成 JSON，尚未生成
   张量或计算统计量。实际输入配置位于
   `inputs[0].range_values.source_case.inputs`，其中 shape 已经过生成约束。
3. `atk task` 在 NPU 主机的 producer 进程中生成实际输入。反向额外在 CPU
   上计算一次公共统计量：实际 dtype 的 x 转 FP64，沿最后一维计算
   means = mean(x)，rstdevs = rsqrt(var(x, correction=0) + eps)，最后将统计量保存为 FP32。
4. `--syc_dataset` 将完整输入包同步到 GPU。receiver 不独立生成输入或统计量。
   CPU 只准备反向输入，不作为精度比较的输出标杆。
5. 前向调用 `op_forward` 并返回本设备统计量；反向将共享统计量放入
   `OperationContext.saved_tensors`，调用 `op_backward`，不执行设备前向。

`zero_centered_gamma=true` 的有效权重为 `gamma + 1`，由设备实现处理。
两端按各自接口要求整理统计量形状，NPU 为 `[M, 1]`、GPU 为 `[M]`，其中
`M = prod(x.shape[:-1])`。输入在调用前克隆，避免上下文清理影响原始数据。

## 文件与输出

- `function_layernorm.py`：自定义输入类型注册、前向/反向执行入口。
- `layernorm_inputs.py`：输入配置包装、公共统计量计算及校验。
- `generate_layernorm_hook.py`：输入间约束、规模约束和独立用例包装。
- `layernorm_forward.yaml` / `layernorm_backward.yaml`：普通用例模板。
- `layernorm_forward_extra.yaml` / `layernorm_backward_extra.yaml`：边界用例模板。
- `collect_nvte_golden.py`：GPU 离线采集器。复用在线输入生成逻辑，导出直接输入
  形态的 case JSON、含共享统计量的 `input.bin` 和每个阶段的 NVTE 输出。
- `result_schema.json`：各入口的输出顺序。

| ATK 输出 | 独立前向 | 独立反向 |
|---|---|---|
| output0 | y | dx |
| output1 | means | dgamma |
| output2 | rstdevs | dbeta |
| output3 | 接口信息 | 接口信息 |

接口信息包含 shape、dtype 和参数结构，以字节形式精确比较。NaN/Inf 输入用例
还记录输出特殊值的位置签名，检查两端传播行为；不替换数值输出或放宽容差。

NPU 前向和反向分别调用 `torch.native_layer_norm` / `torch.ops.aten.native_layer_norm_backward`，
返回底层接口计算的结果。GPU 使用对应版本的 NVIDIA TE 实现。

## 环境与节点配置

NPU 环境安装 TENPU，GPU 环境安装对应版本的 NVIDIA TE（本分支对齐 2.17.1）。
两端均需安装支持自定义数据类型、插件加载和 `--syc_dataset` 的 ATK，
并使用相同版本的本目录及 `tests/atk/module` 下共享插件文件。

节点 YAML 由测试人员按实际环境编写：NPU 为主节点，GPU 为远端比较节点，
配置正确的设备号、GPU 地址、server 端口与结果目录。
下列命令中的 `/path/to/nodes.yaml` 必须替换为实际文件。

`nodes_npu_vs_golden.yaml` 为离线 GPU 输出加载配置，使用 `accuracy_load`。
下列在线对比命令需要单独配置 GPU server 节点，不能直接使用该离线配置。

## 离线测试

离线流程分为 GPU 采集和 NPU 回放两步，无需同时连接两台机器或启动 GPU server。
GPU 采集器复用 `layernorm_inputs.prepare_input_bundle`，生成一次实际输入；反向的
means/rstdevs 也只在此时计算一次并写入 `input.bin`。GPU 和 NPU 均读取该文件，
分别执行对应阶段。CPU 计算的统计量只是共享反向输入，输出标杆来自 NVTE/GPU。

反向所需的量由采集脚本自动准备，无需手动计算：

1. 在 GPU 主机的 CPU 上，将实际输入 x 转为 FP64，沿最后一维计算
   `means = mean(x)` 和 `rstdevs = 1 / sqrt(var(x, correction=0) + eps)`，再保存为 FP32。
2. 将 `x、dy、gamma、beta、eps、zero_centered_gamma、means、rstdevs` 共 8 项写入
   同一个 `input.bin`。GPU 从该文件读取输入，直接执行反向并保存 golden。
3. 按第 3 步将产物目录复制到 NPU 主机。NPU 读取相同的 `input.bin`，将保存的
   means/rstdevs 放入反向上下文，直接执行反向；两端均不执行设备前向来准备反向。

离线模式的跨机器传递由使用者复制文件完成，采集脚本不会自动上传到另一台机器。
在线模式则由 `--syc_dataset` 传输输入包。前向仍直接检查设备前向输出及其统计量。

### 1. 生成普通与 extra 用例

在仓库根目录执行（可在 GPU 或 NPU 主机生成，后续传递同一份 JSON）：

```bash
for suite in forward backward forward_extra backward_extra; do
  atk case -f tests/atk/module/layernorm/layernorm_${suite}.yaml \
    -p tests/atk/module/layernorm/generate_layernorm_hook.py
done
```

输出为 `result/layernorm_<suite>/json/all_layernorm_<suite>.json`。

### 2. GPU：采集输入和 NVTE golden

GPU 上安装对应版本的 NVIDIA TE 和 ATK。使用绝对路径指向本分支仓库，
从仓库外执行，检查导入的确实是 GPU 版 TE；`PYTHONPATH` 也不能包含 TENPU 源码。
下面的 `/path/to/TransformerEngineNPU` 替换为 GPU 主机上的实际路径。

```bash
export TENPU_ATK_ROOT=/path/to/TransformerEngineNPU
cd /tmp
python -c 'import transformer_engine as te; print(te.__file__)'

for suite in forward backward forward_extra backward_extra; do
  python "$TENPU_ATK_ROOT/tests/atk/module/layernorm/collect_nvte_golden.py" \
    --cases "$TENPU_ATK_ROOT/result/layernorm_${suite}/json/all_layernorm_${suite}.json" \
    --input-dir "$TENPU_ATK_ROOT/layernorm_inputs/${suite}" \
    --golden-dir "$TENPU_ATK_ROOT/layernorm_goldens" \
    --cases-out "$TENPU_ATK_ROOT/layernorm_cases_offline" || break
done
```

`--cases-out` 必填。采集器将输入包配置转换为 ATK 能逐项读取的直接输入配置，
独立前向为 6 个输入，独立反向为 8 个输入（追加 FP32 means/rstdevs）。
原始输入包 JSON 保留供在线测试使用。

每套用例独立使用输入目录，防止相同 case ID 的前向、反向或 extra 数据互相覆盖。
产物布局如下，其中 `<stem>` 是 `all_layernorm_forward` 等原始 JSON 文件名去掉扩展名：

```text
layernorm_cases_offline/<stem>.json
layernorm_inputs/<suite>/<case_id>/input.bin
layernorm_goldens/cpu_nvte/<stem>/<case_id>/output_0.pt ... output_3.pt
layernorm_goldens/<stem>_manifest.json
```

manifest 记录 TE 版本和导入路径、case ID、输入 SHA256、输出顺序。
采集器遇到 OOM 或其他失败会报错，不发布该套可运行的离线 JSON。
重新采集时使用新的输入目录、golden 目录和 `--cases-out` 目录，避免覆盖之前的产物。
设备内存需求仍受生成约束影响，可先缩小 YAML 的采样范围生成少量用例。

### 3. NPU：加载同一份输入并比较

将 `layernorm_cases_offline`、`layernorm_inputs`、`layernorm_goldens` 三个目录完整复制到
NPU 仓库根目录。检查 `nodes_npu_vs_golden.yaml` 的设备号和输出路径；默认从当前
工作目录读取 `./layernorm_goldens`。

在 NPU 仓库根目录执行：

普通前向：

```bash
atk task -c layernorm_cases_offline/all_layernorm_forward.json \
  -n tests/atk/module/layernorm/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/layernorm/function_layernorm.py \
  --input_data ./layernorm_inputs/forward \
  --save_data output:bin -tk accuracy
```

普通反向：

```bash
atk task -c layernorm_cases_offline/all_layernorm_backward.json \
  -n tests/atk/module/layernorm/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/layernorm/function_layernorm.py \
  --input_data ./layernorm_inputs/backward \
  --save_data output:bin -tk accuracy
```

extra 前向：

```bash
atk task -c layernorm_cases_offline/all_layernorm_forward_extra.json \
  -n tests/atk/module/layernorm/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/layernorm/function_layernorm.py \
  --input_data ./layernorm_inputs/forward_extra \
  --save_data output:bin -tk accuracy
```

extra 反向：

```bash
atk task -c layernorm_cases_offline/all_layernorm_backward_extra.json \
  -n tests/atk/module/layernorm/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/layernorm/function_layernorm.py \
  --input_data ./layernorm_inputs/backward_extra \
  --save_data output:bin -tk accuracy
```

此处传的是采集器导出的直接输入 JSON，不能改成第 1 步生成的输入包 JSON。
离线回放不需要 `TENPU_LAYERNORM_INPUT_ROLE` 或 `--syc_dataset`；NPU 从文件加载
共享统计量，不重新生成随机输入或执行前向来准备反向。
`cpu_nvte` 节点只加载保存的 GPU 输出，不运行 CPU 算子。

## 在线测试

### GPU：启动 server

从 TENPU 仓库外启动；`PYTHONPATH` 不应将 TENPU 源码置于已安装的 NVIDIA TE 前面。

```bash
export TENPU_LAYERNORM_INPUT_ROLE=receiver
cd /tmp
python -c 'import transformer_engine as te; print(te.__file__)'
atk server --devices 0 --port 9090 \
  --plugin_path /path/to/TransformerEngineNPU/tests/atk/module/layernorm/function_layernorm.py \
  --output_path /tmp/atk_layernorm_gpu
```

### NPU：生成普通用例并执行

```bash
export TENPU_LAYERNORM_INPUT_ROLE=producer
cd /path/to/TransformerEngineNPU

atk case -f tests/atk/module/layernorm/layernorm_forward.yaml \
  -p tests/atk/module/layernorm/generate_layernorm_hook.py
atk case -f tests/atk/module/layernorm/layernorm_backward.yaml \
  -p tests/atk/module/layernorm/generate_layernorm_hook.py

atk task -c result/layernorm_forward/json/all_layernorm_forward.json \
  -n /path/to/nodes.yaml -p tests/atk/module/layernorm/function_layernorm.py \
  --syc_dataset --save_data input:bin --save_data output:bin -tk accuracy
atk task -c result/layernorm_backward/json/all_layernorm_backward.json \
  -n /path/to/nodes.yaml -p tests/atk/module/layernorm/function_layernorm.py \
  --syc_dataset --save_data input:bin --save_data output:bin -tk accuracy
```

### NPU：生成 extra 用例并执行

在仓库根目录执行，保留 producer 设置并沿用上述 GPU server：

```bash
atk case -f tests/atk/module/layernorm/layernorm_forward_extra.yaml \
  -p tests/atk/module/layernorm/generate_layernorm_hook.py
atk case -f tests/atk/module/layernorm/layernorm_backward_extra.yaml \
  -p tests/atk/module/layernorm/generate_layernorm_hook.py

atk task -c result/layernorm_forward_extra/json/all_layernorm_forward_extra.json \
  -n /path/to/nodes.yaml -p tests/atk/module/layernorm/function_layernorm.py \
  --syc_dataset --save_data input:bin --save_data output:bin -tk accuracy
atk task -c result/layernorm_backward_extra/json/all_layernorm_backward_extra.json \
  -n /path/to/nodes.yaml -p tests/atk/module/layernorm/function_layernorm.py \
  --syc_dataset --save_data input:bin --save_data output:bin -tk accuracy
```

`atk case -p` 加载生成插件，`atk task -p` 与 server 的 `--plugin_path` 加载执行
及输入插件，两者不能混用。命令使用 ATK 默认随机种子；复现配置时应保持模板、插件和 ATK 版本一致。
普通前向和反向各生成 150 例，extra 前向和反向各生成 14 例。
YAML 的 `backward: false` 表示不让 ATK 自动追加反向，独立反向由
`function_layernorm_backward` 显式执行。

在线输入包用例通过 ATK 的文件同步机制传输，不接受 `--input_data`。
离线测试请使用上文采集器导出的直接输入用例。
更新插件后必须重启 GPU server 并重新启动 task。若生成逻辑改变，需要重新生成 JSON。

## 验证记录

本次离线适配完成本地 CPU 输入生成、保存和 ATK 文件回读检查；GPU/NPU 实机的
离线精度结果仍需按上述流程验证。

已有在线精度验证记录（不代表本次离线验证结果）：普通前向 150/150、普通反向 150/150、extra 前向 14/14、
extra 反向 14/14 的精度验证，比较标准为 `mixed_tolerance_bm`。

## 生成约束

YAML 保留采样范围，hook 在采样后通过 `../_stress_constraints.py` 约束最终配置：

- 最后一维 H < 128 时，所有 dtype 的行数 M 最多 8192；从左到右将前导维度向下折半至满足上限。
- 随后检查单个 x 最多 2 GiB；FP32 的行数 M 最多 2²¹。
- x 范围含 NaN/Inf 的边界用例最多 2²⁰ 个元素。
- 若仍超过上述字节数、FP32 行数或特殊值元素数上限，反复将最大的前导维度向下折半。
  两个阶段都保留最后一维 H 和维数，再同步其他输入形状。
- 保持用例数量、dtype、数值分布和比较标准。
  调整后的维度可能不在 YAML 原候选集合中，因此最终规模分布会变化。

这些是测试规模上限，不是算子支持范围，也不保证峰值内存或精度结果；
实际运行还需存储 dy、输出、统计量、临时张量及算子工作区。
extra 保留 NaN/Inf 及其产生的统计量；有限 x 仍要求统计量有限、rstdevs 为正。
当前模板不生成空张量或标量 x。
