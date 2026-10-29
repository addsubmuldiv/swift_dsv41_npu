# FAQ

## HCCL error 7 / ERR02200：端口被其他训练进程占用

### 问题现象

分布式启动时所有 rank 报错：

```text
RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140
HCCL function error: hcclCommInitRootInfoConfig(...) error code 7
DIST call hccl api failed
```

各用例报错**逐字一致**。穿透 `HCCLUtils.cpp` / `ERR02200` 表层，在 stderr 深层能找到关键一行：

```text
EI0020: Failed to enable listening ... The IP address 192.168.2.x and port 16666 have already been bound
```

### 问题根因

HCCL 默认 NPU socket 端口（`16666`，遵循 `HCCL_NPU_SOCKET_PORT_RANGE`）被机器上**其他用户的训练进程**占用。属于环境级问题，不是代码回归——判据是跨用例、跨分支报错逐字一致。

### 解决方案

1. 在 stderr 日志中搜索 `EI0020`，定位被占用的 IP/端口及占用进程（如 `torchrun`、`trainer.py`）；
2. 启动前设置 `HCCL_NPU_SOCKET_PORT_RANGE`，把端口范围错开到空闲段；
3. 或等待占用端口的训练结束后再启动。

## 升级 CANN / torch 后报 libstdc++ / CXXABI / GLIBCXX 符号缺失

### 问题现象

加载 torch_npu / TransformerEngineNPU 时运行期报：

```text
ImportError: .../libstdc++.so.6: version `CXXABI_1.3.15' not found
```

（同类还有 `GLIBCXX_3.4.xx not found`。）

### 问题根因

系统自带 libstdc++ 版本较旧（缺 `CXXABI_1.3.15`），而 conda 环境自带的 libstdc++ 够新，却默认不在动态库加载路径里 → 加载 torch_npu 时命中系统旧版。

> [!NOTE]
>
> 重建 conda 环境**不能**解决：新环境的 libstdc++ 版本相同，`conda activate` 依然不把自身 lib 加进加载路径，还是会命中系统旧版。

### 解决方案

让 conda 环境的 libstdc++ 进入加载路径，一行：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

写进 conda 环境的激活脚本 `$CONDA_PREFIX/etc/conda/activate.d/libstdcxx-fix.sh` 后，每次 `conda activate` 自动生效，不用手动执行。

## 昇腾设备初始化失败 / torch_npu owner 警告

### 问题现象

- NPU 初始化失败，或驱动/权限类报错；
- torch_npu 启动时打印 `owner` 相关 WARNING。

### 问题根因

当前用户不在 `HwHiAiUser` 组，无 NPU 设备访问权限。

### 解决方案

将训练用户加入 `HwHiAiUser` 组并重新登录：

```bash
sudo usermod -aG HwHiAiUser <训练用户>
```

`torch_npu` 的 owner WARNING 属于 cosmetic 提示，不影响运行，可忽略。

## 优化器报 Wrapped parameters must be one of torch.cuda.*Tensor

### 问题现象

模型构建阶段、优化器初始化时报错：

```text
TypeError: Wrapped parameters must be one of torch.cuda.FloatTensor,
torch.cuda.HalfTensor, or torch.cuda.BFloat16Tensor. Received torch.BFloat16Tensor
```

触发条件（需同时满足）：

- 使用了 `--use-cpu-initialization=True`，且未开启 `--use-torch-fsdp2`；
- 跑的是 Megatron-LM **`main` 分支**（含 ModelBuilder 集成）。

### 问题根因

Megatron-LM 当前 `main` 分支的模型构建存在两条并行的参数初始化路径，GPU 分配条件不一致：

| 路径 | 条件 | 语义 |
|------|------|------|
| 旧 `get_model()`（正确） | `not (use_torch_fsdp2 and use_cpu_initialization)` | 仅当两 flag 都为 True 才跳过 `.cuda()` |
| `builder.build_distributed_models()`（有 bug） | `not use_torch_fsdp2 and not use_cpu_initialization` | 任一 flag 为 True 就跳过 `.cuda()` |

`--use-cpu-initialization=True` 且默认 `--use-torch-fsdp2=False` 时，走 `builder.build_distributed_models()` 路径会跳过 `.cuda()` → 参数留在 CPU → 被包成 CPU bf16 参数 → 优化器类型校验失败。

### 解决方案

MA 已通过修复类功能提供处理：

- 在启动命令中加 **`--fix-use-cpu-initialization`**。该开关经 `MegatronFixFeature` 注册，
  开启后 `prepare_existing_model_chunks_for_distributed_training_wrapper` 会在调用原函数**之前**对模型做幂等
  `.cuda()`，恢复与旧 `get_model()` 一致的时序（不改上游条件，不影响其他调用方）。
- **无需处理的分支**：`core_r0.18.0` 等 release 分支没有 ModelBuilder 集成路径，buggy 代码不可达，无需该开关。

> **注意**：报错消失只是第一步，重点是训练早期指标——grad_norm 应为非零（如 42–60）、loss 正常下降。
> 若 grad_norm=0.000 且 loss 不收敛，说明设备迁移发生在 DDP wrap 之后、梯度 hook 失效（静默失败），需确认 MA 版本包含对应时序修正。
