# 快速入门

本文档旨在指导用户快速体验 MegatronAdaptor 的核心功能，并在昇腾 NPU 设备上成功启动 Megatron-LM 训练。各组件配套版本要求请参见《[版本说明](release_notes_ma.md)》。

## 环境准备

请先完成[安装指导](install_guide.md)中的所有步骤，并通过其中的"安装验证"章节确认 MegatronAdaptor 安装成功且 NPU 设备可用。

获取 Megatron-LM 源码并切换至配套版本：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.19.0
```

> [!NOTE]
>
> 获取 Megatron-LM 源码需访问 GitHub。若您所处的网络环境无法顺畅访问 `github.com`，请自行配置代理，或使用国内可访问的 Git 镜像（如 GitCode、Gitee 等）拉取源码后，再切换至配套分支，并核对仓库地址与分支，确保源码与配套版本一致。

开始训练前，请确保已加载 CANN 环境变量：

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 导入适配器

在使用 Megatron-LM 进行训练前，必须在其入口文件中引入 MegatronAdaptor。

**在 `Megatron-LM` 目录下修改 `pretrain_gpt.py` 文件**，在 `import torch` 下新增一行 `import megatron_adaptor`：

```Python
# ... 其他导入
import torch
import megatron_adaptor # <-- 新增此行
# ... 其他导入
```

如果您的数据预处理脚本 `tools/preprocess_data.py` 也需要调用 NPU 算子，同样需要在其中添加 `import megatron_adaptor`。

## 数据准备（可选，用于完整训练）

如果只是进行功能验证，可跳过此步。若需进行完整的预训练，请按以下步骤准备数据：

1. **下载 Tokenizer**
    新建 `Megatron-LM/gpt2_tokenizer` 目录，并从 [HuggingFace](https://huggingface.co/Xenova/gpt-3.5-turbo/tree/main) 下载 `vocab.json`、`merges.txt` 和 `config.json`：前两者是 GPT-2 风格的 BPE 词表与合并规则，`config.json` 用于让 HuggingFace Tokenizer 从本地目录识别分词器类型，缺失时会尝试联网获取而导致失败。

    > [!NOTE]
    >
    > 下载 Tokenizer 与数据集需访问 HuggingFace。若您所处的网络环境无法顺畅访问 `huggingface.co`，推荐前往 [ModelScope](https://modelscope.cn) 等国内社区下载相同资源，需关注待下载文件的正确性与安全性。

2. **下载并转换数据集**
    以 Alpaca 数据集为例：

    ```python
    import pandas as pd
    import json

    # 下载并读取数据
    data_df = pd.read_parquet("train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    data_df['text'] = data_df['text'].apply(lambda v: json.dumps({"text": v}))

    # 转换为 JSON 格式
    with open("alpaca_json.json", encoding='utf-8', mode='w') as f:
        for i, row in data_df.iterrows():
            f.write(row['text'])
            f.write('\n')
    ```

3. **生成预训练数据**
    使用 Megatron-LM 自带的预处理脚本将 JSON 转换为二进制格式：

    ```bash
    mkdir -p gpt_pretrain_data

    python tools/preprocess_data.py \
        --input alpaca_json.json \
        --output-prefix ./gpt_pretrain_data/alpaca \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ./gpt2_tokenizer \
        --workers 8 \
        --append-eod
    ```

    > [!NOTE]
    >
    > - Megatron-LM 新版已将分词器实现迁移到 `megatron/core/tokenizers`，其中 `GPT2BPETokenizer` 会按 HuggingFace 模型 id `gpt2` 联网解析配置（离线时报 `Can't load the configuration of 'gpt2'`），因此本文统一使用 `HuggingFaceTokenizer` + 本地 Tokenizer 目录，后续训练脚本也需使用相同配置。

## 启动预训练

在 `Megatron-LM` 目录下创建预训练脚本 `train_distributed.sh`，脚本示例如下：

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
CKPT_DIR=./ckpt
TOKENIZER_PATH=<Specify path to directory>/gpt2_tokenizer
DATA_PATH=<Specify path and file prefix>_text_document
TP=2
PP=2
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
GPT_ARGS="
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 1000 \
    --init-method-std 0.01 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --initial-loss-scale 4096.0 \
    --disable-bias-linear \
    --lr-warmup-fraction 0.01 \
    --fp16
"
DATA_ARGS="
    --split 990,5,5 \
    --data-path $DATA_PATH \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
"
OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
```

请编辑 `train_distributed.sh`，并配置如下路径：

```bash
CKPT_DIR=./ckpt
TOKENIZER_PATH=./gpt2_tokenizer
DATA_PATH=./gpt_pretrain_data/alpaca_text_document
```

**表 1**  train_distributed.sh参数配置

|参数|说明|
|-|-|
|`CKPT_DIR`|权重文件路径|
|`TOKENIZER_PATH`|tokenizer目录，需与数据预处理使用同一目录（含 `config.json`、`vocab.json`、`merges.txt`）|
|`DATA_PATH`|数据集文件|

> [!NOTE]
>
> - 当前示例脚本中 `NPUS_PER_NODE=8` 表示需要8个NPU，如果您的NPU数量低于此配置，请相应调小 `NPUS_PER_NODE`，并同步调整 `TP`、`PP` 等并行参数（需满足 `TP x PP ≤ NPUS_PER_NODE x NNODES`），否则可能遇到OOM等问题。
> - 脚本中的部分参数（如 `--hidden-size`、`--num-layers`、`--seq-length`）需根据实际场景进行适配，避免OOM等现象。
> - `--transformer-impl` 默认推荐 `transformer_engine`（与仓库 CI 脚本一致，基于 TransformerEngineNPU 提供 Transformer 加速）；如仅需验证最小可运行链路且未安装 TransformerEngineNPU，可改为 `local`，但将无法使用昇腾 Transformer 加速算子。

运行脚本启动预训练：

```bash
bash ./train_distributed.sh
```

## 预期输出

训练启动后，您将在控制台看到类似以下的日志输出：

```text
> initializing torch distributed ...
> compiling dataset index builder ...
> building GPT model ...
> building train, validation, and test datasets ...
> finished creating GPT datasets ...
done with setup ...
training ...
iteration        1/    1000 | consumed samples:            4 | elapsed time per iteration (ms): 3986.7 | throughput per GPU (TFLOP/s/GPU): 0.1 | learning rate: 1.000000E-07 | global batch size:     4 | lm loss: 1.041716E+01 | loss scale: 1.0 | grad norm: 15.144 | params norm: 136.026 | number of skipped iterations:   0 | number of nan iterations:   0 |Number of parameters in transformer block in billions:  0.05
iteration        2/    1000 | consumed samples:            8 | elapsed time per iteration (ms): 2173.0 | throughput per GPU (TFLOP/s/GPU): 0.2 | learning rate: 2.000000E-07 | global batch size:     4 | lm loss: 1.041835E+01 | loss scale: 1.0 | grad norm: 15.263 | params norm: 136.026 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
```

## 停止与清理

**停止训练**：

- 前台运行时，按 `Ctrl+C` 即可优雅停止训练。
- 如果训练脚本已被挂起或在后台运行，可以使用 `kill <PID>` 终止进程。

**清理训练产物**：

- 检查点（Checkpoints）：保存在脚本中 `CKPT_DIR` 指定的目录。
- 日志：通常输出在控制台或重定向到文件。如需清理，可直接删除日志文件。

> [!Troubleshooting]
>
> - **问题**：运行时提示 `OOM (Out of Memory)`。
>   - **解决**：减小 `--micro-batch-size`、`--seq-length` 或 `--hidden-size` 参数值。
> - **问题**：`torchrun` 启动失败，提示 `Address already in use`。
>   - **解决**：更换 `MASTER_PORT` 端口号（如 6002, 6003）。
> - **问题**：NPU 设备不可见（`torch.npu.is_available()` 返回 `False`）。
>   - **解决**：执行 `npu-smi info` 确认设备是否在线，并核对 CANN 与驱动版本是否匹配，详见《[安装指导](install_guide.md)》常见问题 Q2。
> - **问题**：数据预处理或训练启动时报 tokenizer / 数据路径错误。
>   - **解决**：确认 `TOKENIZER_PATH`、`DATA_PATH` 已替换为实际路径，且 `DATA_PATH` 指向 `*_text_document` 前缀（二进制数据文件为 `<DATA_PATH>.bin` 与 `<DATA_PATH>.idx`）。
> - **问题**：数据预处理或训练启动时报 `OSError: Can't load the configuration of 'gpt2'`。
>   - **解决**：这是使用 `GPT2BPETokenizer` 且环境无法访问 HuggingFace 导致的。请按本文示例改用 `--tokenizer-type HuggingFaceTokenizer --tokenizer-model <本地目录>`，并确认该目录中存在 `config.json`。
> - **问题**：启动时报 `No module named 'megatron_adaptor'`。
>   - **解决**：确认已按安装指导完成 MegatronAdaptor 安装，且当前 Python 环境已加载 CANN 环境变量（`source /usr/local/Ascend/cann/set_env.sh`）。
> - **问题**：克隆 Megatron-LM 源码或下载 Tokenizer、数据集时，访问 GitHub / HuggingFace 超时或失败。
>   - **解决**：这是网络环境限制，并非配置错误。请自行配置代理，或使用国内可访问的渠道（如 GitCode、Gitee 镜像拉取源码，ModelScope 等社区下载相同资源）；从第三方渠道获取后，请务必核对仓库地址、分支及文件的正确性与安全性。
