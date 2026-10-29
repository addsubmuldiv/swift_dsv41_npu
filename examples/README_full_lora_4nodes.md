# 四机完整模型 LoRA 验证

4 台 × 每台 16 张 64GB NPU。各节点使用已经安装的环境和共享盘上的同一份代码。
在主节点更新共享仓库：`git pull --ff-only`。无需另行下载数据，默认使用仓库自带的 32 条测试样本。

在**四台容器各自的仓库目录**分别执行以下对应的一条命令。
先启动主节点，脚本自动选取本机 IPv4 和通信网卡，将主节点地址原子写入共享输出目录；
其余三台自动读取，不需要填写 IP：

```bash
# 主节点
bash examples/train_full_lora_4nodes.sh 0
# 第二台
bash examples/train_full_lora_4nodes.sh 1
# 第三台
bash examples/train_full_lora_4nodes.sh 2
# 第四台
bash examples/train_full_lora_4nodes.sh 3
```

只运行主节点不会自动拉起其他三台。每次先启动 rank 0，让它更新共享地址文件，再启动其他三台。
同一任务使用相同的共享 `OUTPUT_DIR`；不同任务使用不同目录，避免读取其他任务的地址。
平台需要已配好容器互通和 NPU 跨机通信。主节点默认通过本地默认路由选取地址，
其余节点通过到主节点的路由选取网卡，无需访问外网。多网卡环境若默认路由不适合节点间通信，
可在每台设置 `HCCL_SOCKET_IFNAME` 为本机通信网卡的完整名称；也仍支持第二个位置参数或 `MASTER_ADDR` 手动指定主节点地址。

默认模型路径 `/storage/models/DeepSeek-V4.1-Flash-1017`，使用完整 40 层、384 专家和两个 Engram。
文本 SFT，关闭视觉与 DSpark/MTP 辅助训练；不修改原始 config、权重或索引。
脚本会自动在共享盘生成完整模型的 Engram tokenizer map。

默认配置：64 卡，TP1/PP4/EP16/CP1，BF16 all-linear LoRA rank8/alpha16，
micro batch=1、global batch=64、最大长度128、训练10步，full/uniform 重计算。
使用非融合路径，不依赖 Triton；最后保存 Megatron LoRA checkpoint，不 merge、不导出 safetensors。
这是多机链路验证配置，不是正式数据或效果验证。

输出在 `.local/full_lora_check/`，四台日志在其 `logs/node*.log`。
可通过 `MODEL_DIR`、`DATASET`（本地 JSONL）、`ENGRAM_MAP`、`OUTPUT_DIR`、`TRAIN_ITERS`、
`MAX_LENGTH`、`MASTER_PORT` 覆盖默认值；同一任务四台保持这些设置相同。
新任务可指定新的 `OUTPUT_DIR`。加 `DRY_RUN=1` 可检查地址发现并打印命令，不启动训练。

本机只做了 shell 语法与当前 Swift 参数静态检查；完整模型 Engram map 的生成也已验证。
未启动四机完整模型训练，实际显存和通信以目标机器运行结果为准。
