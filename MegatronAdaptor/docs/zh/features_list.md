# 适配接口列表

## 使用说明

- **目录导航**：本文按功能大类组织，可直接通过左侧目录或章节标题跳转至对应参数表。
- **特性启用**：标注“须 `--optimization-level >= N` 才生效”的参数属于 MegatronAdaptor 功能特性（如 NPU Profiling、NPU DataDump、NPU 确定性计算），未达优化等级时参数虽可传入但不生效。
- **参数不生效排查**：先确认 `--optimization-level` 是否达标，再确认是否已执行 `import megatron_adaptor`；仍不生效时参照《[快速入门](quick_start.md)》的 Troubleshooting 章节。
- **标注约定**：“布尔开关”为 true/false 开关；“Megatron 原生参数”指由 Megatron-LM 定义、MegatronAdaptor 透传或重定向的参数。

## 激活函数 (Activation Functions)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--swiglu` | 使用 SwiGLU 激活函数 | 与 `--squared-relu`、`--quick-geglu` 互斥 |
| `--squared-relu` | 使用 Squared ReLU 激活函数 | 与 `--swiglu` 互斥 |
| `--openai-gelu` | 使用 OpenAI 近似 GELU 激活函数 | 遗留参数，无互斥校验 |
| `--quick-geglu` | 使用快速 GeGLU 激活函数 | 与 `--swiglu` 互斥 |
| `--use-te-activation-func` | 使用 Transformer Engine 实现的激活函数 | 布尔开关；与 bias_activation_fusion 互斥；仅支持 gelu/silu/relu |
| `--use-fused-weighted-squared-relu` | 启用融合的加权 Squared ReLU 内核 | 布尔开关；用于 MoE 场景的融合加权 Squared ReLU 内核 |
| `--no-bias-gelu-fusion` | 禁用 bias-GELU 融合内核 | 布尔开关；当 add_bias_linear=False 时自动禁用 |
| `--no-bias-swiglu-fusion` | 禁用 bias-SwiGLU 融合内核 | 布尔开关；只对 `--swiglu` 生效 |

## 注意力机制 (Attention Mechanisms)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--use-flash-attn` | 启用 Flash Attention 加速注意力计算 | 布尔开关 |
| `--flash-decode` | 启用 Flash 解码模式 | 布尔开关；不能与 `--no-rope-freq` 同时使用 |
| `--group-query-attention` | 启用 Group Query Attention (GQA)，多个 Q 头共享 KV 头 | 布尔开关；需配合 `--num-query-groups` 设置组数；与 `--multi-latent-attention` 互斥 |
| `--multi-latent-attention` | 启用 Multi-Latent Attention (MLA)，通过低秩压缩 KV 降低显存 | 布尔开关；需配合 `--q-lora-rank`、`--kv-lora-rank`、`--qk-head-dim`、`--v-head-dim` 使用；与 `--group-query-attention` 互斥 |
| `--q-lora-rank` | MLA 中 Q 矩阵的低秩压缩维度 | int，默认 None（MLA 配置中默认 512）；依赖 `--multi-latent-attention` |
| `--kv-lora-rank` | MLA 中 KV 矩阵的低秩压缩维度 | int，默认 32（MLA 配置中默认 512）；依赖 `--multi-latent-attention` |
| `--qk-head-dim` | MLA 中每个头的 QK 维度 | int，默认 128；依赖 `--multi-latent-attention` |
| `--qk-pos-emb-head-dim` | MLA 中位置编码部分的头维度 | int，默认 64；依赖 `--multi-latent-attention` |
| `--v-head-dim` | MLA 中每个头的 V 维度 | int，默认 128；依赖 `--multi-latent-attention` |
| `--qk-layernorm` | 对 Q、K 向量分别做 LayerNorm 归一化 | 布尔开关 |
| `--qk-l2-norm` | 对 Q、K 向量做 L2 归一化 | 布尔开关（Llama4 风格） |
| `--relative-attention-num-buckets` | 设置 T5 风格相对注意力偏置的桶数 | int，默认 32；仅 `--position-embedding-type relative` 时有效 |
| `--relative-attention-max-distance` | 设置相对注意力偏置的最大距离 | int，默认 128；依赖 `--position-embedding-type relative` |
| `--apply-query-key-layer-scaling` | 将注意力分数除以 √head_dim 进行缩放 | 布尔开关；启用时自动设置 `--attention-softmax-in-fp32` |
| `--attention-backend` | 指定注意力计算的后端实现 | 可选值：`auto`、`flash`、`fused`、`unfused`、`local`；默认 `auto` |
| `--attention-output-gate` | 在注意力输出上应用门控机制 | 布尔开关 |
| `--window-size` | 设置滑动窗口注意力的窗口大小 | tuple(int,int)，默认 None（全局注意力）；-1 表示无限窗口 |
| `--window-attn-skip-freq` | 窗口注意力的跳过频率，每隔 N 层使用一次全局注意力 | int；依赖 `--window-size` |
| `--disable-bias-linear` | 禁用所有线性层的偏置 | 布尔开关（将 add_bias_linear 设为 False） |
| `--add-qkv-bias` | 仅为 QKV 投影层添加偏置 | 布尔开关，默认 False；当 add_bias_linear=True 时自动设为 True |

## 位置编码 (Position Embeddings)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--position-embedding-type` | 位置编码类型 | 可选值：`learned_absolute`、`rope`、`yarn`、`mrope`、`relative`、`none`；默认 `learned_absolute` |
| `--use-rotary-position-embeddings` | 启用旋转位置编码 (RoPE) | 布尔开关（已废弃）；设置后强制 position_embedding_type='rope' |
| `--rope-type` | RoPE 的具体实现类型 | 可选值：`rope`、`yarn`；默认 None（MLA 默认 `yarn`，普通注意力默认 `rope`） |
| `--rotary-base` | RoPE 频率基数 θ | int，默认 10000 |
| `--rotary-percent` | 对每个头的前 N% 维度应用旋转编码 | float，默认 1.0（全部维度） |
| `--rotary-seq-len-interpolation-factor` | RoPE 序列长度插值因子，用于外推更长序列 | float，默认 None |
| `--use-rope-scaling` | 启用 RoPE 缩放（线性外推） | 布尔开关；需配合 `--rope-scaling-factor` |
| `--rope-scaling-factor` | RoPE 线性缩放因子 | float，默认 8.0；依赖 `--use-rope-scaling` |
| `--no-rope-freq` | 指定跳过 RoPE 的层频率（每 N 层不加 RoPE） | int 或 list，默认 None；不能与 `--flash-decode` 同时使用；int 时须整除 num_layers |
| `--rotary-interleaved` | 使用交错方式应用旋转维度（而非前后对半分） | 布尔开关；不能与 `--multi-latent-attention` 同时使用 |
| `--no-rope-fusion` | 禁用 RoPE 的融合内核优化 | 布尔开关 |
| `--rotary-scaling-factor` | 旋转缩放因子 | float，默认 1.0（MLA 配置中默认 40） |
| `--mscale` | YaRN 中的 mscale 参数，控制注意力温度 | float，默认 1.0 |
| `--mscale-all-dim` | YaRN 中对所有维度应用 mscale | float，默认 0.0 |
| `--mrope-section` | 指定多维 RoPE (M-RoPE) 的维度分段方式 | list of int；`--position-embedding-type mrope` 时必填 |
| `--max-position-embeddings` | 模型支持的最大位置编码长度 | int，默认 None；须 >= seq_length |

## 归一化 (Normalization)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--normalization` | 归一化层类型 | 可选值：`RMSNorm`、`LayerNorm`；默认 `LayerNorm` |
| `--norm-epsilon` | 归一化计算中的 epsilon，防止除零 | float，默认 1e-5 |
| `--apply-residual-connection-post-layernorm` | 残差连接接在 LayerNorm 之后（而非之前） | 布尔开关 |
| `--no-persist-layer-norm` | 禁用持久化 LayerNorm | 布尔开关 |
| `--apply-wd-to-qk-layernorm` | 对 QK LayerNorm 的参数也应用权重衰减 | 布尔开关 |
| `--fused-residual-rmsnorm` | 使用融合的 RMSNorm + 残差连接内核 | 布尔开关；依赖 `--normalization RMSNorm` |

## 混合精度训练 (Mixed Precision Training)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--bf16` | 使用 BF16 混合精度训练 | 布尔开关；与 `--fp16` 互斥 |
| `--fp16` | 使用 FP16 混合精度训练 | 布尔开关；与 `--bf16` 互斥 |
| `--fp32-residual-connection` | 残差连接使用 FP32 精度（提高数值稳定性） | 布尔开关 |
| `--fp8-format` | FP8 低精量化格式 | 可选值：`e4m3`、`hybrid`；依赖 TE 库 |
| `--fp8-recipe` | FP8 量化策略 | 可选值：`tensorwise`、`delayed`、`mxfp8`、`blockwise`；依赖 `--fp8-format` |
| `--fp8-margin` | FP8 amax 计算时的余量值 | int，默认 0 |
| `--fp8-interval` | FP8 缩放因子更新频率（每 N 步更新一次） | int，默认 1 |
| `--fp8-amax-history-len` | FP8 amax 历史窗口长度 | int，默认 1 |
| `--fp8-amax-compute-algo` | FP8 amax 计算策略 | 可选值：`most_recent`、`max` |
| `--no-fp8-wgrad` | 禁用 FP8 权重梯度计算 | 布尔开关 |
| `--fp8-param-gather` | 在参数 AllGather 阶段使用 FP8 通信 | 布尔开关；依赖 `--fp8-format` |
| `--fp8-quantizer-factory` | 自定义 FP8 量化器的 Python 实现路径 | str；用于 `custom` 量化策略 |
| `--reuse-grad-buf-for-mxfp8-param-ag` | MXFP8 参数收集时复用梯度缓冲区以节省显存 | 布尔开关；依赖 `--fp8-recipe mxfp8` |
| `--keep-fp8-transpose-cache` | 保留 FP8 转置缓存 | 布尔开关 |
| `--te-precision-config-file` | Transformer Engine 精度配置文件路径 | str (YAML/JSON) |
| `--first-last-layers-bf16` | 首尾层使用 BF16（其余层可用 FP8） | 布尔开关 |
| `--num-layers-at-start-in-bf16` | 起始使用 BF16 的层数 | int；依赖 `--first-last-layers-bf16` |
| `--num-layers-at-end-in-bf16` | 末尾使用 BF16 的层数 | int；依赖 `--first-last-layers-bf16` |
| `--grad-reduce-in-bf16` | 梯度 AllReduce 使用 BF16 通信 | 布尔开关 |
| `--disable-bf16-reduced-precision-matmul` | 禁用 BF16 低精度矩阵乘法累加 | 布尔开关 |
| `--attention-softmax-in-fp32` | 注意力 softmax 使用 FP32 计算 | 布尔开关；启用 `--apply-query-key-layer-scaling` 时自动开启 |
| `--fp16-lm-cross-entropy` | 语言模型交叉熵使用 FP16 计算 | 布尔开关；依赖 `--fp16` |
| `--loss-scale` | 固定损失缩放值（不使用动态缩放） | float，默认 None（使用动态缩放） |
| `--initial-loss-scale` | 动态损失缩放的初始值 | float，默认 2^32 |
| `--loss-scale-window` | 损失缩放窗口大小（连续 N 步无溢出则翻倍） | int，默认 1000 |
| `--min-loss-scale` | 最小损失缩放值 | float，默认 1.0 |
| `--hysteresis` | 损失缩放滞后参数（允许连续 N 次溢出再缩小） | int，默认 2 |
| `--accumulate-allreduce-grads-in-fp32` | 梯度 AllReduce 累积使用 FP32 | 布尔开关 |

## 并行策略 (Parallelism)

### 张量并行 (Tensor Parallelism)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--tensor-model-parallel-size` | 张量并行度，将模型权重按列/行切分到多个 GPU | int，默认 1 |
| `--tp-comm-overlap` | 启用 TP 通信与计算重叠（AllGather/ReduceScatter） | 布尔开关，默认 False；开启后以下 disable 参数才生效 |
| `--tp-comm-overlap-cfg` | 指定 TP 通信重叠配置 YAML 文件路径 | str (YAML)；依赖 `--tp-comm-overlap` |
| `--disable-tp-comm-bulk-wgrad` | 禁用 TP 通信批量 wgrad 重叠 | 布尔开关；依赖 `--tp-comm-overlap` |
| `--disable-tp-comm-bulk-dgrad` | 禁用 TP 通信批量 dgrad 重叠 | 布尔开关；依赖 `--tp-comm-overlap` |
| `--disable-tp-comm-overlap-ag` | 禁用 TP 通信 AllGather 重叠 | 布尔开关；依赖 `--tp-comm-overlap` |
| `--disable-tp-comm-overlap-rs` | 禁用 TP 通信 ReduceScatter 重叠 | 布尔开关；依赖 `--tp-comm-overlap` |
| `--disable-tp-comm-split-ag` | 禁用 TP 通信 Split AllGather | 布尔开关；依赖 `--tp-comm-overlap` |
| `--disable-tp-comm-split-rs` | 禁用 TP 通信 Split ReduceScatter | 布尔开关；依赖 `--tp-comm-overlap` |
| `--no-clone-scatter-output-in-embedding` | 禁用 Embedding 中 scatter 输出的 clone | 布尔开关 |

### 流水线并行 (Pipeline Parallelism)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--pipeline-model-parallel-size` | 流水线并行度，将模型按层切分到多个 stage | int，默认 1 |
| `--pipeline-model-parallel-layout` | 自定义流水线并行布局（每 stage 的层分配） | str；与 `--decoder-first/last-pipeline-num-layers` 互斥 |
| `--num-layers-per-virtual-pipeline-stage` | 每个虚拟流水线阶段的层数（启用 VPP） | int；需能被总层数整除 |
| `--num-virtual-stages-per-pipeline-rank` | 每个流水线 rank 的虚拟阶段数 | int；与 `--num-layers-per-virtual-pipeline-stage` 二选一 |
| `--decoder-first-pipeline-num-layers` | 解码器第一个流水线阶段的层数 | int；与 `--pipeline-model-parallel-layout` 互斥 |
| `--decoder-last-pipeline-num-layers` | 解码器最后一个流水线阶段的层数 | int；与 `--pipeline-model-parallel-layout` 互斥 |
| `--account-for-embedding-in-pipeline-split` | 在流水线切分时将 Embedding 层参与层数计算 | 布尔开关 |
| `--account-for-loss-in-pipeline-split` | 在流水线切分时将 Loss 层参与层数计算 | 布尔开关 |
| `--microbatch-group-size-per-virtual-pipeline-stage` | VPP 中每个虚拟 stage 连续执行的 micro-batch 数量 | int，默认 1；依赖 VPP 开启 |
| `--pipeline-model-parallel-comm-backend` | 流水线并行通信后端 | 可选值：`nccl`、`ucc` |
| `--overlap-p2p-communication-warmup-flush` | 重叠 P2P 通信预热刷新阶段 | 布尔开关 |
| `--no-overlap-p2p-communication` | 禁用 P2P 通信重叠 | 布尔开关；与 `--overlap-p2p-communication-warmup-flush` 互斥 |

### 序列并行 (Sequence Parallelism)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--sequence-parallel` | 启用序列并行，将 LayerNorm/Dropout 按序列维度切分 | 布尔开关；依赖 `--tensor-model-parallel-size > 1` |

### 上下文并行 (Context Parallelism)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--context-parallel-size` | 上下文并行度，将长序列切分到多个 GPU | int，默认 1 |
| `--cp-comm-type` | 上下文并行通信类型 | 可选值：`p2p`、`a2a`、`a2a+p2p`、`all_gather` |
| `--hierarchical-context-parallel-sizes` | 层次化 CP 各层大小 | list of int |
| `--max-seqlen-per-dp-cp-rank` | 每个 (DP rank, CP rank) 本地批次最大序列长度上限 | int；用于变长序列场景 |

### 专家并行 (Expert Parallelism)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--expert-model-parallel-size` | 专家模型并行度，将 MoE 专家分布到多个 GPU | int，默认 1；依赖 `--num-experts` |
| `--expert-tensor-parallel-size` | 专家张量并行度 | int，默认等于 `--tensor-model-parallel-size` |
| `--overlap-moe-expert-parallel-comm` | 重叠 MOE 专家并行通信与计算 | 布尔开关 |
| `--ep-overlap-early-attn-memory-release` | 在 EP overlap 中更早释放 attention 内存 | 布尔开关；依赖 `--overlap-moe-expert-parallel-comm` |

### 数据并行 (Data Parallelism)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--data-parallel-sharding-strategy` | 数据并行分片策略 | 可选值：`no_shard`、`optim`、`optim_grads`、`optim_grads_params`；默认 `optim_grads_params` |
| `--outer-dp-sharding-strategy` | 外层数据并行分片策略 | 可选值：`no_shard`、`optim`；默认 `no_shard` |
| `--data-parallel-random-init` | 数据并行随机初始化 | 布尔开关 |
| `--ddp-num-buckets` | DDP bucket 数量 | int，默认 None（自动） |
| `--ddp-bucket-size` | DDP bucket 大小 | int，默认 None（自动） |
| `--ddp-pad-buckets-for-high-nccl-busbw` | 为高 NCCL 带宽填充 bucket | 布尔开关 |
| `--ddp-reduce-scatter-with-fp32-accumulation` | DDP ReduceScatter 使用 FP32 累积 | 布尔开关 |
| `--ddp-average-in-collective` | 在集合操作中做平均 | 布尔开关 |

## FSDP (Fully Sharded Data Parallel)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--use-megatron-fsdp` | 启用 Megatron FSDP 全分片数据并行 | 布尔开关 |
| `--fsdp-double-buffer` | 启用 FSDP 双缓冲（预取下一轮参数） | 布尔开关；依赖 `--use-megatron-fsdp` |
| `--torch-fsdp2-no-reshard-after-forward` | FSDP2 前向后不重分片（类似 ZeRO-2） | 布尔开关 |

## 优化器 (Optimizers)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--optimizer` | 优化器类型 | 可选值：`adam`、`sgd`、`muon`、`dist_muon`、`lion`、`soap`、`adaptive_muon`；默认 `adam` |
| `--adam-beta1` | Adam 一阶动量衰减率 | float，默认 0.9 |
| `--adam-beta2` | Adam 二阶动量衰减率 | float，默认 0.999 |
| `--adam-eps` | Adam epsilon，防止除零 | float，默认 1e-8 |
| `--sgd-momentum` | SGD 动量系数 | float，默认 0.9；依赖 `--optimizer sgd` |
| `--muon-momentum` | Muon 动量系数 | float，默认 0.9；依赖 `--optimizer muon` 或 `dist_muon` |
| `--muon-no-split-qkv` | Muon 不拆分 QKV 权重（作为整体优化） | 布尔开关（默认拆分）；依赖 Muon 优化器 |
| `--muon-nesterov` | Muon 启用 Nesterov 动量 | 布尔开关；依赖 Muon 优化器 |
| `--muon-scale-mode` | Muon 缩放模式 | 可选值：`spectral`、`unit_rms_norm`、`shape_scaling`；默认 `spectral`；依赖 Muon 优化器 |
| `--muon-fp32-matmul-prec` | Muon FP32 矩阵乘法精度 | 可选值：`low`、`medium`、`high`；默认 `medium`；依赖 Muon 优化器 |
| `--muon-coefficient-type` | Muon 牛顿-舒尔茨系数类型 | str；依赖 Muon 优化器 |
| `--muon-num-ns-steps` | Muon 牛顿-舒尔茨迭代步数 | int，默认 5；依赖 Muon 优化器 |
| `--muon-tp-mode` | Muon TP 模式 | 可选值：`blockwise`、`duplicated`、`distributed`；默认 `blockwise`；依赖 Muon 优化器 |
| `--muon-extra-scale-factor` | Muon 额外缩放因子 | float，默认 1.0；依赖 Muon 优化器 |
| `--muon-scalar-optimizer` | Muon 标量参数使用的优化器 | 可选值：`adam`、`lion`；默认 `adam`；依赖 Muon 优化器 |
| `--lion-beta1` | Lion 优化器一阶动量 | float，默认 0.95；依赖 `--optimizer lion` |
| `--lion-beta2` | Lion 优化器二阶动量 | float，默认 0.98；依赖 `--optimizer lion` |
| `--use-distributed-optimizer` | 启用分布式优化器（ZeRO-1 风格） | 布尔开关 |
| `--num-distributed-optimizer-instances` | 分布式优化器实例数 | int，默认 1；依赖 `--use-distributed-optimizer` |
| `--distrib-optim-fully-reshardable-mem-efficient` | 启用完全可重分片分布式优化器（节省显存） | 布尔开关；依赖 `--use-distributed-optimizer` |
| `--override-opt-param-scheduler` | 覆盖优化器参数调度器（使用命令行值） | 布尔开关；与 `--use-checkpoint-opt-param-scheduler` 互斥 |
| `--use-checkpoint-opt-param-scheduler` | 使用检查点中保存的优化器参数调度器 | 布尔开关；与 `--override-opt-param-scheduler` 互斥 |

## 学习率调度 (Learning Rate Scheduling)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--lr` | 初始学习率 | float，必填 |
| `--min-lr` | 最小学习率（衰减下限） | float，默认 0.0 |
| `--decoupled-lr` | 解耦学习率（用于 µP 等场景） | float，默认 None |
| `--decoupled-min-lr` | 解耦最小学习率 | float，默认 None |
| `--lr-decay-style` | 学习率衰减策略 | 可选值：`constant`、`linear`、`cosine`、`inverse-square-root`、`WSD`；默认 `linear` |
| `--lr-decay-iters` | 学习率衰减的迭代数 | int；与 `--lr-decay-samples` 二选一 |
| `--lr-decay-samples` | 学习率衰减的样本数 | int；与 `--lr-decay-iters` 二选一 |
| `--lr-warmup-fraction` | 学习率预热占总训练的比例 | float (0~1)；与 `--lr-warmup-iters`/`--lr-warmup-samples` 互斥 |
| `--lr-warmup-init` | 学习率预热初始值 | float，默认 0.0 |
| `--lr-warmup-iters` | 学习率预热的迭代数 | int；与 `--lr-warmup-fraction` 互斥 |
| `--lr-warmup-samples` | 学习率预热的样本数 | int；与 `--lr-warmup-fraction` 互斥 |
| `--lr-wsd-decay-style` | WSD 衰减阶段的子策略 | 可选值：`exponential`、`linear`、`cosine`、`minus_sqrt`；默认 `exponential`；依赖 `--lr-decay-style WSD` |
| `--lr-wsd-decay-iters` | WSD 衰减阶段的迭代数 | int；依赖 `--lr-decay-style WSD` |
| `--lr-wsd-decay-samples` | WSD 衰减阶段的样本数 | int；依赖 `--lr-decay-style WSD` |
| `--weight-decay` | 权重衰减系数 | float，默认 0.01 |
| `--weight-decay-incr-style` | 权重衰减递增策略 | 可选值：`constant`、`linear`、`cosine`；默认 `constant` |
| `--start-weight-decay` | 权重衰减起始值 | float；依赖 `--weight-decay-incr-style` 非 `constant` |
| `--end-weight-decay` | 权重衰减结束值 | float；依赖 `--weight-decay-incr-style` 非 `constant` |
| `--no-weight-decay-cond-type` | 无权重衰减条件类型 | str |

## 重计算策略 (Recompute)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--recompute-activations` | 重计算激活开关 | 布尔开关；等价于 `--recompute-granularity selective` |
| `--recompute-granularity` | 重计算粒度 | 可选值：`full`（整层重算）、`selective`（仅重算部分）；与 `--recompute-activations` 同时配置时生效选择性重计算 |
| `--recompute-method` | 重计算方法 | 可选值：`uniform`（均匀分层）、`block`（按块）；仅 `--recompute-granularity full` 时可用；需同时配置 `--recompute-num-layers` |
| `--recompute-num-layers` | 每个重计算块包含的层数 | int；依赖 `--recompute-granularity full` + `--recompute-method` |
| `--recompute-modules` | 指定需要重计算的模块 | 可选值：`core_attn`、`moe_act`、`layernorm`、`mla_up_proj`、`mlp`、`moe`、`shared_experts`；可多选 |
| `--distribute-saved-activations` | 将保存的激活值分布到多卡上（需 TP > 1） | 布尔开关；依赖 `--recompute-granularity full` + `--recompute-method`；与 `--sequence-parallel` 互斥 |

## MOE (Mixture of Experts)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--num-experts` | MoE 专家数量 | int；启用 MoE 的必填参数 |
| `--moe-layer-freq` | MoE 层与 Dense 层的交替频率 | int，默认 1 |
| `--moe-ffn-hidden-size` | MoE 专家 FFN 隐藏维度 | int |
| `--moe-latent-size` | MoE 潜在投影维度 | int |
| `--moe-shared-expert-intermediate-size` | 共享专家总 FFN 隐藏大小 | int |
| `--moe-shared-expert-gate` | 启用共享专家门控机制 | 布尔开关 |
| `--moe-shared-expert-overlap` | 启用共享专家计算与调度器通信的重叠 | 布尔开关；依赖 `--moe-shared-expert-intermediate-size`；与 `--overlap-moe-expert-parallel-comm` 互斥 |
| `--moe-router-topk` | 每个 token 路由到的专家数量 | int，默认 2 |
| `--moe-router-score-function` | MoE 路由评分函数 | 可选值：`softmax`、`sigmoid`、`sqrtsoftplus`；默认 `softmax` |
| `--moe-router-pre-softmax` | 启用 pre-softmax 路由 | 布尔开关 |
| `--moe-router-topk-scaling-factor` | 路由 Top-K 缩放因子 | float；仅 `--moe-router-pre-softmax` 启用时有效 |
| `--moe-router-num-groups` | 路由分组数 | int，默认 None |
| `--moe-router-group-topk` | 路由组内 Top-K | int；依赖 `--moe-router-num-groups` |
| `--moe-router-dtype` | 路由计算数据类型 | 可选值：`fp32`、`fp64`；默认 None |
| `--moe-router-load-balancing-type` | 路由负载均衡策略 | 可选值：`aux_loss`、`seq_aux_loss`、`global_aux_loss`、`sinkhorn`、`none`；默认 `aux_loss` |
| `--moe-router-enable-expert-bias` | 启用带动态专家偏置的 TopK 路由 | 布尔开关 |
| `--moe-router-bias-update-rate` | 专家偏置更新速率 | float，默认 1e-3；依赖 `--moe-router-enable-expert-bias` |
| `--moe-router-fusion` | 启用 MoE TopK 路由和辅助损失计算的融合 | 布尔开关 |
| `--moe-router-force-load-balancing` | [实验性] 强制负载均衡路由（仅用于基准测试） | 布尔开关 |
| `--moe-enable-routing-replay` | 启用路由重放功能 | 布尔开关 |
| `--moe-aux-loss-coeff` | 辅助损失缩放系数 | float，默认 0.0 |
| `--moe-z-loss-coeff` | Z-loss 缩放系数 | float，默认 None（建议起始值 1e-3） |
| `--moe-input-jitter-eps` | 输入抖动 epsilon（ST-MoE 输入抖动正则化） | float，默认 None |
| `--moe-token-dispatcher-type` | Token 调度器类型 | 可选值：`allgather`、`alltoall`、`flex`；默认 `allgather` |
| `--moe-permute-fusion` | 在 token 调度期间融合 token 重排操作 | 布尔开关 |
| `--moe-expert-capacity-factor` | 每个专家的容量因子 | float，默认 None（不限制） |
| `--moe-pad-expert-input-to-capacity` | 将专家输入填充到容量长度 | 布尔开关；依赖 `--moe-expert-capacity-factor` |
| `--moe-token-drop-policy` | token 丢弃策略 | 可选值：`probs`、`position`；依赖 `--moe-expert-capacity-factor` |
| `--moe-grouped-gemm` | 启用分组 GEMM | 布尔开关；需要 CUTLASS 2.8+ |
| `--moe-apply-probs-on-input` | 将路由概率应用在专家输入上 | 布尔开关 |
| `--moe-use-upcycling` | 加载 Dense 模型检查点并转换为 MoE 模型 | 布尔开关；需配合 `--moe-upcycling-granularity` |
| `--moe-upcycling-granularity` | 专家隐藏大小相对于 Dense FFN 的缩小倍数 | int，默认 1；依赖 `--moe-use-upcycling` |
| `--moe-per-layer-logging` | 启用 MoE 逐层日志记录 | 布尔开关 |

## 检查点 (Checkpointing)

### 保存与加载

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--save` | 检查点保存路径 | str |
| `--save-interval` | 检查点保存间隔（每 N 步保存一次） | int |
| `--load` | 检查点加载路径 | str |
| `--no-load-optim` | 不加载优化器状态 | 布尔开关 |
| `--no-load-rng` | 不加载随机数生成器状态 | 布尔开关 |

### 检查点格式与转换

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--ckpt-format` | 保存检查点的格式 | 可选值：`torch`、`torch_dist`、`torch_dcp`、`fsdp_dtensor`；默认 `torch_dist` |
| `--ckpt-convert-format` | 检查点转换的目标格式 | str |
| `--ckpt-convert-save` | 保存时转换为指定格式 | str |
| `--ckpt-convert-update-legacy-dist-opt-format` | 转换时更新旧版分布式优化器格式 | 布尔开关 |
| `--auto-detect-ckpt-format` | 自动检测检查点格式 | 布尔开关 |
| `--dist-ckpt-optim-fully-reshardable` | 优化器状态完全可重分片保存 | 布尔开关 |
| `--dist-ckpt-strictness` | 分布式检查点加载严格程度 | 可选值：`assume_ok_unexpected`、`log_unexpected`、`log_all`、`raise_unexpected`、`raise_all`、`return_unexpected`、`return_all`、`ignore_all`；默认 `assume_ok_unexpected` |

### 并行保存与加载

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--async-save` | 启用异步保存（不阻塞训练） | 布尔开关；需要 `--use-persistent-ckpt-worker`；仅支持 torch_dist 格式 |
| `--ckpt-fully-parallel-save` | 启用完全并行保存 | 布尔开关（已废弃）；与 `--no-ckpt-fully-parallel-save` 互斥 |
| `--no-ckpt-fully-parallel-save` | 禁用完全并行保存 | 布尔开关；与 `--ckpt-fully-parallel-save` 互斥 |
| `--ckpt-fully-parallel-load` | 启用全并行方式加载检查点 | 布尔开关 |
| `--ckpt-fully-parallel-load-exchange-algo` | 全并行加载交换算法 | 可选值：`broadcast`、`gather_rounds`、`gather_object`；默认 `broadcast` |
| `--ckpt-fully-parallel-save-process-group` | 全并行保存进程组 | str |
| `--ckpt-fully-parallel-load-process-group` | 全并行加载进程组 | str |
| `--ckpt-assume-constant-structure` | 假设检查点模型结构恒定，跳过结构检查 | 布尔开关 |
| `--use-persistent-ckpt-worker` | 使用持久化检查点工作线程 | 布尔开关 |

### 微调与预训练

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--finetune` | 启用微调模式（不加载优化器/调度器状态） | 布尔开关 |
| `--pretrained-checkpoint` | 预训练检查点路径 | str |
| `--ckpt-step` | 加载检查点的具体步骤 | int |
| `--use-checkpoint-args` | 使用检查点中保存的训练参数 | 布尔开关 |
| `--use-mp-args-from-checkpoint-args` | 使用检查点中保存的模型并行参数 | 布尔开关；依赖 `--use-checkpoint-args` |
| `--exit-on-missing-checkpoint` | 检查点不存在时退出程序 | 布尔开关 |

## CPU Offload

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--cpu-offloading-num-layers` | 需要 Offload 到 CPU 的层数 | int，默认 0 |
| `--optimizer-cpu-offload` | 启用优化器状态 CPU Offload | 布尔开关 |
| `--optimizer-offload-fraction` | 优化器 Offload 到 CPU 的比例 | float (0~1)，默认 1.0；依赖 `--optimizer-cpu-offload` |
| `--use-torch-optimizer-for-cpu-offload` | 使用 PyTorch 优化器进行 CPU Offload | 布尔开关；依赖 `--optimizer-cpu-offload` |
| `--overlap-cpu-optimizer-d2h-h2d` | 重叠 CPU 优化器 D2H/H2D 数据传输 | 布尔开关；依赖 `--optimizer-cpu-offload` |
| `--no-pin-cpu-grads` | 不固定 CPU 梯度内存（禁用 pinned memory） | 布尔开关 |
| `--no-pin-cpu-params` | 不固定 CPU 参数内存（禁用 pinned memory） | 布尔开关 |

## Tokenizer & 数据集

### Tokenizer

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--tokenizer-type` | 分词器类型 | 可选值：多种（如 `BertWordPieceLowerCase`、`GPT2BPETokenizer`、`HuggingFaceTokenizer` 等，完整列表以对应版本 `--help` 为准） |
| `--tokenizer-model` | 分词器模型路径 | str |
| `--tokenizer-metadata` | 分词器元数据 JSON 文件路径 | str |
| `--tokenizer-special-tokens` | 特殊 token 列表 | str |
| `--tiktoken-pattern` | Tiktoken 模式 | 可选值：`v1`、`v2` |
| `--tiktoken-num-special-tokens` | Tiktoken 特殊 token 数 | int |
| `--tokenizer-sentencepiece-legacy` | 使用 SentencePiece 旧版兼容模式 | 布尔开关 |
| `--trust-remote-code` | 信任远程代码（加载自定义 tokenizer 时需要） | 布尔开关 |
| `--null-tokenizer-eod-id` | Null Tokenizer EOD ID | int |
| `--null-tokenizer-pad-id` | Null Tokenizer Pad ID | int |
| `--vocab-size` | 词汇表大小 | int |
| `--padded-vocab-size` | 填充后词汇表大小 | int |
| `--vocab-file` | 词汇表文件路径 | str |
| `--merge-file` | BPE 合并文件路径 | str |
| `--vocab-extra-ids` | 额外词汇 ID 数量 | int |
| `--make-vocab-size-divisible-by` | 词汇表大小对齐值 | int |

### 数据集与数据加载

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--data-path` | 数据路径 | str；与 `--data-args-path`、`--mock-data` 互斥 |
| `--train-data-path` | 训练数据路径 | str |
| `--valid-data-path` | 验证数据路径 | str |
| `--test-data-path` | 测试数据路径 | str |
| `--data-args-path` | 数据参数文件路径（替代 --data-path） | str；与 `--data-path` 互斥 |
| `--per-split-data-args-path` | 每个分割数据参数的 JSON 文件路径 | str |
| `--per-dataset-sequences-path` | 每个数据集序列的 JSON 文件路径 | str |
| `--split` | 训练、验证和测试数据集的比例 | str，如 `"99,1,0"` |
| `--seq-length` | 序列长度 | int |
| `--encoder-seq-length` | 编码器序列长度 | int；与 `--seq-length` 互斥 |
| `--decoder-seq-length` | 解码器序列长度 | int |
| `--dataloader-type` | 数据加载器类型 | str |
| `--dataloader-fast-cache-load` | 启用快速缓存加载路径 | 布尔开关；依赖 `--data-cache-path`；与 `--data-path` 互斥 |
| `--dataloader-defer-npy-index-mmap` | 延迟加载 .npy 文件的内存映射 | 布尔开关 |
| `--data-cache-path` | 数据缓存索引文件路径 | str |
| `--no-mmap-bin-files` | 不内存映射二进制文件 | 布尔开关 |
| `--mock-data` | 使用模拟数据（跳过数据加载和验证） | 布尔开关；与 `--data-path` 等互斥；不可用于生产 |
| `--num-workers` | 数据加载工作线程数 | int |
| `--num-dataset-builder-threads` | 数据集构建线程数 | int |
| `--object-storage-cache-path` | 对象存储缓存路径 | str |
| `--mid-level-dataset-surplus` | 中层数据集冗余比例 | float |
| `--sample-rate` | 采样率 | float |
| `--mask-prob` | 掩码概率 | float |
| `--short-seq-prob` | 短序列概率 | float |
| `--reset-position-ids` | 在文档结束符后重置位置 ID | 布尔开关 |
| `--reset-attention-mask` | 在文档结束符后重置注意力掩码 | 布尔开关 |
| `--eod-mask-loss` | 对文档结束位置掩码损失 | 布尔开关 |
| `--no-create-attention-mask-in-dataloader` | 数据加载器不创建注意力掩码 | 布尔开关 |
| `--allow-ambiguous-pad-tokens` | 允许模糊填充 token | 布尔开关 |
| `--phase-transition-iterations` | 阶段转换迭代 | str |

### FIM (Fill-in-the-Middle)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--fim-data` | 启用 FIM 数据 | 布尔开关 |
| `--fim-rate` | FIM 转换概率 | float；依赖 `--fim-data` |
| `--fim-spm-rate` | FIM SPM 格式概率 | float；依赖 `--fim-data` |
| `--fim-split-sample` | FIM 样本分割字符串 | str；依赖 `--fim-data` |
| `--fim-fragment-rate` | FIM 片段比例 | float；依赖 `--fim-data` |
| `--fim-no-prefix` | FIM 无前缀标记 | 布尔开关；依赖 `--fim-data` |
| `--fim-prefix-token` | FIM 前缀 token | int；依赖 `--fim-data` |
| `--fim-middle-token` | FIM 中间 token | int；依赖 `--fim-data` |
| `--fim-suffix-token` | FIM 后缀 token | int；依赖 `--fim-data` |
| `--fim-pad-token` | FIM 填充 token | int；依赖 `--fim-data` |
| `--fim-eod-token` | FIM 文档结束 token | int；依赖 `--fim-data` |

## Hybrid 模型 (Mamba + Transformer)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--mamba-state-dim` | Mamba 状态维度 | int，默认 128；启用 Mamba 层时必填 |
| `--mamba-head-dim` | Mamba 头维度 | int，默认 64；启用 Mamba 层时必填 |
| `--mamba-num-groups` | Mamba 分组数 | int，默认 8；启用 Mamba 层时必填 |
| `--mamba-num-heads` | Mamba 头数 | int |
| `--mtp-num-layers` | 多 Token 预测 (MTP) 层数 | int，默认 None（不启用） |
| `--mtp-loss-scaling-factor` | MTP 损失缩放因子 | float，默认 0.1；依赖 `--mtp-num-layers > 0` |
| `--mtp-standalone` | MTP 独立模式 | 布尔开关；依赖 `--mtp-num-layers > 0` |
| `--calculate-per-token-loss` | 计算每个 token 的损失（而非按序列平均） | 布尔开关 |

## 分布式通信优化

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--use-tp-pp-dp-mapping` | 使用 TP/PP/DP 映射 | 布尔开关 |
| `--overlap-grad-reduce` | 重叠梯度归约与计算 | 布尔开关 |
| `--overlap-param-gather` | 重叠参数收集与计算 | 布尔开关；依赖分布式优化器或 FSDP；必须同时启用 `--overlap-grad-reduce` |
| `--overlap-param-gather-with-optimizer-step` | 重叠参数收集与优化器步骤 | 布尔开关；依赖 `--overlap-param-gather` |
| `--overlap-dispatch-backward-with-experts-wgrad` | 重叠反向调度与专家 wgrad | 布尔开关 |
| `--distributed-backend` | 分布式通信后端 | 可选值：`nccl`、`gloo`；默认 `nccl` |
| `--distributed-timeout-minutes` | 分布式通信超时时间 | int，默认 10（分钟） |
| `--use-sharp` | 启用 SHARP 通信加速 | 布尔开关 |
| `--sharp-enabled-group` | SHARP 启用组 | 可选值：`dp`、`dp_replica`；依赖 `--use-sharp` |
| `--create-all-gather-group` | 创建 all-gather 组 | 布尔开关 |
| `--suggested-communication-unit-size` | 建议通信单元大小 | int |
| `--disable-symmetric-registration` | 禁用对称注册 | 布尔开关 |
| `--symmetric-ar-type` | 对称内存全归约类型 | str |
| `--local-rank` | 本地秩 | int |
| `--no-align-grad-reduce` | 禁用梯度归约对齐 | 布尔开关 |
| `--no-align-param-gather` | 禁用参数收集对齐 | 布尔开关 |
| `--wgrad-deferral-limit` | wgrad 延迟限制 | int，默认 0 |
| `--lazy-mpu-init` | 延迟 MPU 初始化 | 布尔开关 |
| `--disable-gloo-process-groups` | 禁用 Gloo 进程组 | 布尔开关 |

## Fusion 优化

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--no-gradient-reduce-div-fusion` | 禁用梯度归约除法融合 | 布尔开关 |
| `--no-gradient-accumulation-fusion` | 禁用梯度累积融合 | 布尔开关 |
| `--cross-entropy-loss-fusion` | 启用交叉熵损失融合 | 布尔开关 |
| `--cross-entropy-fusion-impl` | 交叉熵融合实现方式 | 可选值：`native`、`te`；依赖 `--cross-entropy-loss-fusion` |
| `--no-masked-softmax-fusion` | 禁用掩码 softmax 融合 | 布尔开关 |
| `--no-bias-dropout-fusion` | 禁用 bias-dropout 融合 | 布尔开关 |
| `--fused-residual-rmsnorm` | 使用融合的 RMSNorm + 残差连接内核 | 布尔开关；依赖 `--normalization RMSNorm` |
| `--disable-jit-fuser` | 禁用 JIT 融合器 | 布尔开关 |

## 训练控制

### 批次与迭代

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--micro-batch-size` | 微批次大小（每个 GPU 每步处理的样本数） | int，必填 |
| `--global-batch-size` | 全局批次大小 | int，必填；须为 micro-batch-size × DP 的倍数 |
| `--rampup-batch-size` | 批次预热配置 | str；与 `--train-iters` 互斥 |
| `--step-batch-size-schedule` | 每 step 的 MBS 调度 | str |
| `--decrease-batch-size-if-needed` | OOM 时自动降批次 | 布尔开关 |
| `--train-iters` | 训练迭代数 | int；与 `--train-samples` 互斥 |
| `--train-samples` | 训练样本数 | int；与 `--train-iters` 互斥 |
| `--eval-iters` | 评估迭代数 | int |
| `--eval-interval` | 评估间隔（每 N 步评估一次） | int，默认 None |
| `--start-eval-at-iter` | 开始评估的迭代 | int |
| `--eval-global-batch-size` | 评估全局批次大小 | int |
| `--eval-micro-batch-size` | 评估微批次大小 | int |
| `--iterations-to-skip` | 跳过的初始迭代数 | int |
| `--skip-train` | 跳过训练（仅评估） | 布尔开关 |
| `--full-validation` | 启用完整验证 | 布尔开关 |
| `--multiple-validation-sets` | 启用多验证集 | 布尔开关 |

### 模型架构

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--num-layers` | Transformer 层数 | int，必填 |
| `--encoder-num-layers` | 编码器层数 | int |
| `--decoder-num-layers` | 解码器层数 | int |
| `--hidden-size` | 隐藏层维度 | int，必填 |
| `--ffn-hidden-size` | FFN 隐藏层维度 | int；默认为 hidden_size × 4 |
| `--num-attention-heads` | 注意力头数 | int，必填 |
| `--num-query-groups` | 查询组数（GQA 时使用） | int |
| `--kv-channels` | KV 通道数 | int；默认为 hidden_size / num_attention_heads |
| `--untie-embeddings-and-output-weights` | 解绑嵌入和输出权重 | 布尔开关 |
| `--attention-dropout` | 注意力 dropout 概率 | float，默认 0.1 |
| `--hidden-dropout` | 隐藏层 dropout 概率 | float，默认 0.1 |
| `--use-mcore-models` | 使用 Megatron-Core 模型实现 | 布尔开关（已废弃，mcore 现为默认） |
| `--use-legacy-models` | 使用旧版模型实现 | 已从代码中移除 |
| `--transformer-impl` | Transformer 实现方式 | 可选值：`local`、`transformer_engine`、`inference_optimized`；默认 `transformer_engine` |
| `--heterogeneous-layers-config-path` | 异构层配置路径 | str |
| `--heterogeneous-layers-config-encoded-json` | 异构层配置 JSON | str；与 `--heterogeneous-layers-config-path` 互斥 |

### 初始化

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--seed` | 随机种子 | int，默认 1234 |
| `--init-method-std` | 初始化标准差 | float，默认 0.02 |
| `--embedding-init-method-std` | 嵌入初始化标准差 | float |
| `--init-model-with-meta-device` | 用 meta 设备初始化模型 | 布尔开关 |
| `--no-initialization` | 跳过初始化 | 布尔开关 |
| `--use-cpu-initialization` | 将模型初始化放在 CPU 上完成 | 布尔开关 |

### 梯度控制

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--clip-grad` | 梯度裁剪阈值（按全局范数裁剪） | float，默认 1.0 |
| `--no-check-for-nan-in-loss-and-grad` | 不检查损失和梯度中的 NaN | 布尔开关 |
| `--check-for-large-grads` | 检查大梯度 | 布尔开关 |
| `--check-for-spiky-loss` | 检测损失尖峰 | 布尔开关 |

### 内存管理

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--manual-gc` | 启用手动垃圾回收 | 布尔开关 |
| `--manual-gc-interval` | GC 间隔（每 N 步执行一次） | int，默认 0；依赖 `--manual-gc` |
| `--no-manual-gc-eval` | 评估时不做 GC | 布尔开关；依赖 `--manual-gc` |
| `--empty-unused-memory-level` | 内存清理级别 | 可选值：0、1、2；默认 0 |
| `--memory-snapshot-path` | 内存快照路径 | str，默认 `snapshot.pickle` |

### 训练稳定性与调试

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--deterministic-mode` | 启用确定性模式（牺牲性能换取可复现性） | 布尔开关 |
| `--npu-deterministic` | 启用 NPU 全链路确定性计算：HCCL 通信算子级确定性、关闭 Matmul K 轴平移、启用 PyTorch 确定性算法、固定 torch_npu RNG 种子（自动在 Megatron 种子初始化后注入）；另写入 PYTHONHASHSEED 审计值 | 布尔开关；须 `--optimization-level >= 2` 才生效；须配合固定 `--seed` 使用；若需固定 Python 哈希随机化，须在启动训练进程前 `export PYTHONHASHSEED=<seed>`（运行时写入不生效）；以性能换可复现性，建议仅排查/精度对照期间开启；与 `--deterministic-mode` 互补可同时开启 |
| `--rerun-mode` | 检测到 step 异常时的重跑策略 | 可选值：`disabled`、`validate_results`、`report_stats`；默认 `validate_results` |
| `--error-injection-rate` | 随机故障注入率 | int，默认 0 |
| `--error-injection-type` | 随机故障注入类型 | 可选值：`correct_result`、`transient_error`、`persistent_error`；默认 `transient_error` |
| `--straggler-ctrlr-port` | Straggler 检测通信端口 | int |
| `--straggler-minmax-count` | Straggler 统计保留样本数 | int |
| `--disable-straggler-on-startup` | 启动时禁用 Straggler 检测 | 布尔开关 |
| `--check-weight-hash-across-dp-replicas-interval` | DP 权重校验间隔 | int |
| `--train-sync-interval` | 训练同步间隔 | int |
| `--result-rejected-tracker-filename` | 结果拒绝跟踪文件 | str |
| `--batch-invariant-mode` | 启用批处理不变模式 | 布尔开关 |
| `--barrier-with-L1-time` | 启用一级时序屏障 | 布尔开关 |
| `--no-barrier-with-level-1-timing` | 禁用一级时序屏障 | 布尔开关；与 `--barrier-with-L1-time` 互斥 |
| `--test-mode` | 启用测试模式 | 布尔开关 |
| `--bert-no-binary-head` | BERT 无二元头 | 布尔开关 |
| `--dump-param-to-param-group-map` | 参数组映射转储路径 | str |

### 退出控制

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--exit-interval` | 退出间隔（每 N 步检查是否退出） | int |
| `--exit-duration-in-mins` | 退出时长（训练超过 N 分钟后退出） | int（分钟） |
| `--exit-signal-handler` | 启用退出信号处理器 | 布尔开关 |
| `--exit-signal-handler-for-dataloader` | 为数据加载器启用退出信号处理器 | 布尔开关 |

### 其他训练参数

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--mlp-chunks-for-prefill` | prefill 阶段 MLP 切分数 | int |

## 日志与输出

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--log-interval` | 日志打印间隔（每 N 步打印一次） | int，默认 100 |
| `--log-params-norm` | 记录参数范数 | 布尔开关 |
| `--log-throughput` | 记录吞吐量 / TFLOPS 等性能指标 | 布尔开关 |
| `--log-progress` | 记录训练进度 | 布尔开关 |
| `--log-num-zeros-in-grad` | 记录梯度零值数量 | 布尔开关 |
| `--logging-level` | 日志输出级别 | str |
| `--timing-log-level` | 时序日志级别 | 可选值：`0`（关闭）、`1`（基础）、`2`（详细） |
| `--timing-log-option` | 计时日志聚合方式 | 可选值：`max`、`minmax`、`all`；默认 `minmax` |
| `--tensorboard-dir` | TensorBoard 日志输出目录 | str |
| `--tensorboard-log-interval` | TensorBoard 日志记录间隔 | int，默认 1 |
| `--tensorboard-queue-size` | TensorBoard 事件写入队列大小 | int，默认 1000 |
| `--log-timers-to-tensorboard` | 记录各阶段耗时到 TensorBoard | 布尔开关；依赖 `--tensorboard-dir` |
| `--no-log-loss-scale-to-tensorboard` | 不记录损失缩放系数到 TensorBoard | 布尔开关 |
| `--log-validation-ppl-to-tensorboard` | 将验证集 PPL 写入 TensorBoard | 布尔开关；依赖 `--tensorboard-dir` |
| `--log-memory-to-tensorboard` | 记录 host 内存到 TensorBoard | 布尔开关；依赖 `--tensorboard-dir` |
| `--log-memory-interval` | 记录 host 内存间隔 | int，默认 None；依赖 `--log-memory-to-tensorboard` |
| `--log-world-size-to-tensorboard` | 记录 rank 数量到 TensorBoard | 布尔开关；依赖 `--tensorboard-dir` |

### 性能剖析 (NPU Profiling)

Megatron 原生 `--profile` 基于 torch.profiler（CUDA/CPU），在昇腾 NPU 上无法采集设备侧数据。MegatronAdaptor 的 `profile` 特性（`optimization_level>=2` 且传 `--profile` 时生效）将其自动重定向为 `torch_npu.profiler` 采集，trace 以 TensorBoard 格式落盘，可用 MindStudio / TensorBoard 查看。

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--profile` | 启用性能剖析（NPU 上自动重定向为 torch_npu.profiler 采集） | 布尔开关，默认 False；须 `--optimization-level >= 2` 才生效；未开启时对训练零影响 |
| `--profile-ranks` | 采集 profiling 数据的 rank 集合，`-1` 表示全部 rank | list of int，默认 `[0]`（Megatron 原生参数） |
| `--profile-step-start` | 采集窗口起始迭代 | int，默认 10（Megatron 原生参数） |
| `--profile-step-end` | 采集窗口结束迭代（不含） | int，默认 12（Megatron 原生参数）；须大于 `--profile-step-start` |
| `--profile-level` | profiler 采集深度 | 可选值：`level0`、`level1`、`level2`；默认 `level0`；level 越高采集粒度与内存开销越大 |
| `--profile-with-cpu` | 同时采集 CPU activity，可对齐 host-device 时序 | 布尔开关，默认 False |
| `--profile-with-stack` | 记录 Python 调用栈 | 布尔开关，默认 False |
| `--profile-with-memory` | 记录内存分配事件 | 布尔开关，默认 False |
| `--profile-record-shapes` | 记录算子输入 shape | 布尔开关，默认 False |
| `--profile-save-path` | trace 落盘目录 | str，默认 `./profile_dir` |

### 精度对照 (NPU DataDump)

跨平台（GPU→NPU）迁移中，整网精度对照需要逐层比对前向/反向数值。MegatronAdaptor 的 `npu_datadump` 特性（`optimization_level>=2` 且传 `--npu-datadump` 时生效）集成了 MSTT（MindStudio Training Tools）msprobe 工具的 `PrecisionDebugger`，在 `train_step` 前启动采集并注入模型、在每个训练步结束钩子后收尾采集并推进步计数，实现一键整网精度数据 dump。使用前需安装 msprobe：`pip install mindstudio-probe`；暂不支持 Lite 后端。

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--npu-datadump` | 启用整网精度数据 dump（msprobe PrecisionDebugger 采集） | 布尔开关，默认 False；须 `--optimization-level >= 2` 才生效；未开启时不导入 msprobe、对训练零影响 |

采集范围（rank/step/采集模式等）通过随包 `config.json` 调整（源码安装：`megatron_adaptor/patches/megatron/npu_datadump/config.json`；pip 安装：site-packages 对应目录），默认采集 RANK0、STEP0 的统计量精度数据（task=statistics、level=L1），落盘到进程工作目录下 `./dump_path`。dump 完成后，可用随包 `compare.json` 模板（配置 `npu_path` 与 `bench_path`）经 msprobe 进行精度比对，配置细节参考 [msprobe 官方文档](https://gitcode.com/Ascend/msprobe)。

> [!NOTE]
>
> - msprobe 暂不支持 PyTorch 2.7 及以上版本的 dynamo 场景。
> - 单卡训练时须将 config.json 中 `rank` 改为空列表 `[]`（指定 rank 会导致 dump 数据为空）。
> - 精度对照场景可配合 Megatron `--deterministic-mode` 固定计算随机性。

## µP (Maximal Update Parametrization)

| 参数名 | 描述 | 约束 |
|--------|------|------|
| `--use-mup` | 启用 µP 参数化 | 布尔开关 |
| `--mup-width-mult` | 宽度乘子 | float，默认 1.0（自动计算为 hidden_size / mup_base_hidden_size）；依赖 `--use-mup` |
| `--mup-base-hidden-size` | 基础隐藏层大小 | int；依赖 `--use-mup` |
| `--mup-embedding-mult` | 嵌入层乘子 | float，默认 1.0；依赖 `--use-mup` |
| `--mup-output-mult` | 输出层乘子 | float，默认 1.0（MuP 启用时自动设为 1/width_mult）；依赖 `--use-mup` |
| `--mup-base-head-dim` | 基础注意力头维度 | int；依赖 `--use-mup` |
| `--mup-attn-scale-power` | 注意力缩放幂次 | float，默认 1.0（0.5=标准 attention，1.0=MuP）；依赖 `--use-mup` |

## 总结

当前 MegatronAdaptor 已验证覆盖了上述大类功能特性，涵盖了 Megatron-LM 的核心能力：

- **核心训练**：激活函数、注意力机制、位置编码、归一化
- **精度优化**：BF16/FP16/FP8 混合精度、Loss Scale 动态调整
- **并行策略**：TP/PP/SP/CP/EP 多维并行、Virtual Pipeline、FSDP
- **优化器**：Adam/SGD/Muon/Lion/SOAP、分布式优化器、CPU Offload
- **内存优化**：重计算、检查点、层 Offload
- **模型架构**：MOE、Hybrid (Mamba+Transformer)、MTP、µP
- **数据处理**：多种 Tokenizer、数据缓存、FIM 训练
- **通信优化**：TP Comm Overlap、SHARP、通信重叠
- **训练稳定性**：Straggler 检测、故障注入、确定性模式、梯度检查
- **可观测性**：TensorBoard 集成、吞吐/内存/时序日志、NPU Profiling（进程内 torch_npu.profiler 采集）、精度对照（msprobe PrecisionDebugger 整网精度数据 dump）
