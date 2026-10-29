# 四机完整模型 LoRA：拓扑与启动配置检查

检查对象是 `examples/train_full_lora_4nodes.sh` 的实际命令及当前五个源码快照。
运行结果：**24 项测试通过，无失败或跳过**。没有分配完整模型权重、创建真实通信组或运行训练。
分组、层切分与发送/接收描述见 [完整 JSON 报告](evidence/four_node_topology.json)。

## 拓扑

采用 MCore 当前实际默认顺序 `tp-cp-ep-dp-pp`，用源码的 `RankGenerator` 生成全部 64 个 rank 的组，
检查每类组覆盖全部 rank 且组内/组间不重复，并对照 Swift 的启动参数。

| 维度 | 大小 | 实际分组 |
| --- | --- | --- |
| TP / ETP | 1 / 1 | 单卡；当前 CSA2 要求 TP1、无 sequence parallel |
| CP | 1 | 单卡；本次不验证 CP>1 |
| PP | 4 | `[i, i+16, i+32, i+48]`，i=0..15 |
| 稠密参数 DP | 16 | 每台机器的 16 个 rank |
| EP | 16 | 与本机 DP 组成员相同；专家通信在节点内 |
| 专家参数 DP | 1 | 每个专家分片只有一个副本 |

稠密网格：`64 = TP1 × CP1 × PP4 × DP16`。
专家网格：`64 = ETP1 × EP16 × PP4 × expert-DP1`。
EP 是对现有 rank 的另一种分组，不能在稠密网格上再乘一个 16。
384 专家平均每卡 24 个，低于当前 NPU grouped-matmul 的 128 专家限制。

`global_batch=64 / (micro_batch=1 × DP16) = 4` 个 microbatch；普通 PP4，无 VPP、P2P overlap 或 FSDP。

## 模型切分与 Engram

| 节点 / PP 阶段 | Global rank | 原始层（0 起） | Hybrid index（0 起） | Engram 原始层 |
| --- | --- | --- | --- | --- |
| 0 | 0–15 | 0–9 | 0–19 | 1 |
| 1 | 16–31 | 10–19 | 20–39 | 14 |
| 2 | 32–47 | 20–29 | 40–59 | 无 |
| 3 | 48–63 | 30–39 | 60–79 | 无 |

40 个原始层映射为 80 个 Hybrid 子层，四段均为完整的 attention+FFN 对。
真实 `get_mcore_model_config`、V4.1 layer-spec 构建和 Engram 启动校验已执行。
配置构造会自动创建依赖真实 process group 的 bridge；静态检查只跳过这项组相关分配，
其余 MCore/Adaptor/bridge 配置校验正常执行，没有构造假训练模型。
两个 Engram 的 1-based Hybrid 位置为 3、29；逐个 prime table 检查了 EP16 分片的连续覆盖、无重叠和行数平衡。
通用 Engram PP 限制与 V4.1 适配区别已有回归测试，TP/SP/ETP 的其他限制保持生效。

## 跨 PP 状态与输入

用真实 Hybrid 状态组件构建四段的 pipeline 描述，并为长度 24、128 生成 meta 张量描述。
三个边界发送与接收的 fingerprint、shape 和 dtype 全部一致：

- 边界 20：hidden、mHC pre_mix、共享 KV、索引。
- 边界 40：hidden、mHC pre_mix；新的 KV/index owner 在下一段开始。
- 边界 60：hidden、mHC pre_mix、共享 KV、indexer K、索引、候选块及长度。

hidden shape 为 `[seq, 1, 20480]`，即原始 5120 宽度乘 4 个 mHC stream；不是普通模型的 5120 宽度。
这依赖当前源码的 typed pipeline payload，不能改回普通单 tensor P2P。
仅提供 placement 元数据来生成描述，没有模拟通信成功。

用真实 Swift sampler 遍历全部 64 个逻辑 rank、每 rank 40 个 microbatch，确认同一 PP 链的输入索引一致，
且同一 microbatch 的 16 个 DP 分片无重复。32 条测试数据会跨 epoch 循环；10 步不要求下载更多数据。
另已验证中间 PP 阶段保留 input_ids/position_ids，只有 labels 被移除。

## 显存的静态估计

只读取本地 safetensors header，不读权重数据。将 FP4/FP8 权重展开为 BF16，F32 张量按 F32 保守计数，
专家和 Engram 表按 EP16 分片、其他文本参数在各阶段内复制；关闭视觉和 DSpark。

| 阶段 | 每卡底座权重估计 GiB |
| --- | --- |
| 0 | 31.921 |
| 1 | 30.669 |
| 2 | 18.946 |
| 3 | 20.164 |

这是底座权重估计，不是实测峰值；不包含 LoRA/优化器、激活、通信工作区、分配器及反量化临时缓冲。
64GB 卡不能仅凭这张表认定一定不 OOM，但底座权重本身没有超过单卡容量。

## 重现与实际验收边界

在已经安装的环境中，使用完整权重和对应的完整 Engram map：

```bash
source env.sh
DSV41_MODEL_DIR=/storage/models/DeepSeek-V4.1-Flash-1017 \
DSV41_ENGRAM_MAP="$PWD/.local/full_lora/engram_tokenizer_map.json" \
python -m pytest -q tests/test_full_lora_topology.py \
    tests/test_engram_pipeline_startup.py tests/test_install_constraints.py
```

未设置模型/map 变量时，完整模型部分会明确跳过；不能把跳过当通过。
没有发现新增的配置、拓扑或 schema 阻塞。真实四机 HCCL、NPU P2P 算子、完整模型首步显存、
forward/backward 和保存仍需在目标四机上确认；上述结果不保证这些运行时环节已经通过。
