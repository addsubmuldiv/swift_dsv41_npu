# TENPU DotProductAttention ATK 看护模块

`transformer_engine.pytorch.DotProductAttention` 是 Megatron
`TEDotProductAttention` 背后的核心 attention 计算。基线为 NVTE golden
（CUDA 采集），数值与 API 契约双重比对。

attention 与其他模块的一点不同：`DotProductAttention` 内部**没有可学习
参数**——QKV 投影权重（`W_q/W_k/W_v`）在 Megatron 的 `Attention` 层里由
外层的 `ColumnParallelLinear`/`RowParallelLinear` 持有（已由 `linear` 模块
看护），本模块接到的是投影后的 q/k/v **激活值**。因此输出槽位为
`forward_output, query_grad, key_grad, value_grad, api_contract`
（梯度只对输入激活求，无参数梯度槽位），契约记录 q/k/v 形状以及 `num_attention_heads`、
`kv_channels`、`num_gqa_groups`、`attn_mask_type`。GQA 用例中，生成器从
`num_gqa_groups`（缺省回退到 `num_attention_heads`）推导 `num_kv_heads`，
使 key/value 形状与模块配置一致。

覆盖范围：BF16/FP16/FP32（各 50 例，共 150 例）、MHA/GQA/MQA、dropout 关闭、前向输出 + q/k/v 输入梯度 +
序列化 API 契约，覆盖 TENPU DPA 支持的全部两种 qkv 格式：

- **`sbhd`**（4 维等长）：mask ∈ {causal, no_mask}；
- **`thd`**（3 维 packed/varlen）：mask ∈ {padding,
  padding_causal}，`cu_seqlens` 由适配器从 token 数按固定规则推导。

无 FP8，不含 `bshd`（TENPU 未实现，API 缺口已登记）。

## 文件构成

- `dot_product_attention.yaml`：ATK 设计文件与精度容差。
- `generate_dot_product_attention.py`：整建路线生成器（候选）。
- `generate_dpa_hook.py`：钩子路线生成器（**当前默认**，见 yaml 的 `generate:`）。
- `function_dot_product_attention.py`：纯设备适配器、契约编码器、allclose 比较器。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case \
  -f tests/atk/module/dot_product_attention/dot_product_attention.yaml \
  -p tests/atk/module/dot_product_attention/generate_dpa_hook.py
```

预期用例文件：`result/dot_product_attention/json/all_dot_product_attention.json`
（10 例，sbhd/thd 双格式 × bf16；`-p` 必须与 yaml 的 `generate:` 字段指向同一条路线）。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行。**必须关闭 TF32**（原因见文末「已知算子限制」）：

```bash
NVIDIA_TF32_OVERRIDE=0 python tests/atk/module/dot_product_attention/collect_nvte_golden.py \
  --cases result/dot_product_attention/json/all_dot_product_attention.json \
  --input-dir ./dot_product_attention_inputs \
  --golden-dir ./dot_product_attention_goldens
```

`NVIDIA_TF32_OVERRIDE=0` 对 cuBLAS/cuDNN 生效（含应用显式请求的 `FAST_TF32`）；
torch 侧的 `torch.backends.cuda.matmul.allow_tf32 = False`、`torch.backends.cudnn.allow_tf32 = False`、
`torch.set_float32_matmul_precision("highest")` 只覆盖 torch 自己发起的调用，**不足以**关掉
上游库内部的 TF32，故以环境变量为准。自查该侧是否真关掉：同进程里算 `(1 + 2**-12) @ 1`——
`1.000244140625` 即 IEEE fp32，`1.0` 就是走了 TF32（10 位尾数把 `2**-12` 舍掉了）。

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/dot_product_attention/json/all_dot_product_attention.json`、`dot_product_attention_inputs` 与 `dot_product_attention_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/dot_product_attention/json/all_dot_product_attention.json \
  -n tests/atk/module/dot_product_attention/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/dot_product_attention/function_dot_product_attention.py \
  --input_data ./dot_product_attention_inputs \
  -tk accuracy
```

## 已知算子限制（生成器已护栏）

- **fp32 用例要求参考侧是 IEEE fp32（TF32 关闭）**：上游库里 fp32 默认可能走 TF32——NVTE 自己的
  GEMM 封装对**纯 fp32 无条件请求 `FAST_TF32`**（`common/gemm/cublaslt_gemm.cu`），cuDNN 的
  fp32 attention 也默认允许 TF32，而 `torch.backends.cuda.matmul.allow_tf32` /
  `torch.backends.cudnn.allow_tf32` 只覆盖 torch 自己发起的调用 ⇒ **采集端必须用
  `NVIDIA_TF32_OVERRIDE=0`（唯一能覆盖库内显式请求的开关，见第 2 节）**。若参考侧没关掉，
  症状是：与 ATK 默认 fp32 容差（`rtol = 9.77e-4 ≈ 2**-10`，正好**一个 TF32 尾数 ULP**）相撞，
  表现为**少数元素（1~4%）越带、没有元素撞绝对门限，且失败数随 |值| 增大**（`dk ∝ q` 最大 ⇒
  失败最多，`out ≤ |v|` 最少）；同形状的 NPU fp32 对精确 fp64 基线是干净的 ⇒ **不要为此放宽
  fp32 标准**，应改采集环境（关 TF32）或改用精确基线判 fp32 用例。

- **thd 的 16 位 stride 上限**：上游 FA 算子（不在本仓库，`ops-transformer` 的
  `attention/flash_attention_score_grad`）的 TND 路径缺 stride 守卫（该侧前向 kernel 有
  `dstStride <= 65535` 守卫与逐行回退，grad kernel 只定义了 `STRIDE_LIMIT = 65535` 却未使用），
  一旦 **`num_heads × align(head_dim, 16) > 65535`** 就**静默算错**：fp32 只错 `dq/dk/dv`（反向），
  **fp16/bf16 连前向输出一起退化**；同等乘积下的 sbhd 正常。判据必须用**对齐后**的 head_dim——
  `head_dim = 120`、512 头时 raw 乘积 61440 看似合规，对齐后 `512 × 128 = 65536` 已越线。
  → `generate_dpa_hook.py` 对 thd 的抽样把 head_dim 按 8 步下钳（sbhd 不夹），并纳入 fail-closed 断言。

- **golden 侧 dq 的 2 GiB 上限**：CUDA 融合 fp16 后向（TE + cuDNN，A100/sm80）在 **dq 张量达到 2 GiB**
  时 dq 大面积未写入——约 72% 元素停在缓冲区初值 0，而前向 out 与 dk/dv 正常、**不报任何错**；
  1 GiB 实测干净，中间区间未测。触发量是 dq 的**字节数**，与 token 数、GQA 组数、后端选择无关。
  → 只影响"用 CUDA 参考判 dq"的用例，与 TENPU 侧无关；生成器按 `generate_dpa_hook._clamp_batch_for_dq_bytes`
  把 batch 维下钳到 dq ≤ 1 GiB（阈值按 dtype 换算，fp32 元素数为 fp16 的一半；sbhd floor=2，
  thd floor 保证 `T = d0·d1 ≥ 3`），使该区间用例不生成（生成期规避，不做断言——ATK 无逐例跳过，
  raise 会中断整跑）。

- **上游算子文档的明写约束**（`ops-transformer` 的 `aclnnFlashAttentionVarLenScoreV4.md` /
  `aclnnFlashAttentionUnpaddingScoreGrad.md` 的"约束说明"）：**T ∈ [1, 1M]、N(heads) ∈ [1, 256]、
  D ∈ [1, 768]**，varlen 仅支持 TND；反向**默认非确定性**（需 `aclrtCtxSetSysParamOpt` 开启）。
  本模块的 dim 池上限 512、head_dim 上钳 512/768 即由此而来。

- 其余已登记的边界：`head_dim = 1024` 被 NPU kernel 直接拒绝（报错 `EZ0026`，故 > 768 钳回 512）；
  thd 侧无对齐约束，sbhd 侧有 cuDNN 的 batch ≥ 2 / seq % 16 / head_dim % 8 三条。
