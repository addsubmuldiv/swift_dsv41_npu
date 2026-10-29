# ATK 泛化约束资料

**面向测试泛化的用例扩展约束**：本文件列出 21 个 ATK 看护模块在 dtype、维度（dim）、shape 上的支持边界与特殊约束。泛化生成用例时**必须遵守**，否则会在适配器、NPU kernel 或 NVTE golden 采集侧直接失败。

> 泛化的操作规则见"四、泛化操作规则"；**泛化规模怎么调、yaml 旋钮的真实语义见"五、泛化规模旋钮（试点：layernorm）"**；各模块当前用例矩阵见各 `generate_<name>.py`。

---

## 一、dtype 约束

ATK 可生成的 dtype 有 8 种（fp32/fp16/bf16/fp8e4m3/fp8e5m2/int32/uint8/bool），但**每个 API 实际只支持其中 2-4 种**。

| 模块 | 允许 dtype | 排除及原因 |
|---|---|---|
| `linear` | bf16, fp16, fp32 | fp8 除外（quantizer 上下文） |
| `rmsnorm` | bf16, fp16, fp32 | fp8 除外 |
| `layernorm` | bf16, fp16, fp32 | fp8 除外 |
| `layernorm_linear` | bf16, fp16, fp32 | fp8 除外 |
| `layernorm_mlp` | bf16, fp16, fp32 | fp8 除外 |
| `grouped_linear` | bf16, fp16, fp32 | fp8 除外 |
| `dot_product_attention` | bf16, fp16, fp32 | fp8 除外；**head_dim 上界见 §3.4** |
| `rope` | t/dy 支持 bf16, fp16, fp32；freqs 只用 fp32 | freqs 必须固定 fp32（API 契约）；head_dim 见 §3.4/§3.1 |
| `fused_activation` | fp32, fp16, bf16 | 其余 dtype 抛 `Unsupported dtype`（TENPU 源码显式检查，`activation.py:131` / `swiglu.py:102`） |
| `float8_quantization` | 输入 bf16, fp16, fp32（量化目标 e4m3/e5m2 是 attr） | fp8 只通过 quantizer 进入；**输入 range 必须远离 0** |
| `mxfp8_quantization` | 输入 bf16, fp16 | **输入 range 必须远离 0**（[0.01, 0.5]，全零 block 使 scale 无定义）；**fp32 输入 kernel 不支持**（`npu_dynamic_mx_quant` 仅收 bf16/fp16；同族 blockwise 的 `npu_dynamic_block_quant` 支持 fp32——两 kernel 支持面不同，勿统一泛化）；shape 必须 32 对齐（纯 yaml 路线靠 dim 池保证）；设备侧执行需 A5 级 NPU（kernel 架构门） |
| `blockwise_quantization` | 输入 bf16, fp16, fp32 | 同 mxfp8（range/对齐/A5），但 **dtype 支持面不同：fp32 输入合法**（`npu_dynamic_block_quant` 与 `npu_dynamic_mx_quant` 是两个 kernel）；shape 必须 128 对齐；设备侧执行需 A5 级 NPU |
| `fused_optimizer` | fp32 | fp16/bf16 master 路径未验证 |
| `multi_tensor` | fp32, bf16, fp16 | — |
| `router` | fp32, bf16, fp16 | 低精度**稀疏**选择跨后端不可比（tie-break，见三、特殊约束）：生成器对非 fp32 的选择类用例强制**稠密** topk==E，稀疏选择只在 fp32 上覆盖 |
| `moe_permute`（含 with_probs/and_pad 变体） | bf16, fp16, fp32 | routing_map 由 spec 字符串确定性重建（attr 的 `routing_dtype` 决定**map** 的 dtype，与激活 dtype 无关） |
| `moe_sort_chunks_by_index`（含 with_probs） | bf16, fp16, fp32 | 无 routing map |
| `moe_unpermute` | bf16, fp16, fp32 | 内部先跑匹配的 permute API，dtype 随之 |

各 dtype 的**验证执行状态**（用例已实跑 / 仅生成待设备 / golden 重采待办）属看护组内部台账，不在本文档维护。

**通用禁令**：

- int32/uint8/bool 输入对全部 21 个 API 非法（这些类型只出现在输出槽：row_id_map、routing_map、契约）；
- fp8 dtype 不能直接作为输入生成——fp8 只通过 quantizer 进入（float8_quantization 模块的 fp8 格式是 **attr 字符串**，不是输入 dtype）。

---

## 二、dim 维度约束

| 模块 | 支持 dim | 已验证形态 | 说明 |
|---|---|---|---|
| `linear` | **2/3/4** | 2D/3D/4D/空输入 | 最后维 = in_features；前导维可任意（含 0 行空输入，合法） |
| `rmsnorm` | **2/3/4** | 同上 | **空输入非法**（kernel 拒绝 shape-0 dy）——与 linear 不同，勿统一泛化 |
| `layernorm` | **2/3/4** | 同上 | 空输入非法，同 rmsnorm |
| `layernorm_linear` | **2/3/4** | 同上 | 空输入非法；lnmlp 的 hidden/ffn 建议保持适中（256-4096），过大拖慢 golden 采集 |
| `layernorm_mlp` | **2/3/4** | 同上 | 同上 |
| `grouped_linear` | **2** | 2D | 3D 理论可行但 Megatron 不使用 |
| `dot_product_attention` | **4（sbhd）或 3（thd）** | sbhd + thd | sbhd 固定 4D 布局 [s, b, h, d]（不是任意 4D）；thd 为 3D `[total_tokens, h, d]` + `cu_seqlens`（见 §3.1/§3.4）。**`bshd` 是 TENPU 的 API 缺口**（NVTE 支持、TENPU `forward` 断言只收 {sbhd, thd}，属实现缺口），不可生成 |
| `rope` | **4**（sbhd/bshd）；THD 为 3 维（[t, h, d]）且要求 `fused=True`（NPU 上非融合 THD 不支持） | sbhd/bshd/thd | thd 需 cu_seqlens attr；泛化 thd 时保持 fused=True |
| `fused_activation` | **任意维数**；gated 类（geglu/reglu/swiglu/scaled 变体）**≥1 维** | 2D/3D | **空输入 NPU 合法 / NVTE 拒绝**（见 3.3） |
| `float8_quantization` | **2/3** | 2D/3D | **空输入非法**（amax 无定义） |
| `mxfp8_quantization` | **2/3** | 2D/3D | 见 shape 对齐约束 |
| `blockwise_quantization` | **2/3** | 2D/3D | 见 shape 对齐约束 |
| `fused_optimizer` | **任意维数**；**空参数合法**（step 空转） | 2D | state 形状不可由用例指定；SGD 超参硬约束见 §3.2 |
| `multi_tensor` | **任意维数** | 2D + 3D | **空输入合法**（任一维 0 含末维 0）；**等形为用例设计约定而非 API 约束**（API 接受 list 内混合形状） |
| `router` | **2** | 2D [T, E] | — |
| `moe_permute` 族 | **2** | 2D [num_tokens, hidden] | — |
| `moe_sort_chunks_by_index` 族 | **2** | 2D | — |
| `moe_unpermute` | **2** | 2D | — |

---

## 三、shape 特殊约束

### 3.1 对齐约束（kernel 要求）

| 模块 | 约束 | 后果 |
|---|---|---|
| `mxfp8_quantization` | **最后维 %32==0 且 前导维乘积 %32==0** | `is_quantizable` 拒绝 |
| `blockwise_quantization` | **最后维 %128==0 且 前导维乘积 %128==0** | 同上 |
| `dot_product_attention` | **sbhd 路线**：seq %16==0、head_dim %8==0、batch ≥2（**thd 路线无对齐约束**） | golden 侧 cuDNN 拒绝（NPU 可能可执行，但采不到 golden） |

### 3.2 kernel 限制（NPU 侧，已登记）

| 模块 | 约束 |
|---|---|
| `rope` | **half（非 interleaved）模式 rot_dim < 1024**（`npu_rotary_mul` kernel 限制）；interleaved 模式无此限。泛化 real 层用例时非 interleaved 的 split ≤ 1024 |
| `fused_optimizer` | **SGD：momentum < 0 无条件非法**；**nesterov=True 要求 momentum > 0**——生成器 reconcile（负值→就近合法值、nesterov×零→0.9） |
| `moe_permute_and_pad_with_probs`（及 `moe_unpermute` 的 pad 模式） | **填充后行数（kernel 的 `numOutTokens`）≤ 1024**（超限报 "numOutTokens should not great than 1024, but got 1088"，aclnnMoeTokenPermuteWithRoutingMap，错误 161002）。生成器在 align_size 搜索里同时约束"整除 E"和该上限 |
| `fused_activation` | **gated 类末维必须为偶数**（geglu/reglu/swiglu/scaled_swiglu/scaled_clamped_qgeglu；swiglu 系 op-plugin kernel 报错、geglu/qgeglu 系 torch 广播错，均为运行时崩)；gated 类 **0 维非法**（dim out of range） |

### 3.3 语义/结构约束

| 模块 | 约束 |
|---|---|
| `grouped_linear` | `m_splits` 之和 == 输入行数（适配器断言，违反报错）；num_gemms ≤ 16（yaml schema 槽位上限）；weight/bias 输入组数 == num_gemms |
| `rope` | freqs 宽度 == rot_dim == t 最后维的一半（plain 形态）。**head_dim 可为奇数**：freqs 宽取 `head_dim // 2`（向下取整），适配器按 `rot_dim = 2*(head_dim//2)` 切分，奇数维留一维不旋转（旧矩阵的 d=7 是同款）；fused-QKV 形态的 q_freqs/k_freqs 宽 == hn、split 三元组满足 [ng*hpg*hn, hn, hn] |
| `float8` / `mxfp8` / `blockwise` | **输入 range 必须远离 0**（[0.01, 0.5] 量级；全零块使 scale 无定义）。纯 yaml 路线没有生成期守卫，靠 yaml 值域声明保证；mxfp8/blockwise 另需 shape 对齐（32/128），靠只放对齐值的 dim 池保证 |
| `moe_permute` 族 | routing_map **不能随机采样**——由 `routing_spec` 字符串确定性重建（token 分布与分组顺序是语义）；泛化新 shape 时**渲染** spec（钩子路线下由钩子按 token 数渲染，见各 `generate_*_hook.py`），不要改为随机 map。**dropped tokens（mask 空行）不要生成**：有 dropped 行时算子输出行数不遵循 `routed×topk` |
| `router` | logits 每行语义为 expert 得分（[T, E]）；**低精度（bf16/fp16）的稀疏 top-k 跨后端不可比**（两 expert 同值时 torch.topk 与 CUDA kernel 可能不同选，而 routing_map 是精确比对）→ 生成器对非 fp32 的选择类用例**强制稠密 topk == E**（边界不存在，平局无从影响）；稀疏选择只在 fp32 上覆盖 |
| `float8_quantization` 等 | 输入 range 保持 [0.01, 0.5] 量级（远离 0，保证 amax/scale 良定义；全零块使 scale 无定义） |
| `rmsnorm`/`layernorm` 族 | 空输入（任一维 0）非法；但 `linear` 的 0 行合法——同族模块规则不同 |
| `fused_activation` | **空输入 NPU 合法、NVTE 拒绝**：CUDA 分配器对 0 字节请求返回 null data_ptr（PyTorch `CUDACachingAllocator::allocate`），TE C++ `CheckInputTensor` 断言 `Input ... is not allocated`（NVTE 侧全 9 个 op 同拒）——两侧都可执行才可测，`has_empty` 保持 false，钩子已加 fail-closed 守卫；0 维标量输入 plain unary 类（gelu/silu/relu）两侧合法，**scaled_srelu 例外**：其 NVTE 反向 `tex.dsrelu` 断言 grad `[1]` ≠ 缓存输入 `[]`（前向 `scale.unsqueeze(-1)` 广播把 0 维输出顶成 `[1]`）→ 钩子 reconcile `[1]`；gated → `[2]` |

### 3.3.1 boundary 轨道支持矩阵（21 模块）

`extra_numbers` 五类边界类别 × 主 tensor 输入的合法性：

| 模块 | empty（空输入） | scalar（0 维） | infnan | lower/upper 值端点 |
|---|---|---|---|---|
| layernorm / rmsnorm / layernorm_linear | ❌ | ❌ | ⚠️ 无意义 | ✅ |
| linear | ⚠️ 仅前导 0 维合法（末维 0 非法） | ❌ | ⚠️ 无意义 | ✅ |
| fused_activation | ❌（NPU 合法、NVTE 拒绝，钩子守卫） | ⚠️ plain unary 合法；scaled / gated 由钩子 reconcile | ⚠️ 无意义 | ✅ |
| multi_tensor | ✅ | ✅ | ⚠️ 无意义 | ✅ |
| fused_optimizer | ✅ | ✅ | ⚠️ 无意义 | ✅ |
| float8_quantization | ❌（amax 无定义） | ❓ 未探 | ⚠️ 无意义 | ✅（值域须远离全零） |
| dot_product_attention | ❌（轨道不可用） | ❌ | ❌ | ✅ |
| rope | ❌（轨道不可用） | ❌ | ⚠️ 无意义 | ✅ |
| router | ❌（轨道不可用） | ❌ | ⚠️ 无意义 | ✅ |
| grouped_linear | ❌（轨道不可用） | ❌ | ⚠️ 无意义 | ✅ |
| `moe_permute` | ❌（退化守卫） | ❌ | ⚠️ 无意义 | ❌（索引域越界） |
| `moe_permute_with_probs` | ❌（同款守卫） | ❌ | ⚠️ 无意义 | ❌（同上） |
| `moe_permute_and_pad_with_probs` | ❌（同款守卫） | ❌ | ⚠️ 无意义 | ❌（同上） |
| `moe_unpermute` | ❌（同款守卫） | ❌ | ⚠️ 无意义 | ❌（同上） |
| `moe_sort_chunks_by_index` | ❌（同款守卫） | ❌ | ⚠️ 无意义 | ❌（同上） |
| `moe_sort_chunks_by_index_with_probs` | ❌（同款守卫） | ❌ | ⚠️ 无意义 | ❌（同上） |
| `mxfp8_quantization` | ❌（结构上不可表达） | ❌（同左） | ❌ **不支持**：inf/nan 使块 amax=NaN → scale 无定义、输出整体无意义（两侧同 NaN 假绿或未定义行为，无判决价值）；yaml `has_infnan` 显式关死 | ✅（值域远离零） |
| `blockwise_quantization` | ❌（结构上不可表达） | ❌（同左） | ❌ **不支持**：golden-free 参考保 inf（round-trip = x/bound）而 kernel 对 inf 走饱和——185 元素破硬门的实测形态，必然假红；nan 使块 amax=NaN → scale 无定义 | ✅（同 mxfp8） |
| `layernorm_mlp` | ❌（钩子守卫） | ❌（钩子守卫） | ❌ | ✅ |

**口径**：❌ 类别必须保持对应 `has_*: false`；✅ 可按需打开；⚠️ infnan 除 DPA（❌）外建议保持关（无判别价值）——量化两模块与 `layernorm_mlp` 已升 ❌ 不支持。

**两点使用提醒**：① empty 列的 ✅ 均为 NPU 单侧结论，开启 `has_empty` 前须先在 GPU 侧确认可执行；
② MoE 族六模块一律 `extra_numbers: 0`，其行内 ❌/❌ 是路线判定（生成期守卫 + 结构分析），非设备实测；
MoE 的「空 chunk」边界是另一回事（属 ✅），勿与空输入合并。

逐模块槽位表与依据见各模块 README。

### 3.4 golden 侧约束 ≠ NPU 侧约束

DPA 的 batch≥2/seq%16/head_dim%8 是 **CUDA 采集主机的 cuDNN 矩阵**限制（**只约束 sbhd 路线**）：NPU 可执行的形状（MQA、奇数头数、非对齐 head_dim）在 golden 侧被拒。**泛化用例前先确认形状两侧都可执行**，否则 golden 采集失败。

**DPA 在 NPU 侧另有一条 head_dim 上界**：
`head_dim = 1024` 的用例**全部**被 NPU flash-attention kernel 拒绝
（`aclnnFlashAttentionVarLenScore` / `aclnnFlashAttentionScore failed`，错误码
1610 / 161002；thd 与 sbhd、bf16 与 fp32 均复现），而 `head_dim ≤ 512` 的用例
全部通过。即 SBHD/THD 两条路线共用同一上界。**精确上界由 kernel 报错给出：D ∈ (0, 768]**（EZ0026 原文
"Parameter D dim of FlashAttentionScore ... can only be in the range of
(0, 768]"）；生成器把 head_dim > 768 的
采样**钳回 512**（ATK 生成器无逐例跳过机制，raise 会中断整个 `atk case` 运行；
(512, 768] 放行——NPU 合法，GPU 侧未验证自担）。**源码依据（ops-transformer `attention/flash_attention_score`）**：
op_api `aclnn_flash_attention_score.cpp:585`（`axes.d > 768` 拒绝，即报错原文出处，
arch 无关）+ tiling `arch35/..._tiling_regbase.cpp:1176-1184`（`dSize`/`dSizeV`
各查一次 ≤ 768，**value 的 head_dim 有独立上限**）；tiling 内部按 64 分档
（{64,128,192,256} 模板，> 256 一律 ALIGNED_768 大 tile——(256,768] 合法但走大
tile）。**D 不要求对齐**：op_api 对未对齐 D 自动内部 pad——本套系的 D%8 约束
服务的是 golden 侧 cuDNN，非 NPU 侧。**头数无硬限**：op_api 对 headNum 仅查
`> 0`（文档口径 1~256 是规格范围非运行时检查；H=2048 实际可跑）。
dim 池保持 512——GPU 侧（NVTE 融合 attention）是否接受 (512, 768] 未验证，
按"两侧都要能跑"的原则，探该区间前需先在 CUDA 主机确认。
文档另登记：B 1~2M、S 1~1M、VarLenScore 与 Score 共用同一套 D 检查
（actualSeqLen 条目数 > 20000 仅告警不拒）。

DPA 的 thd 路线另有一条**由 golden 侧断言定死的格式×mask 耦合**：NVTE
`assert "padding" in attn_mask_type`（`dot_product_attention.py:1368`），
即 **thd 只能用 {padding, padding_causal}**；反过来 sbhd 用 padding 遮罩需要
适配器不提供的 mask 张量。两个 attr 被原生引擎独立采样 ⇒ **非法组合是常态**
，必须进生成器 reconcile（保留 causal 位：causal →
`*_causal`，否则 plain 变体）。注意 **TENPU 侧不校验这条**（8 种组合 NPU 全
可执行），所以耦合规则只能从 golden 侧推出来。

DPA 的 thd 路线还有一条**输出布局跨后端差异**：NVTE 的两个
attention 后端都把输出末两维拍平（`...hd → ...(hd)`；sbhd 返回 3-D
`[s, b, h·d]`，**thd 返回 2-D `[T, H·D]`**），而 **TENPU 的 thd 输出保持 3-D
`[T, H, D]`**（`TND` 直传 `npu_fusion_attention`）。两侧的 sbhd 输出一致（均为
拍平 3-D）。适配器侧归一化：GPU 分支按 `[T, H·D]` 喂 loss、反向结束后把输出
reshape 回 3-D，两侧返回与比对的形状一致；用例的 dy 保持语义形状 `[T, H, D]`
（生成器无需感知该差异）。

---

## 四、泛化操作规则

1. **dtype 扩一档 → golden 全部重采（仅限离线 golden 路线）**。dtype 是序列化契约的字段（exact 比较），离线 golden 与用例必须同版生成（见 README "golden 重采铁律"）。**在线 GPU/NPU 实时对比路线不落 golden，无重采问题**——本表"允许 dtype"列按 API 实际支持面填写，不再为 golden 可采性让路。
2. **dim 泛化只在表二范围内**；表二之外的 dim 是 API 不支持，不是未覆盖。
3. **real 层 shape 惯例保持 128 对齐**（hidden/out_features 为 128 倍数）——全套件一致，为 FP8 泛化预留。
4. **对齐约束（3.1）在生成器里断言**：mxfp8/blockwise/DPA 的生成器已内置检查，泛化工具生成的 shape 会被拒。
5. **扩展生成器而非适配器**：新 shape/dtype 组合进 `generate_<name>.py` 的矩阵；适配器（`function_<name>.py`）不应为泛化改动——改适配器会触发**离线** golden 重采（在线路线不受影响）。
6. **moe 模块（6 个）的 routing map 语义约束**：只扩展 `routing_spec` 的构造（expert 数/topk/drop 计数），不要把 map 改成随机张量——会破坏 token 分布语义，两侧重建不再一致。
7. **比较器口径与切换规则**：
   - 允许切换到 ATK 原生标准的模块：12 个 mixed 模块 → `mixed_tolerance_bm`（yaml key 逐字兼容）；3 个 mare 模块（rmsnorm / fused_activation / float8_quantization）→ `single_bm`（`*_thd` / `mare_ratio` key 同名兼容）；6 个 moe 模块（yaml 当前为 `*_allclose`）→ `mixed_tolerance_bm`（key 兼容，判据升级为带兜底结构）。
   - **不可切换**的 3 个模块：`layernorm_mlp` / `mxfp8_quantization` / `blockwise_quantization`——它们的 yaml 都含 per-output 容差键 `*_output_N`（mxfp8/blockwise 为 fp32 档容差；layernorm_mlp 为 o1 近零救援 `bf16_atol_output_1` / `bf16_max_abs_error_limit_output_1`，见规则 8），ATK 原生构造器遇到该类 key 直接 `ValueError`。状态表标签为 `@mixed_tolerance_custom`。
   - **危险操作警告**：禁止把 mixed 模块改成 `single_bm`——`single_bm` 对不认识的 key（`bf16_rtol` 等）**静默忽略**，任务不报错但阈值被无声替换为默认表。切换方向只能是 mixed→mixed_tolerance_bm、mare→single_bm，不能交叉。
8. **判决域约束（layernorm_mlp）**：判决域是数据流槽——`output_0` 前向与 `output_1` 输入梯度。`output_0` 的 golden 携带 **fp64 物化链基线**（ln_out / fc1 前激活 / 激活输出逐级单次舍入到 case dtype——两侧模块源码描述的数据流；NVTE 的 gelu 融合 epilogue 与瘦 N GEMM 形状实测偏离该链 0.97 量级、TENPU 在 `general_gemm` bias 融合修复后逐位贴合，故 golden 侧替换为精确参考而非调容差；判别器 `diagnose_o0_vs_fp64.py`，负向对照 `verify_exact_baselines.py`）。`output_1` 携带近零救援键（`bf16_atol_output_1` 32 / `max_abs` 64——钳后 x × rsigma=10 边界 × gelu 融合反向链的 errmax 16 外推 2x，rtol 保持模块 0.05）。**参数梯度槽不再产出**（schema v2，适配器只返回 o0/o1/contract）：LayerNormMLP 不在 Megatron-LM 调用路径上、参数梯度组合行为无生产消费方，且对 NVTE golden 本就不可判（o2-o5 被求和项 dln/d_fc1_out 是链派生互噪声 3#15；o6/o7 是批量求和累积顺序互噪声 √N 增长）——`non_judged_outputs` 的"只测不判"阶段随之退役。适配器反向以参数 `requires_grad=False` 运行（跳过 wgrad GEMM 与其所需的大激活保存——多十亿元素形状的峰值显存主宰；A/B 实测 x.grad 与 o0 逐位不变）；`silu_bwd` 同步改 aclnn 融合反向（原手搓公式物化 ~4x 输入的 fp32 临时量，与融合实现逐位一致）。参数梯度的覆盖责任：kernel 级 standalone `layernorm` / `linear` / `layernorm_linear` 守卫，gross 级 `tests/pytorch` sanity。恢复方式：适配器加回参数梯度槽 + golden 重采。
9. **norm 族 x 采样受 rsigma 上界约束**：LayerNorm/RMSNorm 类模块（当前 `layernorm_mlp`；`layernorm_linear` 同款策略）的 x 采样池在生成器里按 `rsigma = 1/sqrt(var + eps) <= 10` 钳制——std 不足的区间绕自身中心放大，case JSON 记录钳后区间，两侧数据一致。eps 主导区（var ≪ eps）下 dX 的绝对误差下限随 rsigma 线性放大，**粗 dtype 与 fp32 同样病态**（fp32 破的不是相对容差而是绝对硬门），故该良态性约束不区分 dtype。未建模的分布桶不钳制（代价是可见红，而非静默错误区间）；钳制后仍违例的建模桶由生成器断言拒绝；零宽桶（常量 x，xhat ≡ 0，无放大路径）放行。

---

## 五、泛化规模旋钮（试点：layernorm）

### 5.1 为什么必须带 generate_*.py

ATK 原生泛化对**每个 tensor 输入独立采样 shape**（各自一套 shapes 配置，未声明时落 `DEFAULT_DIM_NUMBERS`/`DEFAULT_DIM_VALUES` 随机池），没有跨输入耦合机制。而本套件 21 个模块的输入全部存在 shape 依赖（gamma/beta = [x 最后一维]、dy = x.shape、weight = [out, in] 等）。

只用 yaml+function 泛化时，dy/gamma/beta 拿到独立随机 shape，`module.weight.copy_(gamma)` 直接 `RuntimeError: The size of tensor a (64) must match the size of tensor b (16)`。**泛化 = 驱动生成器的旋钮，不是去掉生成器。**

### 5.2 两个 yaml 旋钮的真实语义

| 字段 | 语义 | 说明 |
|---|---|---|
| `dtype_numbers` | **每个 dtype 生成的用例数**（不是总量）；ATK 原生默认 50（`DEFAULT_DTYPE_NUMBERS`） | dtypes=[bf16, fp16] × 50 → 主轨道 100 例。`generate:` 注册后原生引擎不再消费该字段——生成器接管它作为规模旋钮 |
| `extra_numbers` | **边界值轨道**：scalar 形/下界/上界/**空 tensor**/**inf-nan** 五类特殊输入，不是"更多随机 shape" | `all` 或正数会触发 ATK 把生成器**重新实例化再迭代一遍**（`reports.py` 的 `_generate_extra_case`）；自定义生成器必须在该 pass 主动返回空，否则整个矩阵被复制一份标记为 boundary |

**整建路线生成器的配套要求**（见 5.4 路线选择；钩子路线继承父类无需此条）：① `is_gen_extra=True` 时返回空列表；② `__init__` 里显式 `self.extra_len_lst = None`——ATK 探测该属性没有空保护（`hasattr(input_gen.extra_len_lst, ...)`），缺属性时 `extra_numbers: all` 直接 `AttributeError` 崩溃，`None` 会路由到显式的零 extra 分支。

**原生泛化的隐藏坑**：`BoundaryConfig` 默认 `has_empty: true, has_infnan: true`——yaml 不显式关闭时，原生泛化会生成空 tensor 和 inf/nan 输入。norm 族（rmsnorm/layernorm 族）backward 拒绝空 dy，直接失败。若某模块确要走原生泛化，需逐输入配 `boundary: {has_empty: false}`（必要时 `has_infnan: false`）。

### 5.3 layernorm 试点用法（钩子路线）

layernorm 走**钩子路线（单轨）**：原生引擎驱动采样（dtypes / dim_numbers /
dim_values / dtype_numbers 直接消费 yaml），生成器子类只重写官方扩展点
`after_input_config` 补跨输入耦合（dy:=x.shape、gamma/beta:=[x 末维]、
dtype 跟 x），并带 fail-closed 输入布局校验（yaml 输入布局不符直接
raise，不静默补错槽位）。**切换路线需同时改 yaml `generate:` 字段和
`atk case -p` 参数**；路线变更即用例集变更，离线 golden 须重采
。

钩子路线下扩规模/扩 dtype 只改 yaml：

```yaml
dtype_numbers: 50          # 每 dtype 用例数（原生语义直接生效，无需生成器配合）
inputs:
  - type: tensor           # x——只改 inputs[0] 的 dtypes，钩子自动镜像
    dtypes:
      values: [bf16, fp16]
```

复现三要素：**同 yaml + 同 seed（`atk case -s 1234`，CLI 默认）+ 同 ATK
版本 ⇒ 同用例**。在线 GPU/NPU 对比与 golden 复用依赖此性质——基线里要锁
seed 和 ATK commit。

钩子路线的 yaml 配套要求（其余模块通用）：

1. **顶层 `shape_distributions` 必须调到模块量级**：ATK 默认配额要求 30%
   用例 ≥1000 万元素、1% ≥10 亿元素；layernorm 的用例量级在 1M 内，
   配额不可达 → 引擎对每个用例重试 100 轮 reshape 重试
   （10 例 10 分钟跑不完，调对后秒级）。本模块设为 `[1048576, 0.4]`（期望约 40% 大例）。
2. **`max_length`（单个 tensor 的 numel 硬上限）设计三规则**：
   - **可达**：池子里的值必须能组合出 ≤cap 的 shape——cap 低于池子最小
     乘积时，`get_shapes_by_dim` 的拒绝循环**没有次数上限，直接死循环**
     （定位：`parameter_tensor.py:136`，ATK 源码注释自认
     "max_number_ele 较小时可能无限循环"）；
   - **留减半余量**：配额溢出时引擎会把 cap **减半**（`change_tensor_max_ele`），
     连续减半后不能跌破"可达"下限——所以 threshold 要设在池子乘积分布的
     高分位（≤40%），且 cap ≥ 2× 池子最小乘积。threshold=256K（约
     55% 采样达标）在三种 cap 下全部死循环；threshold=1M（约 40%）全稳；
   - **cap 也不能过大**：cap 抬高"乘积可达的尾部"，等效于把 threshold
     推进高密度区。cap=2^32 时：默认配额下引擎**配额猎取**出
     5.37 亿元素（1GB/tensor）的巨型用例，100 例 I/O ≈ 22GiB；调优配额
     `[[1048576, 0.4]]` 下自然大例率过高 → 减半链从 2^32 要 12 轮才收敛，
     同期占位输入（65536）12 轮减半正好触底 16 → **直接死循环**（timeout
     实证）。测试 yaml 的 16M 峰值（=64^4）正是配额猎取落进 [10M, 100M)
     段的结果，由配额机制决定；
   - **匹配目标规模**：cap ≈ 目标最大 shape 的 2 倍（本模块 real 档
     2048×1024=2M → cap=2M；取 2 倍是给配额重试留余量）。测试 yaml 里
     `max_length: 4294967295`（=2^32，等于默认值没设）会采出 1677 万
     元素用例，golden 采集显著变慢。
3. **被耦合的输入（dy/gamma/beta）要给显式最小 shapes 占位**：无 shapes
   声明落默认配置（max_length=2^32），引擎在配额统计时给这些"反正要被
   钩子改写"的输入反复采巨形 shape。占位用 `[1] 维 × [16]` + 小
   max_length 即可，钩子会整体覆盖。
4. **boundary 分类由逐输入开关管**（`BoundaryConfig` 五类：scalar/lower/
   upper/empty/infnan，默认全开）：本模块 tensor 输入关 `has_empty/
   has_infnan/has_scalar`（norm 族空输入 kernel 非法、inf/nan 无对比意义、
   scalar 违反 2-4 维合同），保留 lower/upper 值边界（合法）。attr 输入
   不能带 boundary 字段（design_config 校验拒绝）。`extra_numbers` 默认 0，
   测试可开 `all`——本模板下只生成 lower/upper 值边界用例；漏关
   scalar/empty 时钩子的退化守卫在**生成期** fail-fast（带修复指引的
   报错，不再是 opaque 崩溃，也不再是生成器静默归零）。
5. dtype 扩展只改 x 的 dtypes——钩子把 x.dtype 镜像到 dy/gamma/beta
   （`params_dtype` 必须与激活同 dtype，否则 TypeError）；fp16 新增仍受
   §四.1 golden 重采铁律约束。

### 5.4 路线选择：钩子 vs 整建（按约束类型，不按模块一刀切）

| 约束类型 | 钩子路线（重写 `after_input_config` / `after_case_config`） | 整建路线（生成器拥有完整用例列表） |
|---|---|---|
| 纯 shape 耦合（dy=x、gamma=[x末维]、weight=[out,in]） | ✅ 官方设计用途，代码量 ~1/10，`dtype_numbers`/分布机制免费继承 | ✅ 可行但重 |
| 值语义耦合（moe routing_map 的 token 分布、rope freqs 重复角度格式、grouped_linear m_splits 之和） | ✅ 可行：在 `after_case_config` 渲染语义值（moe 族的 spec 渲染即此路线） | ✅ 可行 |
| 强对齐（mxfp8 %32、blockwise %128、DPA cuDNN 矩阵） | ⚠️ 钩子就地 reconcile（DPA 的 %8/%16/上界钳回）或靠对齐 dim 池结构性保证（量化两模块纯 yaml） | ✅ 可行 |
| 分布质量（hidden/token 分池、curated 锚点） | ⚠️ 单一 dim_values 池，位置角色不分（yaml 无法表达"末维=hidden"） | ✅ 精确分池 |

当前路线分布：**18 个模块走钩子路线**（含 moe 族 / rope / grouped_linear——值语义耦合由钩子在 `after_case_config` 渲染 spec 实现）；**量化三模块零跨输入耦合，走纯 yaml 路线**（`generate: default`，无生成器文件）。凡存在跨输入耦合的模块必须保留生成器（"只用 yaml+function"不可行，见 5.1；零耦合的量化三模块是唯一例外）。
