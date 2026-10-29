# API列表

本文按[简介](../introduction.md)中的六大功能模块组织TransformerEngineNPU（TENPU）的公开接口清单。TENPU的接口与原生TransformerEngine（NVTE）保持同名同签名，并在此基础上提供昇腾专属扩展。

**接口来源说明**

- 原生接口：与NVTE对齐的接口，语义一致，可参考[NVTE官方文档](https://docs.nvidia.com/deeplearning/transformer-engine/)了解详细行为；
- 扩展接口：昇腾扩展接口，仅在TENPU中提供。

> [!NOTE]
>
> TransformerEngineNPU和原生TransformerEngine的库名和模块路径相同（均为transformer_engine），同一环境内请勿同时安装两者。如已安装NVIDIA版TransformerEngine，请先执行`pip uninstall transformer_engine`卸载。

**文档结构**

除环境变量外，各表格均注明了其中接口的导入路径，导入方式示例：

```python
from transformer_engine.pytorch import Linear, autocast
from transformer_engine.common import recipe
```

通过下表可以快速了解各个功能模块的定位。

| 功能模块 | 定位 |
|------|------|
| [高性能基础层模块](#高性能基础层模块) | `Linear`、`LayerNorm`等基础层模块，`DotProductAttention`、RoPE等注意力模块，以及ops可融合算子 |
| [FP8/低精度训练](#fp8低精度训练) | 量化上下文、能力查询、Recipe策略、量化张量与Quantizer |
| [MoE支持](#moe支持) | Token排列与MoE路由 |
| [优化器与多Tensor算子](#优化器与多tensor算子) | 优化器步骤的多tensor融合算子 |
| [训练并行与显存优化](#训练并行与显存优化) | CPU Offload、激活重计算与分布式工具 |
| [数值调试](#数值调试) | 精度定位工具 |
| [环境变量](#环境变量) | 运行时行为开关 |
| [与NVTE的差异](#与nvte的差异) | 未提供/扩展/不支持的接口汇总 |

## 高性能基础层模块

**表 1** 基础层模块列表

从**transformer_engine.pytorch**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `Linear` | 类 | 全连接层，支持FP8/BF16、张量并行、延迟wgrad | 原生接口 |
| `GroupedLinear` | 类 | 分组全连接层（MoE场景），支持FP8与分组wgrad累积 | 原生接口 |
| `LayerNorm` | 类 | LayerNorm归一化层，支持zero-centered gamma、序列并行 | 原生接口 |
| `RMSNorm` | 类 | RMSNorm归一化层，支持zero-centered gamma、序列并行 | 原生接口 |
| `LayerNormLinear` | 类 | 归一化 + 线性层融合模块（`normalization`可选`"LayerNorm"`/`"RMSNorm"`） | 原生接口 |
| `LayerNormMLP` | 类 | 归一化 + 两层MLP融合模块，支持GLU激活 | 原生接口 |
| `TransformerEngineBaseModule` | 类 | 上述模块的公共基类 | 原生接口 |

> [!NOTE]
>
> 以下接口在TENPU中仅为保持接口兼容而保留，无实际功能：
>
> - `Fp8Padding` / `Fp8Unpadding`：NPU上直接透传输入，不执行填充/反填充；
> - `initialize_ub` / `destroy_ub` / `UserBufferQuantizationMode`：NVTE的UserBuffer通信重叠机制为CUDA特有，TENPU未实现。

**表 2** Attention模块列表

从**transformer_engine.pytorch**或**transformer_engine.pytorch.attention**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `DotProductAttention` | 类 | 融合注意力模块：支持GQA/MQA、self/cross注意力、`no_mask`/`causal`/`padding`/`arbitrary`等掩码类型、滑窗注意力（`window_size`）、THD变长序列（`cu_seqlens_q`/`cu_seqlens_kv`）；上下文并行内置Ulysses、Ring、KV-AllGather三种策略（`cp_comm_type`指定） | 原生接口 |
| `DotProductAttention`（Swap策略） | 类 | 上下文并行的Swap策略（`cp_comm_type="swap"`）及配套swap flash attention，昇腾扩展实现 | 扩展接口 |
| `RotaryPositionEmbedding` | 类 | RoPE位置编码，支持`rotary_percent`、位置插值、interleaved布局 | 原生接口 |
| `apply_rotary_pos_emb` | 函数 | 将RoPE应用到输入张量，支持`sbhd`/`bshd`/`thd`格式、`fused=True`融合实现、上下文并行 | 原生接口 |
| `apply_fused_qkv_rotary_pos_emb` | 函数 | 对Q/K/V一次完成融合RoPE（推理场景常用） | 原生接口 |
| `fused_apply_mla_rope_for_q` | 函数 | MLA（multi-latent attention）query 的 YARN RoPE：仅旋转末尾 `emb_dim` 维、前部 `qk_head_dim` 维直通，输出与上游 Triton kernel 逐元素一致；要求 Megatron 重复半区 cos/sin 表（`cat(freqs, freqs)`），其他表布局抛 `MLARopeUnsupportedError`；对应 Megatron `fused_apply_mla_rope_for_q` | 扩展接口 |
| `fused_apply_mla_rope_for_kv` | 函数 | MLA 压缩 KV 拆分：旋转单 head 的 `k_pos_emb` 并广播拼入 key，`value` 以视图返回；输出布局与约束同上；对应 Megatron `fused_apply_mla_rope_for_kv` | 扩展接口 |

**表 3** 可融合基础算子列表

从**transformer_engine.pytorch.ops**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `AllGather` / `AllReduce` / `ReduceScatter` | 类 | 集合通信算子，可与相邻算子融合执行 | 原生接口 |
| `LayerNorm` / `RMSNorm` / `L2Normalization` | 类 | 归一化类算子 | 原生接口 |
| `Quantize` | 类 | 量化算子 | 原生接口 |
| `Bias` / `ConstantScale` / `Identity` / `Reshape` | 类 | 数值与形状操作算子 | 原生接口 |
| `AddExtraInput` / `MakeExtraOutput` | 类 | 梯度累积辅助算子，配合多微批次场景 | 原生接口 |
| `GELU` / `SiLU` / `ReLU` / `QGELU` / `SReLU` / `ScaledSReLU` | 类 | 激活算子 | 原生接口 |
| `GLU` / `GEGLU` / `ReGLU` / `SReGLU` / `QGEGLU` | 类 | 门控激活算子 | 原生接口 |
| `SwiGLU` / `ScaledSwiGLU` / `ClampedSwiGLU` / `ScaledClampedQGeGLU` | 类 | SwiGLU系变体激活算子 | 原生接口 |

**表 4** 算子融合与组合接口列表

从**transformer_engine.pytorch.ops**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `FusibleOperation` | 类 | 可融合算子基类，上述基础算子的公共父类 | 原生接口 |
| `BasicOperation` | 类 | 基础算子基类 | 原生接口 |
| `FusedOperation` | 类 | 融合算子，封装融合后的整体执行 | 原生接口 |
| `Sequential` | 类 | 顺序组合容器，按序执行其中的算子 | 原生接口 |
| `register_forward_fusion` / `register_backward_fusion` | 函数 | 注册算子级的前向/反向融合模式 | 原生接口 |

> [!NOTE]
>
> 可融合算子API为实验性接口，后续版本可能调整。

## FP8/低精度训练

**表 5** 量化上下文接口列表

从**transformer_engine.pytorch**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `autocast` | 类（上下文管理器） | 量化上下文管理器，`enabled=True`时在其中的TENPU模块启用低精度计算 | 原生接口 |
| `fp8_autocast` | 类（上下文管理器） | FP8上下文管理器（`autocast`的FP8特化形式） | 原生接口 |
| `fp8_model_init` | 函数 | 以FP8主权重格式初始化模型 | 原生接口 |
| `quantized_model_init` | 函数 | 以量化格式初始化模型，按recipe自动选择存储类型 | 原生接口 |
| `reset_fp8_amax_history` | 函数 | 重置所有Float8量化的amax历史 | 原生接口 |
| `reset_hif8_amax_history` | 函数 | 重置HIF8量化的amax历史 | 扩展接口 |

**表 6** 量化能力查询接口列表

从**transformer_engine.pytorch.quantization**导入（`is_fp8_available`、`get_default_recipe`亦可从顶层**transformer_engine.pytorch**导入）：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `is_fp8_available` | 函数 | 查询设备/环境是否支持FP8，`return_reason=True`返回原因 | 原生接口 |
| `is_mxfp8_available` | 函数 | 查询MXFP8支持 | 原生接口 |
| `is_mxfp4_available` | 函数 | 查询MXFP4支持 | 扩展接口 |
| `is_fp8_block_scaling_available` | 函数 | 查询128×128块级缩放支持 | 原生接口 |
| `is_w4a8_available` | 函数 | 查询W4A8权重量化支持 | 扩展接口 |
| `is_hif8_delayed_available` | 函数 | 查询HIF8延迟缩放支持 | 扩展接口 |
| `is_nvfp4_available` | 函数 | 查询NVFP4支持（**TENPU当前不支持，恒返回False**） | 原生接口 |
| `get_default_recipe` | 函数 | 返回当前设备推荐的默认量化策略 | 原生接口 |

**表 7** 量化策略（Recipe）列表

从**transformer_engine.common.recipe**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `Recipe` | 类 | 策略基类 | 原生接口 |
| `Format` / `FormatEnum` | 类/枚举 | 量化数据格式定义（`kFloat8E4M3`、`kFloat8E5M2`、`kMXFP8`等） | 原生接口 |
| `DelayedScaling` | 类 | 延迟缩放：维护amax历史（`amax_history_len`）计算scale | 原生接口 |
| `Float8CurrentScaling` | 类 | 当前缩放（per-tensor current scaling） | 原生接口 |
| `Float8BlockScaling` | 类 | 128×128块级FP8缩放 | 原生接口 |
| `MXFP8BlockScaling` | 类 | OCP MX标准1×32 block MXFP8缩放 | 原生接口 |
| `MXFP832x32BlockScaling` | 类 | 32×32 block MXFP8缩放 | 扩展接口 |
| `MXFP4BlockScaling` | 类 | MXFP4块级缩放（NVTE对应能力为NVFP4） | 扩展接口 |
| `W4A8BlockScaling` | 类 | 权重INT4/激活INT8块级缩放 | 扩展接口 |
| `HIF8DelayedScaling` | 类 | HIF8延迟缩放（昇腾FP8格式） | 扩展接口 |
| `CustomRecipe` | 类 | 自定义策略，按QuantizerRole动态生成各角色策略 | 原生接口 |
| `QuantizerRole` | 类 | 量化角色（权重/输入/输出/梯度）定义，用于CustomRecipe | 原生接口 |

配套的RecipeState类（`transformer_engine.pytorch.quantization.state`，如`DelayedScalingRecipeState`）为框架内部使用的策略运行时状态，一般无需直接使用。

**表 8** 量化张量类型列表

从**transformer_engine.pytorch.tensor**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `QuantizedTensor` | 类 | 量化张量基类 | 原生接口 |
| `Float8Tensor` | 类 | per-tensor缩放的FP8张量（顶层亦可直接导入） | 原生接口 |
| `Float8BlockwiseQTensor` | 类 | 128×128块级缩放FP8张量 | 原生接口 |
| `MXFP8Tensor` | 类 | MXFP8缩放张量 | 原生接口 |
| `MXFP4Tensor` | 类 | MXFP4缩放张量 | 扩展接口 |
| `W4A8WeightTensor` | 类 | W4A8权重张量 | 扩展接口 |

**表 9** Quantizer列表

从**transformer_engine.pytorch.tensor**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `Quantizer` | 类 | Quantizer基类 | 原生接口 |
| `Float8Quantizer` | 类 | 延迟缩放FP8量化器 | 原生接口 |
| `Float8CurrentScalingQuantizer` | 类 | 当前缩放FP8量化器 | 原生接口 |
| `Float8BlockQuantizer` | 类 | 块级缩放FP8量化器 | 原生接口 |
| `MXFP8Quantizer` | 类 | MXFP8量化器（1×32） | 原生接口 |
| `MXFP832x32Quantizer` | 类 | 32×32 block MXFP8量化器 | 扩展接口 |
| `MXFP4Quantizer` | 类 | MXFP4量化器 | 扩展接口 |
| `W4A8WeightQuantizer` | 类 | W4A8权重量化器 | 扩展接口 |
| `HIF8Quantizer` | 类 | HIF8量化器 | 扩展接口 |

**表 10** 张量工具函数列表

从**transformer_engine.pytorch.tensor**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `cast_master_weights_to_fp8` | 函数 | 将主权重转换为FP8存储 | 原生接口 |
| `post_all_gather_processing` | 函数 | AllGather后的量化后处理 | 原生接口 |
| `replace_raw_data` | 函数 | 替换量化张量的底层数据 | 原生接口 |

## MoE支持

**表 11** Token排列接口列表

从**transformer_engine.pytorch**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `moe_permute` | 函数 | 按token→专家映射重排token（含可选probs乘法、padding） | 原生接口 |
| `moe_unpermute` | 函数 | `moe_permute`的逆操作，恢复token原顺序 | 原生接口 |
| `moe_sort_chunks_by_index` | 函数 | 按索引对chunk排序 | 原生接口 |
| `moe_sort_chunks_by_index_with_probs` | 函数 | 按索引与概率对chunk排序 | 原生接口 |
| `moe_permute_with_probs` | 函数 | 带expert probs的token重排 | 原生接口 |
| `moe_permute_and_pad_with_probs` | 函数 | 带probs与padding的token重排 | 原生接口 |

**表 12** MoE路由接口列表

从**transformer_engine.pytorch.router**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `fused_topk_with_score_function` | 函数 | 融合top-k路由（带评分函数） | 原生接口 |
| `fused_compute_score_for_moe_aux_loss` | 函数 | 融合计算MoE辅助损失评分 | 原生接口 |
| `fused_moe_aux_loss` | 函数 | 融合计算MoE辅助损失 | 原生接口 |

## 优化器与多Tensor算子

**表 13** 多Tensor算子列表

从**transformer_engine.pytorch.optimizers**导入（`MultiTensorApply`等亦在顶层导出）：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `MultiTensorApply` | 类 | 多tensor融合算子的执行器封装 | 原生接口 |
| `multi_tensor_applier` | 函数 | 按算子类型分发执行 | 原生接口 |
| `multi_tensor_scale` | 函数 | 多tensor缩放 | 原生接口 |
| `multi_tensor_scale_tensor` | 函数 | 多tensor缩放（tensor级接口） | 原生接口 |
| `multi_tensor_l2norm` | 函数 | 计算多tensor的L2范数 | 原生接口 |
| `multi_tensor_unscale_l2norm` | 函数 | 计算多tensor的L2范数并反缩放 | 原生接口 |
| `multi_tensor_compute_scale_and_scale_inv` | 函数 | 计算FP8 scale及其逆 | 原生接口 |
| `multi_tensor_compute_scale_inv_e8m0` | 函数 | 计算E8M0格式的scale逆 | 原生接口 |

## 训练并行与显存优化

**表 14** CPU Offload接口列表

从**transformer_engine.pytorch**导入：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `get_cpu_offload_context` | 函数 | 创建CPU Offload上下文，支持按层序列卸载激活/权重 | 原生接口 |
| `mark_not_offload` | 函数 | 标记tensor不参与offload | 原生接口 |
| `start_offload` | 函数 | 立即启动指定tensor的offload | 原生接口 |
| `is_cpu_offload_enabled` | 函数 | 查询CPU Offload是否启用 | 原生接口 |

**表 15** 分布式与重计算接口列表

从**transformer_engine.pytorch.distributed**导入（`checkpoint`亦可从顶层**transformer_engine.pytorch**导入）：

| API | 类型 | 说明 | 接口来源 |
|-----|------|------|------|
| `checkpoint` | 函数 | 激活重计算封装（基于`torch.utils.checkpoint`），FP8场景下感知量化状态 | 原生接口 |
| `activation_recompute_forward` | 类（上下文管理器） | 标记前向处于重计算阶段 | 原生接口 |
| `get_activation_recompute_contexts` | 函数 | 获取激活重计算上下文 | 原生接口 |
| `is_fp8_activation_recompute_enabled` | 函数 | 查询FP8激活重计算是否启用 | 原生接口 |
| `in_fp8_activation_recompute_phase` | 函数 | 查询当前是否处于FP8重计算阶段 | 原生接口 |
| `get_te_classes` / `has_te_modules` | 函数 | 枚举/检测网络中的TENPU模块（checkpoint兼容处理用） | 原生接口 |
| `set_tensor_model_parallel_attributes` | 函数 | 为tensor设置TP属性 | 原生接口 |
| `get_distributed_world_size` / `get_distributed_rank` | 函数 | 获取（指定通信组的）world size / rank | 原生接口 |
| `get_all_rng_states` / `set_all_rng_states` | 函数 | 获取/恢复全部RNG状态 | 原生接口 |

## 数值调试

**表 16** 数值调试接口列表

| API | 导入路径 | 类型 | 说明 | 接口来源 |
|-----|----------|------|------|------|
| `TEDebugState` | `transformer_engine.debug.pytorch.debug_state` | 类 | 调试状态容器 | 原生接口 |
| `set_weight_tensor_tp_group_reduce` | `transformer_engine.debug.pytorch.debug_state` | 函数 | 控制权重tensor的TP组reduce行为 | 扩展接口 |
| `DebugQuantizer` / `GroupedDebugQuantizer` | `transformer_engine.debug.pytorch.debug_quantization` | 类 | 在量化器外封装观测逻辑的调试Quantizer，用于在线数值观测 | 原生接口 |
| `DebugQuantizedTensor` / `GroupedDebugQuantizedTensor` | `transformer_engine.debug.pytorch.debug_quantization` | 类 | 调试用量化张量 | 原生接口 |
| `GemmComparisonConfig` | `transformer_engine.pytorch` | 类 | GEMM数值比对配置 | 扩展接口 |
| `configure_gemm_comparison` | `transformer_engine.pytorch` | 函数 | 运行时开启量化GEMM与参考实现的数值比对 | 扩展接口 |
| `get_gemm_comparison_config` | `transformer_engine.pytorch` | 函数 | 获取当前GEMM比对配置 | 扩展接口 |

## 环境变量

**表 17** 环境变量列表

| 环境变量 | 说明 | 接口来源 |
|----------|------|------|
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | 置`0`时启用确定性算法，保证结果可复现（损失部分性能） | 原生接口 |
| `NVTE_TORCH_COMPILE` | 置`0`时禁用TENPU算子的`torch.compile`路径 | 原生接口 |
| `NVTE_BACKWARD_OVERRIDE` | 启用反向数值覆写（精度调试用），取值`high_precision`/`dequantized`；`LayerNormMLP`场景暂未实现 | 原生接口 |

> [!NOTE]
>
> 以下NVTE环境变量在TENPU中仅为保持接口兼容而解析保留，设置后无实际作用：
>
> - `NVTE_FWD_LAYERNORM_SM_MARGIN` / `NVTE_BWD_LAYERNORM_SM_MARGIN` / `NVTE_INF_LAYERNORM_SM_MARGIN`：面向CUDA kernel调度的SM预留调优，不参与NPU kernel调度；
> - `NVTE_CPU_OFFLOAD_V1`：用于切换NVTE的CPU Offload v1代码路径，TENPU未实现该路径。

## 与NVTE的差异

### TENPU未提供的接口

以下NVTE接口在TENPU中**暂未提供**，业务代码如依赖这些接口需要适配：

- `MultiheadAttention`、`TransformerLayer`（高层封装模块，可用`DotProductAttention` + 基础模块自行组合）；
- `CudaRNGStatesTracker`（CUDA RNG管理）；
- `parallel_cross_entropy`、`interleave_glu_tensor` / `deinterleave_glu_tensor`；
- ops层的`Linear`、`GroupedLinear`、`BasicLinear`、`Dropout`算子及`register_forward_backward_fusion`注册接口；
- JAX/ONNX相关接口（`export`、`onnx_extensions`）。

### TENPU的昇腾扩展

- HIF8延迟缩放（`HIF8DelayedScaling`、`HIF8Quantizer`、`reset_hif8_amax_history`）；
- W4A8权重量化（`W4A8BlockScaling`、`W4A8WeightTensor`、`W4A8WeightQuantizer`）；
- MXFP4块级缩放（`MXFP4BlockScaling`、`MXFP4Tensor`、`MXFP4Quantizer`，NVTE对应能力为NVFP4）；
- `MXFP832x32BlockScaling`（32×32 block MXFP8）；
- Swap上下文并行注意力策略、GEMM数值比对工具等（见上文扩展接口）。
- MLA RoPE（`fused_apply_mla_rope_for_q` / `fused_apply_mla_rope_for_kv`）：Megatron MLA 的融合 RoPE 入口，NVTE 无对应接口，在昇腾上以 CANN 旋转算子实现，避免上游 Triton kernel 在 NPU 上的性能回退。输出与上游 Triton kernel 逐元素一致（旋转通道按堆叠顺序写回），逐调用回退不影响布局一致性。约束：cos/sin 须为 Megatron 的重复半区表（`cat(freqs, freqs)`，native kernel 反向仅对重复角度是精确伴随），其他表布局抛 `MLARopeUnsupportedError` 回退上游；不支持 packed（thd）、`rotary_interleaved=True` 与非 4D 输入，同样回退。

### 明确不支持

- NVFP4低精度量化（`is_nvfp4_available()`恒返回`False`）。
