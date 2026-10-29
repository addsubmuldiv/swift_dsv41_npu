# TENPU ATK 一致性看护套件

**TENPU（TransformerEngineNPU）vs NVTE（NVIDIA Transformer Engine）数值一致性守护体系**——以 Megatron-Core r0.18 的 TE API 调用清单（110 个唯一 API / 507 处调用）为需求源，对 Megatron 训练链路依赖的 TE 接口建立 ATK 自动化精度看护。

---

## 1. 模块一览（21 个）

| 模块 | 看护对象（Megatron 封装） | 用例 | 基线 | 状态 |
|---|---|---|---|---|
| `linear` | TELinear / Column / RowParallel | 14 | NVTE golden（槽 0/3/4 为精确 fp64 基线） | ✅ 14/14@mixed_tolerance_bm |
| `rmsnorm` | TENorm（RMSNorm 分支） | 前向 150 + extra 14；反向 150 + extra 14 | NVTE golden | ✅ 150/150、150/150、14/14、14/14@mixed_tolerance_bm |
| `layernorm` | TENorm（LayerNorm 分支） | 前向 150 + extra 14；反向 150 + extra 14 | NVTE golden | ✅ 150/150、150/150、14/14、14/14@mixed_tolerance_bm |
| `layernorm_linear` | TELayerNormColumnParallelLinear | 12 | NVTE golden | ✅ 12/12@mixed_tolerance_bm |
| `layernorm_mlp` | （P3 纵深：Megatron 0.18 不直接调用） | 11 | NVTE golden | ✅ 11/11@mixed_tolerance_custom（0.05 + 参数梯度近零救援，自定义口径，不可切原生） |
| `grouped_linear` | TEGroupedLinear（MoE） | 14 | NVTE golden | ✅ 14/14@mixed_tolerance_bm |
| `dot_product_attention` | TEDotProductAttention | 32 | NVTE golden | ✅ 32/32@mixed_tolerance_bm |
| `rope` | apply_rotary_pos_emb + fused-QKV | 21 | NVTE golden | ✅ 21/21@mixed_tolerance_bm |
| `fused_activation` | te.activation（fused gelu/silu 等） | 20 | NVTE golden | ✅ 20/20@mixed_tolerance_bm |
| `float8_quantization` | Float8Tensor（per-tensor E4M3/E5M2） | 普通 150 + extra 13 | NVTE golden | ✅ 150/150、13/13@mixed_tolerance_bm |
| `mxfp8_quantization` | MXFP8Tensor | 10 | **golden-free**（CPU 理论界） | ✅ 10/10@mixed_tolerance_custom（三层守护自定义口径，不可切原生） |
| `blockwise_quantization` | Float8BlockwiseQTensor（128×128） | 10 | **golden-free**（CPU 理论界） | ✅ 10/10@mixed_tolerance_custom（三层守护自定义口径，不可切原生） |
| `fused_optimizer` | FusedAdam | 8 | NVTE golden | ✅ 8/8@mixed_tolerance_bm |
| `multi_tensor` | multi_tensor_scale / l2norm | 8 | NVTE golden | ✅ 8/8@mixed_tolerance_bm |
| `router` | fused_topk / score / moe_aux_loss | 22 | NVTE golden | ✅ 22/22@mixed_tolerance_bm |
| `moe_permute` | moe_permute（MoE token 置换） | 150 | NVTE golden | ✅ 150/150@allclose |
| `moe_permute_with_probs` | moe_permute_with_probs（置换 + probs） | 150 | NVTE golden | ✅ 150/150@allclose |
| `moe_permute_and_pad_with_probs` | moe_permute_and_pad_with_probs（置换 + probs + drop_and_pad） | 150 | NVTE golden（CPU 基线） | ✅ 150/150@allclose |
| `moe_sort_chunks_by_index` | moe_sort_chunks_by_index（chunk 重排） | 150 | NVTE golden | ✅ 150/150@allclose |
| `moe_sort_chunks_by_index_with_probs` | moe_sort_chunks_by_index_with_probs（chunk 重排 + probs） | 150 | NVTE golden | ✅ 150/150@allclose |
| `moe_unpermute` | moe_unpermute（MoE 逆置换） | 150 | NVTE golden / CPU 基线 | ✅ 150/150@allclose |

---

## 2. 基线与环境

**所有 golden 模块的 NVTE 基线为 pinned 版本：`release_v2.17`（v2.17.1，commit `4329ff84`）。**

| 项 | 值 / 获取方式 |
|---|---|
| ATK 工具仓库 | https://gitcode.com/Ascend/ATK.git —— 源码编译安装，**当前 pinned commit `bac1aa7`（内网实测，完整 hash `bac1aa7b77687f4dc9876abb489515f2cbcd9dca`）** |
| NVTE 基线仓库 | https://github.com/NVIDIA/TransformerEngine |
| NVTE 基线分支 / 版本 | `release_v2.17`（v2.17.1）——**golden 采集（`collect_nvte_golden.py`）必须在此构建上运行** |
| TENPU 被测仓库 | 本仓库 `atk_api_consistency` 分支 |
| 需求源 | `Megatron_Core_r0.18.0_TE_API_Analysis_一致性看护细化.xlsx`（110 API / 507 调用） |
| NVTE 侧硬件约束 | per-tensor FP8：任意 CUDA GPU（kernel 有 CC100 运行时 fallback）；blockwise：SM90+ 且 CUDA≥12.9；MXFP8：SM100（Blackwell）——后两者内网不可采，走 golden-free |
| TENPU 侧硬件 | FP8 / MXFP8 / Blockwise kernel 需 A5 级（Ascend 950+）NPU；其余模块 910B3 即可 |
| 参考环境（实测栈） | 8×910B3，CANN 26.0.rc1，torch_npu 2.9.0.post2，conda env `clc`，`source /usr/local/Ascend/ascend-toolkit/set_env.sh` |

> **换 NVTE 基线版本的完整动作链**：checkout 新版本 → 重建 NVTE → **重采全部 golden 模块的 golden**（契约 exact 比较，NVTE 行为变化即旧 golden 失效）→ 全模块复测。golden-free 模块（mxfp8 / blockwise）不受 NVTE 版本影响（基线是 CPU 数学）。

> **换 ATK 版本**：同理 checkout → 重装 → 重采 golden → 复测（比较器/收集器行为随 ATK 变化会使旧 golden 失效）。

**ATK 安装（golden 采集环境与 NPU 环境都要装）**：

```bash
git clone https://gitcode.com/Ascend/ATK.git
cd ATK && git checkout bac1aa7b77687f4dc9876abb489515f2cbcd9dca
pip install wheel && python setup.py bdist_wheel
pip install dist/*.whl
```

安装后 `atk case` / `atk task` 命令与 `python -c "import atk"` 均可用即成功。

**TransformerEngine（NVTE 基线）安装（仅 golden 采集环境需要）**：

```bash
git clone https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine && git checkout release_v2.17   # v2.17.1
```

安装步骤（含 Docker 镜像方式）见 NVTE 官方文档：
https://github.com/NVIDIA/TransformerEngine#setup —— 按所选 PyTorch / CUDA
版本完成构建后，`python -c "import transformer_engine; print(transformer_engine.__version__)"`
输出 2.17.1 即就绪。注意 golden 采集的 Python 进程中 `transformer_engine`
必须解析到**此构建**（如环境内存在其他 TE 安装，先卸载或调整 import 优先级）。

**golden 重采铁律（全局规则）**：契约按精确比较、数值按逐元素比较——用例 JSON、适配器输出列表或契约字段有任何改动，都必须**用同一适配器版本在 CUDA 上重采 golden** 并重跑 NPU 任务。过期 golden 的典型表现是契约尺寸不匹配（`size ... is not equal`）或无法解释的逐用例失败；不要调容差去掩盖，先重采。

> **少数模块的部分槽位基线不是 NVTE 输出而是精确 fp64 参考**（`linear` 的槽 0/3/4、`layernorm_linear` 的 wgrad/dbeta、`grouped_linear` 的 wgrad、`layernorm_mlp` 的前向）。当某个槽位的 NVTE kernel 精度低于被测侧或两侧为合法的不同数据流时，把它换成精确参考是**基线选择而非放宽容差**：真 bug（标度错/轴错/槽位互换/漏 bias）对精确参考仍然必红。这类槽位必须满足两条——(a) 替换只在 golden 侧（被测侧被替换就变成「参考比参考」、永远全绿），(b) 有负向对照脚本证明它没变瞎（各模块 `verify_exact_baselines.py`）。因此**基线替换后，重采 golden 不能自证**，必须用该模块的负向对照脚本另证一次。判据与机制见各模块适配器 docstring 与模块 README。

---

## 3. 快速上手

RMSNorm 已拆分为独立前向和反向。其离线命令见
[RMSNorm README：离线测试](module/rmsnorm/README.md#离线测试)：
GPU 采集脚本自动准备并保存反向共享统计量 rstdevs，复制产物到 NPU 后回放比较；
NPU 使用采集器导出的离线 JSON。普通和 extra 的前向、反向分别采集。

三步流程（以 rope 为例，其余模块同构，命令见各模块 README）：

```bash
# 1) 生成用例（TENPU 仓库根，ATK 在 PYTHONPATH）
atk case -f tests/atk/module/rope/rope.yaml -p tests/atk/module/rope/generate_rope.py

# 2) 收集基线
#    golden 模板（CUDA 主机，需 pinned NVTE）：
PYTHONPATH=/path/to/ATK:/path/to/this/module_dir \
  python tests/atk/module/rope/collect_nvte_golden.py \
  --cases result/rope/json/all_rope.json --input-dir ./rope_inputs --golden-dir ./rope_goldens
#    golden-free 模板（任何机器，纯 CPU 数学）：
python tests/atk/module/mxfp8_quantization/collect_cpu_reference.py \
  --cases result/mxfp8_quantization/json/all_mxfp8_quantization.json \
  --input-dir ./mxfp8_inputs --reference-dir ./mxfp8_quantization_reference

# 3) NPU 主机执行对比
atk task -c result/rope/json/all_rope.json \
  -n tests/atk/module/rope/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/rope/function_rope.py --input_data ./rope_inputs -tk accuracy
```

## 4. 目录结构与共享设施

```
tests/atk/
  ├── README.md                 # 本文档（操作入口）
  ├── GENERALIZATION.md         # 泛化约束资料（dtype/dim/shape 边界，测试泛化必读）
  └── module/
      ├── _comparators.py       # 共享比较器：allclose / mare / mixed_tolerance
      │                         # （按 <module>_<standard> 注册，yaml standard.acc 选型；
      │                         #   支持按输出位覆盖：bf16_atol_output_2 等；
      │                         #   支持判决域收窄：non_judged_outputs 槽位只测不判，
      │                         #   error_info 带 not judged 标记并保留统计）
      ├── _cpu_baselines.py     # 派生自 Megatron-LM（core_v0.13.0rc3, Apache-2.0）的
      │                         #   permute/unpermute CPU 基线：mp_pad_probs_cpu /
      │                         #   moe_unpermute_pad_probs_cpu，供
      │                         #   moe_permute_and_pad_with_probs 及 moe_unpermute
      │                         #   （仅 pad_with_probs 分支）替代 NVTE CUDA golden
      ├── _moe_convert.py       # golden 输出适配：TE row_id_map → NPU 布局
      │                         #   （_convert_row_id_map_to_npu）、未初始化槽确定化
      │                         #   （_mask_unused_row_id_map）、padded 输出按 expert
      │                         #   真实 token 抑制/裁剪（_suppress_padded_permuted_to_npu
      │                         #   / _suppress_padded_probs_to_npu），使 NVTE golden
      │                         #   与 NPU/TENPU 输出结构可比
      └── <name>/               # 21 个模块，各 7 件套
```

**比较器选型速查**：

| 比较器 | 判据 | 适用 |
|---|---|---|
| `*_mixed_tolerance`（默认） | matched_ratio ≥0.99 + abs/ULP 双兜底 | 近零元素相对误差放大会误报的算子（绝大多数） |
| `*_mare` | mean/max 相对误差双门 | 无近零病态的纯数值对比 |
| exact（非浮点自动） | `torch.equal` | 契约、uint8 payload、routing map |

> **判据来源**：ATK 官方《精度标准说明》——
> https://gitcode.com/Ascend/ATK/blob/master/docs/ATK%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/02%20%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/%E7%B2%BE%E5%BA%A6%E6%A0%87%E5%87%86%E8%AF%B4%E6%98%8E.md
> （single_bm 及其阈值体系引自 CANN opbase 算子精度实验标准；本套件的
> `_comparators.py` 为其自包含实现，mixed_tolerance 判据与 ATK 原生
> `mixed_tolerance_bm` 对齐）。

---

## 5. 如何新增一个模块

> 用例的 dtype/dim/shape 支持边界见 [GENERALIZATION.md](GENERALIZATION.md)——新增模块的矩阵设计同样受其约束分类指导（API 合同 / kernel 限制 / golden 侧约束三层）。

1. **选型**：查 NVTE kernel 架构门 → golden 或 golden-free；
2. **复制参考模块**：golden 用 `rmsnorm`，golden-free 用 `mxfp8_quantization`；改名 `<name>`、注册 `generate_<name>_p0` 与 `function_<name>`；
3. **本机验证**再上设备：生成器产出冒烟 + 适配器纯逻辑（参考值/契约编码）CPU 冒烟 + ATK 路径解析模拟（mxfp8 的收集器自检行即此用途）；
4. **设备首跑**：golden 模块先 CUDA 采 golden。
