# TENPU MoE Router ATK 看护模块

看护 `transformer_engine.pytorch.router` 中支撑 Megatron 融合 router 路径的
三个 MoE router API。基线为 NVTE golden（CUDA 采集），数值与 API 契约双重
比对：

- `fused_topk_with_score_function` —— probs（稀疏、logits dtype）+ bool
  routing_map + 播种的 `grad_logits`。
- `fused_compute_score_for_moe_aux_loss` —— bool routing_map + fp32 scores +
  播种的 `grad_logits`。
- `fused_moe_aux_loss` —— 标量 loss + 播种的 `grad_probs`。

## 平分（tie）处理：选择类输出按"索引无关"方式比较

topk 在分数完全相等（平分）时选哪个 expert 都是合法行为，NVTE golden 与
TENPU kernel 的选择可以不同。因此适配器在返回前对选择类输出做**索引无关**
的规范化（`function_router._canonical_rows`）：

- `topk` 变体：`probs` 按行排序后作为主输出；**secondary 槽位不再是 bool
  routing_map，而是 `probs * routing_map` 按行排序**——bool map 排序只是
  每行 popcount 模式，信息量与契约的 popcount 字段重复；乘 map 清零未选中
  位置后排序，得到"选中分数多重集 + 尾部零"，真正校验边界分数取值，且从
  bool 精确比较变为带容差的浮点比较。`probs` 本身已是稀疏的，所以该乘积
  与主输出相同，相当于对主输出的复核。
- `score` 变体：secondary 槽位同样换成 `scores * routing_map` 排序。
  `scores` 本身是 logits 的逐元素函数，与选择无关，保持原位比较。
- `grad_logits` 保持原位：其取值随 expert 不同，不存在平分不变形式。
  根治靠**输入间隔化**（见下）：fp32 选择类用例的 logits 在调 API 前按行
  拉开最小间隔，近平分从源头消除。
- 间隔化（`function_router._separate_rows`）：每行稳定降序排序后做阶梯
  `v'[i] = min(v[i], v'[i-1] - h)`（h=1e-4，无循环实现
  `cummin(v[j]+h*j) - h*i`），仅用正确舍入的逐元素算子 + 稳定排序 +
  cummin + scatter，**跨设备逐位一致**——在线模式下 NPU 和 GPU 各自跑同一
  adapter，任何依赖分数计算的条件分支都会因两端 libm 舍入差异分叉输入。
  拉开后各处选择边界（全局 topk、组排序、掩码后 expert）的分数间隙达数百
  ULP，两端选择必然一致；且无需知道边界在哪，全行成对拉开天然覆盖。
  低精度选择用例走稠密 topk==E（生成器强制），无边界无需间隔化；
  aux_loss 无选择，不处理。
- 间隔化修不掉的残余（概率低，依赖精度门 abs-error 兜底）：expert_bias
  引起的和碰撞（`f(logit)+b`，bias 独立于 logits 间隔）；组分数为多个
  分数之和（`topk // group_topk >= 2`）时的巧合相等（`per == 1` 的已确认
  故障被完整修复）；sigmoid 饱和区近平分（该处 `s'(x) ≈ 0`，grad 本身
  近零，无害）。

规范化对两侧一致生效（同一适配器代码），且排序不掩盖任何取值层面的差异，
只放弃"选中了哪个 expert 位置"——这不是 API 契约的一部分（Megatron 从单一
实现消费 probs/routing_map）。适配器输出语义与契约字段已变：**必须用同一
适配器版本重采 NVTE golden**，旧 golden 与新输出不再可比。间隔化位于
`run_router_case` 内部，在线模式 NPU/GPU 两端自动一致，离线 golden 路径
（`collect_nvte_golden.py` 调用同一函数）也自动获得相同处理。

覆盖范围：三个变体（`topk` / `score` / `aux_loss`）由首个 string attr 分派；
`score_function`（softmax/sigmoid/sqrtsoftplus）、`use_pre_softmax`、组受限
路由（`num_groups`/`group_topk`）、sigmoid 路径上的 `scaling_factor` +
`expert_bias`、`topk` 1..E、bf16/fp16/fp32 三档（各 50 例，共 150 例；非 fp32 的
选择类用例强制稠密 topk==E，见 GENERALIZATION §一）—— 全部由 yaml 采样池驱动，
规模与组合不再固定。

## 文件构成

- `router.yaml`：ATK 设计文件与精度容差，当前选用**混合容差标准**
  （`router_mixed_tolerance`；bool routing_map 与 uint8 契约按整数精确比较）。
- `generate_router_hook.py`：钩子路线生成器（唯一路线）——原生引擎驱动采样，
  生成器补变体决定的跨槽位耦合（t1/g0/g1 形状、a2/a3/a4 的类型与取值、
  组受限路由的配对与窗口合法性）。
- `function_router.py`：纯设备适配器、契约编码器、比较器注册。
- `collect_nvte_golden.py`：CUDA/NVTE golden 与规范输入采集器。
- `nodes_npu_vs_golden.yaml`：NPU + 离线 golden 节点配置。
- `result_schema.json`：把 ATK 编号 tensor 文件映射回语义。

## 1. 生成用例

在 **golden 采集环境（CUDA 主机）**、TENPU 仓库根目录下执行
（ATK 已安装，安装方法与 commit 点见 `../../README.md` 基线与环境节）：

```bash
atk case \
  -f tests/atk/module/router/router.yaml \
  -p tests/atk/module/router/generate_router_hook.py
```

预期用例文件：`result/router/json/all_router.json`（默认 `dtype_numbers: 6`
× 2 个 dtype ⇒ 12 例）。用例集由 yaml 采样池与 `-s <seed>`（CLI 默认 1234）
决定：同 yaml + 同 seed + 同 ATK 版本 ⇒ 同用例。

## 2. 采集规范输入与 NVTE golden（CUDA 主机）

仍在 TENPU 仓库根目录下执行：

```bash
python tests/atk/module/router/collect_nvte_golden.py \
  --cases result/router/json/all_router.json \
  --input-dir ./router_inputs \
  --golden-dir ./router_goldens
```

## 3. NPU 主机上对比 TENPU vs NVTE

把 `result/router/json/all_router.json`、`router_inputs` 与 `router_goldens` 拷贝或挂载进 NPU 环境（json 不拷则需在 NPU 上重新执行第 1 步生成）。`nodes_npu_vs_golden.yaml` 中的路径相对于工作目录，可改为绝对路径。

```bash
atk task \
  -c result/router/json/all_router.json \
  -n tests/atk/module/router/nodes_npu_vs_golden.yaml \
  -p tests/atk/module/router/function_router.py \
  --input_data ./router_inputs \
  -tk accuracy
```

契约字段的任何改动，都要求**用同一适配器版本在 CUDA 上重采集 golden** 并
重跑 NPU 任务。用例集本身的变更（如本轮从 22 例 curated 矩阵切到采样生成的
12 例）同理：旧 golden 与新 case 编号/形状不再对应，必须重采。过期 golden
表现为契约尺寸不匹配或无法解释的逐用例失败；不要调容差去掩盖它——先重采集。
