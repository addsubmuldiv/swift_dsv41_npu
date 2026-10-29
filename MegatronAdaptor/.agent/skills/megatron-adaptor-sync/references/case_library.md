# 飞轮案例库（诊断知识固化）

> **来源**：`nightly_case_library.jsonl` 中的 `case` 类型记录。
> **为什么固化到这里**：案例库是飞轮的运行时输入（① fetch 读 `last_good`、⑥ diag 搜相似案例），
> 同时每夜累积构建结果（`nightly_run` 记录）。其中 `case` 记录是**可复用的诊断知识**（根因 / bisect / 修复），
> 属于「通用内容」，随 skill 版本化、不随运行时文件丢失。
> **分工**：`nightly_run` 构建结果由 ⑨ CI_REPORT 推送至 CI 仓（`sync_build_results/`），代码仓不再跟踪；
> 本文件只收诊断案例。**新案例需人工/脚本同步固化到这里**（当前无脚本自动写入 `case` 记录）。

## 案例一览

| 日期 | 失败用例 | Breaking Commit | 根因类型 | 修复要点 |
|---|---|---|---|---|
| 2026-07-13 | `test_tokenizer_dataset` | ~~052209940~~（已被 07-20 案例纠正为误判） | 条件重构违反 De Morgan 定律 | 补 `.cuda()` + `type_wrapper` 硬化 |
| 2026-07-20 | `test_tokenizer_dataset` | `223e244f5` | **沉睡 bug 激活**（路径路由改变，不改 buggy 代码本身） | wrapper 补 `.cuda()`，不改上游条件 |
| 2026-08-17 | `test_basic_training` / `test_checkpoint_advanced` | TE-NPU `4b63efe`（非 MLM MR） | **TE-NPU 偏离社区 TE 契约** | 改 TE-NPU `state_dict()` 把 step 序列化 int（对标社区），runtime tensor 不动 |

---

## 案例 1：2026-07-13 — `test_tokenizer_dataset` 参数留在 CPU

> ⚠️ 本案例的 breaking commit 判定已被 2026-07-20 案例纠正：`052209940` 是**误判**，
> 真正 breaking commit 是 `223e244f5`（见案例 2 的 `corrects_previous_case`）。本案例的
> **根因与修复思路仍有效**（同一失败现象，wrapper 补 `.cuda()` 方向正确）。

- **错误现象**：`TypeError: Wrapped parameters must be one of torch.cuda.*Tensor. Received torch.BFloat16Tensor`
- **根因**：MLM commit `052209940`（MiMo training）重构了
  `prepare_existing_model_chunks_for_distributed_training`，把 GPU 分配条件从 `not(A and B)`
  错误改为 `not A and not B`（**De Morgan 违反**）。当 `--use-cpu-initialization=True` 且未用 FSDP2 时，
  `.cuda()` 被跳过，参数留在 CPU。
- **Bisect**：代码级二分（3 次 grep 查询）→ `052209940`
- **修复**：① monkey-patch `prepare_existing_model_chunks_for_distributed_training`，
  返回后检测 CPU 参数并调 `.cuda()`；② 硬化 `type_wrapper`，把裸 `torch.X` 类型转为 `torch.cuda.X`。
- **涉及文件**：`patches/megatron/megatron_basic.py`、`features_manager/megatron/megatron_basic.py`、`patches/requirements/requirements.py`
- **验证**：12/12 PASS
- **教训**
  - 条件重构的 De Morgan 违反很常见——检查条件变换
  - `register_patch` 的 wrapper 函数名必须以 `_wrapper` 结尾
  - 代码级 bisect（在中点检查代码特征）比跑测试快，但不如真实验证可靠（见案例 2）

---

## 案例 2：2026-07-20 — `test_tokenizer_dataset` 沉睡 bug 被路径路由激活

- **错误现象**：`TypeError: Received torch.BFloat16Tensor`（与 07-13 同现象，复发）
- **根因**：MLM commit `223e244f5`（Finish ModelBuilder integration）改变**代码路径路由**：
  `setup_model_and_optimizer` 改走 `_build_model_wrapper() → builder.build_distributed_models()
  → prepare_existing_model_chunks_for_distributed_training()`，替代旧的 `get_model()`。
  而 `dist_utils.py` 中有一个写于 `0dc36dfc6` 的**沉睡 bug**（`not use_torch_fsdp2 AND not use_cpu_initialization`），
  GPT 预训练从未走到过这段路径，因此一直未被触发；旧 `get_model()` 用的是正确的
  `not(use_torch_fsdp2 AND use_cpu_initialization)`。
- **Bisect**：`git bisect run` + 真实 NPU 测试执行（685 commits，9 步二分）→ `223e244f5`
- **修复**：在 `prepare_existing_model_chunks_for_distributed_training` 上挂 wrapper，
  返回后检测 CPU 参数并调 `model_module.cuda()`。**不修改上游条件**（其他调用方可能依赖新语义）。
- **涉及文件**：`patches/megatron/dist_training.py`、`features_manager/megatron/megatron_basic.py`
- **验证**：Fix verification PASS，Regression check PASS
- **教训**
  - **沉睡 bug 激活**：breaking commit 可能不改 buggy 代码本身，只改变哪条代码路径到达它
  - **假设派生关系陷阱**：两个位置相似的条件可能是各自独立写的，不是互相派生——不要假设同源
  - **bisect 方法论**：必须用 `git bisect run` + 真实测试执行，不要用代码级 grep 二分（案例 1 因此误判）
  - 跨路径比较条件时，用 `git log --all -S` 独立追踪每个条件的来源
- **纠正**：`corrects_previous_case` — 07-13 案例误判 `052209940`，真实 breaking commit 是 `223e244f5`（路径路由，非条件重构）

---

## 案例 3：2026-08-17 — `test_basic_training` / `test_checkpoint_advanced` FusedAdam step 序列化（TE-NPU 偏离社区契约）

- **错误现象**：checkpoint 保存时 `distrib_optimizer.state_dict()` 里 `assert len(steps) <= 1` 挂掉：
  `AssertionError: steps: [tensor(10, device='npu:1'), tensor(10, device='npu:1')]`
- **根因**：TE-NPU `4b63efe`（2026-08-15）为适配 `torch._fused_adamw_` 把 `group["step"]` 从 Python int 改成 **device tensor**（`__init__` 即初始化，运行时推进）。而 Megatron-LM `distrib_optimizer.py` 对 step 用 `set()` 去重——对 tensor 按 **identity** 去重，两个数值相等的 tensor 都保留 → `len(steps)==2` → assert 挂。
- **对标社区判断归属**（关键）：**社区 TE**（NVIDIA TransformerEngine）`group["step"]` **默认是 Python int**（仅 `capturable=True` 时 tensor），`state_dict()` 序列化 int → Megatron `set()` 去重安全。TE-NPU 无条件改 tensor 是**偏离了它自己对标的社区契约**，且偏离**可避免**：`torch._fused_adamw_` 只在 kernel 调用点要求 tensor `state_steps`，TE-NPU 完全可在调用处临时物化 tensor、保持公开 `param_groups["step"]` 为 int（社区也是 kernel 内部物化）。→ 修复归属是 **TE-NPU**，不是 MA。
- **修复**（TE-NPU，9 行，`transformer_engine/pytorch/optimizers/fused_adam.py` `state_dict()`）：序列化层把 tensor step 转 int，**不动 runtime tensor step**：
  ```python
  for group in state_dict["param_groups"]:
      step = group.get("step")
      if torch.is_tensor(step):
          group["step"] = int(step.detach().item())
  ```
  `load_state_dict` 已实现 int→tensor 回填，save/load 回环无影响。
- **验证**：**MA 无 wrapper**（main @ f14d364 干净基线）下，`test_basic_training` + `test_checkpoint_advanced`（torch + torch_dist 双格式 save/load）全过，loss 收敛（10.417→10.383），零 assert/Traceback/OOM → 证明 TE-NPU 修复**充分**。
- **教训**
  - **判断修复归属先对标社区参考实现**：TE-NPU 是社区 TE 的移植，接口契约以社区为基准；TE-NPU 偏离契约 → 改 TE-NPU，MA 是适配层但不是所有 bug 的默认归属，MA wrapper 只是 TE-NPU 不可控时的下游 shim
  - 区分「runtime 语义」（tensor step 保 `torch._fused_adamw_`）与「序列化契约」（state_dict 输出 int）——修复只碰契约边界，不动 runtime
  - Megatron `distrib_optimizer.py` 的 TE/APEX 分支假设 step 是 int（torch optimizer 分支会用 `.item()`，TE 分支忘了）——社区 TE 开 `capturable=True` 时同样会挂，属潜在上游健壮性缺陷
