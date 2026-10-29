# 断言质量指南

> Step 4.5（生成后上传前）和 Step 5（运行后）的质量检查依据。

---

## L0-L3 判定框架

### L0：测试真的执行到了吗？

跑完用 `--cov-report=term-missing` 看**具体行号**是否从 Miss→Hit，不能只看文件级覆盖率涨了。如果新增 Miss→Hit 行仅在 import / class 定义 / 装饰器行 → 无效覆盖。必须在方法体内部（def 和 return 之间）。

```
跑前:  python run_npu_ut.py --ignore=tests/pytorch/test_<新>.py 2>&1 | grep <目标文件>
跑后:  python run_npu_ut.py 2>&1 | grep <目标文件>
差值 = 新测试带来的行命中
```

### L1：断言的内容有意义吗？

**评分按断言语义，不按断言形式。** grep 命中模式但"语义限定"列不满足时，按降级后的层级计分。

| 层级 | 匹配模式 | 分值 | 语义限定（不满足则降级） |
|------|---------|------|------------------------|
| L1.1 结构 | `assert shape/dtype/isinstance/not None` | 0 | 必要但不充分。`grad is not None`、state 的 dtype 检查、`assert out is not None` 均归此类 |
| L1.2 数值 | `assert_close`, `testing.assert_close`, `.allclose(` | +2 | **仅当 expected 独立构造才计分**：fp64 CPU 参考、手算公式、PyTorch native 对照（如 `F.linear`）。actual 与 expected 同源（同一对象 / 同一表达式 / 被测函数自身输出）→ 0 分 |
| L1.3 状态 | `.grad` 数值校验、`state[...]` 数值校验、可观察全局副作用 | +2 | 必须校验**数值或语义效果**（如 `assert_close(grad, ref_grad)`、`base._ub_communicators is None`）；`grad is not None` 降级为 L1.1（0 分） |
| L1.4 边界 | `pytest.raises(..., match=...)`, zero/NaN 的 `.item()` | +1 | raises 必须带 `match=`，且匹配串包含**具体名称或数值**（能抓住消息文案错误，如漏 f-string 前缀）；无 match 或只匹配 `"error"` 类泛词 → 0 分 |
| L1.5 反向 | 扰动输入验证输出变化；非法输入验证 raise | +1 | 每文件至少 1 条，见下方量化指标 |

**每 test case 总分 ≥ 2 → 合格。** < 2 分 → 补断言或合并到相邻 test。

### L2：是直接调用还是间接触发？

test 体内是否出现目标类/方法名？（grep 即可）。直接调用 ✅；间接命中（如 `LayerNormLinear` 测试触发了 `RMSNorm` 的行）→ ⚠️ 需备注，且该 case 不得作为目标行的主要覆盖证据。

### L3：不同参数走了不同路径吗？

参数化必须触发源码中**不同的 if/else 分支**。dtype × shape 排列（同路径反复跑）→ 冗余 → 精简为 2-3 个代表性组合。

---

## 文件级量化指标（硬指标，逐文件核对）

| 指标 | 下限/上限 |
|------|----------|
| L1.2 / L1.3 数值·语义断言 case 占比 | **≥ 50%** |
| smoke case（仅存在性 / "不报错"） | **≤ 2 条**，且必须注释标记 `# smoke:` |
| 纯 import 测试 | **0 条** — `import transformer_engine` 的可用性由包 `__init__` 自然验证，不单独成测（PR #101 评审意见：无用测试增加整体执行时间） |
| `pytest.raises`-only case 占比 | **≤ 30%**，不能用异常路径替代正常功能覆盖 |
| 反向 / 扰动断言 | **≥ 1 条**：改动输入则输出必须变化，或非法输入必须 raise |
| round-trip 类测试 | load 目标 ≠ save 源头（新实例，或先扰动再恢复） |

---

## 断言风格 7 条

> 优先 NPU 本地风格，不要照搬上游 Transformer Engine 写法。

| # | 场景 | NPU 写法 |
|---|------|---------|
| 1 | 数值比较 | `assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(dtype))` — 不用裸 `torch.testing.assert_close`，会 device mismatch |
| 2 | 结构 | 原生 `assert out.shape == (M, N)` + 消息字符串 |
| 3 | 梯度 | `assert_close(to_cpu(inp.grad), to_cpu(ref_grad), **dtype_tols(dtype))`。`grad is not None` 只是 L1.1，不能单独成 case |
| 4 | 状态 | `assert opt.state[p]["exp_avg"]` 的**数值**；只查 dtype 属 L1.1 |
| 5 | 异常 | `pytest.raises(RuntimeError, match=r"具体类名或数值")` — match 要能抓住消息写错（如漏 f-string 前缀、数值拼错） |
| 6 | 自定义 helper | 同一 assert 模式出现 ≥3 次时提取为 `_assert_*` 函数 |
| 7 | 去重 | 同构 case 重复 ≥3 次（如每个 Test* 类各有一份 `test_forward_shape` / `test_backward_shape`）必须用 `@pytest.mark.parametrize` 合并；helper / 常量已在 `utils.py` 或被测模块（如 `npu_activation.py`）定义时直接 import，**禁止在测试文件重复定义** |
| 8 | 导入位置 | 被测对象和稳定 helper 在文件顶部导入；不要在每个 test 函数内重复 import。只有验证延迟导入、可选依赖缺失或 import 错误路径时，才允许函数内 import 并写明原因 |

**NPU 独有约束**：
- 随机张量 `torch.randn(device="cpu").to("npu")` — 避 NPU 随机数 kernel
- `torch.npu.synchronize()` — 所有 `step()` / `matmul` 后必须 sync
- `FP8GlobalStateManager.reset()` — 涉及 FP8 状态时在 `setup_method` 中 reset
- import 与模块级 helper 放在文件顶部，不要在测试函数体内重复 import / 定义
- dispatch table / enum / 常量测试不要把被测模块中已有的完整集合再复制一份；优先验证关系（如 fwd/bwd key 一致、子集关系、key 映射到预期函数对象）或直接删除低价值断言

**上游 Transformer Engine → NPU 替换**：
- `torch.testing.assert_close(a, b)` → `assert_close(to_cpu(a), to_cpu(b), **dtype_tols(d))`
- `tensor.is_cuda` → 不变（NPU 上返回 True）
- `torch.randn(device="cuda")` → `torch.randn(device="cpu").to("npu")`

---

## 恒真断言禁令（一律 0 分，必须重写）

以下模式断言**永真**，对回归没有任何保护能力，检出即驳回：

- `assert True`、`assert x == x`、actual 与 expected 引用**同一对象**（应改为 `assert out is p.main_grad` 这类身份断言，或换成独立参考值）
- state_dict / pickle **存回同一模块或空张量**再比较 —— 权重物理上不可能变化。正确写法：load 进第二个新实例，或先扰动权重再验证恢复；真正需要 round-trip 的是非空状态
- **无 RNG 算子重 seed 后断言两次输出相等** —— matmul/Linear 不消耗随机状态，必然相等。确定性验证应固定输入后对参考实现（如 `F.linear`）断言数值
- 断言测试自己刚设置的标记（如先 `w.flag = True` 再 `assert w.flag is True`）—— 被测代码是否置位无法区分。正确写法：用未置标记的对象验证函数**主动置位**
- 枚举 membership 断言（`assert v in (全部枚举值)`）—— 列出全部取值即恒真。要么断言具体机型的动态期望值，要么删除
- 对字面 `pass` 的 stub 函数刷"不报错"用例 —— 零信号。stub 只保留 1 条带可观察副作用断言的存在性 case（如 `destroy_ub` 后断言全局态为 None），不伪装成行为测试
- 测试子类重写某 property/方法后再断言该 property 的值 —— 测的是夹具自身。验证基类默认行为时，用不重写的最小子类

---

## 常见错误 Checklist

生成或接收 UT 前逐项过一遍。命中任一项就先修，不要把问题留给远端全量回归。

- [ ] **API 签名未确认**：凭上游 Transformer Engine、旧分支或记忆写调用，常见症状是 `TypeError: unexpected keyword argument`。写调用前先看当前 `TransformerEngineNPU/transformer_engine/...` 的实际 `def`。
- [ ] **假设了目标类未声明的属性或输出契约**：不要因构造参数叫 `normalized_shape` 就断言实例一定保存 `.normalized_shape`，也不要因继承 `torch.nn.Module` 就假设 `extra_repr()` 会输出 shape/配置。逐项确认属性由当前类/父类赋值或注册、方法由目标类覆写且文档声明了输出；否则改为真实功能断言或删除无效 case。
- [ ] **本地方法名 ≠ 远程方法名**：本地 review 通过但远端 `AttributeError`。Step 3.5 必须 grep 远端/当前运行工作区的签名和分支。
- [ ] **远端没有运行本地 HEAD**：pytest 前比较本地、fork 远端和 NPU 工作区 `git rev-parse HEAD`；任一不一致先同步，禁止用旧日志判断新代码。
- [ ] **测试子目录没有 package 边界**：迁移 UT 后添加 `__init__.py`，检查父子目录重复 basename；不要用子目录 `conftest.py` 注入 `sys.path`，否则可能重复导入测试或带注册副作用的依赖。
- [ ] **把 collection 连锁错误当成多个代码问题**：先看第一个完整 traceback。`libhccl.so` 缺失会让 `import torch` 半初始化，后续重试可能连锁报 `TORCH_LIBRARY`/`triton` 重复注册；加载正确 CANN 环境后用干净 Python 进程重试。
- [ ] **缺少仓库声明的测试依赖**：pytest 前预检 `torch_npu`，按目标预检 `triton`、`nvdlfw_inspect`；A5 全量口径下应安装依赖，不得通过 ignore/永久 skip 隐藏对应测试。
- [ ] **裸 `torch.testing.assert_close`**：可能触发 device mismatch。优先用 `assert_close(to_cpu(actual), to_cpu(expected), **dtype_tols(dtype))`。
- [ ] **仅 shape/dtype/isinstance 断言**：只属于 L1.1，不足以独立成为有效 case。补 L1.2 数值比较、L1.3 状态变化或 L1.4 边界断言。
- [ ] **直接 `torch.randn(device="npu")`**：NPU 随机数 kernel 不可靠。用 CPU 生成再 `.to("npu")`。
- [ ] **异步路径忘记同步**：`step()`、matmul、collective 或 fused kernel 后结果不稳定。必要时加 `torch.npu.synchronize()`。
- [ ] **宽泛 `except Exception` / 宽泛 skip**：会吞掉真实 bug。只捕获明确的"不支持后端/特性"异常（`NotImplementedError`）；`RuntimeError`、`TypeError`、`NameError`、断言失败必须失败，不得 `except ... → pytest.skip`。
- [ ] **A2/A3 A 类测试当成有效覆盖**：`_A5 = False`、`_A5_READY = False` 或 A5-only skip 只是占位，报告时标记为 skipped coverage，不计入质量证据。
- [ ] **环境硬编码版本断言**：禁止 `check_npu_version(NPUVersion.A5) is False  # 本机是 A2` 这类写法（PR #101 评审意见）。用 `check_npu_version` / `get_npu_version` 动态对比，换机型不得误 fail。
- [ ] **placeholder 当预期锁定**：`get_cudnn_version() == (0,0,0)`、`get_device_compute_capability() == (9,0)` 这类把 stub 返回值硬编码为预期的断言禁止。改为断言返回类型，或 `pytest.xfail(strict=False)` 注明"待实现"。
- [ ] **xfail 掩盖公开 API 回归**（PR #101 P1 评审意见）：xfail 必须 `strict=True` 并注释根因与跟踪 issue；公开兼容 API（如 deprecated 参数路径）**不允许**用 xfail 接受失败 —— 先修源码，再移除 xfail 让用例正常通过。
- [ ] **重复代码未参数化**（PR #101 评审意见）：同构测试结构重复 ≥3 次必须 parametrize 合并；已存在的 helper / 常量（`utils.py`、被测模块中的定义）禁止在测试文件重复定义。
- [ ] **分布式测试未导入 `torch.distributed as dist`**：引用进程组时必须导入 `dist`，需要 group 的 API 传 `dist.group.WORLD` 或显式 process group。
- [ ] **分布式 rank/world_size 只信环境变量**：`LOCAL_RANK`/`WORLD_SIZE` 可用于设设备，但断言 wrapper 语义时应对齐 `dist.get_rank()` / `dist.get_world_size()`。
- [ ] **分布式算子没传真实进程组**：`AllGather()`、`AllReduce()`、`ReduceScatter()` 等默认 `process_group=None` 时可能 no-op；测试真实 collective 必须传 `dist.group.WORLD` 或显式 group。
- [ ] **collective 只测 shape**：能构造 rank-distinct 输入时，验证 gather/reduce 后的具体值，而不是只看 `.shape`。
- [ ] **HCCL 进程池复用导致第二个 case 失败**：同一 `DistributedTest` class 连续运行多个真实 collective，若第二个用例报 `hcclCommInitRootInfoConfig`，优先设置 `reuse_dist_env = False`；不要改成宽泛 skip 或删掉数值断言。
- [ ] **teardown 额外 barrier**：`pool.starmap()` 已等待所有 worker 返回，销毁进程组前再 `dist.barrier()` 可能在 HCCL teardown 报 `Tensor list mustn't be larger than the number of available NPUs`。
- [ ] **state_dict / round-trip 只比较 `state.keys()` 或同对象自比较**：至少比较 dtype、step、scale/state 值，且 load 目标 ≠ save 源头（见"恒真断言禁令"）。
- [ ] **纯 import / no-op 用例**：纯 import 测试禁止入库（PR #101 评审意见）；仅断言对象存在、"不报错"、`isinstance(result, bool)` 或 `len(sd) > 0` 的 case 删除，或合并为 ≤2 条带 `# smoke:` 标记的 smoke case，不得作为主要覆盖证据。
- [ ] **函数内重复 import**：被测对象和稳定 helper 应在文件顶部导入；每个 test 内重复 `from transformer_engine... import ...` 增加噪声且掩盖结构问题。只有延迟导入/可选依赖行为本身是测试目标时才例外。
- [ ] **重复定义被测模块常量**：不要在测试里复制 `ACTIVATION_FWD`、`GLU_VARIANTS` 等完整常量集合再比较，维护成本高且容易和实现同源。改为验证集合关系、映射到函数对象，或删除低价值断言。
- [ ] **NotImplementedError 测试滥用**：只适合明确未实现且有稳定错误契约的路径。不能用大量 raises 测试替代可运行功能路径覆盖。
- [ ] **容差自定义放宽**：默认且仅默认使用 `dtype_tols(dtype)`；自定义 rtol/atol 必须在自检报告中给出来源和理由。
- [ ] **coverage omit 只放 `[run]`**：已有 `.coverage` 数据重报时 `[run] omit` 不生效；A2 口径必须在 `[run]` 和 `[report]` 同步配置 omit。
- [ ] **没有远端代表性试跑**：静态审查不能替代 NPU 硬件试跑。分布式测试改动必须远端跑对应文件。
- [ ] **提交前未运行仓库门禁**：对 staged/changed files 运行 pre-commit；EOF、Ruff、Pylint、Bandit、codespell、typos 任一失败都要修复并复跑。避免 `--all-files` 格式化无关存量文件，且不得把本地配置或生成的 `__pycache__` 带入提交。

---

## B 类 DistributedTest Checklist

生成或修改 B 类分布式 UT 时，除上面的通用 checklist 外，还必须满足这些约束。

- [ ] **真实 group（硬门禁，PR #101 评审意见）**：collective 用例传 `process_group=None` 一律 ❌ 驳回 —— `None` 会被实现静默降级为 no-op、直接返回输入，测试恒过但没覆盖 HCCL。必须传 `dist.group.WORLD` 或显式创建的 process group，并按 world_size 校验**数值和维度**。
- [ ] **期望值公式**：自检报告中必须写出期望值如何从 rank / world_size 推导（如 all-reduce 期望值 = Σ(rank+1) = world_size·(world_size+1)/2；all-gather 期望值 = 各 rank 常量按 rank 序拼接）。
- [ ] **rank 来源**：设备选择可用 `LOCAL_RANK`，语义断言用 `dist.get_rank()` / `dist.get_world_size()`，必要时和环境变量交叉校验。
- [ ] **rank-distinct 输入**：至少一个 forward case 使用 `rank + 1`、rank id 填充或分 rank arange，确保 all-gather/all-reduce/reduce-scatter 的值可判别。backward 用例的 grad_out 也要按 rank 区分（全 1 梯度的 all-gather 结果恒真）。
- [ ] **数值断言**：all-gather 断言拼接顺序；all-reduce 断言求和值；reduce-scatter 断言输出 shard 形状和值；只测 shape 不合格。
- [ ] **HCCL 复用策略**：同一 class 有多个真实 collective case 时，默认考虑 `reuse_dist_env = False`；若保留复用，必须远端连续跑完整 class。
- [ ] **端口和设备隔离**：远端运行指定空闲卡和独立端口段，例如 `ASCEND_RT_VISIBLE_DEVICES=4,5 HCCL_NPU_SOCKET_PORT_RANGE=22000-22100`。
- [ ] **失败分类**：`hcclCommInitRootInfoConfig` 多半是 HCCL 初始化/复用问题；AllGather 不翻倍多半是 group 为 `None`；teardown barrier 报错不是业务断言失败。

---

## 执行顺序

```
Step 4.5（上传前 — 静态）:
  L1 断言打分: grep assert_close / pytest.raises / \.grad → 按语义限定计分
  恒真断言扫描: 同一对象比较 / 同模块 round-trip / 自己置位自己断言
  文件级指标:   数值断言占比 / smoke 计数 / raises 占比 / 反向断言
  L2 调用链:   grep 目标类名 → 是否直接调用
  L3 分支覆盖: if/else 计数 → 参数化是否覆盖不同分支

Step 5.5（运行后 — 动态）:
  L0 行命中: --cov-report=term-missing 前后 Miss 行号对比
```

---

## 自检报告模板

每个 agent 生成 UT 后必须在输出中附带：

```markdown
## 质量自检

### L0 行命中
- 目标文件: xxx.py
- 新增覆盖行: 57-63, 103-109
- 全部位于方法体内部 ✅ / ⚠️ 部分 import 行 ×N

### L1 断言打分
- N/M ≥ 2 分 ✅
- K/M = 0-1 分 (test_xxx, test_yyy)，原因: ...
- 参考值来源（逐 case 一句话）: test_xxx → fp64 CPU ref / F.linear 对照 / 手算公式 ...
- 容差: 全部 dtype_tols ✅ / 自定义 (test_yyy: rtol=..., 理由 ...)

### 文件级指标
- L1.2/L1.3 数值·语义断言占比: N/M = xx% (≥50%)
- smoke case: K 条 (≤2，已标记 `# smoke:`)
- 纯 import 测试: 0 ✅
- raises-only 占比: K/M = xx% (≤30%)
- 反向/扰动断言: test_xxx ✅ (≥1 条)

### L2 调用链
- K/M 直接调用 ✅ / ⚠️ 间接命中 (test_yyy, ...)

### L3 分支覆盖
- xxx_flag: True/False → 2 分支 ✅
- dtype 排列 → 1 路径 ⚠️

### B 类专项（仅分布式测试）
- process group: 全部 dist.group.WORLD / 显式 group ✅
- 期望值公式: all-reduce 期望 = Σ(rank+1) = 3 (world_size=2) ...
- backward grad_out 按 rank 区分 ✅
```

**出现 ❌ → 先修正再提交。**
