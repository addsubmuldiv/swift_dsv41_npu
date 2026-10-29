# 审查编排：Subagent 嵌入点与信息流

## 管道中的角色拆分

主 agent 生成代码 → 主 agent 机械自检 → **subagent 独立复核** → 主 agent 修正 → 上传运行。

主 agent 不给自己做最终判定。subagent 只看三份材料，不与主 agent 共享推理过程。

## 嵌入点

```
Step 4: 代码生成（主 agent）
  │
Step 4.5: 机械自检（主 agent）  ← grep 打分 + 恒真断言扫描，不做"通过/不通过"判定
  │  输出: 自检报告（结构化文本，含每条 case 的断言类型 + 参考值来源 + 文件级指标 + L0 miss 行列表）
  │
Step 4.6: 独立复核（review subagent）  ← 新增
  │  输入: 源文件 + 测试文件 + 自检报告
  │  输出: 复核意见（agree/disagree + 遗漏项清单 + 最终判定）
  │
  ├── 判定=通过 → Step 5 上传运行
  └── 判定=驳回 → 主 agent 修正 → 回到 Step 4.5 重新自检 → 再次复核
```

## Step 4.6 详细规格

### 触发条件

以下条件**全部满足**时可跳过 subagent（小文件 + 少 case + 全绿，主 agent 自检即可）：

- 目标源文件 < 100 stmts
- 新增 test case ≤ 10 条
- 自检报告 L1 全部 ≥ 2 分（按语义限定计分，非 grep 形式）
- 文件内 smoke / 纯结构 case ≤ 2 条，且含 ≥1 条反向/扰动断言

**任一条件不满足 → 必须启动 subagent。** 自报满分不足以豁免。

### Subagent 输入（由主 agent 在调用时打包）

主 agent 调用 Agent tool 时，在 prompt 中直接内联以下**四份**材料，不传文件路径（subagent 没有文件系统上下文）：

```
【材料1: 被测源文件】
<源文件完整内容 copy-paste>

【材料2: 新生成的测试文件】
<测试文件完整内容 copy-paste>

【材料3: 主 agent 自检报告（结构化）】
L0: 行命中 — 源文件 N stmts, M missed (行号: X-Y, ...)。每行 miss 原因。
L1: 断言打分 — N/M ≥2 分。0 分 case 列表 + 各自的断言类型。
    参考值来源 — 逐 case 一句话（fp64 CPU ref / 手算公式 / PyTorch native 对照）。
    容差 — 全部 dtype_tols 或自定义值+理由。
文件级指标 — 数值断言占比 N/M=xx% (≥50%)；smoke K 条 (≤2)；raises-only xx% (≤30%)；反向断言 test_xxx (≥1)。
L2: 调用链 — N/M 直接调用。间接命中列表（如有）。
L3: 分支覆盖 — 源文件 if/else 分支总数 N。参数化覆盖 K 个。

【材料4: 打分规则 + NPU 约定】
- L1.2 = assert_close / allclose 且 expected 独立构造（fp64 CPU ref / 手算公式 / PyTorch native 对照）→ +2；
  actual 与 expected 同源（同一对象 / 同一表达式 / 被测函数自身输出）→ 0 分并驳回
- L1.3 = .grad / state[...] 的数值校验或可观察全局副作用 → +2；
  `grad is not None`、仅 dtype 检查 → 0 分（L1.1）
- L1.4 = pytest.raises 带 match= 且匹配串含具体名称/数值 → +1；无 match 或泛词 → 0 分
- L1.5 = 反向/扰动断言（扰动输入验证输出变化、非法输入验证 raise）→ +1
- L1.1 = shape/dtype/isinstance/not None → 0。仅 L1.1 不能独立成为 test case
- 恒真断言一律 0 分并驳回：同对象自比较、同模块 round-trip 自证、无 RNG 重 seed 自证、
  自己置位自己断言、枚举全列 membership、stub（字面 pass）空测、夹具自证
- 纯 import 测试 → 直接驳回
- NPU 合法写法: assert_close(to_cpu(x), to_cpu(y), **dtype_tols(d)) — 不是错误
- 能力 `skipif` — 能力不足环境的预期行为，不算失败，也不计有效覆盖；A5 具备能力时必须执行
- xfail — 仅用于有根因和跟踪 issue 的已知软件缺陷，必须 `strict=True`，不得掩盖公开兼容 API 回归
- torch.randn(device="cpu").to("npu") — 避 NPU 随机数 kernel，正确做法
- torch.npu.synchronize() — 异步操作后必须调用，不是冗余代码
```

材料4 就是 `reference/assertion-guide.md` 的核心规则精简版——不必整份文档内联进去，但语义化评分规则、恒真禁令和 4 条 NPU 约定必须有。

### Subagent 处理逻辑（写在 prompt 中）

```
你是 UT 质量审查员。基于四份材料做判断，不假设任何材料之外的信息。
评分规则以【材料4】为准。

执行步骤:
1. L0: 逐行读源文件 → 确认自检 Miss 行号与源码一致 → 确认每行 miss 原因合理
   → 检查是否有 Miss 行被遗漏（不在自检列表但确实未覆盖）
2. L1: 复核全部 0 分 case 和全部 smoke 标记 case（最弱的 case 必被复核），
   再随机补足至 ≥5 条（其中至少 2 条自检声称 ≥2 分的）
   → 用材料4的规则逐条独立打分（按语义限定，不按形式）
   → 对比你的分和自检的分是否一致
   → 检查每个 assert_close 的 expected 是否独立构造；actual/expected 同源直接驳回
   → 扫描恒真断言禁令清单（同对象比较、同模块 round-trip、stub 空测等），命中直接驳回
   → 核对文件级指标是否达标（数值断言占比 / smoke 数 / raises 占比 / 反向断言）
   → 注意: 材料4中列出的 NPU 合法写法（如 assert_close(to_cpu(...))）不应扣分
   → 注意: xfail/skip 标记的 case 不计入 L1 统计；xfail 还必须有 strict=True、根因和跟踪 issue
3. L2: grep 每个 test case → 目标类/方法名是否存在 → 有没有自检标直接但实际间接的
4. L3: 读源文件的 if/else/for/while → 列分支总数 → 统计参数化覆盖了几个
```

### Subagent 输出格式（强制）

```
L0: agree / disagree
  - 差异: [具体行号 + 原因]
L1: agree / disagree
  - 抽查 case 结果:
    - test_xxx: 主检分=X, 复核分=Y, 一致/不一致
    - ...
  - 恒真断言命中: [case 名 + 模式]（如有）
  - oracle 同源命中: [case 名]（如有）
  - 文件级指标: 达标 / 不达标（具体数值）
L2: agree / disagree
  - 差异: [具体 case 名 + 原因]
L3: agree / disagree
  - 差异: [未覆盖分支列表]
遗漏: [未被覆盖但可触达的路径，每行一个]
结论: 通过 / 驳回修正
```

如果结论=驳回，必须列出 **修正清单**（每项一行，主 agent 逐条执行）。

### 主 agent 收到复核意见后的处理

```
If 结论=通过:
  → 进入 Step 5（上传运行）

If 结论=驳回:
  → 按修正清单逐条修改测试代码
  → 重新跑 Step 4.5 自检（只更新自检报告的分数和 case 列表）
  → 再次启动 subagent 复核
  → 最多重试 2 轮。2 轮后仍驳回 → 标记为"需人工介入"，输出当前状态和争议点。
```

## 完整信息流示意

```
主 agent 写测试
  │
  ├→ 机械 grep 打分 + 恒真扫描 (Step 4.5)
  │   输出: 自检报告（纯事实: N条case, K条含assert_close, 参考值来源, 文件级指标, M条仅有shape check...）
  │
  ├→ 打包四份材料 → Agent(subagent, prompt=审查指令+四材料) (Step 4.6)
  │    │
  │    └→ 复核意见: agree/disagree + 遗漏 + 修正清单
  │
  ├→ [disagree] 修正 → 回到 Step 4.5
  │
  └→ [agree] Step 5 上传运行
       │
       └→ --cov term-missing → L0 验证
            │
            ├→ Miss 行变化 ≠ 自检报告预期 → 异常，报告用户
            └→ Miss 行变化 = 自检报告预期 → 完成
```

## 自检报告与 subagent 输入的衔接约束

| 自检报告中的字段 | subagent 如何消费 | 如果缺失会怎样 |
|-----------------|------------------|---------------|
| Miss 行号 + 原因 | 逐行读源码验证 | subagent 需自己从头分析，可能漏判 |
| 每条 case 的断言类型计数 | 抽样时对比自己的打分 | subagent 需逐条打分，耗 token |
| 参考值来源 + 容差 | 验证 oracle 独立性与容差合理性 | subagent 无法判断 assert_close 是否同源 |
| 文件级指标数值 | 核对占比下限/上限 | subagent 需自己统计，可能口径不一 |
| 源文件 if/else 分支列表 | 验证 L3 时直接对照 | subagent 需自己读源码列分支 |
| 材料4 的打分规则 | 作为独立打分的唯一依据 | subagent 用自己训练数据中的惯例，可能与 NPU 约定冲突 |

材料4 虽不是自检报告的一部分，但**必须与自检报告一起打包发给 subagent**。主 agent 在 Step 4.6 调用时负责从 `reference/assertion-guide.md` 中提取语义化评分规则、恒真禁令和 NPU 约定，内联到 prompt 中。
