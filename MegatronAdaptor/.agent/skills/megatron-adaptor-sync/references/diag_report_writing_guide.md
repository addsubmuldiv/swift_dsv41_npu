# 诊断修复报告写作规范（DIAGNOSIS_AND_FIX.md）

> 来源：SKILL.md ⑧.5 章节，写完 DIAGNOSIS_AND_FIX.md 时必须逐条对照检查。

**语言要求**：报告正文一律使用**中文**撰写（标题、正文、表格叙述、结论）。代码、`git diff`、日志摘录等原始内容保留原文，不翻译。

**目标**：报告是诊断全过程的正式记录，面向需要回溯问题或接手修复的工程师。必须做到**一看就懂、可复现、可审计**。

## 报告结构（必须完整）

```markdown
# MegatronAdaptor 夜间验证 — <date> 诊断与修复报告

## 概览（一张表 + 关键数字）
## 一、<失败分支1> — <失败用例>
### 1.1 失败现象（摘录实际错误日志，不是描述）
### 1.2 Megatron-LM 上游变更（精确 diff，不是文字描述）
### 1.3 根因分析（执行时序 + 触发条件 + 崩溃链路）
### 1.4 MegatronAdaptor 修复（最终正确代码，不提中间失败尝试）
### 1.5 验证结果（摘录实际训练日志，不只是"PASS"）
### 1.6 修复提交（表格：仓库/分支/commit/消息/变更统计）
## 二、<通过分支> — 简述
## 三、<失败分支2> — 同上结构，或简化
## 四、总结
```

## 12 条硬性写作要求

### ① 失败现象必须摘录实际日志

**禁止**：只写 "AssertionError: xxx is not initialized" 这种一句话描述。
**必须**：从 `logs/<test>/stderr.log` 中提取关键错误栈（去噪，去掉 warning/deprecation，保留真实 traceback），标注文件:行号。

示例：
````markdown
### 1.1 失败现象
```
[rank0] File ".../megatron/core/parallel_state.py", line 1032, in initialize_model_parallel
[rank0]     group=get_data_parallel_group(with_context_parallel=True),
[rank0]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]   File ".../megatron/core/parallel_state.py", line 1806, in get_data_parallel_group
[rank0]     assert group is not None, f"{description} is not initialized"
[rank0] AssertionError: data parallel group with CP (with GTP_remat) is not initialized
```

**关键特征**: 错误发生在训练启动阶段（非训练过程中），全部 8 个 rank 同位置崩溃。
````

### ② 根因定位过程写成诊断步骤表

每一步是"做了什么操作 → 得到了什么结论"，让读者能复盘诊断逻辑。

```markdown
| 步骤 | 操作 | 发现 |
|------|------|------|
| 1 | 查看错误栈 | 崩溃点在 parallel_state.py:1806 → hasattr 返回 False |
| 2 | 对比昨日基线 | 昨日全通过，今日 MLM 更新到 bc67abde1 |
| 3 | 分析 delta | 新引入 GTP 特性 → with_context_parallel=True 调用 |
| 4 | 检查 Adaptor patch | gtp.py wrapper 拦截了 get_data_parallel_group |
| 5 | 手工 trace wrapper | chr(95)*3 拼出三重下划线，hasattr 永远 False |
```

### ③ 上游变更必须展示精确 diff + 执行时序

**禁止**：用文字描述上游改了什么（"新增了一个参数"）。文字可以被误解，diff 不会。

**必须**：
1. **贴 `git show <breaking_commit> -- <file>` 的关键 diff 片段**，标注具体行号
2. **完整追踪崩溃函数内的执行顺序**，用 A → B → C 标记各阶段，标注每个变量的赋值时间点
3. **回答两个问题**：
   - **触发条件**：什么参数/配置/环境变量让这行代码被执行？
   - **为什么默认情况不触发**：NVIDIA GPU 标准配置下为什么没这个问题？

### ④ 调用链必须用树形图 + 每步注释

不是 flat 文本列表，而是有缩进层次的树形结构，每个关键节点标注**为什么走到这里**。

````markdown
```
pretrain_gpt.py:524 pretrain()
  └─ training.py:1091 initialize_megatron()
       └─ initialize.py:149 finish_mpu_init()
            └─ initialize.py:370 _initialize_distributed()
                 └─ parallel_state.py:1032 get_data_parallel_group(with_context_parallel=True) ← GTP 新增
```
````

### ⑤ Bug 必须有 before/after 代码对比 + 完整 diff

**禁止**：一句话 "修复了 chr typo"。读者看不到代码，无法判断修复是否正确。
**必须**：贴出修复前后的关键代码段 + `diff -u` 格式完整 diff。

**额外要求**：显式列出修复了几个问题（往往不止一个）。

### ⑥ 必须有 Bug 时间线

回答: 这个 bug 是什么时候引入的？什么时候才暴露？为什么中间没发现？

```
7/27  commit 3456713 创建 gtp.py → chr typo 从第一天就存在
      └─ 当时 MLM 未触发 with_context_parallel=True 路径 → 沉睡 ✅
7/29  MLM bc67abde1 拉取 → GTP 正式落地 → 新代码路径激活 wrapper
      └─ hasattr 永远 False → fallback 被触发 → AssertionError ❌
```

### ⑦ "沉睡 Bug 激活"模式必须显式说明

当 breaking commit **不是** buggy 代码的作者时，这个模式必须点名：

> **关键洞察**: 这个 bug 不是今天新引入的。代码 bug 从 N 天前就存在，直到今天上游更新改变了代码路径才暴露。Breaking commit 不改 buggy 代码本身，只改变代码路径路由从而激活沉睡 bug。

### ⑧ 环境类失败必须列出完整受影响节点

如 HCCL 端口冲突，要列出每个 rank 的 IP + 端口 + 错误码（不仅仅是"端口被占用"）。

```markdown
| Rank | Device | IP | 端口 | 错误 |
|------|--------|----|------|------|
| 0 | 0 | 192.168.2.199 | 16666 | Communication_Error_Bind_IP_Port |
| 1 | 1 | 192.168.2.198 | 16666 | 同上 |
```

### ⑨ 排除代码问题必须用结构化检查

不能只说"看起来是环境问题"。要用表格证明：

```markdown
| 检查项 | 结果 |
|--------|------|
| 今日 MLM commit | 963bf39218e8 |
| 上次成功 commit | 963bf39218e8（相同） |
| 代码变更 delta | **0**（last_good == target） |
| 结论 | 非代码问题，环境/进程残留 |
```

### ⑩ 验证结果必须摘录实际训练输出

不是 "loss 正常收敛" 一句话，而是贴出实际训练日志片段。

并显式标注：✅ 0 错误、✅ Loss 收敛方向正确、✅ 吞吐量正常。

### ⑪ 必须有"模式总结与经验教训"章节

至少包含：
- **双失败关联分析**：如果多个分支同时失败，分析是否存在因果链
- **2-4 条可操作的经验教训**：每条格式为 `**教训标题**: 说明 → 建议`
- **长期改进建议**：代码层面或流程层面的改进

### ⑫ 修复追踪表收尾

```markdown
| 分支 | 问题 | 修复 commit | 推送分支 | 状态 |
|------|------|-------------|----------|------|
| main | chr() typo | 057267e | daily_fix_main | ✅ 已验证 |
| core_r0.17.0 | HCCL 端口冲突 | N/A | N/A | 🔧 NEEDS_HUMAN |
| core_r0.18.0 | — | — | — | ✅ PASS |
```

## 字数与详细程度底线

- 每个失败分支的诊断章节**不少于 500 字**（不含代码块）
- 代码 diff **必须是真实 diff**（从 git show/diff 摘录），不得手写伪代码
- **禁止出现**的写法：笼统描述（"修复了一个 typo"）、无日志的错误描述、无 diff 的修复方案
- **必须出现**的元素：实际日志摘录、git diff、时间线、诊断步骤表
