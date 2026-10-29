# Git Bisect 精确定位

> 来源：SKILL.md ⑤.5 章节。触发条件：delta > 10 commits，或 last_good == target，或诊断未能定位到确切 commit。

## 触发条件（满足任一即触发）

- delta commit 数量 > 10
- `last_good_commit == target_commit`（问题在上一次 OK 之前就已引入）
- 上一轮 DIAG 未能产出高置信度的 `breaking_mrs`

## 前置准备（必须先完成，再启动 bisect）

```
0. 构造最小复现用例
   ├── 从失败日志定位触发条件（具体参数组合、子测试编号）
   ├── 编写自包含 bisect_run.sh：注入 adaptor → 设置环境变量 → 跑最小用例 → exit 0=PASS / 1=FAIL
   └── 确保脚本在自己的修改后清理工作区（git checkout -- pretrain_gpt.py 等），否则 bisect 无法 checkout 下一步

0.5 环境初始化顺序（关键）
   ├── ⚠️ 先 source setup_env.sh（CANN 环境），再 import torch_npu / pip install
   └── CANN 未初始化时 pip install TE-NPU 会因 torch_npu 加载失败而报错

1. 验证边界 commit（实际跑测试，不是 grep）
   ├── 在 flywheel 记录的 last_good_commit 上跑 bisect_run.sh → 确认 PASS
   ├── 在 target_commit（或已知失败 commit）上跑 bisect_run.sh → 确认 FAIL
   └── 确认两边报的是同一个错误类型（避免 bisect 过程中因环境问题误判）
```

## Bisect 流程

```
2. 启动 bisect
   git bisect start
   git bisect bad <verified_bad_commit>
   git bisect good <verified_good_commit>

3. 自动二分 — git bisect run bash bisect_run.sh
   注意：bisect_run.sh 退出前必须 git checkout -- pretrain_gpt.py 恢复修改
   超时设置：log₂(N) × 单步耗时 + buffer，通常 685 commits 约需 15-20 分钟
   建议先 dry-run 一步确认单步耗时，再设定 timeout

4. 收敛后强制验证（⚠️ 必须执行，不可跳过）
   ├── git show <result_commit> --stat  ← 确认变更范围
   ├── ⚠️ 检查结果是否为"误中"commit：
   │    如果 result_commit 只改了 CI yaml / docs / 无关注入 → 它是误中（bisect 收敛到了相邻的无害 commit）
   │    → 向前搜索真正引入问题的 commit：git log --oneline <result_commit>^..<result_commit>~5
   │    → 手动测试相邻 commit 找到真正的 breaking commit
   ├── ⚠️ 验证 result_commit^（parent）确实 PASS：
   │    git checkout <result_commit>^ && bash bisect_run.sh
   └── 验证通过后记录：
       ├── git bisect log → bisect_log.txt
       ├── git show <breaking_commit> → breaking_commit_diff.txt
       └── 产出 bisect_report.json
```

## 代码级二分（仅 NPU 不可用时作为最后手段）

**警告**：此方法有根本性缺陷，仅在 NPU 硬件不可用时作为最后手段使用。

**已知缺陷**：
1. **特征可能在 bisect 区间之前就已存在**：如果搜索的特征（如某个条件表达式）在 good commit 之前就已经在代码中，那么每个中点都会命中，二分收拢到区间第一个 commit——但那个 commit 根本没引入这个特征
2. **无法区分"代码被写下"和"代码被激活"**：buggy 条件可能一直沉睡在某函数中，直到另一个 commit 切换了代码路径才触发。代码级二分只能找到条件首次出现的 commit，找不到路径切换的 commit
3. **容易制造误中**：如果 bisect 收敛到一个只改了 CI 配置/docs 的 commit，很可能是前一个 commit 才是真正的 breaking change

**如果必须使用代码级二分，需要额外验证**：
```
1. 提取关键代码特征（注意：必须在 delta 区间内确实发生过变化）
2. 对每个中点 commit：先 git show <commit>^:file | grep 确认 parent 没有此特征
3. 收敛后必须 git show <breaking>^:file 和 git show <breaking>:file 对比
4. 如果 diff 中没有发现任何逻辑变更，说明是误中 → 向前搜索
```

## 结果输入

Bisect 结果直接喂入 ⑥ DIAG：不再需要扫描整个 delta 区间，只需分析单个 commit 的 diff，定位精度从"几十个 MR"提升到"一个 commit"。
