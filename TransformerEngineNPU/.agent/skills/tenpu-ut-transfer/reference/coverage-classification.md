# 覆盖率分类与目标选择

本文件只保存稳定方法，不保存某次运行的百分比、文件数量、硬件卡数或预计增量。仓库和运行环境会变化，所有数值必须从当前分支和本次 coverage 报告生成。

## 先固定口径

运行前记录：

- `git rev-parse HEAD` 与目标基线（通常为最新官仓 `main`）
- 目标硬件及能力检测结果（A2/A3/A5、FP8/HiF8/MXFP8/W4A8、Triton）
- pytest 收集数、passed/skipped/failed 数
- coverage `Stmts`、`Miss`、整数百分比和具体 Missing 行
- `.coveragerc` 的 source/omit 配置及命令行 `--ignore`

**A5 全量口径**：不得 omit A5、量化或 Triton 源码。缺少依赖应补齐环境，不得用 ignore/skip 缩小测试集冒充全量覆盖。

**A2/A3 能力口径**：只有用户明确要求时才允许 omit 硬件不可达源码；`[run]` 与 `[report]` 必须一致，并在结果中逐项披露。删除/跳过测试只改变命中，不改变源码分母。

## 动态分类

| 类别 | 识别信号 | 验证方式 |
|---|---|---|
| A：量化/特定硬件 | `npu_quantize`、FP8/HiF8/MXFP8/W4A8 tensor、Triton kernel | 当前源码能力检测 + 对应硬件真实执行；能力不足时 `skipif` |
| B：分布式 | `dist.*`、all-gather/all-reduce/reduce-scatter、FSDP/CP | `DistributedTest`、真实 group、rank-distinct 数值断言 |
| D：普通 NPU | 单卡模块、helper、错误路径、普通前反向 | 普通 pytest + 独立参考值 |

一个文件可同时包含多类路径。按目标 Miss 行分类，不按文件名一次性归类。

## 选择覆盖目标

1. 从本次 `term-missing` 或 XML/JSON 报告读取 `Stmts - Miss`，不要复用历史百分比。
2. 用 `ceil(Stmts * target) - (Stmts - Miss)` 计算达到目标还需命中的有效行数。
3. 阅读 Missing 行对应分支，剔除硬件不可达、纯防御或无稳定契约路径。
4. 优先选择能以独立数值参考覆盖多个连续分支的功能路径。
5. 对新增 case 前后比较具体 Missing 行；import、装饰器、类定义命中不计功能覆盖。
6. 覆盖率未变化时先确认 case 是否 collected/passed、coverage source 是否正确、自定义 autograd/子进程是否被记录，再考虑继续加测试。

## 禁止的捷径

- 不用删除 A5 测试、ignore 慢用例或增加 skip 来“缩小分母”；这些操作不会减少 `--cov=transformer_engine` 的源码语句数。
- 不把 skipped、xfail、import-only 或构造阶段失败的 case 算作覆盖贡献。
- 不根据显示的整数 `59%` 猜测差距；始终使用 `Stmts/Miss` 计算。
- 不把某次 A2/A5 环境的失败结论写成永久硬件规则；以当前源码能力检测和本次远端试跑为准。
