# 诊断常见陷阱

> 来源：SKILL.md ⑥ DIAG 章节，诊断时必须阅读以避免系统性误判。

## 陷阱 1：假设派生关系

**"这两段代码干同一件事，所以后来者是从前者 refactoring 出来的"**

不同文件中可能各自独立编写了功能相似的代码。例如：
- `training.py:get_model()` 中的 GPU allocation 条件：`not (A and B)`（正确）
- `dist_utils.py:unimodal_build_distributed_models()` 中的条件：`not A and not B`（错误）

容易错误地认为后者是从前者 refactoring 时引入的 De Morgan 违反。实际两者是**独立编写的两份代码，从未有过派生关系**，条件从 `dist_utils.py` 创建起就没变过。

**正确做法**：对每个可疑条件，用 `git log --all -S "<condition>" -- <file>` 追溯其**首次出现**的 commit，不要假设它是在 delta 区间内被改变的。

## 陷阱 2：搜索的错误特征

**"条件是否存在" vs "条件何时被激活"**

buggy 条件可能一直沉睡在某个函数的代码中（从未被执行），直到另一个 commit 改变了**代码路径**才被触发。
- 中间 commits 的代码级 grep 会显示"特征存在" → 只能找到条件被**写下**的 commit
- 实际 breaking commit 可能更晚：是**切换代码路径**的那个 commit

**正确做法**：从报错 traceback 出发，完整追踪当前执行路径。然后用 `git log -S` 找到控制这条路径的分支逻辑在哪个 commit 被修改，而不是搜索报错所在的代码行。

## 陷阱 3：假设"NPU 不兼容"而不验证代码执行顺序

当上游新增一个特性（如 GTP）引入了新的 process group 变量，且该变量在崩溃时值为 `None`，容易直接下结论"NPU 上不支持这个特性"。

**实际可能原因**（按优先级排查）：
1. **时序问题**：变量在函数内的赋值发生在调用点**之后** → 检查崩溃函数的完整执行流，标注每个变量的赋值时间点
2. **条件分支**：赋值被某个 `if` 条件跳过 → 检查条件是什么，为什么在 NPU 上走不到（可能是配置参数不同）
3. **测试脚本传参**：`--use-sharp` 等参数由测试脚本显式传入 → **必须 grep 测试脚本的参数**，不能只看 Megatron-LM 源码的默认值

**案例（2026-07-29 GTP 崩溃 → 2026-07-31 修复）**：
- 初步错误判断："GTP groups 在 NPU 上无法初始化"
- 实际根因：`initialize_model_parallel()` 中 SHARP barrier（调用 `get_data_parallel_group()`）排在 GTP group 别名赋值**之前**。`use_sharp=False` 默认值时 barrier 被跳过，但 `test_attention.sh` 显式传了 `--use-sharp` → barrier 被执行 → 访问尚未赋值的 GTP group → crash
- 教训：追踪完整执行流时，必须从**测试脚本参数**出发，确认每个分支条件在 NPU 环境下的实际值，不能依赖源码默认值
- **修复**（2026-07-31，commit `a203eb7`）：新增 `--fix-gtp-sharp-init-ordering` flag（MegatronFixFeature），wrapper 捕获 `AssertionError` 后用 `with_gtp_remat=False` 重试。修复文件：`patches/megatron/gtp_remat_fallback.py`

---

## 上游 Bug 修复合入策略：MegatronFixFeature vs MegatronBasicFeature

**核心原则**：区分"NPU 专有问题"和"上游源码 Bug（GPU/NPU 均受影响）"。

| 问题类型 | 注册位置 | 生效方式 | 示例 |
|----------|----------|----------|------|
| NPU 不支持 CUDA API | `MegatronBasicFeature` | **始终生效** | `torch.cuda.get_device_capability` → stub |
| NPU 算子等价替换 | `MegatronBasicFeature` | **始终生效** | `FusedScaleMaskSoftmax` → NPU 实现 |
| 上游源码 Bug（GPU/NPU 均受影响） | `MegatronFixFeature` | **CLI flag 控制** | GTP SHARP ordering bug (`--fix-gtp-sharp-init-ordering`) |
| 上游逻辑错误（De Morgan 等） | `MegatronFixFeature` | **CLI flag 控制** | CPU initialization bug (`--fix-use-cpu-initialization`) |

**MegatronFixFeature 模式**（参考 commit `5d2afd7`）：

```
1. megatron_fix.py register_args() → 新增 --fix-<bug-name> flag（默认 False）
2. megatron_fix.py register_patches() → args.<flag> 为 True 时注册 patch
3. 测试脚本 → 添加该 flag
4. 社区修复后 → 删除 flag + 注册代码 + 测试脚本引用，零残留
```

**为什么不用 MegatronBasicFeature**：
- 上游 Bug 是跨平台的，社区有动力修复
- 始终生效的 patch 会掩盖上游修复，导致无法感知社区是否已修复
- CLI flag 使修复变为显式 opt-in，社区修复后直接删代码，不需要改 patch 逻辑

**判断标准**（满足任一即为上游 Bug → MegatronFixFeature）：
1. 同样的问题在 NVIDIA GPU + 等效条件下也能复现
2. 问题由 Megatron-LM 源码的逻辑错误（非 NPU 兼容性）引起
3. 新参数的默认值改变了已有调用点的语义（非 NPU 环境也会走到的代码路径）

**案例（2026-07-29 GTP SHARP ordering）**：
- 初步放入 `MegatronBasicFeature`（始终生效）
- 经分析确认：这是 Megatron-LM `initialize_model_parallel()` 内的执行时序 bug，`use_sharp=True` 时 GPU 也会崩溃
- **修正**：迁移到 `MegatronFixFeature`，新增 `--fix-gtp-sharp-init-ordering` flag

### 复发防护：跨天 statelessness 会打回 BasicFeature

**⚠️ 已修复的上游 Bug 可能在跨天 reset 后复发**。每晚 `git checkout -B daily_fix_<branch> origin/<branch>` 会丢弃尚未合入上游的 FixFeature 修复。

**案例（2026-08-01 ~ 08-03，GTP SHARP 复发三次）**：
- 7/31 commit `a203eb7` 是标准 MegatronFixFeature 修法（`megatron_fix.py` 加 `--fix-gtp-sharp-init-ordering` flag + `gtp_remat_fallback.py` + `test_attention.sh` 传参）
- 每晚 reset 把它整体丢弃 → 新会话把同一 bug **重新分类**为「CUDA-only / NPU 不支持」→ 落回 `MegatronBasicFeature`（始终生效）→ 8/3 时 flag 与 fallback 已从代码库消失
- 复发根因：**陷阱 3**（假设 NPU 不兼容）+ 修法惰性（FixFeature 需改 3 文件 + 测试脚本传参，BasicFeature 只改 2 文件无条件生效 → auto-fix 循环天然偏向省事路径，验证通过后无机制逼它 re-review 架构）

**三条防护规则**：
1. **已知上游 bug 再次出现在 nightly 失败里时，先查 `git reflog` 同一 `[test_name]` 的最近修复 commit**——若历史上有正确的 FixFeature 版本（如 `a203eb7`），直接复用，不要从零重新分类
2. **断言「某修复当前在 BasicFeature/FixFeature」前，先 grep 代码核实现状，不要凭记忆**——跨天 reset 可能已把它换位
3. **合入主仓前防回退**：FixFeature 修复若被人工 PR 检视驳回/重置，注意勿让后续 nightly 再把它打回 BasicFeature

---

## 陷阱 4：pre-commit 校验时序 — "先跑检查再提交" vs "先提交再检查"

**错误流程**（2026-07-31 实际发生）：
```bash
# ❌ 先 commit 再跑 pre-commit
git add <文件>
git commit -m "fix: ..."          # ← 此时无 hook，裸奔
pre-commit run --files <文件>      # ← 自动修复工作区，但 commit 已是旧代码
git add -u && git commit --amend  # ← 容易忘记这步
```

三个致命缺陷：
1. **`pre-commit install` 未执行** → `git commit` 时 hook 不触发
2. **先 commit 后检查** → 即使 step ②.5 的 `git commit --amend` 没忘，也容易在 `--all-files` 先跑过后被误导（工作区已修复 → `pre-commit run --files` 显示 PASS → 以为不用 amend）
3. **多 commit 节点** → 同一次修复拆分成了"实际修复"+"格式化修复"两个 commit，违反"每分支仅一个干净 commit"规范

**正确流程**：
```bash
# 0. 确保 hook 已安装（首次）
pre-commit install

# 1. 修改文件后，先跑 pre-commit（自动修复工作区）
git add <文件>
pre-commit run --files <文件>

# 2. 如果 pre-commit 修改了文件，重新 add
git diff --stat --exit-code || git add -u

# 3. 提交（hook 会自动再跑一次，但文件已修复 → 全部 PASS）
git -c user.name="..." -c user.email="..." commit -m "fix(agent): ..."
```

**核心原则**：`pre-commit run --files` **必须在 `git commit` 之前**执行。pre-commit 修改的是工作区文件，不是已提交的版本。先 commit 再跑 pre-commit → commit 里永远是修复前的代码。

**案例（2026-07-31）**：
- 诊断阶段跑了 `pre-commit run --all-files`，ruff-format 自动格式化了 6 个文件（包括 2 个修复文件）
- 提交阶段跑 `pre-commit run --files` 时，工作区已被前一步修复 → 显示 PASS
- 但 commit 内容是修复前的版本 → 流水线 pre-commit 报错
- 修复后又多了一个"格式化 fix"commit → 需 squash 回一个 commit

---

## 陷阱 5：修复归属默认落到 MA —— 先对标社区 TE 参考实现

**"TE 相关的 bug 就 monkey-patch TE（在 MA 里写 wrapper）"**

TE-NPU 是社区 TransformerEngine 的昇腾移植版，**接口契约应以社区 TE 为基准**。当问题来自 TE-NPU 时，不能默认放 MA `patches/`，要先问：**社区 TE 在问题点的行为是什么？**

**判断流程**：
1. 拉社区 TE 对应源文件（`https://raw.githubusercontent.com/NVIDIA/TransformerEngine/<tag>/transformer_engine/pytorch/...`）
2. 比对 TE-NPU 与社区在问题点的实现差异
3. 分类：
   - TE-NPU **偏离社区契约**且偏离可避免（runtime 实现泄漏成公开接口，如 `group["step"]` 从 int 变 tensor 泄漏进 `state_dict`）→ **改 TE-NPU 仓**，恢复社区契约；MA wrapper 只做 TE-NPU 不可控时的临时 shim
   - TE-NPU **忠实社区**、只是新增了 MA 需要适配的 API → 才是 MA `patches/` 适配
4. **验证充分性**：TE-NPU 修复要能在 **MA 无 wrapper** 时让用例通过，才证明归属正确（否则 wrapper 在掩盖问题）

**案例（2026-08-17，FusedAdam step 序列化）**：MA 已写好 `fused_adam_step.py` wrapper（tensor→int 序列化），表面上修复了，但属归属错位——社区 TE 默认就是 int，TE-NPU 改成 tensor 是契约偏离，正确修复是 TE-NPU `state_dict()` 里 9 行转换。验证证明 MA 无 wrapper 时 TE-NPU 修复已足够 → wrapper 可撤。详见 `references/case_library.md` 案例 3。
