# Patch 代码迭代原理：为什么每轮代码都不一样

> 来源：SKILL.md ⑦ PATCH 修复循环。解释多轮修复中代码持续变化的原因和演进方向。

## 核心原理

修复循环的 3 轮限制不是"同样的代码试 3 次"，而是**每轮重验都注入新信息**，推动代码朝更健壮的方向演进：

```
第 N 轮重验 PASS → 但发现设计缺陷 → 第 N+1 轮改进
第 N 轮重验 FAIL → 读取新日志 → 新错误说明修复方向对但不够 → 修复增强
                  → 旧错误重复说明修复方向错 → 调整策略
```

**代码变不是因为 LLM 随意改写，而是因为每轮重验后对问题的理解加深了一层。**

## 演进驱动力

### 驱动力 1：重验反馈

重验是最硬的信号。PASS 或 FAIL、新错误还是旧错误——直接决定下一步。

```
重验 PASS → 问题解决 → 但可能暴露设计缺陷（re-review 时发现）
重验 FAIL + 新错误 → 说明修复方向对但不够全面 → 补全边缘情况
重验 FAIL + 旧错误重复 → 说明修复方向根本错了 → 换策略
```

### 驱动力 2：跨轮理解深化

每轮都在积累对问题的理解。第一轮只看见崩溃点，第二轮看见设计原则，第三轮看见架构耦合：

```
Round 1: "崩溃了 → 需要 fallback"               （症状驱动）
Round 2: "这是上游 Bug 不是 NPU 问题"             （分类驱动）
Round 3: "不应该和上游内部变量名耦合"              （架构驱动）
```

## 典型案例：GTP SHARP 初始化 Bug 的三版演化

### 背景

Megatron-LM 合入 GTP 特性后，`get_data_parallel_group()` 新增了 `with_gtp_remat=True` 默认参数。`initialize_model_parallel()` 内的 SHARP barrier 在 GTP_REMAT group 别名赋值之前调用了此函数，访问尚未初始化的变量 → `AssertionError`。

### v1（Day 1）— 症状驱动：让代码跑通

```python
def get_data_parallel_group_wrapper(func):
    @wraps(func)
    def wrapper(...):
        if not hasattr(mpu, '_DATA_PARALLEL_GROUP_WITH_GTP_REMAT'):
            return func(..., with_gtp_remat=False, ...)  # fallback
        # ... 逐一检查 3 个 GTP group 变量
```

**阶段理解**：只知道崩溃了，需要降级到非 GTP 路径。

**问题**：
- 放入了 `MegatronBasicFeature`（始终生效），但这是上游 Bug，不该始终生效
- 前一个版本还有 `chr(95)` typo（三重下划线），`hasattr` 永远返回 False

### v2（Day 3 上午）— 分类驱动：修正架构位置

```python
def get_data_parallel_group_wrapper(func):
    """Wrapper that falls back to with_gtp_remat=False when GTP groups are
    not yet initialized. [详细背景说明]"""
    @wraps(func)
    def wrapper(...):
        if with_gtp_remat:
            from megatron.core import parallel_state as mpu
            if getattr(mpu, '_DATA_PARALLEL_GROUP_WITH_GTP_REMAT', None) is None:
                return func(..., with_gtp_remat=False, ...)  # fallback
        return func(...)
```

**阶段理解**：这是 Megatron-LM 源码的**通用时序 Bug**（GPU 也能复现），不是 NPU 专有问题。必须放 `MegatronFixFeature` + CLI flag，让社区修复后零残留。

**改进点**：
| 变化 | 动机 |
|------|------|
| `MegatronBasicFeature` → `MegatronFixFeature` + `--fix-gtp-sharp-init-ordering` | 分类正确性：上游 Bug 不应始终生效的 patch 掩盖 |
| `hasattr` → `getattr(..., None) is None` | 语义精确性：变量存在但未初始化 vs 变量不存在 |
| 简化检查：3 个变量 → 1 个代表性变量 | 冗余消除：GTP 变量同时赋值，检查一个即可 |
| 添加详尽 docstring | 可维护性：后续接手的人需要知道为什么这样改 |

### v3（Day 3 下午）— 架构驱动：消除耦合

```python
def get_data_parallel_group_gtp_fallback_wrapper(orig_fn):
    def wrapper(with_context_parallel=False, with_gtp_remat=True, partial_data_parallel=False):
        try:
            return orig_fn(...)       # 先尝试原函数
        except AssertionError:        # 失败了再降级
            if with_gtp_remat:
                return orig_fn(..., with_gtp_remat=False, ...)
            raise
```

**阶段理解**：v2 的预检查与 Megatron 内部变量名硬耦合（`mpu._DATA_PARALLEL_GROUP_WITH_GTP_REMAT`）。上游改了变量名 → patch 静默失效。应该用 EAFP 模式——不预设内部状态，直接调用，失败了再处理。

**改进点**：
| 变化 | 动机 |
|------|------|
| **策略翻转**：预先检查 → try-catch | 消除耦合：不依赖任何 Megatron 内部变量名 |
| 去掉 `from megatron.core import parallel_state` import | v2 在 wrapper 内 import 上游模块，与内部实现细节耦合 |
| 去掉 `functools.wraps` | 不需要保留元信息（wrapper 工厂模式） |
| 文件重命名 `gtp.py` → `gtp_remat_fallback.py` | 命名精确性：体现 fallback 策略，区分于其他 GTP 适配 |
| 函数重命名 `..._wrapper` → `..._gtp_fallback_wrapper` | 同上 |
| 56 行 → 46 行 | 逻辑简化：EAFP 比 LBYL 更短 |

### 三版对比总结

```
v1: if not hasattr(mpu, '___DATA...')     ← 防御式 + typo + 结构错误
v2: if getattr(mpu, '_DATA...', None)     ← 防御式 + 正确 + 耦合内部变量名
v3: try: orig_fn() except AssertionError  ← EAFP + 零耦合
```

**演化方向**：防御式 → EAFP、耦合 → 零耦合、模糊命名 → 精确命名、结构错误 → 结构正确。

## 每次迭代都改进什么维度

| 维度 | v1→v2 | v2→v3 |
|------|-------|-------|
| **架构位置** | ✅ MegatronBasicFeature → MegatronFixFeature | — |
| **分类正确性** | ✅ NPU 专用 → 跨平台 Bug | — |
| **与上游耦合度** | — | ✅ 硬编码变量名 → try-catch 零耦合 |
| **语义精确性** | ✅ hasattr → getattr | ✅ 防御式检查 → 行为驱动 |
| **命名精确性** | — | ✅ gtp.py → gtp_remat_fallback.py |
| **代码量** | 23 → 56 行（加文档） | 56 → 46 行（去耦合） |
| **可维护性** | ✅ 无注释 → 详细 docstring | ✅ 更清晰的模块文档 |

## 多轮修复的正确心态

1. **第一版不追求完美** — 先跑通，确认修复方向对。`git commit --amend` 给了安全网。
2. **每轮重验后做 code review** — 不只是看 PASS/FAIL，还要看设计原则：分类正确吗？耦合度合理吗？命名精确吗？
3. **第三轮追求"零耦合"** — 最健壮的 patch 是不依赖上游内部实现细节的 patch。上游怎么重构都不影响。
4. **3 轮上限不是限制，是保护** — 防止陷入无限打磨。3 轮后仍不满意 → 标记 NEEDS_HUMAN，而不是继续自动迭代。
