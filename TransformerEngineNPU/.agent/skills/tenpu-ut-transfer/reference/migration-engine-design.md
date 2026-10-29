# UT 迁移引擎设计

## 核心结论

**A/B/D类不需要三套流程。一套统一管道 + 分类驱动的差异分支即可。**

差异点只有三个，而且都集中在管道后半段：

---

## 统一管道（5 步）

```
                  ┌─ 硬件分类 ─┐
上游测试 ─→ 差距分析 ─→ 代码生成 ─→ 验证
                  └─ 结构选择 ─┘
```

### Step 1: 上游定位

**输入**：目标源文件路径（如 `module/rmsnorm.py`）
**动作**：在 `../TransformerEngine/tests/pytorch/` 中 grep 定位上游测试
**规则**：映射表

| NPU 源文件 | → | 上游测试文件（近似） |
|-----------|---|---------------------|
| `module/rmsnorm.py` | → | `test_numerics.py` (TestRMSNorm) |
| `tensor/float8_tensor.py` | → | `test_float8blockwisetensor.py` + `test_float8_current_scaling_exact.py` |
| `distributed.py` | → | `test_sanity.py` (distributed section) + `L1_pytorch_distributed_unittest/` |
| 通用规则 | → | 在 `../TransformerEngine/tests/pytorch/` 中 `grep -l "ClassName\|function_name" *.py` |

**这一步对所有类完全相同。**

### Step 2: 差距分析

**输入**：NPU 源文件 + 上游测试文件 + 已有的 NPU 测试文件
**动作**：
1. `grep "def \|class "` 源文件 → 方法清单
2. `grep` 上游测试文件 → 哪些方法已有测试
3. `grep` NPU 已有测试 → 哪些已被移植
4. 差集 = 需要生成的测试

**输出**：待覆盖方法列表（按优先级：核心 API > 辅助方法 > deprecated 路径）

**这一步对所有类完全相同。** 仅有的差异是：A 类方法需要在输出中标注"FP8 依赖"标记。

### Step 3: 硬件分类（唯一的分叉点）

对 Step 2 输出的每个待覆盖方法，判定硬件依赖：

```
方法体内 grep 后：
  ├─ 调用 npu_quantize / npu_quant_matmul / Float8Tensor(...) → A 类标记
  ├─ 调用 dist.all_reduce / dist.all_gather / get_tensor_model_parallel_* → B 类标记
  ├─ 调用 torch_npu.npu_* 但非 FP8 → C 类标记（Triton kernel）
  └─ 均不依赖 → D 类标记（直接映射到普通 pytest）
```

**这一步是分类的核心，决定了后续的 skip 守卫和测试结构。**

### Step 4: 代码生成（按标记分叉）

统一的生成模板，用标记控制三个变量：

```python
# ═══════════════════════════════════════════════════════════
# 变量 1: 文件头
# ═══════════════════════════════════════════════════════════

# 所有文件统一：
HEADER = """# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.
\"\"\"{docstring}\"\"\"
"""

# 文件级 skip 按类标记：
SKIP_GUARD = {
    "A": 'pytestmark = pytest.mark.skipif(not check_npu_version(NPUVersion.A5), reason="Requires Atlas A5 for FP8 quantization")',
    "B": 'pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")',
    "D": 'pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")',
}

# A 类的 import 多一行：
A_IMPORT = "from transformer_engine.pytorch.constants import NPUVersion\nfrom transformer_engine.pytorch.utils import check_npu_version"

# ═══════════════════════════════════════════════════════════
# 变量 2: 测试结构
# ═══════════════════════════════════════════════════════════

# D 类/A 类：普通 pytest 函数
TEST_FUNC_TEMPLATE = """
class Test{ClassName}:
    @staticmethod
    def setup_method() -> None:
        torch.manual_seed(42)
        torch.npu.manual_seed(42)

    def test_{method_name}(self, ...):
        ...
"""

# B 类：class-based DistributedTest
DIST_TEST_TEMPLATE = """
class Test{ClassName}(DistributedTest):
    world_size = 2  # 或从参数化推导

    def test_{method_name}(self, ...):
        rank = int(os.environ["LOCAL_RANK"])
        ...
"""

# ═══════════════════════════════════════════════════════════
# 变量 3: 设备引用
# ═══════════════════════════════════════════════════════════

# 所有类统一：
DEVICE = 'torch.device("npu")'
SYNC   = 'torch.npu.synchronize()'
RAND   = 'torch.randn(shape, device="cpu").to(device=npu_device)'  # 避 NPU 随机数 kernel

# ═══════════════════════════════════════════════════════════
# 变量 4: 数值对照范式
# ═══════════════════════════════════════════════════════════

# 所有类统一使用 NPU utils.py 提供的：
#   make_reference_and_test_tensors()  — fp64 CPU ref vs NPU test
#   assert_close()                     — device-agnostic assert
#   dtype_tols()                       — per-dtype rtol/atol
#   to_cpu()                           — fp64 CPU conversion
```

**关键认识：A/B/D 三类共享断言设计原则，差异集中在能力守卫和运行结构。A 类使用动态能力 `skipif`，B 类使用 `DistributedTest`，D 类直接执行；不要用 `xfail` 表达硬件能力不足。**

### Step 5: 验证（按标记分叉）

```python
# D 类/A 类单卡测试：
pytest tests/pytorch/test_xxx.py -v

# B 类分布式测试（conftest.py 自动拦截 DistributedTest）：
pytest tests/pytorch/test_xxx.py -v

# 覆盖率：
pytest tests/pytorch/ -v --cov=transformer_engine
```

---

## 总结：能统一到什么程度

| 管道步骤 | A（FP8阻断） | B（分布式） | D（可迁移） | 统一规则？ |
|---------|------------|-----------|----------|----------|
| Step 1 上游定位 | 同 D | grep + 额外关注 `qa/L1_*` | grep 上游测试 | ✅ 完全统一 |
| Step 2 差距分析 | 同 D + FP8 标记 | 同 D + dist 标记 | 标准流程 | ✅ 统一，加标记 |
| Step 3 硬件分类 | 检测到 `npu_quantize` | 检测到 `dist.all_*` | 无依赖 | — 这是分叉点本身 |
| Step 4 代码生成 | 普通 pytest + 能力 `skipif` | `DistributedTest` class | 普通 pytest | **半统一**：测试 body 完全相同，差异在 class 继承和能力守卫 |
| Step 5 验证 | 具备能力时执行；否则明确 skipped | 多进程 runner | 直接 pytest | **半统一**：pytest 命令相同，运行环境不同 |

**三类的测试代码内容 90% 相同，差异只在外壳**：

```
           同一份测试逻辑（body）
        ┌──────────────────────────┐
        │  _to_float8 / quantize   │
        │  dequantize / assert     │
        │  view / reshape / clone  │
        │  matmul / backward       │
        └──────────────────────────┘
           ↑          ↑          ↑
    ┌──────┴──┐ ┌─────┴─────┐ ┌──┴──────┐
    │ D 类壳  │ │ A 类壳    │ │ B 类壳  │
    │ pytest  │ │ pytest    │ │DistTest │
    │ npu()   │ │capability │ │world=2  │
    └─────────┘ └───────────┘ └─────────┘
```
