# 生成前阅读清单（Step 3.5）

> **这是管道中最关键的一步。跳过此步会导致批量调用失败、过期属性错误或无效断言。**


## 强制阅读清单（执行顺序固定）

```
Step 3.5 必须在 Step 4（代码生成）之前完成。四份文件必须全部读完。
```

### 文件 1: NPU 源文件（被测代码）

```
目的：理解 API 签名 + 逻辑分支 + 依赖链
```

| 检查项 | 如何做 | 产出 |
|--------|--------|------|
| 类/方法签名 | `grep "def \|class " <file>` | 方法名 + 参数列表 |
| 实例成员契约 | 检查 `self.<attr>`、`register_parameter/buffer`、父类定义和公开文档 | 允许断言的属性/方法及其输出格式 |
| 分支结构 | 阅读每个方法体中的 `if/else`、`try/except` | 哪些分支未被测试触发 |
| 硬件依赖 | `grep "npu_quantize\|npu_quant_matmul\|dist\.\|all_gather\|reduce_scatter" <file>` | 哪些方法 → A/B 分类 |
| 依赖的 NPU 特定类型 | `grep "Float8Tensor\|MXFP8Tensor\|GroupedTensor" <file>` | 哪些路径需要量化 tensor |
| 分布式默认 group | `grep "process_group=None\|group is None\|dist\\.is_initialized" <file>` | 确认 `None` 是 no-op 还是 WORLD |

**关键：远端必须运行同一 revision。生成前先核对 HEAD，再读取当前源码**：

```bash
git rev-parse HEAD
# 本地与远端 HEAD 不一致时停止测试，先同步分支；不要为两个 revision 各写一套调用。
```

### 文件 2: NPU 已有测试

```
目的：理解本地断言风格 + 避免重复覆盖 + 找到可复用的 helpers
```

| 检查项 | 如何做 | 产出 |
|--------|--------|------|
| 已有哪些 test class | `grep "class Test" tests/pytorch/test_*.py` | 不被重复覆盖的已有测试 |
| assert 风格 | `grep "assert\|assert_close" tests/pytorch/test_<相关>.py` | 该文件断言的具体写法 |
| helpers | 阅读 `tests/pytorch/utils.py` 和测试文件顶部的 helper 函数 | 可用工具：`make_reference_and_test_tensors`、`assert_close`、`dtype_tols`、`to_cpu`、`npu_available` |
| fixtures | 看 `conftest.py` + 文件内 `setup_method` / `setup_class` | `_seed()` 调用方式 |
| skip 守卫写法 | `grep "pytestmark\|skipif" tests/pytorch/test_<相关>.py` | 该文件或同类文件的守卫格式 |
| 分布式复用策略 | 阅读 `distributed_testing.py` 和同类 `DistributedTest` class | 是否需要 `reuse_dist_env = False` |

### 文件 3: 上游对应测试

```
目的：获取覆盖思路 + 缺失的测试场景 + 边界值选法
```

| 检查项 | 如何做 | 产出 |
|--------|--------|------|
| 哪些方法有上游测试 | `grep "目标方法名" ../TransformerEngine/tests/pytorch/test_*.py` | 上游已覆盖的方法 |
| 上游的参考值怎么构造 | 阅读上游测试中 assert 之前的 setup 代码 | 参考值构造思路 |
| 上游的 tolerance 设置 | grep `rtol\|atol\|_tols` | 数值容差参考 |
| 上游的反面验证 | grep `-.*ref\|pytest.raises.*AssertionError` | 哪些测试有反向断言 |

### 文件 4: `tests/pytorch/utils.py`（NPU 测试工具）

```
目的：知道有哪些现成工具可用，避免重新发明
```

所有 NPU 测试共用的工具（必须熟记）：

| 函数 | 用途 | 签名 |
|------|------|------|
| `npu_available()` | 设备可用性检查 | `() -> bool` |
| `dtype_tols(dtype)` | 按 dtype 返回 `{rtol, atol}` | `(torch.dtype) -> dict` |
| `assert_close(actual, expected, ...)` | device-agnostic 数值断言，默认 `check_device=False` | 见 `utils.py` |
| `make_reference_and_test_tensors(shape, ...)` | 构造 fp64 CPU ref + NPU test 张量对，`ref.copy_(test)` 确保同值 | `(shape, *, test_dtype=..., test_device="npu") -> (ref, test)` |
| `to_cpu(tensor)` | 安全转 fp64 CPU（None → None） | `(Tensor?) -> Tensor?` |
| `quantization_tols(name)` | 按量化方案返回 `{rtol, atol}` | `(str) -> dict` |

---

## 阅读完成后才允许进入 Step 4 的门禁

```
读完四份文件后，Agent 必须能在生成代码前回答以下问题：

1. 目标文件的哪些方法尚未被 NPU 已有测试覆盖？
   → 列出待覆盖方法名 + 行号区间

2. 每个待覆盖方法属于 A/B/D 哪类？
   → 输出分类结果（含判定依据）

3. 每个待覆盖方法的参考值怎么构造？
   → 一句话描述（如 "PyTorch native RMSNorm forward" / "手动 Adam step 公式"）

4. 该用哪个 assert helper？
   → assert_close / torch.testing.assert_close / pytest.raises / 自定义 helper

5. 哪些方法在当前环境缺少硬件能力，哪些是已知软件缺陷？
   → 硬件能力不足使用动态 `skipif`；已知软件缺陷仅在有根因和跟踪 issue 时使用 `xfail(strict=True)`

6. 如果是 B 类分布式测试，真实 collective 如何证明？
   → 写出 process group、rank-distinct 输入、期望值公式、远端运行卡和端口段
```

**如果任一问题答不上来 → 不准生成代码。回去重读。**
