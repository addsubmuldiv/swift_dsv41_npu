---
name: tenpu-ut-guardian
description: 生成、迁移、审查和远程验证 TransformerEngineNPU PyTorch 单元测试。用于 TENPU UT 覆盖率、生成的 UT 质量、A/B/D 硬件分类、分布式 DistributedTest 用例、NPU-vs-上游测试迁移或远程 NPU pytest 执行。
metadata:
  short-description: TransformerEngineNPU UT 生成、审查与远程验证约束
---

# TENPU UT Guardian

TransformerEngineNPU 单元测试覆盖率看护、UT 迁移/生成、质量审查与远端 NPU 验证。

## 硬性约束

- 远程环境信息不得写入 skill、模板、脚本默认值或提交内容；开始远程验证前先询问用户提供 host、user、workdir、conda/bin、CANN env script、可用设备/端口段等信息，并通过当前 shell 环境变量传入
- 密码、token、私钥路径等敏感信息只能通过用户本机环境变量或 SSH agent/keychain 提供；不要在命令行示例、`.env_npu`、日志、commit message 或最终回复中回显明文
- 在 `TransformerEngineNPU/` 工作；仅将 `TransformerEngine/` 作为上游参考
- 编写测试前，先读当前相关模块的 NPU 源码、当前 NPU 测试、上游参考测试和 `tests/pytorch/utils.py`
- 从活动 NPU 源码或远程运行时确认目标 API 签名，不要从上游 Transformer Engine、旧分支或记忆推断签名
- 所有 NPU 测试需加门控：`pytestmark = pytest.mark.skipif(not npu_available(), reason="NPU device is required")`
- A 类 FP8/HiF8/MXFP8/W4A8 测试使用当前源码的能力检测做 `skipif`；A5 环境必须实际执行，A2/A3 上的 skipped case 不计有效覆盖，禁止用 xfail 表达硬件不可用
- 分布式 B 类测试使用 `DistributedTest`，导入 `torch.distributed as dist`，传递真实进程组（`dist.group.WORLD` 或显式 group，禁止 `None`），验证 `dist.get_rank()` / `dist.get_world_size()`，并用 rank-distinct 输入做数值断言
- 同一 DistributedTest class 内连续真实 HCCL collective 若在第二个用例报 `hcclCommInitRootInfoConfig`，优先设置 `reuse_dist_env = False`，不要放宽断言或吞异常
- 覆盖率口径先由用户确认：A5 全量口径不得 omit A5/量化/Triton 源码；只有明确要求 A2/A3 能力口径时才可配置对应 omit，并在报告中列出规则。只删除/跳过测试不会缩小源码分母
- 所有新增/修改的测试文件必须全部远程运行通过（不接受只跑代表性样本）；分布式编辑必须运行对应的分布式文件
- 质量硬门槛（细则见 `reference/assertion-guide.md`，以下为入口级约束）：
  - 禁止纯 import 测试（`import transformer_engine` 的可用性由包 `__init__` 自然验证）；禁止恒真断言（同对象自比较、同模块 round-trip 自证、无 RNG 重 seed 自证、自己置位自己断言、枚举全列 membership、stub 空测）
  - 数值断言的 expected 必须独立构造（fp64 CPU 参考 / 手算公式 / PyTorch native 对照），禁止用被测函数自身输出当参考值
  - 梯度断言必须校验数值；`grad is not None` 只是 L1.1 结构断言，不能单独成 case
  - 每文件 L1.2/L1.3 数值·语义断言 case 占比 ≥50%，smoke case ≤2 条（标记 `# smoke:`），raises-only case ≤30%，反向/扰动断言 ≥1 条
  - 版本/能力判断用 `check_npu_version` / `get_npu_version` 动态判断，禁止硬编码"本机是 A2/A5"的断言
  - xfail 必须 `strict=True` 并注明根因与跟踪 issue；公开兼容 API 的回归不允许用 xfail 接受——先修源码再移除 xfail
  - 同构用例重复 ≥3 次必须 `@pytest.mark.parametrize` 合并；禁止重复定义 `utils.py` 或被测模块已有的 helper/常量
  - 被测对象和稳定 helper 在文件顶部导入；禁止在每个 test 函数内重复 import，除非该 import 本身就是被测延迟导入/可选依赖行为
  - 测试不得假设当前类/父类未声明的实例属性或输出格式；构造参数名不等于保存属性名，继承 `torch.nn.Module` 也不代表 `extra_repr()` 有业务输出

## 工作流

### 0. 远程环境配置

- 首次需要远程运行时，先向用户询问非敏感连接信息：`NPU_HOST`、`NPU_USER`、`NPU_WORKDIR`、`NPU_CONDA_BIN`、`NPU_ENV_SCRIPT`、可选 `NPU_ATB_SCRIPT`、空闲设备和 HCCL 端口段
- 要求用户在当前 shell 中设置环境变量；敏感项只允许 `NPU_PASSWORD` 环境变量或 SSH agent/keychain，不写入任何文件
- 运行脚本前检查环境变量是否存在；缺失时询问用户，不要猜测远端路径、用户名、密码或设备占用
- 远端命令中可使用 `ASCEND_RT_VISIBLE_DEVICES`、`HCCL_NPU_SOCKET_PORT_RANGE` 等环境变量，但值应来自用户或实时环境检查结果
- 每个新 shell 先 source 用户提供的 CANN/toolkit 脚本；pytest 前用独立 Python 进程预检 `import torch, torch_npu`，涉及对应测试时再预检 `triton`、`nvdlfw_inspect` 等仓库声明依赖

### 1. 定位目标

- 现有覆盖率工作：运行 `scripts/run_npu_ut.py` 检查 `--cov-report=term-missing`
- 覆盖率工作：先记录 `HEAD`、目标硬件、pytest 收集数、coverage `Stmts/Miss` 和 `.coveragerc`；按 `reference/coverage-classification.md` 动态分类，不复用历史百分比
- 增量工作：检查 `git diff <base>..<target>` 及变更文件
- 用 `rg "ClassName|function_name" ../TransformerEngine/tests/pytorch` 找上游测试

### 2. 分类硬件路径

- **A 类**：FP8/量化路径如 `npu_quantize`、`npu_quant_matmul`、`Float8Tensor`；A5 门控
- **B 类**：分布式/并行路径如 `dist.*`、`gather_along_dim`、`_fsdp_*`；用 `DistributedTest`
- **D 类**：直接 NPU pytest 路径；用普通 pytest 类/函数结构
- 阅读 `reference/coverage-classification.md` 了解动态分类和覆盖缺口选择方法

### 3. 生成或迁移测试

- 编码前遵循 `reference/pre-generation-reading.md`
- 使用模板：
  - D 类：`templates/test-class-d.py.tmpl`
  - A 类：`templates/test-class-a.py.tmpl`
  - B 类：`templates/test-class-b.py.tmpl`
- 使用 `reference/assertion-guide.md` 的 NPU 断言规则
- B 类测试必须额外通过 `reference/assertion-guide.md` 的 "B 类 DistributedTest Checklist"
- 将测试移入子目录时添加 package 边界（通常为 `__init__.py`），检查父子目录重复 basename；禁止用子目录 `conftest.py` 修改 `sys.path`

### 4. 自检质量

- 应用 L0-L3 + 文件级量化指标于 `reference/assertion-guide.md`；L1 按**语义限定**计分，不是 grep 命中形式即得分
- 对照"恒真断言禁令"清单逐条扫描，命中一律重写，不得入库
- 上传前使用"常见错误 Checklist"
- 不接受仅测 shape 的测试、宽泛 `except Exception` / `except RuntimeError → skip`、宽泛 skip、陈旧签名、只比较 `state.keys()` 或同对象自比较的 round-trip、纯 import 测试、函数内重复 import、重复定义被测模块已有常量
- 自检报告必须按 assertion-guide 模板输出，包含参考值来源、容差、文件级指标、B 类专项字段
- 对新增/迁移目录执行 `pytest --collect-only`；大量相同 collection 错误只诊断第一个完整 traceback，环境导入失败后必须换干净 Python 进程重试
- 提交前按仓库配置运行 changed/staged files 的 pre-commit；hook 自动修改文件后重新暂存并复跑。不要用 `--all-files` 顺带格式化与本次 MR 无关的存量文件

### 5. 高风险时的独立审查

- 对大文件、超过 10 个新 case 或弱自检，使用 `reference/review-orchestration.md` 中的协议
- 主 agent 提供源码、测试文件和机械自检；审查者判断通过/失败

### 6. 远程验证

- 使用 `scripts/run_npu_ut.py` 进行直接远程 pytest
- 仅当需要 git push/pull 风格同步时才用 `scripts/sync_remote.py --run`
- 所有新增/修改的测试文件逐个远程运行（不接受只跑代表性样本）
- 前后对比具体 missed 行；仅文件级覆盖率增加不够
- 推送前 fetch 官仓主线并确认 `git merge-base --is-ancestor origin/main HEAD`；不满足时先合并主线、解决冲突并重跑门禁，最后核对远端分支 HEAD 等于本地 HEAD

## 资源地图

| 资源 | 说明 |
|------|------|
| `reference/assertion-guide.md` | L0-L3 质量门、恒真断言禁令、文件级量化指标、断言风格、常见错误清单 |
| `reference/pre-generation-reading.md` | 编码前必读检查清单 |
| `reference/review-orchestration.md` | 独立审查提示/输入协议 |
| `reference/migration-engine-design.md` | A/B/D 分类和统一管道原理 |
| `reference/coverage-classification.md` | 动态硬件分类、覆盖口径和目标选择方法 |
| `templates/` | 类特定测试骨架（含完整测试体示范） |
| `scripts/` | 远程 NPU pytest 和同步辅助脚本 |
