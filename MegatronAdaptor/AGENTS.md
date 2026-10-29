# AGENTS.md — MegatronAdaptor（MA）AI 协作指南

MA 是 Megatron-LM 在昇腾 NPU 上的**非侵入式适配层**：训练入口加一行 `import megatron_adaptor`，即完成算子适配、兼容修复与加速特性启用，不改 Megatron-LM 源码。方案栈为 Megatron-LM + MA + TransformerEngineNPU + CANN；MA 与 Mcore 版本一一对应（分支名即版本号，配套表见 README）。

## 红线

1. **绝不改 Megatron-LM 源码**，一切适配走本仓 Patch 机制，靠 patch 吸收上游演进。
2. **新特性禁止默认使能 patch**（原生特性适配除外），应由训练参数判定或显式开启。
3. **慎用 monkey patch**，优先级：类替换（同时改 `__init__` + `forward`）> 工厂模式（不兼容特性命中同一修改点）> 装饰器（须保证原函数必被执行、语义不变）。
4. **多版本兼容**：改/加 patch 前先确认目标符号在支持范围内可用，参考 `patches/patch_utils.py` 的可用性探测。
5. **特性内聚**：特性本体与框架适配解耦，一特性一目录；禁止上帝类/函数；单函数 ≤200 行、单文件 ≤1000 行。
6. **不兼容要拦截**：代码里显式报错，并在文档写明兼容性说明。

## 架构

`import megatron_adaptor` → `patch_features()` → 特性门控 + 两阶段补丁（pre/post）。

- `features_manager/` 特性层：继承 `MindSpeedFeature` 注册，管参数校验与补丁注册；分 `requirements` / `megatron` / `transformer_engine` / `functional`
- `patches/` 实现层：替换函数实现，经 `PatchesManager.register_patch`（`patches/patch_manager.py`）生效
- `auto_execute.py` 导入期执行控制；`utils/args_utils.py` 参数 ContextVar 存取

细节见 `docs/zh/dir_structure.md`、`docs/zh/introduction.md`。

**命名注意**：本仓的特性/补丁框架源自 MindSpeed，故保留了 `MindSpeedFeature`、`get_mindspeed_args` 等名字，这些**都是本仓自己的类/函数**，不代表改错了仓。真正的上层框架是独立的 `mindspeed` 包：MA 先应用自己的特性，随后 `__init__.py` 尝试 `import mindspeed.megatron_adaptor`，由 MindSpeed 追加其特性（其子类会把 `patch_manager` 指向 `MindSpeedPatchesManager`）。区分依据是 import 路径（`megatron_adaptor.*` vs `mindspeed.*`），不是名字里有没有 MindSpeed。

## 开发约定

- 新特性：`features_manager/` 建目录 → 继承 `MindSpeedFeature` → `register_patch`；同时在 `docs/` 补文档（问题、方案、场景与限制、用法、效果）
- 注释：使用英文注释；适当注释，保持简洁合理，表达清晰准确，不复述代码本身，不堆砌废话
- 必带测试：单测放 `test/unit_tests/` 对应维度目录，优先 `pytest.mark.parametrize`，只写接口级用例——以模块间、类间的主要功能接口（如 `PatchesManager`、`MindSpeedFeature`）为观察对象，通过"调用接口后的外部行为"（属性值、调用结果、状态）断言，内部函数/私有实现不得单独建用例，其行为由接口用例间接覆盖；ST 放 `test/system_tests/feature_tests/` 并注册进 `test_cases.txt`
- 单卡能覆盖就别引入分布式初始化（耗时长）；确需多卡参照 `test/examples/`
- 跑单测：`bash test/unit_tests/run_unit_tests.sh <Megatron-LM路径> [pytest 参数]`，必须传上游路径（脚本内注入 PYTHONPATH），直接 `pytest` 会 import 失败
- ST 是真实多卡训练、依赖环境内固定数据集路径，本地勿尝试执行，交由 CI
- 新文件加版权声明，引入三方代码查协议；提交前过 pre-commit，勿带临时文件（日志、草稿）

## 环境

- 依赖昇腾 NPU 环境，具体版本号以 README「版本说明」为准，勿凭记忆假设
- TENPU 与原生 TransformerEngine 模块名冲突，同环境只能装其一（保留 TENPU）
- 导入即执行 `from torch_npu.contrib import transfer_to_npu`：无 torch_npu 的环境别尝试 import，直接读代码

## 多仓

Megatron-LM 是上游，**只读参考、勿在其中改代码**（确需提 PR 遵守其规范）；TENPU 侧问题不在本仓修；MindSpeed 是独立的上层框架仓，对接关系见上文「命名注意」。

排查「上游变更致适配失效」：上游变更点 → 受影响特性/补丁 → 按红线修复 → 补测试。
