# MegatronAdaptor 项目目录结构

```plaintext
MegatronAdaptor/
├── README.md                            # 项目说明文档，介绍 MegatronAdaptor 的特性、配套版本与使用方法
├── AGENTS.md                            # AI 协作指南，说明仓库红线、架构、开发约定与多仓对接关系
├── setup.py                             # 打包配置，定义 megatron_adaptor 包的安装与构建规则
├── docs/                                # 项目文档目录，包含项目介绍、安装指导、快速入门、已支持特性等
├── megatron_adaptor/                    # 核心源码目录，所有适配实现
│   ├── __init__.py                      # 包入口，import 时调用 patch_features() 自动应用全部适配补丁
│   ├── megatron_adaptor.py              # 补丁总入口，patch_features() 协调两阶段补丁应用与特性门控
│   ├── auto_execute.py                  # 导入期执行控制，在不导入本包的前提下控制适配是否自动执行
│   ├── features_manager/                # 特性管理器，定义各特性的参数注册与补丁注册
│   │   ├── mindspeed_feature.py         # 特性基类，定义 optimization_level 门控、补丁应用状态等公共接口
│   │   ├── features_manager.py          # 特性清单管理，统一执行 pre/post 补丁应用、参数校验与补丁移除
│   │   ├── requirements/                # 基础运行环境适配特性
│   │   ├── megatron/                    # Megatron-LM 基础适配特性
│   │   ├── megatron_fix/                # Megatron-LM 上游缺陷修复特性
│   │   ├── transformer_engine/          # TransformerEngineNPU 适配特性
│   │   └── functional/                  # 功能特性
│   ├── patches/                         # 补丁实现目录，存放各 patch 的替换函数
│   │   ├── patch_manager.py             # 补丁管理器，PatchesManager.register_patch 与目标符号替换
│   │   ├── patch_utils.py               # 补丁工具，上游模块可用性探测（如 megatron.training）
│   │   ├── requirements/                # 基础运行环境补丁实现
│   │   ├── megatron/                    # Megatron-LM 补丁实现
│   │   └── transformer_engine/          # TransformerEngineNPU 补丁实现
│   └── utils/                           # 通用工具目录
│       └── args_utils.py                # 参数工具，全量参数 ContextVar 存取与特性参数缓存管理
├── test/                                # 测试目录，包含所有单元测试与系统测试
├── ci/                                  # 持续集成目录
├── pre-commit/                          # pre-commit 配套配置目录
```
