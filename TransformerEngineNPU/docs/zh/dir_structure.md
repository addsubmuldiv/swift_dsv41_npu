# TransformerEngineNPU项目目录结构

```plaintext
TransformerEngineNPU/
├── README.md                            # 项目说明文档，介绍TransformerEngineNPU的定位、版本配套与安装方法
├── LICENSE                              # 许可证文件
├── setup.py                             # 安装打包脚本，定义包依赖与构建入口
├── pyproject.toml                       # 构建系统配置，声明构建后端与构建期依赖
├── docs/                                # 项目文档目录
│   ├── LICENSE                          # 文档许可证文件（CC-BY 4.0）
│   ├── zh/                              # 中文文档，包含简介、软件安装、API列表、FAQ与安全声明
│   └── en/                              # 英文文档目录，由翻译团队维护
├── transformer_engine/                  # Python包主目录，包含全部核心实现代码
│   ├── common/                          # 框架无关的公共模块
│   │   └── recipe/                      # 量化策略，定义DelayedScaling、MXFP8、BlockScaling等缩放策略
│   ├── debug/                           # 数值调试工具，支持量化行为的在线观测与比对
│   │   └── pytorch/                     # PyTorch后端的调试实现
│   │       ├── debug_state.py           # 调试状态，维护数值调试的全局状态
│   │       ├── debug_quantization.py    # 调试量化器，包含量化过程的数值观测逻辑
│   │       └── utils.py                 # 调试工具函数
│   └── pytorch/                         # PyTorch后端，TENPU对外接口的主要实现
│       ├── constants.py                 # 常量定义，包含注意力后端等全局开关
│       ├── cpu_offload.py               # CPU Offload上下文，提供激活/权重卸载能力
│       ├── distributed.py               # 分布式工具，提供checkpoint、TP属性设置等功能
│       ├── fp8.py                       # FP8元信息，定义量化格式相关的基础设施
│       ├── float8_tensor.py             # Float8张量（旧路径），兼容性保留的实现
│       ├── jit.py                       # JIT工具，支持torch.compile等编译路径
│       ├── permutation.py               # MoE排列，提供token重排与恢复原语
│       ├── quantized_tensor.py          # 量化张量基类，定义QuantizedTensor与Quantizer抽象
│       ├── router.py                    # MoE路由，提供融合top-k与辅助损失计算
│       ├── torch_version.py             # 版本适配，处理不同torch版本的兼容差异
│       ├── utils.py                     # 通用工具函数
│       ├── attention/                   # 注意力模块
│       │   ├── rope.py                  # RoPE旋转位置编码，含融合实现
│       │   └── dot_product_attention/   # 融合注意力实现，含Ulysses/Ring/KV-AllGather/Swap等上下文并行策略
│       ├── module/                      # 基础层模块
│       │   ├── base.py                  # 模块基类，提供FP8状态管理与UserBuffer初始化
│       │   ├── linear.py                # Linear全连接层，支持FP8/BF16与延迟wgrad
│       │   ├── grouped_linear.py        # GroupedLinear分组全连接层，服务MoE场景
│       │   ├── layernorm.py             # LayerNorm归一化层
│       │   ├── rmsnorm.py               # RMSNorm归一化层
│       │   ├── layernorm_linear.py      # LayerNormLinear，归一化与线性层融合
│       │   ├── layernorm_mlp.py         # LayerNormMLP，归一化与两层MLP融合
│       │   ├── fp8_padding.py           # Fp8Padding，token维度对FP8 GEMM友好的填充
│       │   └── fp8_unpadding.py         # Fp8Unpadding，与Fp8Padding配套的反填充
│       ├── ops/                         # 底层算子封装
│       │   ├── basic/                   # 基础算子，逐算子的参考实现
│       │   ├── fused/                   # 融合算子，跨算子的融合实现
│       │   ├── fuser.py                 # 算子融合器，按配置选择融合路径
│       │   └── gemm_comparison.py       # GEMM数值比对，量化GEMM与参考实现在线核对
│       ├── optimizers/                  # 优化器算子
│       │   └── multi_tensor_apply.py    # 多tensor融合算子，合并优化器步骤的同类操作
│       ├── quantization/                # 量化管理
│       │   ├── manager.py               # 量化管理器，提供autocast、fp8_model_init等上下文
│       │   ├── utils.py                 # 量化能力查询，如is_fp8_available
│       │   ├── quantizer_role.py        # 量化角色定义，支撑CustomRecipe
│       │   └── state/                   # 策略运行时状态，各recipe对应的RecipeState实现
│       ├── tensor/                      # 量化张量
│       │   ├── float8_tensor.py         # Float8Tensor与Float8Quantizer实现
│       │   ├── float8_blockwise_tensor.py # 128×128块级缩放FP8张量
│       │   ├── mxfp8_tensor.py          # MXFP8缩放张量
│       │   ├── mxfp4_tensor.py          # MXFP4缩放张量（昇腾扩展）
│       │   ├── hif8_tensor.py           # HIF8张量（昇腾扩展）
│       │   ├── w4a8_tensor.py           # W4A8权重张量（昇腾扩展）
│       │   └── storage/                 # 张量存储层，定义量化张量的序列化与底层存储
│       └── triton/                      # triton-ascend实现，基于Triton的算子kernel
├── tests/                               # 测试目录
│   ├── pytorch/                         # PyTorch接口单元测试
│   ├── atk/                             # ATK相关测试
│   └── system_tests/                    # 系统级测试
├── build_tools/                         # 构建工具，维护版本号文件
├── ci/                                  # 持续集成脚本
└── pre-commit/                          # 代码提交前检查配置，含ruff与拼写检查规则
```
