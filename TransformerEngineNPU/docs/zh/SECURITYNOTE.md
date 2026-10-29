# TransformerEngineNPU 安全声明

本文中列出的安全加固措施为基本的加固建议项。用户应根据自身业务，重新审视整个系统的网络安全加固措施。用户应按照所在组织的安全策略进行相关配置，包括但不限于软件版本、系统加固要求、安全配置（数据存储、调试接口、运行环境等）、权限配置、防火墙设置等。必要时可参考业界优秀加固方案和安全专家的建议。

## 系统安全加固

建议用户在系统中配置开启 ASLR（级别2），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：

```bash
echo 2 > /proc/sys/kernel/randomize_va_space
```

## 运行用户建议

基于安全性考虑，建议您在执行任何命令时，避免使用root等管理员类型账户执行，遵循权限最小化原则。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
- 建议用户对训练所需文件、训练过程中保存的文件、用户个人的隐私数据、商业资产等敏感文件做好权限控制等安全措施，例如多用户共享数据集场景下的数据集文件写权限控制等，设定的权限建议参考表1进行设置。
- TENPU以 `pip install -e` 方式从源码安装，用户安装和使用过程需要做好权限控制，建议参考表1进行设置。如需要保存安装/卸载日志，可在安装/卸载命令后面加上参数 `--log <FILE>`，注意对 `<FILE>` 文件及目录做好权限管控。
- TENPU的Triton算子（`transformer_engine/pytorch/triton/`）在首次执行时会由Triton-Ascend即时编译kernel，TENPU未显式指定缓存目录，缓存位置由Triton-Ascend自身的默认规则决定（通常为 `~/.triton/cache`，可通过 `TRITON_CACHE_DIR` 环境变量指定）。建议用户参考表1对生成文件及目录做好权限控制。
- TENPU中由 `NVTE_TORCH_COMPILE` 控制的 `torch.compile` 路径（默认开启）会在运行时缓存编译产物，缓存位置由PyTorch决定（通常为 `~/.cache/torch/inductor`，可通过 `TORCHINDUCTOR_CACHE_DIR` 环境变量指定）。建议用户参考表1对生成文件及目录做好权限控制。
- TENPU自身不读写磁盘上的数据集、模型权重与checkpoint文件，也不创建日志文件。训练过程中由调用方（Megatron-LM、MegatronAdaptor等）生成的数据集索引文件、checkpoint文件、TensorBoard日志等，其权限依赖系统设定与调用方实现。建议当前执行脚本的用户根据自身需要，对生成文件做好权限控制，设定的权限可参考表1进行设置。
- TENPU的数值调试能力（`transformer_engine/debug/`）依赖第三方 `nvdlfw-inspect` 包输出观测日志，日志的落盘路径由该包的配置文件决定，不在TENPU的控制范围内，建议用户对其输出目录做好权限控制。
- 使用TENPU进行训练时请不要修改除数据资产和训练脚本以外的其他文件的权限。

**表1** 文件（夹）各场景权限管控推荐最大值

| 类型           | Linux权限参考最大值 |
| -------------- | ---------------  |
| 用户主目录                        |   750（rwxr-x---）            |
| 程序文件（含脚本文件、库文件等）       |   550（r-xr-x---）             |
| 程序文件目录                      |   550（r-xr-x---）             |
| 配置文件                          |  640（rw-r-----）             |
| 配置文件目录                      |   750（rwxr-x---）             |
| 日志文件（记录完毕或者已经归档）        |  440（r--r-----）             |
| 日志文件（正在记录）                |    640（rw-r-----）           |
| 日志文件目录                      |   750（rwxr-x---）             |
| Debug文件                         |  640（rw-r-----）         |
| Debug文件目录                     |   750（rwxr-x---）  |
| 临时文件目录                      |   750（rwxr-x---）   |
| 维护升级文件目录                  |   770（rwxrwx---）    |
| 业务数据文件                      |   640（rw-r-----）    |
| 业务数据文件目录                  |   750（rwxr-x---）      |
| 密钥组件、私钥、证书、密文文件目录    |  700（rwx------）      |
| 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
| 加解密接口、加解密脚本            |   500（r-x------）        |

## 数据安全声明

- TENPU在加载模型checkpoint时会反序列化其扩展状态（`_extra_state`），涉及以下两处实现：
  - `transformer_engine/pytorch/module/base.py` 与 `transformer_engine/pytorch/ops/op.py` 的 `set_extra_state` 在**默认格式**（字节tensor）分支中调用 `pickle.loads` 还原状态，两处均已标注 `# nosec`，其输入为本模块 `get_extra_state` 自身序列化产生的数据；
  - `transformer_engine/pytorch/module/base.py` 的 `set_extra_state` 在**兼容旧格式**（`io.BytesIO`）分支中调用 `torch.load(state, map_location="npu")`，未显式指定 `weights_only=True`。
- `torch.load` 在PyTorch 2.6以下版本默认参数 `weights_only=False`，存在潜在安全风险（CVE-2025-32434）。本版本配套的PyTorch为2.10.0，不受该默认值影响；用户在其他PyTorch版本上运行TENPU时，请在加载来源不可信的checkpoint时显式指定 `weights_only=True`。
- 使用TENPU运行过程中可能会执行模型的加载与保存操作，需要特别注意的是，其底层实现可能使用Python `pickle` 模块进行部分文件的序列化/反序列化操作，该模块存在固有的安全风险。请仅加载可信来源的checkpoint与权重文件。
- 为支持TENPU量化张量（`Float8Tensor`、`MXFP8Tensor`、`Float8BlockwiseQTensor` 等）的checkpoint在 `weights_only=True` 下正常加载，TENPU在 `import transformer_engine.pytorch` 时会调用 `torch.serialization.add_safe_globals`，将上述量化张量存储类型、对应的Quantizer类型、`__reduce_ex__` 重构函数以及 `signal.Signals`/`signal.SIGTERM` 加入本进程的反序列化白名单（见 `transformer_engine/pytorch/__init__.py`）。该注册过程会放宽**本进程内所有** `torch.load` 调用的全局白名单范围，请用户知悉。
- TENPU依赖CANN的基础能力实现算子dump、日志记录等功能，用户需要关注上述功能生成文件的权限控制。

## 运行安全声明

- 建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
- TENPU在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。
- TENPU默认开启 `torch.compile` 路径（由环境变量 `NVTE_TORCH_COMPILE` 控制，默认值为 `1`），首次执行算子融合路径时会产生额外的编译开销。如遇编译相关问题或希望规避运行时代码生成，可将其置 `0` 以禁用该路径。
- TENPU中的Triton算子默认关闭autotune多配置搜索（`NVTE_DISABLE_TRITON_AUTOTUNING` 默认值为 `1`），仅使用单一配置执行，以避免在用户环境上执行大范围配置搜索带来额外的编译耗时与资源占用。
- TENPU依赖torch_npu与CANN提供设备侧执行与集合通信能力，分布式场景下通信本身无认证、不加密，其风险与加固方式请参见[通信安全加固](#通信安全加固)与[通信矩阵](#通信矩阵)。

## 公网地址声明

TENPU代码与文档中包含的公网地址如下表所示。除此之外，TENPU源码中不包含其他公网IP地址、公网URL地址、域名或邮箱地址；发布包 `transformer_engine` 自身在导入与运行时**不发起任何网络请求**。

**表2** 公网地址声明列表

|      类型      |                                           开源代码地址                                           |                            文件名                             |             公网IP地址/公网URL地址/域名/邮箱地址             |                   用途说明                    |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------:| :----------------------------------------------------------: |:-----------------------------------------:|
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | setup.py | <https://gitcode.com/Ascend/TransformerEngineNPU> | 打包元数据`url`，指向本仓库源码地址 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | setup.py | <https://gitcode.com/Ascend/TransformerEngineNPU/release> | 打包元数据`download_url`，指向本仓库release页 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | README.md | <https://gitcode.com/Ascend/MegatronAdaptor> | 配套适配库MegatronAdaptor的仓库链接 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://gitcode.com/Ascend/TransformerEngineNPU.git> | 安装指导：克隆TENPU源码 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://gitcode.com/Ascend/MegatronAdaptor.git> | 安装指导：克隆MegatronAdaptor |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://gitcode.com/ascend/pytorch/releases> | 安装指导：TorchNPU插件版本下载入口 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | ci/run_ci.sh | <https://gitcode.com/Ascend/MegatronAdaptor.git> | 门禁ST脚本：克隆MegatronAdaptor |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | ci/run_ci.sh | <https://gitcode.com/GitHub_Trending/me/Megatron-LM.git> | 门禁ST脚本：克隆Megatron-LM |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | tests/atk/README.md | <https://gitcode.com/Ascend/ATK.git> | ATK精度测试工具仓克隆地址 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | tests/atk/README.md | <https://gitcode.com/Ascend/ATK/blob/master/docs/ATK%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/02%20%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/%E7%B2%BE%E5%BA%A6%E6%A0%87%E5%87%86%E8%AF%B4%E6%98%8E.md> | ATK精度标准说明文档 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://www.hiascend.com/hardware/compatibility> | 硬件兼容性查询助手 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0101.html> | CANN操作系统兼容性说明 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://www.hiascend.com/hardware/firmware-drivers> | 固件与驱动下载 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://www.hiascend.com/cann/download> | CANN Toolkit下载 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download> | PyTorch与TorchNPU下载 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://mirrors.huaweicloud.com/ascend/repos/pypi> | 安装triton-ascend时配置的pip扩展源 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | docs/zh/install_guide.md | <https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html> | Triton-Ascend安装指南 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | .pre-commit-config.yaml | <https://gitcode.com/pre-commit/pre-commit-hooks> <br> <https://gitcode.com/gh_mirrors/ru/ruff-pre-commit> <br> <https://gitcode.com/gh_mirrors/co/codespell> <br> <https://gitcode.com/gh_mirrors/pyl/pylint> <br> <https://gitcode.com/gh_mirrors/ba/bandit> <br> <https://gitcode.com/gh_mirrors/ty/typos> <br> <https://gitcode.com/pre-commit-clang/mirrors-clang-format> | pre-commit代码检查hook仓库地址，提交与CI执行pre-commit时拉取其工具代码 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | .gitcode/workflows/pr-pipeline-transformer-engine-npu.yml | <https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta-codecheck/gitleaks> | CI流程下载gitleaks敏感信息扫描工具 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | .pre-commit-config.yaml | <http://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta-codecheck/gitleaks> | 注释中给出的本地执行pre-commit前的gitleaks下载说明 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | .pre-commit-config.yaml | <https://github.com/gitleaks/gitleaks/releases/download/v8.20.0/gitleaks_8.20.0_windows_x64.zip> | 注释中给出的Windows平台gitleaks下载说明 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | .gitcode/workflows/pr-pipeline-transformer-engine-npu.yml | swr.cn-north-4.myhuaweicloud.com/pytorch_images_x86/pytorchx86:v_03 <br> swr.cn-north-4.myhuaweicloud.com/mindspore/mindspore_python3.11:v0.1 <br> swr.cn-north-4.myhuaweicloud.com/mindspeed/pr-mindspeed:910b-openeuler24.03-py3.12-torch-2.9.0-aarch64-v2 | CI流程所用的容器镜像地址 |
| 开源引入 | https://gitcode.com/Ascend/TransformerEngineNPU | .agent/skills/tenpu-daily-orchestrator/scripts/nvte_daily_run.py | <https://api.github.com/repos/NVIDIA/TransformerEngine/commits> | 每日NVTE变更巡检脚本调用GitHub API（需配置GH_TOKEN） |
| 开源引入 | <https://github.com/NVIDIA/TransformerEngine/issues/597> | transformer_engine/pytorch/jit.py | <https://github.com/NVIDIA/TransformerEngine/issues/597> | dropout_fuser实现参考 |
| 开源引入 | <https://github.com/NVIDIA/TransformerEngine/pull/351> <br> <https://github.com/NVIDIA/TransformerEngine/pull/363> | transformer_engine/pytorch/ops/op.py | <https://github.com/NVIDIA/TransformerEngine/pull/351> <br> <https://github.com/NVIDIA/TransformerEngine/pull/363> | CPU拷贝状态处理实现参考 |
| 开源引入 | <https://github.com/Lightning-AI/lightning-thunder/pull/2635> | transformer_engine/pytorch/fp8.py | <https://github.com/Lightning-AI/lightning-thunder/pull/2635> | recipe类向后兼容再导出的实现参考 |
| 开源引入 | <https://github.com/bzhangGo/rmsnorm> | tests/pytorch/test_numerics.py | <https://github.com/bzhangGo/rmsnorm/blob/c6691f20ec0af4128c8159c903071f7575404295/rmsnorm_torch.py> | RMSNorm参考实现来源说明 |
| 开源引入 | <https://github.com/NVIDIA/TransformerEngine> | tests/atk/README.md | <https://github.com/NVIDIA/TransformerEngine.git> | ATK精度测试所需的NVTE基线仓 |
| 开源引入 | <https://github.com/NVIDIA/TransformerEngine#setup> | tests/atk/README.md | <https://github.com/NVIDIA/TransformerEngine#setup> | NVTE编译说明文档 |
| 开源引入 | <https://github.com/NVIDIA/Megatron-LM.git> | docs/zh/install_guide.md | <https://github.com/NVIDIA/Megatron-LM.git> | 安装指导：克隆Megatron-LM源码 |
| 开源引入 | <https://github.com/NVIDIA/Megatron-LM> | tests/atk/module/_cpu_baselines.py | <https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0rc3/LICENSE> | 版权头中的Apache-2.0许可地址（Megatron-LM core_v0.13.0rc3 的 LICENSE） |
| 开源引入 | <https://arxiv.org/abs/1607.06450> | transformer_engine/pytorch/module/layernorm.py <br> transformer_engine/pytorch/ops/basic/layernorm.py | <https://arxiv.org/abs/1607.06450> | Layer Normalization论文地址 |
| 开源引入 | <https://arxiv.org/abs/1910.07467> | transformer_engine/pytorch/module/rmsnorm.py <br> transformer_engine/pytorch/ops/basic/rmsnorm.py | <https://arxiv.org/abs/1910.07467> | RMSNorm论文地址 |
| 开源引入 | <https://arxiv.org/abs/2002.05202> | transformer_engine/pytorch/ops/basic/swiglu.py | <https://arxiv.org/abs/2002.05202> | GLU Variants论文地址 |
| 开源引入 | <https://arxiv.org/abs/1706.03762> <br> <https://arxiv.org/pdf/2305.13245.pdf> <br> <https://arxiv.org/pdf/1911.02150.pdf> <br> <https://arxiv.org/pdf/2309.17453v3> <br> <https://arxiv.org/pdf/2502.16982> | transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py | <https://arxiv.org/abs/1706.03762> <br> <https://arxiv.org/pdf/2305.13245.pdf> <br> <https://arxiv.org/pdf/1911.02150.pdf> <br> <https://arxiv.org/pdf/2309.17453v3> <br> <https://arxiv.org/pdf/2502.16982> | Attention/GQA/MQA/StreamingLLM等算法论文地址 |
| 开源引入 | <https://arxiv.org/abs/2104.09864> <br> <https://arxiv.org/abs/2306.15595> | transformer_engine/pytorch/attention/rope.py | <https://arxiv.org/abs/2104.09864> <br> <https://arxiv.org/abs/2306.15595> | RoPE与位置插值论文地址 |
| 开源引入 | <https://arxiv.org/pdf/1710.03740> | docs/zh/features/low_precision_training/introduction.md | <https://arxiv.org/pdf/1710.03740> | 混合精度训练论文地址 |
| 开源引入 | <https://docs.nvidia.com/deeplearning/transformer-engine/> | docs/zh/api/api_list.md | <https://docs.nvidia.com/deeplearning/transformer-engine/> | NVTE官方文档（接口对齐说明） |
| 开源引入 | <https://docs.nvidia.com/deeplearning/transformer-engine/faq.html#fp8-checkpoint-compatibility> | transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py | <https://docs.nvidia.com/deeplearning/transformer-engine/faq.html#fp8-checkpoint-compatibility> | FP8 checkpoint兼容性说明 |
| 开源引入 | <https://pytorch.org/docs/stable/checkpoint.html> | transformer_engine/pytorch/distributed.py | <https://pytorch.org/docs/stable/checkpoint.html> | `torch.utils.checkpoint`接口文档 |

## 公开接口声明

- TENPU保持与原生TransformerEngine（NVTE）相同的库名、模块路径与接口签名，并在此基础上提供昇腾专属扩展接口。
- **公开接口清单以 [API列表](api/api_list.md) 中记录的接口为准。**该文档中标注为“与NVTE对齐”的接口，其在NVTE中的公开性同样适用于TENPU；标注为“昇腾扩展”的接口为TENPU新增的公开接口。
- 与部分昇腾仓库不同，TENPU的 `transformer_engine/__init__.py` 为对齐NVTE release_v2.17采用eager re-export方式，**不使用 `__all__` 定义导出集合**，因此不以 `__all__` 作为公开接口的判断依据。如果一个函数/类在API列表中已记录，则该接口为公开接口，可以依赖其作为公共函数；其具体使用方法与适用场景请参照该文档及NVTE官方文档。
- 如果需要依赖一个在文档中未记录的函数，请在本仓库主页提交Issue向我们确认该函数是否为公开接口、是否是因意外暴露、或者可能在未来被移除。

## 通信安全加固

TENPU自身不监听端口、不创建socket、不调用 `torch.distributed.init_process_group`，不主动发起任何网络通信。分布式训练场景下的通信能力全部由调用方已初始化的 `torch.distributed` 进程组以及torch_npu/HCCL提供。

因此，本节的通信安全加固要求与CANN、TorchNPU的既有约束一致，具体加固方式（如使用iptables在分布式训练开始前限制外部对通信端口的访问、训练结束后清理防火墙规则，以及相应的脚本模板）请参见：

[通信安全加固说明](https://gitcode.com/Ascend/pytorch/blob/v26.1.0-pytorch2.10.0/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA)

## 通信矩阵

TENPU是算法库，本身不新增任何侦听端口。其运行时的网络通信均由PyTorch分布式组件与CANN的HCCL组件承载，端口、认证与加密特性与调用方环境完全一致，如下表所示。更详细的说明请参见[通信矩阵说明](https://gitcode.com/Ascend/pytorch/blob/v26.1.0-pytorch2.10.0/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5)。

**表3** 通信矩阵信息

| 组件                  |  PyTorch                             |  HCCL                                |
| --------------------- | ------------------------------------ | ------------------------------------ |
| 源设备                | 运行TENPU进程的服务器              | 运行TENPU进程的服务器             |
| 源IP                  | 设备地址IP                            | 设备地址IP                            |
| 源端口                | 操作系统自动分配，分配范围由操作系统自身配置决定      | 默认值为60000，取值范围[1024,65520]。用户可以通过`HCCL_IF_BASE_PORT`环境变量指定Host网卡起始端口号，配置后系统默认占用以该端口起始的16个端口      |
| 目的设备              | 运行TENPU进程的服务器              | 运行TENPU进程的服务器             |
| 目的IP                | 设备地址IP                            | 设备地址IP                            |
| 目的端口（侦听）      | 默认29500/29400，用户可以设定端口号     | 默认值为60000，取值范围[1024,65520]。用户可以通过`HCCL_IF_BASE_PORT`环境变量指定Host网卡起始端口号，配置后系统默认占用以该端口起始的16个端口      |
| 协议                  | TCP                                  | TCP                                   |
| 侦听端口是否可更改     | 是                                    | 是                                    |
| 认证方式              | 无认证方式                             | 无认证方式                             |
| 加密方式              | 无                                    | 无                                    |
| 备注                  | 由开源软件PyTorch控制，配置为PyTorch原生设置，可参考[PyTorch文档](https://pytorch.org/docs/stable/distributed.html#launch-utility) | 由CANN中HCCL组件控制，TENPU与torch_npu均不进行控制，端口范围可参考《环境变量参考》的“执行相关 > 集合通信 > HCCL_IF_BASE_PORT” |

### 特殊场景

| 场景                                  | 使用方法                                         | 端口 | 可能的风险       |
|-------------------------------------| ------------------------------------------------ | ---------- | ---------- |
| 上下文并行（Ulysses/Ring/KV-AllGather/Swap等策略） | 调用`DotProductAttention`并通过`cp_comm_type`指定策略；TENPU在调用方已初始化的进程组之上通过`torch.distributed.new_group`建立上下文并行子通信组 | 复用调用方进程组端口，不新增端口 | 子通信组继承调用方进程组无认证、不加密的特性，风险与调用方分布式环境一致 |
| torch_npu融合通信算子 | TENPU通过`get_hccl_comm_name`读取调用方已创建的HCCL通信域名，作为不透明句柄传入`torch_npu.npu_all_gather_quant_mm`/`torch_npu.npu_quant_mm_reduce_scatter`等融合算子 | 复用调用方HCCL端口，不新增端口 | 通信过程由torch_npu与CANN控制，TENPU不感知其端口细节 |
| CPU Offload | 通过`get_cpu_offload_context`按层序列将激活/权重卸载至主机内存 | 不涉及网络 | 卸载数据落于主机内存，需关注主机侧内存隔离与访问控制 |
| 数值调试 | 通过`DebugQuantizer`/`GroupedDebugQuantizer`配合`nvdlfw-inspect`配置输出在线观测数据 | 不涉及网络 | 调试产物可能包含模型中间数值，需按表1做好文件权限控制 |
| 测试与CI脚本 | `tests/`与`ci/`中的分布式测试脚本使用`MASTER_ADDR=127.0.0.1`/`localhost`等回环地址；`tests/pytorch/utils.py`的`get_free_port`会短暂绑定通配地址的0端口以获取空闲端口 | 回环地址或操作系统随机分配端口 | 仅用于本地单机测试，不对外提供服务；`get_free_port`绑定的是0端口，不会暴露服务 |

## 漏洞响应

我们非常重视社区版本的安全性，昇腾开源社区会接收、调查和披露本社区相关的安全漏洞。具体请参见[Ascend漏洞响应](https://gitcode.com/Ascend/community/blob/master/docs/security.md)。
