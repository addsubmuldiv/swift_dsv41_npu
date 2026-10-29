# TransformerEngineNPU简介

TransformerEngine是NVIDIA提供的一个用于加速Transformer模型训练和推理的库，而TransformerEngineNPU则是其在昇腾设备上的适配版本，旨在为昇腾设备上的大模型训练和推理提供高效的计算支持。

昇腾提供基于Megatron-LM + MegatronAdaptor + TransformerEngineNPU的基础训练能力方案，使用户能够快速在昇腾设备上构建起大模型训练环境，实现开箱可用。

**·** 适配Megatron-LM在昇腾设备上运行的适配库: [MegatronAdaptor](https://gitcode.com/Ascend/MegatronAdaptor)

# 最新消息

**·** [July 15, 2026]: 🚀MegatronAdaptor + TransformerEngineNPU支持Mcore0.18.0版本

# 版本说明

查看[版本说明](docs/zh/release_notes_tenpu.md)，了解TransformerEngineNPU的最新发布版本以及历史版本，包含对应版本的软件配套表、以及当前版本的更新说明。

# 安装

详细的安装步骤和环境配置请参考[TransformerEngineNPU安装指导](docs/zh/install_guide.md)。

# 快速开始

## 概述

要开始使用MegatronAdaptor + TransformerEngineNPU，只需增加一行代码，即可在昇腾设备上启用Megatron-LM。

## 操作方法

以GPT模型为例：在Megatron-LM目录下修改pretrain_gpt.py文件，在import torch下新增一行：import megatron_adaptor，即如下修改：

```python
from functools import partial
from typing import List, Optional, Tuple

import torch
import megatron_adaptor # 新增此行代码

from gpt_builders import gpt_builder
```

## 注意事项

TransformerEngineNPU和原生TransformerEngine的库模块名相同，为了避免冲突，请确保同一环境内不要同时安装两者，仅保留TransformerEngineNPU即可。

# API列表

查看[TransformerEngineNPU API列表](docs/zh/api/api_list.md)，了解各功能模块的公开接口清单、导入路径与环境变量说明。TENPU接口与原生TransformerEngine同名同签名，并在此基础上提供HIF8、W4A8等昇腾专属扩展接口，两者差异在文中有对比说明。

# 分支维护策略

🛠️ TransformerEngineNPU版本分支的维护阶段如下：

| **状态**              | **时间**   | **说明**                                                                                                                                                     |
| --------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 计划 🕐               | 1-3 个月   | 计划特性                                                                                                                                                     |
| 开发 🕔               | 3 个月     | 开发特性                                                                                                                                                     |
| 维护 🕚               | 6-12 个月  | 合入所有已解决的问题并发布版本，针对不同的TransformerEngineNPU版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月                      |
| 无维护 🕛             | 0-3 个月   | 合入所有已解决的问题，无专职维护人员，无版本发布                                                                                                             |
| 生命周期终止（EOL）🚫 | N/A        | 分支不再接受任何修改                                                                                                                                         |

🛠️ TransformerEngineNPU版本维护策略：

| **TransformerEngineNPU分支** | **维护策略** | **当前状态** | **发布时间** | **后续状态**             | **EOL日期** |
| ---------------------------- | ------------ | ------------ | ------------ | ------------------------ | ----------- |
| 26.2.0_release_v2.17         | 常规版本     | 维护         | 2026/09/30   | 预计2027/03/30起无维护   |             |

# 使用限制

**·** TransformerEngineNPU不支持NVFP4低精度量化

# 安全声明

**⚠️** [TransformerEngineNPU安全声明](docs/zh/SECURITYNOTE.md)

# 贡献指南

欢迎贡献TransformerEngineNPU！请查看[贡献指南](CONTRIBUTING.md)了解如何参与项目贡献。

# 免责声明

## 致TransformerEngineNPU使用者

1. TransformerEngineNPU提供的所有内容仅供您用于非商业目的。
2. 对于TransformerEngineNPU测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用TransformerEngineNPU过程中，发现任何问题（包括但不限于功能问题、合规问题），请在GitCode提交issue，我们将及时审视并解决。
4. TransformerEngineNPU功能依赖的torch等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，TransformerEngineNPU仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、修复所有第三方开源软件的漏洞和错误。

## 致数据所有者

如果您不希望您的模型或数据集在TransformerEngineNPU中被提及，或希望更新TransformerEngineNPU中有关的描述，请在GitCode提交issue，我们将根据您的issue要求删除或更新您相关描述。衷心感谢您对TransformerEngineNPU的理解和贡献。

## License声明

TransformerEngineNPU中涉及的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅TransformerEngineNPU根目录。
TransformerEngineNPU产品的使用许可证，具体请参见[LICENSE](./LICENSE)文件。
TransformerEngineNPU docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

# 致谢

TransformerEngineNPU由华为技术有限公司与开发者联合贡献。
感谢来自社区的每一个PR，欢迎贡献。
