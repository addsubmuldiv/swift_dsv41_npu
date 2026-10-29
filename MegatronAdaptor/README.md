# 简介

MegatronAdaptor 是专为 Megatron-LM 在华为昇腾（Ascend）NPU 上高效运行而生的核心组件，面向上层大模型训练框架提供能力适配，让训练框架在昇腾设备上开箱即用、高可靠地运行。依托内置的夜航自动化看护能力（megatron-adaptor-sync），项目已接入 CI 流水线持续运行：每日自动同步上游 Megatron-LM 的演进，跨多分支执行回归验证，并对上游变更引发的兼容性问题自动定位根因、自动生成修复，形成「监测—诊断—修复—回归」的闭环飞轮，让适配能力随上游持续自我进化，为昇腾大模型生态提供长期稳定、可持续演进的坚实保障。

昇腾提供基于Megatron-LM + MegatronAdaptor + TransformerEngineNPU的基础训练能力方案，使用户能够快速在昇腾设备上构建起大模型训练环境，实现开箱可用。

**·** 适配昇腾设备的TransformerEngine加速库: [TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)

# 最新消息

**·** [Aug 28, 2026]: 🚀MegatronAdaptor + TransformerEngineNPU 支持 Mcore 0.19.0 版本。当前分支非正式商用分支，仅完成基础适配，可能存在潜在问题，欢迎反馈。

# 版本说明

当前版本推荐配套表如下：

| 组件 | 版本要求 |
| ------ | ---------- |
| **MegatronAdaptor** | core_r0.19.0 |
| **Mcore** | core_r0.19.0 |
| **TransformerEngineNPU** | main |
| **CANN版本** | 9.2.0 |
| **PyTorch版本** | 2.10.0 |
| **torch_npu版本** | 26.2.0 |
| **Python版本** | Python3.12.x |

# 安装

拉取MegatronAdaptor及TransformerEngineNPU的源码进行安装：

```bash
git clone https://gitcode.com/Ascend/MegatronAdaptor.git
cd MegatronAdaptor
git checkout core_r0.19.0
pip install -e .
cd ..
```

```bash
git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
pip install -e TransformerEngineNPU --no-build-isolation
```

获取并切换Megatron-LM版本至 core_r0.19.0 版本，参考：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.19.0
```

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

TransformerEngineNPU 和 原生TransformerEngine 的库模块名相同，为了避免冲突，请确保同一环境内不要同时安装两者，仅保留TransformerEngineNPU即可。

# 免责声明

## 致MegatronAdaptor使用者

1. MegatronAdaptor提供的所有内容仅供您用于非商业目的。
2. 对于MegatronAdaptor测试用例以及示例文件中所涉及的各模型和数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MegatronAdaptor过程中，发现任何问题（包括但不限于功能问题、合规问题），请在GitCode提交issue，我们将及时审视并解决。
4. MegatronAdaptor功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MegatronAdaptor仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。
