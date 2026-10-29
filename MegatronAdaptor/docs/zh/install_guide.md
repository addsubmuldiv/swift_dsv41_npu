# 软件安装

本文档旨在指导用户从零开始，基于 PyTorch 框架快速、正确地完成MegatronAdaptor的安装。

## 硬件与操作系统要求

**表 1**  产品硬件支持列表

|产品|是否支持（训练场景）|
|--|:-:|
|<term>Ascend 950PR&950DT系列产品</term>|√|
|<term>Atlas A3训练系列产品</term>|√|
|<term>Atlas A3推理系列产品</term>|x|
|<term>Atlas A2训练系列产品</term>|√|
|<term>Atlas A2推理系列产品</term>|x|
|<term>Atlas 200I/500 A2推理产品</term>|x|
|<term>Atlas推理系列产品</term>|x|
|<term>Atlas训练系列产品</term>|x|

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。

- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](release_notes_ma.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

请单击[固件与驱动](https://hiascend.com/hardware/firmware-drivers)，并根据引导完成固件与驱动的安装。

> [!NOTE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 安装MegatronAdaptor

1. 安装 CANN

   安装配套版本的 NPU 驱动固件、CANN 软件（Toolkit、ops 和 NNAL）并配置环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/softwareinst/instg/instg_0000.html)》。

   CANN 软件提供进程级环境变量设置脚本，训练或推理场景下使用 NPU 执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

2. 安装PyTorch以及TorchNPU

   请参考《[TorchNPU 快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C)》，获取配套版本的PyTorch以及TorchNPU软件包。

   > [!NOTE]
   >
   > - 如环境中存在旧版本 MegatronAdaptor，请先[卸载](#卸载)旧版本，再安装新版本。
   >
   > - 更多TorchNPU插件版本请单击[Link](https://gitcode.com/ascend/pytorch/releases)。

3. 源码安装

   ```shell
   git clone https://gitcode.com/Ascend/MegatronAdaptor.git
   cd MegatronAdaptor
   git checkout core_r0.19.0
   pip install -e .
   cd ..
   ```

4. 安装 TransformerEngineNPU

   Megatron-LM 在昇腾设备上的 Transformer 加速能力由 TransformerEngineNPU 提供（[TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)），需拉取源码安装：

   ```shell
   git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
   pip install -e TransformerEngineNPU --no-build-isolation
   ```

   > [!WARNING]
   >
   > TransformerEngineNPU 和原生 TransformerEngine 的库模块名相同，为避免冲突，请确保同一环境内不要同时安装两者，仅保留 TransformerEngineNPU 即可。

5. 获取Megatron-LM源码

   具体操作如下所示：

   ```shell
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_r0.19.0
   cd ..
   ```

   > [!NOTE]
   >
   > - 获取 Megatron-LM 源码需访问 GitHub。若您所处的网络环境无法顺畅访问 `github.com`，请自行配置代理，或使用国内可访问的 Git 镜像（如 GitCode、Gitee 等）拉取源码后，再切换至配套分支。
   > - 拉取前请核对仓库地址与分支/commit，确保所获取的 Megatron-LM 源码与配套版本一致，并关注其正确性与安全性。

## 卸载

如需卸载 MegatronAdaptor，执行以下命令：

```shell
pip uninstall -y megatron_adaptor
```
