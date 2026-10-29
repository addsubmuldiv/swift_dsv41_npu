# TransformerEngineNPU安装指导

本文主要向用户介绍如何基于源码完成TransformerEngineNPU（TENPU）及其配套软件的安装。

## 硬件配套和支持的操作系统

**表 1** 产品硬件支持列表

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

- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](release_notes_tenpu.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

请单击[固件与驱动](https://hiascend.com/hardware/firmware-drivers)，并根据引导完成固件与驱动的安装。

> [!NOTE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
>
> 更多安全相关内容请参见《[TransformerEngineNPU安全声明](SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 安装TransformerEngineNPU

TransformerEngineNPU的典型使用场景是配合Megatron-LM训练，需要同时安装MegatronAdaptor并获取Megatron-LM源码；单独安装TransformerEngineNPU适用于不基于Megatron-LM的自定义训练场景。

1. 安装CANN

   安装配套版本的NPU驱动固件、CANN软件（Toolkit）并配置CANN环境变量，具体请参考《[CANN快速安装](https://www.hiascend.com/cann/download?versionId=793&ids=d806%2Ch0501%2Ch0601%2Ch0702&currentTab=0)》。

   CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   以上命令以默认安装路径为例，请根据set_env.sh的实际路径进行替换。

2. 安装PyTorch以及TorchNPU

   请参考《[TorchNPU快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C)》，获取配套版本的PyTorch以及TorchNPU软件包。

   > [!NOTE]
   >
   > - 如环境中存在旧版本TransformerEngineNPU，请先[卸载](#卸载transformerenginenpu)旧版本，再安装新版本。
   >
   > - 更多TorchNPU插件版本请单击[Link](https://gitcode.com/ascend/pytorch/releases)。

3. 安装Triton-Ascend

   请参考《[Triton-Ascend安装指南](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#id1)》，获取配套版本的Triton-Ascend安装指令。

   可参考如下安装命令：

   ```shell
   pip install triton-ascend==3.6.0 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

4. 安装MegatronAdaptor

   ```shell
   git clone https://gitcode.com/Ascend/MegatronAdaptor.git
   cd MegatronAdaptor
   git checkout main
   pip install -e .
   cd ..
   ```

5. 安装TransformerEngineNPU

   Megatron-LM在昇腾设备上的Transformer加速能力由TransformerEngineNPU提供，需拉取源码安装：

   ```shell
   git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
   pip install -e TransformerEngineNPU --no-build-isolation
   ```

   > [!NOTE]
   >
   > TransformerEngineNPU和原生TransformerEngine的库模块名相同，为避免冲突，请确保同一环境内不要同时安装两者，仅保留TransformerEngineNPU即可。

6. 获取Megatron-LM源码

   具体操作如下所示：

   ```shell
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_r0.18.0
   cd ..
   ```

   > [!NOTE]
   >
   > - 获取Megatron-LM源码需访问GitHub。若您所处的网络环境无法顺畅访问`github.com`，请自行配置代理，或使用国内可访问的Git镜像（如GitCode、Gitee等）拉取源码后，再切换至配套分支`core_r0.18.0`。

## 验证安装

安装完成后，依次执行以下检查确认环境可用。

1. 验证NPU可用

   ```shell
   python -c "import torch; import torch_npu; print(torch.npu.is_available()); print(torch.npu.device_count())"
   ```

   预期输出`True`及可见的NPU数量。

2. 验证TransformerEngineNPU导入与版本

   ```shell
   python -c "import transformer_engine; print(transformer_engine.__version__)"
   ```

   预期输出`2.17.1`。若环境中曾安装过原生版本，可执行`python -c "import transformer_engine; print(transformer_engine.__file__)"`确认导入路径指向TransformerEngineNPU源码目录。

3. 验证FP8可用性

   ```shell
   python -c "from transformer_engine.pytorch import is_fp8_available; print(is_fp8_available())"
   ```

   预期输出`True`。若返回`False`，可携带`return_reason=True`参数获取具体原因。

## 卸载TransformerEngineNPU

如需卸载TransformerEngineNPU，执行以下命令：

```shell
pip uninstall -y transformer_engine #注意：TransformerEngineNPU的包名为transformer_engine
```
