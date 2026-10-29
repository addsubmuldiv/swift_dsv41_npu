# 版本说明

## 关键特性

## 版本配套说明

### 产品版本信息

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>MegatronAdaptor</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>core_r0.19.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>在研版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>2026年9月</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明

**表 1** MegatronAdaptor软件版本配套表

|MegatronAdaptor代码分支名称|CANN版本|TorchNPU版本|Python版本|PyTorch版本|
|--|--|--|--|--|
|core_r0.19.0|9.2.0|26.2.0|Python3.12|2.10.0|
|core_r0.18.0|9.2.0|26.2.0|Python3.12|2.10.0|

>[!NOTE]
>
>用户可根据需要选择MegatronAdaptor代码分支下载源码并进行安装。

## 版本使用注意事项

TransformerEngineNPU 和原生 TransformerEngine 的库模块名相同，为避免冲突，同一环境内请勿同时安装两者，仅保留 TransformerEngineNPU 即可。

## 更新说明

### 新增特性

内置特性覆盖基础运行环境适配、Megatron 基础适配、TransformerEngineNPU 适配、NPU Profiling、NPU 数据 Dump 精度对照、NPU 确定性计算等。

### 删除特性

无

### 接口变更说明

无

### 已解决问题

无

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[MegatronAdaptor快速入门](quick_start.md)》|介绍基于MegatronAdaptor如何实现Megatron-LM在昇腾设备上的高效运行。|-|
|《[MegatronAdaptor安装指导](install_guide.md)》|指导如何在NPU上基于PyTorch框架完成MegatronAdaptor的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件安装的完整安装流程，帮助用户快速搭建大模型分布式训练环境。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

无

### 漏洞修补列表

无
