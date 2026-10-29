# 版本说明

## 关键特性

TransformerEngineNPU是原生TransformerEngine在昇腾（Ascend）设备上的适配版本，为大模型训练与推理提供低精度计算加速；提供`Linear`、`GroupedLinear`、`LayerNormLinear`、`LayerNormMLP`、`DotProductAttention`等高性能基础层模块与`autocast`量化上下文，支持FP8（延迟/当前/块级缩放）、MXFP8、HIF8、W4A8、W4A4-MXFP4等低精度策略，以及MoE路由融合、CPU Offload、Swap上下文并行注意力与`FusedAdam`优化器等特性。

## 版本配套说明

### 产品版本信息

| 项目 | 说明 |
|------|------|
| 产品名称 | MindSpeed |
| 产品版本 | 26.2.0 |
| 版本类型 | Release版本 |
| 组件名称 | TransformerEngineNPU |
| 发布时间 | 2026年10月 |
| 维护周期 | 6个月 |

### 相关产品版本配套说明

**表 1** TransformerEngineNPU软件版本配套表

|TransformerEngineNPU代码分支名称|MegatronAdaptor版本|CANN版本|TorchNPU版本|Python版本|PyTorch版本|Triton-Ascend版本|
|--|--|--|--|--|--|--|
|main（在研版本）|main（在研版本）|在研版本|在研版本|Python3.12|2.10.0|3.6.0|
|26.2.0_release_v2.17|26.2.0_core_r0.18.0|9.2.0|26.2.0|Python3.12|2.10.0|3.6.0|

>[!NOTE]
>
> - 用户可根据需要选择TransformerEngineNPU码分支下载源码并进行安装。
> - Triton-Ascend版本与CANN版本强绑定，Triton-Ascend的使用应该与CANN版本一一对应，详见[Triton-Ascend兼容性](https://triton-ascend.readthedocs.io/zh-cn/latest/release_note.html#id13)。

**表 2** TransformerEngineNPU与原生TransformerEngine版本对应表

| TransformerEngineNPU | TransformerEngine |
|----------------------|-------------------|
| main | main |
| 26.2.0_release_v2.17 | release_v2.17 |

### 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 3**  TransformerEngineNPU与TorchNPU版本兼容

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">TransformerEngineNPU</th>
      <th colspan="1">TorchNPU版本</th>
    </tr>
    <tr>
      <th>26.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.X_release_v2.17</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**表 4**  TransformerEngineNPU与CANN版本兼容

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">TransformerEngineNPU</th>
      <th colspan="1">CANN版本</th>
    </tr>
    <tr>
      <th>9.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.X_release_v2.17</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## 版本使用注意事项

- TransformerEngineNPU和原生TransformerEngine的库模块名相同，为避免冲突，同一环境内请勿同时安装两者，仅保留TransformerEngineNPU即可。
- TransformerEngineNPU不支持NVFP4低精度量化。

## 更新说明

### 新增特性

- 适配MegatronAdaptor与Mcore 0.18.0版本。
- Attention：新增Swap上下文并行注意力及配套Swap Flash Attention实现。
- MoE：新增MoE路由融合接口（`fused_topk_with_score_function`、`fused_moe_aux_loss`等）；`GroupedLinear`支持权重梯度融合累积（`fuse_wgrad_accumulation`）；分组矩阵乘统一调度重构。
- 低精度训练：新增HIF8独立延迟缩放策略、W4A8权重量化策略、W4A4-MXFP4；块级FP8能力增强，支持分片权重重量化与批量主权重重量化；支持量化张量checkpoint序列化；新增`CustomRecipe`自定义策略；新增FP8 amax/scale更新的融合Triton kernel。
- 优化器：`FusedAdam`支持一阶动量`--exp-avg-dtype`和二阶动量`--exp-avg-sq-dtype`可独立配置为 FP32、FP16、BF16 或 FP8，支持混合精度组合。
- 基础模块：`LayerNormLinear`/`LayerNormMLP`重构为`WeightGradStore`延迟权重梯度架构；支持`gradient_accumulation_fusion`；支持`backward_override`与`save_original_input`。
- 调试：对齐上游TransformerEngine的精度调试能力。

### 删除特性

无

### 接口变更说明

无

### 已解决问题

- 修复MXFP8训练显存泄漏问题。
- 修复块级FP8在checkpoint恢复后精度不一致的问题。
- 修复`LayerNormMLP`与`Linear`在无`main_grad`场景下的报错。
- 修复`FusedAdam` group step在`state_dict`不为int时的问题。

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    TransformerEngineNPU以源码方式安装，版本升级需重新构建安装并重启训练任务，期间业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[TransformerEngineNPU简介](introduction.md)》|介绍TransformerEngineNPU的定位、功能特性与目录结构。|-|
|《[TransformerEngineNPU安装指导](install_guide.md)》|指导如何基于源码完成TransformerEngineNPU及其配套软件的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN等基础软件安装的完整流程。|-|
|《[TransformerEngineNPU API列表](api/api_list.md)》|列出所有公开接口，对比其与原生TransformerEngine的差异，并给出环境变量说明。|-|
|《[TransformerEngineNPU FAQ](FAQ.md)》|说明NVFP4能力边界问题。|-|

## 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-10-07 08:00:00.0|2026-10-08|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-10-08 10:11:00|2026-10-08|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101645|2026-10-08|无病毒，无恶意|

## 漏洞修补列表

无

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-10-15 | 第一次正式发布。 |
