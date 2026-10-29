# TransformerEngineNPU简介

## 概述

TransformerEngine是NVIDIA提供的一个用于加速Transformer模型训练和推理的库，而TransformerEngineNPU（下文简称TENPU）则是其在昇腾（Ascend）设备上的适配版本，旨在为昇腾设备上的大模型训练和推理提供高效的计算支持。

TENPU与原生TransformerEngine保持**相同的库名、模块路径和接口签名**（`import transformer_engine.*`），已在CUDA设备上运行的模型代码无需修改import语句，可直接切换到昇腾后端。在原生TransformerEngine之上，TENPU还提供了部分昇腾专属扩展（如HIF8延迟缩放、W4A8权重量化、swap注意力等）。

## TransformerEngineNPU架构

TransformerEngineNPU整体架构自上而下分为四个层次：

- 接口层：保持与原生TransformerEngine相同的库名、模块路径与接口签名，对外提供基础层模块（`Linear`、`LayerNorm`、`DotProductAttention`等）与量化上下文（`autocast`），业务代码无需修改import语句即可切换至昇腾后端；
- 量化框架层：由量化策略（recipe）、量化管理（quantization）与量化张量（tensor）组成，为FP8等低精度训练提供缩放策略、量化状态管理与量化数据结构；
- 算子实现层：对上承接模块层的算子语义，对下封装底层kernel，包括ops算子封装（含算子融合fuser）与基于Triton-Ascend的Triton实现；
- 底层依赖：由TorchNPU、CANN与Triton-Ascend构成的昇腾软件栈，提供设备侧执行能力。

## 功能模块

TransformerEngineNPU由六大功能模块组成，分别为：高性能基础层模块、FP8/低精度训练、MoE支持、优化器与多Tensor算子、训练并行与显存优化以及数值调试。

各模块的接口清单与说明详见[API列表](api/api_list.md)。
