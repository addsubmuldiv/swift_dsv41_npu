<!--
Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
See LICENSE in the repository root for license information.
-->

# 低精度训练

[仓库首页](../../../../README.md)

- [低精度训练入门](introduction.md)
  - [BF16 与 FP16 训练](introduction.md#bf16-与-fp16-训练)
  - [主权重](introduction.md#主权重)
  - [策略与量化上下文](introduction.md#策略与量化上下文)
  - [在同一模型中使用多个策略](introduction.md#在同一模型中使用多个策略)
  - [使用 8 位或 4 位格式的混合精度训练](introduction.md#使用-8-位或-4-位格式的混合精度训练)
  - [低精度与高精度运算的边界](introduction.md#低精度与高精度运算的边界)
  - [Linear 的前向与反向数据流](introduction.md#linear-的前向与反向数据流)
- [FP8 Current Scaling](FP8_Current_Scaling.md)
  - [FP8 数据类型](FP8_Current_Scaling.md#fp8-数据类型)
  - [缩放因子](FP8_Current_Scaling.md#缩放因子)
  - [转置处理](FP8_Current_Scaling.md#转置处理)
  - [分布式训练](FP8_Current_Scaling.md#分布式训练)
- [MXFP8](MXFP8.md)
  - [数据格式](MXFP8.md#数据格式)
  - [转置处理](MXFP8.md#转置处理)
  - [分布式训练](MXFP8.md#分布式训练)
- [低精度训练的性能考量](performance_considerations.md)
  - [处理转置](performance_considerations.md#处理转置)
  - [内存占用](performance_considerations.md#内存占用)
  - [融合层](performance_considerations.md#融合层)
  - [分布式训练](performance_considerations.md#分布式训练)
  - [与 MindSpeed 的配合](performance_considerations.md#与-mindspeed-的配合)
