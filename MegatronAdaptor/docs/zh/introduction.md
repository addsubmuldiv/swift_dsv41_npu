# 简介

## 概述

Megatron-LM是业界主流的大模型分布式训练框架，提供了张量并行、流水线并行等丰富的并行策略，被广泛应用于大模型预训练场景。然而Megatron-LM原生面向GPU设计，直接在昇腾设备上运行面临算子适配、并行通信、计算加速等一系列挑战。

MegatronAdaptor是专为Megatron-LM在华为昇腾设备NPU上运行的基础适配库，为上层模型侧框架提供运行基础。通过非侵入式的Patch机制，MegatronAdaptor在不修改Megatron-LM源码的前提下，完成昇腾设备的算子适配、兼容性修复与加速特性启用，使用户的大模型训练业务能够快速迁移至昇腾设备，确保开箱可用。

## 架构

MegatronAdaptor整体方案架构分为三个层次：

- 训练框架层：原生Megatron-LM，提供模型定义、分布式并行策略与训练流程；
- 适配层：MegatronAdaptor，通过特性注册与Patch机制，完成Megatron-LM在昇腾设备上的适配与增强；
- 加速层：[TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)，为昇腾设备提供Transformer加速算子库。

## 功能特性

MegatronAdaptor采用"特性 + Patch"的解耦设计，核心包含两个组件：

- 特性管理（FeaturesManager）：
  每个特性继承`MindSpeedFeature`基类独立注册，功能内聚、相互解耦；
  特性可根据训练参数自动判断是否需要启用，并支持参数注册与校验，可方便地进行二次开发。

- Patch管理（PatchesManager）：
  采用非侵入式Patch方式对Megatron-LM进行适配，无需修改Megatron-LM源码；
  Patch按特性维度独立管理，方便版本之间的快速迭代，也可轻便快捷地应用至自有框架。
