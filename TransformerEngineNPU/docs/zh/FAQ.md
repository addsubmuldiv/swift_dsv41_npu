# FAQ

本文说明TransformerEngineNPU（TENPU）的能力边界问题（NVFP4支持）。

## 是否支持NVFP4

### 问题现象

使用NVFP4策略（如`recipe.NVFP4BlockScaling`）或将GPU训练脚本中的NVFP4量化配置迁移到昇腾环境时，量化不生效或报不支持。

### 问题根因

TENPU当前不支持NVFP4低精度量化，`is_nvfp4_available()`恒返回`False`。该限制为软件栈能力边界，与版本配套无关。

### 解决方案

在昇腾上改用其他策略（如`DelayedScaling`、`Float8BlockScaling`、`MXFP8BlockScaling`等，完整列表见[API列表](api/api_list.md#fp8低精度训练)）。
