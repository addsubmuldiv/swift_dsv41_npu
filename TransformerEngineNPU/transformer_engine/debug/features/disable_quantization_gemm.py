# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""DisableQuantizationGEMM feature."""

from nvdlfw_inspect.registry import Registry, api_method


from transformer_engine.debug.features.api import TEConfigAPIMapper


@Registry.register_feature(namespace="transformer_engine")
class DisableQuantizationGEMM(TEConfigAPIMapper):
    """Disable selected quantized GEMMs."""

    @api_method
    def fp8_gemm_enabled(self, config, layer_name: str, gemm: str, iteration: int):
        for key in config:
            if key != "gemm":
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')
        return False, iteration + 1
