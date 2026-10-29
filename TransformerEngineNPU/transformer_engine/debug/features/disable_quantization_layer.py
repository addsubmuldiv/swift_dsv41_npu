# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""DisableQuantizationLayer Feature support for nvidia-dlframework-inspect."""

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method


@Registry.register_feature(namespace="transformer_engine")
class DisableQuantizationLayer:
    """
    Disables all quantized GEMMs in the layer, forcing high-precision execution.

    Works with any quantization format (FP8, NVFP4, etc.).
    """

    def parse_config_and_api(self, config, **kwargs):
        """Determines whether to run the API.

        DisableQuantizationLayer disables every GEMM in a matched layer, so it does not
        require the gemms or gemms_struct fields used by TEConfigAPIMapper.
        """
        return config["enabled"], None

    @api_method
    def fp8_gemm_enabled(self, config, layer_name: str, gemm: str, iteration: int):
        """API call responsible for selecting between high-precision and quantized GEMM execution.

        Note: Method name kept as 'fp8_gemm_enabled' for backward compatibility with the debug API,
        but it applies to all quantization formats (FP8, NVFP4, etc.).
        """
        for key in config:
            if key not in ["enabled", "gemm"]:
                raise ValueError(f'[TORCH INSPECT ERROR] Unexpected key in config: "{key}".')
        # If quantized training, disable quantization for the selected layers if this feature is enabled.
        if debug_api.DEBUG_MANAGER is not None:
            debug_api.log_message("Quantization Disabled", layer_name)

        # If this feature is invoked, then quantized GEMM is disabled (returns to high precision).
        # If not, then default behavior in TransformerEngineAPI
        # is that fp8_gemm() API call returns True.
        return False, iteration + 1
