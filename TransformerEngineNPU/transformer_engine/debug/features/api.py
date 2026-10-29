# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""TransformerEngine Ascend namespace API for nvdlfw_inspect."""

import copy
import warnings

import torch

from nvdlfw_inspect.base import BaseConfigAPIMapper, BaseNamespaceAPI
from nvdlfw_inspect.registry import Registry


from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage


class TEConfigAPIMapper(BaseConfigAPIMapper):
    """Route feature config by GEMM or tensor name."""

    def parse_config_and_api(self, config, **kwargs):
        processed_config = None
        config_copy = copy.deepcopy(config)
        gemm_parsing = kwargs.get("gemm_parsing", False)
        tensor_parsing = kwargs.get("tensor_parsing", False)

        if gemm_parsing:
            processed_config = self._process_transformer_engine_config(config_copy, **kwargs)
        elif tensor_parsing:
            processed_config = self._process_tensor_config(config_copy, kwargs["tensor_name"])

        if not processed_config:
            return False, None

        if "enabled" in processed_config:
            processed_config.pop("enabled")
        return True, processed_config

    def _validate_gemm(self, gemm):
        assert gemm in ["fprop", "wgrad", "dgrad"], (
            f"[NVTORCH INSPECT ERROR] Invalid gemm: {gemm}. It must be one of ['fprop', 'wgrad', 'dgrad']."
        )

    def _process_transformer_engine_config(self, config, **kwargs):
        if "gemms_struct" in config:
            for cfg in config["gemms_struct"]:
                self._validate_gemm(cfg["gemm"])
                if cfg["gemm"] == kwargs["gemm"]:
                    if kwargs.get("tensor_parsing", False):
                        cfg = self._process_tensor_config(cfg, kwargs["tensor_name"])
                        if not cfg:
                            return None
                    cfg_copy = copy.deepcopy(cfg)
                    config.pop("gemms_struct")
                    assert "enabled" not in cfg_copy, (
                        "[NVTORCH INSPECT ERROR] Enabled field should not be part of gemms_struct"
                    )
                    config.update(cfg_copy)
                    return config
            return None
        if "gemms" in config:
            for gemm in config["gemms"]:
                self._validate_gemm(gemm)
            if kwargs["gemm"] in config["gemms"]:
                if kwargs.get("tensor_parsing", False):
                    config = self._process_tensor_config(config, kwargs["tensor_name"])
                    if not config:
                        return None
                config["gemm"] = kwargs["gemm"]
                config.pop("gemms")
                return config
            return None
        raise ValueError(
            "[TORCH INSPECT ERROR] Provide 'gemms_struct: List[Dict]' or "
            "'gemms: List[str]' in the config yaml"
        )


required_kwargs = {
    "fp8_gemm_enabled": ["gemm"],
    "modify_tensor_enabled": ["tensor_name", "gemm"],
    "modify_tensor": ["tensor_name", "gemm"],
    "inspect_tensor": ["tensor_name"],
    "inspect_tensor_postquantize": ["tensor_name"],
    "inspect_tensor_enabled": ["tensor_name"],
    "inspect_tensor_postquantize_enabled": ["tensor_name"],
    "default": ["tensor_name", "gemm"],
}


class TEDefaultFeatures:
    """Default behavior when no TE debug feature is selected."""

    def fp8_gemm_enabled(self, config, layer_name: str, gemm: str, iteration: int):
        return True, None

    def modify_tensor_enabled(
        self,
        config,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        iteration: int,
    ):
        return False, None

    def modify_tensor(
        self,
        config,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        tensor,
        default_quantizer=None,
        iteration=None,
        out=None,
        dtype=None,
    ):
        raise NotImplementedError(
            "modify_tensor_enabled() returned True, modify_tensor() was invoked, but it is not "
            "handled by any API."
        )

    def inspect_tensor_enabled(self, config, layer_name: str, tensor_name: str, iteration: int):
        return False, None

    def inspect_tensor(self, config, layer_name: str, **kwargs):
        return None

    def inspect_tensor_postquantize_enabled(
        self,
        config,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        gemm: str = None,
    ):
        return False, None

    def inspect_tensor_postquantize(self, config, layer_name: str, **kwargs):
        return None


@Registry.register_namespace_api(namespace="transformer_engine")
class TransformerEngineAPI(BaseNamespaceAPI):
    """TE namespace API used by debug features."""

    def __init__(self):
        super().__init__()
        self._default_api_impl = TEDefaultFeatures()
        self._cacheable_api_kwargs_map = {
            "fp8_gemm": ["gemm"],
            "modify_tensor": ["tensor_name", "gemm"],
            "inspect_tensor": ["tensor_name"],
            "inspect_tensor_postquantize": ["tensor_name"],
            "inspect_tensor_enabled": ["tensor_name", "iteration"],
            "inspect_tensor_postquantize_enabled": ["tensor_name", "iteration"],
            "modify_tensor_enabled": ["tensor_name"],
        }

    def is_multiple_feature_invocation_allowed(self, api_name):
        return api_name in {
            "fp8_gemm_enabled",
            "inspect_tensor_enabled",
            "inspect_tensor",
            "inspect_tensor_postquantize_enabled",
            "inspect_tensor_postquantize",
        }

    def input_assertions_hook(self, api_name, **kwargs):
        for kwarg in required_kwargs.get(api_name, required_kwargs["default"]):
            assert kwarg in kwargs, (
                "[NVTORCH INSPECT ERROR] Cannot route API, too ambiguous. "
                f"Provide {kwarg} in {api_name}."
            )

    def output_assertions_hook(self, api_name, ret, **kwargs):
        if api_name in {
            "fp8_gemm_enabled",
            "modify_tensor_enabled",
            "inspect_tensor_enabled",
            "inspect_tensor_postquantize_enabled",
        }:
            assert isinstance(ret, (bool, tuple))
        if api_name in {"inspect_tensor", "inspect_tensor_postquantize"}:
            assert ret is None
        if api_name == "modify_tensor":
            if kwargs.get("out") is not None:
                assert ret is None
            else:
                assert isinstance(ret, (torch.Tensor, QuantizedTensorStorage))
                if kwargs.get("dtype") is not None:
                    assert isinstance(ret, torch.Tensor)
                    assert ret.dtype == kwargs["dtype"]

    def routing_condition(self, api_name, config, _, feature_obj, **kwargs):
        tensor_parsing = "tensor_name" in required_kwargs[api_name]
        gemm_parsing = "gemm" in required_kwargs[api_name]
        status, modified_config = feature_obj.parse_config_and_api(
            config,
            gemm_parsing=gemm_parsing,
            tensor_parsing=tensor_parsing,
            **kwargs,
        )
        return status, modified_config

    def call_feature(self, call, feat_config, layer_name, **kwargs):
        """Trim newer kwargs for older feature implementations."""
        kwargs_copy = kwargs.copy()
        if call.__name__ == "inspect_tensor":
            for kwarg in (
                "quantizer",
                "columnwise_quantized_tensor",
                "rowwise_quantized_tensor",
                "tp_size",
            ):
                if kwarg not in call.__code__.co_varnames:
                    kwargs_copy.pop(kwarg, None)
        elif call.__name__ == "inspect_tensor_postquantize":
            warnings.warn(
                "inspect_tensor_postquantize is deprecated, use inspect_tensor instead.",
                DeprecationWarning,
            )
            if "tp_size" not in call.__code__.co_varnames:
                kwargs_copy.pop("tp_size", None)
        return call(feat_config, layer_name, **kwargs_copy)

    def handle_multi_feature_output(
        self, api_name, multi_feature_outputs, features_to_invoke, **kwargs
    ):
        """Combine outputs for APIs where multiple features may run."""
        if "enabled" in api_name:
            all_ret_tuple = all(isinstance(output, tuple) for output in multi_feature_outputs)
            if all_ret_tuple:
                run_current = any(output[0] for output in multi_feature_outputs)
                next_iter = None
                for output in multi_feature_outputs:
                    if next_iter is None:
                        next_iter = output[1]
                    elif output[1] is not None:
                        next_iter = min(next_iter, output[1])
                return run_current, next_iter
            return any(multi_feature_outputs), None
        return super().handle_multi_feature_output(
            api_name, multi_feature_outputs, features_to_invoke, **kwargs
        )

    def step(self):
        from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS

        return STATS_BUFFERS.log_stats()

    def end_debug(self):
        TEDebugState._reset()
