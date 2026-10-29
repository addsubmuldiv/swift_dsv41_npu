# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Feature doing nothing, used for testing purposes."""

from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.api import TEConfigAPIMapper


_inspect_tensor_enabled_call_count = 0
_inspect_tensor_call_count = 0
_inspect_tensor_calls = []


@Registry.register_feature(namespace="transformer_engine")
class TestDummyFeature(TEConfigAPIMapper):
    """Test-only feature that invokes ``inspect_tensor`` without modifying tensors."""

    @api_method
    def inspect_tensor_enabled(self, config, *_args, **_kwargs):
        """Track enablement checks and optionally disable inspection after the first check."""
        import transformer_engine.debug.features._test_dummy_feature as dummy_feature  # pylint: disable=import-self

        dummy_feature._inspect_tensor_enabled_call_count += 1

        if config.get("inspect_only_once", False):
            return False, None
        return True

    @api_method
    def inspect_tensor(self, _config, layer_name, tensor_name, *_args, **_kwargs):
        """Track tensor inspection calls."""
        import transformer_engine.debug.features._test_dummy_feature as dummy_feature  # pylint: disable=import-self

        dummy_feature._inspect_tensor_call_count += 1
        dummy_feature._inspect_tensor_calls.append((layer_name, tensor_name))
