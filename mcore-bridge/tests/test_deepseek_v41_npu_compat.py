# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mcore_bridge.bridge.gpt_bridge import GPTBridge
from mcore_bridge.model.gpts.deepseek_v4 import DeepseekV4Bridge
from mcore_bridge.model.mm_gpts import deepseek_v41
from mcore_bridge.model.mm_gpts.deepseek_v41 import DeepseekV41Bridge
from mcore_bridge.model.modules import engram as engram_adapter


@pytest.mark.parametrize('scale_rows', [2, 64])
@pytest.mark.parametrize('as_bytes', [False, True])
def test_engram_scales(scale_rows, as_bytes):
    weight = (torch.arange(4096).reshape(64, 64) % 8).to(torch.float8_e4m3fn)
    scales = torch.arange(1, scale_rows * 2 + 1, dtype=torch.float32).reshape(scale_rows, 2)
    expected = weight.float() * scales.repeat_interleave(64 // scale_rows, dim=0).repeat_interleave(32, dim=1)
    if as_bytes:
        weight = weight.view(torch.uint8)
    actual = DeepseekV41Bridge._dequantize_engram_rows(weight, scales)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_engram_unquantized_weight():
    weight = torch.ones((4, 4), dtype=torch.bfloat16)
    assert DeepseekV41Bridge._dequantize_engram_rows(weight, None) is weight


@pytest.mark.parametrize('weight_shape,scale_shape', [
    ((64, ), (2, 2)),
    ((64, 64), (2, )),
    ((64, 64), (3, 2)),
    ((64, 64), (2, 3)),
    ((64, 64), (0, 2)),
    ((0, 64), (2, 2)),
])
def test_engram_rejects_invalid_scales(weight_shape, scale_shape):
    with pytest.raises(ValueError, match='Invalid Engram FP8'):
        DeepseekV41Bridge._dequantize_engram_rows(torch.empty(weight_shape), torch.empty(scale_shape))


@pytest.mark.parametrize('native,decorated', [(False, False), (True, False), (True, True)])
def test_hybrid_engram_injected_once(monkeypatch, native, decorated):

    class Wrapper(torch.nn.Module):

        def forward(self, hidden):
            return hidden

    class LegacyWrapper(Wrapper):
        pass

    def native_forward(self, hidden):
        return self.inner_layer._maybe_apply_engram(hidden)

    if native:
        if decorated:

            @wraps(native_forward)
            def forward(self, hidden):
                return native_forward(self, hidden)

            Wrapper.forward = forward
        else:
            Wrapper.forward = native_forward
    monkeypatch.setattr(deepseek_v41, 'HyperConnectionHybridLayer', Wrapper)
    monkeypatch.setattr(deepseek_v41, 'DeepseekV41HyperConnectionHybridLayer', LegacyWrapper)
    layer = Wrapper()
    layer.inner_layer = SimpleNamespace(engram=object())
    plain_layer = Wrapper()
    plain_layer.inner_layer = SimpleNamespace(engram=None)
    loader = object.__new__(deepseek_v41.DeepseekV41Loader)
    loader.config = SimpleNamespace(enable_hyper_connections=True)
    with patch.object(deepseek_v41.inspect, 'getsource', side_effect=OSError('source unavailable')):
        loader._rewrap_engram_hyper_connection_layers(
            SimpleNamespace(decoder=SimpleNamespace(layers=[layer, plain_layer])))
    assert type(layer) is (Wrapper if native else LegacyWrapper)
    assert type(plain_layer) is Wrapper


@pytest.mark.skipif(not engram_adapter.has_native_engram(), reason='Native Engram is unavailable')
@pytest.mark.parametrize('sequence_argument', [False, True])
def test_engram_parallelism_api(monkeypatch, sequence_argument):
    calls = []
    if sequence_argument:

        def validate(self, config, sequence_length):
            calls.append((config.context_parallel_size, sequence_length))
    else:

        def validate(self, config):
            calls.append((config.context_parallel_size, None))

    monkeypatch.setattr(engram_adapter.EngramConfig, '_validate_parallelism', validate)
    config = object.__new__(engram_adapter.DeepseekV41EngramConfig)
    config._validate_parallelism(SimpleNamespace(context_parallel_size=2), 8)
    assert calls == [(1, 4 if sequence_argument else None)]


@pytest.mark.parametrize('language_model_only', [True, False])
def test_image_marker_loading(language_model_only):
    bridge = object.__new__(DeepseekV41Bridge)
    bridge.config = SimpleNamespace(language_model_only=language_model_only)
    bridge._set_state_dict = Mock()
    state_dict = {}
    with patch.object(DeepseekV4Bridge, '_convert_pre_process', return_value={}):
        bridge._convert_pre_process(None, state_dict, '', True)
    if language_model_only:
        bridge._set_state_dict.assert_not_called()
    else:
        assert [call.args[1] for call in bridge._set_state_dict.call_args_list
                ] == ['visual.image_start', 'visual.image_end', 'visual.image_newline']
