"""Exercise the real Engram startup validator without allocating a model or process groups."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pytest

# The adaptor parses CLI arguments during imports; pytest flags are not training flags.
with patch.object(sys, 'argv', sys.argv[:1]):
    import megatron_adaptor  # Apply NPU compatibility before importing MCore.
    from megatron.core.models.engram.config import EngramConfig, TOKENIZER_MAP_FORMAT, TOKENIZER_MAP_VERSION
    from mcore_bridge.model.modules.engram import build_deepseek_v41_engram_config


@pytest.fixture
def engram(tmp_path):
    artifact = tmp_path / 'map.json'
    artifact.write_text(json.dumps({
        'format': TOKENIZER_MAP_FORMAT, 'version': TOKENIZER_MAP_VERSION,
        'source_vocab_size': 8, 'compressed_vocab_size': 8,
        'pad_token_id': 0, 'compressed_pad_token_id': 0,
        'max_ngram_order': 3, 'hash_seed': 0, 'layer_ids': [1],
        'layer_multipliers': {'1': [11, 13, 15]}, 'remap': list(range(8)),
    }))
    return build_deepseek_v41_engram_config(
        placement_layer_ids=(3,), hash_layer_ids=(1,), global_vocab_sizes=(17, 19),
        max_ngram_order=3, num_hash_heads=1, memory_dim=4, kernel_size=1,
        hash_seed=0, boundary_token_id=0, tokenizer_map_path=str(artifact),
    )


def transformer(**overrides):
    values = dict(num_layers=80, pipeline_model_parallel_size=4,
                  virtual_pipeline_model_parallel_size=None, context_parallel_size=1,
                  tensor_model_parallel_size=1, expert_tensor_parallel_size=1,
                  expert_model_parallel_size=16, sequence_parallel=False,
                  recompute_granularity='full', fp16=False, fp4=None,
                  transformer_impl='transformer_engine', cuda_graph_impl='none')
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize('pp', [1, 4])
def test_v41_engram_startup_with_full_recompute(engram, pp):
    config = transformer(pipeline_model_parallel_size=pp)
    engram.validate_startup(config, expected_tokenizer_vocab_size=8)
    assert config.pipeline_model_parallel_size == pp
    assert config.recompute_granularity == 'full'


def test_generic_engram_keeps_pipeline_guard(engram):
    with pytest.raises(ValueError, match='PP=1'):
        EngramConfig._validate_parallelism(engram, transformer())


@pytest.mark.parametrize('overrides, message', [
    ({'tensor_model_parallel_size': 2}, 'TP=1'),
    ({'sequence_parallel': True}, 'TP=1'),
    ({'expert_tensor_parallel_size': 2}, 'expert_tensor_parallel_size'),
])
def test_v41_preserves_other_parallelism_guards(engram, overrides, message):
    with pytest.raises(ValueError, match=message):
        engram.validate_startup(transformer(**overrides), expected_tokenizer_vocab_size=8)


def test_swift_keeps_engram_token_ids_on_middle_pipeline_stages(monkeypatch):
    import torch
    from swift.megatron.trainers import utils

    monkeypatch.setattr(utils, 'get_current_device', lambda: 'cpu')
    monkeypatch.setattr(utils.mpu, 'is_pipeline_last_stage', lambda **kwargs: False)
    tokens = torch.tensor([[1, 2, 3]])
    positions = torch.tensor([[0, 1, 2]])
    batch = utils.get_batch_on_this_pp_rank(
        SimpleNamespace(task_type='causal_lm', pipeline_model_parallel_size=4),
        {'input_ids': tokens, 'position_ids': positions, 'labels': tokens.clone()},
    )
    assert torch.equal(batch['input_ids'], tokens)
    assert torch.equal(batch['position_ids'], positions)
    assert batch['labels'] is None
