"""Static checks for the real 64-rank launcher; no model weights or live groups are created."""
import inspect
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# The adaptor must not interpret pytest CLI flags as model config fields.
with patch.object(sys, 'argv', sys.argv[:1]):
    import megatron_adaptor
    import torch
    from transformers import HfArgumentParser
    from megatron.core import parallel_state
    from megatron.core.models.engram.distributed_embedding import get_contiguous_row_range
    from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateAdapter
    from megatron.core.models.hybrid.hybrid_state import build_hybrid_state_pipeline_plan
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.spec_utils import build_module
    from mcore_bridge.model import get_model_meta
    from mcore_bridge.model.mm_gpts.deepseek_v41 import DeepseekV41HybridStackModel
    from swift.megatron.arguments import MegatronSftArguments
    from swift.megatron.arguments.parallel_args import validate_parallel_args
    from swift.megatron.model.utils import get_mcore_model_config
    from swift.megatron.trainers.batch_sampler import MegatronPretrainingRandomSampler
    from swift.model import get_model_info_meta

ROOT = Path(__file__).resolve().parents[1]
WORLD = 64
ORDER = inspect.signature(parallel_state.initialize_model_parallel).parameters['order'].default


def rank_groups():
    dense = parallel_state.RankGenerator(tp=1, cp=1, ep=1, dp=16, pp=4, order=ORDER)
    expert = parallel_state.RankGenerator(tp=1, cp=1, ep=16, dp=1, pp=4, order=ORDER)
    return {'tp': dense.get_ranks('tp'), 'cp': dense.get_ranks('cp'),
            'dp': dense.get_ranks('dp'), 'pp': dense.get_ranks('pp'),
            'ep': expert.get_ranks('ep'), 'etp': expert.get_ranks('tp'),
            'expert_dp': expert.get_ranks('dp'), 'expert_pp': expert.get_ranks('pp')}


@pytest.mark.parametrize('name,size', [('tp', 1), ('cp', 1), ('dp', 16), ('pp', 4),
                                      ('ep', 16), ('etp', 1), ('expert_dp', 1)])
def test_all_64_ranks_form_disjoint_groups(name, size):
    groups = rank_groups()[name]
    assert len(groups) == WORLD // size
    assert all(len(group) == size and len(set(group)) == size for group in groups)
    assert sorted(itertools.chain.from_iterable(groups)) == list(range(WORLD))


def test_experts_stay_on_node_and_pipeline_keeps_local_rank():
    groups = rank_groups()
    assert groups['ep'] == groups['dp'] == [list(range(i * 16, (i + 1) * 16)) for i in range(4)]
    assert groups['pp'] == groups['expert_pp'] == [[i + 16 * j for j in range(4)] for i in range(16)]
    assert 384 // 16 == 24 and 384 % 16 == 0


@pytest.fixture(scope='module')
def full_config(tmp_path_factory):
    model, token_map = os.getenv('DSV41_MODEL_DIR'), os.getenv('DSV41_ENGRAM_MAP')
    if not model or not token_map:
        pytest.skip('Set DSV41_MODEL_DIR and DSV41_ENGRAM_MAP for the full-model static audit.')
    environment = {**os.environ, 'MODEL_DIR': model, 'ENGRAM_MAP': token_map,
                   'OUTPUT_DIR': str(tmp_path_factory.mktemp('topology')), 'DRY_RUN': '1',
                   'PYTHON': sys.executable}
    environment.pop('MASTER_ADDR', None)
    result = subprocess.run(['bash', str(ROOT / 'examples/train_full_lora_4nodes.sh'), '0'],
                            cwd=ROOT, env=environment, capture_output=True, text=True, check=True)
    command = shlex.split(result.stdout.splitlines()[-1])
    args, extra = HfArgumentParser(MegatronSftArguments).parse_known_args(command[4:])
    assert not extra, extra
    info, metadata = get_model_info_meta(model, model_type='deepseek_v41', torch_dtype=torch.bfloat16,
                                        use_hf=False)
    hf = metadata.loader(info, metadata, load_model=False, model_kwargs={}).get_config(model)
    args.megatron_model_meta = get_model_meta('deepseek_v41')
    args.model_dir, args.torch_dtype = model, torch.bfloat16
    args.attention_backend = AttnBackend.unfused
    args.megatron_extra_kwargs = json.loads(args.megatron_extra_kwargs)
    # Match Swift's ordinary-PP normalization without initializing torch.distributed.
    args.align_param_gather = False
    validate_parallel_args(args, world_size=WORLD)
    # Config construction otherwise creates a bridge which requires initialized groups.
    # Only defer that allocation; execute actual MCore/Adaptor/bridge config validators.
    with patch.object(args.megatron_model_meta.bridge_cls, '__init__', return_value=None), \
            patch.object(parallel_state, 'get_expert_model_parallel_world_size', return_value=16):
        config = get_mcore_model_config(args, hf)
    loader = args.megatron_model_meta.loader.__new__(args.megatron_model_meta.loader)
    loader.config = config
    loader._hybrid_config = loader._build_hybrid_config()
    pattern = DeepseekV41HybridStackModel._segment_main_pattern(loader._hybrid_config)
    spec = loader.get_transformer_layer_spec()
    return SimpleNamespace(args=args, config=config, loader=loader, pattern=pattern, spec=spec)


def test_actual_launcher_and_full_layer_specs(full_config):
    a, c, loader = full_config.args, full_config.config, full_config.loader
    assert (a.tensor_model_parallel_size, a.context_parallel_size,
            a.pipeline_model_parallel_size, a.expert_model_parallel_size,
            a.expert_tensor_parallel_size) == (1, 1, 4, 16, 1)
    assert c.num_layers == 40 and loader._hybrid_config.num_layers == 80
    assert full_config.pattern.split('|') == ['DE' * 10] * 4
    assert c.engram_layer_ids == [1, 14] and c.num_moe_experts == 384
    engram = full_config.spec.submodules.dsa_layer.submodules.engram.params['engram_config']
    assert engram.layer_ids == (3, 29)
    assert not c.mtp_num_layers and not c.dspark_num_layers
    assert a.train_iters == 10 and a.micro_batch_size == 1 and a.global_batch_size == 64
    assert 64 // (1 * 16) == 4
    assert not a.padding_free and not a.save_safetensors and not a.merge_lora
    assert not c.overlap_p2p_comm and c.batch_p2p_comm


def pipeline_specs(bundle, length):
    config = bundle.loader._hybrid_config
    chunks = build_hybrid_state_pipeline_plan(config, bundle.pattern, pp_size=4)
    result = []
    for chunk in chunks:
        # Supply placement metadata only, with no send/recv or process-group implementation.
        pp = SimpleNamespace(size=lambda: 4, rank=lambda rank=chunk.pp_rank: rank)
        pg = SimpleNamespace(cp=None, pp=pp)
        kwargs = dict(config=config, layer_type_list=list(chunk.layer_pattern),
                      pp_layer_offset=chunk.layer_offset, pre_process=chunk.pp_rank == 0,
                      post_process=chunk.pp_rank == 3, is_mtp_layer=False, pg_collection=pg)
        components = tuple(build_module(declaration, **kwargs)
                           for declaration in bundle.spec.submodules.state_components)
        adapter = HybridStateAdapter(**kwargs, components=components,
                                     hidden_size=4 * config.hidden_size, hidden_dtype=torch.bfloat16)
        assert adapter.configure_distributed_pipeline(bundle.pattern, pp) is not None
        result.append(adapter.pipeline_payload_spec(length, 1))  # Creates meta tensors only.
    return result


@pytest.mark.parametrize('length', [24, 128])
def test_all_pipeline_boundary_schemas_match(full_config, length):
    specs = pipeline_specs(full_config, length)
    assert specs[0][0] is None and specs[-1][1] is None
    for stage in range(3):
        send, receive = specs[stage][1], specs[stage + 1][0]
        assert send.fingerprint == receive.fingerprint
        assert send.tensor_specs[0].shape == (length, 1, 20480)
        assert all(s.dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64)
                   for s in send.tensor_specs if s.present)


def test_engram_row_partitions_cover_each_prime_table(full_config):
    engram = full_config.loader._get_engram_config()
    for layer in engram.layer_ids:
        for rows in engram.table_sizes(layer):
            shards = [get_contiguous_row_range(rows, rank, 16) for rank in range(16)]
            assert shards[0][0] == 0 and shards[-1][1] == rows
            assert all(shards[i][1] == shards[i + 1][0] for i in range(15))
            assert max(end - start for start, end in shards) - min(end - start for start, end in shards) <= 1


def test_pipeline_peers_receive_same_samples_across_epoch_boundaries(full_config):
    a = full_config.args
    dataset = [json.loads(line) for line in Path(a.dataset[0]).read_text().splitlines()]
    assert len(dataset) >= 16
    samples = {}
    for rank in range(64):
        sampler = MegatronPretrainingRandomSampler(
            dataset, total_samples=len(dataset), consumed_samples=0, micro_batch_size=1,
            data_parallel_rank=rank % 16, data_parallel_size=16, data_sharding=a.data_sharding,
            shuffle=a.dataset_shuffle, seed=a.data_seed)
        def cycle():
            while True:
                yield from sampler
        samples[rank] = list(itertools.islice(cycle(), 40))
    for group in rank_groups()['pp']:
        assert all(samples[rank] == samples[group[0]] for rank in group)
    assert all(len({samples[rank][micro][0] for rank in range(16)}) == 16 for micro in range(40))


def test_write_static_audit_report(full_config):
    target = os.getenv('DSV41_TOPOLOGY_REPORT')
    if not target:
        return
    specs = pipeline_specs(full_config, 128)
    report = {
        'validation_kind': 'static; no training weights or live communication groups allocated',
        'world_size': WORLD, 'nodes': 4, 'npu_per_node': 16, 'rank_order': ORDER,
        'sizes': {'tp': 1, 'cp': 1, 'pp': 4, 'ep': 16, 'etp': 1, 'dp': 16, 'expert_dp': 1},
        'rank_groups': rank_groups(), 'experts_per_rank': 24, 'microbatches_per_step': 4,
        'stages': [{'node_rank': stage, 'global_ranks': [stage * 16, stage * 16 + 15],
                    'hf_layers': [stage * 10, stage * 10 + 9],
                    'hybrid_indices': [stage * 20, stage * 20 + 19],
                    'engram_hf_layer_ids': [i for i in [1, 14] if stage * 10 <= i < (stage + 1) * 10]}
                   for stage in range(4)],
        'pipeline_boundaries': [
            {'boundary_id': specs[stage][1].boundary_id,
             'send_receive_fingerprints_match': specs[stage][1].fingerprint == specs[stage + 1][0].fingerprint,
             'tensors': [{'name': field.name, 'shape': field.shape, 'dtype': str(field.dtype)}
                         for field in specs[stage][1].tensor_specs if field.present]}
            for stage in range(3)],
        'checks': ['native_rank_generation_all_64_ranks', 'real_config_validators', 'full_layer_specs',
                   'Engram_startup_and_prime_row_shards', 'CSA2_mHC_payload_schema_24_and_128_tokens',
                   'same_PP_inputs_across_10_steps_and_epoch_boundaries'],
        'multi_node_hccl_tested': False, 'full_model_training_tested': False,
    }
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
