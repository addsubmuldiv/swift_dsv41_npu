"""Create a four-layer view of local weights; no weight shards are copied."""
import argparse
import ast
import copy
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
source = Path(args.model).resolve()
out = Path(args.output).resolve()
assert source != out, 'Use a separate output directory'
out.mkdir(parents=True, exist_ok=True)
config = json.loads((source / 'config.json').read_text())
text = config['text_config']
original = copy.deepcopy(text)
layers = 4
assert text['num_hidden_layers'] >= layers
text['num_hidden_layers'] = layers
text['compress_ratios'] = text['compress_ratios'][:layers]
for key in ('kv_source_layer_ids', 'index_source_layer_ids', 'engram_layer_ids'):
    text[key] = [i for i in text[key] if i < layers]
text['engram_num_embeddings'] = [n for i, n in zip(original['engram_layer_ids'], original['engram_num_embeddings']) if i < layers]
candidate = text.get('candidate_source_layer_id')
if candidate is not None and candidate >= layers:
    text.update(candidate_source_layer_id=None, candidate_topk_blocks=0, candidate_block_size=0)
text['num_nextn_predict_layers'] = 0
for key in list(text):
    if key.startswith('dspark_'):
        del text[key]
(out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
for path in source.iterdir():
    if path.name.startswith('tokenizer') or path.name in ('encoding', 'inference', 'generation_config.json', 'preprocessor_config.json', 'processor_config.json') or path.suffix == '.safetensors':
        target = out / path.name
        if target.is_symlink() and target.resolve() != path:
            target.unlink()
        if not target.exists():
            target.symlink_to(path, target_is_directory=path.is_dir())
index = json.loads((source / 'model.safetensors.index.json').read_text())
assert any(key.startswith('layers.0.') for key in index['weight_map']), 'Unexpected checkpoint key layout'
index['weight_map'] = {key: value for key, value in index['weight_map'].items() if
                      (key.startswith('layers.') and int(key.split('.')[1]) < layers) or
                      (not key.startswith(('layers.', 'mtp.', 'vision.', 'aligner.')))}
index['metadata'] = {'description': 'Original shards, restricted to layers 0..3 and backbone endpoints.'}
(out / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
# Use only the checkpoint's official standalone tokenizer normalization/hash helpers.
module = ast.parse((source / 'inference/engram.py').read_text())
functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in
             ('build_compressed_token_map', 'compute_hash_multipliers')]
assert len(functions) == 2, 'Missing checkpoint Engram helpers'
namespace = {'np': np, 'torch': torch}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source / 'inference/engram.py'), 'exec'), namespace)
tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
remap, compressed = namespace['build_compressed_token_map'](tokenizer)
assert compressed == original['engram_compressed_vocab_size']
multipliers = namespace['compute_hash_multipliers'](tuple(text['engram_layer_ids']), text['engram_max_ngram_size'], compressed).tolist()
artifact = {'format': 'megatron-engram-tokenizer-map', 'version': 1, 'max_ngram_order': text['engram_max_ngram_size'],
            'hash_seed': 0, 'pad_token_id': text['engram_pad_token_id'], 'layer_ids': text['engram_layer_ids'],
            'hash_layer_ids': text['engram_layer_ids'], 'remap': remap, 'source_vocab_size': len(remap),
            'compressed_vocab_size': compressed, 'compressed_pad_token_id': remap[text['engram_pad_token_id']],
            'layer_multipliers': {str(i): value for i, value in zip(text['engram_layer_ids'], multipliers)}}
core_config = (Path(__file__).resolve().parents[1] / 'Megatron-LM/megatron/core/models/engram/config.py').read_text()
for field, constant in [('format', 'TOKENIZER_MAP_FORMAT'), ('version', 'TOKENIZER_MAP_VERSION')]:
    match = re.search(r'^' + constant + r'\s*=\s*(.+)$', core_config, re.M)
    if match:
        artifact[field] = ast.literal_eval(match.group(1))
(out / 'engram_tokenizer_map.json').write_text(json.dumps(artifact) + '\n')
print(f'REDUCED_MODEL_READY {out}', flush=True)
