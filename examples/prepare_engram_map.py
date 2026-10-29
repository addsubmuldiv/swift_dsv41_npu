"""Generate the full model Engram tokenizer map without changing its config or weights."""
import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


def prepare_engram_map(model, output):
    text = json.loads((model / 'config.json').read_text())['text_config']
    module = ast.parse((model / 'inference/engram.py').read_text())
    helpers = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in
               ('build_compressed_token_map', 'compute_hash_multipliers')]
    if len(helpers) != 2:
        raise ValueError('Missing official Engram tokenizer helpers in the model directory')
    namespace = {'np': np, 'torch': torch}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(model / 'inference/engram.py'), 'exec'), namespace)
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    remap, compressed = namespace['build_compressed_token_map'](tokenizer)
    if compressed != text['engram_compressed_vocab_size']:
        raise ValueError('Engram compressed vocabulary does not match model config')
    layers = text['engram_layer_ids']
    multipliers = namespace['compute_hash_multipliers'](
        tuple(layers), text['engram_max_ngram_size'], compressed).tolist()
    artifact = {
        'format': 'megatron-engram-tokenizer-map', 'version': 1,
        'max_ngram_order': text['engram_max_ngram_size'], 'hash_seed': 0,
        'pad_token_id': text['engram_pad_token_id'], 'layer_ids': layers, 'hash_layer_ids': layers,
        'remap': remap, 'source_vocab_size': len(remap), 'compressed_vocab_size': compressed,
        'compressed_pad_token_id': remap[text['engram_pad_token_id']],
        'layer_multipliers': {str(i): value for i, value in zip(layers, multipliers)},
    }
    config = (Path(__file__).resolve().parents[1] /
              'Megatron-LM/megatron/core/models/engram/config.py').read_text()
    for field, constant in [('format', 'TOKENIZER_MAP_FORMAT'), ('version', 'TOKENIZER_MAP_VERSION')]:
        match = re.search(r'^' + constant + r'\s*=\s*(.+)$', config, re.M)
        if match:
            artifact[field] = ast.literal_eval(match.group(1))
    output.write_text(json.dumps(artifact) + '\n')
    print(f'Full backbone: {text["num_hidden_layers"]} layers; Engram layers: {layers}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_engram_map(Path(args.model).resolve(), output)


if __name__ == '__main__':
    main()
