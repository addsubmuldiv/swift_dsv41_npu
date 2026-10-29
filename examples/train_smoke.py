"""Single-node smoke training with typed Swift arguments."""
import json
import os
from pathlib import Path
import sys

import megatron_adaptor
import megatron.core
import transformer_engine
import mcore_bridge
import swift
from swift.megatron.arguments import MegatronSftArguments
from swift.megatron.pipelines.train.sft import megatron_sft_main

root = Path(__file__).resolve().parents[1]
for name, module in [('Megatron-LM', megatron.core), ('MegatronAdaptor', megatron_adaptor),
                     ('TransformerEngineNPU', transformer_engine), ('mcore-bridge', mcore_bridge), ('ms-swift', swift)]:
    assert Path(module.__file__).resolve().is_relative_to(root / name), (name, module.__file__)
assert 'mindspeed' not in sys.modules
settings = json.loads((root / 'examples/smoke_config.json').read_text())
settings.update(model=os.environ['SMOKE_MODEL'], model_type='deepseek_v41',
                dataset=[str(root / 'examples/data.jsonl')],
                engram_tokenizer_map=str(Path(os.environ['SMOKE_MODEL']) / 'engram_tokenizer_map.json'),
                output_dir=os.environ['SMOKE_OUTPUT'])
print('SOURCE_CHECK_PASS: all five source packages load from this checkout', flush=True)
megatron_sft_main(MegatronSftArguments(**settings))
