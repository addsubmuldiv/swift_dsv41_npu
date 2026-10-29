"""Check that this checkout provides the five source packages and NPU initialization."""
import importlib.metadata
from pathlib import Path
import sys

import torch
import torch_npu
import megatron_adaptor
import megatron.core
import transformer_engine
import mcore_bridge
import swift
from mcore_bridge.model import get_mcore_model_type, get_model_meta
from mcore_bridge.model.modules.engram import has_native_engram
from swift.megatron.init import init_megatron_env

root = Path(__file__).resolve().parent
modules = {
    'Megatron-LM': megatron.core,
    'MegatronAdaptor': megatron_adaptor,
    'TransformerEngineNPU': transformer_engine,
    'mcore-bridge': mcore_bridge,
    'ms-swift': swift,
}
for name, module in modules.items():
    path = Path(module.__file__).resolve()
    assert path.is_relative_to(root / name), (name, str(path))
    print(f'SOURCE_PASS {name}: {path}', flush=True)
assert 'mindspeed' not in sys.modules
init_megatron_env()
assert has_native_engram(), 'Native Engram missing'
assert get_model_meta(get_mcore_model_type('deepseek_v41')) is not None
assert torch.npu.is_available(), 'Check CANN, driver and container device mounts'
x = torch.ones(4, 4, device='npu')
assert torch.equal((x @ x).cpu(), torch.full((4, 4), 4.0, device='cpu'))
print('PASS: NPU matmul and DeepSeek V4.1 initialization', flush=True)
print('Python:', sys.version.split()[0])
for name in ('torch', 'torch-npu', 'transformers', 'triton-ascend', 'transformer-engine'):
    print(name, importlib.metadata.distribution(name).version)
