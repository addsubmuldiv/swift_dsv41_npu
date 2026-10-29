# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""NPU device-probe compatibility helpers.

NOTE: this module must NOT import megatron at module level.  It is called
from pre_register_patches before megatron is allowed to load; any megatron
import here would (a) run MLM's import-time device probes (e.g. the GDP
kernels' ``megatron/core/ssm/ops/gdp/common.py``) and (b) capture the real
``torch.compile`` into ``megatron.core.jit.jit_fuser`` before the
``torch.compile -> dummy_compile`` stub is applied, breaking JIT warmup.
"""


def get_device_capability(device=None):
    """NPU stub for ``torch.cuda.get_device_capability``.

    transfer_to_npu rewrites ``torch.cuda.is_available()`` -> True but leaves
    ``get_device_capability()`` returning None, which crashes MLM import-time
    probes that index the tuple (e.g. ``common.py``'s
    ``get_device_capability()[0] >= 9``).  Report Hopper-class 9.0 so those
    gates evaluate consistently with the regular patch path.
    """
    return 9, 0


def patch_device_capability():
    """Eagerly backfill ``torch.cuda.get_device_capability`` before megatron import.

    The regular registration (``register_non_mcore_basic_patches``) only
    applies after ``megatron.training`` has been imported, which is too late:
    MLM main #6538 probes device capability at import time.  Mirror the
    ``nvrx_compat`` backfill pattern and install the same stub the normal
    path would install.
    """
    import torch

    torch.cuda.get_device_capability = get_device_capability
