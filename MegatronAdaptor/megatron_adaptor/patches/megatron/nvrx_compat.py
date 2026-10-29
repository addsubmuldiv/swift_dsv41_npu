# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""NVRx compatibility helpers.

NOTE: this module must NOT import megatron at module level. It is called from
pre_register_patches before megatron is allowed to load; any megatron import
here would trigger strategies/torch.py's import-time HAVE_NVRX probe, which is
exactly what the backfill below is meant to precede.
"""


def patch_nvrx_version_attr():
    """Backfill ``nvidia_resiliency_ext.__version__`` before megatron import.

    The adapted NVRx package on NPU does not define ``__version__`` while
    ``strategies/torch.py`` evaluates ``HAVE_NVRX = has_nvrx_async_support()``
    at import time, which reads the attribute via ``is_nvrx_min_version``.
    Setting a large version here keeps that import-time probe from raising
    ``AttributeError`` and makes the minimum-version comparison pass; the
    adapted package's version string does not reflect its actual feature set,
    so the ``async_ckpt`` symbol probes inside ``has_nvrx_async_support``
    remain the real gate for async-checkpointing support.
    """
    try:
        import nvidia_resiliency_ext as nvrx
    except (ImportError, ModuleNotFoundError):
        return

    nvrx.__version__ = '999.999.999'
