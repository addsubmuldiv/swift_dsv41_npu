import importlib.util
from logging import getLogger

from .megatron_adaptor import patch_features

LOG = getLogger(__name__)

patch_features()

if importlib.util.find_spec("mindspeed") is not None:
    try:
        import mindspeed.megatron_adaptor  # noqa: F401

        LOG.info("Successfully imported mindspeed.megatron_adaptor.")
    except ImportError:
        LOG.warning("Failed to import mindspeed.megatron_adaptor.")
