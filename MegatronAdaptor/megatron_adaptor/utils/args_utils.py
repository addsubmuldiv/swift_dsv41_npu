# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import argparse
from contextvars import ContextVar
from logging import getLogger

LOG = getLogger(__name__)
_full_args: ContextVar = ContextVar('full_args', default=None)
_MINDSPEED_ARGS = None


def add_args(args, key, value):
    if key is not None:
        key = key[2:].replace('-', '_')
        if value is None:
            value = True
        elif len(value) == 1:
            value = value[0]
        args[key] = value


def parser_unknown_args(args, unknown):
    i = 0
    key = value = None
    while i < len(unknown):
        if unknown[i].startswith('-'):
            add_args(args, key, value)
            splits = unknown[i].split('=', maxsplit=1)
            key, value = (unknown[i], None) if len(splits) == 1 else (splits[0], [splits[1]])
        else:
            if value is None:
                value = [unknown[i]]
            else:
                value.append(unknown[i])
        i += 1
    add_args(args, key, value)
    return args


def get_mindspeed_args(get_defaults=False):
    global _MINDSPEED_ARGS
    if not get_defaults and _MINDSPEED_ARGS is not None:
        return _MINDSPEED_ARGS

    parser = argparse.ArgumentParser(description='MegatronAdaptor Arguments', allow_abbrev=False)
    from megatron_adaptor.features_manager.features_manager import FeaturesManager

    FeaturesManager.register_features_args(parser)
    args, unknown = parser.parse_known_args() if not get_defaults else parser.parse_known_args([])
    args = parser_unknown_args(vars(args), unknown)
    args = argparse.Namespace(**args)
    if not get_defaults:
        _MINDSPEED_ARGS = args
    return args


def reset_mindspeed_args():
    """Invalidate the cached adaptor args after the feature list changes.

    The cached args are parsed from the feature list at parse time; once the
    list changes (e.g. MindSpeed features are appended), a stale cache would
    miss newly registered feature arguments, so it must be re-parsed.
    """
    global _MINDSPEED_ARGS
    _MINDSPEED_ARGS = None


def get_full_args() -> argparse.Namespace:
    args = _full_args.get()
    if args is not None:
        return args
    try:
        from megatron.training.global_vars import get_args

        full_args = get_args()
        if full_args is None:
            full_args = get_mindspeed_args()
    except ImportError:
        LOG.warning('Failed from megatron.training import get_args, use megatron_adaptor arguments.')
        full_args = get_mindspeed_args()
    except AssertionError as e:
        if 'args is not initialized.' in str(e):
            full_args = get_mindspeed_args()
        else:
            raise e
    return full_args


def set_full_args(args: argparse.Namespace):
    _full_args.set(args)
