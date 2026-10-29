# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps

from megatron_adaptor.features_manager.features_manager import FeaturesManager


def extra_args_provider_decorator(extra_args_provider):
    """Make a extra args parser for megatron."""

    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        FeaturesManager.register_features_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    """Decorate parse_args function of megatron."""

    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def validate_args_wrapper(validate_args):
    """A decorator for megatron arguments validation function."""

    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
        # make prev validation and copy some args.
        FeaturesManager.pre_validate_features_args(args)

        # make megatron args validation then restore args that are copied.
        args = validate_args(args, defaults)

        # make post validation after megatron validation.
        FeaturesManager.post_validate_features_args(args=args)

        FeaturesManager.validate_features_args(args=args)

        return args

    return wrapper
