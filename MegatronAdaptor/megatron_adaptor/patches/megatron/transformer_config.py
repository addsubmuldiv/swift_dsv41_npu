# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""TransformerConfig patches for MindSpeed-specific parameters.

Migrated from ``mindspeed/core/megatron_basic/arguments_basic.py``. These patches make mcore's
dataclass configs grow the extra CLI fields registered by the unified feature
list, and hang the feature validation pipeline onto ``__post_init__``.
"""

import inspect
from dataclasses import make_dataclass, field
from functools import wraps

from megatron_adaptor.features_manager.features_manager import FeaturesManager
from megatron_adaptor.utils.args_utils import get_full_args

# Keys owned by the training-level model config (GPTModelConfig / HybridModelConfig), not by
# TransformerConfig. Injecting them on megatron >= v0.19 trips the model-config __setattr__
# proxy: the value is diverted to the transformer and reads fall back to the dataclass default,
# silently ignoring the CLI value (and, for position_embedding_type, changing the model).
# Membership rule: a model-config field that is a real CLI arg and is never read off a base
# TransformerConfig (rotary_base / rotary_percent are MLATransformerConfig-native -> safe).
_MODEL_CONFIG_OWNED_KEYS = frozenset(
    {
        # Network shape / vocabulary
        'vocab_size',
        'padded_vocab_size',
        'seq_length',
        'make_vocab_size_divisible_by',
        'fp16_lm_cross_entropy',
        # Position / RoPE
        'position_embedding_type',
        'rotary_base',
        'rotary_percent',
        'rope_scaling_factor',
        # Hybrid layer layout
        'hybrid_layer_pattern',
        'hybrid_override_pattern',
        # Misc
        'tp_comm_overlap_cfg',
        # Defensive entries: not CLI args today, kept so injection stays off if upstream
        # later turns them into CLI args.
        'logit_dtype',
        'parallel_output',
        'should_pad_vocab',
        'share_embeddings_and_output_weights',
    }
)


def transformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        # make prev validation and copy some args.
        FeaturesManager.pre_validate_features_args(self)

        fn(self)

        FeaturesManager.post_validate_features_args(args=self)

        FeaturesManager.validate_features_args(args=self)

    return wrapper


def transformer_config_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        known_config = {}
        unknown_config = {}
        full_args = vars(get_full_args()).copy()
        full_args.update(dict(kwargs))

        config_key = inspect.signature(self.__class__).parameters
        for key, value in full_args.items():
            if key in config_key:
                # Skip when the CLI arg is None (unset) and kwargs lacks the key,
                # letting the dataclass default apply (e.g. rope_type defaults to
                # "yarn"; upstream argument_utils pops the unset rope_type).
                # Explicit None in kwargs must pass through as-is: it carries
                # upstream semantics (e.g. num_query_groups=None for non-GQA).
                if value is None and key not in kwargs:
                    continue
                known_config[key] = value
            elif key not in _MODEL_CONFIG_OWNED_KEYS:
                unknown_config[key] = value

        fields = []
        for key, value in unknown_config.items():
            if not hasattr(self, key):
                fields.append((str(key), type(value), field(init=False)))  # pylint: disable=invalid-field-call
        self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))
        for key, value in unknown_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        fn(self, *args, **known_config)

    return wrapper


def transformer_config_init_subclass(cls, **kwargs):
    mutable_types = (list, dict, set, bytearray)
    unknown_config = {}
    full_args = vars(get_full_args()).copy()
    full_args.update(kwargs)

    config_key = inspect.signature(cls).parameters
    for key, value in full_args.items():
        if (key not in config_key) and (key not in _MODEL_CONFIG_OWNED_KEYS):
            unknown_config[key] = value

    for key, value in unknown_config.items():
        if not hasattr(cls, key):
            cls.__annotations__[key] = type(value)
            if callable(value) and not isinstance(value, type):
                value = field(default_factory=value)  # pylint: disable=invalid-field-call
            elif type(value) in mutable_types:
                value = field(default_factory=lambda value=value: value)  # pylint: disable=invalid-field-call
            setattr(cls, key, value)
