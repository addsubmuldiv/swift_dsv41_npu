# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import List, Union
from megatron_adaptor.features_manager.mindspeed_feature import MindSpeedFeature


class FeaturesManager:
    FEATURES_LIST = []

    @classmethod
    def set_features_list(cls, features_list: List[MindSpeedFeature]):
        cls.FEATURES_LIST[:] = features_list
        cls._invalidate_args_cache()

    @classmethod
    def add_feature(cls, feature: Union[MindSpeedFeature, List[MindSpeedFeature]]):
        """Add one feature or a list of features to the feature list."""
        if isinstance(feature, MindSpeedFeature):
            cls.FEATURES_LIST.append(feature)
        else:
            cls.FEATURES_LIST.extend(feature)
        cls._invalidate_args_cache()

    @staticmethod
    def _invalidate_args_cache():
        """Drop the cached adaptor args so the new feature list is re-parsed.

        Drop cached args when feature list changes; otherwise newly
        registered feature args (e.g. ``--optimization-level``) are missed.
        """
        from megatron_adaptor.utils.args_utils import reset_mindspeed_args  # pylint: disable=cyclic-import

        reset_mindspeed_args()

    @classmethod
    def apply_features_pre_patches(cls, args):
        # Apply incrementally: features already applied in a previous stage
        # (e.g. MA entry then MindSpeed entry) are skipped to avoid
        # registering their patches twice.
        patch_managers = []
        for feature in cls.FEATURES_LIST:
            if not feature.is_need_apply(args) or feature.pre_patches_applied:
                continue
            feature.pre_register_patches(feature.patch_manager, args)
            feature.pre_patches_applied = True
            if feature.patch_manager not in patch_managers:
                patch_managers.append(feature.patch_manager)
        for patch_manager in patch_managers:
            patch_manager.apply_patches()

    @classmethod
    def apply_features_patches(cls, args):
        # Same incremental semantics as apply_features_pre_patches.
        patch_managers = []
        for feature in cls.FEATURES_LIST:
            if not feature.is_need_apply(args) or feature.patches_applied:
                continue
            feature.register_patches(feature.patch_manager, args)
            feature.patches_applied = True
            if feature.patch_manager not in patch_managers:
                patch_managers.append(feature.patch_manager)
        for patch_manager in patch_managers:
            patch_manager.apply_patches()

    @classmethod
    def register_features_args(cls, parser):
        for feature in cls.FEATURES_LIST:
            feature.register_args(parser)

    @classmethod
    def pre_validate_features_args(cls, args):
        for feature in cls.FEATURES_LIST:
            feature.pre_validate_args(args)

    @classmethod
    def post_validate_features_args(cls, args):
        for feature in cls.FEATURES_LIST:
            feature.post_validate_args(args)

    @classmethod
    def validate_features_args(cls, args):
        for feature in cls.FEATURES_LIST:
            feature.validate_args(args)

    @classmethod
    def remove_patches(cls):
        # Reset the application state of every feature and remove the patches
        # of all patch managers the feature list is registered to.
        patch_managers = []
        for feature in cls.FEATURES_LIST:
            feature.pre_patches_applied = False
            feature.patches_applied = False
            if feature.patch_manager not in patch_managers:
                patch_managers.append(feature.patch_manager)
        for patch_manager in patch_managers:
            patch_manager.remove_patches()
