# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser, Namespace
from megatron_adaptor.patches.patch_manager import PatchesManager


class MindSpeedFeature:
    """Base class for mindspeed features."""

    def __init__(self, feature_name: str, optimization_level: int = 2):
        self.feature_name = feature_name.lower().strip().replace("-", "_")
        self.optimization_level = optimization_level
        self.default_patches = self.optimization_level == 0
        # Applied flags. The manager may run apply multiple times (the MA
        # entry first, then the MindSpeed entry after its features join the
        # list), so features already applied in an earlier run are skipped.
        self.pre_patches_applied = False
        self.patches_applied = False

    @property
    def patch_manager(self):
        """The patch manager that this feature's patches are registered to.

        MindSpeed subclasses override this so that their patches are owned by
        MindSpeedPatchesManager and can be removed layer by layer.
        """
        return PatchesManager

    @staticmethod
    def _parse_optimization_level(value):
        """Coerce *value* to ``int``."""
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        try:
            return int(value)
        except (ValueError, TypeError):
            raise TypeError(f"optimization_level should be a number, but got: {value!r}")

    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        args_optimization_level = self._parse_optimization_level(getattr(args, "optimization_level", 2))
        return (
            self.optimization_level <= args_optimization_level and getattr(args, self.feature_name, None)
        ) or self.default_patches

    def register_args(self, parser: ArgumentParser):
        """Register cli arguments to enable the feature."""
        pass

    def pre_validate_args(self, args: Namespace):
        """Validate the arguments of mindspeed before megatron args validation
        and store some arguments of the mindspeed temporarily,
        incase that megatron validate faile.
        for example:
            ```python
            origin_context_parallel_size = args.context_parallel_size
            args.context_parallel_size = 1
            ```
        """
        pass

    def validate_args(self, args: Namespace):
        """Restore the arguments of the mindspeed.

        for example:
        ```python
        args.context_parallel_size = origin_context_parallel_size
        ```
        """
        pass

    def post_validate_args(self, args: Namespace):
        """validate mindspeed arguments after megatron arguments validation."""
        pass

    def pre_register_patches(self, patch_manager: PatchesManager, args: Namespace):
        """Register all patch functions before import megatron"""
        pass

    def register_patches(self, patch_manager: PatchesManager, args: Namespace):
        """Register all patch functions the feature is related."""
        pass

    def incompatible_check(self, global_args, check_args):
        """Register all incompatible functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and getattr(global_args, check_args, None):
            raise AssertionError("{} and {} are incompatible.".format(self.feature_name, check_args))

    def dependency_check(self, global_args, check_args):
        """Register all dependency functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and not getattr(global_args, check_args, None):
            raise AssertionError("{} requires {}.".format(self.feature_name, check_args))
