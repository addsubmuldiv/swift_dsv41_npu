# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Offline tests for the ProfileFeature patches (registration and wrappers).

Run:
    pytest test/unit_tests/patches/megatron/test_profile.py -v -s
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

import megatron_adaptor  # noqa: F401  确保补丁入口已执行
from megatron_adaptor.features_manager.functional.profile import ProfileFeature
from megatron_adaptor.patches.megatron import profile as profile_mod


class FakePatchManager:
    """记录 register_patch 调用的假 patch manager。"""

    def __init__(self):
        self.registered = []

    def register_patch(self, orig_func_name, new_func=None, **kwargs):
        self.registered.append((orig_func_name, new_func))


class TestProfilePatchRegistration:
    """register_patches 注册行为与 is_megatron_training_available() 守卫。"""

    def test_feature_constants(self):
        # optimization_level 若被误改为 0，特性将变为 default_patches 对所有训练生效。
        feature = ProfileFeature()
        assert feature.feature_name == 'profile'
        assert feature.optimization_level == 2
        assert feature.default_patches is False

    def test_register_patches_when_megatron_training_available(self):
        pm = FakePatchManager()
        # 校验需要读取窗口参数，给定合法窗口（Megatron 默认值 start=10, end=12）。
        args = argparse.Namespace(profile=True, profile_step_start=10, profile_step_end=12)
        with patch('megatron_adaptor.patches.patch_utils.is_megatron_training_available', return_value=True):
            ProfileFeature().register_patches(pm, args)
        paths = [path for path, _ in pm.registered]
        assert paths == [
            'megatron.training.training.train',
            'megatron.training.training.train_step',
        ]
        funcs = [func for _, func in pm.registered]
        assert funcs == [profile_mod.train_wrapper, profile_mod.train_step_wrapper]

    def test_skip_registration_when_megatron_training_missing(self):
        # 仅 mcore 场景（无 megatron.training）下不应注册任何补丁。
        pm = FakePatchManager()
        with patch('megatron_adaptor.patches.patch_utils.is_megatron_training_available', return_value=False):
            ProfileFeature().register_patches(pm, argparse.Namespace())
        assert not pm.registered

    def test_skip_registration_when_profile_flag_off(self):
        # The feature is meaningless without --profile (train_wrapper only redirects
        # while it is True), so no patch should be registered in that case.
        pm = FakePatchManager()
        args = argparse.Namespace(profile=False, profile_step_start=10, profile_step_end=12)
        with patch('megatron_adaptor.patches.patch_utils.is_megatron_training_available', return_value=True):
            ProfileFeature().register_patches(pm, args)
        assert not pm.registered

    def test_register_patches_accepts_string_window_args(self):
        # get_mindspeed_args() 的 parser 未注册 Megatron 原生 --profile-step-start/end，
        # 命令行传参会被 parser_unknown_args 原样保存为字符串（"8"/"10"）。
        # 必须按数值比较，否则 '10' <= '8' 字典序误判为空窗口。
        pm = FakePatchManager()
        args = argparse.Namespace(profile=True, profile_step_start="8", profile_step_end="10")
        with patch('megatron_adaptor.patches.patch_utils.is_megatron_training_available', return_value=True):
            ProfileFeature().register_patches(pm, args)
        assert len(pm.registered) == 2

    def test_register_patches_rejects_empty_profile_window(self):
        # profile-step-end <= profile-step-start 时 active<=0，schedule 不采集任何步数，
        # trace 为空且无错误提示，须启动期抛 ValueError 帮助定位传参错误。
        pm = FakePatchManager()
        with patch('megatron_adaptor.patches.patch_utils.is_megatron_training_available', return_value=True):
            with pytest.raises(ValueError, match='must be greater than'):
                ProfileFeature().register_patches(
                    pm, argparse.Namespace(profile=True, profile_step_end=3, profile_step_start=5)
                )
            assert not pm.registered
            # 相等值同样构成空窗口。
            with pytest.raises(ValueError, match='must be greater than'):
                ProfileFeature().register_patches(
                    pm, argparse.Namespace(profile=True, profile_step_end=5, profile_step_start=5)
                )
            assert not pm.registered


def _make_train_args(**overrides):
    """构造 train_wrapper/train_step_wrapper 运行时所需的完整参数。"""
    defaults = dict(
        profile=False,
        profile_npu=False,
        profile_ranks=[0],
        profile_step_start=3,
        profile_step_end=5,
        profile_level='level0',
        profile_with_cpu=False,
        profile_with_stack=False,
        profile_with_memory=False,
        profile_record_shapes=False,
        profile_save_path='./profile_dir',
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestTrainWrapper:
    """train_wrapper 的参数重定向与直通行为（不进入真实 profiler 上下文）。"""

    def test_no_profile_passes_through(self):
        train = MagicMock(return_value='train_ret')
        wrapper = profile_mod.train_wrapper(train)
        args = _make_train_args(profile=False)
        with patch.object(profile_mod, 'get_full_args', return_value=args):
            ret = wrapper('a', kw=1)
        assert ret == 'train_ret'
        train.assert_called_once_with('a', kw=1)
        # 未传 --profile 时 profile_npu 为 False，原生 profile 标志保持不变。
        assert args.profile_npu is False
        assert args.profile is False

    def test_profile_flag_redirected_when_rank_misses(self):
        # rank 未命中时不进入 profiler 上下文，但参数重定向必须完成：
        # profile_npu=True 且原生 profile 标志被关闭（屏蔽 torch.profiler 路径）。
        train = MagicMock(return_value='train_ret')
        wrapper = profile_mod.train_wrapper(train)
        args = _make_train_args(profile=True, profile_ranks=[1])
        with (
            patch.object(profile_mod, 'get_full_args', return_value=args),
            patch('torch.distributed.get_rank', return_value=0),
        ):
            ret = wrapper()
        assert ret == 'train_ret'
        train.assert_called_once_with()
        assert args.profile_npu is True
        assert args.profile is False


class TestTrainStepWrapper:
    """train_step_wrapper 的 PROFILE_RECORD.step() 驱动条件（is_profile 两分支）。"""

    def test_step_drives_profiler_when_rank_hits(self):
        train_step = MagicMock(return_value='step_ret')
        wrapper = profile_mod.train_step_wrapper(train_step)
        args = _make_train_args(profile_ranks=[0])
        args.profile_npu = True
        fake_record = MagicMock()
        old_record = profile_mod.PROFILE_RECORD
        try:
            profile_mod.PROFILE_RECORD = fake_record
            with (
                patch.object(profile_mod, 'get_full_args', return_value=args),
                patch('torch.distributed.get_rank', return_value=0),
            ):
                ret = wrapper()
        finally:
            profile_mod.PROFILE_RECORD = old_record
        assert ret == 'step_ret'
        fake_record.step.assert_called_once_with()

    def test_step_not_called_when_rank_misses(self):
        train_step = MagicMock(return_value='step_ret')
        wrapper = profile_mod.train_step_wrapper(train_step)
        args = _make_train_args(profile_ranks=[7])
        args.profile_npu = True
        fake_record = MagicMock()
        old_record = profile_mod.PROFILE_RECORD
        try:
            profile_mod.PROFILE_RECORD = fake_record
            with (
                patch.object(profile_mod, 'get_full_args', return_value=args),
                patch('torch.distributed.get_rank', return_value=0),
            ):
                wrapper()
        finally:
            profile_mod.PROFILE_RECORD = old_record
        fake_record.step.assert_not_called()

    def test_step_no_error_when_profile_npu_attr_missing(self):
        # When train_step is called directly without going through train(), profile_npu
        # (a side-effect set by train_wrapper) is absent on args; treat it as off
        # instead of raising AttributeError.
        train_step = MagicMock(return_value='step_ret')
        wrapper = profile_mod.train_step_wrapper(train_step)
        args = argparse.Namespace(profile_ranks=[0])
        fake_record = MagicMock()
        old_record = profile_mod.PROFILE_RECORD
        try:
            profile_mod.PROFILE_RECORD = fake_record
            with (
                patch.object(profile_mod, 'get_full_args', return_value=args),
                patch('torch.distributed.get_rank', return_value=0),
            ):
                ret = wrapper()
        finally:
            profile_mod.PROFILE_RECORD = old_record
        assert ret == 'step_ret'
        fake_record.step.assert_not_called()


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short", "-s"])
