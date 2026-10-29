# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Dual-registry tests: two patch registries over the SAME function.

MA（megatron_adaptor 自身）与 MindSpeed 都各自持有独立注册表、复用同一个 ``Patch``
类：MA 先 ``register_patch`` 并 ``apply_patches``，随后 ``import mindspeed`` 的
特性对同一符号再注册、再 apply。本组用例固化这一跨注册表叠加的四类组合、分层 remove
顺序、类目标绑定保持与 remove→re-apply 时序下的行为，防止将来重构悄悄改变语义。

Run:
    pytest test/unit_tests/patches/test_dual_registry_patch.py -v -s
"""

import sys
import types

import pytest

import megatron_adaptor  # noqa: F401  确保补丁入口已执行
from megatron_adaptor.patches.patch_manager import Patch, get_module_attr

# 被测模块 sys.modules 注册名（前置 ma_ut_dual_ 避免与真实模块及同仓其他测试冲突）。
DUAL_MOD = 'ma_ut_dual_reg_mod'


class MindSpeedLikeRegistry:
    """模拟 MindSpeedPatchesManager（mindspeed/patch_utils.py）的最小等价实现。

    与 MA 的关系（见 issue93 分析文档）：复用 MA 的 ``Patch`` 类，但持有自己独立的
    ``patches_info`` 注册表，register/apply/remove 全部操作自身注册表。本类仅实现
    测试所需的最小面，不引入 mindspeed 仓依赖。
    """

    def __init__(self):
        self.patches_info = {}

    def register_patch(self, orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        if orig_func_name not in self.patches_info:
            self.patches_info[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            self.patches_info[orig_func_name].set_patch_func(new_func, force_patch)

    def apply_patches(self):
        for patch in self.patches_info.values():
            patch.apply_patch()

    def remove_patches(self):
        for patch in self.patches_info.values():
            patch.remove_patch()
            patch.remove_wrappers()


@pytest.fixture(autouse=True)
def uninstall_dual_registry_modules():
    """用例结束后从 sys.modules 摘除本组测试模块。"""
    yield
    for name in [n for n in sys.modules if n.startswith('ma_ut_dual_')]:
        sys.modules.pop(name, None)


def _install_target():
    """建一个带 target_func 的测试模块并注册进 sys.modules。"""
    module = types.ModuleType(DUAL_MOD)
    sys.modules[DUAL_MOD] = module
    setattr(module, 'target_func', _orig)
    return module


# --- 各注册方安装品与 wrapper（wrapper 名称必须以 wrapper/decorator 结尾，机制据此分派） ---
def _orig(x):
    return 'F(' + x + ')'


def _a(x):
    return 'A(' + x + ')'


def _b(x):
    return 'B(' + x + ')'


def _wA_wrapper(func):
    def wrapped(x):
        return 'WA[' + func(x) + ']'

    return wrapped


def _wB_wrapper(func):
    def wrapped(x):
        return 'WB[' + func(x) + ']'

    return wrapped


class TestReplacePlusReplace:
    """场景 1：MA 替换 + MindSpeed 替换。"""

    def test_ms_replace_silently_overwrites_ma_replace(self):
        # 两注册表各自安装自己的替换函数，后 apply 的 MindSpeed 直接覆盖 MA 产物，
        # 无冲突检测、无告警（RuntimeError 护栏只在同一注册表内生效）。
        target = _install_target()
        ma, ms = MindSpeedLikeRegistry(), MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.target_func', _a)
        ma.apply_patches()
        ms.register_patch(f'{DUAL_MOD}.target_func', _b)
        ms.apply_patches()
        assert get_module_attr(target, 'target_func') is _b
        assert get_module_attr(target, 'target_func')('i') == 'B(i)'
        # 分层 remove：MindSpeed 先还原 MA 产物，MA 再还原原始实现。
        ms.remove_patches()
        assert get_module_attr(target, 'target_func') is _a
        ma.remove_patches()
        assert get_module_attr(target, 'target_func') is _orig

    def test_same_registry_double_replace_raises_but_cross_registry_does_not(self):
        # 同一注册表内二次替换抛 RuntimeError；跨注册表（两个独立 patches_info）
        # 不抛——护栏只存在于单注册表内部。
        _install_target()
        ma = MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.target_func', _a)
        with pytest.raises(RuntimeError, match='the patch of .* exist'):
            ma.register_patch(f'{DUAL_MOD}.target_func', _b)


class TestReplacePlusWrapper:
    """场景 2：MA 替换 + MindSpeed wrapper。"""

    def test_ms_wrapper_stacks_over_ma_replace_and_layered_remove(self):
        target = _install_target()
        ma, ms = MindSpeedLikeRegistry(), MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.target_func', _a)
        ma.apply_patches()
        ms.register_patch(f'{DUAL_MOD}.target_func', _wA_wrapper)
        ms.apply_patches()
        assert get_module_attr(target, 'target_func')('i') == 'WA[A(i)]'
        # 分层 remove 顺序：先 MindSpeed（还原 MA 替换产物），后 MA（还原原始实现）。
        ms.remove_patches()
        assert get_module_attr(target, 'target_func')('i') == 'A(i)'
        ma.remove_patches()
        assert get_module_attr(target, 'target_func')('i') == 'F(i)'


class TestWrapperPlusWrapper:
    """场景 3：MA wrapper + MindSpeed wrapper。"""

    def test_cross_registry_wrappers_nest_in_apply_order(self):
        # MA 先 apply、MindSpeed 后 apply，故外层是后注册的 MindSpeed wrapper。
        target = _install_target()
        ma, ms = MindSpeedLikeRegistry(), MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.target_func', _wA_wrapper)
        ma.apply_patches()
        ms.register_patch(f'{DUAL_MOD}.target_func', _wB_wrapper)
        ms.apply_patches()
        assert get_module_attr(target, 'target_func')('i') == 'WB[WA[F(i)]]'
        # 分层 remove：MindSpeed 还原 MA 的 wrapper 产物，MA 再还原原始实现。
        ms.remove_patches()
        assert get_module_attr(target, 'target_func')('i') == 'WA[F(i)]'
        ma.remove_patches()
        assert get_module_attr(target, 'target_func')('i') == 'F(i)'


class TestWrapperPlusReplace:
    """场景 4：MA wrapper + MindSpeed 替换。"""

    def test_ms_replace_silently_drops_ma_wrapper(self):
        # 后注册的替换函数直接覆盖前注册的 wrapper 链，无告警；MA wrapper 不再生效。
        target = _install_target()
        ma, ms = MindSpeedLikeRegistry(), MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.target_func', _wA_wrapper)
        ma.apply_patches()
        ms.register_patch(f'{DUAL_MOD}.target_func', _b)
        ms.apply_patches()
        assert get_module_attr(target, 'target_func') is _b
        assert get_module_attr(target, 'target_func')('i') == 'B(i)'
        # 分层 remove：MindSpeed 还原 MA 的 wrapper 产物（WA[F] 仍可启动并可用）。
        ms.remove_patches()
        assert get_module_attr(target, 'target_func')('i') == 'WA[F(i)]'
        ma.remove_patches()
        assert get_module_attr(target, 'target_func')('i') == 'F(i)'


class TestClassMethodDualRegistry:
    """类方法目标（classmethod）跨注册表组合：绑定语义保持 + 可分层 remove。"""

    def _install_class_target(self):
        module = _install_target()

        class Holder:
            pass

        setattr(module, 'Holder', Holder)

        def orig_cm(_cls, x):
            return 'F(' + x + ')'

        def a_cm(_cls, x):
            return 'A(' + x + ')'

        def b_cm(_cls, x):
            return 'B(' + x + ')'

        def wA_cm_wrapper(func):
            def wrapped(cls, x):
                return 'WA[' + func(cls, x) + ']'

            return wrapped

        Holder.cm = classmethod(orig_cm)
        return Holder, orig_cm, a_cm, b_cm, wA_cm_wrapper

    def test_ma_replace_and_ms_wrapper_keep_binding_and_layered_remove(self):
        Holder, _orig_cm, a_cm, _b_cm, wA_cm_wrapper = self._install_class_target()
        ma, ms = MindSpeedLikeRegistry(), MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.Holder.cm', a_cm)
        ma.apply_patches()
        ms.register_patch(f'{DUAL_MOD}.Holder.cm', wA_cm_wrapper)
        ms.apply_patches()
        # cls 只注入一次（wrapper 转发不再注入第二个 cls）。
        assert Holder.cm('i') == 'WA[A(i)]'
        assert Holder().cm('i') == 'WA[A(i)]'
        assert isinstance(Holder.__dict__['cm'], classmethod)
        # 分层 remove 后绑定语义仍保持。
        ms.remove_patches()
        assert isinstance(Holder.__dict__['cm'], classmethod)
        assert Holder.cm('i') == 'A(i)'
        ma.remove_patches()
        assert isinstance(Holder.__dict__['cm'], classmethod)
        assert Holder.cm('i') == 'F(i)'

    def test_ms_replace_over_ma_wrapper_drops_wrapper_keeps_binding(self):
        Holder, _orig_cm, _a_cm, b_cm, wA_cm_wrapper = self._install_class_target()
        ma, ms = MindSpeedLikeRegistry(), MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.Holder.cm', wA_cm_wrapper)
        ma.apply_patches()
        ms.register_patch(f'{DUAL_MOD}.Holder.cm', b_cm)
        ms.apply_patches()
        # 后注册替换覆盖 wrapper 链；classmethod 形状保持、cls 绑定不丢。
        assert isinstance(Holder.__dict__['cm'], classmethod)
        assert Holder.cm('i') == 'B(i)'
        ms.remove_patches()
        assert Holder.cm('i') == 'WA[F(i)]'
        ma.remove_patches()
        assert Holder.cm('i') == 'F(i)'


class TestCrossRegistryRepatchTiming:
    """MindSpeed 侧 remove→re-apply（两阶段补丁时序）不得吞掉 MA 已装补丁。"""

    def test_ms_remove_reapply_keeps_ma_patch_alive(self):
        target = _install_target()
        ma, ms = MindSpeedLikeRegistry(), MindSpeedLikeRegistry()
        ma.register_patch(f'{DUAL_MOD}.target_func', _a)
        ma.apply_patches()
        ms.register_patch(f'{DUAL_MOD}.target_func', _wA_wrapper)
        ms.apply_patches()
        assert get_module_attr(target, 'target_func')('i') == 'WA[A(i)]'
        ms.remove_patches()
        # MindSpeed 清掉自己的 wrapper 后，MA 的替换仍存活。
        assert get_module_attr(target, 'target_func')('i') == 'A(i)'
        # MindSpeed 重新注册并 apply 新 wrapper，叠加在 MA 产物之上重建。
        ms.register_patch(f'{DUAL_MOD}.target_func', _wB_wrapper)
        ms.apply_patches()
        assert get_module_attr(target, 'target_func')('i') == 'WB[A(i)]'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
