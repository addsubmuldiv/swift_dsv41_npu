# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Offline tests for the ``Patch`` class in patch_manager (patch_manager.py).

Directly exercise the Patch-level interface (set_patch_func / apply_patch /
remove_patch) covering the nvdlfw_inspect regression fix: dynamic-proxy modules
whose module-level __getattr__ (PEP 562) raises RuntimeError before initialize()
must not break apply_patch()/remove_patch().

Run:
    pytest test/unit_tests/patches/test_patch_manager.py -v -s
"""

import sys
import types

import pytest

import megatron_adaptor  # noqa: F401  确保补丁入口已执行
from megatron_adaptor.patches.patch_manager import MISSING_ATTR, Patch, get_module_attr

# 被测模块 / 别名模块 / 动态代理模块的 sys.modules 注册名（前置 ma_ut_ 避免与真实模块冲突）。
TARGET_MOD = 'ma_ut_target_module'
ALIAS_MOD = 'ma_ut_alias_module'
PROXY_MOD = 'ma_ut_proxy_api'


class NvdlfwLikeError(RuntimeError):
    """与 nvdlfw_inspect.api 未初始化时抛出一致的异常类型。"""


class DynamicProxyModule(types.ModuleType):
    """模拟 nvdlfw_inspect.api：模块级 __getattr__ 在未 initialize 时抛 RuntimeError。"""

    # 模拟已初始化的调试管理器 API。
    initialize = staticmethod(lambda: None)

    # 视为有子模块搜索路径的包：使 import_module('proxy_api.child') 正常抛
    # ModuleNotFoundError（否则 import 机制访问 __path__ 反而触发下面的 __getattr__）。
    __path__ = []

    def __getattr__(self, name):
        raise NvdlfwLikeError('[NVDLFW INSPECT ERROR] Debug manager not initialized. Call initialize first.')


class StrictAttrModule(types.ModuleType):
    """模拟 torch.fx.experimental._config：非预声明属性禁止 setattr，抛 AttributeError。"""

    # 静态声明以满足 pylint 成员检查；实例化后 __name__ 会被 ModuleType 赋为真实模块名。
    __name__ = 'strict_attr_module'

    def __setattr__(self, name, value):
        if not name.startswith('__'):
            raise AttributeError(f'{self.__name__}.{name} does not exist')
        super().__setattr__(name, value)


@pytest.fixture(autouse=True)
def uninstall_test_modules():
    """用例结束后从 sys.modules 摘除测试模块。"""
    yield
    for name in (TARGET_MOD, ALIAS_MOD, PROXY_MOD):
        sys.modules.pop(name, None)


def _set_attr(module, name, value):
    """给测试模块动态设置属性（模块无法静态声明成员，避开 pylint E1101）。"""
    setattr(module, name, value)
    return module


def _install(proxy_module=True):
    """安装 目标/别名 模块（可选动态代理模块），返回 (target, alias, proxy)。"""
    target = types.ModuleType(TARGET_MOD)
    sys.modules[TARGET_MOD] = target
    alias = types.ModuleType(ALIAS_MOD)
    sys.modules[ALIAS_MOD] = alias
    proxy = None
    if proxy_module:
        proxy = DynamicProxyModule(PROXY_MOD)
        sys.modules[PROXY_MOD] = proxy
    return target, alias, proxy


@pytest.fixture(autouse=True)
def uninstall_extra_modules():
    """用例结束后摘除临时塞进 sys.modules 的额外模块（三个常量模块由上一 fixture 负责）。"""
    yield
    known = (TARGET_MOD, ALIAS_MOD, PROXY_MOD)
    for name in [n for n in sys.modules if n.startswith('ma_ut_') and n not in known]:
        sys.modules.pop(name, None)


def _register(name, **attrs):
    """建一个注册进 sys.modules 的额外测试模块。"""
    module = types.ModuleType(name)
    sys.modules[name] = module
    for key, value in attrs.items():
        _set_attr(module, key, value)
    return module


def _orig(x):
    return 'orig'


def _patched(x):
    return 'patched'


class TestGetModuleAttr:
    """get_module_attr 只查模块 __dict__、不触发模块级 __getattr__。"""

    def test_reads_real_binding_and_missing_returns_sentinel(self):
        target = _set_attr(types.ModuleType(TARGET_MOD), 'foo', _orig)
        # 真实绑定可读；缺失属性、非模块对象均返回 MISSING_ATTR（区分“缺失”与“值为 None”）。
        assert get_module_attr(target, 'foo') is _orig
        assert get_module_attr(target, 'bar') is MISSING_ATTR
        assert get_module_attr(object(), 'foo') is MISSING_ATTR

    def test_does_not_trigger_proxy_getattr(self):
        # 修复核心：get_module_attr 对动态代理模块返回 MISSING_ATTR，不会冒泡 RuntimeError；
        # hasattr 仍会触发 __getattr__（旧实现缺陷根因，hasattr 仅吞 AttributeError）。
        _, _, proxy = _install()
        assert get_module_attr(proxy, 'foo') is MISSING_ATTR
        with pytest.raises(NvdlfwLikeError):
            hasattr(proxy, 'foo')


class TestApplyPatchLevel:
    """直接调用 Patch.apply_patch()/remove_patch()，不经过 PatchesManager。"""

    def test_apply_remove_survives_proxy_in_sys_modules(self):
        # 目标函数名与被 patch 目标一致，扫描必然触发代理模块 __getattr__；
        # 修复前 apply/remove 都会因 RuntimeError 中断。
        target, alias, _ = _install()
        target = _set_attr(target, 'foo', _orig)
        alias = _set_attr(alias, 'foo', _orig)
        patch = Patch(f'{TARGET_MOD}.foo', _patched, False)
        patch.apply_patch()
        assert get_module_attr(target, 'foo') is _patched
        assert get_module_attr(alias, 'foo') is _patched
        patch.apply_patch()  # is_applied=True，幂等不重复替换。
        assert get_module_attr(target, 'foo') is _patched
        patch.remove_patch()
        assert get_module_attr(target, 'foo') is _orig
        assert get_module_attr(alias, 'foo') is _orig

    def test_apply_remove_without_proxy_still_roundtrips(self):
        # 回归：无动态代理模块时 apply/remove 语义保持不变。
        target, alias, _ = _install(proxy_module=False)
        target = _set_attr(target, 'foo', _orig)
        alias = _set_attr(alias, 'foo', _orig)
        patch = Patch(f'{TARGET_MOD}.foo', _patched, False)
        patch.apply_patch()
        assert get_module_attr(target, 'foo') is _patched
        assert get_module_attr(alias, 'foo') is _patched
        patch.remove_patch()
        assert get_module_attr(target, 'foo') is _orig
        assert get_module_attr(alias, 'foo') is _orig

    def test_dummy_patch_does_not_touch_unrelated_or_strict_modules(self):
        # 回归流水线报错：create_dummy 场景下原函数缺失，current_func 为 None，
        # 曾把“缺失属性”误判为 None 而对所有模块执行 setattr，最终写崩
        # torch.fx.experimental._config 这类只读模块（AttributeError: ... does not exist）。
        target, alias, _ = _install()
        sys.modules[PROXY_MOD] = StrictAttrModule(PROXY_MOD)
        missing_module = 'ma_ut_missing_te'
        missing_func = 'thd_get_partitioned_indices'
        try:
            patch = Patch(f'{missing_module}.{missing_func}', _patched, create_dummy=True)
            patch.apply_patch()
            # 无关模块不得被塞入该属性，只读/受限模块不得被 setattr 写崩。
            assert missing_func not in vars(target)
            assert missing_func not in vars(alias)
            assert missing_func not in vars(sys.modules[PROXY_MOD])
            patch.remove_patch()
            assert missing_func not in vars(target)
        finally:
            sys.modules.pop(f'{missing_module}.{missing_func}', None)
            sys.modules.pop(missing_module, None)

    def test_remove_patch_class_target_survives_proxy_module(self):
        # remove_patch 遍历 sys.modules 时，类对象补丁分支曾用 hasattr(value, 类名) 探测
        # 任意模块，同样会触发动态代理模块的 __getattr__ 冒泡 RuntimeError；
        # 卸载类对象补丁也必须只读模块 __dict__（get_module_attr），仅对定位到的类用 getattr 兜底。
        target, alias, _ = _install()

        class FakeClass:
            pass

        target = _set_attr(target, 'FakeClass', FakeClass)
        alias = _set_attr(alias, 'FakeClass', FakeClass)
        patch = Patch(f'{TARGET_MOD}.FakeClass.method', _patched, create_dummy=True)
        patch.apply_patch()
        # 修复前 remove_patch 在扫描到代理模块时抛 NvdlfwLikeError 中断。
        patch.remove_patch()
        # 卸载后 FakeClass.method 还原为 dummy，且代理模块未被写坏。
        assert FakeClass.__dict__['method'].__name__ == 'dummy_function'
        assert not patch.is_applied
        assert patch.final_patch_func is None

    def test_parse_path_create_dummy_survives_proxy_parent(self):
        # parse_path 在子模块 import 失败后会探测父模块属性，曾用 hasattr(父模块, 子名)
        # 触发动态代理模块 __getattr__；应只读父模块 __dict__ 走 dummy 路径而非崩溃。
        _, _, proxy = _install()  # noqa: F841
        nested = f'{PROXY_MOD}.non_existent_child'
        patch = Patch(f'{nested}.some_func', _patched, create_dummy=True)
        try:
            # 修复前此处冒泡 NvdlfwLikeError。
            patch.apply_patch()
            assert sys.modules.get(nested) is not None
            patch.remove_patch()
        finally:
            sys.modules.pop(f'{nested}.some_func', None)
            sys.modules.pop(nested, None)

    def test_readonly_alias_does_not_abort_apply_or_remove(self):
        """只读模块（如 torch.fx.experimental._config）持有目标函数别名时 setattr 抛
        AttributeError；单个写不进去的别名不得中断整轮 apply/remove。
        """
        target, alias, _ = _install(proxy_module=False)
        _set_attr(target, 'foo', _orig)
        _set_attr(alias, 'foo', _orig)
        readonly = StrictAttrModule('ma_ut_readonly')
        readonly.__dict__['foo'] = _orig  # 绕过 __setattr__ 预置别名
        sys.modules['ma_ut_readonly'] = readonly
        patch = Patch(f'{TARGET_MOD}.foo', _patched, False)
        patch.apply_patch()
        assert get_module_attr(target, 'foo') is _patched
        assert get_module_attr(alias, 'foo') is _patched
        assert readonly.__dict__['foo'] is _orig  # 写不进去，保持原值
        patch.remove_patch()
        assert get_module_attr(target, 'foo') is _orig
        assert get_module_attr(alias, 'foo') is _orig

    def test_own_namespace_aliases_skipped_but_targets_still_unload(self):
        """apply/remove 一致跳过 MA/MindSpeed 命名空间的别名，避免改写或残留旧补丁；
        被跳过的命名空间自身的模块级/类级主目标仍须经捕获的 holder 正常装卸。
        """
        target, _, _ = _install(proxy_module=False)
        _set_attr(target, 'foo', _orig)
        own_ns_names = ('ma_ut_mindspeed_holder', 'ma_ut_megatron_adaptor_holder')
        holders = [_set_attr(_register(name), 'foo', _orig) for name in own_ns_names]
        patch = Patch(f'{TARGET_MOD}.foo', _patched, False)
        patch.apply_patch()
        assert get_module_attr(target, 'foo') is _patched
        assert all(get_module_attr(holder, 'foo') is _orig for holder in holders)
        patch.remove_patch()
        assert get_module_attr(target, 'foo') is _orig
        assert all(get_module_attr(holder, 'foo') is _orig for holder in holders)

        # 主目标（模块级）：MindSpeed 会 patch 自家符号，即便位于被跳过的命名空间，
        # 也要经捕获的 holder 装卸，而不是跟着别名一起被跳过。
        ms_target = _register('ma_ut_mindspeed_target', foo=_orig)
        ms_patch = Patch('ma_ut_mindspeed_target.foo', _patched, False)
        ms_patch.apply_patch()
        assert get_module_attr(ms_target, 'foo') is _patched
        ms_patch.remove_patch()
        assert get_module_attr(ms_target, 'foo') is _orig

        # 主目标（类级，如 mindspeed.optimizer.adamw.AdamW.step）：类只能经由所在模块定位，
        # 该模块被跳过时仍须直接还原捕获的 holder。
        class AdamWLike:
            def step(self, x):
                return ('orig', x)

        def patched_step(self, x):
            return ('patched', x)

        _set_attr(_register('ma_ut_mindspeed_pkg'), 'AdamWLike', AdamWLike)
        cls_patch = Patch('ma_ut_mindspeed_pkg.AdamWLike.step', patched_step, False)
        cls_patch.apply_patch()
        assert AdamWLike().step(1) == ('patched', 1)
        cls_patch.remove_patch()
        assert AdamWLike().step(1) == ('orig', 1)


class TestSetPatchFunc:
    """set_patch_func 的替换/装饰分派与 force_patch 语义。"""

    def test_dummy_created_for_missing_func(self):
        patch = Patch(f'{TARGET_MOD}.nonexistent', None, False)
        with pytest.raises(RuntimeError, match='no exist'):
            patch.patch_func()

    def test_reject_duplicate_and_allow_force_overwrite(self):
        target, _, _ = _install()
        target = _set_attr(target, 'foo', _orig)
        patch = Patch(f'{TARGET_MOD}.foo', _patched, False)
        assert patch.patch_func is _patched
        assert not patch.wrappers
        with pytest.raises(RuntimeError, match='the patch of .* exist'):
            patch.set_patch_func(_orig)
        # force 覆盖后替换函数与原函数相同，apply 为无操作。
        patch.set_patch_func(_orig, force_patch=True)
        patch.apply_patch()
        assert get_module_attr(target, 'foo') is _orig

    def test_wrapper_stacking(self):
        target, _, _ = _install()
        target = _set_attr(target, 'foo', _orig)
        patch = Patch(f'{TARGET_MOD}.foo', _patched, False)

        def wrapper_wrapper(func):
            def wrapped(x):
                return 'wrap(' + func(x) + ')'

            return wrapped

        patch.set_patch_func(wrapper_wrapper)
        assert wrapper_wrapper in patch.wrappers
        assert patch.patch_func is _patched
        patch.apply_patch()
        assert get_module_attr(target, 'foo')('a') == 'wrap(patched)'


class TestDescriptorBindingLifecycle:
    """类方法目标（staticmethod/classmethod）在 apply/remove 往返中保持绑定语义。

    回归背景：remove 后再 apply 时，Patch 曾用解包后的普通函数装回类，丢失staticmethod 描述符；
    实例访问 self.method(...) 会注入 self，产生多余的位置参数。
    """

    def _install_class(self):
        target, _, _ = _install()

        class Holder:
            pass

        _set_attr(target, 'Holder', Holder)
        return target, Holder

    def test_staticmethod_binding_survives_roundtrip(self):
        def _sm(x, y=2):
            return {'x': x, 'y': y}

        target, Holder = self._install_class()  # noqa: F841
        Holder.sm = staticmethod(_sm)
        patch = Patch(f'{TARGET_MOD}.Holder.sm', _patched, False)
        patch.apply_patch()
        # 安装产物保持 staticmethod：类访问与实例访问都不注入 self。
        assert isinstance(patch.final_patch_func, staticmethod)
        assert Holder.sm('a') == 'patched'
        assert Holder().sm('a') == 'patched'  # 修复前以普通函数安装，此处 TypeError
        patch.remove_patch()
        assert isinstance(Holder.__dict__['sm'], staticmethod)
        assert Holder.sm(10, y=3) == {'x': 10, 'y': 3}
        assert Holder().sm(10) == {'x': 10, 'y': 2}
        # 多轮往返稳定：remove 后重新注册再应用，描述符不再次丢失。
        patch.set_patch_func(_patched)
        patch.apply_patch()
        assert Holder().sm('b') == 'patched'
        patch.remove_patch()
        assert Holder().sm(1) == {'x': 1, 'y': 2}

    def test_classmethod_binding_survives_roundtrip(self):
        def _cm(cls, x):
            return (cls.__name__, x)

        def _patched_cm(cls, x):
            return ('patched', cls)

        target, Holder = self._install_class()  # noqa: F841
        Holder.cm = classmethod(_cm)
        patch = Patch(f'{TARGET_MOD}.Holder.cm', _patched_cm, False)
        patch.apply_patch()
        assert isinstance(patch.final_patch_func, classmethod)
        assert Holder.cm('a') == ('patched', Holder)
        assert Holder().cm('a') == ('patched', Holder)
        patch.remove_patch()
        assert isinstance(Holder.__dict__['cm'], classmethod)
        assert Holder.cm(1) == ('Holder', 1)
        assert Holder().cm(1) == ('Holder', 1)

    @pytest.mark.parametrize('wrapped', [False, True], ids=['bare_func', 'classmethod_wrapped'])
    def test_c_level_classmethod_descriptor_shape_preserved(self, wrapped):
        """C 层 classmethod_descriptor 目标（如 object.__init_subclass__）保持 classmethod 形状。

        回归背景：_build_final_func 会 strip 注册处 classmethod 包装，而原描述符是 C 层
        classmethod_descriptor（非 classmethod 实例），wrap_descriptor 匹配不上便以裸函数
        装回；CPython type.__new__ 创建子类时不绑定 cls，make_dataclass 建子类直接报
        "missing 1 required positional argument: 'cls'"（TransformerConfig 实际故障）。
        """
        target, Holder = self._install_class()  # noqa: F841
        # 目标必须是继承自 object 的 C 层 classmethod_descriptor，而非 Python 层 classmethod。
        assert '__init_subclass__' not in Holder.__dict__
        assert isinstance(object.__dict__['__init_subclass__'], types.ClassMethodDescriptorType)

        received = []

        def patched_init_subclass(cls, **kwargs):
            received.append(cls)

        # 注册形式与生产侧（megatron_basic.py）一致：显式 classmethod 包装。
        new_func = classmethod(patched_init_subclass) if wrapped else patched_init_subclass
        patch = Patch(f'{TARGET_MOD}.Holder.__init_subclass__', new_func, False)
        try:
            patch.apply_patch()
            # 修复前此处为裸函数：子类创建时 cls 不再自动绑定。
            assert isinstance(Holder.__dict__['__init_subclass__'], classmethod)

            class Child(Holder):
                pass

            # 绑定必须注入 cls（新子类）；修复前此处抛 TypeError: missing 'cls'。
            assert received == [Child]
        finally:
            patch.remove_patch()
        # remove 后按捕获的原始描述符还原（C 层 classmethod_descriptor 装回类上语义等价于继承）。
        assert isinstance(Holder.__dict__['__init_subclass__'], types.ClassMethodDescriptorType)

    @pytest.mark.parametrize('instance_access', [False, True], ids=['class', 'instance'])
    def test_classmethod_wrapper_binds_class_once(self, instance_access):
        """classmethod 目标叠加装饰器时，转发器不得给原方法注入第二个 cls；
        类访问与实例访问都只绑定一次 cls。
        """

        def _cm(cls, x):
            return (cls, x)

        def forwarding_wrapper(func):
            def wrapped(*args, **kwargs):
                return ('wrapped', func(*args, **kwargs))

            return wrapped

        _, Holder = self._install_class()
        Holder.cm = classmethod(_cm)
        receiver = Holder() if instance_access else Holder
        patch = Patch(f'{TARGET_MOD}.Holder.cm', forwarding_wrapper, False)
        try:
            patch.apply_patch()
            # Forwarding must not pass a second cls to the original method.
            assert receiver.cm('value') == ('wrapped', (Holder, 'value'))
        finally:
            patch.remove_patch()
        assert receiver.cm('value') == (Holder, 'value')

    @pytest.mark.parametrize('remove_after_replacement', [False, True], ids=['replaced', 'removed'])
    def test_staticmethod_alias_tracks_forced_replacement(self, remove_after_replacement):
        """staticmethod 目标被 force 替换后别名须同步刷新：主目标、类与别名三方行为一致；
        随后 remove 时一并还原原值，防止替换失败掩盖残留旧别名。
        """

        def _sm(x):
            return ('original', x)

        def first_patch(x):
            return ('first', x)

        def second_patch(x):
            return ('second', x)

        _, Holder = self._install_class()
        Holder.sm = staticmethod(_sm)
        alias = sys.modules[ALIAS_MOD]
        _set_attr(alias, 'sm', Holder.sm)
        patch = Patch(f'{TARGET_MOD}.Holder.sm', first_patch, False)
        try:
            patch.apply_patch()
            patch.set_patch_func(second_patch, force_patch=True)
            patch.apply_patch()
            expected = ('second', 'value')
            if remove_after_replacement:
                patch.remove_patch()
                expected = ('original', 'value')
            assert Holder.sm('value') == expected
            assert Holder().sm('value') == expected
            # Check removal independently so replacement failure cannot hide a stale alias.
            assert alias.sm('value') == expected
        finally:
            patch.remove_patch()

    def test_apply_skips_entry_cleared_by_remove(self):
        # remove 后注册表残留但未重新注册：apply 不得用 orig_func 提前重装，
        # 否则会覆盖外部（如 MindSpeed pre 阶段）已装好的实现。
        def _sm(x):
            return {'x': x}

        target, Holder = self._install_class()  # noqa: F841
        Holder.sm = staticmethod(_sm)
        patch = Patch(f'{TARGET_MOD}.Holder.sm', _patched, False)
        patch.apply_patch()
        patch.remove_patch()
        assert patch.patch_func is None and not patch.wrappers
        third_party = staticmethod(lambda v: ('third', v))
        Holder.sm = third_party
        patch.apply_patch()
        assert Holder.__dict__['sm'] is third_party
        assert not patch.is_applied

    def test_module_slots_receive_bare_callable(self):
        """模块属性查找不走 __get__：classmethod 对象永不可调用，staticmethod 对象虽可调用但非裸函数；
        模块槽位（主目标与别名）统一收裸函数，类上保持描述符。
        """
        _, Holder = self._install_class()
        Holder.sm = staticmethod(lambda x: ('original', x))
        alias = sys.modules[ALIAS_MOD]
        _set_attr(alias, 'sm', Holder.sm)

        # 别名槽位：替换函数本身是普通函数。
        patch = Patch(f'{TARGET_MOD}.Holder.sm', _patched, False)
        patch.apply_patch()
        assert not isinstance(vars(alias)['sm'], (staticmethod, classmethod))
        assert alias.sm('a') == 'patched'
        assert isinstance(Holder.__dict__['sm'], staticmethod)
        patch.remove_patch()
        assert alias.sm(1) == ('original', 1)

        # 模块主目标：替换函数自带 @staticmethod，装进模块前必须剥掉描述符层。
        module_target = _register('ma_ut_module_slot', foo=_orig)
        slot_patch = Patch('ma_ut_module_slot.foo', staticmethod(_patched), False)
        slot_patch.apply_patch()
        assert not isinstance(vars(module_target)['foo'], (staticmethod, classmethod))
        assert get_module_attr(module_target, 'foo')('a') == 'patched'
        slot_patch.remove_patch()
        assert get_module_attr(module_target, 'foo')('a') == 'orig'

    def test_replacement_descriptor_yields_to_original_shape(self):
        """目标原为 classmethod 而替换函数自带 @staticmethod 时，以原始形状装回
        classmethod，cls 仍注入一次（旧实现直接透传 staticmethod 静默丢 cls）。
        """
        _, Holder = self._install_class()
        Holder.m = classmethod(lambda cls, x: ('orig', cls.__name__, x))
        replacement = staticmethod(lambda cls, x: ('repl', cls.__name__, x))
        patch = Patch(f'{TARGET_MOD}.Holder.m', replacement, False)
        patch.apply_patch()
        assert isinstance(Holder.__dict__['m'], classmethod)
        assert Holder().m('a') == ('repl', 'Holder', 'a')
        patch.remove_patch()
        assert Holder().m('a') == ('orig', 'Holder', 'a')

    def test_reapply_keeps_original_shape_after_third_party_reshape(self):
        """remove 后目标被第三方改成其他形状时，re-apply 必须以捕获的原始描述符为准，
        避免把第三方形状误当原始形状、以普通函数装回类导致实例访问注入 self。
        """
        _, Holder = self._install_class()
        Holder.sm = staticmethod(lambda x: ('original', x))
        patch = Patch(f'{TARGET_MOD}.Holder.sm', _patched, False)
        patch.apply_patch()
        patch.remove_patch()

        def third_party_method(self, x):
            return ('third', self, x)

        Holder.sm = third_party_method
        patch.set_patch_func(_patched)
        patch.apply_patch()
        assert isinstance(Holder.__dict__['sm'], staticmethod)
        assert Holder().sm('a') == 'patched'
        patch.remove_patch()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
