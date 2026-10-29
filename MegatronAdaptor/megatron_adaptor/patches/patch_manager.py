import functools
import importlib
import inspect
import sys
import types
from typing import Dict, Union, List


def get_func_name(func):
    if isinstance(func, str):
        return func
    return '.'.join((func.__module__, func.__qualname__))


def dummy_function_wrapper(func_name):
    def dummy_function(*args, **kwargs):
        raise RuntimeError('function {} no exist'.format(func_name))

    return dummy_function


# Sentinel distinguishing a missing attribute from one whose value is None.
MISSING_ATTR = object()

_METHOD_DESCRIPTOR_TYPES = (staticmethod, classmethod)


def get_module_attr(value, name):
    """Look up only the module's own __dict__; never trigger the module-level
    __getattr__ (PEP 562). Returns MISSING_ATTR for non-modules or missing
    attributes, separating "missing" from "None" so the id-based match never
    hits every module.
    """
    if isinstance(value, types.ModuleType):
        return vars(value).get(name, MISSING_ATTR)
    return MISSING_ATTR


def probe_attr(value, name):
    """Probe a binding without risking arbitrary __getattr__ side effects:
    modules are read only via their __dict__ (never triggering the module-level
    __getattr__, PEP 562); non-module objects (e.g. class objects) keep the
    regular getattr fallback (class attribute chains / metaclasses). Missing
    returns MISSING_ATTR.
    """
    if isinstance(value, types.ModuleType):
        return get_module_attr(value, name)
    return getattr(value, name, MISSING_ATTR)


def get_static_attr(target, name, default=MISSING_ATTR):
    """Read a class attribute without __get__ unpacking, else a safe probe.
    Missing -> default.
    """
    if inspect.isclass(target):
        return inspect.getattr_static(target, name, default)
    attr = probe_attr(target, name)
    return default if attr is MISSING_ATTR else attr


def capture_descriptor(target, name, fallback):
    """Capture the original class attribute without __get__ unpacking;
    returns fallback when the attribute is missing.
    """
    if name is not None:
        raw = get_static_attr(target, name)
        if raw is not MISSING_ATTR:
            return raw
    return fallback


def strip_method_descriptor(value):
    """Return the callable behind a staticmethod/classmethod object, else value.

    Keeps a bound method bound — only the descriptor layer goes, which is what module
    ``__dict__`` slots and wrapper stacking need.
    """
    if isinstance(value, _METHOD_DESCRIPTOR_TYPES):
        return value.__func__
    return value


def wrap_descriptor(descriptor, func):
    """Make func bind like the original attribute did; safe to call twice.

    The original shape always wins. C-level classmethod descriptors (e.g.
    ``object.__init_subclass__``) are not ``classmethod`` instances and cannot be
    rebuilt by ``type(descriptor)(func)``, so re-wrap as a plain ``classmethod``.
    """
    # Normalize the input to a bare function; a bound method stays bound on purpose.
    func = strip_method_descriptor(func)
    if isinstance(descriptor, _METHOD_DESCRIPTOR_TYPES):
        return type(descriptor)(func)
    if isinstance(descriptor, types.ClassMethodDescriptorType):
        return classmethod(func)
    return func


def unwrap_binding(value):
    """Strip descriptor (staticmethod/classmethod) and bound-method layers so
    identity comparison reaches the underlying function no matter how a module,
    class or alias stores the binding.

    Unlike strip_method_descriptor, a bound method is also flattened here.
    """
    if isinstance(value, types.MethodType):
        return value.__func__
    return strip_method_descriptor(value)


def iter_alias_modules():
    """Yield (name, module) for loaded modules that may hold aliases of a patch target.

    apply_patch and remove_patch must use the same filter: an alias rewritten on
    apply must also be visible on remove. Skips MA/MindSpeed namespaces (their own
    symbols go through the captured holder) and non-module entries.
    """
    for name, value in sys.modules.copy().items():
        if 'megatron_adaptor' in name or 'mindspeed' in name or name == 'torch.classes':
            continue
        if not isinstance(value, types.ModuleType):
            continue
        yield name, value


class Patch:
    def __init__(self, orig_func_name, new_func, create_dummy):
        split_name = orig_func_name.rsplit('.', 1)
        if len(split_name) == 1:
            self.orig_module_name, self.orig_func_name = orig_func_name, None
        else:
            self.orig_module_name, self.orig_func_name = split_name
        self.orig_module = None
        self.orig_func = None
        self.orig_descriptor = None
        self.patch_func = None
        self.final_patch_func = None
        self.wrappers = []
        if new_func is None:
            new_func = dummy_function_wrapper(orig_func_name)
        self.set_patch_func(new_func)
        self.is_applied = False
        self.create_dummy = create_dummy

    @property
    def orig_func_id(self):
        return id(self.orig_func)

    @property
    def patch_func_id(self):
        return id(self.patch_func)

    def set_patch_func(self, new_func, force_patch=False):
        if hasattr(new_func, '__name__') and new_func.__name__.endswith(('wrapper', 'decorator')):
            if new_func not in self.wrappers:
                self.wrappers.append(new_func)
        else:
            if self.patch_func and not force_patch:
                raise RuntimeError('the patch of {} exist !'.format(self.orig_func_name))
            self.patch_func = new_func
        self.is_applied = False

    def remove_wrappers(self, wrapper_names: Union[str, List[str]] = None):
        if wrapper_names is None:
            self.wrappers.clear()
            return
        if isinstance(wrapper_names, str):
            wrapper_names = [wrapper_names]
        for name in wrapper_names:
            i = 0
            while i < len(self.wrappers):
                if self.wrappers[i].__name__ == name:
                    self.wrappers.pop(i)
                else:
                    i += 1

    def remove_patch(self):
        if self.orig_func_name is not None and self.final_patch_func is not None:
            installed_func = unwrap_binding(self.final_patch_func)
            # Own target, not an alias: restore unconditionally, even in a namespace the
            # alias scan skips or one already gone from sys.modules.
            self._restore_attr(self.orig_module, installed_func)
            class_name = self.orig_module_name.split('.')[-1] if inspect.isclass(self.orig_module) else None
            for _, module in iter_alias_modules():
                holder = module
                if class_name is not None:
                    found_class = get_module_attr(module, class_name)
                    if found_class is not MISSING_ATTR:
                        holder = found_class
                self._restore_attr(holder, installed_func)
        # Keep orig_module / orig_func / orig_descriptor: a re-apply must rebuild from the
        # true original shape, not re-capture whatever sits on the target by then.
        self.patch_func = None
        self.final_patch_func = None
        self.is_applied = False

    def _restore_attr(self, holder, installed_func):
        """Put the original binding back on one holder (module or class)."""
        if not isinstance(holder, types.ModuleType) and not inspect.isclass(holder):
            # Never probe an arbitrary object: its __getattr__ may raise.
            return
        if unwrap_binding(get_static_attr(holder, self.orig_func_name)) is not installed_func:
            # Not our binding, or a third party patched on top of it.
            return
        original = self.orig_descriptor if inspect.isclass(holder) else self.orig_func
        try:
            setattr(holder, self.orig_func_name, original)
        except (AttributeError, TypeError):
            # One unwritable alias must not abort the whole removal.
            pass

    def apply_patch(self):
        if self.is_applied:
            return
        # A removed-but-not-re-registered patch must not re-install orig_func/dummy,
        # which would overwrite the target (MindSpeed two-phase repatch, pre stage).
        if self.orig_module is not None and self.patch_func is None and not self.wrappers:
            return
        current_module, current_func = Patch.parse_path(self.orig_module_name, self.orig_func_name, self.create_dummy)
        if self.orig_module is None:
            self.orig_module, self.orig_func = current_module, current_func
        if self.orig_descriptor is None:
            self.orig_descriptor = capture_descriptor(current_module, self.orig_func_name, current_func)
        final_patch_func = self._build_final_func()
        if self.orig_func_name is not None:
            setattr(self.orig_module, self.orig_func_name, final_patch_func)
            # Alias sync: modules holding the same binding (from X import f) get the bare
            # callable — a module attribute is never unpacked by __get__. Match on the
            # current binding so a forced replacement also refreshes stale aliases.
            alias_func = strip_method_descriptor(final_patch_func)
            current_binding = unwrap_binding(current_func)
            for _, module in iter_alias_modules():
                if unwrap_binding(get_module_attr(module, self.orig_func_name)) is not current_binding:
                    continue
                try:
                    setattr(module, self.orig_func_name, alias_func)
                except (AttributeError, TypeError):
                    # Read-only module: the main target is already patched, keep going.
                    continue
        self.is_applied = True
        self.final_patch_func = final_patch_func

    def _build_final_func(self):
        """Assemble the callable to install, then repeat the original descriptor shape."""
        if self.patch_func is not None:
            # Wrappers must stack on a real callable; wrap_descriptor restores the shape.
            base = strip_method_descriptor(self.patch_func)
        else:
            # Stack on the unbound function: a classmethod is already bound to cls, and
            # re-wrapping it below would inject cls twice.
            base = unwrap_binding(self.orig_func)
        wrapped = functools.reduce(lambda acc, wrapper: wrapper(acc), self.wrappers, base)
        return wrap_descriptor(self.orig_descriptor, wrapped)

    @staticmethod
    def parse_path(module_path, function_name, create_dummy):
        from importlib.machinery import ModuleSpec

        modules = module_path.split('.')
        for i in range(1, len(modules) + 1):
            parent = '.'.join(modules[: i - 1])
            path = '.'.join(modules[:i])
            try:
                importlib.import_module(path)
            except ModuleNotFoundError as e:
                # Deliberately probe only the parent's __dict__: a module-level __getattr__ (PEP 562)
                # must never be triggered here (it may raise, e.g. uninitialized proxy modules).
                # Trade-off: benign lazy attributes exposed solely via __getattr__ are treated as missing (dummy path).
                if not parent or get_module_attr(importlib.import_module(parent), modules[i - 1]) is MISSING_ATTR:
                    if not create_dummy:
                        raise ModuleNotFoundError(e) from e
                    sys.modules[path] = types.ModuleType(path)
                    sys.modules[path].__file__ = 'mindspeed.dummy_module.py'
                    sys.modules[path].__spec__ = ModuleSpec(path, None)
                    if parent:
                        setattr(importlib.import_module(parent), modules[i - 1], sys.modules[path])
                else:
                    # The attribute lives in the parent's __dict__ (checked above); probe_attr
                    # handles the value being a module or a class object.
                    module = get_module_attr(importlib.import_module(parent), modules[i - 1])
                    attr = probe_attr(module, function_name)
                    if attr is not MISSING_ATTR:
                        return module, attr
                    if create_dummy:
                        return module, dummy_function_wrapper(function_name)
                    raise RuntimeError('no exist {} of {}'.format(function_name, module))

        target_module = sys.modules[module_path]
        if function_name is not None and get_module_attr(target_module, function_name) is MISSING_ATTR:
            setattr(target_module, function_name, None)
        return target_module, (get_module_attr(target_module, function_name) if function_name is not None else None)


class PatchesManager:
    patches_info: Dict[str, Patch] = {}

    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        """Patch registration method. When this method is executed, the patch does not take effect in real time.
        It takes effect only after the apply_patches method is invoked. Other details are as follows:

        1. If `orig_func_name` does not exist and create_dummy is set to True, a dummy function is created to ensure
        that the import is normal.
        2. If `orig_func_name` is not None, `orig_func_name` is replaced with `new_func`.
        3. If the `new_func` function name ends with `wrapper` or `decorator`, then `new_func` is decorated on
        `orig_func_name` as a decorator, and the decorator can be superimposed repeatedly.
        4. When force_patch=False, a function cannot be replaced repeatedly (but can be decorated repeatedly),
        otherwise the replacement is overwritten.
        """
        if orig_func_name not in PatchesManager.patches_info:
            PatchesManager.patches_info[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            PatchesManager.patches_info.get(orig_func_name).set_patch_func(new_func, force_patch)

    @staticmethod
    def remove_wrappers(orig_func_name, wrappers_name, remove_check=True):
        """Remove wrapper registered in orig_func_name."""
        if orig_func_name not in PatchesManager.patches_info:
            raise ValueError('The function <{}> not exist.'.format(orig_func_name))
        patch = PatchesManager.patches_info.get(orig_func_name)
        wrappers_len = len(patch.wrappers)
        patch.remove_wrappers(wrappers_name)
        if remove_check and wrappers_len == len(patch.wrappers):
            raise RuntimeError('Remove wrappers has not remove anything.')

    @staticmethod
    def remove_patches():
        for patch in PatchesManager.patches_info.values():
            patch.remove_patch()
            patch.remove_wrappers()

    @staticmethod
    def apply_patches():
        for patch in PatchesManager.patches_info.values():
            patch.apply_patch()

    @staticmethod
    def get_patch(orig_func_name):
        return PatchesManager.patches_info.get(orig_func_name)
