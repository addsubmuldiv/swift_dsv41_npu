#!/usr/bin/env python3
"""Stage ②.5 SIGCHECK — Deterministic signature cross-check. Zero LLM involvement.

Parses all register_patch(orig_func_name, ...) calls from MegatronAdaptor source,
inspects the CURRENT Megatron function/method signatures, and cross-checks each
against the adapter's replacement/wrapper definition in the adaptor source.

A target is flagged CHANGED when the current Megatron signature exposes a parameter
that the adapter replacement does not accept (by name) — the replacement would fail
to bind at call time. This catches upstream signature additions / renames before
LAUNCH wastes training time.

Comparison rules (designed against real patches, no false positives on them):
- Parameter names are compared with trailing underscores stripped ('input_' ≡ 'input'),
  so adapters that rename to avoid shadowing builtins are not false-flagged.
- Decorator-style wrappers (*_wrapper / *_decorator) take (func) and re-dispatch —
  their own signature is not comparable; treated as compatible.
- Replacements declaring **kwargs accept any keyword — treated as compatible.
- Class targets are compared per method: only methods present on BOTH the upstream
  class and the adapter class are checked.
- Targets whose module/function is not importable (skipped) or is a C builtin
  (uninstructable) are NOT treated as changes.

Runs after SETUP (pip install done), before LAUNCH.
"""

import argparse
import ast
import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_WRAPPER_SUFFIXES = ("_wrapper", "_decorator")
_SELF_PARAMS = {"self", "cls"}


def _norm(name: str) -> str:
    """Normalize a parameter name: strip trailing underscores ('input_' ≡ 'input')."""
    return name.rstrip("_")


def _sig_to_contract(sig: inspect.Signature) -> dict:
    """Split an inspect.Signature into a comparable contract.

    Returns {"required": set, "optional": set, "has_args": bool, "has_kwargs": bool}.
    self/cls are excluded; *args/**kwargs are recorded as flags.
    """
    required, optional = set(), set()
    has_args = has_kwargs = False
    for pname, p in sig.parameters.items():
        if pname in _SELF_PARAMS:
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            has_args = True
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            has_kwargs = True
        elif p.default is inspect.Parameter.empty:
            required.add(_norm(pname))
        else:
            optional.add(_norm(pname))
    return {"required": required, "optional": optional, "has_args": has_args, "has_kwargs": has_kwargs}


def get_registered_targets(adaptor_path: str) -> dict:
    """Extract all register_patch(orig_func_name, new_func_name) calls.

    Returns {orig_func_name: new_func_name} mapping.
    """
    targets = {}
    patches_dir = Path(adaptor_path) / "megatron_adaptor"

    for py_file in patches_dir.rglob("*.py"):
        try:
            content = py_file.read_text()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_register = False
            if isinstance(func, ast.Attribute) and func.attr == "register_patch":
                is_register = True
            elif isinstance(func, ast.Name) and func.id == "register_patch":
                is_register = True
            if not is_register or len(node.args) < 1:
                continue

            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                orig_name = first_arg.value
                new_func = orig_name.rsplit(".", 1)[-1]  # default fallback
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Name):
                    new_func = node.args[1].id
                for kw in node.keywords:
                    if kw.arg == "new_func" and isinstance(kw.value, ast.Name):
                        new_func = kw.value.id
                targets[orig_name] = new_func

    return targets


def _ast_params(func_node) -> dict:
    """Contract from an AST function node (adapter source, no runtime import)."""
    required, optional = set(), set()
    has_args = bool(func_node.args.vararg)
    has_kwargs = bool(func_node.args.kwarg)
    defaults = func_node.args.defaults or []
    ndefaults = len(defaults)
    args_with_default = set()
    # defaults align to the LAST n positional args
    if ndefaults:
        args_with_default = {a.arg for a in func_node.args.args[-ndefaults:]}
    for arg in func_node.args.posonlyargs + func_node.args.args:
        if arg.arg in _SELF_PARAMS:
            continue
        if arg.arg in args_with_default:
            optional.add(_norm(arg.arg))
        else:
            required.add(_norm(arg.arg))
    kw_defaults = func_node.args.kw_defaults or []
    kw_offset = len(func_node.args.kwonlyargs) - len(kw_defaults)
    for i, arg in enumerate(func_node.args.kwonlyargs):
        if arg.arg in _SELF_PARAMS:
            continue
        if kw_defaults and i >= kw_offset and kw_defaults[i] is not None:
            optional.add(_norm(arg.arg))
        else:
            required.add(_norm(arg.arg))
    return {"required": required, "optional": optional, "has_args": has_args, "has_kwargs": has_kwargs}


def get_adaptor_definitions(adaptor_path: str) -> dict:
    """Walk MegatronAdaptor source; collect definitions by name (AST, no import).

    Returns {name: {
        "kind": "function" | "class",
        "required": set, "optional": set, "has_args": bool, "has_kwargs": bool,
        "methods": {method_name: contract},   # classes only
    }}
    """
    defs = {}
    root = Path(adaptor_path) / "megatron_adaptor"

    def _record_func(name, contract):
        defs.setdefault(name, {"kind": "function", **contract, "methods": {}})

    for py_file in root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _record_func(node.name, _ast_params(node))
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[sub.name] = _ast_params(sub)
                defs[node.name] = {
                    "kind": "class",
                    "required": set(),
                    "optional": set(),
                    "has_args": False,
                    "has_kwargs": False,
                    "methods": methods,
                }
    return defs


def inspect_target_signature(orig_func_name: str) -> dict:
    """Resolve the current Megatron target and inspect its signature.

    The dotted path may point to a module-level function, a class, or a class
    method. We import the longest importable module prefix, then walk the
    remaining attribute chain.
    """
    parts = orig_func_name.split(".")
    if len(parts) < 2:
        return {"status": "skipped", "reason": f"cannot parse: {orig_func_name}"}

    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError):
            continue

        obj = module
        for attr in parts[i:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                return {
                    "status": "skipped",
                    "reason": f"'{'.'.join(parts[i:])}' not found in {module_path}",
                }
        return _describe_callable(obj)

    return {"status": "skipped", "reason": f"no importable module prefix for {orig_func_name}"}


def _describe_callable(obj) -> dict:
    """Inspect a resolved upstream function/class into a comparable contract."""
    if inspect.isclass(obj):
        methods = {}
        for name in dir(obj):
            member = getattr(obj, name, None)
            if not (inspect.isfunction(member) or inspect.ismethod(member)):
                continue
            try:
                methods[name] = _sig_to_contract(inspect.signature(member))
            except (ValueError, TypeError):
                continue
        return {
            "status": "ok",
            "kind": "class",
            "required": set(),
            "optional": set(),
            "has_args": False,
            "has_kwargs": False,
            "methods": methods,
        }

    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        # C built-ins cannot be introspected; NOT a signature change.
        return {"status": "uninstructable", "reason": str(sys.exc_info()[1])}
    contract = _sig_to_contract(sig)
    return {
        "status": "ok",
        "kind": "function",
        "methods": {},
        **contract,
    }


def check_compatible(upstream: dict, adapter: dict, new_func: str) -> tuple[str, str]:
    """Cross-check an upstream contract against the adapter replacement.

    Returns (status, reason); status in {"ok", "changed"}.
    """
    if adapter["has_kwargs"]:
        return "ok", "replacement accepts **kwargs (flexible)"
    missing = (upstream["required"] | upstream["optional"]) - (adapter["required"] | adapter["optional"])
    if missing:
        return (
            "changed",
            f"Megatron params not accepted by adapter '{new_func}': {sorted(missing)}",
        )
    return "ok", "signatures compatible"


def check_class_compatible(upstream: dict, adapter: dict, new_func: str) -> tuple[str, str]:
    """Cross-check class targets by their shared methods."""
    changed = []
    up_methods = upstream.get("methods", {})
    for mname, adef in adapter.get("methods", {}).items():
        um = up_methods.get(mname)
        if um is None:
            continue  # adapter patches a method upstream lacks — not a break signal
        if adef["has_kwargs"]:
            continue
        missing = (um.get("required", set()) | um.get("optional", set())) - (adef["required"] | adef["optional"])
        if missing:
            changed.append(f"{mname}({sorted(missing)})")
    if changed:
        return "changed", f"class '{new_func}' methods miss Megatron params: {changed}"
    return "ok", "class methods compatible"


def _jsonable(contract: dict) -> dict:
    """Convert contract sets to lists for JSON serialization."""
    return {
        "required": sorted(contract.get("required", [])),
        "optional": sorted(contract.get("optional", [])),
        "has_args": contract.get("has_args", False),
        "has_kwargs": contract.get("has_kwargs", False),
        "methods": {k: _jsonable(v) for k, v in contract.get("methods", {}).items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check MegatronAdaptor patch targets against current Megatron signatures"
    )
    parser.add_argument("--adaptor-path", required=True, help="Path to MegatronAdaptor repository")
    parser.add_argument("--env-report", required=True, help="env_report.json from Stage ①")
    parser.add_argument("--output", required=True, help="Output sigcheck_report.json path")
    args = parser.parse_args()

    with open(args.env_report, encoding="utf-8") as f:
        env_data = json.load(f)

    check_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    targets = get_registered_targets(args.adaptor_path)
    adaptor_defs = get_adaptor_definitions(args.adaptor_path)

    results = {}
    changed = []
    ok_count = 0
    skipped_count = 0
    uninstructable_count = 0

    for orig_name, new_func in sorted(targets.items()):
        sig_info = inspect_target_signature(orig_name)
        adapter = adaptor_defs.get(new_func)

        entry = {
            "new_func": new_func,
            "kind": sig_info.get("kind"),
            "adapter_def": adapter.get("kind") if adapter else None,
        }

        if sig_info["status"] == "skipped":
            entry["status"] = "skipped"
            entry["reason"] = sig_info.get("reason", "")
            skipped_count += 1
        elif sig_info["status"] == "uninstructable":
            entry["status"] = "uninstructable"
            entry["reason"] = sig_info.get("reason", "")
            uninstructable_count += 1
        else:  # ok — real comparison
            if adapter is None:
                entry["status"] = "skipped"
                entry["reason"] = f"adapter definition '{new_func}' not found in source"
                skipped_count += 1
            elif adapter["kind"] == "class":
                status, reason = check_class_compatible(sig_info, adapter, new_func)
                entry["status"] = status
                entry["reason"] = reason
                ok_count += 1 if status == "ok" else 0
                if status == "changed":
                    changed.append(orig_name)
            elif new_func.endswith(_WRAPPER_SUFFIXES):
                entry["status"] = "ok"
                entry["reason"] = "decorator-style wrapper (flexible signature)"
                ok_count += 1
            else:
                status, reason = check_compatible(sig_info, adapter, new_func)
                entry["status"] = status
                entry["reason"] = reason
                ok_count += 1 if status == "ok" else 0
                if status == "changed":
                    changed.append(orig_name)

        entry["params"] = sorted(sig_info.get("required", set()) | sig_info.get("optional", set()))
        if adapter is not None:
            entry["adapter_signature"] = _jsonable(adapter)
        results[orig_name] = entry

    overall = "pass" if len(changed) == 0 else "fail"

    report = {
        "check_time": check_time,
        "megatron_commit": env_data.get("megatron", {}).get("commit_sha", ""),
        "adaptor_path": str(Path(args.adaptor_path).resolve()),
        "targets_total": len(targets),
        "targets_ok": ok_count,
        "targets_skipped": skipped_count,
        "targets_uninstructable": uninstructable_count,
        "targets_changed": len(changed),
        "targets": results,
        "changed_targets": changed,
        "overall": overall,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"RESULT: {overall}")
    print(
        f"  Targets: {len(targets)} total, {ok_count} ok, "
        f"{skipped_count} skipped, {uninstructable_count} uninstructable, "
        f"{len(changed)} changed"
    )
    if changed:
        print(f"  Changed targets ({len(changed)}):")
        for c in changed:
            print(f"    - {c}: {results[c].get('reason', 'N/A')}")

    if overall == "fail":
        print("STOP: Signature changes detected. Blocking LAUNCH.", file=sys.stderr)


if __name__ == "__main__":
    main()
