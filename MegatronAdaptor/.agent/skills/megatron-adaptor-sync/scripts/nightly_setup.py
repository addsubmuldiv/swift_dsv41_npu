#!/usr/bin/env python3
"""Stage ② SETUP — Deterministic environment setup. Zero LLM involvement.

Installs MegatronAdaptor + TransformerEngineNPU (both from cloned paths in
env_report.json), configures NPU environment variables, and verifies NPU
availability.  Megatron-LM is NOT installed — it is only used as code
reference via PYTHONPATH.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: list, timeout: int = 300, env: dict = None) -> subprocess.CompletedProcess:
    """Run a subprocess command with error handling."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout, env=merged_env)
    except subprocess.CalledProcessError as e:
        return e
    except subprocess.TimeoutExpired:
        print(f"WARNING: Command timed out: {' '.join(cmd)}", file=sys.stderr)
        raise


def _write_failure_log(stderr: str, log_path: str = None) -> str:
    """Persist the full stderr of a failed command to ``log_path``.

    Returns the path that was written (or ``None`` if no path was given), so
    the report can reference it.  Ensures the parent directory exists.
    """
    if not log_path:
        return None
    try:
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(stderr or "", encoding="utf-8", errors="replace")
        return str(log)
    except OSError as e:
        print(f"WARNING: could not write failure log to {log_path}: {e}", file=sys.stderr)
        return None


def pip_install_editable(
    package_path: str, no_deps: bool = False, no_build_isolation: bool = False, log_path: str = None
) -> dict:
    """pip install -e a package and return result info.

    On failure the *full* stderr is written to ``log_path`` (if provided), so
    the report keeps a short error summary while the complete install log is
    preserved on disk for diagnosis.
    """
    cmd = [sys.executable, "-m", "pip", "install", "-e", package_path]
    if no_deps:
        cmd.insert(4, "--no-deps")
    if no_build_isolation:
        cmd.insert(4, "--no-build-isolation")
    result = run_cmd(cmd, timeout=600)
    success = result.returncode == 0
    log_file = None
    if not success:
        print(f"WARNING: pip install failed for {package_path}: {result.stderr[:500]}")
        log_file = _write_failure_log(result.stderr, log_path)
    return {
        "path": package_path,
        "installed": success,
        "error": result.stderr[:300] if not success else None,
        "log": log_file,
    }


def get_package_version(package_name: str) -> str:
    """Get installed package version via importlib."""
    try:
        result = run_cmd(
            [
                sys.executable,
                "-c",
                f"import importlib.metadata; print(importlib.metadata.version('{package_name}'))",
            ]
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass  # nosec B110 — version unknown is handled by the caller
    return "unknown"


def check_te_available() -> dict:
    """Verify TransformerEngine is importable and functional (REQUIRED path)."""
    check_script = """
import transformer_engine; print('importable')
try:
    ver = transformer_engine.__version__
    print(f'version={ver}')
except AttributeError:
    print('version=unknown')
"""
    result = run_cmd([sys.executable, "-c", check_script])
    available = result.returncode == 0 and "importable" in result.stdout
    version = "unknown"
    if available:
        for line in result.stdout.splitlines():
            if line.startswith("version="):
                version = line.split("=", 1)[1]
    return {"available": available, "version": version}


def check_npu_availability() -> dict:
    """Verify NPU is available via torch.npu."""
    check_script = """
import torch
try:
    import torch_npu
    npu_available = torch.npu.is_available()
    device_count = torch.npu.device_count() if npu_available else 0
    device_name = torch.npu.get_device_name(0) if device_count > 0 else "N/A"
    print(f"npu_available={npu_available}")
    print(f"device_count={device_count}")
    print(f"device_name={device_name}")
except ImportError as e:
    print(f"error={e}")
"""
    result = run_cmd([sys.executable, "-c", check_script])
    info = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            info[key.strip()] = val.strip()

    if "npu_available" in info:
        info["npu_available"] = info["npu_available"] == "True"
        info["device_count"] = int(info.get("device_count", 0))
    else:
        info["npu_available"] = False
        info["device_count"] = 0

    return info


def get_python_torch_versions() -> dict:
    """Get Python, torch, and torch_npu versions."""
    check_script = """
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
try:
    import torch_npu
    print(f"torch_npu={torch_npu.__version__}")
except ImportError:
    print("torch_npu=not_installed")
"""
    result = run_cmd([sys.executable, "-c", check_script])
    versions = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            versions[key.strip()] = val.strip()
    return versions


def load_npu_env_vars(adaptor_path: str) -> dict:
    """Parse NPU environment variables from env_npu.sh."""
    env_sh = Path(adaptor_path) / "test" / "system_tests" / "env_npu.sh"
    env_vars = {}

    if not env_sh.exists():
        print(f"WARNING: env_npu.sh not found at {env_sh}")
        env_vars["ASCEND_LAUNCH_BLOCKING"] = "0"
        env_vars["TASK_QUEUE_ENABLE"] = "1"
        env_vars["HCCL_WHITELIST_DISABLE"] = "1"
        return env_vars

    for line in env_sh.read_text().splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            decl = line[len("export ") :]
            if "=" in decl:
                key, val = decl.split("=", 1)
                val = val.strip('"').strip("'")
                env_vars[key] = val
    return env_vars


def smoke_test_adaptor_import(log_path: str = None) -> dict:
    """2-second import smoke test — exposes import chain breakage before training.

    On failure the *full* stderr (including the traceback) is written to
    ``log_path`` (if provided) so the report can carry a compact one-line
    error while the complete context survives on disk for diagnosis.
    """
    check_script = "import megatron_adaptor; print('ok')"
    result = run_cmd([sys.executable, "-c", check_script], timeout=120)
    passed = result.returncode == 0 and "ok" in result.stdout
    error = None
    log_file = None
    if not passed:
        stderr_text = result.stderr or ""
        log_file = _write_failure_log(stderr_text, log_path)
        stderr_lines = [line for line in stderr_text.splitlines() if line.strip()]

        # Search for the actual exception.  The last stderr line is often a
        # harmless DeprecationWarning; the real ImportError / Traceback is
        # buried in the middle.  Scan from the end for known exception types.
        EXCEPTION_SIGNATURES = [
            "ImportError:",
            "ModuleNotFoundError:",
            "AttributeError:",
            "TypeError:",
            "ValueError:",
            "RuntimeError:",
            "NameError:",
            "SyntaxError:",
            "FileNotFoundError:",
            "KeyError:",
        ]
        for line in reversed(stderr_lines):
            if any(sig in line for sig in EXCEPTION_SIGNATURES):
                error = line.strip()
                break

        if not error:
            # Fallback: last line that mentions "error" but isn't a deprecation warning
            for line in reversed(stderr_lines):
                lowered = line.lower()
                if "error" in lowered and "deprecationwarning" not in lowered:
                    error = line.strip()
                    break

        if not error:
            error = stderr_lines[-1] if stderr_lines else f"exit_code={result.returncode}"
    return {"passed": passed, "error": error, "log": log_file}


def check_dataset(dataset_path: str) -> dict:
    """Verify dataset files exist."""
    dataset = Path(dataset_path)

    candidates = [
        dataset,
        Path(str(dataset_path) + "_text_document.bin"),
        Path(str(dataset_path) + "_text_document.idx"),
        Path(str(dataset_path) + ".bin"),
        Path(str(dataset_path) + ".idx"),
    ]

    found = [str(c) for c in candidates if c.exists()]

    return {
        "path": str(dataset_path),
        "exists_on_disk": dataset.exists() or len(found) > 0,
        "found_files": found[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Setup environment for MegatronAdaptor nightly validation")
    parser.add_argument("--env-report", required=True, help="env_report.json from Stage ① FETCH")
    parser.add_argument("--output", required=True, help="Output setup_report.json path")
    parser.add_argument(
        "--dataset-path", default="/home/dataset/llama2/alpaca_text_document", help="Training dataset path"
    )
    args = parser.parse_args()

    with open(args.env_report, encoding="utf-8") as f:
        env_data = json.load(f)

    setup_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    errors = []

    # Failure logs (full stderr of pip installs / import smoke test) are
    # written next to the report so a failure carries complete context.
    output_dir = Path(args.output).parent
    adaptor_pip_log = str(output_dir / "pip_install_megatron_adaptor.log")
    te_npu_pip_log = str(output_dir / "pip_install_transformer_engine_npu.log")
    smoke_log = str(output_dir / "smoke_test_import_stderr.log")

    # ── Resolve repo paths from env_report.json ──
    # Prefer clone_dir (set by updated ① FETCH), fall back to path (legacy).
    adaptor_path = env_data.get("adaptor", {}).get("clone_dir", "") or env_data.get("adaptor", {}).get("path", "")
    te_npu_path = env_data.get("transformer_engine_npu", {}).get("clone_dir", "")
    # Legacy fallback: if TE-NPU not in env_report, derive from adaptor path
    if not te_npu_path and adaptor_path:
        te_npu_path = str(Path(adaptor_path).parent / "TransformerEngineNPU")

    # NOTE: Megatron-LM is NOT installed here.  It is only a code reference
    # (injected via PYTHONPATH by the orchestrator).

    # ── Install MegatronAdaptor ──
    adaptor_install = {}
    if adaptor_path and Path(adaptor_path).exists():
        adaptor_install = pip_install_editable(adaptor_path, log_path=adaptor_pip_log)
        if not adaptor_install["installed"]:
            errors.append(f"MegatronAdaptor install failed: {adaptor_install.get('error', '')}")
    else:
        errors.append(f"MegatronAdaptor path not found: {adaptor_path}")
        adaptor_install = {"path": adaptor_path, "installed": False}

    # ── Install TransformerEngineNPU (REQUIRED) ──
    te_npu_install = {}
    if te_npu_path and Path(te_npu_path).exists():
        # --no-deps: Conda env already provides torch/torch_npu/TE deps.
        # --no-build-isolation: use conda's setuptools, avoid pulling
        #   incompatible build deps from PyPI.
        te_npu_install = pip_install_editable(
            str(te_npu_path), no_deps=True, no_build_isolation=True, log_path=te_npu_pip_log
        )
        if not te_npu_install["installed"]:
            errors.append(f"TransformerEngineNPU install failed: {te_npu_install.get('error', '')}")
    else:
        errors.append(f"TransformerEngineNPU not found at {te_npu_path} — TE is REQUIRED")

    # ── Check TE status (MUST be available from installed TE-NPU) ──
    te_info = check_te_available()
    if not te_info["available"]:
        errors.append("TransformerEngine is NOT importable — TE is REQUIRED")

    # ── Get version info ──
    versions = get_python_torch_versions()

    # ── NPU check ──
    npu_info = check_npu_availability()
    if not npu_info.get("npu_available", False):
        errors.append("NPU is NOT available — cannot run training tests")

    # ── Load NPU env vars ──
    env_vars = {}
    if adaptor_path:
        env_vars = load_npu_env_vars(adaptor_path)

    # ── Dataset check ──
    dataset_info = check_dataset(args.dataset_path)
    if not dataset_info["exists_on_disk"]:
        errors.append(f"Dataset not found at {args.dataset_path}")

    # ── Import smoke test (左移: 2秒暴露 import 链断裂) ──
    adaptor_import_smoke = smoke_test_adaptor_import(log_path=smoke_log)
    if not adaptor_import_smoke["passed"]:
        smoke_hint = f" (full stderr: {adaptor_import_smoke.get('log')})" if adaptor_import_smoke.get("log") else ""
        errors.append(f"Adaptor import smoke test failed: {adaptor_import_smoke['error']}{smoke_hint}")

    # ── Build report ──
    status = "ok" if not errors else "fail"

    report = {
        "setup_time": setup_time,
        "packages": {
            "megatron_adaptor_installed": adaptor_install.get("installed", False),
            "adaptor_import_smoke": adaptor_import_smoke["passed"],
            "te_available": te_info["available"],
            "te_version": te_info.get("version", "unknown"),
            "te_npu_installed": te_npu_install.get("installed", False),
            "te_npu_path": te_npu_path,
            "te_path": "required",
            **versions,
        },
        "npu": npu_info,
        "dataset": dataset_info,
        "env_vars": env_vars,
        "status": status,
        "errors": errors,
        "logs": {
            "adaptor_pip_install": adaptor_install.get("log"),
            "te_npu_pip_install": te_npu_install.get("log"),
            "smoke_test_import": adaptor_import_smoke.get("log"),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"OK: setup_report.json written to {args.output}")
    print(f"  Status: {status}")
    print(f"  Adaptor installed: {report['packages']['megatron_adaptor_installed']}")
    print(f"  TE available: {te_info['available']} (MUST be True — TE path required)")
    print(f"  NPU available: {npu_info.get('npu_available', False)} (devices: {npu_info.get('device_count', 0)})")
    print(f"  Dataset: {dataset_info['exists_on_disk']}")

    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
