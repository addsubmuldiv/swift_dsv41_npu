# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Collect NVTE fused-optimizer P0 golden outputs in ATK ``accuracy_load`` layout.

This script is meant to run on a CUDA host that has the pinned NVIDIA
Transformer Engine build importable as ``transformer_engine``. It rejects a
TENPU import (path containing ``TransformerEngineNPU``), records the actual
import path/version and input SHA256 values, and writes NVTE outputs in the
layout expected by the ``cpu_nvte`` accuracy-load node. It shares the exact
device execution path (``run_fused_optimizer_case``) used on the NPU side, so
both sides run the same N optimizer steps on the same canonical inputs.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from atk.configs.case_config import CaseConfig
from atk.tasks.dataset.base_dataset import OpsDataset

from atk.case_generator.utils.plugins_register import init_registry


import sys

# Make the collector runnable from any working directory (e.g. the TENPU
# repo root): it imports its sibling adapter module by bare name.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from function_fused_optimizer import RESULT_SCHEMA, run_fused_optimizer_case  # noqa: E402


def _flatten_tensors(value) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_args(args):
    """Make ``dataset.args`` ``torch.load(weights_only=True)``-safe.

    ATK's ``OpsDataset`` samples ``float`` attrs (e.g. ``lr``/``weight_decay``)
    as ``numpy.float32`` scalars; ``torch.save`` pickles them verbatim, and ATK's
    device-side loader uses ``torch.load(weights_only=True)`` which refuses numpy
    globals. Convert numpy scalars to Python scalars (numerically identical) so
    the saved ``input.bin`` loads on both the golden and the NPU side.
    """
    import numpy as np

    def _clean(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list):
            return [_clean(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_clean(item) for item in value)
        return value

    return [_clean(value) for value in args]


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="ATK generated all_fused_optimizer.json")
    parser.add_argument("--input-dir", required=True, help="Canonical input artifact directory")
    parser.add_argument("--golden-dir", required=True, help="Root used by the accuracy_load node")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend-label", default="cpu_nvte")
    parser.add_argument(
        "--reject-import-token",
        default="TransformerEngineNPU",
        help="Fail if transformer_engine.__file__ contains this token; empty disables the check",
    )
    return parser.parse_args()


def main():
    # Plain-script entry: the parameter-type plugins are only registered by
    # the atk CLI's init_registry(); without it OpsDataset's materialization
    # dies on KeyError 'tensor' (pip-installed ATK on the golden host).
    init_registry()
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to collect NVTE golden outputs")

    import transformer_engine

    import_path = str(Path(transformer_engine.__file__).resolve())
    if args.reject_import_token and args.reject_import_token.lower() in import_path.lower():
        raise RuntimeError(
            "Expected NVTE, but imported transformer_engine from {}".format(import_path)
        )

    cases_path = Path(args.cases).resolve()
    input_root = Path(args.input_dir).resolve()
    golden_root = Path(args.golden_dir).resolve()
    save_name = cases_path.stem
    records = []
    skipped = []

    with cases_path.open("r", encoding="utf-8") as stream:
        raw_cases = json.load(stream)

    for raw_case in raw_cases:
        case = CaseConfig(**raw_case)
        dataset = next(OpsDataset([case.model_dump()], case.id))

        input_case_dir = input_root / str(case.id)
        input_case_dir.mkdir(parents=True, exist_ok=True)
        input_file = input_case_dir / "input.bin"
        torch.save(_sanitize_args(dataset.args), input_file)

        dataset.to_device(args.device)
        try:
            outputs = run_fused_optimizer_case(dataset, args.device)
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            skipped.append({"case_id": case.id, "reason": "cuda_oom"})
            print(
                "[skip] case {} hit CUDA OOM on the golden host; golden not "
                "collected (case JSON and ids unchanged)".format(case.id)
            )
            gc.collect()
            torch.cuda.empty_cache()
            continue

        output_dir = golden_root / args.backend_label / save_name / str(case.id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        for index, output in enumerate(_flatten_tensors(outputs)):
            output_file = output_dir / "output_{}.pt".format(index)
            torch.save(output.detach().cpu(), output_file)
            output_files.append(output_file.name)

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        records.append(
            {
                "case_id": case.id,
                "input_sha256": _sha256(input_file),
                "outputs": output_files,
                "schema": RESULT_SCHEMA,
            }
        )

    manifest = {
        "schema_version": 1,
        "backend": "nvte_cuda",
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
        "transformer_engine_import": import_path,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "cases_file": str(cases_path),
        "case_count": len(records),
        "result_schema": RESULT_SCHEMA,
        "skipped": skipped,
        "records": records,
    }
    golden_root.mkdir(parents=True, exist_ok=True)
    manifest_path = golden_root / "fused_optimizer_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print("Collected {} NVTE fused-optimizer cases into {}".format(len(records), golden_root))


if __name__ == "__main__":
    main()
