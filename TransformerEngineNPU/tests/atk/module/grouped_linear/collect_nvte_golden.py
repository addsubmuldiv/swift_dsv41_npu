# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Collect NVTE GroupedLinear golden outputs in ATK ``accuracy_load`` layout.

Run on a CUDA host with the pinned NVIDIA Transformer Engine importable as
``transformer_engine``. Rejects a TENPU import, records import path/version and
input SHA256, and writes outputs in the layout expected by the ``cpu_nvte``
accuracy-load node. Shares the exact device execution path
(``run_grouped_linear_case``) used on the NPU side.

The wgrad (and, when bias is on, bgrad) slots are collected as exact fp64
references rather than NVTE's kernel output — the same substitution the online
GPU side performs (``run_grouped_linear_case(..., exact_baselines=True)``), so
the offline and online golden paths cannot drift. The manifest records the
per-slot baseline kind and how far NVTE's own output sat from the baseline that
replaced it (``slot_baseline`` / ``device_vs_baseline``).
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
_MODULE_ROOT = str(Path(_HERE).parent)
for _path in (_HERE, _MODULE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from function_grouped_linear import (  # noqa: E402
    _BGRAD_SLOT,
    _WGRAD_SLOT,
    RESULT_SCHEMA,
    run_grouped_linear_case,
)
from _comparators import MixedToleranceAccuracy, load_design_acc_kwargs  # noqa: E402


def _comparator():
    """The module's own standard, so device-vs-baseline is judged as ATK would."""
    return MixedToleranceAccuracy(
        None, **load_design_acc_kwargs(Path(_HERE) / "grouped_linear.yaml")
    )


def _judge(raw: torch.Tensor, baseline: torch.Tensor, index: int, comparator) -> dict:
    """ATK-style verdict of NVTE's own output against the baseline that replaced it."""
    result = comparator.compute_accuracy_result(raw, baseline, "output_{}.pt".format(index))
    return {"result": bool(result.result), "error_info": result.error_info}


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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="ATK generated all_grouped_linear.json")
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
        torch.save(dataset.args, input_file)

        dataset.to_device(args.device)
        # The device's own slots are captured for provenance before the exact
        # references replace them; the substitution policy itself lives in
        # run_grouped_linear_case so the offline and online paths share one
        # definition.
        raw_slots = {}
        try:
            outputs = run_grouped_linear_case(
                dataset,
                args.device,
                exact_baselines=True,
                raw_slots_out=raw_slots,
            )
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
        # Adapter arg layout: inp, dy, weight_0..g-1, bias_0..g-1, num_gemms,
        # m_splits, bias_enabled — bias_enabled is the last arg.
        bias_enabled = bool(list(dataset.args)[-1])
        comparator = _comparator()

        slot_baseline = {}
        device_vs_baseline = {}
        for index, output in enumerate(_flatten_tensors(outputs)):
            torch.save(output.detach().cpu(), output_dir / "output_{}.pt".format(index))
            output_files.append("output_{}.pt".format(index))
            if index == _WGRAD_SLOT:
                slot_baseline[str(index)] = "exact_fp64_grouped_wgrad"
            elif index == _BGRAD_SLOT:
                slot_baseline[str(index)] = (
                    "exact_fp64_grouped_bgrad" if bias_enabled else "device_empty"
                )
            else:
                slot_baseline[str(index)] = "device"
            raw = raw_slots.get(index)
            if raw is not None:
                device_vs_baseline[str(index)] = _judge(
                    raw.detach().cpu(), output.detach().cpu(), index, comparator
                )

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        records.append(
            {
                "case_id": case.id,
                "input_sha256": _sha256(input_file),
                "outputs": output_files,
                "schema": RESULT_SCHEMA,
                "slot_baseline": slot_baseline,
                "device_vs_baseline": device_vs_baseline,
            }
        )

    manifest = {
        "schema_version": 2,
        "backend": "nvte_cuda",
        "backend_detail": "nvte_cuda device outputs + exact fp64 grouped wgrad/bgrad-slot baselines",
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
    manifest_path = golden_root / "grouped_linear_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print("Collected {} NVTE GroupedLinear cases into {}".format(len(records), golden_root))


if __name__ == "__main__":
    main()
