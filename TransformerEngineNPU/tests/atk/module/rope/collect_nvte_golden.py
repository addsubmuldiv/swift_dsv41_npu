# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Collect NVTE RoPE P0 golden outputs in ATK ``accuracy_load`` layout.

Run on a CUDA host with the pinned NVIDIA Transformer Engine importable as
``transformer_engine``. Rejects a TENPU import, records import path/version and
input SHA256, and writes NVTE outputs in the layout expected by the ``cpu_nvte``
accuracy-load node. Shares the exact device execution paths (``run_rope_case``
/ ``run_rope_fused_qkv_case``) used on the NPU side; per-case dispatch mirrors
the adapter (first string attr: "rope" vs "rope_fused_qkv").
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

from function_rope import (  # noqa: E402
    RESULT_SCHEMA,
    RESULT_SCHEMA_FUSED_QKV,
    run_rope_case,
    run_rope_fused_qkv_case,
)


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
    parser.add_argument("--cases", required=True, help="ATK generated all_rope.json")
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
        # Dispatch mirrors the adapter: the variant attr sits at position 3 for
        # the plain shape (t, freqs, dy, variant, ...) and position 4 for the
        # fused-QKV shape (qkv, q_freqs, k_freqs, dy, variant, ...).
        arg_list = list(dataset.args)
        variant = ""
        for probe in (3, 4):
            if len(arg_list) > probe and isinstance(arg_list[probe], str):
                variant = str(arg_list[probe])
                break
        try:
            if variant == "rope_fused_qkv":
                outputs = run_rope_fused_qkv_case(dataset, args.device)
                schema = RESULT_SCHEMA_FUSED_QKV
            else:
                outputs = run_rope_case(dataset, args.device)
                schema = RESULT_SCHEMA
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
                "schema": schema,
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
        "result_schema_fused_qkv": RESULT_SCHEMA_FUSED_QKV,
        "skipped": skipped,
        "records": records,
    }
    golden_root.mkdir(parents=True, exist_ok=True)
    manifest_path = golden_root / "rope_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print("Collected {} NVTE RoPE cases into {}".format(len(records), golden_root))


if __name__ == "__main__":
    main()
