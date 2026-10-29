# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Shared offline-golden collector driver for the split LayerNorm/RMSNorm guards.

The split guards (``function_layernorm_forward/backward`` and their RMSNorm
counterparts) consume a serialized input bundle in the online mode: the NPU
producer materializes the source inputs, computes the shared FP32 statistics
once (fp64 on host), and ``--syc_dataset`` syncs the bundle to the GPU
receiver. This driver is the offline twin: it produces, on the CUDA host,

* ``<input-dir>/<case_id>/input.bin`` — the source args (plus the shared
  statistics for backward cases), materialized with the SAME OpsDataset recipe
  and the SAME ``compute_shared_statistics`` call as the online producer, so
  both tracks see bit-identical inputs;
* ``<cases-out>/<stem>.json`` — source-form case JSONs for the NPU task. The
  generated case files are in bundle form (one custom-dtype input); ATK's
  ``--input_data`` loader zips saved args against the case's input specs
  positionally, so backward cases get the statistics appended as explicit
  FP32 ``[rows]`` tensor specs (never sampled; ``input_data`` mode loads
  ``input.bin`` directly). The file is written once after the loop, so an
  interrupted run leaves it absent rather than truncated;
* ``<golden-dir>/<backend>/<stem>/<case_id>/output_N.pt`` — the NVTE outputs
  of the phase entry, plus a manifest with per-case input SHA256 and the
  OOM-skipped ids.

The bundle-form checks in the adapters only trigger for bundle-typed single
input specs; the source-form cases written here carry the original six/five
tensor/attr specs, so the offline ``accuracy_load`` path needs no
``--syc_dataset`` and no role environment variable.
"""

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from atk.case_generator.utils.plugins_register import init_registry
from atk.configs.case_config import CaseConfig
from atk.tasks.dataset.base_dataset import OpsDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_name(device):
    if str(device).startswith("npu"):
        return torch.npu.get_device_name(torch.npu.current_device())
    return torch.cuda.get_device_name(torch.device(device))


def _empty_cache(device):
    if str(device).startswith("npu"):
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tensors(item)


def _stat_spec(rows):
    """One appended FP32 ``[rows]`` spec for a shared-statistics slot."""
    return {
        "name": None,
        "type": "tensor",
        "required": True,
        "dtype": "fp32",
        "shape": [rows],
        "range_values": [0.0, 1.0],
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def collect_split_norm_goldens(
    args,
    *,
    inputs_module,
    run_forward,
    run_backward,
    eps_position,
    manifest_prefix,
):
    """Drive one bundle-form case file through offline golden collection.

    ``eps_position`` is the source-arg index of eps (LayerNorm 4, RMSNorm 3);
    the statistics are computed from the CPU source args exactly like the
    online producer (before any device move).
    """
    init_registry()

    import transformer_engine

    import_path = str(Path(transformer_engine.__file__).resolve())
    if args.reject_import_token and args.reject_import_token.lower() in import_path.lower():
        raise RuntimeError(
            "Expected NVTE, but imported transformer_engine from {}".format(import_path)
        )

    cases_path = Path(args.cases).resolve()
    input_root = Path(args.input_dir).resolve()
    golden_root = Path(args.golden_dir).resolve()
    cases_out = Path(args.cases_out).resolve() if args.cases_out else None
    stem = cases_path.stem
    records = []
    skipped = []
    offline_cases = []

    with cases_path.open("r", encoding="utf-8") as stream:
        raw_cases = json.load(stream)

    for raw_case in raw_cases:
        # Config-level validation only (the bundle metadata must parse); the
        # per-input spec validation lives in the plugin module.
        CaseConfig(**raw_case)
        specs = raw_case["inputs"]
        if len(specs) != 1 or specs[0].get("dtype") != inputs_module.BUNDLE_DTYPE:
            raise ValueError(
                "Expected a {} bundle case file (generate with the "
                "forward/backward yaml + hook), got {}".format(
                    inputs_module.BUNDLE_DTYPE, cases_path
                )
            )
        metadata = specs[0]["range_values"]
        phase = metadata["phase"]
        source_case = metadata["source_case"]
        case_id = source_case["id"]

        source_args = list(next(OpsDataset([source_case], metadata["seed"])).args)
        source_args = [v.item() if isinstance(v, np.generic) else v for v in source_args]
        source_args = [
            v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for v in source_args
        ]
        eps = float(source_args[eps_position])

        rows = source_args[0].numel() // source_args[0].shape[-1]
        if phase == "backward":
            # LayerNorm returns (means, rstdevs); RMSNorm returns a single
            # tensor — ``list()`` would explode the latter into scalars.
            stats = inputs_module.compute_shared_statistics(source_args[0], eps)
            if isinstance(stats, tuple):
                source_args += list(stats)
            else:
                source_args.append(stats)

        input_case_dir = input_root / str(case_id)
        input_case_dir.mkdir(parents=True, exist_ok=True)
        input_file = input_case_dir / "input.bin"
        torch.save(source_args, input_file)

        offline_case = json.loads(json.dumps(source_case))
        offline_case["save_name"] = raw_case.get("save_name") or stem
        for spec in offline_case["inputs"]:
            spec["backward"] = False
        if phase == "backward":
            offline_case["inputs"] += [_stat_spec(rows) for _ in inputs_module.STAT_NAMES]
        offline_cases.append(offline_case)

        device_args = [a.to(args.device) if isinstance(a, torch.Tensor) else a for a in source_args]
        holder = SimpleNamespace(args=device_args)
        try:
            outputs = (
                run_forward(holder, args.device)
                if phase == "forward"
                else run_backward(holder, args.device)
            )
            if str(args.device).startswith("npu"):
                torch.npu.synchronize()
            else:
                torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            skipped.append({"case_id": case_id, "reason": "cuda_oom"})
            print(
                "[skip] case {} hit CUDA OOM on the golden host; golden not "
                "collected (case JSON and ids unchanged)".format(case_id)
            )
            gc.collect()
            _empty_cache(args.device)
            continue

        output_dir = golden_root / args.backend_label / stem / str(case_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        for index, output in enumerate(_flatten_tensors(outputs)):
            output_file = output_dir / "output_{}.pt".format(index)
            torch.save(output.detach().cpu(), output_file)
            output_files.append(output_file.name)
        del outputs
        gc.collect()
        _empty_cache(args.device)

        records.append(
            {
                "case_id": case_id,
                "phase": phase,
                "input_sha256": _sha256(input_file),
                "outputs": output_files,
            }
        )

    if cases_out is not None:
        cases_out.mkdir(parents=True, exist_ok=True)
        with (cases_out / "{}.json".format(stem)).open("w", encoding="utf-8") as stream:
            json.dump(offline_cases, stream, indent=2, ensure_ascii=False)

    manifest = {
        "schema_version": 1,
        "backend": "nvte_cuda",
        "backend_detail": "offline twin of the online producer/receiver bundle flow",
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
        "transformer_engine_import": import_path,
        "torch_version": torch.__version__,
        "device": _device_name(args.device),
        "cases_file": str(cases_path),
        "case_count": len(records),
        "skipped": skipped,
        "records": records,
    }
    golden_root.mkdir(parents=True, exist_ok=True)
    manifest_path = golden_root / "{}_manifest.json".format(manifest_prefix)
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print(
        "Collected {} NVTE cases into {} ({} skipped: {})".format(
            len(records), golden_root, len(skipped), [s["case_id"] for s in skipped]
        )
    )
