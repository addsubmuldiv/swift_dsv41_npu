# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Collect the CPU reference artifacts for the blockwise golden-free guard.

Runs anywhere (no accelerator). Mirrors ../mxfp8_quantization/collect_cpu_reference.py:

- output_0/1 (round-trip references): the input tensor itself, / bound.
- output_2 (cross-consistency reference): zeros.
- output_3/4 (scale-canonicity references): exact CPU canonical
  ``amax_block / fp8_max`` per 128x128 tile (fp32, unrounded — TENPU's
  ``force_pow_2_scales`` is a dead knob, so no pow-2 constraint is assumed).
- output_5: serialized contract (exact compare, same builder both sides).
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

import sys

# Make the collector runnable from any working directory (e.g. the TENPU
# repo root): it imports its sibling adapter module by bare name.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from function_blockwise_quantization import (  # noqa: E402
    RESULT_SCHEMA,
    cpu_reference_outputs,
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


def _generate_case_inputs(raw_case: dict) -> List:
    """Construct the case inputs directly, WITHOUT ATK's ``OpsDataset``.

    Mirrors ATK's dataset RNG recipe exactly (seed = case id, then
    ``np.random.uniform`` for each range-bounded tensor in input order), so
    the emitted ``input.bin`` is bit-identical to what the source-tree ATK
    generates — while removing the dependency on the parameter-type plugin
    registry entirely: pip-installed ATK builds moved ``ParameterFactory``
    to ``atk.case_generator.utils.plugins_register`` and a standalone python
    process does not have the built-in 'tensor'/'attr' input types
    registered, which made ``OpsDataset.__next__`` fail with
    ``KeyError: 'tensor'`` regardless of import order. The ATK task side
    never regenerates inputs either: with ``--input_data`` it loads this
    ``input.bin`` directly (``DatasetExecutor.load_dataset``).
    """
    seed = raw_case.get("id")
    seed = 0 if seed is None else int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args: List = []
    for spec in raw_case["inputs"]:
        if spec["type"] == "tensor":
            # Exact ATK recipe (data_torch.gen_nonbound_tensor_data): sample
            # via numpy, cast through the numpy dtype table (bf16 maps to
            # np.float16!), then convert to torch and re-cast — the double
            # rounding fp32->fp16->bf16 is what ATK actually produces and must
            # be mirrored bit-for-bit or the input.bin differs by 1 ULP on ~6%
            # of elements.
            low, high = spec["range_values"]
            array = np.random.uniform(low=low, high=high, size=spec["shape"])
            if spec["dtype"] == "bf16":
                tensor = torch.from_numpy(array.astype(np.float16)).bfloat16()
            elif spec["dtype"] == "fp16":
                tensor = torch.from_numpy(array.astype(np.float16))
            elif spec["dtype"] == "fp32":
                tensor = torch.from_numpy(array.astype(np.float32))
            else:
                raise ValueError(
                    "unsupported tensor dtype in case {}: {}".format(seed, spec["dtype"])
                )
            args.append(tensor)
        elif spec["type"] == "attr":
            value = spec["range_values"]
            args.append(value.item() if isinstance(value, np.generic) else value)
        else:
            raise ValueError("unsupported input type in case {}: {}".format(seed, spec["type"]))
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_args(args):
    """Make ``dataset.args`` ``torch.load(weights_only=True)``-safe."""

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
    parser.add_argument(
        "--cases", required=True, help="ATK generated all_blockwise_quantization.json"
    )
    parser.add_argument("--input-dir", required=True, help="Canonical input artifact directory")
    parser.add_argument(
        "--reference-dir", required=True, help="Root used by the accuracy_load node"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend-label", default="cpu_nvte")
    return parser.parse_args()


def _cpu_reference(args_list, device: str):
    """Compute the golden-free CPU reference outputs for one case.

    Delegates to the adapter's ``cpu_reference_outputs`` (single source): the
    offline artifacts collected here and the ONLINE GPU-side reference branch
    must come from the same code, or the two paths drift. The computation is
    always done on CPU; ``device`` only decides where the result tensors land
    (matching this collector's historical ``--device`` semantics).
    """
    x, fp8_format = args_list
    return cpu_reference_outputs(x, fp8_format, out_device=torch.device(device))


def main():
    args = _parse_args()

    cases_path = Path(args.cases).resolve()
    input_root = Path(args.input_dir).resolve()
    reference_root = Path(args.reference_dir).resolve()
    save_name = cases_path.stem
    records = []

    with cases_path.open("r", encoding="utf-8") as stream:
        raw_cases = json.load(stream)

    for raw_case in raw_cases:
        case_id = raw_case.get("id")
        args_list = _generate_case_inputs(raw_case)
        # ATK resolves BOTH sides FLAT (no "small"/"real" tier):
        # - task --input_data loads  <input_data>/<case_id>/input.bin
        #   (DatasetExecutor.get_input_file_path: join(custom_base, str(id)));
        # - the accuracy_load reference is resolved as
        #   <output_path>/<backend_label>/<STEM>/<case_id>/ where STEM is the
        #   cases-FILE stem (the runner OVERWRITES every case's save_name with
        #   it at task creation, main.py create_case_by_index).
        case_save_name = save_name  # == cases-file stem == task-side save_name
        input_case_dir = input_root / str(case_id)
        input_case_dir.mkdir(parents=True, exist_ok=True)
        input_file = input_case_dir / "input.bin"
        torch.save(_sanitize_args(args_list), input_file)

        outputs = _cpu_reference(args_list, args.device)
        output_dir = reference_root / args.backend_label / str(case_save_name) / str(case_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        for index, output in enumerate(_flatten_tensors(outputs)):
            output_file = output_dir / "output_{}.pt".format(index)
            torch.save(output.detach().cpu(), output_file)
            output_files.append(output_file.name)

        records.append(
            {
                "case_id": case_id,
                "input_sha256": _sha256(input_file),
                "outputs": output_files,
                "schema": RESULT_SCHEMA,
            }
        )

    manifest = {
        "schema_version": 1,
        "backend": "cpu_reference",
        "guard_mode": "golden_free",
        "note": (
            "CPU-computed reference for the golden-free blockwise guard: "
            "round-trip reference is the input itself (theoretical bound); "
            "scale reference is the exact canonical amax/fp8_max per 128x128 "
            "tile. Not an NVTE golden — NVTE's blockwise kernel requires "
            "Hopper (SM90) + CUDA >= 12.9, unavailable on the intranet."
        ),
        "torch_version": torch.__version__,
        "device": "cpu",
        "cases_file": str(cases_path),
        "case_count": len(records),
        "result_schema": RESULT_SCHEMA,
        "records": records,
    }
    reference_root.mkdir(parents=True, exist_ok=True)
    manifest_path = reference_root / "blockwise_quantization_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    sample = raw_cases[0] if raw_cases else {}
    sample_id = sample.get("id", 0)
    print(
        "ATK accuracy_load will resolve: {}/{}/{}/{}/output_*.pt "
        "(task-side save_name == the cases-FILE STEM, which the runner "
        "overwrites onto every case; must sit under the accuracy_load "
        "node output_path in nodes yaml)".format(
            reference_root, args.backend_label, save_name, sample_id
        )
    )
    print("Collected {} CPU-reference blockwise cases into {}".format(len(records), reference_root))


if __name__ == "__main__":
    main()
