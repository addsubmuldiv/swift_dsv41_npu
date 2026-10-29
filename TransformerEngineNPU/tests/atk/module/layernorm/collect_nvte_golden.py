# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Collect independent LayerNorm NVTE outputs and inputs for ATK offline replay.

Run once per generated forward/backward case file on a CUDA host. The collector
uses the online producer's input recipe, saves positional inputs (including
shared backward statistics), and exports matching direct-input case JSON for
``atk task --input_data``. See README.md for the complete workflow.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Resolve sibling plugins before importing; the collector also runs outside the repo.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402
from atk.case_generator.utils.plugins_register import init_registry  # noqa: E402
from atk.configs.case_config import CaseConfig  # noqa: E402
from function_layernorm import run_layernorm_backward, run_layernorm_forward  # noqa: E402
from layernorm_inputs import (  # noqa: E402
    BUNDLE_DTYPE,
    BUNDLE_FORMAT,
    STAT_NAMES,
    compute_file_sha256,
    prepare_input_bundle,
    validate_inputs,
    validate_inputs_against_case,
)

SCHEMAS = {
    "forward": ["forward_output", "means", "rstdevs", "api_contract_json_utf8"],
    "backward": ["input_grad", "weight_grad", "bias_grad", "api_contract_json_utf8"],
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", required=True, help="bundle-form case JSON (forward or backward)"
    )
    parser.add_argument(
        "--input-dir", required=True, help="input artifact directory for this case file"
    )
    parser.add_argument("--golden-dir", required=True, help="root used by the accuracy_load node")
    parser.add_argument(
        "--cases-out", required=True, help="directory for direct-input case JSON consumed by NPU"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend-label", default="cpu_nvte")
    parser.add_argument(
        "--reject-import-token",
        default="TransformerEngineNPU",
        help="Fail if transformer_engine.__file__ contains this token; empty disables the check",
    )
    return parser.parse_args()


def _read_cases(path):
    with path.open("r", encoding="utf-8") as stream:
        cases = json.load(stream)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty ATK case list")
    ids = set()
    phases = set()
    for raw_case in cases:
        case = CaseConfig(**raw_case)
        specs = raw_case["inputs"]
        if len(specs) != 1 or specs[0].get("dtype") != BUNDLE_DTYPE:
            raise ValueError("Generate LayerNorm bundle cases with the phase YAML and hook")
        metadata = specs[0]["range_values"]
        phase = metadata.get("phase")
        if metadata.get("format") != BUNDLE_FORMAT or phase not in SCHEMAS:
            raise ValueError("Invalid LayerNorm bundle format/phase")
        expected_api = f"function_layernorm_{phase}"
        if case.api_type != expected_api or metadata["source_case"]["api_type"] != expected_api:
            raise ValueError("LayerNorm case API does not match its bundle phase")
        if case.id in ids:
            raise ValueError(f"Duplicate LayerNorm case ID: {case.id}")
        ids.add(case.id)
        phases.add(phase)
    if len(phases) != 1:
        raise ValueError("Collect forward and backward case files separately")
    return cases


def _offline_case(raw_case, values, save_name):
    """Match ATK's positional file loader, preserving the outer report case ID."""
    metadata = raw_case["inputs"][0]["range_values"]
    case = copy.deepcopy(raw_case)
    case["inputs"] = copy.deepcopy(metadata["source_case"]["inputs"])
    case["backward"] = False
    case["outputs"] = None
    case["save_name"] = save_name
    for spec in case["inputs"]:
        spec["backward"] = False
    if metadata["phase"] == "backward":
        for statistic in values[6:]:
            case["inputs"].append(
                {
                    "name": None,
                    "type": "tensor",
                    "dtype": "fp32",
                    "shape": list(statistic.shape),
                    "range_values": [0.0, 1.0],
                    "backward": False,
                }
            )
    CaseConfig(**case)
    return case


def collect_goldens(args):
    """Save inputs and all four phase outputs; any failed case aborts the export."""
    init_registry()
    import transformer_engine

    import_path = str(Path(transformer_engine.__file__).resolve())
    if args.reject_import_token and args.reject_import_token.lower() in import_path.lower():
        raise RuntimeError(f"Expected NVTE, but imported transformer_engine from {import_path}")
    cases_path = Path(args.cases).resolve()
    raw_cases = _read_cases(cases_path)
    input_root = Path(args.input_dir).resolve()
    golden_root = Path(args.golden_dir).resolve()
    cases_out = Path(args.cases_out).resolve() / cases_path.name
    if cases_out == cases_path:
        raise ValueError("--cases-out must not overwrite the generated bundle case JSON")
    if cases_out.exists():
        raise FileExistsError(f"Use a new --cases-out directory: {cases_out}")
    golden_suite_dir = golden_root / args.backend_label / cases_path.stem
    if golden_suite_dir.exists():
        raise FileExistsError(f"Use a new --golden-dir for this case file: {golden_suite_dir}")
    records, offline_cases = [], []
    for raw_case in raw_cases:
        metadata = raw_case["inputs"][0]["range_values"]
        phase = metadata["phase"]
        values = prepare_input_bundle(metadata)["values"]
        validate_inputs_against_case(metadata["source_case"], values[:6])
        validate_inputs(values, backward=phase == "backward")
        case_id = raw_case["id"]
        input_file = input_root / str(case_id) / "input.bin"
        # Refuse reuse: changing this file could invalidate goldens from another phase.
        if input_file.exists():
            raise FileExistsError(f"Use a new --input-dir for each case file: {input_file}")
        input_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, input_file)
        # Execute the saved artifact, just as the NPU task will do.
        values = torch.load(input_file, map_location="cpu", weights_only=True)
        offline_cases.append(_offline_case(raw_case, values, cases_path.stem))
        holder = SimpleNamespace(
            args=[
                value.to(args.device) if isinstance(value, torch.Tensor) else value
                for value in values
            ]
        )
        runner = run_layernorm_forward if phase == "forward" else run_layernorm_backward
        outputs = runner(holder, args.device)
        torch.cuda.synchronize(args.device)
        if len(outputs) != len(SCHEMAS[phase]):
            raise ValueError("Expected four LayerNorm phase outputs")
        output_dir = golden_suite_dir / str(case_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        for index, output in enumerate(outputs):
            output_file = output_dir / f"output_{index}.pt"
            torch.save(output.detach().cpu(), output_file)
            output_files.append(output_file.name)
        records.append(
            {
                "case_id": case_id,
                "source_case_id": metadata["source_case"]["id"],
                "phase": phase,
                "input_sha256": compute_file_sha256(input_file),
                "outputs": output_files,
                "schema": SCHEMAS[phase],
            }
        )
        del outputs, holder, values
        torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "backend": "nvte_cuda",
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
        "transformer_engine_import": import_path,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "cases_file": str(cases_path),
        "case_count": len(records),
        "input_dir": str(input_root),
        "offline_cases_file": str(cases_out),
        "shared_statistic_names": list(STAT_NAMES),
        "records": records,
    }
    golden_root.mkdir(parents=True, exist_ok=True)
    manifest_path = golden_root / f"{cases_path.stem}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    # Publish runnable cases only after every GPU output and the manifest were saved.
    cases_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = cases_out.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(offline_cases, stream, indent=2, ensure_ascii=False)
    temporary.replace(cases_out)
    print(f"Collected {len(records)} NVTE LayerNorm cases; NPU cases: {cases_out}")


def main():
    args = _parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to collect NVTE golden outputs")
    collect_goldens(args)


if __name__ == "__main__":
    main()
