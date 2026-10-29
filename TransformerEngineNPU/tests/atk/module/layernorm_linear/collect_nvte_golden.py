# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Collect NVTE LayerNormLinear P0 golden outputs in ATK ``accuracy_load`` layout.

Run on a CUDA host with the pinned NVIDIA Transformer Engine importable as
``transformer_engine``. Rejects a TENPU import, records import path/version and
input SHA256, and writes NVTE outputs in the layout expected by the ``cpu_nvte``
accuracy-load node. Shares the exact device execution path
(``run_layernorm_linear_case``) used on the NPU side.

The ``forward_output`` / ``ln_weight_grad`` / ``ln_bias_grad`` /
``weight_grad`` / ``bias_grad`` slots are written as **exact fp64
references**, not the NVTE kernel output (baseline selection — the Linear
inside LayerNormLinear shares te.Linear's cuBLASLt
path, the LayerNorm's parameter grads are token-batch reductions against a
fused kernel, and NVTE's ``transB=T`` forward is coarse on large-K/narrow-N
shapes). The policy lives in one place
(``run_layernorm_linear_case(exact_baselines=True)``), so the offline and
online golden paths cannot drift apart. The manifest records, per replaced
slot, how far NVTE's own output sits from the baseline that replaced it.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import torch

from atk.configs.case_config import CaseConfig
from atk.tasks.dataset.base_dataset import OpsDataset

from atk.case_generator.utils.plugins_register import init_registry


import sys

# Make the collector runnable from any working directory (e.g. the TENPU
# repo root): it imports its sibling adapter module by bare name, and the
# shared comparators / exact-reference helpers live one level up.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = str(Path(_HERE).resolve().parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from function_layernorm_linear import RESULT_SCHEMA, run_layernorm_linear_case  # noqa: E402
from _comparators import MixedToleranceAccuracy, load_design_acc_kwargs  # noqa: E402

# Slots whose content is an exact reference rather than the device output.
_FORWARD_SLOT = RESULT_SCHEMA.index("forward_output")
_LNGAMMA_SLOT = RESULT_SCHEMA.index("ln_weight_grad")
_LNBETA_SLOT = RESULT_SCHEMA.index("ln_bias_grad_or_empty")
_WGRAD_SLOT = RESULT_SCHEMA.index("weight_grad")
_BGRAD_SLOT = RESULT_SCHEMA.index("bias_grad_or_empty")
_EXACT_SLOTS = (_FORWARD_SLOT, _LNGAMMA_SLOT, _LNBETA_SLOT, _WGRAD_SLOT, _BGRAD_SLOT)


def _flatten_tensors(value) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_tensors(item)


def _judge(local: torch.Tensor, remote: torch.Tensor, index: int, comparator) -> Dict:
    """Judge one slot with the real ATK comparator, baseline as the reference."""
    result = comparator.compute_accuracy_result(local, remote, "output_{}.pt".format(index))
    return {"result": bool(result.result), "error_info": result.error_info}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_args(args):
    """Make ``dataset.args`` ``torch.load(weights_only=True)``-safe.

    ATK's ``OpsDataset`` samples ``float`` attrs (e.g. ``epsilon``) as
    ``numpy.float32`` scalars; ``torch.save`` pickles them verbatim, and ATK's
    device-side loader uses ``torch.load(weights_only=True)`` which refuses
    numpy globals. Convert numpy scalars to Python scalars (numerically
    identical) so the saved ``input.bin`` loads on both the golden and the NPU
    side. ``int``/``bool``/``str``/tensor values pass through unchanged.
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
    parser.add_argument("--cases", required=True, help="ATK generated all_layernorm_linear.json")
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

    acc_kwargs = load_design_acc_kwargs(Path(__file__).resolve().parent / "layernorm_linear.yaml")
    # Judges "how far is NVTE's own output from the reference", i.e. exactly
    # the question ATK asks of the NPU. local = device output, remote = baseline.
    comparator = MixedToleranceAccuracy(None, **acc_kwargs)

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
        # The device outputs run through the module first; the exact fp64
        # references replace the wgrad/bgrad slots inside run_case (single
        # source — the online GPU path shares the same definition).
        raw_slots = {}
        try:
            outputs = run_layernorm_linear_case(
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
        slot_baseline = {}
        device_vs_baseline = {}
        for index, output in enumerate(_flatten_tensors(outputs)):
            torch.save(output.detach().cpu(), output_dir / "output_{}.pt".format(index))
            output_files.append("output_{}.pt".format(index))
            if index == _FORWARD_SLOT:
                slot_baseline[str(index)] = "exact_fp64_forward"
            elif index == _LNGAMMA_SLOT:
                slot_baseline[str(index)] = "exact_fp64_lngamma"
            elif index == _LNBETA_SLOT:
                slot_baseline[str(index)] = (
                    "exact_fp64_lnbeta" if output.numel() else "empty_passthrough"
                )
            elif index == _WGRAD_SLOT:
                slot_baseline[str(index)] = "exact_fp64_wgrad"
            elif index == _BGRAD_SLOT:
                slot_baseline[str(index)] = (
                    "exact_fp64_bgrad" if output.numel() else "empty_passthrough"
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
        "backend_detail": (
            "nvte_cuda device outputs + exact fp64 grad-slot baselines (slots {})".format(
                ", ".join(str(slot) for slot in _EXACT_SLOTS)
            )
        ),
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
        "transformer_engine_import": import_path,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "cases_file": str(cases_path),
        "case_count": len(records),
        "result_schema": RESULT_SCHEMA,
        "grad_slot_baseline": {
            "mode": "exact_fp64",
            "slots": {
                str(_FORWARD_SLOT): (
                    "norm_forward_reference(x).to(case dtype) @ W^T [+ b], "
                    "fp64 accumulated over row chunks -- the LN output is "
                    "rounded to the case dtype once (the module's dataflow: it "
                    "is the GEMM's A operand), and the bias is included "
                    "(apply_bias = bias_enabled and not return_bias; this "
                    "adapter pins return_bias=False)"
                ),
                str(_LNGAMMA_SLOT): (
                    "colsum(xhat * dln) in fp64, with dln = round_dtype(dy @ W) "
                    "— the LN backward's own input, materialized in the case "
                    "dtype exactly as the module's dgrad buffer is; "
                    "zero_centered_gamma does not enter this slot (only dX "
                    "carries gamma_eff); RMSNorm keeps the same formula "
                    "(xhat = x * rms)"
                ),
                str(_LNBETA_SLOT): (
                    "colsum(dln) in fp64; no LN bias parameter under RMSNorm, "
                    "so the empty slot passes through"
                ),
                str(_WGRAD_SLOT): (
                    "dy^T @ norm_forward_reference(x).to(case dtype), fp64 "
                    "accumulated over token chunks — beta included (it enters "
                    "the GEMM's A operand); the LN output is rounded to the "
                    "case dtype once, mirroring the module's dataflow (LN "
                    "output materialized in case dtype before the GEMM); "
                    "RMSNorm has no beta by construction"
                ),
                str(_BGRAD_SLOT): (
                    "column sum of grad_output, fp64 (bias-off cases: empty slot passes through)"
                ),
            },
            "mechanism": (
                "LayerNormLinear's Linear shares te.Linear's cuBLASLt "
                "CUBLASLT_EPILOGUE_BGRADB fused path; its in-GEMM wgrad/db "
                "accumulation is coarser than the NPU's fp32 accumulation on "
                "large-token fp16/bf16 shapes (linear's measurements: NPU "
                "1.0000 vs NVTE 0.8802-0.9689 against exact fp64). "
                "The forward slot is the same story through the "
                "transB=T kernel NVTE picks on large-K/narrow-N shapes "
                "(all_linear id73/id137 at 0.9616/0.9851 with the NPU at "
                "1.0000; measured for this module on the second generalized "
                "round, case 65 at 0.9620 vs 1.000000). The LayerNorm's own "
                "parameter grads are batch reductions over every token with "
                "the same NVTE-vs-NPU question; the per-slot judge puts that "
                "divergence in the operand too — the device's reduction is "
                "exact given its own dln (matched_ratio 1.000000, 0 abs "
                "violations, 9/9 cases), and the reference reproduces it only "
                "with the one dtype rounding of dln that the module's "
                "dataflow performs (3#31). Baseline choice, not a tolerance "
                "relaxation."
            ),
            "history": (
                "The bgrad column sum was previously substituted here in the "
                "collector only (3#17 treatment); centralising into "
                "run_layernorm_linear_case also closes the online-mode bypass."
            ),
        },
        "skipped": skipped,
        "records": records,
    }
    golden_root.mkdir(parents=True, exist_ok=True)
    manifest_path = golden_root / "layernorm_linear_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print("Collected {} NVTE LayerNormLinear cases into {}".format(len(records), golden_root))


if __name__ == "__main__":
    main()
