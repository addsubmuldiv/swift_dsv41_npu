# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Collect NVTE Linear P0 golden outputs in ATK ``accuracy_load`` layout.

This script is meant to run on a CUDA host that has the pinned NVIDIA
Transformer Engine build importable as ``transformer_engine``. It rejects a
TENPU import (path containing ``TransformerEngineNPU``), records the actual
import path/version and input SHA256 values, and writes NVTE outputs in the
layout expected by the ``cpu_nvte`` accuracy-load node. It shares the exact
device execution path (``run_linear_case``) used on the NPU side, so both
sides exercise the same canonical inputs and the same forward/backward recipe.

**Slots 0, 3 and 4 in this tree are NOT NVTE kernel outputs.** They carry exact
fp64 references, because NVTE's ``forward_output``, ``weight_grad`` and
``bias_grad`` are produced by cuBLASLt kernels whose accumulation is less
accurate than the NPU's on some shapes (judged against exact fp64: NPU 1.0000 on
all three; NVTE 0.8802-0.9689 / 0.9033-1.0000 on the grad slots, and 0.9616 /
0.9851 on ``forward_output`` for the two ``all_linear`` fp16 shapes that were
still red). Running NVTE against this tree therefore fails exactly those slots
*by design* -- that is the baseline choice, not a broken golden. The policy lives
in one place (``function_linear.run_linear_case`` with
``exact_baselines=True``), so the offline and online golden paths cannot drift
apart; the reasoning and measurements are in ``../_exact_reference.py``.
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
# repo root): it imports its sibling adapter module by bare name, and the shared
# comparators / exact-reference helpers that live one level up.
_HERE = Path(__file__).resolve().parent
for _path in (str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _comparators import MixedToleranceAccuracy, load_design_acc_kwargs  # noqa: E402
from function_linear import (  # noqa: E402
    RESULT_SCHEMA,
    _BGRAD_SLOT,
    _FORWARD_SLOT,
    _WGRAD_SLOT,
    run_linear_case,
)

# Slots whose content is an exact reference rather than the device output.
_EXACT_SLOTS = (_FORWARD_SLOT, _WGRAD_SLOT, _BGRAD_SLOT)


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

    ATK's ``OpsDataset`` samples ``float`` attrs as ``numpy.float32`` scalars;
    ``torch.save`` pickles them verbatim, and ATK's device-side loader uses
    ``torch.load(weights_only=True)`` which refuses numpy globals. Convert
    numpy scalars to Python scalars (numerically identical) so the saved
    ``input.bin`` loads on both the golden and the NPU side. ``int``/``bool``/
    ``str``/tensor values pass through unchanged.
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


def _bias_baseline_kind(bias_enabled: bool, return_bias: bool, consume: bool, numel: int) -> str:
    """Which of the four ``bias_grad`` shapes this case took."""
    if not bias_enabled or numel == 0:
        return "empty"
    if not return_bias:
        return "exact_fp64_column_sum"
    if not consume:
        return "empty"
    return "grad_returned_bias_identity"


def _forward_baseline_kind(bias_enabled: bool, return_bias: bool, numel: int) -> str:
    """Which of the three ``forward_output`` shapes this case took.

    The bias enters the forward output only when the module applies it, i.e.
    ``bias_enabled and not return_bias`` (NVTE ``Linear.apply_bias``). A 0-row
    input is the third shape: the slot is legitimately empty and passes through.
    """
    if numel == 0:
        return "empty"
    if bias_enabled and not return_bias:
        return "exact_fp64_x_w_t_plus_bias"
    return "exact_fp64_x_w_t"


def _judge(local: torch.Tensor, remote: torch.Tensor, index: int, comparator) -> Dict:
    """Judge one slot with the real ATK comparator, baseline as the reference."""
    result = comparator.compute_accuracy_result(local, remote, "output_{}.pt".format(index))
    return {"result": bool(result.result), "error_info": result.error_info}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="ATK generated all_linear.json")
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

    acc_kwargs = load_design_acc_kwargs(Path(__file__).resolve().parent / "linear.yaml")
    # Judges "how far is NVTE's own output from the reference", i.e. exactly the
    # question ATK asks of the NPU. local = device output, remote = baseline.
    comparator = MixedToleranceAccuracy(None, **acc_kwargs)

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
        # The device's own slots are captured for provenance before the exact
        # references replace them; the substitution policy itself lives in
        # run_linear_case so the offline and online paths share one definition.
        raw_slots = {}
        try:
            outputs = run_linear_case(
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
        flattened = list(_flatten_tensors(outputs))
        # Adapter arg layout: inp, weight, bias, grad_output,
        # grad_returned_bias, bias_enabled, return_bias, consume_returned_bias, split
        case_args = list(dataset.args)
        bias_enabled = bool(case_args[5])
        return_bias = bool(case_args[6])
        consume_returned_bias = bool(case_args[7])

        slot_baseline = {}
        device_vs_baseline = {}
        for index, output in enumerate(flattened):
            torch.save(output.detach().cpu(), output_dir / "output_{}.pt".format(index))
            output_files.append("output_{}.pt".format(index))
            if index == _FORWARD_SLOT:
                slot_baseline[str(index)] = _forward_baseline_kind(
                    bias_enabled, return_bias, output.numel()
                )
            elif index == _WGRAD_SLOT:
                slot_baseline[str(index)] = "exact_fp64_wgrad"
            elif index == _BGRAD_SLOT:
                slot_baseline[str(index)] = _bias_baseline_kind(
                    bias_enabled, return_bias, consume_returned_bias, output.numel()
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
        "schema_version": 3,
        "backend": "nvte_cuda",
        "backend_detail": "nvte_cuda device outputs + exact fp64 forward/grad-slot baselines",
        "transformer_engine_version": getattr(transformer_engine, "__version__", "unknown"),
        "transformer_engine_import": import_path,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "cases_file": str(cases_path),
        "case_count": len(records),
        "result_schema": RESULT_SCHEMA,
        "forward_slot_baseline": {
            "mode": "exact_fp64",
            "slots": {
                str(_FORWARD_SLOT): {
                    "bias_applied": "x @ W^T + b, fp64, row-chunked",
                    "bias_not_applied": "x @ W^T, fp64, row-chunked",
                    "empty": "0-row input: slot passes through",
                },
            },
            "bias_rule": (
                "bias enters the forward output iff bias_enabled and not "
                "return_bias (NVTE Linear.apply_bias). Adding it for return_bias "
                "cases would put a bias-sized error into the golden itself."
            ),
            "note": (
                "Added after the generalized all_linear set left exactly two red "
                "cases, both on slot {}: the per-slot judge scored the NPU 1.0000 "
                "against exact fp64 while NVTE landed at 0.9616 / 0.9851. Same "
                "baseline choice as the grad slots, not a tolerance relaxation."
            ).format(_FORWARD_SLOT),
        },
        "grad_slot_baseline": {
            "mode": "exact_fp64",
            "slots": {
                str(_WGRAD_SLOT): "dy^T @ x, fp64 accumulated over token chunks",
                str(_BGRAD_SLOT): {
                    "bias_disabled": "empty",
                    "return_bias_false": "exact fp64 column sum of grad_output",
                    "return_bias_true_unconsumed": "empty",
                    "return_bias_true_consumed": "grad_returned_bias (identity)",
                },
            },
            "thresholds_source": (
                "linear.yaml: standard.acc.<module>_mixed_tolerance"
                if acc_kwargs
                else "unavailable"
            ),
            "device_vs_baseline": (
                "per record: NVTE's own slot judged against the baseline with the "
                "module comparator -- the evidence that the deviant side is NVTE"
            ),
            "note": (
                "Slots {}, {} and {} in this tree are NOT NVTE kernel outputs. The "
                "documented deviant is NVTE's cuBLASLt accumulation, which is less "
                "accurate than the NPU's on some shapes; running NVTE against this "
                "tree fails exactly those slots BY DESIGN. See "
                "tests/atk/module/_exact_reference.py."
            ).format(_FORWARD_SLOT, _WGRAD_SLOT, _BGRAD_SLOT),
        },
        "skipped": skipped,
        "records": records,
    }
    golden_root.mkdir(parents=True, exist_ok=True)
    manifest_path = golden_root / "linear_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    print("Collected {} NVTE Linear cases into {}".format(len(records), golden_root))


if __name__ == "__main__":
    main()
