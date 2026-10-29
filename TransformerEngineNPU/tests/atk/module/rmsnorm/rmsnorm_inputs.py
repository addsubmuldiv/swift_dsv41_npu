# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# See LICENSE for license information.

"""Validation and device-independent fixtures for the split RMSNorm tests.

CPU prepares shared backward inputs, never an output benchmark.
The serialized FP32 statistics must be copied to both devices.
"""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

import torch


def validate_inputs(args, backward=False):
    expected = 6 if backward else 5
    if len(args) != expected:
        raise ValueError(f"Expected {expected} RMSNorm inputs, got {len(args)}")
    x, dy, gamma, eps, zero_centered = args[:5]
    if not all(isinstance(t, torch.Tensor) for t in (x, dy, gamma)):
        raise ValueError("x, dy and gamma must be tensors")
    if x.ndim not in (2, 3, 4) or not x.numel():
        raise ValueError("RMSNorm cases require nonempty 2D/3D/4D x")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported input dtype: {x.dtype}")
    if dy.shape != x.shape or gamma.shape != (x.shape[-1],):
        raise ValueError("Expected dy.shape == x.shape and gamma.shape == [hidden]")
    if any(t.dtype != x.dtype for t in (dy, gamma)):
        raise ValueError("x, dy and gamma must have the same dtype")
    if not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(zero_centered, bool):
        raise ValueError("zero_centered_gamma must be a bool")
    if backward:
        rows = x.numel() // x.shape[-1]
        rstdevs = args[5]
        if (
            not isinstance(rstdevs, torch.Tensor)
            or rstdevs.dtype != torch.float32
            or rstdevs.shape != (rows,)
        ):
            raise ValueError(f"rstdevs must be a serialized FP32 tensor of shape [{rows}]")
        # Nonfinite x is an intentional extra case. Its canonical rstd can
        # be zero (Inf rows) or NaN; preserve it for both device backends.
        if torch.isfinite(x).all():
            if not torch.isfinite(rstdevs).all() or not (rstdevs > 0).all():
                raise ValueError("Finite x requires finite positive rstdevs")


def compute_shared_statistics(x, eps):
    """Compute once from the actual, already quantized input; store in FP32."""
    flat = x.detach().to(device="cpu", dtype=torch.float64).reshape(-1, x.shape[-1])
    mean_square = flat.square().mean(dim=-1)
    return torch.rsqrt(mean_square + float(eps)).float().contiguous()


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
BUNDLE_DTYPE = "tenpu_rmsnorm_inputs"
BUNDLE_FORMAT = "tenpu_rmsnorm_inputs_v1"
# Shared-statistics slot names appended to offline backward cases.
STAT_NAMES = ("rstdevs",)


def load_saved_inputs(path):
    # ATK bac1aa7 torch_save_safe uses pickle_protocol=4 and numpy scalar attrs;
    # weights_only=True cannot read all of these files on the target Torch 2.9.
    # Match ATK's own trusted-artifact loader, then normalize scalar attributes.
    values = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{path}: expected positional inputs saved as a list/tuple")
    return [v.item() if isinstance(v, np.generic) else v for v in values]


def compute_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs_against_case(case, args):
    validate_inputs(args)
    specs = case["inputs"]
    if len(specs) != 5:
        raise ValueError("RMSNorm source cases require five input specifications")
    for spec, tensor in zip(specs[:3], args[:3]):
        if list(tensor.shape) != spec["shape"] or tensor.dtype != DTYPES[spec["dtype"]]:
            raise ValueError(f"Case {case['id']}: input.bin shape/dtype disagrees with case JSON")
    for spec, actual in zip(specs[3:], args[3:]):
        expected = spec["range_values"]
        if isinstance(expected, list) and len(expected) == 1:
            expected = expected[0]
        if isinstance(actual, bool):
            matches = isinstance(expected, bool) and actual == expected
        else:
            # ATK may materialize eps through numpy.float32.
            matches = abs(float(actual) - float(expected)) <= abs(float(expected)) * 1e-6
        if not matches:
            raise ValueError(f"Case {case['id']}: input.bin attributes disagree with case JSON")


def create_input_bundle_case(case, phase, input_dir=None, seed=0):
    """Describe a single ATK custom-data argument containing all coupled inputs."""
    if phase not in ("forward", "backward") or len(case["inputs"]) != 5:
        raise ValueError("Expected a forward/backward phase and five-input source case")
    if not isinstance(case["id"], int) or case["id"] < 0:
        raise ValueError("Source case ID must be a nonnegative integer")
    converted = copy.deepcopy(case)
    converted["api_type"] = f"function_rmsnorm_{phase}"
    converted["backward"] = False
    converted["outputs"] = None
    converted["save_name"] = f"rmsnorm_{phase}"
    source_case = copy.deepcopy(case)
    source_case["api_type"] = converted["api_type"]
    source_case["backward"] = False
    for spec in source_case["inputs"]:
        spec["backward"] = False
    converted["inputs"] = [
        {
            "name": None,
            "type": "attr",
            "dtype": BUNDLE_DTYPE,
            "shape": None,
            "backward": False,
            "range_values": {
                "format": BUNDLE_FORMAT,
                "phase": phase,
                "source_case": source_case,
                "input_dir": str(Path(input_dir).resolve()) if input_dir else None,
                "seed": seed + case["id"],
            },
        }
    ]
    return converted


def prepare_input_bundle(metadata):
    """Create CPU inputs and shared statistics for online sync or offline replay."""
    if metadata.get("format") != BUNDLE_FORMAT or metadata.get("phase") not in (
        "forward",
        "backward",
    ):
        raise ValueError("Invalid RMSNorm bundle configuration")
    case = metadata["source_case"]
    source = None
    if metadata["input_dir"]:
        source = Path(metadata["input_dir"]) / str(case["id"]) / "input.bin"
        args = load_saved_inputs(source)
    else:
        from atk.tasks.dataset.base_dataset import OpsDataset

        args = next(OpsDataset([case], metadata["seed"])).args
        args = [v.item() if isinstance(v, np.generic) else v for v in args]
    validate_inputs_against_case(case, args)
    args = [v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for v in args]
    args[3], args[4] = float(args[3]), bool(args[4])
    if metadata["phase"] == "backward":
        args.append(compute_shared_statistics(args[0], args[3]))
    bundle = {
        "format": BUNDLE_FORMAT,
        "phase": metadata["phase"],
        "case_id": case["id"],
        "source_input_sha256": compute_file_sha256(source) if source else None,
        "values": args,
    }
    return bundle
