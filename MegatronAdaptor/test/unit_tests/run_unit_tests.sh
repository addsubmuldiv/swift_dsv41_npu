#!/bin/bash
# One-shot runner for MegatronAdaptor unit tests.
#
# Prerequisites (not handled by this script; the caller must ensure them beforehand):
#   1. The CANN environment is sourced (provides libhccl.so etc., required by import torch_npu).
#   2. This script runs inside an activated conda environment that has pytest / torch_npu
#      and other dependencies installed; it uses the current environment's python interpreter.
#
# Usage:
#   bash test/unit_tests/run_unit_tests.sh <MEGATRON_LM_PATH> [pytest args...]
#
# Examples:
#   # Run all unit tests under test/unit_tests/
#   bash test/unit_tests/run_unit_tests.sh /path/to/Megatron-LM
#   # Specify a file / pass through pytest args (-k filter, -s, etc.)
#   bash test/unit_tests/run_unit_tests.sh /path/to/Megatron-LM \
#       test/unit_tests/patches/megatron/test_preload_tensors.py -s

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <MEGATRON_LM_PATH> [pytest args...]" >&2
    exit 1
fi

MEGATRON_PATH="$1"
shift

if [ ! -d "${MEGATRON_PATH}/megatron" ]; then
    echo "[error] megatron package not found under path: ${MEGATRON_PATH}" >&2
    exit 1
fi

# The argument may be a relative path; convert to an absolute path.
MEGATRON_PATH="$(cd "${MEGATRON_PATH}" && pwd)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Add Megatron-LM to PYTHONPATH (otherwise import megatron fails).
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH}"

# Default to running the whole unit_tests directory; if args are given, pass them straight to pytest.
if [ "$#" -eq 0 ]; then
    set -- test/unit_tests/ -v
fi

cd "${REPO_ROOT}"
exec python -m pytest "$@"
