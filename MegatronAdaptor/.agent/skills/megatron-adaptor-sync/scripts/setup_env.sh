#!/bin/bash
# Conda environment activation and CANN initialization.
# Usage: source setup_env.sh [conda_env_name]
#   conda_env_name: optional, default ma_agent_py312
#   All branches use ma_agent_py312 (dedicated conda env for this skill)
#
# Paths are configurable via environment variables:
#   CONDA_INIT           — path to conda.sh (default: /home/miniconda3/etc/profile.d/conda.sh)
#   ASCEND_TOOLKIT_PATH  — path to ascend-toolkit (default: /home/ma_daily_ci/CANN/ascend-toolkit)
#   ATB_PATH             — path to nnal/atb (default: /home/ma_daily_ci/CANN/nnal/atb)

CONDA_ENV="${1:-ma_agent_py312}"

# ── Conda ──
_conda_init="${CONDA_INIT:-/home/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$_conda_init" ]; then
    source "$_conda_init"
else
    echo "WARNING: conda.sh not found at $_conda_init — conda may not be available" >&2
fi
conda activate "$CONDA_ENV"

# ── CANN (Ascend NPU toolkit) ──
_ascend_toolkit="${ASCEND_TOOLKIT_PATH:-/home/ma_daily_ci/CANN/ascend-toolkit}"
_atb="${ATB_PATH:-/home/ma_daily_ci/CANN/nnal/atb}"

if [ -f "$_ascend_toolkit/set_env.sh" ]; then
    source "$_ascend_toolkit/set_env.sh"
else
    echo "WARNING: ascend-toolkit set_env.sh not found at $_ascend_toolkit" >&2
fi

if [ -f "$_atb/set_env.sh" ]; then
    source "$_atb/set_env.sh"
fi
