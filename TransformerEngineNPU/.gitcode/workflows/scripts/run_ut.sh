#!/bin/bash
# TransformerEngineNPU gate UT script.

set -e

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <workspace> <pr_id> <target_branch>" >&2
  exit 2
fi

WORKSPACE=$1
pr_id=$2
branch=$3
echo "pr_id=${pr_id}"
echo "branch=${branch}"

readonly CODE_DIR="${WORKSPACE}/CODE"
readonly MODIFY_FILE="${WORKSPACE}/modify.txt"
readonly PYTHON_VERSION="3.12"
readonly -a UT_TRIGGER_PATHS=(
  "transformer_engine/"
  "transformer_engine_torch/"
  "tests/"
  "setup.py"
  "pyproject.toml"
  "requirements.txt"
)

run_ut() {
  # Ascend environment setup scripts probe optional shell variables such as
  # ZSH_VERSION, so nounset is intentionally not enabled in this script.
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  export ENABLE_ATB=1

  cd "${CODE_DIR}"

  echo "Selected Python version: ${PYTHON_VERSION}"

  # TODO: Fix pip install for pyproject.toml.
  rm -f pyproject.toml
  "pip${PYTHON_VERSION}" install -e .

  export PYTHONPATH="${PYTHONPATH:-}:${CODE_DIR}"

  "pip${PYTHON_VERSION}" list

  local ut_root="./tests"

  echo "============================================"
  echo "Running TransformerEngineNPU UTs..."
  echo "============================================"

  "python${PYTHON_VERSION}" -m pytest --color=no --timeout=1800 -x \
    "${ut_root}/" --ignore tests/pytorch/debug
  "python${PYTHON_VERSION}" -m pytest --color=no --timeout=1800 -x \
    tests/pytorch/debug
}

cd "${CODE_DIR}"
git diff-tree -r --name-only --no-commit-id "origin/${branch}" HEAD > "${MODIFY_FILE}"
cat "${MODIFY_FILE}"

for path in "${UT_TRIGGER_PATHS[@]}"; do
  if grep -Fq "${path}" "${MODIFY_FILE}"; then
    echo "UT-triggering change found: ${path}"
    run_ut
    exit 0
  fi
done

echo "No UT-triggering changes found, skipping UT."
