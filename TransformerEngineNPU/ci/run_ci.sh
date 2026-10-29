#!/bin/bash
# TransformerEngineNPU 门禁UT/ST 脚本
set -e

WORKSPACE=$1
pr_id=$2
branch=$3
echo "branch=${branch}"

# 把需要在容器内做的事写成一个代码块，通过环境变量传入参数

list_ut_file=("transformer_engine/" "transformer_engine_torch/" "tests/" "setup.py" "pyproject.toml" "requirements.txt")
list_st_file=("transformer_engine/" "tests/system_tests/" "megatron")

run_ut() {
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  export ENABLE_ATB=1

  cd "${WORKSPACE}/CODE/"

  PYTHON_VERSION="3.12"
  echo "Selected Python version: $PYTHON_VERSION"

  # TODO : fix pip install for pyproject.toml
  rm -rf pyproject.toml
  pip$PYTHON_VERSION install -e .

  export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/CODE

  pip$PYTHON_VERSION list

  UT_ROOT="./tests"

  if [ ! -d "${UT_ROOT}" ]; then
    echo "No tests directory found, skipping UT."
    return 0
  fi

  echo "============================================"
  echo "Running TransformerEngineNPU UTs..."
  echo "============================================"

  python$PYTHON_VERSION -m pytest --color=no --timeout=1800 -x ${UT_ROOT}/ --ignore tests/pytorch/debug --ignore tests/system_tests
  python$PYTHON_VERSION -m pytest --color=no --timeout=1800 -x tests/pytorch/debug
}

run_st() {
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh
  export ENABLE_ATB=1

  # 获取 Megatron-LM (main)
  cd "${WORKSPACE}"
  if [ ! -d "Megatron-LM" ]; then
    git clone https://gitcode.com/GitHub_Trending/me/Megatron-LM.git
  fi
  cd "${WORKSPACE}/Megatron-LM"
  git fetch origin main
  git checkout main
  git reset --hard origin/main

  # 获取 MegatronAdaptor (main)
  cd "${WORKSPACE}"
  if [ ! -d "MegatronAdaptor" ]; then
    git clone https://gitcode.com/Ascend/MegatronAdaptor.git
  fi
  cd "${WORKSPACE}/MegatronAdaptor"
  git fetch origin main
  git checkout main
  git reset --hard origin/main

  # 安装 TransformerEngineNPU (PR 检出代码)
  cd "${WORKSPACE}/CODE"
  rm -rf pyproject.toml
  pip install -e .

  # 安装 MegatronAdaptor
  cd "${WORKSPACE}/MegatronAdaptor"
  pip install -e . -v

  # 编译 Megatron-LM 的 C++ 扩展 helpers_cpp（Megatron-LM 以源码方式运行，不通过 pip 安装）。
  # gpt_dataset 构建时无条件 `from megatron.core.datasets.helpers_cpp import *`，
  # 不编译会在 dataset 构建阶段报错: No module named 'megatron.core.datasets.helpers_cpp'。
  # 放在此处编译以复用上文 pip install -e 已装好的 pybind11。
  cd "${WORKSPACE}/Megatron-LM"
  python setup.py build_ext --inplace

  # 使能 megatron_adaptor
  cd "${WORKSPACE}/Megatron-LM"
  if ! grep -q "import megatron_adaptor" pretrain_gpt.py; then
    sed -i '/^import torch$/a\import megatron_adaptor' pretrain_gpt.py
  fi

  # 运行 ST 用例 (在 Megatron-LM 目录下执行，pretrain_gpt.py 位于当前目录；
  # 用例脚本通过自身路径定位 env_npu.sh，无需软链进 Megatron-LM)
  bash -e "${WORKSPACE}/CODE/tests/system_tests/feature_tests/test_basic_training.sh"
}

cd "${WORKSPACE}/CODE"
git diff-tree -r --name-only --no-commit-id origin/${branch} HEAD > ${WORKSPACE}/modify.txt
cat ${WORKSPACE}/modify.txt
for file in "${list_ut_file[@]}"; do
  if grep -q "${file}" "${WORKSPACE}/modify.txt"; then
    echo "find ${file}"
    run_ut
    break
  fi
done
for file in "${list_st_file[@]}"; do
  if grep -q "${file}" "${WORKSPACE}/modify.txt"; then
    echo "find ${file}"
    run_st
    break
  fi
done
