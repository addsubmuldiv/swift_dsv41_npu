#!/bin/bash
set -ex

WORKSPACE=${1:-${WORKSPACE}}  # 工作空间, 入参为空时从环境变量获取
PR_ID=$2                      # pull request id
TARGET_BRANCH=${3:-"main"}    # MegatronAdaptor 分支名称, 默认值为 "main"

if [ -z "$WORKSPACE" ]; then
    echo "ERROR: WORKSPACE is not set. Please provide it as an argument or set the WORKSPACE environment variable."
    exit 1
fi
MA_PATH="${WORKSPACE}"/CODE

# 判断是否有源码变更: 有源码变更返回0, 仅文档变更返回1
# 文档变更指 *.md 文件及 docs/ 目录下的文件
# 无法获取目标分支时保守处理, 视为有源码变更, 继续执行
function code_changed() {
    cd "${MA_PATH}"
    if ! git fetch origin "${TARGET_BRANCH}"; then
        echo "WARNING: fetch ${TARGET_BRANCH} failed, run tests anyway."
        return 0
    fi

    local files
    if ! files=$(git diff --name-only "origin/${TARGET_BRANCH}"...HEAD); then
        echo "WARNING: diff against origin/${TARGET_BRANCH} failed, run tests anyway."
        return 0
    fi

    echo "Changed files:"
    echo "${files}"

    local f
    while IFS= read -r f; do
        [ -z "${f}" ] && continue
        case "${f}" in
            *.md|docs/*) ;;        # 文档变更, 跳过
            *) return 0 ;;         # 存在源码变更
        esac
    done <<< "${files}"
    return 1
}

# 下载依赖的代码
function download_code() {
    cd "${WORKSPACE}"
    if [ ! -d "TransformerEngineNPU" ]; then
        git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
    fi
    cd "${WORKSPACE}"/TransformerEngineNPU
    git fetch origin main
    git checkout main
    git reset --hard origin/main
    cd ..

    checkout_lines=$(grep "git checkout" "${WORKSPACE}"/CODE/README.md | awk '{print substr($0, index($0,$3))}' | tail -n 1)
    if [ -z "$checkout_lines" ]; then
        echo "No valid git checkout line found in README.md"
        exit 1
    fi

    cd "${WORKSPACE}"
    if [ ! -d "Megatron-LM" ]; then
        git clone https://gitcode.com/GitHub_Trending/me/Megatron-LM.git
    fi
    cd "${WORKSPACE}"/Megatron-LM
    git fetch origin "$checkout_lines"
    git checkout "$checkout_lines"
    git reset --hard origin/"$checkout_lines"
    cd ..
}

# 环境准备
function prepare_env() {
    PYTHON_BIN=$(which python3)
    PYTHON_DIR=$(dirname "$PYTHON_BIN")
    PYTHON_CONFIG="$PYTHON_DIR/python3-config"
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
    PYTHON_VERSION_CONFIG="$PYTHON_DIR/python${PYTHON_VERSION}-config"

    if [ ! -f "$PYTHON_CONFIG" ] && [ ! -L "$PYTHON_CONFIG" ]; then
        echo "python3-config not found, checking $PYTHON_VERSION_CONFIG..."
        if [ -f "$PYTHON_VERSION_CONFIG" ]; then
            ln -sf "$PYTHON_VERSION_CONFIG" "$PYTHON_CONFIG"
            echo "Created symlink: $PYTHON_CONFIG -> $PYTHON_VERSION_CONFIG"
        else
            echo "WARNING: Neither python3-config nor $PYTHON_VERSION_CONFIG found"
        fi
    else
        echo "python3-config already exists: $PYTHON_CONFIG"
    fi

    # 安装 MegatronAdaptor
    echo "Installing MegatronAdaptor..."
    cd "${MA_PATH}"
    pip install -e . -v

    # 安装TransformerEngineNPU
    echo "Installing TransformerEngineNPU..."
    cd "${WORKSPACE}"
    pip install -e TransformerEngineNPU -v --no-build-isolation

    # 使能megatron_adaptor
    cd "${WORKSPACE}"/Megatron-LM
    if ! grep -q "import megatron_adaptor" pretrain_gpt.py; then
        sed -i '/^import torch$/a\import megatron_adaptor' pretrain_gpt.py
    fi
    cd ..

    # print all packages
    pip list
}

# 环境清理
function clean_env() {
    echo "Cleaning up environment..."
    pip uninstall -y megatron_adaptor transformer_engine

    rm -f "${WORKSPACE}"/Megatron-LM/test
}

# 执行单测
function run_unit_test() {
    cd "${WORKSPACE}/Megatron-LM"
    bash "${MA_PATH}/test/unit_tests/run_unit_tests.sh" "./"
}

function run_test() {
    cd "${WORKSPACE}"/Megatron-LM

    ln -sf "${MA_PATH}/test" ./test

    # 控制执行时间，先只跑一个用例
    test_script="./test/system_tests/feature_tests/test_basic_training.sh"
    bash -e "$test_script"
}

function main() {
    # 仅文档变更时不触发用例构建
    if ! code_changed; then
        echo "Only documentation changed, skip test build."
        exit 0
    fi

    # 更新依赖
    echo "Download Code..."
    download_code

    # 环境准备
    echo "Setting up environment..."
    prepare_env

    # 执行单测
    echo "Running unit tests..."
    run_unit_test

    # 执行用例
    echo "Running tests..."
    run_test

    # 环境清理
    echo "Cleaning up environment..."
    clean_env
}

main
