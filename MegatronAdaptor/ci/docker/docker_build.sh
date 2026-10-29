#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

NPU_TYPE="910b"
OS="openeuler24.03"
BASE_IMAGE_VERSION="9.1.0-beta.3"
BASE_IMAGE=""
PYTHON_VERSION="3.12"
TORCH_VERSION="2.10.0"
TORCH_NPU_VERSION="2.10.0"
TRITON_VERSION="3.2.1"
TRITON_RELEASE_BASE_URL="https://gitcode.com/Ascend/triton-ascend/releases/download"
EMERGING_OPTIMIZERS_VERSION="v0.2.0"
EMERGING_OPTIMIZERS_REPO="https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git"
IMAGE_NAME="megatronadaptor-ci"
NO_CACHE=""
CLEANUP_ON_FAIL=false
NPU_TYPE_EXPLICIT=false
OS_EXPLICIT=false

cleanup_dangling() {
    echo ">>> Cleaning up dangling images and corresponding containers..."
    local dangling_images
    dangling_images=$(docker images -f "dangling=true" -q 2>/dev/null || true)
    if [ -n "$dangling_images" ]; then
        for img_id in $dangling_images; do
            local containers
            containers=$(docker ps -a -q --filter "ancestor=$img_id" 2>/dev/null || true)
            if [ -n "$containers" ]; then
                docker rm -f $containers 2>/dev/null || true
            fi
        done
        docker rmi $dangling_images 2>/dev/null || true
    fi
}

prepare_build_context() {
    local wheels_dir="${SCRIPT_DIR}/wheels"
    local third_party_dir="${SCRIPT_DIR}/third_party"
    mkdir -p "$wheels_dir" "$third_party_dir"

    if ls "${wheels_dir}"/triton_ascend-*_"${ARCH_NAME}".whl >/dev/null 2>&1; then
        echo ">>> triton-ascend wheel for ${ARCH_NAME} already present, skip download"
    else
        local cp_tag="cp${PYTHON_VERSION//./}"
        local whl_file="triton_ascend-${TRITON_VERSION}-${cp_tag}-${cp_tag}-manylinux_2_27_${ARCH_NAME}.manylinux_2_28_${ARCH_NAME}.whl"
        local whl_url="${TRITON_RELEASE_BASE_URL}/v${TRITON_VERSION}/${whl_file}"
        echo ">>> Downloading triton-ascend wheel: ${whl_url}"
        if ! wget -O "${wheels_dir}/${whl_file}" "${whl_url}"; then
            rm -f "${wheels_dir}/${whl_file}"
            echo "ERROR: Failed to download triton-ascend wheel: ${whl_url}"
            exit 1
        fi
    fi

    local emerging_dir="${third_party_dir}/Emerging-Optimizers"
    if [ -f "${emerging_dir}/pyproject.toml" ] || [ -f "${emerging_dir}/setup.py" ]; then
        echo ">>> Emerging-Optimizers source already present, skip clone"
    else
        rm -rf "${emerging_dir}"
        echo ">>> Cloning Emerging-Optimizers ${EMERGING_OPTIMIZERS_VERSION}..."
        if ! git clone --depth 1 --branch "${EMERGING_OPTIMIZERS_VERSION}" "${EMERGING_OPTIMIZERS_REPO}" "${emerging_dir}"; then
            echo "ERROR: Failed to clone Emerging-Optimizers: ${EMERGING_OPTIMIZERS_REPO}"
            exit 1
        fi
    fi
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build MegatronAdaptor Docker Image for CI Build

Options:
    -t, --npu-type TYPE       NPU type: a3 or 910b (default: 910b)
    -o, --os OS               OS: openeuler24.03 or ubuntu22.04 (default: openeuler24.03, auto-detected from --base-image if not specified)
    -i, --image-name NAME     Image name (default: megatronadaptor)
    -n, --no-cache            Build without cache
    --base-image-version VER  Base image CANN version (default: 9.1.0-beta.3)
    --base-image IMAGE        Full base image name (higher priority than --base-image-version; passed through unchanged)
    --python-version VER      Python tag in the CANN base image (default: 3.12)
    --torch-version VER       PyTorch version (default: 2.7.1)
    --torch-npu-version VER   torch_npu version (default: 2.7.1.post6)
    --cleanup-on-fail         Clean dangling images/containers if build fails
    -h, --help                Show help

Examples:
    bash $0
    bash $0 -t a3 -o openeuler24.03 --base-image-version 9.1.0-beta.3
    bash $0 --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.3-910b-openeuler24.03-py3.12

Note:
    CANN base image tags use lowercase chip names, such as a3 and 910b. A full --base-image value is used exactly as provided.
EOF
}

parse_base_image_tag() {
    local image="$1"
    local tag="${image##*:}"
    local tag_lower
    tag_lower=$(echo "$tag" | tr '[:upper:]' '[:lower:]')

    if [[ "$tag_lower" == *"910b"* ]]; then
        DETECTED_NPU_TYPE="910b"
    elif [[ "$tag_lower" == *"-a3-"* ]] || [[ "$tag_lower" == *"-a3-py"* ]]; then
        DETECTED_NPU_TYPE="a3"
    fi

    if [[ "$tag_lower" == *"openeuler24.03"* ]]; then
        DETECTED_OS="openeuler24.03"
    elif [[ "$tag_lower" == *"ubuntu22.04"* ]]; then
        DETECTED_OS="ubuntu22.04"
    fi

    if [[ "$tag_lower" =~ py([0-9]+\.[0-9]+) ]]; then
        DETECTED_PYTHON_VERSION="${BASH_REMATCH[1]}"
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--npu-type)        NPU_TYPE="$2"; NPU_TYPE_EXPLICIT=true; shift 2 ;;
        -o|--os)              OS="$2"; OS_EXPLICIT=true; shift 2 ;;
        -i|--image-name)      IMAGE_NAME="$2"; shift 2 ;;
        -n|--no-cache)        NO_CACHE="--no-cache"; shift ;;
        --base-image-version) BASE_IMAGE_VERSION="$2"; shift 2 ;;
        --base-image)         BASE_IMAGE="$2"; shift 2 ;;
        --python-version)     PYTHON_VERSION="$2"; shift 2 ;;
        --torch-version)      TORCH_VERSION="$2"; shift 2 ;;
        --torch-npu-version)  TORCH_NPU_VERSION="$2"; shift 2 ;;
        --cleanup-on-fail)    CLEANUP_ON_FAIL=true; shift ;;
        -h|--help)            show_help; exit 0 ;;
        *)                    echo "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

DETECTED_NPU_TYPE=""
DETECTED_OS=""
DETECTED_PYTHON_VERSION=""
if [ -n "$BASE_IMAGE" ]; then
    parse_base_image_tag "$BASE_IMAGE"
    if [ "$NPU_TYPE_EXPLICIT" = false ] && [ -n "$DETECTED_NPU_TYPE" ]; then
        NPU_TYPE="$DETECTED_NPU_TYPE"
    fi
    if [ "$OS_EXPLICIT" = false ] && [ -n "$DETECTED_OS" ]; then
        OS="$DETECTED_OS"
    fi
    if [ -n "$DETECTED_PYTHON_VERSION" ]; then
        PYTHON_VERSION="$DETECTED_PYTHON_VERSION"
    fi
fi

NPU_TYPE_LOWER=$(echo "$NPU_TYPE" | tr '[:upper:]' '[:lower:]')
OS=$(echo "$OS" | tr '[:upper:]' '[:lower:]')

if [ "$NPU_TYPE_LOWER" != "a3" ] && [ "$NPU_TYPE_LOWER" != "910b" ]; then
    echo "Error: NPU type must be a3 or 910b"
    exit 1
fi

if [ "$OS" != "openeuler24.03" ]; then
    echo "Error: OS must be openeuler24.03"
    exit 1
fi

case "$OS" in
    openeuler*) OS_FAMILY="openeuler"; REPO_SCRIPT="configure_yum_repo.sh" ;;
esac

HOST_ARCH=$(uname -m)
case "$HOST_ARCH" in
    arm64) ARCH_NAME="aarch64" ;;
    *)     ARCH_NAME="$HOST_ARCH" ;;
esac

cd "$SCRIPT_DIR"
cp "${SCRIPT_DIR}/${REPO_SCRIPT}" configure_repo.sh
trap 'rm -f configure_repo.sh' EXIT

BUILD_ARGS="--build-arg OS=${OS}"
BUILD_ARGS="$BUILD_ARGS --build-arg OS_FAMILY=${OS_FAMILY}"
BUILD_ARGS="$BUILD_ARGS --build-arg NPU_TYPE=${NPU_TYPE_LOWER}"
BUILD_ARGS="$BUILD_ARGS --build-arg PYTHON_VERSION=${PYTHON_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_VERSION=${TORCH_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_NPU_VERSION=${TORCH_NPU_VERSION}"

if [ -n "$BASE_IMAGE" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE=${BASE_IMAGE}"
else
    BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE_VERSION=${BASE_IMAGE_VERSION}"
fi

echo "=========================================="
echo "Build Configuration"
echo "=========================================="
echo "NPU Type:           ${NPU_TYPE_LOWER}"
echo "OS:                 ${OS}"
echo "OS Family:          ${OS_FAMILY}"
echo "CPU Architecture:   ${ARCH_NAME}"
echo "Dockerfile:         ${DOCKERFILE}"
echo "Image Name:         ${IMAGE_NAME}"
echo "Base Image Version: ${BASE_IMAGE_VERSION}"
if [ -n "$BASE_IMAGE" ]; then
    echo "Base Image:         ${BASE_IMAGE}"
fi
echo "Python Version:     ${PYTHON_VERSION}"
echo "PyTorch Version:    ${TORCH_VERSION}"
echo "torch_npu Version:  ${TORCH_NPU_VERSION}"
echo "triton Version:     ${TRITON_VERSION}"
echo "Emerging-Optimizers:${EMERGING_OPTIMIZERS_VERSION}"
echo "No Cache:           ${NO_CACHE:-No}"
echo "=========================================="

prepare_build_context

set +e
docker build \
    -t "$IMAGE_NAME" \
    -f "$DOCKERFILE" \
    $BUILD_ARGS \
    $NO_CACHE \
    --network=host \
    .
BUILD_RESULT=$?
set -e

if [ $BUILD_RESULT -eq 0 ]; then
    echo "=========================================="
    echo "Build Complete!"
    echo "Image: ${IMAGE_NAME}"
    echo "=========================================="
    exit 0
fi

echo "=========================================="
echo "Build Failed!"
echo "=========================================="
if [ "$CLEANUP_ON_FAIL" = true ]; then
    cleanup_dangling
fi
exit $BUILD_RESULT
