# ============================================
# MegatronAdaptor Docker Image for CI Build
# NPU Type: Configurable (910b, a3, etc.)
# Supports: x86_64 and aarch64
# Supports: Ubuntu and openEuler CANN base images
# ============================================

ARG OS=openeuler24.03
ARG BASE_IMAGE_VERSION=9.1.0-beta.3
ARG NPU_TYPE=910b
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=""

FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py${PYTHON_VERSION}}

USER root
SHELL ["/bin/bash", "-c"]

ARG OS_FAMILY=openeuler
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1.post6

# Prepare required system dependencies
COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm -f /tmp/configure_repo.sh

RUN ARCH=$(uname -m) && \
    echo "Detected CPU architecture: ${ARCH}" && \
    if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then \
        echo "ERROR: Unsupported architecture: ${ARCH}"; \
        exit 1; \
    fi

RUN echo "Installing system dependencies for ${OS_FAMILY}..." && \
    if [ "$OS_FAMILY" = "ubuntu" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            build-essential \
            cmake \
            curl \
            gcc \
            g++ \
            git \
            jq \
            libnuma-dev \
            vim \
            wget && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    elif [ "$OS_FAMILY" = "openeuler" ]; then \
        yum install -y \
            cmake \
            curl \
            findutils \
            gcc \
            gcc-c++ \
            git \
            jq \
            make \
            numactl-devel \
            tar \
            vim \
            wget \
            which && \
        yum clean all && \
        rm -rf /var/cache/yum; \
    else \
        echo "ERROR: Unsupported OS family: ${OS_FAMILY}"; \
        exit 1; \
    fi && \
    rm -rf /tmp/* /var/tmp/* && \
    pip install --upgrade pip packaging setuptools==80.10.2 && \
    pip cache purge

RUN PYTHON_BIN=$(which python3) && \
    PYTHON_DIR=$(dirname "$PYTHON_BIN") && \
    PYTHON_CONFIG="$PYTHON_DIR/python3-config" && \
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2) && \
    PYTHON_VERSION_CONFIG="$PYTHON_DIR/python${PYTHON_VERSION}-config" && \
    if [ ! -f "$PYTHON_CONFIG" ] && [ ! -L "$PYTHON_CONFIG" ]; then \
        echo "python3-config not found, checking $PYTHON_VERSION_CONFIG..."; \
        if [ -f "$PYTHON_VERSION_CONFIG" ]; then \
            ln -sf "$PYTHON_VERSION_CONFIG" "$PYTHON_CONFIG"; \
            echo "Created symlink: $PYTHON_CONFIG -> $PYTHON_VERSION_CONFIG"; \
        else \
            echo "WARNING: Neither python3-config nor $PYTHON_VERSION_CONFIG found"; \
        fi; \
    else \
        echo "python3-config already exists: $PYTHON_CONFIG"; \
    fi

# Install required dependencies
COPY wheels/ /tmp/wheels/
COPY third_party/ /tmp/third_party/
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"; \
    else \
        pip config set global.extra-index-url "https://repo.huaweicloud.com/repository/pypi/simple"; \
    fi && \
    DEVLIB_DIR=$(find /usr/local/Ascend -type d -path "*/devlib/linux/${ARCH}" | head -n 1) && \
    if [ -n "$DEVLIB_DIR" ]; then \
        export LD_LIBRARY_PATH="${DEVLIB_DIR}:${LD_LIBRARY_PATH}"; \
    fi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then \
        source /usr/local/Ascend/nnal/atb/set_env.sh; \
    fi && \
    pip install torch==${TORCH_VERSION} torch_npu==${TORCH_NPU_VERSION} && \
    pip install transformers==4.57.1 && \
    pip install sentencepiece einops pydantic importlib-metadata && \
    TRITON_WHL=$(ls /tmp/wheels/triton_ascend-*_${ARCH}.whl 2>/dev/null | head -n 1) && \
    if [ -z "$TRITON_WHL" ]; then \
        echo "ERROR: triton-ascend wheel for ${ARCH} not found under /tmp/wheels/"; \
        exit 1; \
    fi && \
    pip install "$TRITON_WHL" && \
    EMERGING_SRC=/tmp/third_party/Emerging-Optimizers && \
    if [ ! -f "$EMERGING_SRC/pyproject.toml" ] && [ ! -f "$EMERGING_SRC/setup.py" ]; then \
        echo "ERROR: Emerging-Optimizers source not found under docker/third_party/Emerging-Optimizers"; \
        exit 1; \
    fi && \
    pip install "$EMERGING_SRC" && \
    rm -rf /tmp/* /var/tmp/* && \
    pip cache purge

# Show install results
RUN pip list

# Setting Default Commands
CMD ["/bin/bash"]
