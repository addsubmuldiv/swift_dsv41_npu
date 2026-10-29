#!/usr/bin/env bash
_dsv41_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z ${ASCEND_HOME_PATH:-} ]]; then
    for _dsv41_cann in /usr/local/Ascend/cann/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
        if [[ -f $_dsv41_cann ]]; then
            _dsv41_shell_flags=$-
            set +u
            source "$_dsv41_cann"
            [[ $_dsv41_shell_flags != *u* ]] || set -u
            break
        fi
    done
fi
unset _dsv41_cann _dsv41_shell_flags
export MEGATRON_LM_PATH="$_dsv41_root/Megatron-LM"
export PYTHONPATH="$_dsv41_root/Megatron-LM:$_dsv41_root/MegatronAdaptor:$_dsv41_root/TransformerEngineNPU:$_dsv41_root/mcore-bridge/src:$_dsv41_root/ms-swift"
export NO_VCS_VERSION=1
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/swift_dsv41_npu-$UID/torch_extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/swift_dsv41_npu-$UID/triton}
unset _dsv41_root
