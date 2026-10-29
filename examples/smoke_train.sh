#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/env.sh"
: "${MODEL_DIR:?Set MODEL_DIR to your local DeepSeek-V4.1-Flash-1017 weights directory}"
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5}
IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ ${#devices[@]} == 6 ]] || { echo 'This EP6 smoke example requires exactly six visible NPUs.' >&2; exit 1; }
export OMP_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=32
export HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto
export HF_HUB_OFFLINE=1 MODELSCOPE_OFFLINE=1 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export SMOKE_MODEL=${SMOKE_MODEL:-$ROOT/.local/reduced4}
export SMOKE_OUTPUT=${SMOKE_OUTPUT:-$ROOT/.local/runs/$(hostname)-$(date +%Y%m%d-%H%M%S)-$$}
"$PYTHON" "$ROOT/examples/prepare_reduced.py" --model "$MODEL_DIR" --output "$SMOKE_MODEL"
"$PYTHON" -m torch.distributed.run --nproc_per_node=6 --master_port="${MASTER_PORT:-29852}" "$ROOT/examples/train_smoke.py"
echo 'DEEPSEEK_V41_SMOKE_PASS'
