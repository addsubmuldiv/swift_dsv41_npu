#!/usr/bin/env bash
# On each node: bash examples/train_full_lora_4nodes.sh NODE_RANK
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/env.sh"
PYTHON=${PYTHON:-$(command -v python || command -v python3)}
export NODE_RANK=${1:?Pass node rank 0/1/2/3}
MASTER_ADDR=${2:-${MASTER_ADDR:-}}
export NNODES=4 NPROC_PER_NODE=16 MASTER_PORT=${MASTER_PORT:-29500}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export OMP_NUM_THREADS=2 CUDA_DEVICE_MAX_CONNECTIONS=32
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
MODEL_DIR=${MODEL_DIR:-/storage/models/DeepSeek-V4.1-Flash-1017}
DATASET=${DATASET:-$ROOT/examples/data.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/.local/full_lora_check}
ENGRAM_MAP=${ENGRAM_MAP:-$ROOT/.local/full_lora/engram_tokenizer_map.json}
TRAIN_ITERS=${TRAIN_ITERS:-10}
[[ $NODE_RANK =~ ^[0-3]$ ]] || { echo 'Node rank must be 0, 1, 2 or 3' >&2; exit 1; }

# Start rank 0 first; other nodes read its address from the shared output directory.
mkdir -p "$OUTPUT_DIR"
if [[ $NODE_RANK != 0 && -z $MASTER_ADDR ]]; then
    echo 'Waiting for the master address on the shared disk...'
    deadline=$((SECONDS + 120))
    until [[ -s $OUTPUT_DIR/master_addr ]]; do
        (( SECONDS < deadline )) || { echo 'Start rank 0 first, using the same shared OUTPUT_DIR.' >&2; exit 1; }
        sleep 1
    done
    read -r MASTER_ADDR MASTER_PORT < "$OUTPUT_DIR/master_addr"
fi
read -r node_ip node_iface < <("$PYTHON" - "${MASTER_ADDR:-1.1.1.1}" <<'PYNET'
import fcntl
import os
import socket
import struct
import sys

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    def address(name):
        request = struct.pack('256s', name.encode()[:15])
        return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, request)[20:24])

    interface = os.environ.get('HCCL_SOCKET_IFNAME')
    if interface:
        ip = address(interface)
    else:
        # UDP connect asks the local routing table; it does not send a packet.
        sock.connect((sys.argv[1], 9))
        ip = sock.getsockname()[0]
        for _, name in socket.if_nameindex():
            try:
                if address(name) == ip:
                    interface = name
                    break
            except OSError:
                pass
    if not interface or ip.startswith('127.') or ip == '0.0.0.0':
        raise SystemExit('No routable IPv4 interface found; set HCCL_SOCKET_IFNAME to the communication interface.')
    print(ip, interface)
PYNET
)
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$node_iface}
if [[ $NODE_RANK == 0 ]]; then
    MASTER_ADDR=${MASTER_ADDR:-$node_ip}
    address_tmp=$(mktemp "$OUTPUT_DIR/.master_addr.XXXXXX")
    printf '%s %s\n' "$MASTER_ADDR" "$MASTER_PORT" > "$address_tmp"
    mv -f "$address_tmp" "$OUTPUT_DIR/master_addr"
fi
export MASTER_ADDR MASTER_PORT
echo "node=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT interface=$HCCL_SOCKET_IFNAME"

args=(sft
    --model "$MODEL_DIR" --model_type deepseek_v41
    --dataset "$DATASET" --split_dataset_ratio 0 --check_model false
    --engram_tokenizer_map "$ENGRAM_MAP" --output_dir "$OUTPUT_DIR" --add_version false
    --language_model_only true --mtp_num_layers 0
    --tuner_type lora --target_modules all-linear
    --lora_rank 8 --lora_alpha 16 --lora_dropout 0 --lora_dtype bfloat16 --merge_lora false
    --tensor_model_parallel_size 1 --pipeline_model_parallel_size 4
    --expert_model_parallel_size 16 --expert_tensor_parallel_size 1 --context_parallel_size 1
    --sequence_parallel false --overlap_p2p_comm false --batch_p2p_comm true
    --micro_batch_size 1 --global_batch_size 64
    --train_iters "$TRAIN_ITERS" --max_length "${MAX_LENGTH:-128}"
    --bf16 true --fp16 false --torch_dtype bfloat16 --use_distributed_optimizer true
    --optimizer adam --adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-8
    --weight_decay 0.1 --clip_grad 1
    --lr 1e-4 --min_lr 0 --lr_decay_style cosine --lr_decay_iters "$TRAIN_ITERS" --lr_warmup_iters 0
    --attention_backend unfused --use_fused_mhc false
    --gradient_accumulation_fusion false --moe_permute_fusion false
    --moe_shared_expert_overlap false --moe_grouped_gemm true
    --cross_entropy_loss_fusion false --masked_softmax_fusion false --padding_free false
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1
    --seed 42 --data_seed 42 --logging_steps 1 --dataloader_num_workers 0 --dataset_num_proc 1
    --save_steps "$TRAIN_ITERS" --save_safetensors false --no_save_optim true --no_save_rng true
    --megatron_extra_kwargs '{"hidden_dropout":0.0,"attention_dropout":0.0,"dsa_kernel_backend":"none","dspark_num_layers":0,"dspark_block_size":0}'
)
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "$PYTHON" -m swift.cli._megatron.main "${args[@]}"
    printf '\n'
    exit 0
fi
# The shared lock prevents four nodes from generating the same map concurrently.
mkdir -p "$(dirname -- "$ENGRAM_MAP")" "$OUTPUT_DIR/logs"
(
    flock 9
    if [[ ! -s $ENGRAM_MAP ]]; then
        "$PYTHON" "$ROOT/examples/prepare_engram_map.py" --model "$MODEL_DIR" --output "$ENGRAM_MAP"
    fi
) 9>"$ENGRAM_MAP.lock"
export HF_HUB_OFFLINE=1 MODELSCOPE_OFFLINE=1
"$PYTHON" -m swift.cli._megatron.main "${args[@]}" \
    2>&1 | tee "$OUTPUT_DIR/logs/node${NODE_RANK}-$(date +%Y%m%d-%H%M%S).log"
