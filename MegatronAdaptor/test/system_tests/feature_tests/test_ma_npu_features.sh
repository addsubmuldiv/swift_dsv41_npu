#!/bin/bash
# MegatronAdaptor 新增 NPU 专项特性的冒烟级系统测试
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6115}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_ma_npu_features
rm -rf ${CKPT_DIR}
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

TP=2
PP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BASE_ARGS=(
    --transformer-impl transformer_engine \
    --use-flash-attn \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 2048 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --make-vocab-size-divisible-by 1 \
    --num-layers 4 \
    --hidden-size 1024 \
    --ffn-hidden-size 4000 \
    --num-attention-heads 16 \
    --num-query-groups 8 \
    --kv-channels 64 \
    --normalization LayerNorm \
    --apply-residual-connection-post-layernorm \
    --no-persist-layer-norm \
    --position-embedding-type rope \
    --no-masked-softmax-fusion \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --lr 1.0e-6 \
    --min-lr 1.0e-7 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --bf16 \
    --attention-softmax-in-fp32 \
    --initial-loss-scale 4096 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --use-distributed-optimizer \
    --group-query-attention \
    --seed 42 \
    --use-cpu-initialization \
    --rampup-batch-size 4 4 10 \
    --eval-micro-batch-size 1 \
    --eval-global-batch-size 4 \
    --train-iters 20
)

DATA_ARGS=(
    --data-path ${DATA_PATH} \
    --split 949,50,1
)

OUTPUT_ARGS=(
    --save ${CKPT_DIR} \
    --log-throughput \
    --log-interval 1 \
    --log-progress \
    --logging-level 20 \
    --eval-interval 100000 \
    --eval-iters 1 \
    --save-interval 100000
)

PROFILE_DIR=./profile_dir_ma_st
rm -rf ${PROFILE_DIR}

PROFILE_ARGS=(
    --profile
    --profile-ranks 0
    --profile-step-start 2
    --profile-step-end 4
    --profile-level level0
    --profile-with-cpu
    # --profile-with-stack
    --profile-with-memory
    --profile-record-shapes
    --profile-save-path ${PROFILE_DIR}
    --npu-deterministic
)

if python -c "import msprobe.pytorch" 2>/dev/null; then
    NPU_DATADUMP_ARGS=(--npu-datadump)
else
    echo "SKIP npu-datadump: mindstudio-probe (msprobe) is not installed"
fi

NPU_DETERMINISTIC_ARGS=(
    --npu-deterministic
)

echo "=== [ST] train with MA NPU features: profile + npu-deterministic [+ npu-datadump] ==="
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${BASE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${PROFILE_ARGS[@]} \
    ${NPU_DATADUMP_ARGS[@]} \
    ${NPU_DETERMINISTIC_ARGS[@]} \
    --exit-interval 8

# 校验 profile trace 产物已生成（窗口采集在训练退出时落盘）
if ! find ${PROFILE_DIR} -type f | grep -q .; then
    echo "FAIL: no trace file generated under ${PROFILE_DIR}"
    exit 1
fi
echo "=== [ST] profile: trace files found ==="

set +x
