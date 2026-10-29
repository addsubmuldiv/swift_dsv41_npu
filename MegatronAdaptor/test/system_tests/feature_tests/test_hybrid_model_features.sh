#!/bin/bash
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6101}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_hybrid_model_features
rm -rf ${CKPT_DIR}
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

TP=2
PP=2
CP=1
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

SPECIAL_ARGS=(
    --mamba-state-dim 64 \
    --mamba-head-dim 32 \
    --mamba-num-groups 4 \
    --mamba-num-heads 4 \
    --is-hybrid-model \
    --add-qkv-bias \
    --mtp-num-layers 2 \
    --mtp-standalone
    --no-barrier-with-level-1-timing
)

# Activation function
ACT_ARGS=(
    --openai-gelu
)

# Mixed precision
MIX_PRC_ARGS=(
    --bf16 \
    --first-last-layers-bf16 \
    --num-layers-at-start-in-bf16 0 \
    --num-layers-at-end-in-bf16 1 \
    --grad-reduce-in-bf16 \
    --disable-bf16-reduced-precision-matmul \
    --attention-softmax-in-fp32 \
    --hysteresis 2
)

# Normalization
NORM_ARGS=(
    --normalization LayerNorm \
    --apply-residual-connection-post-layernorm \
    --no-persist-layer-norm
)

# Network size
NET_SIZE_ARGS=(
    --hidden-size 1024 \
    --ffn-hidden-size 4000 \
    --num-layers 4 \
    --num-attention-heads 16 \
    --num-query-groups 8 \
    --kv-channels 64
)

# Loss
LOSS_ARGS=(
    --initial-loss-scale 4096 \
    --loss-scale-window 50 \
    --min-loss-scale 1.0 \
    --calculate-per-token-loss \
    --mtp-loss-scaling-factor 0.2
)

# Learning rate schedule
LR_SCHED_ARGS=(
    --lr 1.0e-6 \
    --min-lr 1.0e-7 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01
)

# Parallelism
PARAL_ARGS=(
    --sequence-parallel \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --tensor-model-parallel-size ${TP} \
)

BASE_ARGS=(
    --transformer-impl transformer_engine \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    --use-flash-attn \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 2048 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --make-vocab-size-divisible-by 1 \
    --train-iters 1000 \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --no-masked-softmax-fusion \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --group-query-attention \
    --seed 42
)

DATA_ARGS=(
    --data-path ${DATA_PATH} \
    --split 949,50,1
)

OUTPUT_ARGS=(
    --save ${CKPT_DIR} \
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 50 \
    --eval-iters 10 \
)

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${ACT_ARGS[@]} \
    ${MIX_PRC_ARGS[@]} \
    ${NET_SIZE_ARGS[@]} \
    ${NORM_ARGS[@]} \
    ${LOSS_ARGS[@]} \
    ${LR_SCHED_ARGS[@]} \
    ${PARAL_ARGS[@]} \
    ${BASE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${SPECIAL_ARGS[@]} \
    --exit-interval 10

set +x
