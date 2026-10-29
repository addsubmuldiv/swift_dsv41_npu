#!/bin/bash
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6104}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_activation_variants
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

# 激活函数
# 注意：MLM #5940 起 activation-func-clamp-value 用于 SwiGLU 时必须配 MoE，
# 非 MoE 配置不能再传该参数，故从基础 swiglu 配置中移除。
ACT_ARGS=(
    --swiglu
)

ACT_ARGS1=(
    --squared-relu
    --no-bias-gelu-fusion
)

# Quick-GeGLU activation
ACT_ARGS2=(
    --quick-geglu
    --no-bias-gelu-fusion
    --glu-linear-offset 0.0
    --activation-func-clamp-value 10.0
)

# 学习率
LR_SCHED_ARGS=(
    --lr 1.0e-6 \
    --min-lr 1.0e-7 \
    --lr-decay-style linear \
    --lr-decay-samples 40 \
    --lr-warmup-init 1.0e-7 \
    --lr-warmup-samples 20 \
    --lr-wsd-decay-style linear \
    --lr-wsd-decay-samples 40 \
    --train-samples 4000
)

LR_SCHED_ARGS1=(
    --lr 1.0e-6 \
    --min-lr 1.0e-7 \
    --lr-decay-style WSD \
    --lr-decay-iters 10 \
    --lr-warmup-init 1.0e-7 \
    --lr-warmup-iters 5 \
    --lr-wsd-decay-style linear \
    --lr-wsd-decay-iters 10 \
    --train-iters 1000
)

# Decoupled learning rate
LR_SCHED_ARGS2=(
    --lr 1.0e-6
    --min-lr 1.0e-7
    --decoupled-lr 1.0e-6
    --decoupled-min-lr 1.0e-7
    --lr-decay-style cosine
    --lr-warmup-fraction 0.01
    --train-iters 1000
)

# Mixed precision
MIX_PRC_ARGS=(
    --bf16 \
    --accumulate-allreduce-grads-in-fp32
)

# Normalization
NORM_ARGS=(
    --normalization RMSNorm
)

# Network size
NET_SIZE_ARGS=(
    --hidden-size 1024 \
    --ffn-hidden-size 4000 \
    --encoder-num-layers 4 \
    --decoder-num-layers 4 \
    --num-attention-heads 16 \
    --num-query-groups 8 \
)

# Loss
LOSS_ARGS=(
    --loss-scale 1.0
)

# Checkpoint
CKPT_ARGS=(
    --save ${CKPT_DIR} \
    --ckpt-format torch \
    --save-interval 500
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
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 50 \
    --eval-iters 10 \
)

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${MIX_PRC_ARGS[@]} \
    ${NET_SIZE_ARGS[@]} \
    ${NORM_ARGS[@]} \
    ${LOSS_ARGS[@]} \
    ${PARAL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${BASE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${ACT_ARGS[@]} \
    ${LR_SCHED_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${MIX_PRC_ARGS[@]} \
    ${NET_SIZE_ARGS[@]} \
    ${NORM_ARGS[@]} \
    ${LOSS_ARGS[@]} \
    ${PARAL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${BASE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${ACT_ARGS1[@]} \
    ${LR_SCHED_ARGS1[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${MIX_PRC_ARGS[@]} \
    ${NET_SIZE_ARGS[@]} \
    ${NORM_ARGS[@]} \
    ${LOSS_ARGS[@]} \
    ${PARAL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${BASE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${ACT_ARGS2[@]} \
    ${LR_SCHED_ARGS2[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

set +x
