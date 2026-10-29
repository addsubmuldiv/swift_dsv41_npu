#!/bin/bash
# test_mup.sh - µP (Maximal Update Parametrization) feature tests
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6108}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_mup
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

GPT_ARGS=(
    --transformer-impl transformer_engine
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --num-layers-per-virtual-pipeline-stage 1
    --sequence-parallel
    --num-layers 4
    --hidden-size 1024
    --ffn-hidden-size 4000
    --num-attention-heads 16
    --num-query-groups 8
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --seq-length 1024
    --max-position-embeddings 1024
    --micro-batch-size 1
    --global-batch-size 4
    --make-vocab-size-divisible-by 1
    --lr 1.0e-4
    --train-iters 50
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --init-method-std 0.01
    --disable-bias-linear
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --no-masked-softmax-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --bf16
    --group-query-attention
    --seed 42
)

DATA_ARGS=(
    --data-path ${DATA_PATH}
    --split 949,50,1
)

OUTPUT_ARGS=(
    --save ${CKPT_DIR}
    --log-throughput
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 10
)

# µP config
MUP_ARGS=(
    --use-mup
    --mup-width-mult 1.0
    --mup-base-hidden-size 1024
    --mup-embedding-mult 1.0
    --mup-output-mult 1.0
    --mup-base-head-dim 64
    --mup-attn-scale-power 1
)

echo "Running µP (Maximal Update Parametrization) test..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${MUP_ARGS[@]} \
    --exit-interval 10

set +x
