#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6114}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./ckpt_fp8
rm -rf ${CHECKPOINT_PATH}
VOCAB_FILE=/home/dataset/enwiki/gpt2-vocab.json
MERGE_FILE=/home/dataset/enwiki/gpt2-merges.txt
DATA_PATH=/home/dataset/enwiki/my-t5_text_sentence

TP=2
PP=2
EP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --expert-model-parallel-size ${EP} \
    --num-experts 4 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type allgather \
    --moe-router-topk 2 \
    --moe-aux-loss-coeff 0.02 \
"

GPT_ARGS="
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 4 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 50 \
    --lr 1.0e-4 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.01 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-bias-dropout-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16
"

DATA_ARGS="
    --tokenizer-type NullTokenizer \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --vocab-size 50257 \
    --num-workers 4 \
    --split 949,50,1
"

OUTPUT_ARGS="
    --save $CHECKPOINT_PATH \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"

MXFP8_ARGS="
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --fp8-margin 0 \
    --fp8-interval 1 \
    --fp8-amax-history-len 1 \
    --fp8-amax-compute-algo most_recent \
    --no-fp8-wgrad \
    --keep-fp8-transpose-cache \
"

FP8_HYBRID_ARGS="
    --fp8-format hybrid \
    --fp8-recipe tensorwise \
    --fp8-margin 0 \
    --fp8-interval 1 \
    --fp8-amax-history-len 1 \
    --fp8-amax-compute-algo most_recent \
"

FP8_DELAYED_ARGS="
    --fp8-format e4m3 \
    --fp8-recipe delayed \
    --fp8-margin 0 \
    --fp8-interval 2 \
    --fp8-amax-history-len 10 \
    --fp8-amax-compute-algo max \
"

FP8_BLOCK_ARGS="
    --fp8-format e4m3 \
    --fp8-recipe blockwise \
    --fp8-margin 0 \
    --fp8-interval 1 \
    --fp8-amax-history-len 1 \
    --fp8-amax-compute-algo most_recent \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MXFP8_ARGS \
    --exit-interval 10

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FP8_HYBRID_ARGS \
    --exit-interval 10

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FP8_DELAYED_ARGS \
    --exit-interval 10

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $FP8_BLOCK_ARGS \
    --exit-interval 10 \

set +x
