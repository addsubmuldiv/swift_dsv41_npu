#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/example/env_npu_ma.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./ckpt_gpt
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
    --train-iters 10 \
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
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --exit-interval 10

set +x
