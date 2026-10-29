#!/bin/bash
# test_checkpoint_convert.sh
# Test 1: save a torch-format source checkpoint
# Test 2: convert checkpoint from torch to torch_dist (--ckpt-convert-save/-format)
# Test 3: load with --pretrained-checkpoint
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6112}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_convert_src
CKPT_DIR_CONVERTED=./ckpt_convert_dst
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

GPT_ARGS=(
    --transformer-impl transformer_engine
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --sequence-parallel
    --num-layers 4
    --hidden-size 1024
    --ffn-hidden-size 4000
    --num-attention-heads 16
    --seq-length 1024
    --max-position-embeddings 1024
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --micro-batch-size 1
    --global-batch-size 4
    --train-iters 100
    --lr 1.0e-4
    --min-lr 1.0e-7
    --lr-decay-style cosine
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-5
    --init-method-std 0.01
    --disable-bias-linear
    --group-query-attention
    --num-query-groups 8
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --bf16
    --seed 42
)

DATA_ARGS=(
    --data-path ${DATA_PATH}
    --split 949,50,1
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 10
)

CKPT_SAVE_ARGS=(
    --save ${CKPT_DIR}
    --ckpt-format torch
    --save-interval 5
)

CKPT_CONVERT_ARGS=(
    --load ${CKPT_DIR}
    --ckpt-format torch
    --finetune
    --exit-on-missing-checkpoint
    --ckpt-convert-save ${CKPT_DIR_CONVERTED}
    --ckpt-convert-format torch_dist
    --ckpt-convert-update-legacy-dist-opt-format
)

CKPT_PRETRAINED_ARGS=(
    --pretrained-checkpoint ${CKPT_DIR}
    --ckpt-format torch
    --finetune
    --exit-on-missing-checkpoint
)

# Test 1: save source checkpoint (torch format)
echo "========================================"
echo "Running checkpoint save test (torch format)..."
echo "========================================"
rm -rf ${CKPT_DIR} ${CKPT_DIR_CONVERTED}
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${CKPT_SAVE_ARGS[@]} \
    --exit-interval 5

# Test 2: convert checkpoint to torch_dist format（转换在加载后立即执行并退出，不会实际训练，exit-interval 无实际作用）
echo "========================================"
echo "Running checkpoint convert test (torch -> torch_dist)..."
echo "========================================"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${CKPT_CONVERT_ARGS[@]} \
    --exit-interval 1

# Test 3: load with --pretrained-checkpoint（验证加载成功即可，只跑 1 步）
echo "========================================"
echo "Running checkpoint load test (--pretrained-checkpoint)..."
echo "========================================"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${CKPT_PRETRAINED_ARGS[@]} \
    --exit-interval 1

set +x
