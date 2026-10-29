#!/bin/bash
# test_checkpoint_advanced.sh
# Tests 1/2: torch-format synchronous save and load
# Tests 3/4: torch_dist-format asynchronous save and load
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6111}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_adv
rm -rf ${CKPT_DIR}
CKPT_DIR_ASYNC=./ckpt_adv_async
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
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 4 \
    --train-iters 1000 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --lr 1.0e-4 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --init-method-std 0.01 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-bias-dropout-fusion \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16
)

DATA_ARGS=(
    --data-path ${DATA_PATH} \
    --split 949,50,1
)

OUTPUT_ARGS=(
    --log-throughput \
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 10
)

CKPT_SAVE_ARGS=(
    --save ${CKPT_DIR} \
    --ckpt-format torch \
    --save-interval 5 \
    --no-save-optim \
    --no-save-rng \
    --ckpt-fully-parallel-save \
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_unexpected
    --ckpt-assume-constant-structure
    --use-persistent-ckpt-worker
    --ckpt-fully-parallel-save-process-group dp
)

CKPT_LOAD_ARGS=(
    --load ${CKPT_DIR} \
    --ckpt-format torch \
    --no-load-optim \
    --no-load-rng \
    --auto-detect-ckpt-format
    --finetune
    --exit-on-missing-checkpoint
    --use-checkpoint-args
    --ckpt-fully-parallel-load-process-group dp
    --no-ckpt-load-validate-sharding-integrity
)

CKPT_ASYNC_SAVE_ARGS=(
    --save ${CKPT_DIR_ASYNC} \
    --ckpt-format torch_dist \
    --save-interval 5 \
    --async-save \
    --use-persistent-ckpt-worker \
    --use-distributed-optimizer \
    --dist-ckpt-optim-fully-reshardable
    --ckpt-fully-parallel-load
    --ckpt-fully-parallel-load-exchange-algo broadcast
)

CKPT_ASYNC_LOAD_ARGS=(
    --load ${CKPT_DIR_ASYNC} \
    --ckpt-format torch_dist \
    --use-distributed-optimizer \
    --ckpt-fully-parallel-load
    --ckpt-fully-parallel-load-exchange-algo broadcast
    --use-checkpoint-opt-param-scheduler
    --use-mp-args-from-checkpoint-args
)

# Test 1: save checkpoint
echo "========================================"
echo "Running checkpoint save test..."
echo "========================================"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${CKPT_SAVE_ARGS[@]} \
    --exit-interval 5

# Test 2: load checkpoint（验证加载成功即可，只跑 1 步）
echo "========================================"
echo "Running checkpoint load test..."
echo "========================================"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${CKPT_LOAD_ARGS[@]} \
    --exit-interval 1

# Test 3: torch_dist async checkpoint save
echo "========================================"
echo "Running checkpoint async (torch_dist) save test..."
echo "========================================"
rm -rf ${CKPT_DIR_ASYNC}
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${CKPT_ASYNC_SAVE_ARGS[@]} \
    --exit-interval 10

# Test 4: load the torch_dist async checkpoint and resume training（验证加载成功即可，只跑 1 步）
echo "========================================"
echo "Running checkpoint async (torch_dist) load test..."
echo "========================================"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${CKPT_ASYNC_LOAD_ARGS[@]} \
    --exit-interval 1

set +x
