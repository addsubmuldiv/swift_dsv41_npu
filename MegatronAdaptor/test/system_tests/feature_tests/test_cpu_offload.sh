#!/bin/bash
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=TRUE

DEVICES_PER_NODE=${NPU_PER_NODE:-8}
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6113}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($DEVICES_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_cpu_offload
rm -rf ${CKPT_DIR}
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

TP_SIZE=2
PP_SIZE=1

DISTRIBUTED_ARGS="
    --nproc_per_node $DEVICES_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

CPUOFFLOAD_ARGS=(
    --cpu-offloading-num-layers 2 \
    --optimizer-cpu-offload \
    --optimizer-offload-fraction 1.0 \
    --use-torch-optimizer-for-cpu-offload \
    --overlap-cpu-optimizer-d2h-h2d \
    --use-precision-aware-optimizer \
    --use-distributed-optimizer \
    --no-pin-cpu-grads
    --no-pin-cpu-params
)

GPT_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE:-1} \
    --pipeline-model-parallel-size ${PP_SIZE:-1} \
    --sequence-parallel \
    --num-layers 8 \
    --hidden-size 512 \
    --ffn-hidden-size 1376 \
    --num-attention-heads 8 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size ${MBS:-2} \
    --global-batch-size ${GBS:-16} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-5 \
    --train-iters 10 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-9 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 2
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
)

DATA_ARGS=(
    --data-path ${DATA_PATH} \
    --split 949,50,1
)

OUTPUT_ARGS=(
    --save ${CKPT_DIR} \
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 10 \
    --eval-iters 2 \
)

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${CPUOFFLOAD_ARGS[@]} \
    --distributed-backend nccl \
    --exit-interval 10 \

set +x
