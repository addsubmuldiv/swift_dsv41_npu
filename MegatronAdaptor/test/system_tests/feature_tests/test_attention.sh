#!/bin/bash
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6105}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_attention
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

GPT_ARGS=(
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --use-flash-attn \
    --num-layers 4 \
    --hidden-size 1024 \
    --ffn-hidden-size 4000 \
    --num-attention-heads 16 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 2048 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 100 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --disable-bias-linear \
    --num-query-groups 8 \
    --lr-warmup-fraction 0.01 \
    --bf16 \
    --seed 42
)

DATA_ARGS=(
    --data-path $DATA_PATH \
    --split 949,50,1
)

OUTPUT_ARGS=(
    --save ${CKPT_DIR} \
    --log-throughput \
    --log-interval 1 \
    --log-params-norm \
    --log-progress \
    --timing-log-level 0 \
    --logging-level 20 \
    --log-num-zeros-in-grad \
    --save-interval 10000 \
    --eval-interval 50 \
    --eval-iters 10 \
    --timing-log-option max
)

export NVTE_APPLY_QK_LAYER_SCALING=1
ATTN_COMMON_ARGS=(
    --attention-backend auto
    --flash-decode
)

GQA_ARGS=(
    --group-query-attention
    --apply-query-key-layer-scaling
    --qk-layernorm
    --relative-attention-num-buckets 8
    --relative-attention-max-distance 128
    --qk-l2-norm
    --apply-wd-to-qk-layernorm
    --attention-output-gate
    --rotary-interleaved
)

MLA_ARGS=(
    --multi-latent-attention \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
)

DP_ARGS=(
    --data-parallel-sharding-strategy no_shard \
    --data-parallel-random-init \
    --ddp-num-buckets 8 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --ddp-average-in-collective \
    --outer-dp-sharding-strategy no_shard
)

EMBEDDINGS_ARGS=(
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --rotary-base 10000 \
    --rotary-percent 1.0 \
    --rotary-seq-len-interpolation-factor 1 \
    --use-rope-scaling \
    --rope-scaling-factor 8.0 \
    --no-rope-fusion \
    --rotary-scaling-factor 1.0 \
    --mscale 1.0 \
    --mscale-all-dim 1.0 \
    --rope-type rope
)

# 与 EMBEDDINGS_ARGS 相同但刻意不传 --rope-type
EMBEDDINGS_ARGS_DEFAULT_ROPE=(
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --rotary-base 10000 \
    --rotary-percent 1.0 \
    --rotary-seq-len-interpolation-factor 1 \
    --use-rope-scaling \
    --rope-scaling-factor 8.0 \
    --no-rope-fusion \
    --rotary-scaling-factor 1.0 \
    --mscale 1.0 \
    --mscale-all-dim 1.0 \
)

COMM_ARGS=(
    --use-tp-pp-dp-mapping \
    --sequence-parallel \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --overlap-param-gather-with-optimizer-step \
    --tp-comm-overlap
    --disable-tp-comm-bulk-wgrad
    --disable-tp-comm-bulk-dgrad
    --disable-tp-comm-overlap-ag
    --disable-tp-comm-overlap-rs
    --disable-tp-comm-split-ag
    --disable-tp-comm-split-rs
    --no-clone-scatter-output-in-embedding
    --suggested-communication-unit-size 4194304
    --disable-symmetric-registration
    --no-align-grad-reduce
    --no-align-param-gather
    --symmetric-ar-type two_shot
)

DIST_ARGS=(
    --use-distributed-optimizer \
    --distributed-backend nccl \
    --distributed-timeout-minutes 3 \
    --ckpt-format torch \
    --use-sharp \
    --sharp-enabled-group dp \
    --fix-gtp-sharp-init-ordering \
    --create-all-gather-group \
)

RECOMPUTE_ARGS=(
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 4 \
)

DECAY_ARGS=(
    --wgrad-deferral-limit 1 \
    --weight-decay-incr-style cosine \
    --start-weight-decay 1 \
    --end-weight-decay 5 \
)

STRAGGLER_ARGS=(
    --straggler-ctrlr-port 6012 \
    --straggler-minmax-count 1 \
    --disable-straggler-on-startup \
)

FUSION_ARGS=(
    --no-gradient-reduce-div-fusion \
    --no-bias-gelu-fusion \
    --no-bias-swiglu-fusion \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    --use-fused-weighted-squared-relu \
)

# GQA
export NVTE_APPLY_QK_LAYER_SCALING=1
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${ATTN_COMMON_ARGS[@]} \
    ${GQA_ARGS[@]} \
    ${DP_ARGS[@]} \
    ${EMBEDDINGS_ARGS[@]} \
    ${COMM_ARGS[@]} \
    ${DIST_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${DECAY_ARGS[@]} \
    ${STRAGGLER_ARGS[@]} \
    ${FUSION_ARGS[@]} \
    --exit-interval 10 \

# MLA
export NVTE_APPLY_QK_LAYER_SCALING=0
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${ATTN_COMMON_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${DP_ARGS[@]} \
    ${EMBEDDINGS_ARGS[@]} \
    ${COMM_ARGS[@]} \
    ${DIST_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${DECAY_ARGS[@]} \
    ${STRAGGLER_ARGS[@]} \
    ${FUSION_ARGS[@]} \
    --exit-interval 10 \

# MLA + 缺省 RoPE（回归）：刻意不传 --rope-type，验证 CLI 缺省的 None 不会覆盖
# MLATransformerConfig.rope_type 的 dataclass 默认值 yarn，模型可正常构建；
export NVTE_APPLY_QK_LAYER_SCALING=0
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${ATTN_COMMON_ARGS[@]} \
    ${MLA_ARGS[@]} \
    ${DP_ARGS[@]} \
    ${EMBEDDINGS_ARGS_DEFAULT_ROPE[@]} \
    ${COMM_ARGS[@]} \
    ${DIST_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${DECAY_ARGS[@]} \
    ${STRAGGLER_ARGS[@]} \
    ${FUSION_ARGS[@]} \
    --exit-interval 5

set +x
