#!/bin/bash
# test_moe.sh - MOE特性综合测试
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6107}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_moe
rm -rf ${CKPT_DIR}
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# MOE V2配置（均匀PP）
MOE_V2_ARGS=(
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 2 \
    --num-experts 8 \
    --moe-router-topk 1 \
    --moe-router-score-function softmax \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 0.25 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-shared-expert-intermediate-size 4096 \
    --moe-shared-expert-gate \
    --moe-grouped-gemm \
    --moe-permute-fusion \
    --moe-apply-probs-on-input \
    --moe-per-layer-logging \
    --moe-latent-size 2048 \
    --moe-ffn-hidden-size 1408 \
    --moe-layer-freq 1 \
    --moe-z-loss-coeff 1e-3 \
    --moe-input-jitter-eps 0.01 \
    --moe-enable-routing-replay \
    --moe-router-force-load-balancing \
    --moe-token-dispatcher-type alltoall
)

# MOE V2 EP通信重叠配置（与--overlap-dispatch-backward-with-experts-wgrad互斥）
MOE_V2_EP_OVERLAP_ARGS=(
    --overlap-moe-expert-parallel-comm
    --ep-overlap-early-attn-memory-release
)

# MOE V2 backward dispatch重叠配置（要求关闭--overlap-moe-expert-parallel-comm）
MOE_V2_DISPATCH_OVERLAP_ARGS=(
    --overlap-dispatch-backward-with-experts-wgrad
)

# MOE V3配置（不均匀PP）
MOE_V3_ARGS=(
    --expert-model-parallel-size 2 \
    --expert-tensor-parallel-size 1 \
    --num-experts 16 \
    --moe-router-topk 8 \
    --moe-router-score-function sigmoid \
    --moe-router-pre-softmax \
    --moe-router-topk-scaling-factor 0.25 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --moe-router-enable-expert-bias \
    --moe-router-bias-update-rate 1e-3 \
    --moe-router-dtype fp32 \
    --moe-router-fusion \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-shared-expert-intermediate-size 4096 \
    --moe-shared-expert-gate \
    --moe-shared-expert-overlap \
    --moe-grouped-gemm \
    --moe-permute-fusion \
    --moe-per-layer-logging \
    --moe-expert-capacity-factor 1.5 \
    --moe-pad-expert-input-to-capacity \
    --moe-token-drop-policy probs \
    --moe-ffn-hidden-size 1408 \
    --moe-layer-freq 1 \
    --moe-z-loss-coeff 1e-3 \
    --moe-input-jitter-eps 0.01 \
    --moe-token-dispatcher-type alltoall
)

# PP配置（均匀）
PP_UNIFORM_ARGS=(
    --pipeline-model-parallel-size 2 \
    --num-layers-per-virtual-pipeline-stage 1 \
    --microbatch-group-size-per-virtual-pipeline-stage 4 \
    --account-for-embedding-in-pipeline-split \
    --account-for-loss-in-pipeline-split \
    --overlap-p2p-communication-warmup-flush
)

# PP配置（不均匀）
PP_UNEVEN_ARGS=(
    --pipeline-model-parallel-size 4 \
    --decoder-first-pipeline-num-layers 2 \
    --decoder-last-pipeline-num-layers 2 \
    --num-virtual-stages-per-pipeline-rank 2 \
    --pipeline-model-parallel-comm-backend nccl
    --no-overlap-p2p-communication
)

GPT_ARGS=(
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --num-layers 8 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 1000 \
    --lr 5.0e-7 \
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
    --bf16 \
    --seed 42
)

DATA_ARGS=(
    --data-path $DATA_PATH \
    --split 949,50,1
)

OUTPUT_ARGS=(
    --save ${CKPT_DIR} \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10
)

# 测试1：MOE V2 + EP通信重叠 + 均匀PP
echo "Running MOE V2 with uniform PP..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${MOE_V2_ARGS[@]} \
    ${MOE_V2_EP_OVERLAP_ARGS[@]} \
    ${PP_UNIFORM_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

# 测试2：MOE V3 + 不均匀PP
echo "Running MOE V3 with uneven PP..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${MOE_V3_ARGS[@]} \
    ${PP_UNEVEN_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

# 测试3：MOE V2 + backward dispatch重叠 + 均匀PP
echo "Running MOE V2 with overlap-dispatch-backward-with-experts-wgrad..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${MOE_V2_ARGS[@]} \
    ${MOE_V2_DISPATCH_OVERLAP_ARGS[@]} \
    ${PP_UNIFORM_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

set +x
