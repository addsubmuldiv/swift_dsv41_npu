#!/bin/bash
# test_tokenizer_dataset.sh - Tokenizer和数据集特性测试
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-6103}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_tokenizer_dataset
rm -rf ${CKPT_DIR}

TP=2
PP=2
CP=2
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# tokenizer参数配置1：NullTokenizer
# --tokenizer-metadata 要求传入 JSON 元数据文件路径（而非内联 JSON 字符串），
# 因此先将元数据写入文件，再把文件路径传给该参数。
TOKENIZER_METADATA_FILE="./tokenizer_metadata.json"
echo '{"source": "st_test"}' > "$TOKENIZER_METADATA_FILE"
TOKENIZER_ARGS=(
    --tokenizer-type NullTokenizer \
    --vocab-size 32000 \
    --padded-vocab-size 32768 \
    --vocab-extra-ids 100 \
    --make-vocab-size-divisible-by 1
    --null-tokenizer-eod-id 0
    --null-tokenizer-pad-id 1
    --tokenizer-metadata $TOKENIZER_METADATA_FILE
)

# tokenizer参数配置2：Llama2Tokenizer
TOKENIZER_MODEL1="/home/dataset/model/llama-2-7b-hf/tokenizer.model"
TOKENIZER_ARGS1=(
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL1} \
)

# tokenizer参数配置3：HuggingFaceTokenizer + 外部 vocab/merges（验证 --vocab-file / --merge-file）
# 在脚本执行目录下生成一个仅含配置、不含词表的本地 gpt2 tokenizer 目录，
# 词表完全依赖 --vocab-file / --merge-file 加载，从而真正验证这两个参数生效。
GPT2_VOCAB_FILE="/home/dataset/enwiki/gpt2-vocab.json"
GPT2_MERGE_FILE="/home/dataset/enwiki/gpt2-merges.txt"
TOKENIZER_MODEL2="./gpt2_bpe_tokenizer"
mkdir -p "$TOKENIZER_MODEL2"
echo '{"model_type": "gpt2"}' > "$TOKENIZER_MODEL2/config.json"
echo '{"tokenizer_class": "GPT2Tokenizer"}' > "$TOKENIZER_MODEL2/tokenizer_config.json"
TOKENIZER_ARGS2=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL2
    --vocab-file $GPT2_VOCAB_FILE
    --merge-file $GPT2_MERGE_FILE
    --tokenizer-hf-no-use-fast
    --trust-remote-code
)

# 数据路径配置1：构建数据缓存
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/home/dataset/cache/megatron_st_cache}"
DATA_ARGS=(
    --train-data-path $DATA_PATH \
    --valid-data-path $DATA_PATH \
    --test-data-path $DATA_PATH \
    --seq-length 4096 \
    --sample-rate 0.8 \
    --mask-prob 0.15 \
    --short-seq-prob 0.1 \
    --num-workers 4 \
    --num-dataset-builder-threads 2 \
    --mid-level-dataset-surplus 0.005 \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss \
    --allow-ambiguous-pad-tokens
    --data-cache-path $DATA_CACHE_PATH
    --dataloader-type single
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
)

# 数据路径配置2：使用数据缓存（需与对应tokenizer配合构建缓存后再加载）
DATA_ARGS1_BUILD=(
    --train-data-path $DATA_PATH \
    --valid-data-path $DATA_PATH \
    --test-data-path $DATA_PATH \
    --seq-length 4096 \
    --data-cache-path $DATA_CACHE_PATH
)

DATA_ARGS1_LOAD=(
    --train-data-path $DATA_PATH \
    --valid-data-path $DATA_PATH \
    --test-data-path $DATA_PATH \
    --seq-length 4096 \
    --data-cache-path $DATA_CACHE_PATH \
    --dataloader-fast-cache-load \
    --dataloader-defer-npy-index-mmap
)

# 数据路径配置3：FIM（Fill-in-the-Middle）
DATA_ARGS2=(
    --data-path $DATA_PATH \
    --split 90,5,5 \
    --seq-length 4096 \
    --fim-data \
    --fim-rate 0.5 \
    --fim-spm-rate 0.5 \
    --fim-split-sample '<|file_separator|>' \
    --fim-fragment-rate 0.8 \
    --fim-no-prefix 'def' \
    --fim-prefix-token '<|fim_prefix|>' \
    --fim-middle-token '<|fim_middle|>' \
    --fim-suffix-token '<|fim_suffix|>' \
    --fim-pad-token '<|fim_pad|>' \
    --fim-eod-token '<|end_of_text|>'
)

TRAIN_ARGS=(
    --manual-gc \
    --manual-gc-interval 100 \
    --empty-unused-memory-level 1 \
    --no-pin-cpu-params \
    --no-pin-cpu-grads
    --memory-snapshot-path ./memory_snapshot \
    --check-weight-hash-across-dp-replicas-interval 100 \
    --no-check-for-nan-in-loss-and-grad \
    --check-for-large-grads \
    --check-for-spiky-loss \
    --result-rejected-tracker-filename result_rejected_tracker.json \
    --use-mcore-models \
    --disable-gloo-process-groups \
    --disable-jit-fuser \
    --dump-param-to-param-group-map ./param_group \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --weight-decay 0.1 \
    --no-weight-decay-cond-type apply_wd_to_qk_layernorm \
    --clip-grad 1.0 \
    --optimizer adam \
    --override-opt-param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --sgd-momentum 0.9 \
    --use-cpu-initialization \
    --fix-use-cpu-initialization \
    --init-method-std 0.01 \
    --embedding-init-method-std 0.014
    --no-manual-gc-eval
)

GPT_ARGS=(
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --context-parallel-size ${CP} \
    --cp-comm-type p2p \
    --use-flash-attn \
    --num-layers 4 \
    --hidden-size 1024 \
    --ffn-hidden-size 4000 \
    --num-attention-heads 16 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --decrease-batch-size-if-needed \
    --lr 1.0e-6 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
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
    --group-query-attention \
    --num-query-groups 8 \
    --lr-warmup-fraction 0.01 \
    --bf16 \
    --seed 42
)

OUTPUT_ARGS=(
    --save ${CKPT_DIR} \
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
    --train-sync-interval 100 \
    --exit-interval 1000 \
    --exit-duration-in-mins 30 \
    --exit-signal-handler \
    --exit-signal-handler-for-dataloader
)

# 测试1：NullTokenizer + 数据缓存构建
echo "Running NullTokenizer with data cache build..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

# 测试2：Llama2Tokenizer + 数据缓存加载
echo "Running Llama2Tokenizer with data cache build..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${TOKENIZER_ARGS1[@]} \
    ${DATA_ARGS1_BUILD[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

echo "Running Llama2Tokenizer with data cache load..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${TOKENIZER_ARGS1[@]} \
    ${DATA_ARGS1_LOAD[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

# 测试3：HuggingFaceTokenizer + FIM数据
echo "Running HuggingFaceTokenizer with FIM data..."
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${TRAIN_ARGS[@]} \
    ${TOKENIZER_ARGS2[@]} \
    ${DATA_ARGS2[@]} \
    ${OUTPUT_ARGS[@]} \
    --exit-interval 10

set +x
