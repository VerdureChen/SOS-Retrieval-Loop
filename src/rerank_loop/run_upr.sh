#!/usr/bin/env bash

# 指定GPU设备
export CUDA_VISIBLE_DEVICES=0

# 定义世界大小（分布式训练的进程数）
export WORLD_SIZE=1

MODEL_NAMES=(rankgpt) #monot5 upr bge rankgpt
DATA_NAMES=(nq)
BASE_METHOD=(bm25)
PORT=6000
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for DATA_NAME in "${DATA_NAMES[@]}"
  do
    echo "Running rerank for ${MODEL_NAME} on ${DATA_NAME}..."
    CONFIG_PATH="upr_configs/${MODEL_NAME}-config-${DATA_NAME}.json"
    LOG_DIR="logs/${MODEL_NAME}_${DATA_NAME}_rerank_base_${BASE_METHOD}.log"

    # 运行分布式Python脚本
    torchrun  --nproc_per_node ${WORLD_SIZE} \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr localhost \
        --master_port $PORT \
        rerank_for_loop.py \
        --config "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
    PORT=$((PORT+1))
    # 等待所有进程结束
#    wait
  done
done