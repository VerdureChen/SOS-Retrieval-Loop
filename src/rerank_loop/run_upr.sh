#!/usr/bin/env bash

# 指定GPU设备
export CUDA_VISIBLE_DEVICES=0

# 定义世界大小（分布式训练的进程数）
export WORLD_SIZE=1

RERANK_MODEL_NAMES=(bge monot5) #monot5 upr bge rankgpt
QUERY_DATA_NAMES=(nq) #nq webq tqa pop
RETRIEVAL_MODEL_NAMES=(bm25 all-mpnet bge-base bge-large contriever dpr retromae llm-embedder)
PORT=13423
LOOP_CONFIG_PATH_NAME="../run_configs/original_retrieval_config"

TOTAL_LOG_DIR="../run_logs/original_retrieval_log"
TOTAL_OUTPUT_DIR="../../data_v2/loop_output/DPR/original_retrieval_result"
mkdir -p "${TOTAL_LOG_DIR}"
mkdir -p "${TOTAL_OUTPUT_DIR}"


for RERANK_MODEL_NAME in "${RERANK_MODEL_NAMES[@]}"; do
  for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"; do
    for RETRIEVAL_MODEL_NAME in "${RETRIEVAL_MODEL_NAMES[@]}"; do
      RETRIEVAL_OUTPUT_NAME="${QUERY_DATA_NAME}-test-${RETRIEVAL_MODEL_NAME}"
      RETRIEVAL_OUTPUT_PATH="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_OUTPUT_NAME}"
      OUTPUT_DIR="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}"
      mkdir -p "${OUTPUT_DIR}"
      CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${QUERY_DATA_NAME}_${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}_rerank.json"
      LOG_DIR="${TOTAL_LOG_DIR}/${QUERY_DATA_NAME}_${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}_rerank.log"
      RERANK_OUTPUT_NAME="${QUERY_DATA_NAME}-${RERANK_MODEL_NAME}_rerank_based_on_${RETRIEVAL_MODEL_NAME}"
      python ../rewrite_configs.py --method "${RERANK_MODEL_NAME}" \
                          --data_name "${QUERY_DATA_NAME}" \
                          --stage "rerank" \
                          --output_dir "${CONFIG_PATH}" \
                          --overrides '{"retriever_topk_passages_path": "'"${RETRIEVAL_OUTPUT_PATH}"'", "special_suffix": "'"${RERANK_OUTPUT_NAME}"'", "reranker_output_dir": "'"${OUTPUT_DIR}"'"}'

      wait
      echo "Running rerank for ${RERANK_MODEL_NAME} on ${QUERY_DATA_NAME}..."
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
      wait

    done
  done
done

