#!/usr/bin/env bash

#循环次数
TOTAL_LOOP_NAME=3
#方法设置
RETRIEVAL_MODEL_NAME=dpr # dpr contriever retromae all-mpnet bge-base llm-embedder bm25
RERANK_MODEL_NAME=monot5 #monot5 upr bge rankgpt
CORPUS_NAME=psgs_w100
QUERY_DATA_NAME=nq
QUERY_FILE_PATH="../../data_v2/input_data/DPR/${QUERY_DATA_NAME}-test-h10.jsonl"
GENERATE_MODEL_NAMES=(gpt-3.5-turbo) #running: pop trivia finished: nq wq
#GENERATE_DATA_NAMES=(nq)

#创建目录
TIMESTAMP=$(date +%Y%m%d%H%M%S)
RUN_DIR=$(pwd)
LOOP_CONFIG_PATH_NAME="${RUN_DIR}/run_configs/loop_config_${TIMESTAMP}"
mkdir -p $LOOP_CONFIG_PATH_NAME
TOTAL_LOG_DIR="${RUN_DIR}/run_logs/loop_log_${TIMESTAMP}"
mkdir -p $TOTAL_LOG_DIR
OUTPUT_DIR="${RUN_DIR}/../data_v2/test_output/DPR/loop_output_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

#write your config in total_config.json
USER_CONFIG_PATH="${RUN_DIR}/test_function/test_configs/template_total_config.json"



# 指定rerank GPU设备
export CUDA_VISIBLE_DEVICES=0

# 定义rerank世界大小（分布式训练的进程数）
export WORLD_SIZE=1
PORT=6000

# if RETRIEVAL_MODEL_NAME is not bm25, add bm25 to indexing model name
if [[ "${RETRIEVAL_MODEL_NAME}" != "bm25" ]]; then
  INDEXING_MODEL_NAMES=("bm25" "${RETRIEVAL_MODEL_NAME}")
else
  INDEXING_MODEL_NAMES=("${RETRIEVAL_MODEL_NAME}")
fi

# indexing before loop
cd retrieval_loop
for INDEXING_MODEL_NAME in "${INDEXING_MODEL_NAMES[@]}"
do
  echo "rewrite config file for ${INDEXING_MODEL_NAME} on ${CORPUS_NAME}..."
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${INDEXING_MODEL_NAME}_${CORPUS_NAME}_indexing_loop_0.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${INDEXING_MODEL_NAME}_${CORPUS_NAME}_indexing_loop_0.log"
  python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                          --method "${INDEXING_MODEL_NAME}" \
                          --data_name "${CORPUS_NAME}" \
                          --loop "0" \
                          --stage "indexing" \
                          --output_dir "${CONFIG_PATH}" \
                          --overrides '{"index_name": "'"${INDEXING_MODEL_NAME}_test_index"'", "index_exists": false, "index_add_path":"'"${TOTAL_LOG_DIR}"'"}'

  echo "Running indexing for ${INDEXING_MODEL_NAME} on ${CORPUS_NAME}..."
  python embedding_index_incremental_corpus.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
done
wait

# shellcheck disable=SC1073
for ((LOOP_NUM=1; LOOP_NUM<=${TOTAL_LOOP_NAME}; LOOP_NUM++))
do

#retrieve
cd ../retrieval_loop

echo "rewrite config file for ${RETRIEVAL_MODEL_NAME} on ${QUERY_DATA_NAME}..."
CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}.json"
LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}.log"
RETRIEVAL_OUTPUT_PATH="${OUTPUT_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}.json"

python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                        --method "${RETRIEVAL_MODEL_NAME}" \
                        --data_name "${QUERY_DATA_NAME}" \
                        --loop "${LOOP_NUM}" \
                        --stage "retrieval" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"index_name": "'"${RETRIEVAL_MODEL_NAME}_test_index"'", "query_files": ["'"${QUERY_FILE_PATH}"'"], "output_files": ["'"${RETRIEVAL_OUTPUT_PATH}"'"]}'
wait
echo "Running retrieval for ${RETRIEVAL_MODEL_NAME} on ${QUERY_DATA_NAME}..."
python retrieve_methods.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

wait

#rerank
cd ../rerank_loop

echo "rewrite config file for ${RERANK_MODEL_NAME} on ${QUERY_DATA_NAME}..."
CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RERANK_MODEL_NAME}_${QUERY_DATA_NAME}_rerank_loop_${LOOP_NUM}.json"
LOG_DIR="${TOTAL_LOG_DIR}/${RERANK_MODEL_NAME}_${QUERY_DATA_NAME}_rerank_loop_${LOOP_NUM}.log"
RERANK_OUTPUT_NAME="${RERANK_MODEL_NAME}_${QUERY_DATA_NAME}_rerank_loop_${LOOP_NUM}_based_on_${RETRIEVAL_MODEL_NAME}"
python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                        --method "${RERANK_MODEL_NAME}" \
                        --data_name "${QUERY_DATA_NAME}" \
                        --loop "${LOOP_NUM}" \
                        --stage "rerank" \
                        --output_dir "${CONFIG_PATH}" \
                        --overrides '{"index_name": "bm25_test_index", "retriever_topk_passages_path": "'"${RETRIEVAL_OUTPUT_PATH}"'", "special_suffix": "'"${RERANK_OUTPUT_NAME}"'", "reranker_output_dir": "'"${OUTPUT_DIR}"'"}'

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
#PORT=$((PORT+1))

wait

#generate
cd ../llm_zero_generate
for MODEL_NAME in "${GENERATE_MODEL_NAMES[@]}"
do

  echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_generate_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_generate_loop_${LOOP_NUM}.log"
  GENERATE_OUTPUT_NAME="${OUTPUT_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_generate_based_on_${RERANK_MODEL_NAME}_${RETRIEVAL_MODEL_NAME}_loop_${LOOP_NUM}.jsonl"
  python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                          --method "${MODEL_NAME}" \
                          --data_name "${QUERY_DATA_NAME}" \
                          --loop "${LOOP_NUM}" \
                          --stage "generate" \
                          --output_dir "${CONFIG_PATH}" \
                          --overrides '{"index_name": "bm25_test_index", "question_file_path": "'"${OUTPUT_DIR}/${RERANK_OUTPUT_NAME}.json"'", "output_file_path": "'"${GENERATE_OUTPUT_NAME}"'"}'
  wait
  echo "Running generate for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
  python get_response_llm.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &


  wait

  # postprocess
  cd ../post_process
  echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.log"
  POSTPROCESS_OUTPUT_NAME="${OUTPUT_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}_based_on_${RERANK_MODEL_NAME}_${RETRIEVAL_MODEL_NAME}.jsonl"
  python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                          --method "${MODEL_NAME}" \
                          --data_name "${QUERY_DATA_NAME}" \
                          --loop "${LOOP_NUM}" \
                          --stage "post_process" \
                          --output_dir "${CONFIG_PATH}" \
                          --overrides '{"loop_num": "'"${LOOP_NUM}"'", "gen_model_name": "'"${MODEL_NAME}"'", "input_file": "'"${GENERATE_OUTPUT_NAME}"'", "output_dir": "'"${POSTPROCESS_OUTPUT_NAME}"'"}'
  wait
  echo "Running postprocess for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
  python process_llm_text.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &


#incremental indexing
  cd ../retrieval_loop


  echo "re-indexing for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
  for INDEXING_MODEL_NAME in "${INDEXING_MODEL_NAMES[@]}"
  do
    CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_indexing_${INDEXING_MODEL_NAME}_loop_${LOOP_NUM}.json"
    LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_indexing_${INDEXING_MODEL_NAME}_loop_${LOOP_NUM}.log"
    python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                            --method "${INDEXING_MODEL_NAME}" \
                            --data_name "${CORPUS_NAME}" \
                            --loop "${LOOP_NUM}" \
                            --stage "indexing" \
                            --output_dir "${CONFIG_PATH}" \
                            --overrides '{"query_set_name":"'"${QUERY_DATA_NAME}"'","index_name": "'"${INDEXING_MODEL_NAME}_test_index"'", "index_exists": true, "new_text_file": "'"${POSTPROCESS_OUTPUT_NAME}"'", "page_content_column": "response", "index_add_path":"'"${TOTAL_LOG_DIR}"'"}'
    wait
    echo "Running ${INDEXING_MODEL_NAME} indexing for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
    python embedding_index_incremental_corpus.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &


    done
    wait
done

done