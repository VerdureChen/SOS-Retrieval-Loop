#!/usr/bin/env bash

#MODEL_NAMES=(chatglm3-6b qwen-14b-chat) # running:pop  finished: wq nq trivia
#enumarate all the models
#MODEL_NAMES=(llama2-13b-chat baichuan2-13b-chat gpt-3.5-turbo) #running: pop trivia finished: nq wq

MODEL_NAMES=(chatglm3-6b) #running: pop trivia finished: nq wq
DATA_NAMES=(webq nq pop tqa)
QUESTION_FILE_NAMES=(
  "-test-bge-base"
  "-test-bm25"
  "-test-contriever"
  "-test-llm-embedder"
  "-bge_rerank_based_on_bm25.json"
  "-bge_rerank_based_on_bge-base.json"
  "-monot5_rerank_based_on_bge-base.json"
  "-monot5_rerank_based_on_bm25.json"
  "-upr_rerank_based_on_bge-base.json"
  "-upr_rerank_based_on_bm25.json"
)
LOOP_CONFIG_PATH_NAME="../run_configs/trash"

TOTAL_LOG_DIR="../run_logs/trash"
QUESTION_FILE_PATH_TOTAL="../../data_v2/loop_output/DPR/original_retrieval_result"
TOTAL_OUTPUT_DIR="../../data_v2/loop_output/DPR/trash"
mkdir -p "${TOTAL_LOG_DIR}"
mkdir -p "${TOTAL_OUTPUT_DIR}"
mkdir -p "${LOOP_CONFIG_PATH_NAME}"

LOOP_NUM=10
for ((i=0;i<${LOOP_NUM};i++))
do
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for QUERY_DATA_NAME in "${DATA_NAMES[@]}"
  do
    for QUESTION_FILE_NAME in "${QUESTION_FILE_NAMES[@]}"
    do
      OUTPUT_DIR="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}"
      mkdir -p "${OUTPUT_DIR}"
      QUESTION_FILE_PATH="${QUESTION_FILE_PATH_TOTAL}/${QUERY_DATA_NAME}/${QUERY_DATA_NAME}${QUESTION_FILE_NAME}"
      echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}${QUESTION_FILE_NAME}_generate.json"
      LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}${QUESTION_FILE_NAME}_generate.log"
      GENERATE_OUTPUT_NAME="${OUTPUT_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}${QUESTION_FILE_NAME}"
      python ../rewrite_configs.py --method "${MODEL_NAME}" \
                              --data_name "${QUERY_DATA_NAME}" \
                              --stage "generate" \
                              --output_dir "${CONFIG_PATH}" \
                              --overrides '{"question_file_path": "'"${QUESTION_FILE_PATH}"'", "output_file_path": "'"${GENERATE_OUTPUT_NAME}"'"}'
      wait
      echo "Running generate for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      python get_response_llm.py --config_file_path "${CONFIG_PATH}" > "${LOG_DIR}" 2>&1 &
      wait
    done

  done
done
done