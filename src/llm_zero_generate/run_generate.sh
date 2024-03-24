#!/usr/bin/env bash


MODEL_NAMES=(chatglm3-6b) #chatglm3-6b qwen-14b-chat llama2-13b-chat baichuan2-13b-chat gpt-3.5-turbo
GENERATE_BASE_AND_KEY=(
   "gpt-3.5-turbo http://124.16.138.150:8113/v1 xxx"
   "chatglm3-6b http://124.16.138.150:8113/v1 xxx"
   "qwen-14b-chat http://124.16.138.150:8113/v1 xxx"
   "llama2-13b-chat http://124.16.138.150:8113/v1 xxx"
   "baichuan2-13b-chat http://124.16.138.150:8113/v1 xxx"
  )

DATA_NAMES=(tqa pop nq webq)
CONTEXT_REF_NUM=1
QUESTION_FILE_NAMES=(
  "-test-bm25"
  "-test-contriever"
  "-test-bge-base"
  "-test-llm-embedder"
  "-upr_rerank_based_on_bm25.json"
  "-monot5_rerank_based_on_bm25.json"
  "-bge_rerank_based_on_bm25.json"
  "-upr_rerank_based_on_bge-base.json"
  "-monot5_rerank_based_on_bge-base.json"
  "-bge_rerank_based_on_bge-base.json"
)
LOOP_CONFIG_PATH_NAME="../run_configs/original_retrieval_config"

TOTAL_LOG_DIR="../run_logs/original_retrieval_log"
QUESTION_FILE_PATH_TOTAL="../../data_v2/loop_output/DPR/original_retrieval_result"
TOTAL_OUTPUT_DIR="../../data_v2/loop_output/DPR/original_retrieval_result"
mkdir -p "${TOTAL_LOG_DIR}"
mkdir -p "${TOTAL_OUTPUT_DIR}"
mkdir -p "${LOOP_CONFIG_PATH_NAME}"

LOOP_NUM=1
for ((i=0;i<${LOOP_NUM};i++))
do
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  # 遍历键值对数组
    for entry in "${GENERATE_BASE_AND_KEY[@]}"; do
        if [[ ${entry} == $MODEL_NAME* ]]; then
            # 读取URL和key
            read -ra ADDR <<< "$entry"
            API_BASE=${ADDR[1]}
            API_KEY=${ADDR[2]}
            break
        fi
    done
  for QUERY_DATA_NAME in "${DATA_NAMES[@]}"
  do
    for QUESTION_FILE_NAME in "${QUESTION_FILE_NAMES[@]}"
    do
      OUTPUT_DIR="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}"
      mkdir -p "${OUTPUT_DIR}"
      QUESTION_FILE_PATH="${QUESTION_FILE_PATH_TOTAL}/${QUERY_DATA_NAME}/${QUERY_DATA_NAME}${QUESTION_FILE_NAME}"
      echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}${QUESTION_FILE_NAME}_generate_context_ref_num_${CONTEXT_REF_NUM}.json"
      LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}${QUESTION_FILE_NAME}_generate_context_ref_num_${CONTEXT_REF_NUM}.log"
      GENERATE_OUTPUT_NAME="${OUTPUT_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}${QUESTION_FILE_NAME}_generate_context_ref_num_${CONTEXT_REF_NUM}.json"
      python ../rewrite_configs.py --method "${MODEL_NAME}" \
                              --data_name "${QUERY_DATA_NAME}" \
                              --stage "generate" \
                              --output_dir "${CONFIG_PATH}" \
                              --overrides '{"question_file_path": "'"${QUESTION_FILE_PATH}"'", "output_file_path": "'"${GENERATE_OUTPUT_NAME}"'", "context_ref_num": "'"${CONTEXT_REF_NUM}"'", "api-base": "'"${API_BASE}"'", "api-key": "'"${API_KEY}"'"}'
      wait
      echo "Running generate for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      python get_response_llm.py --config_file_path "${CONFIG_PATH}" > "${LOG_DIR}" 2>&1 &
      wait
    done

  done
done
done