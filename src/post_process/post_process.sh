#!/usr/bin/env bash

MODEL_NAMES=(chatglm3-6b-chat baichuan2-13b-chat qwen-14b-chat llama2-13b-chat gpt-3.5-turbo)
QUERY_DATA_NAMES=(nq pop tqa webq)
LOOP_NUM=0
LOOP_CONFIG_PATH_NAME="../run_configs/zero-shot_retrieval_config"

TOTAL_LOG_DIR="../run_logs/zero-shot_retrieval_log"
TOTAL_OUTPUT_DIR="../../data_v2/zero_gen_data/DPR/post_processed_sampled_data"
mkdir -p "${TOTAL_LOG_DIR}"
mkdir -p "${TOTAL_OUTPUT_DIR}"
mkdir -p "${LOOP_CONFIG_PATH_NAME}"

INPUT_FILE_PATH="../../data_v2/zero_gen_data/DPR/sampled_data"

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"
  do
#    OUTPUT_DIR="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}"
#    mkdir -p "${OUTPUT_DIR}"
    echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
    INPUT_FILE_NAME="${INPUT_FILE_PATH}/${QUERY_DATA_NAME}-test-gen-${MODEL_NAME}.jsonl"
    CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.json"
    LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.log"
    POSTPROCESS_OUTPUT_NAME="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}-test-gen-${MODEL_NAME}-postprocessed.jsonl"
    python ../rewrite_configs.py --method "${MODEL_NAME}" \
                            --data_name "${QUERY_DATA_NAME}" \
                            --loop "${LOOP_NUM}" \
                            --stage "post_process" \
                            --output_dir "${CONFIG_PATH}" \
                            --overrides '{"loop_num": "'"${LOOP_NUM}"'", "gen_model_name": "'"${MODEL_NAME}"'", "input_file": "'"${INPUT_FILE_NAME}"'", "output_dir": "'"${POSTPROCESS_OUTPUT_NAME}"'"}'
    wait
    echo "Running postprocess for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
    python process_llm_text.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

  done
done
