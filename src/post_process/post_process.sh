#!/usr/bin/env bash

MODEL_NAMES=(qwen-0.5b-chat qwen-1.8b-chat qwen-4b-chat)
QUERY_DATA_NAMES=(nq pop tqa webq)
LOOP_NUM=0
LOOP_CONFIG_PATH_NAME="../run_configs/update_configs"
FROM_METHOD="update"

TOTAL_LOG_DIR="../run_logs/update_log"
TOTAL_OUTPUT_DIR="../../data_v2/update_data/DPR/update_passage_processed"
mkdir -p "${TOTAL_LOG_DIR}"
mkdir -p "${TOTAL_OUTPUT_DIR}"
mkdir -p "${LOOP_CONFIG_PATH_NAME}"

INPUT_FILE_PATH="../../data_v2/loop_output/DPR/zero_update_retrieval_result"

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"
  do
#    OUTPUT_DIR="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}"
#    mkdir -p "${OUTPUT_DIR}"
    echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
    INPUT_FILE_NAME="${INPUT_FILE_PATH}/${QUERY_DATA_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}-test-sample-200.jsonl_generate_context_ref_num_0.json"
    CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.json"
    LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.log"
    POSTPROCESS_OUTPUT_NAME="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}-test-gen-${MODEL_NAME}-postprocessed.jsonl"
    python ../rewrite_configs.py --method "${MODEL_NAME}" \
                            --data_name "${QUERY_DATA_NAME}" \
                            --loop "${LOOP_NUM}" \
                            --stage "post_process" \
                            --output_dir "${CONFIG_PATH}" \
                            --overrides '{"loop_num": "'"${LOOP_NUM}"'", "gen_model_name": "'"${MODEL_NAME}"'", "input_file": "'"${INPUT_FILE_NAME}"'", "output_dir": "'"${POSTPROCESS_OUTPUT_NAME}"'", "from_method": "'"${FROM_METHOD}"'"}'
    wait
    echo "Running postprocess for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
    python process_llm_text.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

  done
done
