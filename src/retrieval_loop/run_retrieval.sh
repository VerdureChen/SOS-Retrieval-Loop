#!/usr/bin/env bash


MODEL_NAMES=(contriever) #(bm25 dpr bge-base contriever retromae bge-large all-mpnet llm-ebedder)
DATA_NAMES=(nq)

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for DATA_NAME in "${DATA_NAMES[@]}"
  do
    echo "Running retrieval for ${MODEL_NAME}..."
    CONFIG_PATH="retrieve_configs/${MODEL_NAME}-config-${DATA_NAME}.json"
    LOG_DIR="../run_logs/test/${MODEL_NAME}_retrieval.log"

    python retrieve_methods.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
  done
wait
done
