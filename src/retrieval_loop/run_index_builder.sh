#!/usr/bin/env bash


MODEL_NAMES=(llm-embedder bge-base) # dpr contriever) # retromae all-mpnet bge llm-embedder bm25 contriever dpr)
DATA_NAMES=(psgs_w100)

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for DATA_NAME in "${DATA_NAMES[@]}"
  do
    echo "Running index bulider for ${MODEL_NAME} on ${DATA_NAME}..."
    CONFIG_PATH="configs/${MODEL_NAME}-config-${DATA_NAME}.json"
    LOG_DIR="logs/${MODEL_NAME}_${DATA_NAME}_indexing.log"

    python embedding_index_incremental_corpus.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
  done
done
