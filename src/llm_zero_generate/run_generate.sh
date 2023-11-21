#!/usr/bin/env bash

#MODEL_NAMES=(chatglm3-6b qwen-14b-chat) # running:pop  finished: wq nq trivia
#enumarate all the models
#MODEL_NAMES=(llama2-13b-chat baichuan2-13b-chat) #running: pop trivia finished: nq wq

MODEL_NAMES=(gpt-3.5-turbo) #running: pop trivia finished: nq wq
DATA_NAMES=(trivia)

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
  for DATA_NAME in "${DATA_NAMES[@]}"
  do
    echo "Generating responses for ${MODEL_NAME} on ${DATA_NAME}..."
    CONFIG_PATH="configs/${MODEL_NAME}-config-${DATA_NAME}.json"
    LOG_DIR="logs/${MODEL_NAME}_${DATA_NAME}_gen_rerun.log"

    python get_response_llm.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
  done
done
