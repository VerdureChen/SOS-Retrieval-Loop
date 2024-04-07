#!/usr/bin/env bash

RUN_DIR=$(pwd)
elasticsearch_url="http://124.16.138.150:9978"
#循环次数
TOTAL_LOOP_NUM=10
#方法设置
RETRIEVAL_MODEL_NAME=bm25 # dpr contriever retromae all-mpnet bge-base llm-embedder bm25
RERANK_MODEL_NAME=None #monot5 upr bge rankgpt
CORPUS_NAME=psgs_w100
FILTER_METHOD_NAME=None #None, filter_source, filter_bleu
GENERATE_TASK=generate
#QUERY_DATA_NAME=nq
#QUERY_FILE_PATH="../../data_v2/input_data/DPR/${QUERY_DATA_NAME}-test-h10.jsonl"

QUERY_DATA_NAMES=(nq webq)
QUERY_DATA_PATH="${RUN_DIR}/../data_v2/input_data/DPR/sampled_query"
QUERY_NAME_FORMAT="-test-sample-200.jsonl"

CONTEXT_REF_NUM=5
#NORMALIZE_EMBEDDINGS=False
GENERATE_MODEL_NAMES_F3=(qwen-0.5b-chat qwen-1.8b-chat qwen-4b-chat)
GENERATE_MODEL_NAMES_F7=(qwen-7b-chat llama2-7b-chat baichuan2-7b-chat)
GENERATE_MODEL_NAMES_F10=(gpt-3.5-turbo qwen-14b-chat llama2-13b-chat)
#GENERATE_MODEL_NAMES=(gpt-3.5-turbo chatglm3-6b qwen-14b-chat llama2-13b-chat baichuan2-13b-chat) #running: pop trivia finished: nq wq
#GENERATE_DATA_NAMES=(nq)
GENERATE_BASE_AND_KEY=(
   "gpt-3.5-turbo https://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "chatglm3-6b http://xxx.xxx.xxx.xxx:xx/v1 xxx"
#   "qwen-14b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "llama2-7b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "baichuan2-7b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "llama2-13b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "baichuan2-13b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "qwen-0.5b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "qwen-1.8b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "qwen-4b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "qwen-7b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
   "qwen-14b-chat http://xxx.xxx.xxx.xxx:xx/v1 xxx"
  )



#创建目录
TIMESTAMP=$(date +%Y%m%d%H%M%S)
#TIMESTAMP=20240206164013
# concanate all the query data names
QUERY_DATA_NAME=$(IFS=_; echo "${QUERY_DATA_NAMES[*]}")

LOOP_CONFIG_PATH_NAME="${RUN_DIR}/run_configs/update_${QUERY_DATA_NAME}_loop_config_${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}_total_loop_${TOTAL_LOOP_NUM}_${TIMESTAMP}"
mkdir -p $LOOP_CONFIG_PATH_NAME
TOTAL_LOG_DIR="${RUN_DIR}/run_logs/update_${QUERY_DATA_NAME}_loop_log_${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}_total_loop_${TOTAL_LOOP_NUM}_${TIMESTAMP}"
mkdir -p $TOTAL_LOG_DIR
OUTPUT_DIR="${RUN_DIR}/../data_v2/loop_output/DPR/update_${QUERY_DATA_NAME}_loop_output_${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}_total_loop_${TOTAL_LOOP_NUM}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "create config dir: ${LOOP_CONFIG_PATH_NAME}"
echo "create log dir: ${TOTAL_LOG_DIR}"
echo "create output dir: ${OUTPUT_DIR}"


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
#for INDEXING_MODEL_NAME in "${INDEXING_MODEL_NAMES[@]}"
#do
#  echo "rewrite config file for ${INDEXING_MODEL_NAME} on ${CORPUS_NAME}..."
#  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${INDEXING_MODEL_NAME}_${CORPUS_NAME}_indexing_loop_0.json"
#  LOG_DIR="${TOTAL_LOG_DIR}/${INDEXING_MODEL_NAME}_${CORPUS_NAME}_indexing_loop_0.log"
#  python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
#                          --method "${INDEXING_MODEL_NAME}" \
#                          --data_name "${CORPUS_NAME}" \
#                          --loop "0" \
#                          --stage "indexing" \
#                          --output_dir "${CONFIG_PATH}" \
#                          --overrides '{"index_name": "'"${INDEXING_MODEL_NAME}_test_index"'", "index_exists": false, "index_add_path":"'"${TOTAL_LOG_DIR}"'"}'
#
#  echo "Running indexing for ${INDEXING_MODEL_NAME} on ${CORPUS_NAME}..."
#  python embedding_index_incremental_corpus.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
#done
wait





# shellcheck disable=SC1073
for ((LOOP_NUM=1; LOOP_NUM<=${TOTAL_LOOP_NUM}; LOOP_NUM++))
do



  #retrieve
  cd ../retrieval_loop

  echo "rewrite config file for ${RETRIEVAL_MODEL_NAME} on ${QUERY_DATA_NAME}..."
  CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}.json"
  LOG_DIR="${TOTAL_LOG_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}.log"
  #RETRIEVAL_OUTPUT_PATH="${OUTPUT_DIR}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}.json"


  # Initialize empty strings for file lists
  QUERY_FILE_LIST=""
  OUTPUT_FILE_LIST=""

    # Loop through QUERY_DATA_NAMES to build up the file lists
    for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"; do
      QUERY_FILE_PATH="${QUERY_DATA_PATH}/${QUERY_DATA_NAME}${QUERY_NAME_FORMAT}"
      # Append the file path to the list, surrounded by quotes, separated by commas if not the first item
      if [ -z "${QUERY_FILE_LIST}" ]; then
        QUERY_FILE_LIST="\"${QUERY_FILE_PATH}\""
      else
        QUERY_FILE_LIST="${QUERY_FILE_LIST},\"${QUERY_FILE_PATH}\""
      fi
      OUTPUT_DIR_QUERY="${OUTPUT_DIR}/${QUERY_DATA_NAME}"
      mkdir -p $OUTPUT_DIR_QUERY
      OUTPUT_FILE_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}"
      if [ -z "${OUTPUT_FILE_LIST}" ]; then
        OUTPUT_FILE_LIST="\"${OUTPUT_FILE_PATH}\""
      else
        OUTPUT_FILE_LIST="${OUTPUT_FILE_LIST},\"${OUTPUT_FILE_PATH}\""
      fi
    done


  python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                          --method "${RETRIEVAL_MODEL_NAME}" \
                          --data_name "nq" \
                          --loop "${LOOP_NUM}" \
                          --stage "retrieval" \
                          --output_dir "${CONFIG_PATH}" \
                          --overrides '{"query_files": ['"${QUERY_FILE_LIST}"'], "output_files": ['"${OUTPUT_FILE_LIST}"'] , "elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": false}'
  wait
  echo "Running retrieval for ${RETRIEVAL_MODEL_NAME}"
  # if loop_num is 7, jump the retrieval
#  if [[ "${LOOP_NUM}" == "2" ]]; then
#    echo "jump the retrieval for ${RETRIEVAL_MODEL_NAME}"
#  else
#    python retrieve_methods.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
#  fi
  python retrieve_methods.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &

  wait







  #rerank

  for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}";
  do
    if [[ "${RERANK_MODEL_NAME}" != "None" ]]; then
    cd ../rerank_loop
    echo "rewrite config file for ${RERANK_MODEL_NAME} on ${QUERY_DATA_NAME}..."
    CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${RERANK_MODEL_NAME}_${QUERY_DATA_NAME}_rerank_loop_${LOOP_NUM}.json"
    LOG_DIR="${TOTAL_LOG_DIR}/${RERANK_MODEL_NAME}_${QUERY_DATA_NAME}_rerank_loop_${LOOP_NUM}.log"
    RERANK_OUTPUT_NAME="${RERANK_MODEL_NAME}_${QUERY_DATA_NAME}_rerank_loop_${LOOP_NUM}_based_on_${RETRIEVAL_MODEL_NAME}"
    RETRIEVAL_OUTPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}"
    python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                            --method "${RERANK_MODEL_NAME}" \
                            --data_name "${QUERY_DATA_NAME}" \
                            --loop "${LOOP_NUM}" \
                            --stage "rerank" \
                            --output_dir "${CONFIG_PATH}" \
                            --overrides '{"retriever_topk_passages_path": "'"${RETRIEVAL_OUTPUT_PATH}"'", "special_suffix": "'"${RERANK_OUTPUT_NAME}"'", "reranker_output_dir": "'"${OUTPUT_DIR}/${QUERY_DATA_NAME}"'", "elasticsearch_url": "'"${elasticsearch_url}"'"}'

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
    fi

    if [[ "${RERANK_MODEL_NAME}" != "None" ]]; then
      FILTER_INPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RERANK_OUTPUT_NAME}.json"
    else
      FILTER_INPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}"
    fi

    if [[ "${FILTER_METHOD_NAME}" != "None" ]]; then
      cd ../filtering
      echo "rewrite config file for ${FILTER_METHOD_NAME} on ${QUERY_DATA_NAME}..."
      CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${FILTER_METHOD_NAME}_${QUERY_DATA_NAME}_filter_loop_${LOOP_NUM}.json"
      LOG_DIR="${TOTAL_LOG_DIR}/${FILTER_METHOD_NAME}_${QUERY_DATA_NAME}_filter_loop_${LOOP_NUM}.log"
      FILTER_OUTPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/filter_${FILTER_METHOD_NAME}_${QUERY_DATA_NAME}_loop_${LOOP_NUM}_based_on_${RERANK_MODEL_NAME}_${RETRIEVAL_MODEL_NAME}"
      python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                              --method "${FILTER_METHOD_NAME}" \
                              --data_name "${QUERY_DATA_NAME}" \
                              --loop "${LOOP_NUM}" \
                              --stage "${FILTER_METHOD_NAME}" \
                              --output_dir "${CONFIG_PATH}" \
                              --overrides '{"input_file": "'"${FILTER_INPUT_PATH}"'", "output_file": "'"${FILTER_OUTPUT_PATH}"'", "elasticsearch_url": "'"${elasticsearch_url}"'", "max_self_bleu": 0.4, "num_docs": "'"${CONTEXT_REF_NUM}"'"}'
      wait
      echo "Running filter for ${FILTER_METHOD_NAME} on ${QUERY_DATA_NAME}..."
      python filter_for_loop.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
      wait
    fi



    #generate
    if [[ "${FILTER_METHOD_NAME}" != "None" ]]; then
      GENERATE_INPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/filter_${FILTER_METHOD_NAME}_${QUERY_DATA_NAME}_loop_${LOOP_NUM}_based_on_${RERANK_MODEL_NAME}_${RETRIEVAL_MODEL_NAME}"
    elif [[ "${RERANK_MODEL_NAME}" != "None" ]]; then
      GENERATE_INPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RERANK_OUTPUT_NAME}.json"
    else
      GENERATE_INPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_MODEL_NAME}_${QUERY_DATA_NAME}_retrieval_loop_${LOOP_NUM}"
    fi

    # if CONTEXT_REF_NUM less or equal to 5, use the first set of models, else use the second set
    if [[ "${LOOP_NUM}" -le 3 ]]; then
      GENERATE_MODEL_NAMES=("${GENERATE_MODEL_NAMES_F3[@]}")
    elif [[ "${LOOP_NUM}" -le 7 ]]; then
      GENERATE_MODEL_NAMES=("${GENERATE_MODEL_NAMES_F7[@]}")
    else
      GENERATE_MODEL_NAMES=("${GENERATE_MODEL_NAMES_F10[@]}")
    fi

    for MODEL_NAME in "${GENERATE_MODEL_NAMES[@]}"
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
      cd ../llm_zero_generate
      echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_generate_loop_${LOOP_NUM}.json"
      LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_generate_loop_${LOOP_NUM}.log"
      GENERATE_OUTPUT_NAME="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_generate_based_on_${RERANK_MODEL_NAME}_${RETRIEVAL_MODEL_NAME}_loop_${LOOP_NUM}_ref_${CONTEXT_REF_NUM}.jsonl"
      python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                              --method "${MODEL_NAME}" \
                              --data_name "${QUERY_DATA_NAME}" \
                              --loop "${LOOP_NUM}" \
                              --stage "${GENERATE_TASK}" \
                              --output_dir "${CONFIG_PATH}" \
                              --overrides '{"question_file_path": "'"${GENERATE_INPUT_PATH}"'", "output_file_path": "'"${GENERATE_OUTPUT_NAME}"'","context_ref_num": "'"${CONTEXT_REF_NUM}"'", "elasticsearch_url": "'"${elasticsearch_url}"'", "with_context": true, "api-base": "'"${API_BASE}"'", "api-key": "'"${API_KEY}"'"}'
#      wait
      echo "Running generate for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      python get_response_llm.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
    done
    wait

    for MODEL_NAME in "${GENERATE_MODEL_NAMES[@]}"
    do
      # postprocess
      cd ../post_process
      echo "rewrite config file for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.json"
      LOG_DIR="${TOTAL_LOG_DIR}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}.log"
      POSTPROCESS_OUTPUT_PATH="${OUTPUT_DIR}/${LOOP_NUM}"
      mkdir -p $POSTPROCESS_OUTPUT_PATH
      POSTPROCESS_OUTPUT_NAME="${POSTPROCESS_OUTPUT_PATH}/${MODEL_NAME}_${QUERY_DATA_NAME}_postprocess_loop_${LOOP_NUM}_based_on_${RERANK_MODEL_NAME}_${RETRIEVAL_MODEL_NAME}_ref_${CONTEXT_REF_NUM}.jsonl"
      FROM_METHOD="${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}"
      GENERATE_OUTPUT_NAME="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${MODEL_NAME}_${QUERY_DATA_NAME}_generate_based_on_${RERANK_MODEL_NAME}_${RETRIEVAL_MODEL_NAME}_loop_${LOOP_NUM}_ref_${CONTEXT_REF_NUM}.jsonl"
      python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                              --method "${MODEL_NAME}" \
                              --data_name "${QUERY_DATA_NAME}" \
                              --loop "${LOOP_NUM}" \
                              --stage "post_process" \
                              --output_dir "${CONFIG_PATH}" \
                              --overrides '{"loop_num": "'"${LOOP_NUM}"'", "gen_model_name": "'"${MODEL_NAME}"'", "input_file": "'"${GENERATE_OUTPUT_NAME}"'", "output_dir": "'"${POSTPROCESS_OUTPUT_NAME}"'", "from_method": "'"${FROM_METHOD}"'", "query_set_name": "'"${QUERY_DATA_NAME}"'"}'
      wait
      echo "Running postprocess for ${MODEL_NAME} on ${QUERY_DATA_NAME}..."
      python process_llm_text.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
      wait
    done

  done
  wait


  #incremental indexing
  cd ../retrieval_loop
  # merge all files in NEW_FILE_PATH into one
  mkdir -p "${POSTPROCESS_OUTPUT_PATH}/merged_file"
  cat ${POSTPROCESS_OUTPUT_PATH}/*.jsonl > ${POSTPROCESS_OUTPUT_PATH}/merged_file/merged.jsonl
  
  NEW_FILE_ADDR="${POSTPROCESS_OUTPUT_PATH}/merged_file/merged.jsonl"

  for INDEXING_MODEL_NAME in "${INDEXING_MODEL_NAMES[@]}"
  do
    echo "re-indexing for ${INDEXING_MODEL_NAME}..."
    CONFIG_PATH="${LOOP_CONFIG_PATH_NAME}/indexing_${INDEXING_MODEL_NAME}_loop_${LOOP_NUM}.json"
    LOG_DIR="${TOTAL_LOG_DIR}/indexing_${INDEXING_MODEL_NAME}_loop_${LOOP_NUM}.log"
    python ../rewrite_configs.py --total_config "${USER_CONFIG_PATH}" \
                            --method "${INDEXING_MODEL_NAME}" \
                            --data_name "${CORPUS_NAME}" \
                            --loop "${LOOP_NUM}" \
                            --stage "indexing" \
                            --output_dir "${CONFIG_PATH}" \
                            --overrides '{ "index_exists": true, "new_text_file": "'"${NEW_FILE_ADDR}"'", "page_content_column": "response", "index_add_path":"'"${TOTAL_LOG_DIR}"'","elasticsearch_url": "'"${elasticsearch_url}"'", "normalize_embeddings": false}'
    wait
    echo "Running ${INDEXING_MODEL_NAME} indexing..."
    python embedding_index_incremental_corpus.py --config_file_path "$CONFIG_PATH" > "$LOG_DIR" 2>&1 &
    wait

  done
  wait


done
