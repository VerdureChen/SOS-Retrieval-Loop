#!/usr/bin/env bash
# 定义元组数组 "item retrieval_method rerank_method"
RUN_DIR=$(pwd)
CORPUS_NAME=psgs_w100
QUERY_DATA_NAMES=(nq pop tqa webq)
QUERY_DATA_PATH="${RUN_DIR}/../data_v2/input_data/DPR/sampled_query"
QUERY_NAME_FORMAT="-test-sample-200.jsonl"
OUTPUT_DIR="${RUN_DIR}/../data_v2/loop_output/DPR/zero-shot_retrieval_result"
for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"; do
    mkdir -p "${OUTPUT_DIR}/${QUERY_DATA_NAME}"
done
NEW_FILE_PATH="${RUN_DIR}/../data_v2/zero_gen_data/DPR/post_processed_sampled_data"
LOG_PATH="${RUN_DIR}/run_logs/zero-shot_retrieval_log"
CONFIG_PATH="${RUN_DIR}/run_configs/zero-shot_retrieval_config"
mkdir -p "${LOG_PATH}"
mkdir -p "${CONFIG_PATH}"
PORT=29500
WORLD_SIZE=1
run_items=(
#  "item1 bm25 None"
#  "item2 contriever None"
#  "item3 bge-base None"
#  "item4 llm-embedder None"
#  "item5 bm25 bge"
  "item6 bm25 monot5"
#  "item7 bm25 upr"
#  "item8 bge-base bge"
  "item9 bge-base monot5"
#  "item10 bge-base upr"
)
GENERATE_MODEL_NAMES=(baichuan2-13b-chat qwen-14b-chat gpt-3.5-turbo llama2-13b-chat chatglm3-6b) #running: pop trivia finished: nq wq
# 新建一个空数组用于存储去重后的检索方法名称
uniq_retrieval_methods=()

# 遍历run_items数组
for item in "${run_items[@]}"; do
  # 使用空格分割item的值
  IFS=' ' read -r -a tuple <<< "$item"

  # 获取检索方法名称
  retrieval_method="${tuple[1]}"

  # 判断检索方法名称是否已经存在于去重后的数组中
  # shellcheck disable=SC2199
  # shellcheck disable=SC2076
  if [[ ! " ${uniq_retrieval_methods[@]} " =~ " ${retrieval_method} " ]]; then
    # 将检索方法名称添加到去重后的数组中
    uniq_retrieval_methods+=("${retrieval_method}")
  fi
done

# if BM25 is not in uniq_retrieval_methods, add it
# shellcheck disable=SC2199
# shellcheck disable=SC2076
if [[ ! " ${uniq_retrieval_methods[@]} " =~ " bm25 " ]]; then
  uniq_retrieval_methods+=("bm25")
fi


# 打印去重后的检索方法名称
echo "去重后的检索方法名称:"
for retrieval_method in "${uniq_retrieval_methods[@]}"; do
  echo "$retrieval_method"
done





# merge all files in NEW_FILE_PATH into one
cat ${NEW_FILE_PATH}/*.jsonl > ${NEW_FILE_PATH}/merged_file/merged.jsonl
NEW_FILE_ADDR="${NEW_FILE_PATH}/merged_file/merged.jsonl"
# reindex
cd retrieval_loop
for retrieval_method in "${uniq_retrieval_methods[@]}"; do
  # create log dir
  LOG_DIR="${LOG_PATH}/${retrieval_method}_add_index.log"
#  mkdir -p "${LOG_DIR}"
  # create config dir
  CONFIG_DIR="${CONFIG_PATH}/${retrieval_method}_add_index.json"
  echo "reindexing ${retrieval_method}..."
  python ../rewrite_configs.py --method "${retrieval_method}" \
                              --data_name "${CORPUS_NAME}" \
                              --stage "indexing" \
                              --output_dir "${CONFIG_DIR}" \
                              --overrides '{"index_exists": true, "new_text_file": "'"${NEW_FILE_ADDR}"'", "page_content_column": "response", "index_add_path":"'"${LOG_PATH}"'"}'
  wait
  echo "Running ${retrieval_method} indexing..."
  python embedding_index_incremental_corpus.py --config_file_path "$CONFIG_DIR" > "$LOG_DIR" 2>&1 &
done

for RETRIEVAL_MODEL_NAME in "${uniq_retrieval_methods[@]}"; do
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

    OUTPUT_FILE_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${QUERY_DATA_NAME}-test-${RETRIEVAL_MODEL_NAME}"
    if [ -z "${OUTPUT_FILE_LIST}" ]; then
      OUTPUT_FILE_LIST="\"${OUTPUT_FILE_PATH}\""
    else
      OUTPUT_FILE_LIST="${OUTPUT_FILE_LIST},\"${OUTPUT_FILE_PATH}\""
    fi
  done
#
#
#
  echo "rewrite config file for ${RETRIEVAL_MODEL_NAME}"
  CONFIG_DIR="${CONFIG_PATH}/${RETRIEVAL_MODEL_NAME}_retrieval.json"
  LOG_DIR="${LOG_PATH}/${RETRIEVAL_MODEL_NAME}_retrieval.log"

  python ../rewrite_configs.py --method "${RETRIEVAL_MODEL_NAME}" \
                          --data_name "nq" \
                          --stage "retrieval" \
                          --output_dir "${CONFIG_DIR}" \
                          --overrides '{ "query_files": ['"${QUERY_FILE_LIST}"'], "output_files": ['"${OUTPUT_FILE_LIST}"'] }'
  wait
  echo "Running retrieval for ${RETRIEVAL_MODEL_NAME} on ${QUERY_DATA_NAME}..."
  python retrieve_methods.py --config_file_path "$CONFIG_DIR" > "$LOG_DIR" 2>&1 &
  wait
done
#
#wait



cd ../rerank_loop
# 遍历数组，检索方法和rerank方法
for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"; do
  for item in "${run_items[@]}"; do
    # 使用空格分割item的值
    IFS=' ' read -r -a tuple <<< "$item"

    # 获取retrieval方法和rerank方法
    RETRIEVAL_MODEL_NAME="${tuple[1]}"
    RERANK_MODEL_NAME="${tuple[2]}"

    # 打印方法
    echo "Retrieval method: $RETRIEVAL_MODEL_NAME"
    echo "Rerank method: $RERANK_MODEL_NAME"
    # if RERANK_MODEL_NAME is None, skip
    if [ "$RERANK_MODEL_NAME" = "None" ]; then
      continue
    fi

    # rerank
    RETRIEVAL_OUTPUT_NAME="${QUERY_DATA_NAME}-test-${RETRIEVAL_MODEL_NAME}"
    RETRIEVAL_OUTPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}/${RETRIEVAL_OUTPUT_NAME}"
    OUTPUT_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}"
    CONFIG_DIR="${CONFIG_PATH}/${QUERY_DATA_NAME}_${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}_rerank.json"
    LOG_DIR="${LOG_PATH}/${QUERY_DATA_NAME}_${RETRIEVAL_MODEL_NAME}_${RERANK_MODEL_NAME}_rerank.log"
    RERANK_OUTPUT_NAME="${QUERY_DATA_NAME}-${RERANK_MODEL_NAME}_rerank_based_on_${RETRIEVAL_MODEL_NAME}"
    python ../rewrite_configs.py --method "${RERANK_MODEL_NAME}" \
                        --data_name "${QUERY_DATA_NAME}" \
                        --stage "rerank" \
                        --output_dir "${CONFIG_DIR}" \
                        --overrides '{"retriever_topk_passages_path": "'"${RETRIEVAL_OUTPUT_PATH}"'", "special_suffix": "'"${RERANK_OUTPUT_NAME}"'", "reranker_output_dir": "'"${OUTPUT_PATH}"'"}'

    wait
    echo "Running rerank for ${RERANK_MODEL_NAME} on ${QUERY_DATA_NAME}..."
    # 运行分布式Python脚本
    torchrun  --nproc_per_node ${WORLD_SIZE} \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr localhost \
        --master_port $PORT \
        rerank_for_loop.py \
        --config "$CONFIG_DIR" > "$LOG_DIR" 2>&1 &
    PORT=$((PORT+1))
    # 等待所有进程结束
    wait

    echo "-----------------------------"
  done
done



cd ../llm_zero_generate

QUESTION_FILE_NAMES=()
for RETRIEVAL_MODEL_NAME in "${uniq_retrieval_methods[@]}"; do
  QUESTION_FILE_NAME="-test-${RETRIEVAL_MODEL_NAME}"
  QUESTION_FILE_NAMES+=("${QUESTION_FILE_NAME}")
done
for item in "${run_items[@]}"; do
    # 使用空格分割item的值
    IFS=' ' read -r -a tuple <<< "$item"

    # 获取retrieval方法和rerank方法
    RETRIEVAL_MODEL_NAME="${tuple[1]}"
    RERANK_MODEL_NAME="${tuple[2]}"
    if [ "$RERANK_MODEL_NAME" = "None" ]; then
      continue
    fi
    QUESTION_FILE_NAME="-${RERANK_MODEL_NAME}_rerank_based_on_${RETRIEVAL_MODEL_NAME}.json"
    QUESTION_FILE_NAMES+=("${QUESTION_FILE_NAME}")
done
echo "QUESTION_FILE_NAMES:"
for QUESTION_FILE_NAME in "${QUESTION_FILE_NAMES[@]}"; do
  echo "$QUESTION_FILE_NAME"
done
#
#
##QUESTION_FILE_NAMES=(
###  "-test-bge-base"
###  "-test-bm25"
###  "-test-contriever"
###  "-test-llm-embedder"
###  "-bge_rerank_based_on_bm25.json"
##  "-bge_rerank_based_on_bge-base.json"
###  "-monot5_rerank_based_on_bge-base.json"
###  "-monot5_rerank_based_on_bm25.json"
###  "-upr_rerank_based_on_bge-base.json"
###  "-upr_rerank_based_on_bm25.json"
##)
LOOP_CONFIG_PATH_NAME=$CONFIG_PATH

TOTAL_LOG_DIR=$LOG_PATH
TOTAL_OUTPUT_DIR=$OUTPUT_DIR
mkdir -p "${TOTAL_LOG_DIR}"
mkdir -p "${TOTAL_OUTPUT_DIR}"

for MODEL_NAME in "${GENARATE_MODEL_NAMES[@]}"
do
  for QUERY_DATA_NAME in "${QUERY_DATA_NAMES[@]}"
  do
    for QUESTION_FILE_NAME in "${QUESTION_FILE_NAMES[@]}"
    do
      OUTPUT_DIR="${TOTAL_OUTPUT_DIR}/${QUERY_DATA_NAME}"
      QUESTION_FILE_PATH="${OUTPUT_DIR}/${QUERY_DATA_NAME}${QUESTION_FILE_NAME}"
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
#      python get_response_llm.py --config_file_path "${CONFIG_PATH}" > "${LOG_DIR}" 2>&1 &
      wait
  done
  done
done
