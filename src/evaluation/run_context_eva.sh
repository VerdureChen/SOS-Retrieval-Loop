#!/usr/bin/env bash

QUERY_DATA_NAMES=(nq webq pop tqa)
#RESULT_NAMES=(
#    "nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20231227041949"
#    "nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240113075935"
#    "nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20231229042900"
#    "nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240116050642"
#    "nq_webq_pop_tqa_loop_output_bm25_upr_total_loop_10_20231231125441"
#    "nq_webq_pop_tqa_loop_output_bm25_monot5_total_loop_10_20240101125941"
#    "nq_webq_pop_tqa_loop_output_bm25_bge_total_loop_10_20240103144945"
#    "nq_webq_pop_tqa_loop_output_bge-base_upr_total_loop_10_20240106093905"
#    "nq_webq_pop_tqa_loop_output_bge-base_monot5_total_loop_10_20240108014726"
#    "nq_webq_pop_tqa_loop_output_bge-base_bge_total_loop_10_20240109090024"
#)
# shellcheck disable=SC2054
#RESULT_NAMES=(
#    "mis_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240124142811"
#    "mis_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240129064151"
#    "mis_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240125140045"
#    "mis_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240123121401"
#)

#RESULT_NAMES=(
#    "filter_bleu_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240131140843"
#    "filter_bleu_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240130134307"
#    "filter_bleu_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240131141029"
#    "filter_bleu_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240131141119"
#)

RESULT_NAMES=(
    "filter_source_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240204104046"
    "filter_source_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240203141208"
    "filter_source_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240204091108"
    "filter_source_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240204103944"
)

#RESULT_NAMES=( "mis_passage_processed" )
RESULT_DIR="../../data_v2/loop_output/DPR"
#RESULT_DIR="../../data_v2/misinfo_data/DPR"
#TASK="context_answer"
#TASK="retrieval"
#TASK="bleu"
#TASK="QA"
#TASK="misQA"
#TASK="QA_llm_mis"
#TASK="QA_llm_right"
#TASK="filter_bleu_retrieval"
#TASK="filter_bleu_percentage"
#TASK="filter_bleu_context_answer"
#TASK="filter_source_retrieval"
#TASK="filter_source_percentage"
TASK="filter_source_context_answer"
#TASK="percentage"
for ((i=0;i<${#QUERY_DATA_NAMES[@]};i++))
do
  for ((j=0;j<${#RESULT_NAMES[@]};j++))
  do
    QUERY_DATA_NAME=${QUERY_DATA_NAMES[i]}
    RESULT_NAME=${RESULT_NAMES[j]}
    RESULT_PATH="${RESULT_DIR}/${RESULT_NAME}/${QUERY_DATA_NAME}"
    echo "QUERY_DATA_NAME: ${QUERY_DATA_NAME}"
    echo "RESULT_NAME: ${RESULT_NAME}"
    echo "RESULT_PATH: ${RESULT_PATH}"
    python3 eva_pipe.py --config_file_path none --directory ${RESULT_PATH} --task $TASK
  done
done
