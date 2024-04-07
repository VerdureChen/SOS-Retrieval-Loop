#!/usr/bin/env bash

datasets=(nq pop webq tqa)
models=(qwen chatglm llama baichuan)
# first generate false answer using gpt
# shellcheck disable=SC2068
for dataset in ${datasets[@]}
do

        python gen_misinfo_llm.py --config_file_path  mis_config/${dataset}_mis_config_answer_gpt.json

done

wait

#generate misinformation passages using models
# shellcheck disable=SC2068
for dataset in ${datasets[@]}
do
    # shellcheck disable=SC2068
    # shellcheck disable=SC2034
    for model in ${models[@]}
    do
        python gen_misinfo_llm.py --config_file_path  mis_config/${dataset}_mis_config_passage_${model}.json > logs/${dataset}_misinfo_passage_${model}.log 2>&1 &
    done
    wait
done