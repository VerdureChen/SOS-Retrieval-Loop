import os
import pandas as pd
from collections import defaultdict


# Your provided path names and dataset names
# path_names = [
#     "mis_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240129064151",
#     "mis_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240124142811",
#     "mis_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240125140045",
#     "mis_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240123121401"
# ]

# path_names = [
#     "filter_bleu_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240130134307",
#     "filter_bleu_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240131140843",
#     "filter_bleu_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240131141029",
#     "filter_bleu_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240131141119"
# ]

path_names = [
    "filter_source_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240203141208",
    "filter_source_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240204104046",
    "filter_source_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240204091108",
    "filter_source_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240204103944"
]

dataset_names = ["nq", "webq", "pop", "tqa"]
base_dir = "/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/data_v2/loop_output/DPR"

# 存储最终结果
final_results = defaultdict(dict)

# 遍历每个数据集

for dataset in dataset_names:
    for path_name in path_names:
        file_path = os.path.join(base_dir, path_name, dataset, "results", f"{dataset}_QA.tsv")
        print(f"file_path: {file_path}")
        # 读取并解析文件
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            methods_scores = defaultdict(dict)
            ref_num = None
            model_name = None
            loop_order = []
            method_name = None

            for line in lines:
                if line.startswith("Ref Num:"):
                    ref_num = line.strip().split(": ")[1]
                elif line.startswith("Generate Model:"):
                    model_name = line.strip().split(": ")[1]
                elif line.startswith(dataset):
                    continue
                elif line == "\n":
                    continue
                else:
                    print(f"model_name: {model_name}")
                    print(f"line: {line}")
                    # llm-embedder	0.635	0.58	0.64	0.62	0.615	0.61	0.605	0.605	0.585	0.59
                    method_name = line.strip().split("\t")[0]
                    method_scores = line.strip().split("\t")[1:]
                    # print(method_name, method_scores)
                    # change the 2nd item in method_scores to last
                    index_changed = method_scores.pop(1)
                    method_scores.append(index_changed)
                    print(f"method_name: {method_name}, method_scores: {method_scores}\n\n\n")
                    methods_scores[model_name][method_name] = method_scores

            # 保存结果
            # print(f"ref_num: {ref_num},  methods_scores: {methods_scores}")
            for model_name in methods_scores.keys():
                for method_name in methods_scores[model_name].keys():
                    if dataset not in final_results.keys():
                        final_results[dataset] = defaultdict(dict)
                    if model_name not in final_results[dataset].keys():
                        final_results[dataset][model_name] = defaultdict(dict)
                    final_results[dataset][model_name][method_name] = methods_scores[model_name][method_name].copy()

# 保存结果
for dataset in final_results.keys():
    print(f"dataset: {dataset}")
    for model_name in final_results[dataset].keys():
        print(f"model_name: {model_name}")
        for method_name in final_results[dataset][model_name].keys():
            print(f"method_name: {method_name}, scores: {final_results[dataset][model_name][method_name]}")
        print("\n\n")


output_path = os.path.join("sum_tsvs", "sum_QA_filter_source.tsv")
'''

output_format:
EM	ref=5				
GPT					
nq					
BM25	BM25+UPR	BM25+MonoT5	BM25+BGE_reranker	BGE-Base
loop1	0.53	0.55	0.53	0.565	0.56
loop2	0.55	0.53	0.535	0.54	0.565
loop3	0.55	0.515	0.53	0.555	0.57
loop4	0.525	0.515	0.53	0.55	0.545
loop5	0.515	0.52	0.535	0.545	0.54
loop6	0.515	0.515	0.515	0.535	0.55

'''





with open(output_path, 'w') as file:
    file.write("EM\tref=5\n")
    for dataset in dataset_names:
        for model_name in final_results[dataset].keys():
            file.write(f"{model_name}\n")
            file.write(f'{dataset}\n')
            file.write(f"Method\t")
            for method_name in final_results[dataset][model_name].keys():
                file.write(f"{method_name}\t")
            file.write("\n")
            for loop_order in range(1, 11):
                file.write(f"loop{loop_order}\t")
                for method_name in final_results[dataset][model_name].keys():
                    file.write(f"{final_results[dataset][model_name][method_name][loop_order - 1]}\t")
                file.write("\n")
            file.write("\n")
        file.write("\n")




