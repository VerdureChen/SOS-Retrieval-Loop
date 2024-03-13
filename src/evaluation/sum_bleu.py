#summarze percentage tsvs

import os
import sys
import json
import csv


path_names = [
    "nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20231227041949",
    "nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240113075935",
    "nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20231229042900",
    "nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240116050642",
    "nq_webq_pop_tqa_loop_output_bm25_upr_total_loop_10_20231231125441",
    "nq_webq_pop_tqa_loop_output_bm25_monot5_total_loop_10_20240101125941",
    "nq_webq_pop_tqa_loop_output_bm25_bge_total_loop_10_20240103144945",
    "nq_webq_pop_tqa_loop_output_bge-base_upr_total_loop_10_20240106093905",
    "nq_webq_pop_tqa_loop_output_bge-base_monot5_total_loop_10_20240108014726",
    "nq_webq_pop_tqa_loop_output_bge-base_bge_total_loop_10_20240109090024"
]

dataset_names = [
    "nq",
    "webq",
    "pop",
    "tqa"
]

topk =[
    5,
]

ref = 5

# 要合并的文件类型
file_types = ["bleu_top5_ref5_bigram.tsv", "bleu_top5_ref5_trigram.tsv"]

# 输出文件
output_file = "summary_bleu.tsv"

# 创建或清空输出文件
f = open(output_file, "w")
f.close()

# 准备写入数据
with open(output_file, 'a') as outfile:

    total_path = "/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/data_v2/loop_output/DPR"
    # 遍历每个文件类型（topk_ref）
    for file_type in file_types:
        # 写入topk和ref作为新的部分的第一行
        topk_ref = file_type.replace("bleu_", "").replace(".tsv", "")
        outfile.write(topk_ref + "\n")
        # write a dataset name row, each dataset name has 12 blank columns after it
        for dataset_name in dataset_names:
            outfile.write(dataset_name + "\t" * 13)
        outfile.write("\n")

        # 写入表头
        headers = ['ref_num', "method", "generate_model_name"] + [str(i) for i in range(1, 11)]
        headers_with_space = "\t".join(headers) + "\t\t"
        outfile.write(headers_with_space * len(dataset_names) + "\n")




        # 遍历每个数据集
        # 初始化每个文件类型的数据行
        data_rows = [[] for _ in dataset_names]

        # 遍历每个文件夹进行数据集的读取
        for path in path_names:
            # 遍历每个数据集
            for dataset_index, dataset in enumerate(dataset_names):
                # 构建文件路径
                file_path = os.path.join(total_path, path, dataset, "results", file_type)
                # 检查文件是否存在
                if os.path.exists(file_path):
                    # 读取文件内容
                    with open(file_path, 'r') as infile:
                        # 跳过表头
                        next(infile)
                        reader = csv.reader(infile, delimiter='\t')
                        rows = list(reader)
                        if len(rows) > 3:
                            for row in rows[3:]:
                                # 跳过空行
                                if len(row) == 0:
                                    continue
                                data_rows[dataset_index].append("\t".join(row))
                        else:
                            for row in rows:
                                # 跳过空行
                                if len(row) == 0:
                                    continue
                                data_rows[dataset_index].append("\t".join(row))
                else:
                    # 如果文件不存在，添加空行
                    print("file not exist: ", file_path)
        # 将每个数据集的对应行作为一行写入文件，中间用\t\t
        for data_row in range(len(data_rows[0])):
            for dataset_index, dataset in enumerate(dataset_names):
                outfile.write(data_rows[dataset_index][data_row] + "\t\t")
            outfile.write("\n")
