#summarze percentage tsvs

import os
import sys
import json
import csv


# path_names = [
#     "nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20231227041949",
#     "nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240113075935",
#     "nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20231229042900",
#     "nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240116050642",
#     "nq_webq_pop_tqa_loop_output_bm25_upr_total_loop_10_20231231125441",
#     "nq_webq_pop_tqa_loop_output_bm25_monot5_total_loop_10_20240101125941",
#     "nq_webq_pop_tqa_loop_output_bm25_bge_total_loop_10_20240103144945",
#     "nq_webq_pop_tqa_loop_output_bge-base_upr_total_loop_10_20240106093905",
#     "nq_webq_pop_tqa_loop_output_bge-base_monot5_total_loop_10_20240108014726",
#     "nq_webq_pop_tqa_loop_output_bge-base_bge_total_loop_10_20240109090024"
# ]

# path_names = [
#     "mis_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240129064151",
#     "mis_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240124142811",
#     "mis_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240125140045",
#     "mis_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240123121401"
# ]

path_names = [
    "filter_bleu_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240130134307",
    "filter_bleu_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240131140843",
    "filter_bleu_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240131141029",
    "filter_bleu_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240131141119"
]
# path_names = [
#     "filter_source_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240203141208",
#     "filter_source_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240204104046",
#     "filter_source_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240204091108",
#     "filter_source_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240204103944"
# ]

dataset_names = [
    "nq",
    "webq",
    "pop",
    "tqa"
]
total_path = "/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/data_v2/loop_output/DPR"
# 指定要合并的文件类型
file_types = ["context_answer_ref5_cutoff5_em0.tsv", "context_answer_ref5_cutoff5_em1.tsv"]

# 创建目标文件的表头
header = ["ref_num", "model", "method", "right_num", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# 处理每个文件类型
for file_type in file_types:
    # 初始化数据集字典，用于存储每个数据集的数据
    data_to_write = {dataset_name: [] for dataset_name in dataset_names}

    # 遍历每个路径和数据集名称，收集数据
    for path_name in path_names:
        for dataset_name in dataset_names:
            # 构建文件的完整路径
            file_path = os.path.join(total_path, path_name, dataset_name, "results", file_type)
            if os.path.exists(file_path):
                with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
                    reader = csv.reader(tsvfile, delimiter='\t')
                    for row in reader:
                        # 跳过空行和不正确的表头
                        if not row or row[:4] == header[:4]:
                            continue
                        # 转换行为字典
                        row_dict = dict(zip(header, row))
                        data_to_write[dataset_name].append(row_dict)

    # 写入合并后的文件
    output_file_name = f'sum_tsvs/filter_bleu_merged_{file_type}'
    with open(output_file_name, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t')

        # 写入新的表头
        extended_header = [[f'{dataset_name}_ref5_cutoff5'] + header for dataset_name in dataset_names]
        extended_header = [item for sublist in extended_header for item in sublist]
        writer.writerow(extended_header)

        # 写入合并后的数据
        max_rows = max(len(data_to_write[dataset_name]) for dataset_name in dataset_names)
        for i in range(max_rows):
            row_to_write = []
            for dataset_name in dataset_names:
                data_list = data_to_write[dataset_name]
                if i < len(data_list):
                    row_to_write.extend(['']+list(data_list[i].values()))
                else:
                    # 如果数据集的行不够，用空字符串填充
                    row_to_write.extend([''] * len(header))
            writer.writerow(row_to_write)
