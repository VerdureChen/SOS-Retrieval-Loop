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

# 输出文件
output_file = "sum_tsvs/summary_change_index.tsv"

# 创建或清空输出文件
f = open(output_file, "w")
f.close()


# 函数以提取特定模型的Change Index
def extract_change_index(data, model_name):
    change_index = {loop: {'0->1': set(), '1->0': set()} for loop in range(1, 11)}
    current_model = None
    loop_num = 0
    for row in data:
        if row == []:
            continue

        if row and row[0].startswith('Model:') and model_name in row[0] and 'Change Index' in row[0]:
            current_model = model_name
        elif current_model and row[0].startswith('Ref Num'):
            continue
        elif current_model and change_index and row[0].startswith('5'):
            print(row)
            loop_num = int(row[1])
            change_index[loop_num]['0->1'] = set(row[2].split(','))
            change_index[loop_num]['1->0'] = set(row[3].split(','))
        elif current_model and not row[0].startswith('5'):
            current_model = None
    return change_index


# 函数以计算增量变化平均数量
def calculate_average_increment(models_change_index):
    increments = {'0->1': [], '1->0': []}

    # 循环计算所有loop的增量变化
    for loop in range(2, 11):
        loop_increments = {'0->1': 0, '1->0': 0}
        for model_change_index in models_change_index:
            prev_0_to_1 = model_change_index[loop - 1]['0->1']
            prev_1_to_0 = model_change_index[loop - 1]['1->0']
            loop_increments['0->1'] += len(model_change_index[loop]['0->1'] - prev_0_to_1)/200*100
            loop_increments['1->0'] += len(model_change_index[loop]['1->0'] - prev_1_to_0)/200*100

        increments['0->1'].append(loop_increments['0->1'] / len(models_change_index))
        increments['1->0'].append(loop_increments['1->0'] / len(models_change_index))

    return increments


# 函数以提取特定模型的排名数据
def extract_rank_data(data, model_name, label):
    ranks = {loop: {'avg_answer_rank': [], 'avg_human_answer_rank': []} for loop in range(1, 11)}
    current_model = None
    loop_num = 0
    for row in data:
        if row == []:
            continue

        if row and row[0].startswith('Model:') and model_name in row[0] and f'Label: {label}' in row[0]:
            current_model = model_name
        elif current_model and row[0].startswith('Ref Num'):
            continue
        elif current_model and ranks and row[0].startswith('5'):
            print(current_model)
            print(row)
            print(label)
            loop_num = int(row[1])
            ranks[loop_num]['avg_answer_rank'].append(float(row[2]))
            ranks[loop_num]['avg_human_answer_rank'].append(float(row[3]))
        elif current_model and not row[0].startswith('5'):
            current_model = None
    return ranks

# 函数以计算排名的平均值
# 函数以计算每个loop中所有模型的平均排名
def calculate_average_ranks(ranks_data):
    averages = {'avg_answer_rank': [], 'avg_human_answer_rank': []}

    # 循环从1到10包括所有模型的每个loop
    for loop in range(1, 11):
        total_answer_rank = 0
        total_human_answer_rank = 0
        models_count = 0

        # 循环遍历所有模型
        for model_data in ranks_data:
            if model_data[loop]['avg_answer_rank'] and model_data[loop]['avg_human_answer_rank']:
                total_answer_rank += sum(model_data[loop]['avg_answer_rank'])
                total_human_answer_rank += sum(model_data[loop]['avg_human_answer_rank'])
                models_count += len(model_data[loop]['avg_answer_rank'])  # 假设每个loop的答案排名数量相同

        # 计算平均值并添加到列表中
        avg_answer_rank = total_answer_rank / models_count if models_count else 0
        avg_human_answer_rank = total_human_answer_rank / models_count if models_count else 0

        averages['avg_answer_rank'].append(avg_answer_rank)
        averages['avg_human_answer_rank'].append(avg_human_answer_rank)

    return averages


total_path = "/home/xiaoyang2020/chenxiaoyang_11/Rob_LLM/data_v2/loop_output/DPR"
# 主处理循环
models = ['gpt-3.5-turbo', 'baichuan2-13b-chat', 'qwen-14b-chat', 'chatglm3-6b', 'llama2-13b-chat']  # 这里替换成实际的模型名称



def sum_change_index(path_names, dataset_names, total_path, models):
    for path_name in path_names:
        for dataset_name in dataset_names:
            file_to_read = os.path.join(total_path, path_name, dataset_name, 'results', f"{dataset_name}_rank.tsv")
            print(os.path.exists(file_to_read))
            if os.path.exists(file_to_read):
                with open(file_to_read, 'r', encoding='utf-8') as file:
                    tsv_reader = csv.reader(file, delimiter='\t')
                    data = list(tsv_reader)

                models_change_index = [extract_change_index(data, model) for model in models]
                print(models_change_index)
                average_increments = calculate_average_increment(models_change_index)

                with open(output_file, 'a') as outfile:
                    outfile.write(f"{path_name}\t{dataset_name}\n")
                    outfile.write("Loop\tAverage 0->1\tAverage 1->0\n")
                    for loop in range(2, 11):  # Loop 1 is the base, so we start from Loop 2
                        outfile.write(
                            f"{loop-1}->{loop}\t{average_increments['0->1'][loop - 2]:.2f}\t{average_increments['1->0'][loop - 2]:.2f}\n")


def sum_avg_rank(path_names, dataset_names, total_path, models):
    # 主处理循环
    for path_name in path_names:
        for dataset_name in dataset_names:
            file_to_read = os.path.join(total_path, path_name, dataset_name, 'results', f"{dataset_name}_rank.tsv")
            if os.path.exists(file_to_read):
                with open(file_to_read, 'r', encoding='utf-8') as file:
                    tsv_reader = csv.reader(file, delimiter='\t')
                    data = list(tsv_reader)

                for label in [0, 1]:  # 分别处理标签0和1
                    output_file = f"summary_rank_label_{label}.tsv"  # 输出文件名根据标签变化
                    models_rank_data = [extract_rank_data(data, model, label) for model in models]
                    print(models_rank_data)
                    average_ranks = calculate_average_ranks(models_rank_data)

                    with open(output_file, 'a') as outfile:
                        outfile.write(f"{path_name}\t{dataset_name}\tLabel {label}\n")
                        outfile.write("Loop\tAverage Answer Rank\tAverage Human Answer Rank\n")
                        for loop in range(1, 11):
                            outfile.write(
                                f"{loop}\t{average_ranks['avg_answer_rank'][loop - 1]:.2f}\t{average_ranks['avg_human_answer_rank'][loop - 1]:.2f}\n")

sum_change_index(path_names, dataset_names, total_path, models)
# sum_avg_rank(path_names, dataset_names, total_path, models)