import pandas as pd
from scipy.stats import pearsonr
import os
from prettytable import PrettyTable

# 读取TSV文件的函数
def read_tsv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        data = {}
        for i in range(1, len(lines), 4):  # 跳过标题行，每个模型3行
            # print(lines[i])
            # Generate Model: gpt-3.5-turbo
            # print(lines[i].strip().split(':')[1])
            model = lines[i].strip().split(':')[1].strip()
            # 使用列表推导式获取bm25值，注意跳过前两个元素
            bm25_values = [float(x) for x in lines[i + 2].strip().split('\t')[1:]]
            data[model] = bm25_values
    return data


# 计算两组数据的皮尔逊相关系数的函数
def calculate_correlation(data1, data2):
    # print(data1)
    # print(data2)

    correlations = {}
    for model in data1.keys():
        correlation, _ = pearsonr(data1[model], data2[model])
        correlations[model] = correlation
    return correlations


# 主程序
def main():
    data_names = ['nq', 'webq', 'pop', 'tqa']
    method_names = [
        'loop_output/DPR/mis_nq_webq_pop_tqa_loop_output_bm25_None_total_loop_10_20240129064151',
        'loop_output/DPR/mis_nq_webq_pop_tqa_loop_output_contriever_None_total_loop_10_20240124142811',
        'loop_output/DPR/mis_nq_webq_pop_tqa_loop_output_bge-base_None_total_loop_10_20240125140045',
        'loop_output/DPR/mis_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240123121401'
    ]
    file_type = [
        'right',
        'mis'
    ]
    model_avg_dict = {}
    # 假设你的数据已经保存在两个.tsv文件中
    for data_name in data_names:
        for method_name in method_names:
            for type in file_type:
                # ../../data_v2/loop_output/DPR/mis_nq_webq_pop_tqa_loop_output_llm-embedder_None_total_loop_10_20240123121401/nq/results/nq_QA.tsv
                file1_path = os.path.join('../../data_v2', method_name, data_name, 'results', f'{data_name}_QA.tsv')
                file2_path = os.path.join('../../data_v2', method_name, data_name, 'results', f'{data_name}_QA_llm_{type}.tsv')
                # print(file1_path)
                # print(file2_path)
                # 判断文件是否存在
                if not os.path.exists(file1_path) or not os.path.exists(file2_path):
                    print(f"File {file1_path} or {file2_path} does not exist!")
                    continue
                # 读取数据
                data1 = read_tsv(file1_path)
                data2 = read_tsv(file2_path)

                # 计算相关系数
                correlations = calculate_correlation(data1, data2)

                # 输出相关性结果
                for model, corr in correlations.items():
                    print(f"Model {model}: Pearson Correlation Coefficient = {corr:.3f}")
                    if model not in model_avg_dict:
                        model_avg_dict[model] = {}
                    if data_name not in model_avg_dict[model]:
                        model_avg_dict[model][data_name] = {}
                    if type not in model_avg_dict[model][data_name]:
                        model_avg_dict[model][data_name][type] = []
                    model_avg_dict[model][data_name][type].append(corr)

    # print a pretty table like this:
    # type = right
    # +------------------+---------+---------+---------+---------+
    # | Model            | nq      | webq    | pop     | tqa     |
    # +------------------+---------+---------+---------+---------+
    # | gpt-3.5-turbo    | 0.000   | 0.000   | 0.000   | 0.000   |
    # +------------------+---------+---------+---------+---------+
    # | contriever        | 0.000   | 0.000   | 0.000   | 0.000   |
    # +------------------+---------+---------+---------+---------+
    # | bge-base         | 0.000   | 0.000   | 0.000   | 0.000   |
    # +------------------+---------+---------+---------+---------+
    # | llm-embedder     | 0.000   | 0.000   | 0.000   | 0.000   |
    # +------------------+---------+---------+---------+---------+
    # | Average          | 0.000   | 0.000   | 0.000   | 0.000   |
    # +------------------+---------+---------+---------+---------+
    # type = mis

    def print_pretty_table(model_avg_dict, file_type):
        for type in file_type:
            print(f"type = {type}")
            table = PrettyTable()
            table.field_names = ["Model"] + data_names  # 表头

            for model in model_avg_dict.keys():
                row = [model]  # 开始构建行
                for data_name in data_names:
                    # 计算每种数据类型的平均相关性
                    avg_corr = sum(model_avg_dict[model][data_name][type]) / len(model_avg_dict[model][data_name][type])
                    row.append(f"{avg_corr:.3f}")
                table.add_row(row)  # 添加行到表格

            # 添加平均行
            avg_row = ["Average"]
            for data_name in data_names:
                all_corrs = [model_avg_dict[model][data_name][type] for model in model_avg_dict if model not in ["gpt-3.5-turbo"]]
                avg_corr = sum(sum(corrs) for corrs in all_corrs) / sum(len(corrs) for corrs in all_corrs)
                avg_row.append(f"{avg_corr:.3f}")
            table.add_row(avg_row)

            print(table)  # 打印表格

    print_pretty_table(model_avg_dict, file_type)





# 执行主程序
main()

