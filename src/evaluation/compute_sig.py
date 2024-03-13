import json
import argparse
from scipy.stats import ttest_ind
import os

def calculate_acc_at_5(data):
    acc_scores = []
    for query_id, query_data in data.items():
        top_5_contexts = query_data['contexts'][:20]  # 取前5个结果
        acc_scores.append(1 if any(ctx['has_answer'] for ctx in top_5_contexts) else 0)  # 检查是否有正确答案
    return acc_scores


def compare_significance(acc1_scores, acc2_scores):
    # 使用t-test比较两组数据的平均值
    t_stat, p_value = ttest_ind(acc1_scores, acc2_scores)
    return t_stat, p_value


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def print_results(file1_mean, file2_mean, t_stat, p_value):
    print(f"File 1 ACC@5 Average: {file1_mean}")
    print(f"File 2 ACC@5 Average: {file2_mean}")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        print("There is a significant difference between the two files at the 5% significance level.")
    else:
        print("There is no significant difference between the two files at the 5% significance level.")

    if file1_mean > file2_mean:
        print("File 1 has a higher ACC@5 average value.")
    elif file1_mean < file2_mean:
        print("File 2 has a higher ACC@5 average value.")
    else:
        print("Both files have the same ACC@5 average value.")


def main(file1_path, file2_path):
    # 读取文件
    file1_data = read_json_file(file1_path)
    file2_data = read_json_file(file2_path)

    # 计算acc@5
    acc1_scores = calculate_acc_at_5(file1_data)
    acc2_scores = calculate_acc_at_5(file2_data)

    # 计算平均值
    file1_mean = sum(acc1_scores) / len(acc1_scores)
    file2_mean = sum(acc2_scores) / len(acc2_scores)

    # 比较两个文件
    t_stat, p_value = compare_significance(acc1_scores, acc2_scores)

    # 打印结果
    print_results(file1_mean, file2_mean, t_stat, p_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare ACC@5 significance between two JSON files and determine which has the higher average.')
    parser.add_argument('ref', type=str, help='Path to the first JSON file')
    parser.add_argument('run', type=str, help='Path to the second JSON file')

    args = parser.parse_args()

    retrieve_model_names = [
        'bm25',
        'contriever',
        'bge-base',
        'llm-embedder',
    ]

    rerank_model_names = [
        'upr',
        'monot5',
        'bge'
    ]
    data_names = [
        'nq',
        'webq',
        'pop',
        'tqa'
    ]

    # for all files in two folders
    for data_name in data_names:
        for retrieve_model_name in retrieve_model_names:
            ref_path = os.path.join(args.ref, f'{data_name}/{data_name}-test-{retrieve_model_name}')
            run_path = os.path.join(args.run, f'{data_name}/{data_name}-test-{retrieve_model_name}')
            print(f'Comparing {ref_path} and {run_path}')
            main(ref_path, run_path)
            for rerank_model_name in rerank_model_names:
                if retrieve_model_name not in ['llm-embedder','contriever']:
                    ref_path = os.path.join(args.ref, f'{data_name}/{data_name}-{rerank_model_name}_rerank_based_on_{retrieve_model_name}.json')
                    run_path = os.path.join(args.run, f'{data_name}/{data_name}-{rerank_model_name}_rerank_based_on_{retrieve_model_name}.json')
                    print(f'Comparing {ref_path} and {run_path}')
                    main(ref_path, run_path)



