# evaluate the performance of the pipeline
# 1. retrieval accuracy
# 2. QA EM
# 3. AIGC avg rank
# 4. AIGC quality

import os
import sys
import json
import csv
import argparse
import numpy as np
import re
from tqdm import tqdm
import datasets
from collections import OrderedDict
from eva_retrieval_acc import evaluate_retrieval
from eva_generate_em import evaluate as evaluate_em
from eva_rank import compute_true_wrong_first_answer_rank, compute_change_rank

generate_model_names = [
    'gpt-3.5-turbo',
    'baichuan2-13b-chat',
    'qwen-14b-chat',
    'chatglm3-6b',
    'llama2-13b-chat'
]

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

retrieval_order_dict = OrderedDict()
for retrieve_model_name in retrieve_model_names:
    retrieval_order_dict[retrieve_model_name] = {}
for retrieve_model_name in retrieve_model_names:
    if retrieve_model_name == 'bm25' or retrieve_model_name == 'bge-base':
        for rerank_model_name in rerank_model_names:
            retrieval_order_dict[f'{retrieve_model_name}+{rerank_model_name}'] = {}

# print(retrieval_order_dict)

QA_order_dict = OrderedDict()
for generate_model_name in generate_model_names:
    QA_order_dict[generate_model_name] = OrderedDict()
    for key in retrieval_order_dict.keys():
        QA_order_dict[generate_model_name][key] = {}

rank_order_dict = OrderedDict()
for generate_model_name in generate_model_names:
    rank_order_dict[generate_model_name] = OrderedDict()
    for key in retrieval_order_dict.keys():
        rank_order_dict[generate_model_name][key] = {}

def evaluate_retrieval_acc(retrieval_file, topk=None, regex=False):
    if topk is None:
        topk = [5, 20, 100]
    print('evaluate retrieval accuracy...')
    retrieval_acc = evaluate_retrieval(retrieval_file, topk, regex)
    return retrieval_acc


def evaluate_qa_em(qa_file):
    dataset = datasets.load_dataset("json", data_files=qa_file)["train"]
    print('evaluate QA EM...')
    prediction = evaluate_em(dataset)
    EM = sum(prediction['exact_match']) / len(prediction['exact_match'])
    return EM


def extract_method_name(file_name, directory_name):
    # 动态提取方法名
    generate_model_name = None
    retrieval_method_name = None
    rerank_model_name = None
    ref_num = None
    loop_num = None
    if directory_name in file_name and not file_name.endswith('.trec'):
        for model_name in generate_model_names:
            if model_name in file_name:
                generate_model_name = model_name
                # remove the model name from file name
                file_name = file_name.replace(model_name, '')
        for model_name in retrieve_model_names:
            if model_name in file_name:
                retrieval_method_name = model_name
                # remove the model name from file name
                file_name = file_name.replace(model_name, '')
        for model_name in rerank_model_names:
            if model_name in file_name:
                rerank_model_name = model_name
                # remove the model name from file name
                file_name = file_name.replace(model_name, '')

        if 'ref' in file_name or 'ref_num' in file_name:
            ref_num = file_name.split('_')[-1].split('.')[0]
            # remove the ref_num from file name
            file_name = file_name.replace(f'_ref_num_{ref_num}', '')
            file_name = file_name.replace(f'_ref_{ref_num}', '')
        else:
            ref_num = '5'

        if 'loop' in file_name:
            # get loop_num n from 'loop_n' by regex
            loop_num = re.findall(r'loop_\d+', file_name)[0].split('_')[-1]
            # remove the loop_num from file name
            file_name = file_name.replace(f'_loop_{loop_num}', '')
        else:
            loop_num = '0'
        return [generate_model_name, retrieval_method_name, rerank_model_name, ref_num, loop_num]
    return None


def format_tsv_output(directory_name, results, task, output_file_name=None):

    with open(output_file_name, 'w') as f:
        if task == 'retrieval':
            # 按照ref_num划分区域
            print(results)
            # remove the empty method
            for method in list(results.keys()):
                if results[method] == {}:
                    results.pop(method)
            for ref_num in sorted(next(iter(next(iter(results.values())).values())).keys()):
                f.write(f"Ref Num: {ref_num}\n")
                # 按照loop_num划分表格
                for loop_num in sorted(results[next(iter(results.keys()))].keys()):
                    f.write(f"Loop Num: {loop_num}\n")
                    f.write(f"{directory_name}\tacc@5\tacc@10\tacc@100\n")
                    # 格式化输出
                    for method in results.keys():
                        try:
                            scores = results[method][loop_num][ref_num]
                            score_5 = scores.get('5', '-')
                            score_10 = scores.get('20', '-')  # 假设你有acc@10的数据
                            score_100 = scores.get('100', '-')
                            f.write(f"{method}\t{score_5}\t{score_10}\t{score_100}\n")
                        except KeyError:
                            f.write(f"{method}\t-\t-\t-\n")
                    f.write('\n')  # 打印空行以分隔不同的表格
        elif task == 'QA':
            # 按照ref_num划分区域
            for model in list(results.keys()):
                for method in list(results[model].keys()):
                    if results[model][method] == {}:
                        results[model].pop(method)
            for ref_num in sorted(next(iter(next(iter(next(iter(results.values())).values())).values())).keys()):
                f.write(f"Ref Num: {ref_num}\n")
                # 按照loop_num划分表格
                #for loop_num in sorted(results[next(iter(results.keys()))][next(iter(next(iter(results.values())).keys()))].keys()):
                #按照generate_model_name划分表格
                for generate_model_name in results.keys():
                    f.write(f"Generate Model: {generate_model_name}\n")
                    f.write(f"{directory_name}-EM\t")
                    # 按照loop_num划分每一列
                    for loop_num in sorted(results[generate_model_name][next(iter(results[generate_model_name].keys()))].keys()):
                        f.write(f"{loop_num}\t")
                    f.write('\n')
                    # 格式化输出
                    for method in results[generate_model_name].keys():
                        f.write(f"{method}\t")
                        for loop_num in sorted(results[generate_model_name][method].keys()):
                            try:
                                EM = results[generate_model_name][method][loop_num][ref_num]
                            except KeyError:
                                EM = '-'
                            f.write(f"{EM}\t")
                        f.write('\n')
                    f.write('\n')  # 打印空行以分隔不同的表格
        elif task == 'rank':
            tsv_writer = csv.writer(f, delimiter='\t')
            for generate_model_name, methods in rank_order_dict.items():
                for retrieval_method_name, loops in methods.items():
                    if loops == {}:
                        continue
                    for ref_num in loops['1']:  # 假设每个loop_num都有相同的ref_num
                        # 为每个标签和ref_num创建表格
                        table_0 = []
                        headers = ["Ref Num", "Loop Num", "Avg Answer Rank", "Avg Human Answer Rank", "Accuracy@5",
                                   "Accuracy@3", "Accuracy@1"]
                        for label in ['0', '1']:
                            # 写入表头
                            tsv_writer.writerow([f"Model: {generate_model_name} | Method: {retrieval_method_name} | Ref Num: {ref_num} | Label: {label}"])
                            tsv_writer.writerow(headers)
                            # 写入表格
                            # 遍历循环
                            for loop_num in sorted(int(loop_num) for loop_num in loops.keys()):
                                data = loops[str(loop_num)][ref_num]
                                # 对于每个标签，添加行到对应的表格列表
                                table_0.append([ref_num, loop_num, data['avg_answer_rank'][int(label)],
                                    data['avg_human_answer_rank'][int(label)],
                                    data['accuracy_at_5'][int(label)], data['accuracy_at_3'][int(label)],
                                    data['accuracy_at_1'][int(label)]])
                            # 写入表格数据
                            tsv_writer.writerows(table_0)
                            tsv_writer.writerow([])  # 添加空行作为分隔
                            table_0 = []
                        # 写入change rank
                        tsv_writer.writerow([f"Model: {generate_model_name} | Method: {retrieval_method_name} | Ref Num: {ref_num} | Change Rank"])
                        tsv_writer.writerow(["Ref Num", "Loop Num", "0->1", "1->0", "0->0", "1->1"])
                        for loop_num in sorted(int(loop_num) for loop_num in loops.keys()):
                            data = loops[str(loop_num)][ref_num]
                            tsv_writer.writerow([ref_num, str(loop_num), data['change_em']['0->1'], data['change_em']['1->0'], data['change_em']['0->0'], data['change_em']['1->1']])
                        tsv_writer.writerow([])
                        # 写入change index
                        tsv_writer.writerow([f"Model: {generate_model_name} | Method: {retrieval_method_name} | Ref Num: {ref_num} | Change Index"])
                        tsv_writer.writerow(["Ref Num", "Loop Num", "0->1", "1->0"])
                        for loop_num in sorted(int(loop_num) for loop_num in loops.keys()):
                            data = loops[str(loop_num)][ref_num]
                            tsv_writer.writerow([ref_num, str(loop_num)] + [','.join([str(i) for i in data['change_index']['0->1']]), ','.join([str(i) for i in data['change_index']['1->0']])])
                        tsv_writer.writerow([])


        else:
            raise ValueError(f"Unknown task: {task}")




def compute_res(directory, task):
    directory_name = os.path.basename(directory)
    output_file_dir = os.path.join(directory, 'results')
    if not os.path.exists(output_file_dir):
        os.mkdir(output_file_dir)
    output_file_name = os.path.join(output_file_dir, f'{directory_name}_{task}.tsv')
    if task == 'rank':
        retrieval_file_dict = {}
        QA_file_dict = {}

    # 遍历给定目录
    for file_name in tqdm(os.listdir(directory)):
        method_name = extract_method_name(file_name, directory_name)
        if method_name:
            if method_name[0] is None and task == 'retrieval':
                # retrieval file
                generate_model_name, retrieval_method_name, rerank_model_name, ref_num, loop_num = method_name
                file_path = os.path.join(directory, file_name)
                scores = evaluate_retrieval_acc(file_path)
                if rerank_model_name is not None:
                    method = f'{retrieval_method_name}+{rerank_model_name}'
                else:
                    method = retrieval_method_name
                if method not in retrieval_order_dict.keys():
                    continue
                if retrieval_order_dict[method].get(loop_num) is None:
                    retrieval_order_dict[method][loop_num] = {}
                retrieval_order_dict[method][loop_num][ref_num] = scores

            elif method_name[0] is not None and task == 'QA':
                # QA file
                generate_model_name, retrieval_method_name, rerank_model_name, ref_num, loop_num = method_name
                file_path = os.path.join(directory, file_name)
                EM = evaluate_qa_em(file_path)
                if rerank_model_name is not None:
                    method = f'{retrieval_method_name}+{rerank_model_name}'
                else:
                    method = retrieval_method_name
                if method not in QA_order_dict[generate_model_name].keys():
                    continue
                if QA_order_dict[generate_model_name][method].get(loop_num) is None:
                    QA_order_dict[generate_model_name][method][loop_num] = {}
                QA_order_dict[generate_model_name][method][loop_num][ref_num] = EM
            elif task == 'rank':
                # rank file
                generate_model_name, retrieval_method_name, rerank_model_name, ref_num, loop_num = method_name
                file_path = os.path.join(directory, file_name)
                if rerank_model_name is not None:
                    method = f'{retrieval_method_name}+{rerank_model_name}'
                else:
                    method = retrieval_method_name

                if generate_model_name is None:
                    retrieval_file_dict[f'{method}_{loop_num}_{ref_num}'] = file_path
                else:
                    QA_file_dict[f'{generate_model_name}_{method}_{loop_num}_{ref_num}'] = file_path

        else:
            print(f"Unknown file: {file_name}")

    if task == 'rank':
        # pair retrieval and QA file name
        print('pair retrieval and QA file name...')
        for retrieval_file_name in retrieval_file_dict.keys():
            for QA_file_name in QA_file_dict.keys():
                retrieval_method_name, loop_num, ref_num = retrieval_file_name.split('_')
                generate_model_name, retrieval_method_name2, loop_num2, ref_num2 = QA_file_name.split('_')
                if retrieval_method_name == retrieval_method_name2 and loop_num == loop_num2 and ref_num == ref_num2:
                    if rank_order_dict[generate_model_name][retrieval_method_name].get(loop_num) is None:
                        rank_order_dict[generate_model_name][retrieval_method_name][loop_num] = {}
                    rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num] = {}
                    rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['retrieval_file'] = retrieval_file_dict[retrieval_file_name]
                    rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['QA_file'] = QA_file_dict[QA_file_name]
        # compute rank
        print('compute rank...')
        for generate_model_name in tqdm(rank_order_dict.keys()):
            for retrieval_method_name in rank_order_dict[generate_model_name].keys():
                for loop_num in rank_order_dict[generate_model_name][retrieval_method_name].keys():
                    for ref_num in rank_order_dict[generate_model_name][retrieval_method_name][loop_num].keys():
                        retrieval_file = rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['retrieval_file']
                        QA_file = rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['QA_file']
                        avg_answer_rank, avg_human_answer_rank, accuracy_at_5, accuracy_at_3, accuracy_at_1, _ = compute_true_wrong_first_answer_rank(retrieval_file, QA_file)
                        rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['avg_answer_rank'] = avg_answer_rank
                        rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['avg_human_answer_rank'] = avg_human_answer_rank
                        rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['accuracy_at_5'] = accuracy_at_5
                        rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['accuracy_at_3'] = accuracy_at_3
                        rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['accuracy_at_1'] = accuracy_at_1
                        # compute change rank between first loop and other loops
                        if loop_num == '1':
                            rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['change_em'] = {
                                '0->1': 0,
                                '1->0': 0,
                                '0->0': 0,
                                '1->1': 0,
                            }
                            rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['change_index'] = {
                                '0->1': [],
                                '1->0': [],
                            }
                        else:
                            retrieval_files = [
                                rank_order_dict[generate_model_name][retrieval_method_name]['1'][ref_num]['retrieval_file'],
                                rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['retrieval_file']
                            ]
                            QA_files = [
                                rank_order_dict[generate_model_name][retrieval_method_name]['1'][ref_num]['QA_file'],
                                rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['QA_file']
                            ]
                            change_em, change_index = compute_change_rank(retrieval_files, QA_files)
                            rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['change_em'] = change_em
                            rank_order_dict[generate_model_name][retrieval_method_name][loop_num][ref_num]['change_index'] = change_index



    # print(retrieval_order_dict)
    # print(QA_order_dict)
    if task == 'retrieval':
        format_tsv_output(directory_name, retrieval_order_dict, task, output_file_name)
    elif task == 'QA':
        format_tsv_output(directory_name, QA_order_dict, task, output_file_name)
    elif task == 'rank':
        format_tsv_output(directory_name, rank_order_dict, task, output_file_name)
    else:
        raise ValueError(f"Unknown task: {task}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='config.json')
    parser.add_argument('--directory', type=str, default=None)
    parser.add_argument('--task', type=str, default='retrieval')
    args = parser.parse_args()
    config = read_config_from_json(args.config_file_path)
    # json likes
    # {
    #     "directory": "data",
    #     "task": "retrieval"
    # }

    # 使用配置文件中的参数覆盖命令行参数
    args = override_args_by_config(args, config)

    print(f'args: {args}')

    return args


# 函数用于读取 JSON 配置文件
def read_config_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            args_dict = json.load(json_file)
        return args_dict
    except FileNotFoundError:
        print(f"Configuration file {json_file_path} not found.")
        return {}


# 函数用于覆盖命令行参数
def override_args_by_config(args, config):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


if __name__ == '__main__':
    args = get_args()
    compute_res(args.directory, args.task)



