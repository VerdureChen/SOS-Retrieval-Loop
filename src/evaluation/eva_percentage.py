import json
import re
from collections import defaultdict

# 使用正则表达式来匹配LLM名字
def extract_llm_name(docid):
    match = re.match(r'([a-zA-Z0-9.-]+)_', str(docid))
    return match.group(1) if match else None

def calculate_llm_human_proportion(retrieval_file, topks, llm_names_preset=None):
    with open(retrieval_file, 'r') as file:
        data = json.load(file)

    # 初始化结果字典，包括每个LLM和人类的计数
    results = defaultdict(lambda: defaultdict(int))
    if llm_names_preset:
        for topk in topks:
            for name in llm_names_preset:
                results[topk][name] = 0
            results[topk]['human'] = 0

    # 遍历每个查询
    for query_id, query_data in data.items():
        contexts = query_data['contexts']
        for topk in topks:
            # 获取当前topk的文档
            topk_contexts = contexts[:topk]
            llm_names = defaultdict(int)

            # 计算LLM和人类生成文本的数量
            for ctx in topk_contexts:
                llm_name = extract_llm_name(ctx['docid'])
                if llm_name:
                    llm_names[llm_name] += 1
                else:
                    # 假设数字id表示人类生成的文本
                    llm_names['human'] += 1

            # 更新结果
            for name, count in llm_names.items():
                results[topk][name] += count

    # 计算比例
    proportions = defaultdict(dict)
    # merge results[topk]['chatglm3-6b'] and results[topk]['chatglm3-6b-chat'] into results[topk]['chatglm3-6b']
    for topk in topks:
        if 'chatglm3-6b-chat' in results[topk]:
            results[topk]['chatglm3-6b'] += results[topk]['chatglm3-6b-chat']
            del results[topk]['chatglm3-6b-chat']

    for topk in topks:
        total = sum(results[topk].values())
        for name in results[topk]:
            proportions[topk][f'{name}_proportion'] = results[topk][name] / total
        print(f'Proportions for top-{topk}: {proportions[topk]}')
        print(f'sum: {sum(proportions[topk].values())}')
    # return dict like {5: {'human': 100, 'human_proportion': 0.5, 'gpt-3.5-turbo': 100, 'gpt-3.5-turbo_proportion': 0.5}}

    return proportions

# 使用示例
# retrieval_file = 'path_to_your_file.json'  # 替换为实际文件路径
# topks = [5, 20, 50]
# proportions = calculate_llm_human_proportion(retrieval_file, topks)
# print(proportions)
