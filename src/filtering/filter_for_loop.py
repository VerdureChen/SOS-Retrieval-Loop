import pandas as pd
from datasets import Dataset
from transformers import pipeline
import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from fast_bleu import SelfBLEU
from collections import defaultdict
import sys
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# filter the top docs of the retrieved docs by self-bleu score, to keep the self-bleu score of the top 5 docs =< 0.5


sys.path.append('../retrieval_loop')
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
from evaluate_dpr_retrieval import has_answers, SimpleTokenizer, evaluate_retrieval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='config.json')
    parser.add_argument('--input_file', type=str, default='data/llm_text.json')
    parser.add_argument('--output_file', type=str, default='data/llm_text_processed')
    parser.add_argument('--elasticsearch_url', type=str, default='http://localhost:9200')
    parser.add_argument('--index_name', type=str, default='bm25_psgs_index')
    parser.add_argument('--max_self_bleu', type=float, default=0.4)
    parser.add_argument('--num_docs', type=int, default=5)
    parser.add_argument('--task', type=str, default='filter_bleu')
    args = parser.parse_args()
    # 读取 JSON 配置文件
    # json:
    # {
    #     "input_file": "data/llm_text.json",
    #     "output_file": "data/llm_text_processed",
    #     "elasticsearch_url": "http://localhost:9200",
    #     "index_name": "bm25_psgs_index",
    #     "max_self_bleu": 0.4,
    #     "num_docs": 5
    # }

    config = read_config_from_json(args.config_file_path)

    # 使用配置文件中的参数覆盖命令行参数
    args = override_args_by_config(args, config)

    print(f'args: {args}')

    return args


# 函数用于读取 JSON 配置文件
def read_config_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            args_dict = json.load(json_file)
            # 如果配置文件中的参数是字符串类型，需要转换为对应的类型
            if isinstance(args_dict['num_docs'], str):
                args_dict['num_docs'] = int(args_dict['num_docs'])
            if isinstance(args_dict['max_self_bleu'], str):
                args_dict['max_self_bleu'] = float(args_dict['max_self_bleu'])
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


def read_result_file(input_file):
    with open(input_file, 'r') as f:
        result = json.load(f)

    #result format:
    # {
    # "12": {
    # "question": "right to property according to the constitution of india is a?",
    # "answers": [
    #     "constitutional right"
    # ],
    # "contexts": [
    #     {
    #         "docid": "baichuan2-13b-chat_nq_from_llm-embedder_None_loop4_21_20240116132218",
    #         "score": 0.9580259323120117,
    #         "has_answer": false
    #     },
    #     {
    #         "docid": "baichuan2-13b-chat_nq_from_llm-embedder_None_loop4_21_20240116132218",
    #         "score": 0.9580259323120117,
    #         "has_answer": false
    #     }...
    # ]
    # }
    # }

    return result


def get_index(elasticsearch_url, index_name):
    index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name, verbose=False)
    return index


def gather_LLM_gen_text(retrieval_file_path):
    # load 10 json files into a list
    # format:
    #{"id":"baichuan2-13b-chat_nq_from_bge-base_bge_loop1_21_20240109095603","question":"right to property according to the constitution of india is a?","answers":["constitutional right"],"response":"Constitution of India grants citizens the Right to Property (RtP), safeguarded under Article 300A. Initially, RtP was part of Article 19(f)butwas stripped offthrough the 44thAmendmentActof 1978makingitalegal right instead offundamental right. Nevertheless,it was reinstated via the 45th Amendment Act of 1982 along with necessary caveats."}
    #{"id":"baichuan2-13b-chat_nq_from_bge-base_bge_loop1_31_20240109095603","question":"how many countries are a part of opec?","answers":["14"],"response":"The background document provides important contextual information regarding the topic under consideration – namely, how many countries make up the Organization of the Petroleum Exporting Countries (OPEC). Founded in 1960, OPEC initially consisted of five countries – Iran, Iraq, Kuwait, Saudi Arabia, and Venezuela. Since then, it has grown into a permanent intergovernmental organization comprising 13 member countries today; all of them are major oil producers. They are joined by Algeria, Angola, Congo, Ecuador, Equatorial Guinea, Gabon, Libya, and the United Arab Emirates. By doing so, they aim to stabilize the volatile price of"}
    try:
        data = []
        for i in range(1, 11):
            file_path = os.path.join(retrieval_file_path, f'{i}', 'merged_file', 'merged.jsonl')
            with open(file_path, 'r') as f:
                print(f'loading {file_path}')
                for line in f:
                    data.append(json.loads(line))
        zero_paths = ["../../data_v2/zero_gen_data/DPR/post_processed_sampled_data/merged_file/merged.jsonl",
                    "../../data_v2/misinfo_data/DPR/mis_passage_processed/merged_file/merged.jsonl",]
        for zero_path in zero_paths:
            with open(zero_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        # convert data into a dict like this:
        # {'baichuan2-13b-chat_nq_from_bge-base_bge_loop1_21_20240109095603':{ 'question': 'right to property according to the constitution of india is a?', 'answers': ['constitutional right'], 'response': 'Constitution of India grants citizens the Right to Property (RtP), safeguarded under Article 300A. Initially, RtP was part of Article 19(f)butwas stripped offthrough the 44thAmendmentActof 1978makingitalegal right instead offundamental right. Nevertheless,it was reinstated via the 45th Amendment Act of 1982 along with necessary caveats.'}...}
        llm_data = {}
        for item in data:
            llm_data[item['id']] = item
    except Exception as e:
        print(e)
        llm_data = None

    return llm_data


# def get_doc_text(docid, index, task, llm_data):
#     # if docid is formed by digits, it is a human generated text, get it from index
#     # if docid has alpha, it is a LLM generated text, get it from llm_data
#     if not docid.isdigit():
#         try:
#             text = llm_data[docid]['response']
#         except Exception as e:
#             print(f'error docid: {docid}')
#
#     else:
#         text = index.get_document_by_id(docid).page_content
#
#     if task == 'filter_source':
#         return text
#     else:
#         return text.split()


def get_doc_text(docid, index, task='filter_source'):
    text = index.get_document_by_id(docid).page_content

    if task == 'filter_source':
        return text  # Split the text into words for BLEU computation
    else:
        return text.split()


def get_detector(checkpoint, device):
    detector = pipeline('text-classification', model=checkpoint, device=device, framework='pt')
    return detector


def compute_source(question, documents, detector):
    # paired = [dict(text=q, text_pair=a) for q, a in zip(batch['question'], batch['answer'])]
    # out = detector(paired , max_length=512, truncation=True)

    question = [question] * len(documents)
    paired = [dict(text=q, text_pair=a) for q, a in zip(question, documents)]
    out = detector(paired , max_length=512, truncation=True)
    return out


def find_docs_with_source(query_id, result, index, detector, num_docs=100, llm_data=None, task='filter_source'):
    original_contexts = result['contexts']
    doc_ids = [context['docid'] for context in original_contexts]
    # compute label, if the doc_id is digit, it is human generated, if not, it is LLM generated

    origin_label = [0 if doc_id.isdigit() else 1 for doc_id in doc_ids]

    question = result['question']
    doc_texts = [get_doc_text(doc_id, index, task) for doc_id in doc_ids]

    candidate_contexts = original_contexts
    candidate_docs = doc_texts

    current_source = compute_source(question, candidate_docs, detector)
    # print(doc_ids)
    # print(current_source)
    # [{'label': 'LABEL_1', 'score': 0.999941349029541}, {'label': 'LABEL_1', 'score': 0.9999754428863525}, {'label': 'LABEL_1', 'score': 0.9999765157699585}, {'label': 'LABEL_0', 'score': 0.8834363222122192}, {'label': 'LABEL_1', 'score': 0.999976634979248}]
    # LABEL_0: Human, LABEL_1: chatgpt
    #get pred
    pred = [int(item['label'][-1]) for item in current_source]
    new_contexts = []
    human_count = 0

    for i, context in enumerate(candidate_contexts):
        if human_count < num_docs:
            if current_source[i]['label'] == 'LABEL_0':
                new_contexts.append(context)
                human_count += 1
            else:
                print(f'del llm doc: {context["docid"]}')
        else:
            new_contexts.append(context)

    if len(new_contexts) < num_docs:
        # if there are not enough human docs, add from the first of the original_contexts
        for i, context in enumerate(original_contexts):
            if len(new_contexts) < num_docs and pred[i] == 1:
                new_contexts.append(context)
                print(f'add llm doc for query {query_id}: {context["docid"]}')

    result['contexts'] = new_contexts
    print(f'pred: {pred}')
    print(f'len of new_contexts: {len(new_contexts)}')
    return result, origin_label, pred


def compute_self_bleu(documents):
    # Set weights for trigram only
    weights = {'trigram': (1 / 3., 1 / 3., 1 / 3.)}
    self_bleu = SelfBLEU(documents, weights)
    scores = self_bleu.get_score()
    # Since we are only interested in the trigram score, we will return that directly
    average_score = np.mean(scores['trigram'])
    return average_score


def find_docs_with_self_bleu_constraint(query_id, result, index, max_self_bleu=0.4, num_docs=5, llm_data=None, task='filter_bleu'):
    original_contexts = result['contexts']
    doc_ids = [context['docid'] for context in original_contexts]

    doc_texts = [get_doc_text(doc_id, index, task) for doc_id in doc_ids]

    candidate_contexts = original_contexts[:num_docs]
    candidate_docs = doc_texts[:num_docs]

    current_self_bleu = compute_self_bleu(candidate_docs)
    print(f'Initial Self-BLEU (trigram): {current_self_bleu:.4f}')

    next_docs_pool = doc_texts[num_docs:]
    next_contexts_pool = original_contexts[num_docs:]

    while current_self_bleu > max_self_bleu and next_docs_pool:
        bleu_scores_excluding_each_doc = [compute_self_bleu(candidate_docs[:i] + candidate_docs[i + 1:]) for i in
                                          range(len(candidate_docs))]
        min_bleu_idx = np.argmin(bleu_scores_excluding_each_doc)
        removed_context = candidate_contexts.pop(min_bleu_idx)
        candidate_docs.pop(min_bleu_idx)

        new_doc = next_docs_pool.pop(0)
        new_context = next_contexts_pool.pop(0)

        candidate_docs.append(new_doc)
        candidate_contexts.append(new_context)

        new_self_bleu = compute_self_bleu(candidate_docs)

        print(
            f'query_id: {query_id}, removed docid: {removed_context["docid"]}, new docid: {new_context["docid"]},  old self-bleu: {current_self_bleu:.4f}, new self-bleu: {new_self_bleu:.4f}')

        current_self_bleu = new_self_bleu

    # Append the remaining documents that were not filtered out
    candidate_contexts.extend(next_contexts_pool)

    result['contexts'] = candidate_contexts
    return result


def run_filter_source(input_file, output_file, elasticsearch_url, index_name, max_self_bleu, num_docs):
    parent_dir = os.path.dirname(os.path.dirname(input_file))
    llm_data = gather_LLM_gen_text(parent_dir)
    # llm_data = None
    result = read_result_file(input_file)
    index = get_index(elasticsearch_url, index_name)
    detector = get_detector('../../ret_model/chatgpt-qa-detector-roberta', device)
    label = []
    pred = []
    # parent_dir = os.path.dirname(os.path.dirname(input_file))
    # llm_data = gather_LLM_gen_text(parent_dir)
    for query_id, query_result in tqdm(result.items()):
        filtered_result, label_q, pred_q = find_docs_with_source(query_id, query_result, index, detector, num_docs, llm_data=llm_data, task='filter_source')
        result[query_id] = filtered_result
        label.extend(label_q)
        pred.extend(pred_q)

    with open(output_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Filtered results have been written to {output_file}")

    print(f'evaluating detection...')
    print(classification_report(label, pred, target_names=['human','chatgpt'], output_dict=True))

    print(f'evaluating input...')
    evaluate_retrieval(input_file, [5, 20, 100], False)
    print(f'evaluating output...')
    evaluate_retrieval(output_file, [5, 20, 100], False)


def run_filter_bleu(input_file, output_file, elasticsearch_url, index_name, max_self_bleu, num_docs):
    result = read_result_file(input_file)
    index = get_index(elasticsearch_url, index_name)
    parent_dir = os.path.dirname(os.path.dirname(input_file))
    llm_data = gather_LLM_gen_text(parent_dir)
    # llm_data = None
    for query_id, query_result in tqdm(result.items()):
        filtered_result = find_docs_with_self_bleu_constraint(query_id, query_result, index, max_self_bleu, num_docs, llm_data=llm_data, task='filter_bleu')
        result[query_id] = filtered_result

    with open(output_file, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Filtered results have been written to {output_file}")

    print(f'evaluating input...')
    evaluate_retrieval(input_file, [5, 20, 100], False)
    print(f'evaluating output...')
    evaluate_retrieval(output_file, [5, 20, 100], False)


if __name__ == '__main__':
    args = get_args()
    if args.task == 'filter_source':
        run_filter_source(args.input_file, args.output_file, args.elasticsearch_url, args.index_name, args.max_self_bleu, args.num_docs)
    else:
        run_filter_bleu(args.input_file, args.output_file, args.elasticsearch_url, args.index_name, args.max_self_bleu, args.num_docs)



