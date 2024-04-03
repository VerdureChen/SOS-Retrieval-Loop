#compute top5 doc self bleu
import os
import sys
from fast_bleu import BLEU, SelfBLEU
import json
import re
from collections import defaultdict
sys.path.append('../retrieval_loop')
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
import datasets


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
        for i in range(1, 10):
            file_path = os.path.join(retrieval_file_path, f'{i}', 'merged_file', 'merged.jsonl')
            with open(file_path, 'r') as f:
                print(f'loading {file_path}')
                for line in f:
                    data.append(json.loads(line))
        zero_paths = ["../../data_v2/zero_gen_data/DPR/post_processed_sampled_data/merged_file/merged.jsonl",
                      "../../data_v2/misinfo_data/DPR/mis_passage_processed/merged_file/merged.jsonl",
                      "../../data_v2/update_data/DPR/update_passage_processed/merged_file/merged.jsonl"]
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

def get_doc_text(docid, index, llm_data):
    # if docid is formed by digits, it is a human generated text, get it from index
    # if docid has alpha, it is a LLM generated text, get it from llm_data
    if not docid.isdigit():
        try:
            text = llm_data[docid]['response']
        except Exception as e:
            print(f'error docid: {docid}')

    else:
        text = index.get_document_by_id(docid).page_content

    return text


def calculate_self_bleu(retrieval_file, index, llm_data, topks=[1, 5, 10]):
    with open(retrieval_file, 'r') as file:
        try:
            data = json.load(file)
        except Exception as e:
            print(f'error file: {retrieval_file}')


    # 初始化结果字典，包括bigram和trigram
    result = defaultdict(lambda: defaultdict(list))
    for topk in topks:
        result[topk]['bigram'] = []
        result[topk]['trigram'] = []


    # 遍历每个查询
    for query_id, query_data in data.items():
        contexts = query_data['contexts']
        for topk in topks:
            # 获取当前topk的文档
            topk_contexts = contexts[:topk]
            context_texts = []
            # 计算当前topk的文档的self bleu
            for ctx in topk_contexts:
                docid = ctx['docid']
                try:
                    text = get_doc_text(docid, index, llm_data)
                except Exception as e:
                    print(f'error docid: {docid}')
                # split text into word list
                text = re.split(r'\W+', text)
                context_texts.append(text)
            # calculate self bleu
            weights = {'bigram': (1 / 2., 1 / 2.), 'trigram': (1 / 3., 1 / 3., 1 / 3.)}
            self_bleu = SelfBLEU(context_texts, weights)
            scores = self_bleu.get_score()
            # compute average bleu score
            for ngram in ['bigram', 'trigram']:
                result[topk][ngram].append(sum(scores[ngram]) / len(scores[ngram]))

    # compute average bleu score for each topk
    for topk in topks:
        for ngram in ['bigram', 'trigram']:
            result[topk][ngram] = sum(result[topk][ngram]) / len(result[topk][ngram])
    # result format:
    # {1: {'bigram': 0.0, 'trigram': 0.0}, 5: {'bigram': 0.0, 'trigram': 0.0}, 10: {'bigram': 0.0, 'trigram': 0.0}}
    return result


# def calculate_self_bleu(retrieval_file, elasticsearch_url, index_name, llm_data, topks=[1, 5, 10]):
#     with open(retrieval_file, "r", encoding='utf-8') as f:
#         question_dataset = json.load(f)
#     formatted_data = [
#         {
#             "id": id,
#             "question": details["question"],
#             "answers": details["answers"],
#             "contexts": details["contexts"],
#         }
#         for id, details in question_dataset.items()
#     ]
#     dataset = datasets.Dataset.from_list(formatted_data)
#
#     # 定义一个函数example处理，用于计算一个query的self bleu
#     def example_process(example, elasticsearch_url=None, index_name=None, topks=None, llm_data=None):
#         index = get_index(elasticsearch_url, index_name)
#         contexts = example['contexts']
#         for topk in topks:
#             # 获取当前topk的文档
#             topk_contexts = contexts[:topk]
#             context_texts = []
#             # 计算当前topk的文档的self bleu
#             for ctx in topk_contexts:
#                 docid = ctx['docid']
#                 text = get_doc_text(docid, index, llm_data)
#                 # split text into word list
#                 text = re.split(r'\W+', text)
#                 context_texts.append(text)
#             # calculate self bleu
#             weights = {'bigram': (1 / 2., 1 / 2.), 'trigram': (1 / 3., 1 / 3., 1 / 3.)}
#             self_bleu = SelfBLEU(context_texts, weights)
#             scores = self_bleu.get_score()
#             # compute average bleu score
#             for ngram in ['bigram', 'trigram']:
#                 example[f'self_bleu_{topk}_{ngram}'] = sum(scores[ngram]) / len(scores[ngram])
#         return example
#
#     fn_args = {'elasticsearch_url': elasticsearch_url, 'index_name': index_name, 'topks': topks, 'llm_data': llm_data}
#     # 对dataset中的每个example进行处理
#     dataset = dataset.map(example_process, num_proc=16, fn_kwargs=fn_args)
#     # 计算平均值
#     result = defaultdict(dict)
#     for topk in topks:
#         for ngram in ['bigram', 'trigram']:
#             result[topk][ngram] = dataset[f'self_bleu_{topk}_{ngram}']
#     result[topk][ngram] = sum(result[topk][ngram]) / len(result[topk][ngram])
#
#     # result format:
#     # {1: {'bigram': 0.0, 'trigram': 0.0}, 5: {'bigram': 0.0, 'trigram': 0.0}, 10: {'bigram': 0.0, 'trigram': 0.0}}
#     return result







