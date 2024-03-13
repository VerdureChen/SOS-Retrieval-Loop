#delete docs from index according to the doc id

import sys
import os
import json
import argparse
import time

sys.path.append('../retrieval_loop')

from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
from faiss_search import Batch_FAISS


def read_ids_from_file(file_path):
    ids = []
    with open(file_path) as f:
        for line in f:
            ids.append(line.strip())
    return ids




def get_args():
    parser = argparse.ArgumentParser()
    # a list of docid files
    parser.add_argument('--id_files', type=str, nargs='+', default=[])
    # model name
    parser.add_argument('--model_name', type=str, default='bm25')
    # index path
    parser.add_argument('--index_path', type=str, default='')
    # index name
    parser.add_argument('--index_name', type=str, default='')
    # elasticsearch url
    parser.add_argument('--elasticsearch_url', type=str, default='http://localhost:9200')
    # config file path
    parser.add_argument('--config_file_path', type=str, default='../config/delete_doc_from_index.json')
    #delete log file path
    parser.add_argument('--delete_log_path', type=str, default='../logs')

    args = parser.parse_args()
    # 读取 JSON 配置文件
    # {
    #    "id_files": ["../data/ids.txt"],
    #    "model_name": "bm25",
    #    "index_path": "",
    #    "index_name": "index",
    #    "elasticsearch_url": "http://localhost:9200"
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


def delete_docs_from_index(id_files, model_name, index_path, index_name, elasticsearch_url, delete_log_path):
    delete_log_file = os.path.join(delete_log_path, f'delete_log_{model_name}_{index_name}_{time.strftime("%Y%m%d-%H%M%S")}.log')
    index_p = os.path.join(index_path, index_name)
    if model_name == 'bm25':
        index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name)
        index_size_before = index.get_document_count()
    elif model_name == 'faiss':
        index = Batch_FAISS.load_local(index_p, None)
        index_size_before = len(index.docstore._dict)

    else:
        raise ValueError(f'Invalid model name: {model_name}')

    with open(delete_log_file, 'w') as f:
        f.write(f'index size before: {index_size_before}\n')
        # wether id_files are directories or files
        delete_id_files = []
        for id_file in id_files:
            if os.path.isdir(id_file):
                for file in os.listdir(id_file):
                    if file.startswith(index_name):
                        delete_id_files.append(os.path.join(id_file, file))
            else:
                delete_id_files.append(id_file)

        for id_file in delete_id_files:
            if model_name not in id_file:
                continue
            ids = read_ids_from_file(id_file)

            if model_name == 'bm25':
                index.delete_documents_by_id(ids)
                index_size_after = index.get_document_count()
            elif model_name == 'faiss':
                try:
                    index.delete(ids)
                except Exception as e:
                    print(e)
                    pass
                index_size_after = len(index.docstore._dict)
            else:
                raise ValueError(f'Invalid model name: {model_name}')
            # write ids to delete log file
            f.write(f'delete ids from {id_file}: {ids}\n')
            print(f'index size before: {index_size_before}, delete ids from {id_file}, index size after: {index_size_after}')
            f.write(f'index size after: {index_size_after}\n')
    if model_name == 'faiss':
        index.save_local(index_p)


if __name__ == '__main__':
    args = get_args()
    delete_docs_from_index(args.id_files, args.model_name, args.index_path, args.index_name, args.elasticsearch_url, args.delete_log_path)


