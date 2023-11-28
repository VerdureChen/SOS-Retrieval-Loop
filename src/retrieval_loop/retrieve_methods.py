from langchain.document_loaders import HuggingFaceDatasetLoader
from sentence_transformers import SentenceTransformer
from faiss_search import Batch_FAISS, Batch_HuggingFaceEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
import os
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
import elasticsearch
from langchain.vectorstores import ElasticsearchStore
from tqdm import tqdm
from evaluate_dpr_retrieval import has_answers, SimpleTokenizer, evaluate_retrieval
import json
import argparse
import torch


def get_args():
    # get config_file_path, which is the path to the config file
    # config file formatted as a json file:
    # {
    #   "query_file": "../../data_v2/zero_gen_data/DPR/nq-test-gen-baichuan2-13b-chat.jsonl",
    #   "retrieval_method": "DPR", # BM25, DPR, Contriever, RetroMAE, all-mpnet, BGE, LLM-Embedder
    #   "index_name": "DPR_faiss_index",
    #   "index_path": "../../data_v2/indexes",
    #   "page_content_column": "question"
    # }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    print("config: ", config)
    return config


def load_retrieval_embeddings(retrieval_model, normalize_embeddings=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings,
                     'batch_size': 128,
                     'show_progress_bar': True
                     }
    embeddings = Batch_HuggingFaceEmbeddings(model_name=retrieval_model, model_kwargs=model_kwargs,
                                             encode_kwargs=encode_kwargs)
    return embeddings


def Retrieval(query_files, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings,
              output_files):


    # map retrieval model names: DPR, Contriever, RetroMAE, all-mpnet, BGE, LLM-Embedder
    query_instruction = ''
    doc_instruction = ''
    if 'DPR' in retrieval_model.upper():
        retrieval_model = '../../ret_model/DPR/facebook-dpr-ctx_encoder-multiset-base'
        query_model = '../../ret_model/DPR/facebook-dpr-question_encoder-multiset-base'
    elif 'CONTRIEVER' in retrieval_model.upper():
        retrieval_model = '../../ret_model/contriever-base-msmarco'
    elif 'RETROMAE' in retrieval_model.upper():
        retrieval_model = '../../ret_model/RetroMAE_BEIR'
    elif 'ALL-MPNET' in retrieval_model.upper():
        retrieval_model = '../../ret_model/all-mpnet-base-v2'
    elif 'BGE-LARGE' in retrieval_model.upper():
        retrieval_model = '../../ret_model/bge-large-en-v1.5'
        query_instruction = 'Represent this sentence for searching relevant passages: '
    elif 'BGE-BASE' in retrieval_model.upper():
        retrieval_model = '../../ret_model/bge-base-en-v1.5'
        query_instruction = 'Represent this sentence for searching relevant passages: '
    elif 'LLM-EMBEDDER' in retrieval_model.upper():
        retrieval_model = '../../ret_model/llm-embedder'
        query_instruction = 'Represent this query for retrieving relevant documents: '
        doc_instruction = "Represent this document for retrieval: "
    elif 'BM25' in retrieval_model.upper():
        retrieval_model = 'BM25'
    else:
        raise ValueError(f'unknown retrieval model: {retrieval_model}')

    # load the query embedder and index
    if "DPR" in retrieval_model.upper():
        embeddings = load_retrieval_embeddings(retrieval_model, normalize_embeddings=normalize_embeddings)
        query_embeddings = load_retrieval_embeddings(query_model, normalize_embeddings=normalize_embeddings)
        print(f'loaded ctx embedder: {retrieval_model}')
        print(f'loaded query embedder: {query_model}')
        index_p = os.path.join(index_path, index_name)
        index = Batch_FAISS.load_local(index_p, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        index_size = len(index.docstore._dict)

    elif retrieval_model != "BM25":
        embeddings = load_retrieval_embeddings(retrieval_model, normalize_embeddings=normalize_embeddings)
        print(f'loaded query embedder: {retrieval_model}')
        index_p = os.path.join(index_path, index_name)
        index = Batch_FAISS.load_local(index_p, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        index_size = len(index.docstore._dict)
    else:
        print('Please make sure you have started elastic search server')
        elasticsearch_url = "http://0.0.0.0:9978"
        index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name)
        index_size = index.get_document_count()

    print(f'loaded index: {index_name}, size: {index_size}')

    for query_file, output_file in zip(query_files, output_files):
        # load queries
        loader = HuggingFaceDatasetLoader('json', data_files=query_file,
                                          page_content_column=page_content_column)
        queries = loader.load()

        # retrieve
        print('retrieving ...')
        output_file_json = output_file + '.json'
        output_file_trec = output_file + '.trec'


        with open(output_file_trec, 'w', encoding='utf-8') as f_t:
            retrieval = {}
            tokenizer = SimpleTokenizer()
            batch_size = 1024  # Set the batch size as per your requirement
            batch_queries = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]

            for batch in tqdm(batch_queries):
                print(f'processing batch of size {len(batch)} ...')
                batch_query_texts = [query_instruction + query.page_content for query in batch]
                batch_metadata = [query.metadata for query in batch]
                batch_query_ids = [metadata['id'] for metadata in batch_metadata]
                batch_answers = [metadata['answer'] for metadata in batch_metadata]

                assert all(type(answer) == list for answer in batch_answers)

                if "DPR" in retrieval_model.upper():
                    embedded_queries = query_embeddings.embed_queries(batch_query_texts)
                    batch_docs_and_scores = index.similarity_search_with_score_by_vector(embedded_queries, k=100)
                elif retrieval_model != "BM25":
                    embedded_queries = embeddings.embed_queries(batch_query_texts)
                    batch_docs_and_scores = index.similarity_search_with_score_by_vector(embedded_queries, k=100)
                else:
                    batch_docs_and_scores = [index.get_relevant_documents(query_text, num_docs=100) for query_text in
                                             tqdm(batch_query_texts)]

                for query_idx, (query, docs_and_scores) in enumerate(zip(batch, batch_docs_and_scores)):
                    query_id = batch_query_ids[query_idx]
                    answer = batch_answers[query_idx]

                    if query_id not in retrieval:
                        retrieval[query_id] = {'question': query.page_content, 'answers': answer, 'contexts': []}

                    for i, (doc, score) in enumerate(docs_and_scores):
                        rank = i + 1
                        doc_id = doc.metadata['id']
                        doc_content = doc.page_content
                        if doc_instruction != '':
                            assert doc_content.startswith(doc_instruction)
                            doc_content = doc_content[len(doc_instruction):]
                        title, text = doc_content.split('\n')
                        tag = retrieval_model.split('/')[-1]
                        f_t.write(f'{query_id} Q0 {doc_id} {rank} {score} {tag}\n')

                        answer_exist = has_answers(text, answer, tokenizer, False)
                        retrieval[query_id]['contexts'].append(
                            {'docid': doc_id, 'score': float(score), 'has_answer': answer_exist}
                        )
            json.dump(retrieval, open(output_file_json, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

        # evaluate
        evaluate_retrieval(output_file_json, [20, 100], False)


if __name__ == '__main__':
    config = get_args()
    query_files = config["query_files"]
    page_content_column = config["query_page_content_column"]
    retrieval_model = config["retrieval_model"]
    index_name = config["index_name"]
    index_path = config["index_path"]
    normalize_embeddings = config["normalize_embeddings"]
    output_files = config["output_files"]

    Retrieval(query_files, page_content_column, retrieval_model, index_name, index_path, normalize_embeddings,
              output_files)
