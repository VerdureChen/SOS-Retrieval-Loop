import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import string
def read_docs(docs_file):
    with open(docs_file, 'r') as f_docs:
        docs_data = [json.loads(line) for line in f_docs]
    id_to_content = {doc['id']: doc['contents'] for doc in docs_data}
    return id_to_content


def read_qrels(qrels_file, id_to_content):
    with open(qrels_file, 'r') as f_qrels:
        qrels_data = [json.loads(line) for line in f_qrels]
    c_qrels = {}
    for qrel in qrels_data:
        if 'question' in qrel:
            qid = qrel['question']
            # only extract alpha words
            qid = ''.join([i for i in qid if i.isalpha()])
            for id, content in id_to_content.items():
                k = 'output' if 'output' in qrel else 'gpt_output'
                if qrel[k][0].strip() == content:
                    if qid not in c_qrels:
                        c_qrels[qid] = [id]
                    else:
                        c_qrels[qid].append(id)
    return c_qrels


def read_retrieval_results(retrieval_results_file, cut_off=1000):
    with open(retrieval_results_file, 'r') as f:
        data = json.loads(f.read())
    h_qrels = {}
    retrieval_results = {}
    for item in data:
        qid = data[item]['question']
        # only extract alpha words
        qid = ''.join([i for i in qid if i.isalpha()])
        for rank, context in enumerate(data[item]['contexts']):
            docid = context['docid']
            score = float(context['score'])
            has_answer = context['has_answer']
            if int(docid) < 30000000:
                 # docid小于30000000的为h数据集
                if has_answer:
                    if qid not in h_qrels:
                        h_qrels[qid] = [docid]
                    else:
                        h_qrels[qid].append(docid)
            else:
                pass
            if qid not in retrieval_results:
                retrieval_results[qid] = {}
            retrieval_results[qid][docid] = (rank+1, score, has_answer)
            if rank == cut_off:
                break
    return h_qrels, retrieval_results


def compute_avg_rank(retrieval_results, qrels, print_or_not=False):
    '''

    :param retrieval_results:
    :param qrels:
    :return:
    '''
    avg_rank = 0
    pid_in_top = 0
    qid_in_top = 0
    pid_not_in_top = 0
    for qid in retrieval_results:
        if qid in qrels:
            for pid in qrels[qid]:
                if pid in retrieval_results[qid]:
                    avg_rank += retrieval_results[qid][pid][0]
                    pid_in_top += 1
                    qid_in_top = 1
        if qid_in_top != 1:
            pid_not_in_top += 1
            if print_or_not:
                print('qid not in top: ', qid)
        qid_in_top = 0

    avg_rank = avg_rank / pid_in_top
    return avg_rank, pid_in_top, pid_not_in_top


def compute_max_rank(retrieval_results, qrels, print_or_not=False):
    '''

    :param retrieval_results:
    :param qrels:
    :return:
    '''
    total_top_rank = 0
    pid_in_top = 0
    qid_in_top = 0
    pid_not_in_top = 0
    for qid in retrieval_results:
        top_rank_in_qid = 1000
        if qid in qrels:
            for pid in qrels[qid]:
                if pid in retrieval_results[qid]:
                    if retrieval_results[qid][pid][0] < top_rank_in_qid:
                        top_rank_in_qid = retrieval_results[qid][pid][0]

                    qid_in_top = 1
        if qid_in_top != 1:
            pid_not_in_top += 1
            if print_or_not:
                print('qid not in top: ', qid)
        else:
            pid_in_top += 1
        qid_in_top = 0
        if top_rank_in_qid != 1000:
            total_top_rank += top_rank_in_qid

    avg_top_rank = total_top_rank / pid_in_top
    return avg_top_rank, pid_in_top, pid_not_in_top


def compute_has_answer_rate(retrieval_results, qrels, print_or_not=False):
    '''

    :param retrieval_results:
    :param qrels:
    :return:
    '''
    has_answer = 0
    pid_in_top = 0
    qid_in_top = 0
    pid_not_in_top = 0
    for qid in retrieval_results:
        if qid in qrels:
            for pid in qrels[qid]:
                if pid in retrieval_results[qid]:
                    if retrieval_results[qid][pid][2]:
                        has_answer += 1
                    qid_in_top += 1
                    pid_in_top = 1
        if pid_in_top != 1:
            pid_not_in_top += 1
            if print_or_not:
                print('pid not in top: ', pid)
        pid_in_top = 0

    has_answer_rate = has_answer / qid_in_top
    return has_answer_rate, qid_in_top, pid_not_in_top


def compute_win_rate(retrieval_results, c_qrels, h_qrels, cut_off=1000):
    # compute winning rate of C_qrel compared with H_qrel
    c_win = 0
    h_win = 0
    c_win_rank = 0
    h_win_rank = 0
    total = 0
    for qid in retrieval_results:
        for pid in c_qrels[qid]:
            if pid in retrieval_results[qid]:
                c_rank = retrieval_results[qid][pid][0]
                if qid in h_qrels:
                    for pid in h_qrels[qid]:
                        if pid in retrieval_results[qid]:
                            h_rank = retrieval_results[qid][pid][0]
                            if c_rank < h_rank:
                                c_win += 1
                                c_win_rank += h_rank - c_rank
                                total += 1
                            elif c_rank > h_rank:
                                h_win += 1
                                h_win_rank += c_rank - h_rank
                                total += 1
                        else:
                            c_win += 1
                            c_win_rank += cut_off - c_rank
                            total += 1
                else:
                    c_win += 1
                    c_win_rank += cut_off - c_rank
                    total += 1
            else:
                if qid in h_qrels:
                    for pid in h_qrels[qid]:
                        if pid in retrieval_results[qid]:
                            h_win_rank += cut_off - retrieval_results[qid][pid][0]
                            h_win += 1
                            total += 1
                        else:
                            total += 1
                else:
                    total += 1

    c_win_rank = c_win_rank / c_win
    h_win_rank = h_win_rank / h_win
    c_win = c_win / total
    h_win = h_win / total

    return c_win, h_win, c_win_rank, h_win_rank


if __name__ == '__main__':
    # task = 'nq'
    # task = 'wq'
    task = 'triva'
    retrieval_results_file = '/home1/cxy/Rob_LLM/LLM4Ranking/data/Genread_DPR/runs/trivia.retromae.run.json'
    if task == 'nq':
        docs_file = '/home1/cxy/Rob_LLM/LLM4Ranking/data/Genread_DPR/add_corpus_jsonl/nq-test.jsonl'
        qrels_file = '/home1/cxy/Rob_LLM/Genread/GenRead/indatasets/backgrounds-greedy-text-davinci-002/nq-test/nq-test-p1.jsonl'
    elif task == 'wq':
        docs_file = '/home1/cxy/Rob_LLM/LLM4Ranking/data/Genread_DPR/add_corpus_jsonl/webq-test.jsonl'
        qrels_file = '/home1/cxy/Rob_LLM/Genread/GenRead/indatasets/backgrounds-greedy-text-davinci-002/webq-test/webq-test-p1.jsonl'
    elif task == 'triva':
        docs_file = '/home1/cxy/Rob_LLM/LLM4Ranking/data/Genread_DPR/add_corpus_jsonl/tqa-test.jsonl'
        qrels_file = '/home1/cxy/Rob_LLM/Genread/GenRead/indatasets/backgrounds-greedy-text-davinci-002/tqa-test/tqa-test-p1.jsonl'
    id_to_content = read_docs(docs_file)
    c_qrels = read_qrels(qrels_file, id_to_content)
    h_qrels, retrieval_results = read_retrieval_results(retrieval_results_file, cut_off=5)

    c_avg_rank, c_pid_in_top, c_pid_not_in_top = compute_avg_rank(retrieval_results, c_qrels, print_or_not=False)
    c_top_rank, c_pid_in_top, c_pid_not_in_top = compute_max_rank(retrieval_results, c_qrels, print_or_not=False)
    c_has_answer, _, _ = compute_has_answer_rate(retrieval_results, c_qrels, print_or_not=False)
    print('c_top_rank: ', c_top_rank, 'c_avg_rank: ', c_avg_rank, 'c_has_answer: ', c_has_answer, 'c_pid_not_in_top: ', c_pid_not_in_top)


    h_avg_rank, h_pid_in_top, h_pid_not_in_top = compute_avg_rank(retrieval_results, h_qrels, print_or_not=False)
    h_top_rank, h_pid_in_top, h_pid_not_in_top = compute_max_rank(retrieval_results, h_qrels, print_or_not=False)
    print('h_top_rank: ', h_top_rank, 'h_avg_rank: ', h_avg_rank, 'h_pid_not_in_top: ', h_pid_not_in_top)


    c_win, h_win, c_win_rank, h_win_rank = compute_win_rate(retrieval_results, c_qrels, h_qrels, cut_off=5)
    print('c_win: ', c_win, 'h_win: ', h_win, 'c_win_rank: ', c_win_rank, 'h_win_rank: ', h_win_rank)



