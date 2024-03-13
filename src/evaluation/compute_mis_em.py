#compute top5 doc self bleu
import os
import sys
from fast_bleu import BLEU, SelfBLEU
import json
import re
from collections import defaultdict
import datasets
import string


def evaluate(predictions, mis_answer_dict):
    # evaluate the predictions with exact match
    def _normalize_answer(s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def exact_match_score(example, mis_answer_dict):
        question = example['question']
        ground_truths = mis_answer_dict[question]
        assert type(ground_truths) == list, f'ground_truths is not a list, id:{example["id"]}, ground_truth:{ground_truths}'
        prediction = example['response']
        example['exact_match'] = 0
        if not prediction:
            print(f'no prediction for qid {example["qid"]}, {example["query"]}')
            return example
        for ground_truth in ground_truths:
            if _normalize_answer(ground_truth) in _normalize_answer(prediction):
                example['exact_match'] = 1
                break
        return example

    fn_args = {'mis_answer_dict': mis_answer_dict}
    predictions = predictions.map(exact_match_score, fn_kwargs=fn_args)
    return predictions


def get_mis_answer(retrieval_file_path):
    # mis_parent_path = os.path.dirname(os.path.dirname(os.path.dirname(retrieval_file_path)))
    # mis_answer_path = os.path.join(mis_parent_path, 'misinfo_data', 'mis_passage_processed', 'merged_file', 'merged.jsonl')
    mis_answer_dict = {}
    mis_answer_path = '../../data_v2/misinfo_data/DPR/mis_passage_processed/merged_file/merged.jsonl'
    with open(mis_answer_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            mis_answer_dict[data['question']] = data['false_answer']
    assert len(mis_answer_dict) == 800
    return mis_answer_dict



def calculate_mis_em(retrieval_file, mis_answer_path):
    mis_answer_dict = get_mis_answer(mis_answer_path)
    dataset = datasets.load_dataset("json", data_files=retrieval_file)["train"]
    print('evaluate QA EM...')
    prediction = evaluate(dataset, mis_answer_dict)
    EM = sum(prediction['exact_match']) / len(prediction['exact_match'])
    return EM






