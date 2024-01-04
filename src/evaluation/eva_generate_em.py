import datasets
import string
import re
import os


def evaluate(predictions):
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

    def exact_match_score(example):
        ground_truths = example['answers']
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

    predictions = predictions.map(exact_match_score)
    return predictions


def main():
    # load the predictions
    res_file_path = '../../data_v2/zero_gen_data/DPR'
    data_names = ['nq', 'webq', 'tqa', 'pop']
    model_names = ['chatglm3-6b-chat', 'qwen-14b-chat', 'baichuan2-13b-chat', 'llama2-13b-chat', 'gpt-3.5-turbo']
    for data_name in data_names:
        for model_name in model_names:
            res_path = f'{res_file_path}/{data_name}-test-gen-{model_name}.jsonl'
            # check if the file exists
            if not os.path.exists(res_path):
                continue
            predictions = datasets.load_dataset('json', data_files=res_path)['train']
            predictions = evaluate(predictions)
            print('-' * 20)
            print(f'{data_name}-{model_name}')
            # compute the exact match score
            # 'list' object has no attribute 'sum'
            # predictions['exact_match'] is a list
            print(sum(predictions['exact_match']) / len(predictions['exact_match']))
            print('-' * 20)


if __name__ == '__main__':
    main()

