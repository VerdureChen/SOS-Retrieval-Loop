#compute top5 doc self bleu
import os
import sys
from fast_bleu import BLEU, SelfBLEU
import json
import re
from collections import defaultdict
import datasets
import string
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

def get_openai_api(model_name):
    if model_name == "Qwen":
        print("Using Qwen")
        openai.api_base = "http://124.16.138.150:8111/v1"
        openai.api_key = "xxx"
    elif model_name == "Llama":
        print("Using Llama")
        openai.api_base = "http://124.16.138.150:8223/v1"
        openai.api_key = "xxx"
    elif model_name == "chatglm3":
        print("Using chatglm3")
        openai.api_base = "http://124.16.138.150:8113/v1"
        openai.api_key = "xxx"
    elif model_name == "baichuan2-13b-chat":
        print("Using baichuan2-13b-chat")
        openai.api_base = "http://124.16.138.150:8222/v1"
        openai.api_key = "xxx"
    elif model_name == "gpt-3.5-turbo":
        print("Using gpt-3.5-turbo")
        # openai.api_base = "https://one-api.ponte.top/v1"
        # openai.api_key = "sk-9y7d4TtOJO3xzHXn6dCa9fE02d18436d86Be540a00DcD1F8"
        openai.api_base = "http://47.245.109.131:5555/v1"
        openai.api_key = "sk-Y5UPxxoh10M9h50RBe8e70EfEc484556A80a8717623aEb2f"
    else:
        raise ValueError("Model name not supported")


def is_supported_by_gpt(model_name, question, response, ground_truth):
    # Construct the prompt for GPT
    prompt = f"Does the following response support the answer to the question? \nQuestion: {question}\n" \
             f"Response: {response}\n" \
             f"Answer: {ground_truth}\n" \
             f"Just answer 'yes' or 'no'.\n"


    # Get the response from the model
    # if 'yes' or 'no' not in response, continue to try
    gpt_response = ''

    # retry 10 times
    for i in range(10):
        try:
            gpt_response = get_response_llm(model_name, prompt)
        except Exception as e:
            print(f"Error in getting response from GPT: {e}")
            gpt_response = ''
        if 'yes' in gpt_response.lower() or 'no' in gpt_response.lower():
            break

    if gpt_response == '':
        print(f"Error in getting response from GPT, return False")
        return False

    # Consider the response affirmative if GPT says 'yes' or similar affirmative answer
    return "yes" in gpt_response.lower()


@retry(stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, max=10))
def get_response_llm(model_name, text, filter_words=None):
    # print(f"Getting response from {model_name} for text: {text}")
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user",
             "content": text},
        ],
        max_tokens=10,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=filter_words,
        stream=False,
    )


    # print(text)
    # print(f'GPT response: {completion.choices[0].message.content.strip()}')
    # print("\n")

    resp = completion.choices[0].message.content.strip().replace("\n", " ")
    # memory cleanup
    del completion
    return resp


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

    def exact_match_score(example, mis_answer_dict, model_name='gpt-3.5-turbo'):
        question = example['question']
        if mis_answer_dict != {}:
            ground_truths = mis_answer_dict[question]
        else:
            ground_truths = example['answers']
        assert type(ground_truths) == list, f'ground_truths is not a list, id:{example["id"]}, ground_truth:{ground_truths}'
        prediction = example['response']
        example['exact_match'] = 0
        if not prediction:
            print(f'no prediction for qid {example["id"]}, {example["question"]}')
            return example
        for ground_truth in ground_truths:
            if _normalize_answer(ground_truth) in _normalize_answer(prediction):
                if is_supported_by_gpt(model_name, question, prediction, ground_truth):
                    example['exact_match'] = 1
                    break
        return example

    get_openai_api('gpt-3.5-turbo')
    fn_args = {'mis_answer_dict': mis_answer_dict, 'model_name': 'gpt-3.5-turbo-0613'}
    predictions = predictions.map(exact_match_score, fn_kwargs=fn_args, num_proc=4)
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



def calculate_mis_em_llm(retrieval_file, mis_answer_path):
    mis_answer_dict = get_mis_answer(mis_answer_path)
    dataset = datasets.load_dataset("json", data_files=retrieval_file)["train"]
    print('evaluate QA EM...')
    prediction = evaluate(dataset, mis_answer_dict)
    EM = sum(prediction['exact_match']) / len(prediction['exact_match'])
    return EM


def calculate_right_em_llm(retrieval_file):
    dataset = datasets.load_dataset("json", data_files=retrieval_file)["train"]
    print('evaluate QA EM...')
    mis_answer_dict = {}
    prediction = evaluate(dataset, mis_answer_dict)
    EM = sum(prediction['exact_match']) / len(prediction['exact_match'])
    return EM


if __name__ == '__main__':
    base_path = '../../data_v2/misinfo_data/DPR/mis_passage_processed'
    dataset_names = ['nq']
    output_name = 'retrieval_em_llm.tsv'
    # get all the retrieval file, end with .jsonl
    for dataset_name in dataset_names:
        file_path = os.path.join(base_path, dataset_name)
        if not os.path.exists(os.path.join(file_path, 'results')):
            os.makedirs(os.path.join(file_path, 'results'))
        with open(os.path.join(file_path, 'results', output_name), 'a') as f:
            f.write('misinfo_type\tfile_name\tEM\n')
            for file in os.listdir(file_path):
                if file.endswith('.jsonl'):
                    retrieval_file = os.path.join(file_path, file)
                    print(f'Calculating mis EM for {retrieval_file}')
                    EM = calculate_mis_em_llm(retrieval_file, base_path)
                    f.write(f'misinfo\t{file}\t{EM}\n')
    for dataset_name in dataset_names:
        file_path = os.path.join(base_path, dataset_name)
        with open(os.path.join(file_path, 'results', output_name), 'a') as f:
            f.write('rightinfo_type\tfile_name\tEM\n')
            for file in os.listdir(file_path):
                if file.endswith('.jsonl'):
                    retrieval_file = os.path.join(file_path, file)
                    print(f'Calculating right EM for {retrieval_file}')
                    EM = calculate_right_em_llm(retrieval_file)
                    f.write(f'right\t{file}\t{EM}\n')
    print('Done')










