import argparse
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import openai
import string
import re
import datasets
import sys
sys.path.append('../retrieval_loop')
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
sys.path.append('../llm_zero_generate')
from get_response_llm import get_openai_api
from eva_mis import evaluate
import random
# define global variables Mode
Mode = {
    "Mis_Answer": 0,
    "Mis_Passage": 0,
}
'''
misanswer prompt:
Generate a false answer to the given
question. It should be of short (less than
five words in general) and look plausible,
compared to the reference answer.
Question: Question
Reference Answers: Answer

mis_passage prompt:
Generate a background 
document in support of the 
answer to the given question.
Question: Question
Answers: Answer
'''
Prompt = {
    "Mis_Answer": "Generate a false answer to the given question. It should be of short (less than ten words in general) and look plausible, compared to the reference answer."
                 "\n\nQuestion:{question} \n\nReference Answers:{answers}, \n\nfalse answer:",
    "Mis_Passage": "Suppose you are generating realistic-looking claims for a quiz contest. You need to generate a background document less than 100 words in support of the answer to the given question. "
                   "Don't contain any word in the original answers in {ref_answers}. The background document must contain the following given Answers with their original form. "
                      "\n\nQuestion:{question} \n\nAnswers:{answers}, \n\nbackground document:",
}



def get_args():
    # get config_file_path, which is the path to the config file
    # config file formatted as a json file:
    # {
    #   "model_name": "Qwen",
    #   "question_file_path": "data/qa/qa.json",
    #   "output_file_path": "data/qa/qa_with_response.json"
    # }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


@retry(stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, max=10))
def get_response_llm(model_name, text, filter_words=None, n=1):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user",
             "content": text},
        ],
        max_tokens=128,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=filter_words,
        stream=False,
        n=n,
    )
    print(completion)
    # for choice in completion.choices:
    #     print(choice.message.content.strip().replace("\n", " "))
    # print("\n")
    # print(text)
    # if len(completion.choices) > 1, return the list of responses
    # else return the response
    if len(completion.choices) > 1:
        resp = [choice.message.content.strip().replace("\n", " ") for choice in completion.choices]
        # memory cleanup
        del completion
        return resp
    resp = completion.choices[0].message.content.strip().replace("\n", " ")
    # memory cleanup
    del completion
    return [resp]


def prepare_input(question, answers, Mode=None, ref_answers=None):
    # print(f'zero-shot:{Mode["Zero-shot"]}')
    # print(f'with-context:{Mode["With-context"]}')
    # print(f'mode:{mode}')
    if Mode["Mis_Answer"] == 1:
        prompt = Prompt["Mis_Answer"]
        prompt = prompt.format(question=question, answers=answers)
    elif Mode["Mis_Passage"] == 1:
        prompt = Prompt["Mis_Passage"]
        prompt = prompt.format(question=question, answers=answers, ref_answers=ref_answers)
    else:
        raise ValueError("Mode not supported")
    print(prompt)
    return prompt




def read_dataset(question_file_path, misinfo_type):
    print(f'misinfo_type:{misinfo_type}')
    print(misinfo_type == "mis_answer")
    if misinfo_type == "mis_answer":
        Mode["Mis_Answer"] = 1
        try:
            question_dataset = datasets.load_dataset("json", data_files=question_file_path)["train"]
        except:
            with open(question_file_path, "r", encoding='utf-8') as f:
                question_dataset = json.load(f)
            formatted_data = [
                {
                    "id": qid,
                    "question": details["question"],
                    "answers": details["answer"],
                }
                for qid, details in question_dataset.items()
            ]
            question_dataset = datasets.Dataset.from_list(formatted_data)
    elif misinfo_type == "mis_passage":
        Mode["Mis_Passage"] = 1
        try:
            question_dataset = datasets.load_dataset("json", data_files=question_file_path)["train"]
        except:
            with open(question_file_path, "r", encoding='utf-8') as f:
                question_dataset = json.load(f)
            formatted_data = [
                {
                    "id": qid,
                    "question": details["question"],
                    "ref_answers": details["ref_answers"],
                    "false_answer": details["false_answer"],
                }
                for qid, details in question_dataset.items()
            ]
            question_dataset = datasets.Dataset.from_list(formatted_data)
    else:
        raise ValueError("misinfo_type not supported")

    return question_dataset


def get_response(example, Mode=None, model_name=None):
    if Mode is None:
        Mode = {}
    question = example["question"]
    if Mode["Mis_Answer"] == 1:
        answers = example["answer"]
        ref_answers = example["answer"]
    elif Mode["Mis_Passage"] == 1:
        # randomly choose a false answer
        answers = random.choice(example["false_answer"])
        example["false_answer"] = [answers]
        ref_answers = example["answers"]
    else:
        raise ValueError("Mode not supported")
    input_text = prepare_input(question, answers, Mode, ref_answers)
    filter_words = [" Human:", " AI:", "sorry", " Sorry"]
    # add ref_answer to filter_words
    if Mode["Mis_Answer"] == 1:
        filter_words.extend(answers)
    elif Mode["Mis_Passage"] == 1:
        filter_words.extend(example["answers"])

    # if question contains filter_words, remove the word from filter_words
    for filter_word in filter_words:
        if Mode["Mis_Answer"] == 1:
            if filter_word.strip().lower() in question.lower():
                filter_words = [word for word in filter_words if word != filter_word]
        elif Mode["Mis_Passage"] == 1:
            if filter_word.strip().lower() in question.lower() or filter_word.strip().lower() in answers.lower():
                filter_words = [word for word in filter_words if word != filter_word]


    response = []

    flag = True

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

    def check_response(response):
        if len(response) == 0:
            flag = True
            return flag
        for item in response:
            if item.strip() == "" \
                    or len(item.split(" ")) < 1 \
                    or [word for word in filter_words if word.lower() in item.lower()] != [] \
                    or (Mode["Mis_Passage"] == 1 and _normalize_answer(answers) not in _normalize_answer(item)):
                flag = True
            else:
                flag = False
        return flag
    count = 0
    while check_response(response):
        if Mode["Mis_Answer"] == 1:
            response = get_response_llm(model_name, input_text, filter_words=[], n=5)
        elif Mode["Mis_Passage"] == 1:
            response = get_response_llm(model_name, input_text, filter_words=[])
        else:
            raise ValueError("Mode not supported")
        count += 1
        if count > 10:
            break
    # if response > 100 words, truncate to 10 words

    if Mode["Mis_Answer"] == 1:
        example["false_answer"] = response
        for i, item in enumerate(response):
            if len(item.split(" ")) > 10:
                response[i] = " ".join(item.split(" ")[:10])
    elif Mode["Mis_Passage"] == 1:
        example["response"] = response[0]
        for i, item in enumerate(response):
            if len(item.split(" ")) > 100:
                response[i] = " ".join(item.split(" ")[:100])
    return example


if __name__ == '__main__':
    config = get_args()
    model_name = config["model_name"]
    question_file_path = config["question_file_path"]
    output_file_path = config["output_file_path"]
    misinfo_type = config["misinfo_type"]
    api_key = config["api-key"]
    api_base = config["api-base"]
    print(config)
    # config = {
    #     "model_name": "Qwen",
    #     "question_file_path": "data/qa/qa.json",
    #     "output_file_path": "data/qa/qa_with_response.json",
    #     "misinfo_type": "mis_answer"
    # }

    # set openai api
    get_openai_api(model_name, api_base, api_key)
    if model_name == "gpt-3.5-turbo":
        model_name = "gpt-3.5-turbo-0613"
    # read dataset
    question_dataset = read_dataset(question_file_path, misinfo_type)
    # set kwargs
    map_fn_kwargs = {"Mode": Mode,
                     'model_name': model_name}
    # check mode
    assert Mode["Mis_Answer"] + Mode["Mis_Passage"] == 1, "Only one mode can be set to 1"
    print(f"Number of examples: {len(question_dataset)}")



    # question_dataset = question_dataset.map(get_response, num_proc=4, fn_kwargs=map_fn_kwargs)
    # question_dataset.to_json(output_file_path, force_ascii=False)

    shard_size = 50
    num_shards = len(question_dataset) // shard_size
    if num_shards == 0:
        num_shards = 1

    shard_names = [f"{output_file_path}_shard_{i}.json" for i in range(num_shards)]
    existing_shards = [shard_name for shard_name in shard_names if os.path.exists(shard_name)]
    print(f"Existing shards: {existing_shards}")
    print(f"Shard names: {shard_names}")

    for shard_name in shard_names:
        if shard_name not in existing_shards:
            print('-' * 50)
            print(f"Creating shard {shard_name}")
            print(f"Shard index: {shard_names.index(shard_name)}")
            print('-' * 50)
            shard_dataset = question_dataset.shard(num_shards, index=shard_names.index(shard_name))
            shard_dataset = shard_dataset.map(get_response, num_proc=8, fn_kwargs=map_fn_kwargs)
            # only keep 'id', 'question', 'answers', 'response'
            # shard_dataset = shard_dataset.select_columns(['id', 'question', 'answers', 'response'])
            shard_dataset.to_json(shard_name, force_ascii=False)
        else:
            print('-' * 50)
            print(f"Skipping shard {shard_name} of index {shard_names.index(shard_name)}")

    # combine shards
    print('-' * 50)
    print("Combining shards")
    print('-' * 50)
    combined_dataset = datasets.load_dataset("json", data_files=shard_names)["train"]
    # 先使用map方法创建一个临时的整数类型的id字段，例如命名为"id_int"
    combined_dataset = combined_dataset.map(lambda example: {"id_int": int(example["id"])})

    # 根据这个新字段排序
    combined_dataset = combined_dataset.sort("id_int")

    # 删除这个新字段
    combined_dataset = combined_dataset.remove_columns(["id_int"])

    #rename answer to answers if misinfo_type is mis_answer
    if misinfo_type == "mis_answer":
        combined_dataset = combined_dataset.rename_column("answer", "answers")

    # evaluate
    print('-' * 50)
    print("Evaluating")
    print('-' * 50)
    if misinfo_type != "mis_answer":
        prediction = evaluate(combined_dataset)
        EM_true = sum(prediction['exact_match_true']) / len(prediction['exact_match_true'])
        EM_false = sum(prediction['exact_match_false']) / len(prediction['exact_match_false'])
        print(f"EM_true: {EM_true}")
        print(f"EM_false: {EM_false}")

        #write metrics to file
        metrics_file_path = os.path.join(os.path.dirname(output_file_path), f"{model_name}_rag_metrics.json")
        with open(metrics_file_path, "a") as f:
            f.write(f"{output_file_path}_EM_true: {EM_true}\n")
            f.write(f"{output_file_path}_EM_false: {EM_false}\n")


    combined_dataset.to_json(output_file_path, force_ascii=False)

    # delete shards
    print('-' * 50)
    print("Deleting shards")
    print('-' * 50)
    for shard_name in shard_names:
        os.remove(shard_name)
    print("Done")




