import argparse
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import openai
import os
import datasets


def get_openai_api(model_name):
    if model_name == "Qwen":
        print("Using Qwen")
        openai.api_base = "http://0.0.0.0:8111/v1"
        openai.api_key = "xxx"
    elif model_name == "Llama":
        print("Using Llama")
        openai.api_base = "http://0.0.0.0:8223/v1"
        openai.api_key = "xxx"
    elif model_name == "chatglm3":
        print("Using chatglm3")
        openai.api_base = "http://0.0.0.0:8113/v1"
        openai.api_key = "xxx"
    elif model_name == "baichuan2-13b-chat":
        print("Using baichuan2-13b-chat")
        openai.api_base = "http://0.0.0.0:8222/v1"
        openai.api_key = "xxx"
    elif model_name == "gpt-3.5-turbo":
        print("Using gpt-3.5-turbo")
        openai.api_base = "http://47.245.109.131:5555/v1"
        openai.api_key = "sk-Y5UPxxoh10M9h50RBe8e70EfEc484556A80a8717623aEb2f"
    else:
        raise ValueError("Model name not supported")

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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
def get_response_llm(model_name, text, filter_words=None):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user",
             "content": text},
        ],
        max_tokens=128,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=filter_words,
        stream=False,
    )


    # print(text)
    # print(completion.choices[0].message.content)
    # print("\n")

    resp = completion.choices[0].message.content.strip().replace("\n", " ")
    # memory cleanup
    del completion
    return resp


if __name__ == '__main__':
    config = get_args()
    model_name = config["model_name"]
    question_file_path = config["question_file_path"]
    output_file_path = config["output_file_path"]
    print(config)

    get_openai_api(model_name)
    question_dataset = datasets.load_dataset("json", data_files=question_file_path)["train"]

    def get_response(example):
        question = example["question"]
        prompt = f"Provide a background document in 100 words according to your knowledge from Wikipedia to answer the given question."\
                 f"\n\nQuestion:{question} \n\nBackground Document:"
        filter_words = [" Human:", " AI:", "sorry", " Sorry"]
        # if question contains filter_words, remove the word from filter_words
        for filter_word in filter_words:
            if filter_word.strip().lower() in question.lower():
                # remove all instances of filter_words
                filter_words = [word for word in filter_words if word != filter_word]

        response = get_response_llm(model_name, prompt.format(question=question), filter_words=filter_words)
        while response.strip() == "" \
                or len(response.split(" ")) < 5 \
                or [word for word in filter_words if word in response] != []:
            response = get_response_llm(model_name, prompt.format(question=question), filter_words=filter_words)
        # if response > 100 words, truncate to 100 words
        response = " ".join(response.split(" ")[:100])
        example["response"] = response
        return example

    # question_dataset = question_dataset.map(get_response, num_proc=4)
    # question_dataset.to_json(output_file_path, force_ascii=False)

    shard_size = 500
    num_shards = len(question_dataset) // shard_size

    shard_names = [f"{output_file_path[:-6]}_shard_{i}.json" for i in range(num_shards)]
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
            shard_dataset = shard_dataset.map(get_response, num_proc=4)
            shard_dataset.to_json(shard_name, force_ascii=False)
        else:
            print('-' * 50)
            print(f"Skipping shard {shard_name} of index {shard_names.index(shard_name)}")

    # combine shards
    print('-' * 50)
    print("Combining shards")
    print('-' * 50)
    combined_dataset = datasets.load_dataset("json", data_files=shard_names)["train"]

    combined_dataset = combined_dataset.sort("id")
    combined_dataset.to_json(output_file_path, force_ascii=False)

    # delete shards
    print('-' * 50)
    print("Deleting shards")
    print('-' * 50)
    for shard_name in shard_names:
        os.remove(shard_name)
    print("Done")




