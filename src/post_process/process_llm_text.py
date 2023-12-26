#process the llm text into a format that can be used by the model

# input file format:
# {"id":"0","question":"who got the first nobel prize in physics?","answer":["Wilhelm Conrad Röntgen"],"response":"Sure, I'd be happy to help! The first Nobel Prize in Physics was awarded to Wilhelm Conrad Röntgen in 1901 for his discovery of X-rays. According to Wikipedia, Röntgen was a German physicist who made this groundbreaking discovery while working at the University of Würzburg. He discovered X-rays while experimenting with cathode rays and observed that they could penetrate human tissue and create images of internal structures. This revolutionary discovery has had a profound impact on medicine and science, leading to countless advancements in medical imag"}
# {"id":"1","question":"when is the next deadpool movie being released?","answer":["May 18 , 2018"],"response":"Sure, I'd be happy to help! According to Wikipedia, the next Deadpool movie, titled \"Deadpool 3,\" is currently in development and is set to be released on September 6, 2024. The film will once again star Ryan Reynolds as the titular character and will be directed by Shawn Levy, who also directed the previous installment, \"Deadpool 2.\" The movie is expected to continue the adventures of the Merc with a Mouth and will feature a mix of action, comedy, and pop culture references. However, it's important to"}
# ...

# other input arguments:
# --output_dir: the directory to save the output files
# --loop_num: the number of loop of the input file
# --gen_model_name: the name of the model used to generate the response
# --query_set_name: the name of the query set used to generate the response


import argparse
import json
import os
import sys
import datasets
import time


def read_input_dataset(input_file):
    input_dataset = datasets.load_dataset('json', data_files=input_file)['train']
    return input_dataset


def process_dataset(input_dataset, output_dir, loop_num, gen_model_name, query_set_name, from_method):
    # map the input dataset to the output dataset
    func_args = {
        'loop_num': loop_num,
        'gen_model_name': gen_model_name,
        'query_set_name': query_set_name,
        'from_method': from_method
    }
    output_dataset = input_dataset.map(process_example, fn_kwargs=func_args)
    # save the output dataset
    output_dataset.to_json(output_dir, force_ascii=False)


def filter_response(response):
    # for text start with the following words, remove the text
    filter_whole_sentence = [
        "Sure, I'd be happy to help!",
        "Thank you for asking! I'd be happy to help answer your question.",
        "Sure, I can provide you with accurate and helpful information!",
        "Sure!",
        "Sure, I'd be happy to assist!",
        "Hello! I'm here to assist you with your question.",
        "Hello! I'd be happy to help you with your question.",
        "Hello! I'm here to assist you in a responsible and positive manner.",
        "Thank you for asking!",
        "I'd be happy to help answer your question.",
        "Sure thing!",
        "Certainly!",
        "Based on the provided context information,"
        ]

    filter_the_chunk_if_starts_with = [
        "According to Wikipedia, ",
        "Hello! I'd be happy to help you with your question."
        "I'd be happy to help",
        "I'd be happy to assist you with your question.",
        "However, I think there may be a mistake in the question.",
        "Based on my knowledge from Wikipedia,",
        "According to my knowledge from Wikipedia,",
        "According to my knowledge,",
        "As of my knowledge cutoff date in [[训练时间]],",
        "According to the latest data available,",
        "Based on the provided context information,",
        "Based on the context information provided,"
        ]

    filter_the_sentence_if_starts_with = [
        "Here is",
        "Here's my response",
        "Sure,",
        "Here's the background",
        "Here is the background",
        "Here's a background",
        "However, I would like to",
        "However, I must point out",
        "However, I want to",
        "Of course! I'd be happy",
        "Here is a 100-word background",
        "Here's a 100-word background",
        "As a helpful and respectful assistant,",
        "However, I noticed that",
        "I'd be happy to help",
        "I apologize",
        "As an AI language model,",
        "Sorry",
        "Here's my answer",
        "here's the answer",
        "here is a 100-word"
        ]


    for sentence in filter_whole_sentence:
        if response.startswith(sentence):
            response = response[len(sentence):].strip()
            break

    for chunk_start in filter_the_chunk_if_starts_with:
        if response.startswith(chunk_start):
            response = response[len(chunk_start):].strip()
            break

    for sentence_start in filter_the_sentence_if_starts_with:
        # find the end of the sentence in the response starts with the sentence_start
        # or appear a ':' before the end of the sentence
        end_of_sentence = len(response)
        for i in range(len(sentence_start), len(response)):
            if response[i] == ':':
                end_of_sentence = i + 1
                break
            if response[i] == '.':
                end_of_sentence = i + 1
                break
        # remove the sentence
        if response.startswith(sentence_start):
            response = response[end_of_sentence:]
            break

    return response.strip()




def process_example(example, loop_num, gen_model_name, query_set_name, from_method):
    # assign new id
    # get a timestamp, format: 20210801120000
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    example['id'] = f'{gen_model_name}_{query_set_name}_from_{from_method}_loop{loop_num}_{example["id"]}_{timestamp}'
    # filter the response
    example['response'] = filter_response(example['response'])
    return example


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default='config.json')
    parser.add_argument('--input_file', type=str, default='data/llm_text.json')
    parser.add_argument('--output_dir', type=str, default='data/llm_text_processed')
    parser.add_argument('--loop_num', type=int, default=0)
    parser.add_argument('--gen_model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--query_set_name', type=str, default='nq')
    parser.add_argument('--from_method', type=str, default='zreo-shot')
    args = parser.parse_args()
    # 读取 JSON 配置文件
    # json:
    # {
    #     "loop_num": 0,
    #     "gen_model_name": "gpt-3.5-turbo",
    #     "query_set_name": "nq",
    #     "output_dir": "data/llm_text_processed",
    #     "input_file": "data/llm_text.json"
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


def main():
    args = get_args()
    # read the input dataset
    input_dataset = read_input_dataset(args.input_file)
    # process the input dataset
    process_dataset(input_dataset, args.output_dir, args.loop_num, args.gen_model_name, args.query_set_name, args.from_method)


if __name__ == '__main__':
    main()

