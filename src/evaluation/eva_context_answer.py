import datasets
import json
from eva_generate_em import evaluate as evaluate_em
from datasets.utils.logging import disable_progress_bar
from collections import defaultdict
disable_progress_bar()

def read_result_file(file_name):
    with open(file_name, "r", encoding='utf-8') as f:
        question_dataset = json.load(f)
    formatted_data = [
        {
            "id": qid,
            "question": details["question"],
            "answers": details["answers"],
            "contexts": details["contexts"],
        }
        for qid, details in question_dataset.items()
    ]
    result_dataset = datasets.Dataset.from_list(formatted_data)

    return result_dataset


def read_generated_file(file_name):
    generated_dataset = datasets.load_dataset('json', data_files=file_name)['train']
    prediction = evaluate_em(generated_dataset) # format: for each sample, example['exact_match'] = 1 or 0

    return prediction


def get_top_answer_number(example, cut_off=100):
    count = 0
    for i, context in enumerate(example['contexts']):
        if context['has_answer'] == True:
            count += 1
        if i == cut_off - 1:
            break

    example['top_answer_number'] = count
    return example


def get_top_answer_number_dataset(dataset, cut_off=100):
    map_fn_kwargs = {'cut_off': cut_off}
    dataset = dataset.map(get_top_answer_number, fn_kwargs=map_fn_kwargs, num_proc=4)
    return dataset


def compute_top_answer_number(result_file, generated_file, cut_off=5):
    result_dataset = read_result_file(result_file)
    generated_dataset = read_generated_file(generated_file)
    result_dataset = get_top_answer_number_dataset(result_dataset, cut_off)
    assert result_dataset['id'] == generated_dataset['id']
    em = generated_dataset['exact_match']
    top_answer_number = result_dataset['top_answer_number']

    top_answer_number_dict = defaultdict(list)
    for i in range(len(em)):
        top_answer_number_dict[f'{cut_off}_{em[i]}'].append(top_answer_number[i])

    # print(top_answer_number_dict)
    # print(top_answer_number_dict)
    return top_answer_number_dict







