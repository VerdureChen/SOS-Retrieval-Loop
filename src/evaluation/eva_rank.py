import datasets
import json
from eva_generate_em import evaluate as evaluate_em
from datasets.utils.logging import disable_progress_bar
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


def get_first_answer_rank(example, cut_off=100):
    rank = cut_off
    for i, context in enumerate(example['contexts']):
        if context['has_answer'] == True:
            rank = min(rank, i+1)
            break

    example['first_answer_rank'] = rank
    return example


def get_first_answer_rank_dataset(dataset, cut_off=100):
    map_fn_kwargs = {'cut_off': cut_off}
    dataset = dataset.map(get_first_answer_rank, fn_kwargs=map_fn_kwargs, num_proc=4)
    return dataset


def get_first_human_answer_rank(example, cut_off=100):
    rank = cut_off
    for i, context in enumerate(example['contexts']):
        if context['has_answer'] == True and context['docid'].isdigit():
            rank = min(rank, i+1)
            break

    example['first_human_answer_rank'] = rank
    return example


def get_first_human_answer_rank_dataset(dataset, cut_off=100):
    map_fn_kwargs = {'cut_off': cut_off}
    dataset = dataset.map(get_first_human_answer_rank, fn_kwargs=map_fn_kwargs, num_proc=4)
    return dataset


def compute_true_wrong_first_answer_rank(result_file, generated_file, cut_off=100):
    result_dataset = read_result_file(result_file)
    generated_dataset = read_generated_file(generated_file)
    result_dataset = get_first_answer_rank_dataset(result_dataset, cut_off)
    result_dataset = get_first_human_answer_rank_dataset(result_dataset, cut_off)

    assert result_dataset['id'] == generated_dataset['id']
    em = generated_dataset['exact_match']
    first_answer_rank = result_dataset['first_answer_rank']
    first_human_answer_rank = result_dataset['first_human_answer_rank']

    # compute avg answer rank and avg human answer rank, grouped by exact_match (0 or 1)
    avg_answer_rank = {}
    avg_human_answer_rank = {}
    for i in range(len(em)):
        if em[i] not in avg_answer_rank:
            avg_answer_rank[em[i]] = []
            avg_human_answer_rank[em[i]] = []
        avg_answer_rank[em[i]].append(first_answer_rank[i])
        avg_human_answer_rank[em[i]].append(first_human_answer_rank[i])


    # compute accuracy@5
    accuracy_at_5 = {}
    for i in avg_answer_rank:
        accuracy_at_5[i] = round(sum([1 for rank in avg_answer_rank[i] if rank <= 5]) / len(avg_answer_rank[i]),4)
    accuracy_at_3 = {}
    for i in avg_answer_rank:
        accuracy_at_3[i] = round(sum([1 for rank in avg_answer_rank[i] if rank <= 3]) / len(avg_answer_rank[i]),4)
    accuracy_at_1 = {}
    for i in avg_answer_rank:
        accuracy_at_1[i] = round(sum([1 for rank in avg_answer_rank[i] if rank <= 1]) / len(avg_answer_rank[i]),4)

    for i in avg_answer_rank:
        avg_answer_rank[i] = round(sum(avg_answer_rank[i]) / len(avg_answer_rank[i]),4)
    for i in avg_human_answer_rank:
        avg_human_answer_rank[i] = round(sum(avg_human_answer_rank[i]) / len(avg_human_answer_rank[i]),4)
    return avg_answer_rank, avg_human_answer_rank, accuracy_at_5, accuracy_at_3, accuracy_at_1, em



def compute_change_rank(result_files, generated_files, cut_off=100):
    em = []

    for result_file, generated_file in zip(result_files, generated_files):
        _, _, _, _, _, em_i = compute_true_wrong_first_answer_rank(result_file, generated_file, cut_off)
        em.append(em_i)

    # compute em: 0->1, 1->0, 0->0, 1->1
    # em format:[[0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]], compute the item change from 0 to 1 of each item

    change_em = {}
    change_index = {}
    for i in range(len(em[0])):
        if em[0][i] == 0 and em[1][i] == 1:
            change_em['0->1'] = change_em.get('0->1', 0) + 1
            change_index['0->1'] = change_index.get('0->1', []) + [i]
        elif em[0][i] == 1 and em[1][i] == 0:
            change_em['1->0'] = change_em.get('1->0', 0) + 1
            change_index['1->0'] = change_index.get('1->0', []) + [i]
        elif em[0][i] == 0 and em[1][i] == 0:
            change_em['0->0'] = change_em.get('0->0', 0) + 1
        elif em[0][i] == 1 and em[1][i] == 1:
            change_em['1->1'] = change_em.get('1->1', 0) + 1
        else:
            print('error')
    # check dict key exist
    for key in ['0->1', '1->0', '0->0', '1->1']:
        if key not in change_em:
            change_em[key] = 0
    for key in ['0->1', '1->0']:
        if key not in change_index:
            change_index[key] = []
    return change_em, change_index



