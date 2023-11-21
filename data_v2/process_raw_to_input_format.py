import datasets

#process raw data to input format
raw_data_paths = [
    'raw_data/DPR/nq-test.jsonl',
    'raw_data/DPR/tqa-test.jsonl',
    'raw_data/DPR/webq-test.jsonl',
    'raw_data/DPR/pop-test.jsonl',
]

def process_raw_data(raw_data_path):
    output_path = raw_data_path.replace('raw_data', 'input_data')
    raw_data = datasets.load_dataset('json', data_files=raw_data_path)['train']
    def reassign_id_with_idx(example, idx):
        example['id'] = str(idx)
        if 'query' in example:
            # keep the special tokens like è
            example['question'] = example['query']
        if 'possible_answers' in example:
            example['answer'] = example['possible_answers']
        return example
    raw_data = raw_data.map(reassign_id_with_idx, with_indices=True)
    # keep id, question, answer columns
    raw_data = raw_data.select_columns(['id', 'question', 'answer'])
    # keep the special tokens like è
    raw_data.to_json(output_path, force_ascii=False)


if __name__ == '__main__':
    for raw_data_path in raw_data_paths:
        process_raw_data(raw_data_path)
