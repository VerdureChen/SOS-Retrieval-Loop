#convert psgs_w100.tsv to jsonl format
#add generated docs meanwhile

import json
from tqdm import tqdm
import ast

def convert_psgs_to_jsonl(psgs_file, output_file):
    with open(psgs_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out:
        for i, line in tqdm(enumerate(f), desc='Converting psgs to jsonl', total=21015325):
            if i == 0:
                continue
            line = line.strip().split('\t')
            doc_id = line[0]
            text = line[1][1:-1]
            title = line[2]
            new_text = title + '\n' + text
            doc = {'id': doc_id, 'contents': new_text}
            # force_ascii = False
            out.write(json.dumps(doc, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    psgs_file = 'raw_data/DPR/psgs_w100.tsv'
    output_file = 'input_data/DPR/psgs_w100.jsonl'
    convert_psgs_to_jsonl(psgs_file, output_file)
