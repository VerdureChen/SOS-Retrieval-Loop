#rewrite config files before running a loop
#read retrieval, rerank, generate, post_process and indexing config

import os
import sys
import argparse
import json


#read config files

def read_config(template_path):
    with open(template_path, 'r') as f:
        config = json.load(f)
    return config


def get_args():

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='argument-parser')

    group.add_argument('--stage', type=str, required=True, help='stage name: retrieval, rerank, generate, indexing')

    group.add_argument('--output_dir', type=str, required=True, help='config output dir')

    group.add_argument('--method', type=str, required=True, help='method name')

    group.add_argument('--data_name', type=str, required=True, help='data name')

    group.add_argument('--loop', type=int, default=0, help='loop number')

    group.add_argument('--total_config', type=str, default=None, help='total config file')

    # Accept a JSON string from the command line to override config settings
    group.add_argument('--overrides', type=str, help='JSON string to override config values')


    args = parser.parse_args()

    if args.overrides:
        print(f"Overriding config with: {args.overrides}")
        # {"index_name": "bm25_test_index", "index_exists": false}
        try:
            args.overrides = json.loads(args.overrides)
        except json.JSONDecodeError:
            print("Overrides must be a valid JSON string")
            sys.exit(1)

    return args

def get_template_path(stage_name, method_name, data_name):
    if stage_name == 'retrieval':
        template_path = f'retrieve_configs/{method_name}-config-{data_name}.json'
    elif stage_name == 'rerank':
        template_path = f'rerank_configs/{method_name}-config-{data_name}.json'
    elif stage_name == 'generate':
        template_path = f'rag_configs/{method_name}-config-{data_name}.json'
    elif stage_name == 'indexing':
        template_path = f'index_configs/{method_name}-config-{data_name}.json'
    elif stage_name == 'post_process':
        template_path = f'process_configs/template_config.json'
    else:
        raise ValueError('stage name error')
    return template_path


def rewrite_config(stage_name, method_name, data_name, loop, output_dir, total_config, overrides):
    template_path = get_template_path(stage_name, method_name, data_name)
    config = read_config(template_path)
    if total_config is not None:
        running_config = read_config(total_config)[stage_name]

        # for any key in config_template, if it is in running_config, use the value in running_config
        # otherwise, use the value in config_template
        for key in config:
            if key in running_config:
                config[key] = running_config[key]


    # if overrides is not None:
    # Apply overrides from the command line
    if overrides:
        for key, value in overrides.items():
            config[key] = value

    with open(output_dir, 'w') as f:
        json.dump(config, f, indent=4)

    print(f'loop {loop} config file is saved in {output_dir}')
    print(f'stage: {stage_name}, method: {method_name}, data: {data_name}, loop: {loop}')
    print(f'config: {config}')


def main():
    args = get_args()
    rewrite_config(args.stage, args.method, args.data_name, args.loop, args.output_dir, args.total_config, args.overrides)


if __name__ == '__main__':
    main()

