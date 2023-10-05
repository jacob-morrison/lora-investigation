import csv
from pprint import pprint
import os
import json

seeds = [
    '1',
    '2',
    '3',
]

models = [
    'llama-7b',
    't5-xxl',
    'tk-xxl',
    
    # 't5-xl',
    # 'tk-xl',
    # 'gpt2-xl',
    # 'deberta-v2-xl',
]

tasks = [
    'case-hold',
    'sciq',
    'squad',
]

start_dir = '/net/nfs.cirrascale/allennlp/jacobm/lora-investigation/' # 'llama2-7b/sciq/lora_2521/'
for elem in os.walk(start_dir):
    if os.path.isfile(elem[0] + '/metrics.json'):
        dir = elem[0]
        dir_tokens = dir.split('/')
        seed = dir_tokens[-1].split('_')[-1]
        if 'lora' in dir_tokens[-2]:
            method = 'lora'
            rank = dir_tokens[-2].split('_')[-1]
        else:
            method = 'full finetuning'
            rank = -1
        task = dir_tokens[-3]
        model = dir_tokens[-4]
        print('seed: ' + seed)
        print('method: ' + method)
        print('rank: ' + rank)
        print('task: ' + task)
        print('model: ' + model)
        with open(dir + '/metrics.json') as f:
            data = json.load(f)
        print(data)
        print()