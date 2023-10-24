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
    'deberta-v2-xxlarge',
    
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

results = {}
deberta_results = {}
start_dir = '/net/nfs.cirrascale/allennlp/jacobm/lora-investigation/' # 'llama2-7b/sciq/lora_2521/'
for elem in os.walk(start_dir):
    if os.path.isfile(elem[0] + '/metrics.json'):
        dir = elem[0]
        dir_tokens = dir.split('/')
        print(dir_tokens)
        if dir_tokens[6] == 'deberta-v2-xxlarge':
            learning_rate = dir_tokens[-1]
            seed = dir_tokens[-2].split('_')[-1]
            if 'lora' in dir_tokens[-3]:
                method = dir_tokens[-3]
                rank = dir_tokens[-3].split('_')[-1]
            else:
                method = 'full finetuning'
                rank = '-1'
            task = dir_tokens[-4]
            model = dir_tokens[-5]
            with open(dir + '/metrics.json') as f:
                data = json.load(f)
            if task not in results:
                results[task] = {}
            if model not in results[task]:
                results[task][model] = {}
            if method not in results[task][model]:
                results[task][model][method] = {}
            if learning_rate not in results[task][model][method]:
                results[task][model][method][learning_rate] = {}
            if seed not in results[task][model][method][learning_rate]:
                results[task][model][method][learning_rate][seed] = data['eval_accuracy']
        else:
            seed = dir_tokens[-1].split('_')[-1]
            if 'lora' in dir_tokens[-2]:
                method = dir_tokens[-2]
                rank = dir_tokens[-2].split('_')[-1]
            else:
                method = 'full finetuning'
                rank = '-1'
            task = dir_tokens[-3]
            model = dir_tokens[-4]
            # print('seed: ' + seed)
            # print('method: ' + method)
            # print('rank: ' + rank)
            # print('task: ' + task)
            # print('model: ' + model)
            with open(dir + '/metrics.json') as f:
                data = json.load(f)
            if task not in results:
                results[task] = {}
            if model not in results[task]:
                results[task][model] = {}
            if method not in results[task][model]:
                results[task][model][method] = {'1e-4': {}}
            if seed not in results[task][model][method]['1e-4']:
                results[task][model][method]['1e-4'][seed] = data['eval_accuracy']
pprint(results)
