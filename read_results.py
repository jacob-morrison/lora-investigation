import csv
from pprint import pprint
import pandas as pd
import json

def read_results_file(task):
    data = {}
    with open('results/' + task + '.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['metrics_eval_accuracy'] == '':
                continue
            accuracy = float(row['metrics_eval_accuracy'])
            if row['env_MODEL'] not in data:
                data[row['env_MODEL']] = {}
            if row['env_METHOD'] == 'LoRA':
                method = 'LoRA ' + row['env_RANK']
            else:
                method = 'Full Finetuning'
            if method not in data[row['env_MODEL']]:
                data[row['env_MODEL']][method] = {}
            if row['env_LEARNING_RATE'] not in data[row['env_MODEL']][method]:
                data[row['env_MODEL']][method][row['env_LEARNING_RATE']] = {}
            if row['env_SEED'] not in data[row['env_MODEL']][method][row['env_LEARNING_RATE']]:
                data[row['env_MODEL']][method][row['env_LEARNING_RATE']][row['env_SEED']] = accuracy
            elif accuracy > data[row['env_MODEL']][method][row['env_LEARNING_RATE']][row['env_SEED']]:
                data[row['env_MODEL']][method][row['env_LEARNING_RATE']][row['env_SEED']] = accuracy
            # else:
                # print('duplicate!!')
                # print(row['task_name'])
                # print(accuracy)
                # print(data[row['env_MODEL']][method][row['env_LEARNING_RATE']])
    return data

def create_data_frame(data):
    max_scores = {}

    blobs = []
    for model in data:
        for method in data[model]:
            for learning_rate in data[model][method]:
                for seed in data[model][method][learning_rate]:
                    blobs.append({
                        'model': model,
                        'method': method.split()[0] if 'LoRA' in method else method,
                        'rank': int(method.split()[1]) if 'LoRA' in method else -1,
                        'method and rank': method,
                        'learning rate': float(learning_rate),
                        'seed': seed,
                        'accuracy': float(data[model][method][learning_rate][seed])
                    })

                    if model not in max_scores:
                        max_scores[model] = {}
                    if method not in max_scores[model]:
                        max_scores[model][method] = {
                            'best score': float(data[model][method][learning_rate][seed]),
                            'best learning rate': learning_rate, # best
                            'lowest learning rate': learning_rate, # lowest
                            'highest learning rate': learning_rate # highest
                        }
                    if float(data[model][method][learning_rate][seed]) > max_scores[model][method]['best score']:
                        # max_scores[model][method] = (float(data[model][method][learning_rate][seed]), learning_rate)
                        max_scores[model][method]['best score'] = float(data[model][method][learning_rate][seed])
                        max_scores[model][method]['best learning rate'] = learning_rate
                    if float(learning_rate) < float(max_scores[model][method]['lowest learning rate']):
                        max_scores[model][method]['lowest learning rate'] = learning_rate
                    if float(learning_rate) > float(max_scores[model][method]['highest learning rate']):
                        max_scores[model][method]['highest learning rate'] = learning_rate


    pprint(max_scores)
    print()

    for model in max_scores:
        for method in max_scores[model]:
            if max_scores[model][method]['best learning rate'] == max_scores[model][method]['lowest learning rate']: 
                print('Best matches lowest LR:')
                print(model)
                print(method)
                print(max_scores[model][method])
                print()
            elif max_scores[model][method]['best learning rate'] == max_scores[model][method]['highest learning rate']:
                print('Best matches highest LR:')
                print(model)
                print(method)
                print(max_scores[model][method])
                print()

    # flatten by building a list of maps, model + method + rank (if applicable) + LR + seed
    return pd.DataFrame(blobs), max_scores

def transform_lora_name(method):
    if method == 'full finetuning':
        return 'Full Finetuning'
    if '_' in method:
        rank = method.split('_')[-1]
        if rank == 'Full Finetuning':
            return rank
        return 'LoRA ' + rank
    else:
        return method

with open('results/nfs-results.json') as f:
    nfs_data = json.load(f)
reformatted_data = {}
for task in nfs_data:
    reformatted_data[task] = {}
    for model in nfs_data[task]:
        reformatted_data[task][model] = {}
        for method in nfs_data[task][model]:
            reformatted_data[task][model][transform_lora_name(method)] = {}
            for seed in nfs_data[task][model][method]:
                reformatted_data[task][model][transform_lora_name(method)][seed] = nfs_data[task][model][method][seed]

pprint(reformatted_data)

def get_data(task):
    results = read_results_file(task)
    if task == 'case-hold':
        for model in reformatted_data['case-hold']:
            results[model] = reformatted_data['case-hold'][model]
    if task == 'sciq':
        for model in reformatted_data['sciq']:
            results[model] = reformatted_data['sciq'][model]
    return create_data_frame(results)

tasks = [
    'case-hold',
    'qnli',
    'arc-easy',
    'arc-challenge',
    'sciq',
    'mnli',
    'hellaswag',
    'yelp',
    'piqa',
    'mathqa',

    # 'squad',
]

methods = [
    'Full Finetuning',
    'LoRA 1',
    'LoRA 2',
    'LoRA 4',
    'LoRA 8',
    'LoRA 16',
    'LoRA 32',
    'LoRA 64',
]

models = {
    ### encoder only ###
    'microsoft/deberta-v3-xsmall': 1,
    'microsoft/deberta-v3-large': 2,
    'microsoft/deberta-v2-xxlarge': 4,

    ### decoder only ###
    'gpt2': 1,
    'gpt2-large': 4,
    'llama2-7b': 8, # probably use llama 2 instead?

    ### encoder/decoder ###
    ### single task ###
    'google/t5-small-lm-adapt': 1,
    'google/t5-large-lm-adapt': 4,
    't5-xxl': 8,

    ### multi task ###
    'jacobmorrison/tk-instruct-small-lora-experiments': 1,
    'jacobmorrison/tk-instruct-large-lora-experiments': 4,
    'jacobmorrison/tk-instruct-xxl-lora-experiments': 8,
}

LoRA_ranks = {
    'microsoft/deberta-v3-xsmall': 3843, # 769, 
    'microsoft/deberta-v3-large': 4426,
    'microsoft/deberta-v2-xxlarge': 5314,

    'gpt2': 3376,
    'gpt2-large': 4200,
    'llama2-7b': 12603, # probably use llama 2 instead?

    'google/t5-small-lm-adapt': 1414,
    'google/t5-large-lm-adapt': 2548,
    't5-xxl': 8,

    'jacobmorrison/tk-instruct-small-lora-experiments': 1413,
    'jacobmorrison/tk-instruct-large-lora-experiments': 2548,
    'jacobmorrison/tk-instruct-xxl-lora-experiments': 8,

}

model_specific_lora_ranks = {}

coefficients = [
    0.2,
    0.4,
    0.6,
    0.8
]
    
from math import ceil

for model in LoRA_ranks:
    model_specific_lora_ranks[model] = []
    if LoRA_ranks[model] != 1:
        for coefficient in coefficients:
            model_specific_lora_ranks[model].append('lora_' + str(int(ceil(coefficient * LoRA_ranks[model]))))
        model_specific_lora_ranks[model].append('lora_' + str(int(LoRA_ranks[model])))


max_scores = {}
for task in tasks:
    _, task_scores = get_data(task)
    max_scores[task] = task_scores

best_scores = {}
for task in max_scores:
    for model in max_scores[task]:
        if model not in best_scores:
            best_scores[model] = {}
        for method in max_scores[task][model]:
            if method not in best_scores[model]:
                best_scores[model][method] = {}
            best_scores[model][method][task] = max_scores[task][model][method]['best score']

pprint(best_scores)

# Model	Method	CaseHOLD	QNLI	ARC Easy	ARC Challenge	SciQ	MNLI	HellaSwag	Yelp	PIQA	MathQA	SQuAD
with open('results/combined-results.csv', 'w') as f:
    f.write('Model,Method,CaseHOLD,QNLI,ARC Easy,ARC Challenge,SciQ,MNLI,HellaSwag,Yelp,PIQA,MathQA,SQuAD\n')
    for model in models:
        for method in methods + model_specific_lora_ranks[model]:
            f.write(model + ',' + transform_lora_name(method))
            for task in tasks:
                if model not in best_scores or \
                    (transform_lora_name(method) not in best_scores[model] and transform_lora_name(method) in methods) or \
                        transform_lora_name(method) not in best_scores[model] or \
                        task not in best_scores[model][transform_lora_name(method)]:
                    f.write(',0.0')
                else:
                    f.write(',' + str(best_scores[model][transform_lora_name(method)][task]))
            f.write('\n')