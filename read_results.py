import csv
from pprint import pprint
import pandas as pd

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

def get_data(task):
    return create_data_frame(read_results_file(task))