import copy
import subprocess
import yaml
import random
from datetime import date
import time
import os
from math import ceil
from read_results import get_data


today = date.today().strftime("%m%d%Y")

# with open("beaker_configs/default_experiment.yaml", 'r') as f:
#     default_yaml = f.read()
# d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

def set_argument_value(arguments, name, value):
    if name not in arguments:
        raise ValueError(f"{name} not in arguments.")
    idx = arguments.index(name)
    assert not (isinstance(arguments[idx+1], str) and arguments[idx+1].startswith("-")) # make sure the next argument is the value
    arguments[idx+1] = value
    return arguments

# ---- run all experiments ---- #
seeds = [
    1,
    2,
    3,
    # 4,
    # 5,
]

experiments = [
    # 'case-hold',

    'qnli',
    'arc-easy',
    # 'arc-challenge',
    # 'sciq',
    # 'mnli',
    # 'hellaswag',
    # 'yelp',
    # 'piqa',
    # 'mathqa',


    # 'squad',
]

# TODO: do more learning rates if a good LR is on a boundary, or if none work at all
learning_rates = [
    # '1e-2',
    # '5e-3',

    # TODO: limit which we choose
    # '1e-3',
    # '5e-4',
    # '1e-4',
    # '5e-5',
    # '1e-5',
    # '5e-6',
    # '1e-6',
    # '5e-7',
    '1e-7'
]

models = {
    ### don't use long term ###
    # 'roberta-base': 1,
    # 'microsoft/deberta-base': 1,
    # 'roberta-large': 2,
    # 'microsoft/deberta-large': 2,
    # t5-v1_1s too, probably
    # 'google/t5-v1_1-small': 1,
    # 'google/t5-v1_1-base': 2,
    # 'google/t5-v1_1-large': 2,
    # 'google/t5-v1_1-xl': 4,
    # 'google/t5-v1_1-xxl': 8,

    # not using these anymore either
    # 'microsoft/deberta-v3-small': 1,
    # 'microsoft/deberta-v3-base': 1,
    # 'gpt2-medium': 2,
    # 'google/t5-base-lm-adapt': 2,
    # 'jacobmorrison/tk-instruct-base-lora-experiments': 2,
    # 'microsoft/deberta-v2-xlarge': 4,
    # 'gpt2-xl': 4,
    # '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/13B': 8,
    # 'decapoda-research/llama-7b-hf': 8,
    # 'jacobmorrison/tk-instruct-xl-lora-experiments': 4,
    # 'google/t5-xl-lm-adapt': 4,


    ### encoder only ###
    # 'microsoft/deberta-v3-xsmall': 1,
    # 'microsoft/deberta-v3-large': 2,
    # 'microsoft/deberta-v2-xxlarge': 4,

    ### decoder only ###
    # 'gpt2': 1,
    # 'gpt2-large': 4,
    # '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B': 8, # probably use llama 2 instead?

    ### encoder/decoder ###
    ### single task ###
    # 'google/t5-small-lm-adapt': 1,
    'google/t5-large-lm-adapt': 4,
    # 'google/t5-xxl-lm-adapt': 8,

    ### multi task ###
    # 'jacobmorrison/tk-instruct-small-lora-experiments': 1,
    # 'jacobmorrison/tk-instruct-large-lora-experiments': 4,
    # 'jacobmorrison/tk-instruct-xxl-lora-experiments': 8,
}
    
xl_models = {
    'gpt2-xl': 4,
    'microsoft/deberta-v2-xxlarge': 4,
    'google/t5-xl-lm-adapt': 4,
    'jacobmorrison/tk-instruct-xl-lora-experiments': 4,
}

LoRA_ranks = {
    # 'roberta-base': 3398,
    # 'roberta-large': 3626,

    # 'microsoft/deberta-base': 3776,
    # 'microsoft/deberta-large': 4133,

    # 'microsoft/deberta-v2-xlarge': 6016,
    'microsoft/deberta-v2-xxlarge': 5314,

    'microsoft/deberta-v3-xsmall': 3843, # 769, 
    # 'microsoft/deberta-v3-small': 7699,
    # 'microsoft/deberta-v3-base': 5003,
    'microsoft/deberta-v3-large': 4426,

    'gpt2': 3376,
    # 'gpt2-medium': 3610,
    'gpt2-large': 4200,
    # 'gpt2-xl': 5071,

    'google/t5-small-lm-adapt': 1414,
    # 'google/t5-base-lm-adapt': 2021,
    'google/t5-large-lm-adapt': 2548,
    # 'google/t5-xl-lm-adapt': 1,
    'google/t5-xxl-lm-adapt': 1,

    'jacobmorrison/tk-instruct-small-lora-experiments': 1413,
    # 'jacobmorrison/tk-instruct-base-lora-experiments': 2021,
    'jacobmorrison/tk-instruct-large-lora-experiments': 2548,
    # 'jacobmorrison/tk-instruct-xl-lora-experiments': 1,
    'jacobmorrison/tk-instruct-xxl-lora-experiments': 1,

    '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B': 12603,
    '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/13B': 1,

    # Decide if we want to run these
    # 'huggyllama/llama-13b',
    # 'llama-2-7b,
    # 'llama-2-13b,
    # 'google/t5-xxl-lm-adapt',
    # 'jacobmorrison/tk-instruct-xxl-lora-experiments',
}

methods = [
    # 'full_finetuning',
    'lora_1',
    'lora_2',
    'lora_4',
    # 'lora_8',
    'lora_16',
    'lora_32',
    'lora_64',
    
    # TODO: programmatically add 20%, 40%, 60%, 80%, 100% trainable parameters
]
    
# 3 seeds * 3 learning rates * 22 models * 13 methods * 10 tasks

model_specific_lora_ranks = {}

coefficients = [
    0.2,
    0.4,
    0.6,
    0.8
]

_, max_scores = get_data('case-hold')
    
for model in LoRA_ranks:
    model_specific_lora_ranks[model] = []
    if LoRA_ranks[model] != 1:
        model_specific_lora_ranks[model].append('lora_' + str(int(LoRA_ranks[model])))
        for coefficient in coefficients:
            model_specific_lora_ranks[model].append('lora_' + str(int(ceil(coefficient * LoRA_ranks[model]))))


for experiment in experiments:
    with open('configs/base-config.yaml', 'r') as f:
        default_yaml = f.read()
    d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

    for model in models:
        if 'llama' in model:
            with open('configs/base-config-llama.yaml', 'r') as f:
                default_yaml = f.read()
            d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)
        else:
            with open('configs/base-config.yaml', 'r') as f:
                default_yaml = f.read()
            d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)
        for seed in seeds:
            for learning_rate in learning_rates:
                for method in methods + model_specific_lora_ranks[model]:
                    print(learning_rate)
                    if model in max_scores:
                        if method == 'full_finetuning':
                            learning_rate = max_scores[model]['Full Finetuning']['best learning rate']
                        else:
                            learning_rate = max_scores[model]['LoRA ' + method.split('_')[-1]]['best learning rate']
                    print(learning_rate)

                    if model in xl_models:
                        batch_size_constant = xl_models[model]
                    else:
                        batch_size_constant = 8
                    num_instances_for_eval = 10000
                    # print(models[model])
                    num_gpus = int(models[model])
                    # print(num_gpus)
                    eval_batch_size = batch_size_constant
                    train_batch_size = int(batch_size_constant / num_gpus)
                    max_train_steps = int(150000 / (train_batch_size * num_gpus))
                    save_steps = int(num_instances_for_eval / batch_size_constant)
                    eval_steps = int(num_instances_for_eval / batch_size_constant)

                    d = copy.deepcopy(d1)
                    if 't5' not in model and 'tk' not in model:
                        d['tasks'][0]['envVars'][4]['value'] = 'requirements-non-t5.txt'
                    d['tasks'][0]['envVars'][8]['value'] = seed # SEED
                    d['tasks'][0]['envVars'][5]['value'] = model # MODEL
                    d['tasks'][0]['envVars'][10]['value'] = experiment # TASK
                    d['tasks'][0]['resources']['gpuCount'] = num_gpus
                    if experiment in ['hellaswag', 'yelp', 'mathqa', 'piqa', 'qnli']:
                        if '--do_pred' in  d['tasks'][0]['arguments']:
                            d['tasks'][0]['arguments'].remove('--do_pred')
                        if '--do_predict' in  d['tasks'][0]['arguments']:
                            d['tasks'][0]['arguments'].remove('--do_predict')

                    for i in range(len(d['tasks'][0]['arguments'])):
                        if '$EXPERIMENT' in d['tasks'][0]['arguments'][i]:
                            if experiment == 'squad':
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$EXPERIMENT', 'question_answering')
                            else:
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$EXPERIMENT', 'sequence_classification')
                        if '$TASK' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$TASK', experiment)
                        if '$MODEL' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$MODEL', model)
                        if '$SEED' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$SEED', '1')
                        # TODO: decide if we care about LOWER (we don't right now)
                        if '$LOWER' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$LOWER', 'True' if 'uncased' in model else 'False')
                        if '$USE_LORA' in d['tasks'][0]['arguments'][i]:
                            if 'lora' in method:
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$USE_LORA', 'True')
                                d['tasks'][0]['envVars'][6]['value'] = 'LoRA' # METHOD
                            else:
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$USE_LORA', 'False')
                                d['tasks'][0]['envVars'][6]['value'] = 'full_finetuning' # METHOD
                        if '$LORA_RANK' in d['tasks'][0]['arguments'][i]:
                            if 'lora' in method:
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$LORA_RANK', method.split('_')[-1])
                                d['tasks'][0]['envVars'][7]['value'] = method.split('_')[-1] # RANK
                            else:
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$LORA_RANK', '0')
                        # TODO: fix these
                        if '$MAX_SEQ_LENGTH' in d['tasks'][0]['arguments'][i]:
                            if 'gpt2' in model: # TODO: fix for llama
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$MAX_SEQ_LENGTH', '1024')
                            else:
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$MAX_SEQ_LENGTH', '512')
                        if '$EVAL_STEPS' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$EVAL_STEPS', str(eval_steps))
                        if '$SAVE_STEPS' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$SAVE_STEPS', str(save_steps))
                        if '$MAX_STEPS' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$MAX_STEPS', str(max_train_steps))
                        if '$LEARNING_RATE' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$LEARNING_RATE', str(learning_rate))
                            d['tasks'][0]['envVars'][9]['value'] = str(learning_rate) # LEARNING_RATE
                        if '$DEVICE_TRAIN_BATCH_SIZE' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$DEVICE_TRAIN_BATCH_SIZE', str(train_batch_size))
                        if '$DEVICE_EVAL_BATCH_SIZE' in d['tasks'][0]['arguments'][i]:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$DEVICE_EVAL_BATCH_SIZE', str(eval_batch_size))


                    if model in xl_models:
                        d['tasks'][0]['constraints']['cluster'] = ['ai2/allennlp-cirrascale', 'ai2/general-cirrascale-a100-80g-ib']
                        # d['tasks'][0]['constraints']['cluster'] = ['ai2/general-cirrascale-a5000']

                    model_for_name = model.replace('/', '-')
                    name = f'{experiment}-{model_for_name}-{method}-lr_{learning_rate}-seed_{seed}'

                    d['description'] = name
                    d['tasks'][0]['name'] = name

                    print(d)

                    fn = "configs/{}.yaml".format(name)
                    file = open(fn, "w")
                    yaml.dump(d, file, default_flow_style=True)
                    file.close()

                    workspace_subset = ''
                    if experiment != 'case-hold':
                        workspace_subset = '-' + experiment
                    cmd = ("beaker experiment create {} --workspace ai2/lora-vs-full-finetuning" + workspace_subset).format(fn)
                    subprocess.Popen(cmd, shell=True)
                    time.sleep(3)

                    os.remove(fn)