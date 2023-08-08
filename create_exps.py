import copy
import subprocess
import yaml
import random
from datetime import date
import time
import os

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
    4,
    5,
]

experiments = [
    'case-hold',
    # 'qnli',
    # 'piqa',
    # 'arc-easy',
    # 'arc-challenge',
    # 'sciq',
    # 'hellaswag',
    # 'mathqa',
    # 'mnli',
    # 'yelp',
    # 'squad',
]

# TODO: do more learning rates if a good LR is on a boundary, or if none work at all
learning_rates = [
    # '1e-2',
    # '5e-3',
    # '1e-3',
    # '5e-4',
    '1e-4',
    '5e-5',
    '1e-5',
    '5e-6',
    '1e-6',
]

models = {
    ### don't use long term ###
    # 'roberta-base': 1,
    # 'microsoft/deberta-base': 1,
    # 'roberta-large': 2,
    # 'microsoft/deberta-large': 2,
    # t5-v1_1s too, probably


    ### round 1 ###
    # 'microsoft/deberta-v3-xsmall': 1,
    # 'gpt2': 1,
    # 'google/t5-v1_1-small': 1,
    # 'jacobmorrison/tk-instruct-small-lora-experiments': 1,

    ### round 2 ###
    # 'google/t5-small-lm-adapt': 1,
    # 'microsoft/deberta-v3-small': 1,
    # 'microsoft/deberta-v3-base': 1,

    ### round 3 ###
    'microsoft/deberta-v3-large': 2,
    'gpt2-medium': 2,

    ### round 3.5 ###
    # 'google/t5-v1_1-base': 2,
    # 'google/t5-base-lm-adapt': 2,
    # 'jacobmorrison/tk-instruct-base-lora-experiments': 2,

    ### round 4 ###
    # 'microsoft/deberta-v2-xlarge': 2,
    # 'gpt2-large': 2,
    # 'google/t5-v1_1-large': 2,
    # 'jacobmorrison/tk-instruct-large-lora-experiments': 2,

    ### round 5 ###
    # 'microsoft/deberta-v2-xxlarge': 4,
    # 'gpt2-xl': 4,
    # 'google/t5-v1_1-xl': 4,
    # 'jacobmorrison/tk-instruct-xl-lora-experiments': 4,

    # TODO: decide # of GPUs for these ones
    ### round 6 ###
    # 'google/t5-v1_1-xxl': 8,
    # 'jacobmorrison/tk-instruct-xxl-lora-experiments': 8,
    # '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B': 8,
    # '/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/13B': 8,
}

LoRA_ranks = {
    'roberta-base': 3398,
    'roberta-large': 3626,

    'microsoft/deberta-base': 3776,
    'microsoft/deberta-large': 4133,

    'microsoft/deberta-v2-xlarge': 6016,
    'microsoft/deberta-v2-xxlarge': 5314,

    'microsoft/deberta-v3-xsmall': 1,
    'microsoft/deberta-v3-small': 1,
    'microsoft/deberta-v3-base': 1,
    'microsoft/deberta-v3-large': 1,

    'gpt2': 3376,
    'gpt2-medium': 3610,
    'gpt2-large': 4200,
    'gpt2-xl': 5071,

    'google/t5-v1_1-small': 1,
    'google/t5-v1_1-base': 1,
    'google/t5-v1_1-large': 1,
    'google/t5-v1_1-xl': 1,

    'jacobmorrison/tk-instruct-small-lora-experiments': 1,
    'jacobmorrison/tk-instruct-base-lora-experiments': 1,
    'jacobmorrison/tk-instruct-large-lora-experiments': 1,
    'jacobmorrison/tk-instruct-xl-lora-experiments': 1,

    'huggyllama/llama-7b': 1,

    # Decide if we want to run these
    # 'huggyllama/llama-13b',
    # 'llama-2-7b,
    # 'llama-2-13b,
    # 'google/t5-v1_1-xxl',
    # 'jacobmorrison/tk-instruct-xxl-lora-experiments',
}

methods = [
    'full_finetuning',
    # 'lora_1',
    # 'lora_2',
    # 'lora_4',
    'lora_8',
    # 'lora_16',
    # 'lora_32',
    # 'lora_64',
    
    # TODO: programmatically add 20%, 40%, 60%, 80%, 100% trainable parameters
]

for experiment in experiments:
    with open('configs/base-config.yaml', 'r') as f:
        default_yaml = f.read()
    d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

    for model in models:
        for seed in seeds:
            for learning_rate in learning_rates:
                for method in methods:
                    num_instances_for_eval = 10000
                    num_gpus = int(models[model])
                    eval_batch_size = 8
                    train_batch_size = int(8 / num_gpus)
                    max_train_steps = int(150000 / (train_batch_size * num_gpus))
                    save_steps = int(num_instances_for_eval / 8)
                    eval_steps = int(num_instances_for_eval / 8)

                    d = copy.deepcopy(d1)
                    d['tasks'][0]['envVars'][8]['value'] = seed # SEED
                    d['tasks'][0]['envVars'][5]['value'] = model # MODEL
                    d['tasks'][0]['envVars'][10]['value'] = experiment # TASK
                    d['tasks'][0]['resources']['gpuCount'] = num_gpus
                    for i in range(len(d['tasks'][0]['arguments'])):
                        if '$EXPERIMENT' in d['tasks'][0]['arguments'][i]:
                            if experiment == 'squad':
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$EXPERIMENT', 'question_answering')
                            else:
                                d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$EXPERIMENT', 'sequence_classification')
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


                    if 'llama' in model or 'gpt2-xl' in model:
                        d['tasks'][0]['constraints']['cluster'] = ['ai2/allennlp-cirrascale']

                    model_for_name = model.replace('/', '-')
                    name = f'{experiment}-{model_for_name}-{method}-lr_{learning_rate}-seed_{seed}'

                    d['description'] = name
                    d['tasks'][0]['name'] = name

                    print(d)

                    fn = "configs/{}.yaml".format(name)
                    file = open(fn, "w")
                    yaml.dump(d, file, default_flow_style=True)
                    file.close()

                    cmd = "beaker experiment create {} --workspace ai2/lora-vs-full-finetuning".format(fn)
                    subprocess.Popen(cmd, shell=True)
                    time.sleep(3)

                    os.remove(fn)