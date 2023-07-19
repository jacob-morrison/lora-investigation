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

# modify here for different set of experiments
# experiment_group = "train_squad_models"
# experiment_group = "train_and_eval_squadlike"

# experiment_group = "train_nli_models"
# experiment_group = "train_and_eval_qa_subset"
# experiment_group = "retrain_tk_qa_eval"
# experiment_group = "train_full_squad"
# experiment_group = "train_and_eval_actually_only_qa"
# # experiment_group = "train_qa_training_curves"
experiment_group = "none"
# experiment_group = "train_full_nli"
# experiment_group = "train_full_squad_lora"
# experiment_group = "train_full_nli_lora"
# experiment_group = "train_low_data_nli_lora"
# experiment_group = "retrain_tk_sentiment_analysis"
# experiment_group = "train_full_sentiment_analysis"
# experiment_group = "train_low_data_sentiment_analysis"
# experiment_group = "train_low_data_sentiment_analysis_lora"
# experiment_group = "train_full_sentiment_analysis_lora"
# experiment_group = "retrain_tk_sentiment_analysis_lora"
# experiment_group = "retrain_tk_standard_lora"
# experiment_group = "eval_models"

encodings = {
    # "input_only": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False},
    # "task_name_input": {"add_task_name": True, "add_task_definition": False, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False},
    # "pos_1_input": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 1, "num_neg_examples": 0, "add_explanation": False},
    # "pos_2_input": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False},
    # "pos_4_input": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 4, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_pos_1_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 1, "num_neg_examples": 0, "add_explanation": False},
    "instruct_pos_2_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_pos_4_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 4, "num_neg_examples": 0, "add_explanation": False},
    # "instruct_pos_2_neg_2_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": False},
    # "instruct_pos_2_neg_2_explanation_input": {"add_task_name": False, "add_task_definition": True, "num_pos_examples": 2, "num_neg_examples": 2, "add_explanation": True},
    # "tk_instruct": {"add_task_name": False, "add_task_definition": False, "num_pos_examples": 0, "num_neg_examples": 0, "add_explanation": False, "tk_instruct": True},
}

# ---- run all experiments ---- #
experiments = [
    'case-hold',
    'qnli',
]

models = {
    'roberta-base': 1,
    'roberta-large': 1,

    'microsoft/deberta-base': 1,
    'microsoft/deberta-large': 1,

    'microsoft/deberta-v2-xlarge': 2,
    'microsoft/deberta-v2-xxlarge': 4,

    'microsoft/deberta-v3-xsmall': 1,
    'microsoft/deberta-v3-small': 1,
    'microsoft/deberta-v3-base': 1,
    'microsoft/deberta-v3-large': 1,

    'gpt2': 1,
    'gpt2-medium': 1,
    'gpt2-large': 2,
    'gpt2-xl': 4,

    'google/t5-v1_1-small': 1,
    'google/t5-v1_1-base': 1,
    'google/t5-v1_1-large': 2,
    'google/t5-v1_1-xl': 4,

    'jacobmorrison/tk-instruct-small-lora-experiments': 1,
    'jacobmorrison/tk-instruct-base-lora-experiments': 1,
    'jacobmorrison/tk-instruct-large-lora-experiments': 2,
    'jacobmorrison/tk-instruct-xl-lora-experiments': 4,

    'huggyllama/llama-7b': 8,

    # TODO: decide # of GPUs for these ones

    # Decide if we want to run these
    # 'huggyllama/llama-13b',
    # 'llama-2-7b,
    # 'llama-2-13b,
    # 'google/t5-v1_1-xxl',
    # 'jacobmorrison/tk-instruct-xxl-lora-experiments',
}

LoRA_ranks = {
    'roberta-base': 1,
    'roberta-large': 1,

    'microsoft/deberta-base': 3776,
    'microsoft/deberta-large': 1,

    'microsoft/deberta-v2-xlarge': 1,
    'microsoft/deberta-v2-xxlarge': 1,

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
    'lora_2',
    # 'lora_4',
    # 'lora_8',
    # 'lora_16',
    # 'lora_32',
    # 'lora_64',
    # 'lora_128',
    # 'lora_256',
    # 'lora_512',
    # 'lora_1024',
]

for experiment in experiments:
    with open('configs/base-config-' + experiment + '.yaml', 'r') as f:
        default_yaml = f.read()
    d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

    for seed in seeds:
        for model in models:
            for method in methods:
                d = copy.deepcopy(d1)
                for i in range(len(d['tasks'][0]['arguments'])):
                    if '$MODEL' in d['tasks'][0]['arguments'][i]:
                        d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$MODEL', model)
                    if '$METHOD' in d['tasks'][0]['arguments'][i]:
                        d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$METHOD', method)
                    if '$SEED' in d['tasks'][0]['arguments'][i]:
                        d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$SEED', '1')
                    if '$LOWER' in d['tasks'][0]['arguments'][i]:
                        d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$LOWER', 'True' if 'uncased' in model else 'False')
                    if '$USE_LORA' in d['tasks'][0]['arguments'][i]:
                        if 'lora' in method:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$USE_LORA', 'True')
                        else:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$USE_LORA', 'False')
                    if '$LORA_RANK' in d['tasks'][0]['arguments'][i]:
                        if 'lora' in method:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$LORA_RANK', method.split('_')[-1])
                        else:
                            d['tasks'][0]['arguments'][i] = d['tasks'][0]['arguments'][i].replace('$LORA_RANK', '0')

                if 'llama' in model or 'gpt2-xl' in model:
                    d['tasks'][0]['constraints']['cluster'] = ['ai2/allennlp-cirrascale']

                model_for_name = model.replace('/', '-')
                name = f'{experiment}-{model_for_name}-{method}-seed_{seed}'

                d['description'] = name
                d['tasks'][0]['name'] = name

                print(d)

                fn = "configs/{}.yaml".format(name)
                file = open(fn, "w")
                yaml.dump(d, file, default_flow_style=True)
                file.close()

                cmd = "beaker experiment create {} --workspace ai2/lexglue-tasks".format(fn)
                subprocess.Popen(cmd, shell=True)
                time.sleep(3)

                os.remove(fn)

#--------------- experiments about number of supervision tasks -------------------------

if experiment_group == "num_of_tasks":
    train_task_nums = [8, 32, 128, 256, 512]
    for train_task_num in train_task_nums:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{train_task_num}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/cross_category/train_{train_task_num}")
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- experiments about instances per task -------------------------

if experiment_group == "num_of_instances":
    instance_per_task_nums = [8, 32, 64, 128, 256, 512]
    for num_instance_per_task in instance_per_task_nums:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{num_instance_per_task}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", num_instance_per_task)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- experiments about model variants -------------------------

if experiment_group == "eval_models":
    checkpoints = [
        # checkpoint_name, beaker_dataset, checkpoint (if None, will use the root beaker output dir)
        # ("input_only", "01FZHYPTKGEN16404MV5TTMJAK", None),
        # ("task_name_input", "01FZHYPV1CNJFNDGNWDJ84G2XV", None),
        # ("instruct_input", "01FZHYPS9XC47R5CZA7RN8TTPQ", None),
        # ("pos_1_input", "01FZHYPSR0Z8S7F901ZTK2SDER", None),
        # ("instruct_pos_1_input", "01FZHYPSZ4SPGV47R0GVR1JFDT", None),
        # ("pos_2_input", "01FZHYPTTE4YGQEXECSPBZGA28", None),
        # ("instruct_pos_2_input", "01FZHYPSGWBER9A9B2NV8TXE4Q", None),
        # ("instruct_pos_2_neg_2_input", "01FZHYPT60T8R7GVF4V5FKSK5E", None),
        # ("instruct_pos_2_neg_2_explanation_input", "01FZHYPS30B5J8P3P8MVDFZJSY", None),
        # ("pos_4_input", "01FZHYPV88MHTW3SHGSTZ753E6", None),
        # ("instruct_pos_4_input", "01FZHYPTCTA9XRD45KKQBTBDZ0", None),   
        # ("tk_instruct", "01FZK3EKQPZNCY30KFECK49YZN", "checkpoint-5000"),
        ("retrain-baseline", "retrain-tk-for-interpolations-small-2", None)
    ]

    # TODO: Add checkpoint support
    # TODO: This includes using the subpath field

    for checkpoint_name, beaker_dataset_id, checkpoint_step in checkpoints:
        for encoding_name, encoding in encodings.items():
            d = copy.deepcopy(d1)

            # d['tasks'][0]['context']['cluster'] = cluster
            d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
            d['tasks'][0]['image']['beaker'] = 'jacobm/eval-sni'

            name = f"ni_{experiment_group}_{checkpoint_name}_test_encoding_{encoding_name}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            assert d['tasks'][0]['command'][3].endswith(".py")
            d['tasks'][0]['command'] = ["python"] + d['tasks'][0]['command'][3:] 
            d['tasks'][0]['command'].remove("--do_train")
            d['tasks'][0]['command'].remove("--bf16")
            d['tasks'][0]['command'].remove("--deepspeed")
            d['tasks'][0]['command'].remove("ds_configs/stage2.config")

            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            # set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            # set_argument_value(d['tasks'][0]['command'], "--disable_tqdm", False)

            set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
            set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
            set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
            set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
            set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
            set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", "no")
            set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/default')
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
            d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/sni-for-evals'

            if "tk_instruct" in encoding:
                set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])

            d['tasks'][0]['datasets'].append({"mountPath": "/models/", "source": {"beaker": 'jacobm/' + beaker_dataset_id}})            
            set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", "/models/" + (checkpoint_step if checkpoint_step else ""))            

            d['tasks'][0]['resources']['gpuCount'] = 1
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 4)

            set_argument_value(d['tasks'][0]['command'], "--run_name", name) 
            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
            subprocess.Popen(cmd, shell=True)
            time.sleep(3)

### In this experiment, we're creating new Tk baselines with modified sentiment analysis datasets

if experiment_group == "retrain_tk_standard_lora":
    model_names = [
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        # d['tasks'][0]['image']['beaker'] = 'jacobm/train-with-lora-no-delete'
        d['tasks'][0]['image']['beaker'] = 'jacobm/retrain_tk_sentiment_analysis'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 5)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 100)
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/retrain-tk'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/default')
        # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{str(100)}_{today}"
        name = f"retrain_tk-instruct_lora_{model_name.split('/')[-1]}"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['priority'] = 'preemptible'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
            d['tasks'][0]['context']['priority'] = 'preemptible'
        
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)


#--------------- experiments about model variants -------------------------

### In this experiment, we're creating new Tk baselines with modified sentiment analysis datasets

if experiment_group == "retrain_tk_sentiment_analysis_lora":
    model_names = [
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/train-with-lora-no-delete'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 5)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        # for dataset, splits, num_examples in datasets:
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 100)
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/data-for-sentiment-analysis'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/sentiment_analysis')
        # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{str(100)}_{today}"
        name = f"retrain_tk-instruct_sentiment-analysis_lora_{model_name.split('/')[-1]}"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['priority'] = 'preemptible'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 8
            # d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            d['tasks'][0]['context']['priority'] = 'high'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
            # d['tasks'][0]['context']['priority'] = 'preemptible'
        
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)


#--------------- experiments about model variants -------------------------

### In this experiment, we're creating new Tk baselines with modified sentiment analysis datasets

if experiment_group == "retrain_tk_sentiment_analysis":
    model_names = [
        "google/t5-small-lm-adapt",
        "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/retrain_tk_sentiment_analysis'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 5)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        # for dataset, splits, num_examples in datasets:
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 100)
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/data-for-sentiment-analysis'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/sentiment_analysis')
        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{str(100)}_{today}_2"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            print('wrong size??????????')
        
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)


#--------------- experiments about model variants -------------------------

### In this experiment, we're comparing *training data*
# In other words, for the same amount of data, does it make sense
# to use only SQuAD, or a mixture of QA datasets?
# We'll be evaluating every 2k steps (total of just under 32k steps) to build curves

if experiment_group == "retrain_tk_or_full_squad_training_curves":
    model_names = [
        # Baselines -> For both SQuAD and retraining
        "google/t5-small-lm-adapt",
        "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",

        # Tk-Instruct -> Only for SQuAD
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        # 'allenai/tk-instruct-3b-def-pos'

        # "bigscience/T0_3B",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        datasets = [
            # ('jacobm/squad-for-t5', '/data/splits/squad', 75600, 150000, 'ai2/allennlp-cirrascale'),
            # ('jacobm/squad-for-t5', '/data/splits/squad', 2500, 5000, 'ai2/general-cirrascale'),
            # ('jacobm/retrain-tk', '/data/splits/default', 100)
            # ('jacobm/data-for-nli', '/data/splits/default', 75600, 150000, 'ai2/allennlp-cirrascale'),
            # ('jacobm/data-for-nli', '/data/splits/default', 2500, 5000, 'ai2/general-cirrascale'),
            ('jacobm/sentiment-analysis-training3', '/data/splits/sentiment-analysis', 74117, 140000, 'ai2/general-cirrascale'),
            ('jacobm/sentiment-analysis-training3', '/data/splits/sentiment-analysis', 2500, 5000, 'ai2/general-cirrascale'),
        ]

        # d['tasks'][0]['image']['beaker'] = 'jacobm/retrain_tk_or_full_squad_training_curves'
        # d['tasks'][0]['image']['beaker'] = 'jacobm/retrain_tk_sentiment_analysis'
        images = [
            'jacobm/train-with-lora-2',
            'jacobm/train-with-lora-4',
            # 'jacobm/train-with-lora-8',
            'jacobm/train-with-lora-16',
            'jacobm/train-with-lora-32',
            'jacobm/train-with-lora-64',
        ]
        for image in images:
            # image = 'jacobm/train-with-lora-2'
            # image = 'jacobm/train-with-lora-4'
            # image = 'jacobm/train-with-lora-16'
            # image = 'jacobm/train-with-lora-32'
            # image = 'jacobm/train-with-lora-256'
            # image = 'jacobm/train-with-lora-512'
            d['tasks'][0]['image']['beaker'] = image
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
            



            set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
            # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
            set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 5)
            set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
            set_argument_value(d['tasks'][0]['command'], "--logging_steps", 2500000)
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)

            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

            for dataset, splits, num_examples, save_steps, cluster in datasets:
                if save_steps == 5000 and ('small' in model_name or 'base' in model_name):
                    d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
                else:
                    d['tasks'][0]['context']['cluster'] = cluster
                set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", num_examples)
                d['tasks'][0]['datasets'][0]['source']['beaker'] = dataset
                set_argument_value(d['tasks'][0]['command'], "--data_dir", splits)
                set_argument_value(d['tasks'][0]['command'], "--save_steps", save_steps)
                # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{str(num_examples)}_{today}"
                # name = f"retrain_tk_instruct_for_merge_baseline_{model_name.split('/')[-1]}_{str(num_examples)}_{today}"
                if save_steps == 5000:
                    amount_of_data = 'low-data'
                else:
                    amount_of_data = 'high-data'
                name = f"{dataset.split('/')[-1]}_{model_name.split('/')[-1]}_{today}_lora-{image.split('-')[-1]}_{amount_of_data}"
                d['description'] = name
                d['tasks'][0]['name'] = name
                set_argument_value(d['tasks'][0]['command'], "--run_name", name)
                # if "small" in model_name:
                #     d['tasks'][0]['resources']['gpuCount'] = 1
                #     d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
                #     set_argument_value(d['tasks'][0]['command'], "--save_steps", 20000)
                #     # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
                # elif "base" in model_name:
                #     d['tasks'][0]['resources']['gpuCount'] = 1
                #     d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
                #     set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
                #     # d['tasks'][0]['resources']['gpuCount'] = 2
                #     # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
                #     # set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
                #     # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
                # elif "large" in model_name:
                #     d['tasks'][0]['resources']['gpuCount'] = 1
                #     d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
                #     set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
                #     # d['tasks'][0]['resources']['gpuCount'] = 4
                #     # d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
                #     # set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
                #     # d['tasks'][0]['context']['priority'] = 'preemptible'
                #     # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
                # elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
                #     d['tasks'][0]['resources']['gpuCount'] = 1
                #     d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
                #     set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
                    # d['tasks'][0]['resources']['gpuCount'] = 1
                    # d['tasks'][0]['resources']['gpuCount'] = 8
                    # d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
                    # set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
                    # d['tasks'][0]['context']['priority'] = 'preemptible'
                    # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
                    # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
                    # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
                # elif "11b" in model_name or "-xxl" in model_name:
                #     print('wrong size??????????')
                #     print('wrong size??????????')
                
                print(d)

                fn = "beaker_configs/{}.yaml".format(name)
                file = open(fn, "w")
                yaml.dump(d, file, default_flow_style=True)
                file.close()

                cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
                subprocess.Popen(cmd, shell=True)
                time.sleep(3)

#--------------- experiments about model variants -------------------------

### In this experiment, we're comparing *training data*
# In other words, for the same amount of data, does it make sense
# to use only SQuAD, or a mixture of QA datasets?
# We'll be evaluating every 2k steps (total of just under 32k steps) to build curves

if experiment_group == "train_qa_training_curves":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        datasets = [
            ('jacobm/squad-for-t5', '/data/splits/squad', 15700),
            # ('jacobm/tk-instruct-actually-only-qa', '/data/splits/actually-only-qa', 100)
        ]

        d['tasks'][0]['image']['beaker'] = 'jacobm/training-curves-for-qa'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'



        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 5)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 15000)
        set_argument_value(d['tasks'][0]['command'], "--save_steps", 2000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        for dataset, splits, num_examples in datasets:
            set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", num_examples)
            d['tasks'][0]['datasets'][0]['source']['beaker'] = dataset
            set_argument_value(d['tasks'][0]['command'], "--data_dir", splits)
            name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{str(num_examples)}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name
            set_argument_value(d['tasks'][0]['command'], "--run_name", name)
            if "small" in model_name:
                d['tasks'][0]['resources']['gpuCount'] = 1
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
                # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            elif "base" in model_name:
                d['tasks'][0]['resources']['gpuCount'] = 1
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
                # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            elif "large" in model_name:
                d['tasks'][0]['resources']['gpuCount'] = 1
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
                # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
                # d['tasks'][0]['resources']['gpuCount'] = 1
                d['tasks'][0]['resources']['gpuCount'] = 1
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
                # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
                # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
                # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            elif "11b" in model_name or "-xxl" in model_name:
                print('wrong size??????????')
            
            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
            subprocess.Popen(cmd, shell=True)
            time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_low_data_sentiment_analysis_lora":
    model_names = [
        # Baselines
        "google/t5-small-lm-adapt",
        "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",

        # My new Tk's
        "jacobmorrison/tk-small-minus-sentiment-analysis",
        "jacobmorrison/tk-base-minus-sentiment-analysis",
        "jacobmorrison/tk-large-minus-sentiment-analysis",
        "jacobmorrison/tk-xl-minus-sentiment-analysis",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/train-with-lora'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/sentiment-analysis-training3'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/sentiment-analysis')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 2500)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 150000)
        set_argument_value(d['tasks'][0]['command'], "--save_steps", 100)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_low_data_sentiment_analysis":
    model_names = [
        # Baselines
        "google/t5-small-lm-adapt",
        "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",

        # My new Tk's
        "jacobmorrison/tk-small-minus-sentiment-analysis",
        "jacobmorrison/tk-base-minus-sentiment-analysis",
        "jacobmorrison/tk-large-minus-sentiment-analysis",
        "jacobmorrison/tk-xl-minus-sentiment-analysis",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/image_for_sentiment_analysis'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/sentiment-analysis-training3'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/sentiment-analysis')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 2500)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 150000)
        set_argument_value(d['tasks'][0]['command'], "--save_steps", 100)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_full_sentiment_analysis_lora":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",

        # My new Tk's
        # "jacobmorrison/tk-small-minus-sentiment-analysis",
        "jacobmorrison/tk-base-minus-sentiment-analysis",
        # "jacobmorrison/tk-large-minus-sentiment-analysis",
        # "jacobmorrison/tk-xl-minus-sentiment-analysis",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/train-with-lora-no-delete'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/sentiment-analysis-training3'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/sentiment-analysis')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 74117)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 150000)
        set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_full_sentiment_analysis":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",

        # My new Tk's
        # "jacobmorrison/tk-small-minus-sentiment-analysis",
        # "jacobmorrison/tk-base-minus-sentiment-analysis",
        # "jacobmorrison/tk-large-minus-sentiment-analysis",
        "jacobmorrison/tk-xl-minus-sentiment-analysis",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/image_for_sentiment_analysis'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/sentiment-analysis-training3'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/sentiment-analysis')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 74117)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 150000)
        set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_full_nli":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",

        # T0
        "bigscience/T0_3B",

        # # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        'allenai/tk-instruct-3b-def-pos'
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/data-for-nli'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/data-for-nli'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/default')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 75600)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 10000000)
        # set_argument_value(d['tasks'][0]['command'], "--save_steps", 100)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 2
            # set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 2
            # set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 4
            # set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_full_squad_lora":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",

        # # T0
        # "bigscience/T0_3B",

        # # # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        'allenai/tk-instruct-3b-def-pos',

        # Flan
        'google/flan-t5-small',
        # 'google/flan-t5-base',
        # 'google/flan-t5-large',
        # 'google/flan-t5-xl',
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/train-with-lora-4'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/squad-for-t5'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/squad')

        # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        low_data = True
        if low_data:
            name = f"train_squad_low_data_lora-rank_4_{model_name.split('/')[-1]}"
            set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 2500)
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 100)
        else:
            name = f"train_squad_full_data_lora-rank_4_{model_name.split('/')[-1]}"
            set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 75600)
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_low_data_nli_lora":
    model_names = [
        # # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",

        # # T0
        "bigscience/T0_3B",

        # # # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        # 'allenai/tk-instruct-3b-def-pos'

        # My lora Tk's
        # 'jacobmorrison/tk-small-lora',
        # 'jacobmorrison/tk-base-lora',
        # 'jacobmorrison/tk-large-lora',
        # 'jacobmorrison/tk-xl-lora',
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        # d['tasks'][0]['image']['beaker'] = 'jacobm/lora-base-train-lora'
        d['tasks'][0]['image']['beaker'] = 'jacobm/data-for-nli'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/data-for-nli'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/default')

        # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        name = f"train-low-data-nli-{model_name.split('/')[-1]}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 2500)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 10000)
        set_argument_value(d['tasks'][0]['command'], "--save_steps", 100)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_full_nli_lora":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",

        # T0
        # "bigscience/T0_3B",

        # # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        # 'allenai/tk-instruct-3b-def-pos'

        # My lora Tk's
        'jacobmorrison/tk-small-lora',
        'jacobmorrison/tk-base-lora',
        'jacobmorrison/tk-large-lora',
        'jacobmorrison/tk-xl-lora',
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/lora-base-train-lora'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/data-for-nli'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/default')

        # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        name = f"train-full-nli-lora-{model_name.split('/')[-1]}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 75600)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
            d['tasks'][0]['context']['priority'] = 'preemptible'
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_full_squad":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",

        # T0
        # "bigscience/T0_3B",

        # # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        # 'allenai/tk-instruct-3b-def-pos'

        # Flan
        # 'google/flan-t5-small',
        # 'google/flan-t5-base',
        # 'google/flan-t5-large',
        'google/flan-t5-xl',

        # LoRA Tk's
        # 'jacobmorrison/tk-small-lora',
        # 'jacobmorrison/tk-base-lora',
        # 'jacobmorrison/tk-large-lora',
        # 'jacobmorrison/tk-xl-lora',
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        lora = False
        low_data = True

        if lora:
            # d['tasks'][0]['image']['beaker'] = 'jacobm/train-with-lora-no-delete'
            d['tasks'][0]['image']['beaker'] = 'jacobm/lora-base-train-lora'
        else:
            d['tasks'][0]['image']['beaker'] = 'jacobm/training-curves-with-final-model-eval2'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/squad-for-t5'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/squad')

        # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}_LoRA"
        name = f"train_full_squad_{model_name.split('/')[-1]}"
        if lora:
            name += '_with-lora'
        if low_data:
            name += '_low-data'
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 75600)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 10)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000000)
        set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 5000)
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "large" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 4
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            # d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name or "3B" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 1
            # set_argument_value(d['tasks'][0]['command'], "--save_steps", 1250)
            # d['tasks'][0]['resources']['gpuCount'] = 8
            # d['tasks'][0]['context']['priority'] = 'preemptible'
            # d['tasks'][0]['context']['cluster'] = 'ai2/general-cirrascale-a100-80g-ib'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            d['tasks'][0]['resources']['gpuCount'] = 4
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 2500)
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")

        if lora:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 10000)

        if low_data:
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 2500)
            set_argument_value(d['tasks'][0]['command'], "--save_steps", 100)
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "retrain_tk_qa_eval":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/training-curves-qa-subset'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'Retrain Tk-Instruct'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/sni-eval-on-squadlike'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/default')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 100)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 16980)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 1
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "large" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_and_eval_actually_only_qa":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/training-curves-qa-subset'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'T5 + Only ACTUAL QA'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/tk-instruct-actually-only-qa'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/actually-only-qa')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 100)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 16980)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 2000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 1
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "large" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'

            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        # elif "11b" in model_name or "-xxl" in model_name:
        #     set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        #     set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
        #     set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
        #     set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
        #     d['tasks'][0]['resources']['gpuCount'] = 8
        #     d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
        #     #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
        #     set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
        #     set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_and_eval_qa_subset":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        # "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",

        # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        # 'allenai/tk-instruct-3b-def-pos'
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/training-curves-qa-subset'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'T5 + All QA'
        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/sni-eval-on-squadlike'
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/qa')

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 100)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 16980)
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 2000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 1
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
        elif "base" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "large" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'

            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        # elif "11b" in model_name or "-xxl" in model_name:
        #     set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        #     set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
        #     set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
        #     set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
        #     d['tasks'][0]['resources']['gpuCount'] = 8
        #     d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
        #     #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
        #     set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
        #     set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------

if experiment_group == "train_and_eval_squadlike":
    model_names = [
        # Baselines
        "google/t5-small-lm-adapt",
        "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",
        # "google/t5-xxl-lm-adapt",

        # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        # 'allenai/tk-instruct-base-def-pos',
        # 'allenai/tk-instruct-large-def-pos',
        # 'allenai/tk-instruct-3b-def-pos'
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/training-curves-nli'
        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 6500)
        # set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 100)
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/qa')
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 1000)

        set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 1)
        set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)

        if "small" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "base" in model_name:
            d['tasks'][0]['resources']['gpuCount'] = 2 # 1 # 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "large" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            # set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            # set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] =  2 # 4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name:
            # d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['resources']['gpuCount'] = 4 # 1 # 4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------
# seeds = [
#     70,
#     71,
# ]
if experiment_group == "train_nli_models":
    model_names = [
        # Baselines
        # "google/t5-small-lm-adapt",
        "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        "google/t5-xl-lm-adapt",

        # Tk-Instruct
        # 'allenai/tk-instruct-small-def-pos',
        'allenai/tk-instruct-base-def-pos',
        'allenai/tk-instruct-large-def-pos',
        'allenai/tk-instruct-3b-def-pos'
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['image']['beaker'] = 'jacobm/training-curves-nli'
        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        # set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)
        set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 128)
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 100000)
        # set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_eval_task", 900)
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/nli')
        set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", 'steps')
        set_argument_value(d['tasks'][0]['command'], "--logging_steps", 500)


        if "small" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 1
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "base" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 2
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "large" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] =  4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
        elif "3b" in model_name or "-xl" in model_name:
            # set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 2)
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            # d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['resources']['gpuCount'] = 4
            d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
            # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
            # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)

#--------------- experiments about model variants -------------------------
seeds = [
    70,
    71,
]
if experiment_group == "train_squad_models":
    model_names = [
        # "t5-3b", 
        # "t5-11b",
        # "t5-small", 
        # "t5-large", 
        # "t5-base", 
        # "google/t5-v1_1-small", 
        # "google/t5-v1_1-base", 
        # "google/t5-v1_1-large",
        # "google/t5-v1_1-xl",
        # "google/t5-small-lm-adapt",
        # "google/t5-base-lm-adapt",
        "google/t5-large-lm-adapt",
        # "google/t5-xl-lm-adapt",
        # "google/t5-xxl-lm-adapt",
        ]
        
    for model_name in model_names:
        for seed in seeds:
            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
            set_argument_value(d['tasks'][0]['command'], "--run_name", name)
            set_argument_value(d['tasks'][0]['command'], "--seed_for_data", seed)

            if "small" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
                set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
                d['tasks'][0]['resources']['gpuCount'] = 1
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            elif "base" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
                set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
                d['tasks'][0]['resources']['gpuCount'] = 1 # 2
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            elif "large" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
                set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
                d['tasks'][0]['resources']['gpuCount'] = 1 # 2 # 4
                # d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-elanding-a100-40g'
            elif "3b" in model_name or "-xl" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 2)
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
                set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
                # d['tasks'][0]['resources']['gpuCount'] = 8
                d['tasks'][0]['resources']['gpuCount'] = 1 # 4
                d['tasks'][0]['context']['cluster'] = 'ai2/allennlp-cirrascale'
                # d['tasks'][0]['command'].remove("--bf16")  # stage 3 is currently 4x slower with bf16
                # set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
                # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            elif "11b" in model_name or "-xxl" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
                set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
                set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
                d['tasks'][0]['resources']['gpuCount'] = 8
                d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
                #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
                set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
                set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
                
            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/jacobm-default-workspace".format(fn)
            subprocess.Popen(cmd, shell=True)
            time.sleep(3)

#--------------- experiments about learning rate and batch size -------------------------
if experiment_group == "hyper_tuning":
    learning_rates = [1e-5, 3e-5, 5e-5, 1e-4, 1e-3]
    acc_steps = [2, 4, 8, 16]
    for lr in learning_rates:
        for acc_step in acc_steps:

            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"ni_training_lr_{lr}_accu_{acc_step}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--learning_rate", lr)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", acc_step)
            set_argument_value(d['tasks'][0]['command'], "--run_name", name)

            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
            subprocess.Popen(cmd, shell=True)

# --------------- experiments about the encodings of NI elements -------------------------
if experiment_group == "encoding":
    for encoding_name, encoding in encodings.items():
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_t0_subset_{experiment_group}_{encoding_name}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
        set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
        set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
        set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
        set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
        if "tk_instruct" in encoding:
            set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- different test sets -------------------------
if experiment_group == "test_sets":

    for set_idx in range(10):
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_test_set_{set_idx}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/cross_category/set_{set_idx}")

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)

#--------------- different splits -------------------------
if experiment_group == "splits":

    for split_name in ["default", "no_synthetic", "supervised"]:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_{split_name}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/cross_category/{split_name}")

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)


#--------------- no-finetuning transfer of pretrained models -------------------------
if experiment_group == "eval_pretrained_models":
    model_names = [
        # "google/t5-xl-lm-adapt",
        # "google/t5-xxl-lm-adapt",
        # "bigscience/T0",
        # "bigscience/T0_3B",
        # "t5-large",
        # "google/t5-large-lm-adapt",
        # "google/mt5-xxl",
        "allenai/tk-instruct-11b-def-pos",
        # "allenai/tk-instruct-3b-def-pos", 
        # "allenai/mtk-instruct-3b-def-pos", 
    ]

    for model_name in model_names:
        for encoding_name, encoding in encodings.items():
            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"ni_evaluation_model_{model_name.split('/')[-1]}_encoding_{encoding_name}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            assert d['tasks'][0]['command'][3].endswith(".py")
            # d['tasks'][0]['command'] = ["python"] + d['tasks'][0]['command'][3:] 
            d['tasks'][0]['command'].remove("--do_train")
            d['tasks'][0]['command'].remove("--bf16")
            # d['tasks'][0]['command'].remove("--deepspeed")
            # d['tasks'][0]['command'].remove("ds_configs/stage2.config")

            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--disable_tqdm", False)

            set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
            set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
            set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
            set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
            set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
            set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", "no")

            if "tk_instruct" in encoding:
                set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])
            
            
            # set model and resources
            set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
            d['tasks'][0]['resources']['gpuCount'] = 1
            if "small" in model_names:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            elif "base" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            elif "large" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            elif "3b" in model_name or "3B" in model_name or "-xl" in model_name:
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 4)
            elif "11b" in model_name or "11B" in model_name or "-xxl" in model_name or model_name == "bigscience/T0":
                set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 2)
                d['tasks'][0]['resources']['gpuCount'] = 8
                set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
            set_argument_value(d['tasks'][0]['command'], "--run_name", name)
            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
            subprocess.Popen(cmd, shell=True)

#--------------- evaluation of beaker checkpoints -------------------------
if experiment_group == "eval_ckpt":
    checkpoints = [
        # checkpoint_name, beaker_dataset, checkpoint (if None, will use the root beaker output dir)
        # ("input_only", "01FZHYPTKGEN16404MV5TTMJAK", None),
        # ("task_name_input", "01FZHYPV1CNJFNDGNWDJ84G2XV", None),
        # ("instruct_input", "01FZHYPS9XC47R5CZA7RN8TTPQ", None),
        # ("pos_1_input", "01FZHYPSR0Z8S7F901ZTK2SDER", None),
        # ("instruct_pos_1_input", "01FZHYPSZ4SPGV47R0GVR1JFDT", None),
        # ("pos_2_input", "01FZHYPTTE4YGQEXECSPBZGA28", None),
        # ("instruct_pos_2_input", "01FZHYPSGWBER9A9B2NV8TXE4Q", None),
        # ("instruct_pos_2_neg_2_input", "01FZHYPT60T8R7GVF4V5FKSK5E", None),
        # ("instruct_pos_2_neg_2_explanation_input", "01FZHYPS30B5J8P3P8MVDFZJSY", None),
        # ("pos_4_input", "01FZHYPV88MHTW3SHGSTZ753E6", None),
        # ("instruct_pos_4_input", "01FZHYPTCTA9XRD45KKQBTBDZ0", None),   
        # ("tk_instruct", "01FZK3EKQPZNCY30KFECK49YZN", "checkpoint-5000"),
        ("mt5-xl", "01G0A8CYHZF5VV2SW3V10Y9CZT", None)
    ]

    for checkpoint_name, beaker_dataset_id, checkpoint_step in checkpoints:
        for encoding_name, encoding in encodings.items():
            d = copy.deepcopy(d1)

            d['tasks'][0]['context']['cluster'] = cluster

            name = f"ni_{experiment_group}_{checkpoint_name}_test_encoding_{encoding_name}_{today}"
            d['description'] = name
            d['tasks'][0]['name'] = name

            assert d['tasks'][0]['command'][3].endswith(".py")
            d['tasks'][0]['command'] = ["python"] + d['tasks'][0]['command'][3:] 
            d['tasks'][0]['command'].remove("--do_train")
            d['tasks'][0]['command'].remove("--bf16")
            d['tasks'][0]['command'].remove("--deepspeed")
            d['tasks'][0]['command'].remove("ds_configs/stage2.config")

            # set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            # set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
            set_argument_value(d['tasks'][0]['command'], "--disable_tqdm", False)

            set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
            set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
            set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
            set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
            set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
            set_argument_value(d['tasks'][0]['command'], "--evaluation_strategy", "no")

            if "tk_instruct" in encoding:
                set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])

            d['tasks'][0]['datasets'].append({"mountPath": "/models/", "source": {"beaker": beaker_dataset_id}})            
            set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", "/models/" + (checkpoint_step if checkpoint_step else ""))            

            d['tasks'][0]['resources']['gpuCount'] = 1
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 4)

            set_argument_value(d['tasks'][0]['command'], "--run_name", name) 
            print(d)

            fn = "beaker_configs/{}.yaml".format(name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
            subprocess.Popen(cmd, shell=True)


#--------------- supervised upper bound -------------------------
if experiment_group == "supervised":
    for encoding_name, encoding in encodings.items():
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_supervised_upper_bound_encoding_{encoding_name}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--add_task_name", encoding["add_task_name"])
        set_argument_value(d['tasks'][0]['command'], "--add_task_definition", encoding["add_task_definition"])
        set_argument_value(d['tasks'][0]['command'], "--num_pos_examples", encoding["num_pos_examples"])
        set_argument_value(d['tasks'][0]['command'], "--num_neg_examples", encoding["num_neg_examples"])
        set_argument_value(d['tasks'][0]['command'], "--add_explanation", encoding["add_explanation"])
        if "tk_instruct" in encoding:
            set_argument_value(d['tasks'][0]['command'], "--tk_instruct", encoding["tk_instruct"])

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/supervised/multilingual")
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 1000)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True) 


#--------------- multilingual -------------------------
if experiment_group == "multilingual":
    model_names = [
        "google/mt5-xl",
    ]
        
    for model_name in model_names:
        d = copy.deepcopy(d1)

        d['tasks'][0]['context']['cluster'] = cluster

        name = f"ni_training_{experiment_group}_supervised_{model_name.split('/')[-1]}_{today}"
        d['description'] = name
        d['tasks'][0]['name'] = name

        set_argument_value(d['tasks'][0]['command'], "--master_port", random.randint(25000, 35000))
        set_argument_value(d['tasks'][0]['command'], "--model_name_or_path", model_name)
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)

        if "small" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 32)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 1
        elif "base" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 16)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 2
        elif "large" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 4)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            d['tasks'][0]['resources']['gpuCount'] = 4
        elif "3b" in model_name or "-xl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 2)
            d['tasks'][0]['resources']['gpuCount'] = 8
        elif "11b" in model_name or "-xxl" in model_name:
            set_argument_value(d['tasks'][0]['command'], "--per_device_train_batch_size", 1)
            set_argument_value(d['tasks'][0]['command'], "--per_device_eval_batch_size", 8)
            set_argument_value(d['tasks'][0]['command'], "--gradient_accumulation_steps", 1)
            set_argument_value(d['tasks'][0]['command'], "--denser_evaluation", False)
            d['tasks'][0]['resources']['gpuCount'] = 8
            d['tasks'][0]['command'].remove("--bf16") # stage 3 is currently 4x slower with bf16
            #set_argument_value(d['tasks'][0]['command'], "--max_source_length", 1024)
            set_argument_value(d['tasks'][0]['command'], "--generation_max_length", 10)
            set_argument_value(d['tasks'][0]['command'], "--deepspeed", "ds_configs/stage3.config")
            
        set_argument_value(d['tasks'][0]['command'], "--data_dir", f"/data/supervised/multilingual/")
        set_argument_value(d['tasks'][0]['command'], "--max_num_instances_per_task", 1000)
        
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        
        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)


