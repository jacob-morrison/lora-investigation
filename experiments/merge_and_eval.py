from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    Seq2SeqTrainer,
    HfArgumentParser,
    EvalPrediction,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoConfig,
)
from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import PeftModel, PeftConfig
from collections import OrderedDict
import argparse
import sys
from sequence_classification_helpers import Split, T2TMultipleChoiceDataset
import os
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from transformers.trainer_utils import is_main_process

@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use LoRA or not"}
    )
	lora_rank: Optional[int] = field(
        default=None, metadata={"help": "If using LoRA, what rank to use"}
    )
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_flash_attn: Optional[bool] = field(
		default=False,
		metadata={"help": "Do you want to use flash attention with llama"},
	)

@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	task_name: str = field(default="case-hold", metadata={"help": "The name of the task to train on"})
	max_seq_length: int = field(
		default=256,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
			"value if set."
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
	)

def compute_metrics(p: EvalPrediction):
    # logits = p.predictions.transpose([1, 0, 2])[0].transpose()[tokenized_labels].transpose()
    if len(p.predictions) == 2:
        logits = p.predictions[0]
    else:
        logits = p.predictions
    # preds = np.argmax(p.predictions, axis=1)
    # preds = tokenized_labels[np.argmax(logits, axis=1)]
    print('predictions')
    print(logits)
    preds = np.argmax(logits, axis=1)
    print(preds)
    true_labels = p.label_ids.squeeze()
    print(true_labels)
    # Compute macro and micro F1 for 5-class CaseHOLD task
    accuracy = accuracy_score(y_true=true_labels, y_pred = preds)
    macro_f1 = f1_score(y_true=true_labels, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=true_labels, y_pred=preds, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1, 'accuracy': accuracy}

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
# Add custom arguments for computing pre-train loss
parser.add_argument("-b", "--base_model", help = "Which base model", type = str)
model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=data_args.task_name != 'case-hold',
)

if 't5' in model_args.model_name_or_path or 'tk' in model_args.model_name_or_path:
    eval_dataset = \
        T2TMultipleChoiceDataset(
            tokenizer=tokenizer,
            task=data_args.task_name,
            device=training_args.device,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=False,
            mode=Split.dev,
            text_to_text=True,
            max_samples=data_args.max_eval_samples,
        )
else:
    eval_dataset = \
        T2TMultipleChoiceDataset(
            tokenizer=tokenizer,
            task=data_args.task_name,
            device=training_args.device,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=False,
            mode=Split.dev,
            text_to_text=False,
            max_samples=data_args.max_eval_samples,
        )

if training_args.do_predict:
    if 't5' in model_args.model_name_or_path or 'tk' in model_args.model_name_or_path:
        predict_dataset = \
            T2TMultipleChoiceDataset(
                tokenizer=tokenizer,
                task=data_args.task_name,
                device=training_args.device,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=False,
                mode=Split.test,
                text_to_text=True,
                max_samples=data_args.max_predict_samples,
            )
    else:
        predict_dataset = \
            T2TMultipleChoiceDataset(
                tokenizer=tokenizer,
                task=data_args.task_name,
                device=training_args.device,
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=False,
                mode=Split.test,
                text_to_text=False,
                max_samples=data_args.max_predict_samples,
            )

new_state_dict = OrderedDict()

num_classes = {
		'case-hold': 5,
		'qnli': 2,
		'arc-easy': 4,
		'arc-challenge': 4,
		'sciq': 4,
		'hellaswag': 4,
		'mnli': 3,
		'yelp': 2,
		'mathqa': 5,
		'piqa': 2,
	}

# Load pretrained model and tokenizer
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_classes[data_args.task_name],
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    token='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',
)
config.use_cache = False

target_model_path = '/model/'
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
)

if model_args.use_lora:
    print("Merging the lora modules...")
    lora_base_model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)
    lora_model = PeftModel.from_pretrained(lora_base_model, target_model_path)
    model_to_merge = lora_model.base_model.merge_and_unload()
    print("Done merging lora modules")
else:
    model_to_merge = AutoModelForSequenceClassification.from_pretrained(target_model_path)

model_weights = [
    (0.0, 1.0),
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    (1.0, 0.0),
]
base_model.resize_token_embeddings(len(tokenizer))

for target_model_weight, base_model_weight in model_weights:
    for key in base_model.state_dict():
        if base_model.state_dict()[key].shape == model_to_merge.state_dict()[key].shape:
            new_state_dict[key] = torch.mul(base_model.state_dict()[key], base_model_weight) + torch.mul(model_to_merge.state_dict()[key], target_model_weight)
        else:
            print('size mismatch for key: ' + str(key))
            print('base model shape: ' + str(base_model.state_dict()[key].shape))
            print('target model shape:' + str(model_to_merge.state_dict()[key].shape))
            print('exiting now')
            sys.exit(1)

    merged_model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, state_dict = new_state_dict, config=base_model.config)

    # Initialize our Trainer
    if 't5' in model_args.model_name_or_path or 'tk' in model_args.model_name_or_path:
        trainer = Seq2SeqTrainer(
            model=merged_model,
            args=training_args,
            train_dataset=[],
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=merged_model),
            compute_metrics=compute_metrics,
            callbacks=[]
        )
    else:
        trainer = Trainer(
            model=merged_model,
            args=training_args,
            train_dataset=[],
            eval_dataset=eval_dataset,
            data_collator=None,
            compute_metrics=compute_metrics,
            callbacks=[]
        )

    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    if model_args.use_lora:
        prefix = f"lora-{model_args.rank}"
    else:
        prefix = "full-finetuning"

    os.rename(
        os.path.join(training_args.output_dir, 'all_results.json'),
        os.path.join(training_args.output_dir, f'{data_args.task_name}/{model_args.model_name_or_path.replace("/", "-")}-{prefix}-{training_args.learning_rate}-{training_args.seed}/target-weight-{target_model_weight}-base-weight-{base_model_weight}-metrics.json')
    )

