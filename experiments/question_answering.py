#!/usr/bin/env python
# coding=utf-8
""" Finetuning models on CaseHOLD (e.g. Bert, RoBERTa, LEGAL-BERT)."""

import logging
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import random

import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import shutil
import glob
from transformers.trainer_utils import is_main_process
from sequence_classification_helpers import Split, T2TMultipleChoiceDataset
from sklearn.metrics import f1_score, accuracy_score
from peft import get_peft_model, LoraConfig, TaskType
import json
from question_answering_helpers import compute_metrics, compute_grouped_metrics, LlamaForQuestionAnswering


logger = logging.getLogger(__name__)

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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # Add custom arguments for computing pre-train loss
    parser.add_argument("--ptl", type=bool, default=False)
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        logger.info(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    if model_args.use_flash_attn:
        from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    transformers.logging.set_verbosity_error()

    # Set seed
    set_seed(training_args.seed)

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
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        token='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',
    )
    config.use_cache = False

    print('num classes: ' + str(num_classes[data_args.task_name]))

    if config.model_type == 'big_bird':
        config.attention_type = 'original_full'
    elif config.model_type == 'longformer':
        config.attention_window = [data_args.max_seq_length] * config.num_hidden_layers

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # Default fast tokenizer is buggy on CaseHOLD task, switch to legacy tokenizer
        use_fast=data_args.task_name != 'case-hold', # True,
        token='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',
    )
    if 'gpt2' in model_args.model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        if 'gpt2' in model_args.model_name_or_path:
            config.pad_token_id = config.eos_token_id

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        print('adding llama tokens')
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    else:
        print('what is the tokenizer?')
        print(tokenizer)
    # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        # num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    models = [
        'gpt2',
        'llama',
        'deberta-v2',
        't5',
    ]
    

    print('model type')
    print(config.model_type)
    task_type = TaskType.SEQ_2_SEQ_LM # add causal lm
    if config.model_type in models:
        if config.model_type == 'llama':
            model = LlamaForQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                token='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',
                torch_dtype=torch.bfloat16,
            )
        elif config.model_type == 't5':
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                token='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',
            )
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                token='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',
            )

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type=task_type, inference_mode=False, r=model_args.lora_rank, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    train_dataset = None
    eval_dataset = None

    if training_args.do_train:
        if config.model_type == 't5':
            train_dataset = \
                T2TMultipleChoiceDataset(
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    device=training_args.device,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.train,
                    text_to_text=True,
                    max_samples=data_args.max_train_samples,
                )
        elif config.model_type in sequence_classification_models:
            train_dataset = \
                T2TMultipleChoiceDataset(
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    device=training_args.device,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.train,
                    text_to_text=False,
                    max_samples=data_args.max_train_samples,
                )
        else:
            print('This is broken')

    # If do_eval or do_predict passed, eval_dataset by default loads dev split from file named dev.csv in data directory
    if training_args.do_eval:
        if config.model_type == 't5':
            eval_dataset = \
                T2TMultipleChoiceDataset(
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    device=training_args.device,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.dev,
                    text_to_text=True,
                    max_samples=data_args.max_eval_samples,
                )
        elif config.model_type in sequence_classification_models:
            eval_dataset = \
                T2TMultipleChoiceDataset(
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    device=training_args.device,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.dev,
                    text_to_text=False,
                    max_samples=data_args.max_eval_samples,
                )
        else:
            print('This is broken')

    if training_args.do_predict:
        if config.model_type == 't5':
            predict_dataset = \
                T2TMultipleChoiceDataset(
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    device=training_args.device,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.test,
                    text_to_text=True,
                    max_samples=data_args.max_predict_samples,
                )
        elif config.model_type in sequence_classification_models:
            predict_dataset = \
                T2TMultipleChoiceDataset(
                    tokenizer=tokenizer,
                    task=data_args.task_name,
                    device=training_args.device,
                    max_seq_length=data_args.max_seq_length,
                    overwrite_cache=data_args.overwrite_cache,
                    mode=Split.test,
                    text_to_text=False,
                    max_samples=data_args.max_predict_samples,
                )
        else:
            print('This is broken')
            
    print('args')
    print(data_args)
    print(training_args)
    print(model_args)
    print(train_dataset)

    # if config.model_type != 't5':
    if training_args.do_train:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset[:data_args.max_eval_samples]

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset[:data_args.max_predict_samples]
    
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
    
    # Metric
    def compute_ni_metrics(dataset, preds, save_prefix=None):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [e["Instance"]["output"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
        result.update(result_per_task)
        categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    if config.model_type == 't5':
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=compute_ni_metrics,
            callbacks=[]
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=None,
            compute_metrics=compute_metrics,
            callbacks=[]
        )

    print('device info')
    print(model.device)

    print('listing directory contents')
    for (root,dirs,files) in os.walk(training_args.output_dir, topdown=True):
        print(root)
        print(dirs)
        print(files)
        print('--------------------------------')

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    print('detecting last checkpoint')
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        print('calling function')
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.info(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    print('last checkpoint: ' + str(last_checkpoint))

    # TODO: uncomment this after debugging

    # if last_checkpoint is None:
    #     logger.info("*** Evaluate ***")

    #     metrics = trainer.evaluate()

    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        print('model here')
        # print(peft_config)
        print(config)
        print(model)
        trainer.train(
            resume_from_checkpoint=checkpoint
            # model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # # Re-save the tokenizer for model sharing
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation on eval_dataset
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict on eval_dataset
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
        # if trainer.is_world_process_zero():
        #     with open(output_predict_file, "w") as writer:
        #         for index, pred_list in enumerate(predictions):
        #             if len(pred_list.shape) > 1:
        #                 pred_list = pred_list.squeeze()
        #             pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
        #             writer.write(f"{index}\t{pred_line}\n")

    # Clean up checkpoints
    if is_main_process(training_args.local_rank):
        checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
        for checkpoint in checkpoints:
            shutil.rmtree(checkpoint)

    if is_main_process(training_args.local_rank):
        os.rename(training_args.output_dir + 'all_results.json', training_args.output_dir + 'metrics.json')

if __name__ == "__main__":
    main()
