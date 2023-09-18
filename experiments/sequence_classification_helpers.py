import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from typing import Any, List, NewType, Optional
import random

import tqdm

from transformers import PreTrainedTokenizer, is_torch_available
import datasets
import torch

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class T2TMultipleChoiceDataset(Dataset):
        """
        PyTorch multiple choice dataset class
        """

        features=List

        def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            task: str,
            device,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            text_to_text: bool=False,
            max_samples: Optional[int] = None,
            model_type: Optional[str]=None,
        ):
            if task == 'case-hold':
                dataset = datasets.load_dataset('lex_glue', 'case_hold')
            elif task == 'qnli': # no test labels
                dataset = datasets.load_dataset('glue', 'qnli')    
            elif task == 'arc-easy':
                dataset = datasets.load_dataset("ai2_arc", 'ARC-Easy')
            elif task == 'arc-challenge':
                dataset = datasets.load_dataset("ai2_arc", 'ARC-Challenge')
            elif task == 'sciq':
                dataset = datasets.load_dataset("sciq")
            elif task == 'mnli': # validation_matched # validation_mismatched
                dataset = datasets.load_dataset("multi_nli")
            elif task == 'hellaswag': # no test labels
                dataset = datasets.load_dataset("hellaswag")
            elif task == 'yelp': # no validation, limit dataset size
                dataset = datasets.load_dataset("yelp_polarity")
            elif task == 'mathqa': # no test labels
                dataset = datasets.load_dataset("math_qa")
            elif task == 'piqa': # no test labels
                dataset = datasets.load_dataset("piqa") 
            else:
                print('invalid task: ' + task)


            if mode == Split.dev:
                if task == 'mnli':
                    examples = dataset['validation_matched']
                elif task == 'yelp':
                    if max_samples is not None:
                        examples = dataset['test']
                    else:
                        examples = dataset['test'][:10000]
                else:
                    examples = dataset['validation']
            elif mode == Split.test:
                if task == 'mnli':
                    examples = dataset['validation_mismatched']
                else:
                    examples = dataset['test']
            elif mode == Split.train:
                examples = dataset['train']
            logger.info("Training examples: %s", len(examples))
            if max_samples is not None:
                examples = examples[:max_samples]
            self.features = convert_examples_to_text_to_text(
                examples,
                max_seq_length,
                tokenizer,
                task,
                device,
                text_to_text=text_to_text,
                mode=mode,
            )

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i):
            return self.features[i]
        
def convert_examples_to_text_to_text(
        examples: datasets.Dataset,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        task: str,
        device,
        include_instruction: bool=False,
        prepare_decoder_input_ids_from_labels: bool=False,
        text_to_text: bool=False,
        model_type: Optional[str]=None,
        mode: Split = Split.train,
):
    """
    Loads a data file into a text to text format
    """

    # put each example together

    if task == 'case-hold':
        choices = [
            'A',
            'B',
            'C',
            'D',
            'E',
        ]
        contexts = examples['context']
        endings = examples['endings']
        labels = examples['label']
    elif task == 'qnli':
        choices = [
            'true',
            'false',
        ]
        contexts = examples['sentence']
        questions = examples['question']
        labels = examples['label']
    elif task == 'arc-easy' or task == 'arc-challenge':
        choices = ['A', 'B', 'C', 'D']
        label_map = {
            '1': 'A',
            '2': 'B',
            '3': 'C',
            '4': 'D',
        }

        def fix_labels(label):
            if label in label_map:
                return label_map[label]
            else:
                return label

        contexts = examples['question']
        endings = examples['choices']
        labels = examples['answerKey']
        labels = list(map(fix_labels, labels))
    elif task == 'sciq':
        choices = ['A', 'B', 'C', 'D']
        contexts = examples['question']
        endings = list(zip(examples['distractor1'], examples['distractor2'], examples['distractor3'], examples['correct_answer']))
        labels = []
    elif task == 'mnli':
        choices = [
            'true',
            'neutral',
            'false',
        ]
        contexts = examples['premise']
        endings = examples['hypothesis']
        labels = examples['label']
    elif task == 'hellaswag':            
        choices = ['A', 'B', 'C', 'D']

        def map_label_to_choice(str_label):
            return choices[int(str_label)]

        contexts = examples['ctx']
        endings = examples['endings']
        labels = examples['label']
        labels = list(map(map_label_to_choice, labels))
        print(labels)
    elif task == 'yelp':
        contexts = examples['text']
        labels = examples['label']
    elif task == 'piqa':
        choices = ['A', 'B']
        contexts = examples['goal']
        endings = list(zip(examples['sol1'], examples['sol2']))
        labels = examples['label']
    elif task == 'mathqa':
        choices = ['A', 'B', 'C', 'D', 'E']
        contexts = examples['Problem']
        endings = examples['options']
        labels = examples['correct']

        def switch_to_numbers(label):
            label_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            return label_map[label]

        labels = list(map(switch_to_numbers, labels))
    else:
        print('invalid task: ' + task)
    
    processed_examples = []
    input_endings = []
    labels_list = []
    for (ex_index, context) in tqdm.tqdm(enumerate(contexts), desc="convert examples to t2t"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(contexts)))
        if task == 'case-hold':
            processed_example = context + '.'
            ending = ' '
            if include_instruction:
                if task == 'case-hold':
                    processed_example = 'What is the correct holding statement for the following text?\nText: ' + processed_example

            for choice, option in zip(choices, endings[ex_index]):
                ending += '\n(' + choice + '): ' + option + ' '
            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
            label = labels[ex_index]
        elif task == 'qnli':
            processed_example = context + '.'
            ending = ' ' + questions[ex_index] + ' '
            if include_instruction:
                pass
            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
            label = labels[ex_index]
        elif task == 'arc-easy' or task == 'arc-challenge':
            processed_example = context + '.'
            ending = ' '
            for choice, option in zip(choices, endings[ex_index]):
                ending += '\n(' + choice + '): ' + option + ' '
            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
            print(labels)
            if labels[ex_index] not in choices:
                print('weird label: ' + str(labels[ex_index]))
                continue
            label = choices.index(labels[ex_index])
        elif task == 'sciq':
            processed_example = context + '.'
            ending = ' '
            (distractor1, distractor2, distractor3, correct_answer) = endings[ex_index]
            options = [
                (distractor1, 'incorrect'),
                (distractor2, 'incorrect'),
                (distractor3, 'incorrect'),
                (correct_answer, 'correct'),
            ]
            random.shuffle(options)
            for choice, (option, status) in zip(choices, options):
                ending += '\n(' + choice + '): ' + option + ' '
                if status == 'correct':
                    labels.append(choice)
            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
            label = choices.index(labels[ex_index])
        elif task == 'hellaswag':
            processed_example = context
            ending = ' '
            for choice, option in zip(choices, endings[ex_index]):
                ending += '\n(' + choice + '): ' + option + ' '
            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
            label = choices.index(labels[ex_index])
        elif task == 'mnli':
            if labels[ex_index] == -1:
                continue
            processed_example = context + '.'
            ending = ' ' + endings[ex_index] + ' '
            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
            label = labels[ex_index]
        elif task == 'yelp':
            processed_example = context + '.'
            ending = ''
            if ex_index == 0:
                print(processed_example)
                print(ending)
            label = labels[ex_index]
        elif task == 'piqa':
            processed_example = context + '.'
            ending = ' '
            for choice, option in zip(choices, endings[ex_index]):
                ending += '\n(' + choice + '): ' + option + ' '

            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
            label = labels[ex_index]
        elif task == 'mathqa':
            processed_example = context
            ending = ' '
            for choice, option in zip(choices, ending.split(',')):
                ending += '\n(' + choice + '): ' + option.split(')')[-1].strip() + ' '

            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)

            label = labels[ex_index]
        else:
            print('invalid task: ' + task)
    
        processed_examples.append(processed_example)
        input_endings.append(ending)
        if text_to_text:
            labels_list.append([int(label)])
        else:
            labels_list.append(torch.tensor(int(label)))

    # if len(input_endings[0]) > 0:
    model_inputs = tokenizer(
        processed_examples,
        input_endings,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # else:
    #     model_inputs = tokenizer(
    #         processed_examples,
    #         add_special_tokens=True,
    #         max_length=max_length,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    model_inputs["labels"] = labels_list

    outputs = []
    padded = 0
    if 'token_type_ids' in model_inputs:
        for input_ids, attention_mask, token_type_ids, label in zip(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs['token_type_ids'],
            model_inputs["labels"]
        ):
            if 0 in attention_mask:
                padded += 1
            elif mode == Split.train:
                print('skipping train example')
                continue
            outputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': label,
            })
    else:
        for input_ids, attention_mask, label in zip(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["labels"]
        ):
            if 0 in attention_mask:
                padded += 1
            elif mode == Split.train:
                print('skipping train example')
                continue
            outputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label,
            })
    print('total: ' + str(len(model_inputs['input_ids'])))
    print('padded: ' + str(padded))

    return outputs
