import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
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
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            text_to_text: bool=False,
            max_samples: Optional[int] = None,
        ):
            dataset = datasets.load_dataset('lex_glue', task)

            if text_to_text:
                if mode == Split.dev:
                    examples = dataset['validation']
                elif mode == Split.test:
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
                )

        # NEED TO IMPLEMENT THESE CORRECTLY
        def __len__(self):
            return len(self.features)

        def __getitem__(self, i):
            return self.features[i]

    class MultipleChoiceDataset(Dataset):
        """
        PyTorch multiple choice dataset class
        """

        features: List[InputFeatures]

        def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            text_to_text: bool=False,
        ):
            dataset = datasets.load_dataset('lex_glue', task)

            tokenizer_name = re.sub('[^a-z]+', ' ', tokenizer.name_or_path).title().replace(' ', '')
            cached_features_file = os.path.join(
                '.cache',
                task,
                "cached_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer_name,
                    str(max_seq_length),
                    task,
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            if not os.path.exists(os.path.join('.cache', task)):
                if not os.path.exists('.cache'):
                    os.mkdir('.cache')
                os.mkdir(os.path.join('.cache', task))
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {task}")
                    if mode == Split.dev:
                        examples = dataset['validation']
                    elif mode == Split.test:
                        examples = dataset['test']
                    elif mode == Split.train:
                        examples = dataset['train']
                    logger.info("Training examples: %s", len(examples))
                    self.features = convert_examples_to_features(
                        examples,
                        max_seq_length,
                        tokenizer,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]
        
def convert_examples_to_text_to_text(
        examples: datasets.Dataset,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        include_instruction: bool=False,
        prepare_decoder_input_ids_from_labels: bool=False,
):
    """
    Loads a data file into a text to text format
    """

    # put each example together
    # TODO: are we including an instruction? idk, probably not if we're using raw T5, yes if multitask?

    choices = [
        '(A)',
        '(B)',
        '(C)',
        '(D)',
        '(E)',
    ]
    contexts = examples['context']
    endings = examples['endings']
    labels = examples['label']
    inputs = []
    processed_examples = []
    labels_list = []
    for (ex_index, context) in tqdm.tqdm(enumerate(contexts), desc="convert examples to t2t"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(contexts)))
        processed_example = context + '. '
        if include_instruction:
            pass

        for choice, option in zip(choices, endings[ex_index]):
            processed_example += ' ' + choice + ' ' + option
        processed_examples.append(processed_example)
        labels_list.append(choices[int(labels[ex_index])].replace('(', '').replace(')', ''))

        # processed_examples.append(processed_example)
        # labels_list.append(choices[int(example['label'])])
    model_inputs = tokenizer(
        processed_examples,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        tokenized_labels = tokenizer(
            labels_list,
            max_length=max_length,
            padding="max_length",
            # return_tensors=self.return_tensors,
            truncation=True,
            # pad_to_multiple_of=self.pad_to_multiple_of
        )
        # label_mask = labels["attention_mask"].bool()
        # TODO: fix label_pad_token_id?
        # label_pad_token_id = -100
    model_inputs["labels"] = tokenized_labels["input_ids"]#.masked_fill(~label_mask, label_pad_token_id)

        # TODO: I think T5 takes care of this automatically
        # if prepare_decoder_input_ids_from_labels:
            # decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            # model_inputs["decoder_input_ids"] = decoder_input_ids

        # inputs.append(model_inputs)

    # for f in inputs[:2]:
    #     logger.info("*** Example ***")
    #     logger.info("input: %s" % f)
    
    # return inputs

    outputs = []

    for input_ids, attention_mask, label in zip(model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["labels"]):
        # print('shapes')
        # print(input_ids)
        # print(attention_mask)
        # print(label)
        input_ids = torch.tensor(input_ids)#.unsqueeze(0)
        attention_mask = torch.tensor(attention_mask)#.unsqueeze(0)
        label = torch.tensor(label)
        # print(input_ids)
        # print(attention_mask)
        # print(label)
        # input_ids = input_ids.unsqueeze(0)
        # attention_mask = attention_mask.unsqueeze(0)
        outputs.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label,
        })

    # print('model inputs')
    # print(model_inputs)

    return outputs


def convert_examples_to_features(
    examples: datasets.Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, ending in enumerate(example['endings']):
            context = example['context']
            inputs = tokenizer(
                context,
                ending,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )

            choices_inputs.append(inputs)
        
        label = example['label']

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features
