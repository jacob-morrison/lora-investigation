import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping

import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available, PreTrainedTokenizerBase
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
            if task == 'case_hold':
                dataset = datasets.load_dataset('lex_glue', task)
            elif task == 'qnli':
                dataset = datasets.load_dataset('glue', task)

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
                task,
                device,
                text_to_text=text_to_text,
                mode=mode,
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
            device,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            text_to_text: bool=False,
        ):
            if task == 'case_hold':
                dataset = datasets.load_dataset('lex_glue', task)
            elif task == 'qnli':
                dataset = datasets.load_dataset('glue', task)

            tokenizer_name = re.sub('[^a-z]+', ' ', tokenizer.name_or_path).title().replace(' ', '')
            if tokenizer_name == 'RobertaBase':
                tokenizer_name += '2'
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
                        task,
                        device,
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
    # TODO: are we including an instruction? idk, probably not if we're using raw T5, yes if multitask?

    if task == 'case_hold':
        choices = [
            'A',
            'B',
            'C',
            'D',
            'E',
        ]
        # label_map = {
        #     '(A)': 0,
        #     '(B)': 1,
        #     '(C)': 2,
        #     '(D)': 3,
        #     '(E)': 4,
        # }
        contexts = examples['context']
        endings = examples['endings']
        labels = examples['label']
    elif task == 'qnli':
        choices = [
            'true',
            'false',
        ]
        # label_map = {
        #     'true': 0,
        #     'false': 1,
        # }
        contexts = examples['sentence']
        questions = examples['question']
        labels = examples['label']
    
    # with tokenizer.as_target_tokenizer():
    #     tokenized_labels = tokenizer(
    #         choices,
    #         max_length=1, #max_length,
    #         padding="max_length",
    #         add_special_tokens=False,
    #         return_tensors="pt",
    #         truncation=False,
    #         # pad_to_multiple_of=self.pad_to_multiple_of
    #     )
    # tokenized_labels = tokenized_labels.input_ids.squeeze(1).numpy()

    # label_map = {}
    # for i in range(len(choices)):
    #     label_map[choices[i]] = tokenized_labels[i]
    
    inputs = []
    processed_examples = []
    input_endings = []
    labels_list = []
    for (ex_index, context) in tqdm.tqdm(enumerate(contexts), desc="convert examples to t2t"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(contexts)))
        if task == 'case_hold':
            processed_example = context + '.'
            ending = ' '
            if include_instruction:
                if task == 'case_hold':
                    processed_example = 'What is the correct holding statement for the following text?\nText: ' + processed_example

            for choice, option in zip(choices, endings[ex_index]):
                ending += '\n(' + choice + '): ' + option + ' '
            ending += '\nOutput: '
            if ex_index == 0:
                print(processed_example)
                print(ending)
        elif task == 'qnli':
            processed_example = context + '.'
            ending = ' ' + questions[ex_index] + ' '
            if include_instruction:
                pass
        processed_examples.append(processed_example)
        input_endings.append(ending)
        # print(processed_example)
        # print(len(processed_example.split()))
        # label_list = list(range(len(choices)))
        if text_to_text:
            # labels_list.append([int(labels[ex_index])])
            labels_list.append(choices[labels[ex_index]])
            # true_label = int(labels[ex_index])
            # labels_list.append([1 if label == true_label else 0 for label in label_list])
        else:
            labels_list.append(torch.tensor(int(labels[ex_index])))
            # true_label = int(labels[ex_index])
            # labels_list.append([1 if label == true_label else 0 for label in label_list])

        # processed_examples.append(processed_example)
        # labels_list.append(choices[int(example['label'])])
    # print(processed_examples)
    model_inputs = tokenizer(
        processed_examples,
        input_endings,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    if text_to_text:
        with tokenizer.as_target_tokenizer():
            labels_list = tokenizer(
                labels_list,
                max_length=1, #max_length,
                padding="max_length",
                add_special_tokens=False,
                return_tensors="pt",
                truncation=False,
                # pad_to_multiple_of=self.pad_to_multiple_of
            )['input_ids']

        # label_mask = labels["attention_mask"].bool()
        # TODO: fix label_pad_token_id?
        # label_pad_token_id = -100
    model_inputs["labels"] = labels_list #tokenized_labels["input_ids"]#.masked_fill(~label_mask, label_pad_token_id)

        # TODO: I think T5 takes care of this automatically
        # if prepare_decoder_input_ids_from_labels:
            # decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            # model_inputs["decoder_input_ids"] = decoder_input_ids

        # inputs.append(model_inputs)

    outputs = []

    padded = 0

    if 'token_type_ids' in model_inputs:
        for input_ids, attention_mask, token_type_ids, label in zip(model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs['token_type_ids'], model_inputs["labels"]):
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print()
            if 0 in attention_mask:
                padded += 1
            elif mode == Split.train:# or mode == Split.dev:
                if mode == Split.train:
                    print('skipping train example')
                else:
                    print('skipping validation example')
                continue
            outputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': label,
            })
    else:
        for input_ids, attention_mask, label in zip(model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["labels"]):
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print()
            if 0 in attention_mask:
                padded += 1
            elif mode == Split.train:# or mode == Split.dev:
                if mode == Split.train:
                    print('skipping train example')
                else:
                    print('skipping validation example')
                continue
            outputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label,
            })
    print('total: ' + str(len(model_inputs['input_ids'])))
    print('padded: ' + str(padded))

    # print('model inputs')
    # print(model_inputs)

    return outputs


def convert_examples_to_features(
    examples: datasets.Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    task: str,
    device,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        if task == 'case_hold':
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
        elif task == 'qnli':
            sentence = example['sentence']
            question = example['question']
            inputs = tokenizer(
                sentence,
                question,
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
        
    max_input_ids_length = -1
    labels_list = []
    for f in features:
        for input_id in f.input_ids:
            if len(input_id) > max_input_ids_length:
                max_input_ids_length = len(input_id)
        
        if f.label not in labels_list:
            labels_list.append(f.label)
        
    print('max input id length: ' + str(max_input_ids_length))
    print('labels list: ' + str(labels_list))

    return features


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

@dataclass
class MyDataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["labels"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
