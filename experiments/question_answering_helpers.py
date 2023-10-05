import string
import re
import json
import sys
import os
import argparse
import logging
from collections import Counter
from rouge import rouge_scorer
from typing import Optional, Tuple, Union
from transformers import (
    AutoTokenizer, 
    LlamaPretrainedModel,
    LlamaModel,
    QuestionAnsweringModelOutput,
)
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import collections


logger = logging.getLogger(__name__)

class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word. 
        # But for the first word of a sentence, there is no space before it. 
        # So, we remove all the added spaces ("Ġ"). 
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_f1(prediction, ground_truth, xlingual=False):
    gold_toks = get_tokens(ground_truth)
    pred_toks = get_tokens(prediction)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL, f1 = 0., 0., 0., 0.
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        f1 += metric_max_over_ground_truths(
            compute_f1, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    f1 = 100.0 * f1 / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL, "f1": f1}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     with open(args.predictions) as fin:
#         examples = [json.loads(l) for l in fin]

#     predictions = [e["prediction"] for e in examples]
#     references = [e["Instance"]["output"] for e in examples]
#     tasks = []
#     for e in examples:
#         if e["Task"] == "task121_atomic_question_rewriting":
#             e["Task"] = "task121_zest_question_rewriting"
#         tasks.append(e["Task"])

#     results = compute_metrics(predictions, references, xlingual=args.track == "xlingual")
#     print("======== Overall Metrics ========")
#     print("all_rougeL", results["rougeL"])
#     print("all_EM", results["exact_match"])
#     print("all_f1", results["f1"])
#     print()
    
#     category_metrics = [
#         ("Textual Entailment", "exact_match"),
#         ("Cause Effect Classification", "exact_match"),
#         ("Coreference Resolution", "exact_match"),
#         ("Dialogue Act Recognition", "exact_match"),
#         ("Answerability Classification", "exact_match"),
#         ("Word Analogy", "exact_match"),
#         ("Overlap Extraction", "rougeL"),
#         ("Keyword Tagging", "rougeL"),
#         ("Question Rewriting", "rougeL"),
#         ("Title Generation", "rougeL"),
#         ("Data to Text", "rougeL"),
#         ("Grammar Error Correction", "rougeL"),
#         ("SQuAD-like", "f1"),
#     ]
#     category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}

#     if args.compute_per_category_metrics:
#         print("======== Metrics per category ========")
#         task_category = {}
#         for task in set(tasks):
#             with open(os.path.join("./data/tasks/", task+".json")) as fin:
#                 task_data = json.load(fin)
#                 task_category[task] = "_".join(task_data["Categories"][0].lower().split())
#         categories = [task_category[e["Task"]] for e in examples] 
#         results.update(compute_grouped_metrics(predictions, references, categories, xlingual=args.track=="xlingual"))
        
#         for category, metric in category_metrics.items():
#             # category = "_".join(category.lower().split())
#             if f"{metric}_for_{category}" in results:
#                 print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
#         print()
            
#     if args.compute_per_task_metrics:
#         print("======== Metrics per task ========")
#         results_by_task = compute_grouped_metrics(predictions, references, tasks, xlingual=args.track=="xlingual")
#         for task in sorted(list(set(tasks))):
#             category = task_category[task]
#             metric = category_metrics[category]
#             print(task, results_by_task[f"{metric}_for_{task}"])
#         print()

# preprocess squad this way?
for i in range(len(inputs['offset_mapping'])):
    if inputs['offset_mapping'][i][0] == start_char - 1:
        start_token = i
    elif inputs['offset_mapping'][i][0] == end_char + 1:
        end_token = i
        break

class LlamaForQuestionAnswering(LlamaPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )