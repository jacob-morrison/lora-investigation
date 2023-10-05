import csv
from pprint import pprint
import pandas as pd
import os

seeds = [
    '1',
    '2',
    '3',
]

models = [
    'llama-7b',
    't5-xxl',
    'tk-xxl',
    
    # 't5-xl',
    # 'tk-xl',
    # 'gpt2-xl',
    # 'deberta-v2-xl',
]

tasks = [
    'case-hold',
    'sciq',
    'squad',
]

start_dir = '/net/nfs.cirrascale/allennlp/jacobm/lora-investigation/' # 'llama2-7b/sciq/lora_2521/'
print(os.walk(start_dir))