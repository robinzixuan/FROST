"""
GSM8K dataset module for loading and processing GSM8K math problem data
(from Hugging Face hub instead of local JSON files).
"""

import os
import re
import logging
from torch.utils.data import Dataset
from datasets import load_dataset
from utils import *

logger = logging.getLogger(__name__)


class GSM8K(Dataset):
    """
    Dataset class for GSM8K math problems (downloaded from Hugging Face).
    """

    def __init__(self, args, with_reasoning=True, cache=True, name=None, budget=None, split='train'):
        """
        Initialize the GSM8K dataset.

        Args:
            args: Command line arguments containing configuration
            with_reasoning (bool): Whether to include step-by-step reasoning
            cache (bool): Whether to cache the processed data
            name (str, optional): Name of the dataset variant
            budget (int, optional): Token budget for prompt generation
            split (str): Dataset split ('train' or 'test')
        """
        from utils import create_gsm8k_prompt
        self.args = args
        self.cache = cache
        self.split = split
        self.with_reasoning = with_reasoning
        if budget is not None:
            global gsm8k_prompts
            gsm8k_prompts = create_gsm8k_prompt(budget)
        self.gsm8k_std_data_sets = self._load_data()
        logger.info(f"Loading dataset from the GSM8K-{split}!")
        self.dataset = sum(self.gsm8k_std_data_sets.values(), [])

    def _generate_configs(self):
        config = [{
            'abbr': 'GSM8K',
            'path': 'gsm8k',
            'name': f'GSM8K-{self.split}',
            'reader_cfg': {
                'input_column': 'question',
                'output_column': 'answer'
            },
            'meta_prompt': {
                'round': gsm8k_prompts['reasoning'] if self.with_reasoning else gsm8k_prompts['no_reasoning']
            }
        }]
        return config

    @staticmethod
    def _generate_std_subset(raw_data, cfg):
        examples = []
        prompt_template = cfg["meta_prompt"]["round"][0]['prompt']
        for item in raw_data:
            examples.append(dict(
                gold=item['answer'],
                reasoning_process_main=item['reasoning_process_main'],
                reasoning_process_socratic=item['reasoning_process_socratic'],
                round=[
                    {
                        "role": "HUMAN",
                        "prompt": prompt_template.replace("{question}", item['question'])
                    },
                    {
                        "role": "BOT",
                        "prompt": "{answer}"
                    },
                ]
            ))
        return examples

    def _generate_formal_info(self, cfg):
        """
        Load GSM8K from Hugging Face and extract fields.
        """
        def find_answer(text):
            pattern = r"#### (-?\d+(?:\.\d+)?|\d+/\d+)"
            return re.findall(pattern, text)[-1]

        # Load main dataset
        raw = load_dataset("gsm8k", "main", split=self.split)
        raw_socratic = load_dataset("gsm8k", "socratic", split=self.split)

        data = []
        for idx in range(len(raw)):
            assert raw[idx]['question'] == raw_socratic[idx]['question']
            data.append({
                'question': raw[idx]['question'].strip(),
                'answer': find_answer(raw[idx]['answer']),
                'reasoning_process_main': raw[idx]['answer'],
                'reasoning_process_socratic': raw_socratic[idx]['answer'],
            })
            assert data[-1]['answer'] in raw[idx]['answer'] and data[-1]['answer'] in raw_socratic[idx]['answer']
        return data

    def _load_data(self):
        from utils import save_to_jsonl
        cfgs = self._generate_configs()
        std_data_sets = {}
        for cfg in cfgs:
            info = self._generate_formal_info(cfg)
            std_subset = self._generate_std_subset(info, cfg)
            os.makedirs('./.cache', exist_ok=True)
            save_to_jsonl(std_subset, os.path.join('./.cache', cfg["abbr"]) + f'-{self.split}.jsonl')
            std_data_sets[cfg["abbr"]] = std_subset
        return std_data_sets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
