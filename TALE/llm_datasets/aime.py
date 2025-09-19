"""
AIME 2024 dataset module for loading and processing AIME math problem data.
"""

import os
import logging
from torch.utils.data import Dataset
from datasets import load_dataset
from utils import *

logger = logging.getLogger(__name__)


class AIME2024(Dataset):
    """
    Dataset class for AIME 2024 math problems (from Hugging Face).
    """

    def __init__(self, args, with_reasoning=True, cache=True, name=None, budget=None, split="train"):
        """
        Initialize the AIME2024 dataset.

        Args:
            args: Command line arguments containing configuration
            with_reasoning (bool): Whether to include step-by-step reasoning (future-proof)
            cache (bool): Whether to cache the processed data
            name (str, optional): Name of the dataset variant
            budget (int, optional): Token budget for prompt generation
            split (str): Dataset split ('train' or 'test')
        """
        from utils import aime_prompts  # reuse prompt template logic
        self.args = args
        self.cache = cache
        self.split = split
        self.with_reasoning = with_reasoning

        if budget is not None:
            global aime_prompts
            aime_prompts = aime_prompts(budget)  # reuse GSM8K prompt factory

        self.aime_std_data_sets = self._load_data()
        logger.info(f"Loading dataset from AIME2024-{split}!")
        self.dataset = sum(self.aime_std_data_sets.values(), [])

    def _generate_configs(self):
        """
        Generate configuration for dataset loading.
        """
        config = [{
            "abbr": "AIME2024",
            "path": "HuggingFaceH4/aime_2024",
            "name": f"AIME2024-{self.split}",
            "reader_cfg": {
                "input_column": "problem",
                "output_column": "answer"
            },
            "meta_prompt": {
                "round": aime_prompts["reasoning"] if self.with_reasoning else aime_prompts["no_reasoning"]
            }
        }]
        return config

    @staticmethod
    def _generate_std_subset(raw_data, cfg):
        """
        Generate standardized subset of the dataset.
        """
        examples = []
        prompt_template = cfg["meta_prompt"]["round"][0]["prompt"]
        for item in raw_data:
            print(item)
            examples.append(dict(
                gold=item["answer"],
                reasoning_process_main=None,     # AIME2024 has no reasoning chains
                reasoning_process_socratic=None, # placeholder for consistency
                round=[
                    {
                        "role": "HUMAN",
                        "prompt": prompt_template.replace("{question}", item["question"])
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
        Load AIME 2024 dataset from Hugging Face hub and extract fields.
        """
        raw = load_dataset(cfg["path"], split=self.split)
        data = []
        for item in raw:
            data.append({
                "question": item["problem"].strip(),
                "answer": str(item["answer"]).strip(),  # answers are integers
                "reasoning_process_main": None,
                "reasoning_process_socratic": None,
            })
        return data

    def _load_data(self):
        """
        Load and process the dataset.
        """
        from utils import save_to_jsonl
        cfgs = self._generate_configs()
        std_data_sets = {}
        for cfg in cfgs:
            info = self._generate_formal_info(cfg)
            std_subset = self._generate_std_subset(info, cfg)
            os.makedirs("./.cache", exist_ok=True)
            save_to_jsonl(std_subset, os.path.join("./.cache", cfg["abbr"]) + f"-{self.split}.jsonl")
            std_data_sets[cfg["abbr"]] = std_subset
        return std_data_sets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]
