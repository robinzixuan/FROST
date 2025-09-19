"""
MATH-500 dataset module for loading and processing MATH-500 problem data.
"""

import os
import logging
from torch.utils.data import Dataset
from datasets import load_dataset
from utils import *

logger = logging.getLogger(__name__)


class MATH500(Dataset):
    """
    Dataset class for MATH-500 problems (from Hugging Face).
    """

    def __init__(self, args, with_reasoning=True, cache=True, name=None, budget=None, split="train"):
        """
        Initialize the MATH-500 dataset.

        Args:
            args: Command line arguments containing configuration
            with_reasoning (bool): Whether to include step-by-step reasoning (future-proof)
            cache (bool): Whether to cache the processed data
            name (str, optional): Name of the dataset variant
            budget (int, optional): Token budget for prompt generation
            split (str): Dataset split ('train' or 'test')
        """
        from utils import create_math_prompt  # reuse prompt template logic
        self.args = args
        self.cache = cache
        self.split = split
        self.with_reasoning = with_reasoning

        if budget is not None:
            global math_prompts
            math_prompts = create_math_prompt(budget)  # reuse GSM8K prompt factory

        self.math500_std_data_sets = self._load_data()
        logger.info(f"Loading dataset from MATH-500-{split}!")
        self.dataset = sum(self.math500_std_data_sets.values(), [])

    def _generate_configs(self):
        """
        Generate configuration for dataset loading.
        """
        config = [{
            "abbr": "MATH500",
            "path": "HuggingFaceH4/MATH-500",
            "name": f"MATH500-{self.split}",
            "reader_cfg": {
                "input_column": "problem",
                "output_column": "answer"
            },
            "meta_prompt": {
                "round": math_prompts["reasoning"] if self.with_reasoning else math_prompts["no_reasoning"]
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
            examples.append(dict(
                gold=item["answer"],
                reasoning_process_main=None,     # no reasoning chains provided
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
        Load MATH-500 dataset from Hugging Face hub and extract fields.
        """
        raw = load_dataset(cfg["path"], split=self.split)
        data = []
        for item in raw:
            data.append({
                "question": item["problem"].strip(),
                "answer": str(item["answer"]).strip(),
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
