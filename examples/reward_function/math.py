# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
import torch
import numpy as np
import pandas as pd
from mathruler.grader import extract_boxed_content, grade_answer
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

def configure_model_for_gradient_checkpointing(model):
    """
    Configure model to be compatible with gradient checkpointing by disabling caching.
    """
    if hasattr(model, 'config'):
        model.config.use_cache = False
    
    # For Phi models specifically, ensure gradient checkpointing compatibility
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model


def load_model_tokenizer(model_name_or_path):
    # Initialize tokenizer for BPE tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model for attention calculation
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # Force eager attention for attention weight extraction
    )

    # Configure model for gradient checkpointing compatibility
    model = configure_model_for_gradient_checkpointing(model)

    # Force eager attention implementation for attention weight extraction
    if hasattr(model, 'config'):
        model.config._attn_implementation = "eager"
        print("Forced eager attention implementation for attention weight extraction")

    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer


def split_prediction_into_sentences(prediction_text):
    """
    Split prediction text into sentences and return sentence boundaries.
    """
    # Download nltk data if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Split into sentences
    sentences = sent_tokenize(prediction_text)
    
    # Find sentence boundaries in the original text
    sentence_boundaries = []
    current_pos = 0
    
    for sentence in sentences:
        # Find the position of this sentence in the original text
        start_pos = prediction_text.find(sentence, current_pos)
        end_pos = start_pos + len(sentence)
        sentence_boundaries.append((start_pos, end_pos))
        current_pos = end_pos
    
    return sentences, sentence_boundaries

def find_sentence_token_boundaries(sentence_boundaries, prompts, prediction_text, tokenizer):
    """
    Find token positions for each sentence boundary.
    """
    # Combine question and prediction with space
    full_text = prompts + ' ' + prediction_text
    
    # Tokenize the full text
    full_tokens = tokenizer.tokenize(full_text)
    
    # Find the position of space after question
    question_tokens = tokenizer.tokenize(prompts)
    space_token_pos = len(question_tokens)
    
    # Find the position of </think>
    think_token_pos = None
    for i, token in enumerate(full_tokens):
        if '</think>' in token or token == '</think>':
            think_token_pos = i
            break
    
    # Find token boundaries for each sentence
    sentence_token_boundaries = []
    
    for start_char, end_char in sentence_boundaries:
        # Convert character positions to token positions
        # We need to find which tokens correspond to this sentence
        
        # Create text up to the end of this sentence
        text_up_to_sentence_end = prompts + ' ' + prediction_text[:end_char]
        tokens_up_to_sentence_end = tokenizer.tokenize(text_up_to_sentence_end)
        
        # Create text up to the start of this sentence
        text_up_to_sentence_start = prompts + ' ' + prediction_text[:start_char]
        tokens_up_to_sentence_start = tokenizer.tokenize(text_up_to_sentence_start)
        
        # Token boundaries for this sentence
        sentence_start_token = len(tokens_up_to_sentence_start)
        sentence_end_token = len(tokens_up_to_sentence_end)
        
        sentence_token_boundaries.append((sentence_start_token, sentence_end_token))
    
    return sentence_token_boundaries, space_token_pos, think_token_pos

def calculate_attention_matrix(content, tokenizer, model, max_length=2048):
    """
    Calculate attention matrix for a given content using the model's attention weights.
    Based on analysis6.py approach.
    
    Args:
        content: The text content to analyze
        tokenizer: The tokenizer to use
        model: The model to extract attention weights from (if None, will use global model)
        max_length: Maximum sequence length
    
    Returns:
        attention_matrix: A 2D numpy array representing aggregated attention weights
    """
   
    # Tokenize the content
    inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=max_length)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract attention weights
    with torch.no_grad():
        # Ensure caching is disabled for gradient checkpointing compatibility
        if hasattr(model, 'config'):
            original_use_cache = model.config.use_cache
            model.config.use_cache = False
        
        outputs = model(**inputs, output_attentions=True)
        
        # Restore original cache setting
        if hasattr(model, 'config'):
            model.config.use_cache = original_use_cache
    

    # Get attention weights from all layers
    attention_weights = outputs.attentions  # List of [layer_num, batch_size, num_heads, seq_len, seq_len]
    
    # Get attention weights from specific layer 31 (last layer) and head 0
    target_layer = 31
    target_head = 0
    
    # Check if the target layer exists
    if target_layer >= len(attention_weights):
        print(f"Warning: Target layer {target_layer} not found. Using last available layer.")
        target_layer = len(attention_weights) - 1
    
    layer_attention = attention_weights[target_layer][0]  # [num_heads, seq_len, seq_len]
    
    # Check if the target head exists
    if target_head >= layer_attention.shape[0]:
        print(f"Warning: Target head {target_head} not found. Using last available head.")
        target_head = layer_attention.shape[0] - 1
    
    # Extract attention from specific layer and head
    attention_matrix = layer_attention[target_head].cpu().numpy()  # [seq_len, seq_len]
    
    print(f"Using layer {target_layer}, head {target_head}")
    print(f"Attention matrix shape: {attention_matrix.shape}")

    return attention_matrix






def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def length_reward(response: str, answer: str, tokenizer, **kwargs):
    """Reward function that penalizes completions that exceed the token budget.
    Returns lower reward for longer completions.
    """

    
    # Get the token budget from the dataset example
    # Calculate token budget based on the answer length
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    budget = len(answer_tokens)
    
    
    response_tokens = tokenizer.encode(response, add_special_tokens=False)
    num_tokens = len(response_tokens)
    # Count tokens using BPE tokenization
   
    
    # Calculate reward based on budget - shorter is better
    if num_tokens <= budget:
        # Reward shorter responses more than longer ones within budget
        # Use a scaling factor to encourage brevity
        brevity_bonus = max(0, (budget - num_tokens) / budget) * 0.3  # Up to 30% bonus for being shorter
        reward = 1.0 + brevity_bonus
    elif num_tokens >= 10 * budget:
        # Linearly decrease reward as length exceeds budget
        # Penalty factor: reduce reward by 0.1 for each token over budget
        penalty = (num_tokens - 10 * budget) * 0.0002
        reward = max(0, 1 - penalty)
    else:
        penalty = min(0.9, (num_tokens - budget) * 0.001)
        reward = max(0.1, 1 - penalty)
        
    
    return reward


def attention_reward(response: str, prompt: str, tokenizer, model, **kwargs):
    """Reward function that encourage the model to put more attention to </think> tokens,
    which highly influence the final answer.
    """
        # Count tokens using BPE tokenization
    tokens = tokenizer.encode(response, add_special_tokens=False)
    
    # Find the location of </think> token
    think_token_pos = None
    think_token_id = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_token_id) == 1:
        think_token_id = think_token_id[0]
        for i, token in enumerate(tokens):
            if token == think_token_id:
                think_token_pos = i
                break
    
    if think_token_pos is None:
        # No </think> token found, give zero reward
        reward = 0
    
    # Extract the instruction from the first dataset example (assuming all examples have the same instruction)
    combined_text = prompt + ' ' + response
    
    # Calculate attention matrix for this content
    attention_matrix = calculate_attention_matrix(combined_text, tokenizer, model)
    
    print(f"Attention matrix shape: {attention_matrix.shape}")
    
    # Calculate the attention score of </think> tokens on sentence level
    sentence_attention_sums = []
    # Get the sentence token boundaries
    sentences, sentence_boundaries = split_prediction_into_sentences(response)
    sentence_token_boundaries, _, _ = find_sentence_token_boundaries(sentence_boundaries, prompt, response, tokenizer)

    for i, (start_token, end_token) in enumerate(sentence_token_boundaries):
        if start_token < len(attention_matrix) and end_token <= len(attention_matrix) and think_token_pos < len(attention_matrix):
            sentence_attention_weights = attention_matrix[think_token_pos, start_token:end_token]
            sentence_attention_sum = np.mean(sentence_attention_weights)
            sentence_attention_sums.append(sentence_attention_sum)
    
    if sentence_attention_sums:
        reward = np.sum(sentence_attention_sums) * 10
    else:
        reward = 0.0

    
    return reward




def compute_score(
         reward_inputs: list[dict[str, Any]], 
         model_name_or_path: str = 'Qwen/Qwen3-1.7B',
         format_weight: float = 0.05, 
         length_weight: float = 0.25,
         attention_weight: float = 0.2,
         ) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    model, tokenizer = load_model_tokenizer(model_name_or_path)

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        length_score = length_reward(response, reward_input["answer"], tokenizer)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + length_weight * length_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "length": length_score,
            }
        )

    return scores
