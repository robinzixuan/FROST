#!/usr/bin/env python3
"""
Evaluation script for Maxwell-Jia/AIME_2024 dataset
Evaluates model performance on math problems, extracting answers from \boxed{} format
and comparing with ground truth answers.
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Qwen_attention import Qwen3AttentionExtrea
import numpy as np
from tqdm import tqdm
import time

def load_aime_dataset():
    """Load the AIME dataset directly from Hugging Face."""
    from datasets import load_dataset
    print("Loading AIME_2024 dataset from Hugging Face...")
    dataset = load_dataset("Maxwell-Jia/AIME_2024")
    
    # Get the test split (or use train if test doesn't exist)
    if 'test' in dataset:
        data = dataset['test']
    elif 'train' in dataset:
        data = dataset['train']
    else:
        # If no specific split, use the first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
    
    # Convert to list of dictionaries
    data_list = [{"Problem": item["Problem"], "Answer": item["Answer"]} for item in data]
    print(f"Loaded {len(data_list)} problems from {list(dataset.keys())[0]} split")
    return data_list

def extract_boxed_answer(response: str):
    """
    Extract answer from \boxed{} format in the model response.
    Returns the content inside \boxed{} or None if not found.
    """
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(boxed_pattern, response)
    
    if match:
        return match.group(1).strip()
    
    # Also look for boxed without backslash (common in some models)
    boxed_pattern2 = r'boxed\{([^}]*)\}'
    match = re.search(boxed_pattern2, response)
    
    if match:
        return match.group(1).strip()
    
    return None

def normalize_answer(answer: str):
    """Normalize answer for comparison (remove spaces, convert to lowercase, etc.)."""
    if not answer:
        return ""
    
    # Remove common LaTeX commands and symbols
    answer = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', answer)  # Remove LaTeX commands
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)  # Remove single LaTeX commands
    
    # Remove common math symbols that might cause issues
    answer = re.sub(r'[{}]', '', answer)  # Remove braces
    answer = re.sub(r'\\', '', answer)  # Remove backslashes
    
    # Normalize whitespace and convert to lowercase
    answer = re.sub(r'\s+', '', answer.lower())
    
    return answer

def is_answer_correct(predicted: str, ground_truth: str):
    """Check if predicted answer matches ground truth."""
    if not predicted or not ground_truth:
        return False
    
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Direct match
    if pred_norm == gt_norm:
        return True
    
    # Try to extract just the final number from complex expressions
    # This handles cases where the model gives the full expression instead of just the answer
    pred_numbers = re.findall(r'\d+', pred_norm)
    gt_numbers = re.findall(r'\d+', gt_norm)
    
    if pred_numbers and gt_numbers:
        # Check if any of the predicted numbers match the ground truth
        return any(pred_num == gt_num for pred_num in pred_numbers for gt_num in gt_numbers)
    
    return False

def evaluate_model(model, tokenizer, dataset, max_new_tokens=1000):
    """Evaluate the model on the AIME dataset."""
    results = {
        'total_problems': len(dataset),
        'correct_answers': 0,
        'incorrect_answers': 0,
        'no_boxed_found': 0,
        'token_usage': [],
        'response_times': [],
        'detailed_results': []
    }
    
    print(f"Evaluating {len(dataset)} problems...")
    
    for i, problem in enumerate(tqdm(dataset, desc="Evaluating problems")):
        try:
            # Prepare the problem
            problem_text = problem.get('Problem', '')
            ground_truth = problem.get('Answer', '')
            
            if not problem_text or not ground_truth:
                print(f"Warning: Problem {i} missing Problem or Answer field")
                continue
            
            # Create the prompt
            messages = [
                {"role": "user", "content": problem_text}
            ]
            
            # Tokenize input
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            
            input_tokens = inputs["input_ids"].shape[-1]
            
            # Generate response
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_time = time.time()
            
            # Decode response
            response = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
            
            # Calculate token usage
            total_tokens = outputs[0].shape[0]
            new_tokens = total_tokens - input_tokens
            
            # Extract answer from \boxed{}
            extracted_answer = extract_boxed_answer(response)
            
            # Check correctness
            is_correct = False
            if extracted_answer:
                is_correct = is_answer_correct(extracted_answer, ground_truth)
                if is_correct:
                    results['correct_answers'] += 1
                else:
                    results['incorrect_answers'] += 1
            else:
                results['no_boxed_found'] += 1
            
            # Store results
            results['token_usage'].append({
                'input_tokens': input_tokens,
                'output_tokens': new_tokens,
                'total_tokens': total_tokens
            })
            
            results['response_times'].append(end_time - start_time)
            
            results['detailed_results'].append({
                'problem_id': i,
                'problem': problem_text[:100] + "..." if len(problem_text) > 100 else problem_text,
                'ground_truth': ground_truth,
                'extracted_answer': extracted_answer,
                'full_response': response[:200] + "..." if len(response) > 200 else response,
                'is_correct': is_correct,
                'input_tokens': input_tokens,
                'output_tokens': new_tokens,
                'response_time': end_time - start_time
            })
            
        except Exception as e:
            print(f"Error processing problem {i}: {e}")
            continue
    
    return results

def print_results(results):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("AIME 2024 DATASET EVALUATION RESULTS")
    print("="*60)
    
    total = results['total_problems']
    correct = results['correct_answers']
    incorrect = results['incorrect_answers']
    no_boxed = results['no_boxed_found']
    
    print(f"Total Problems: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Incorrect Answers: {incorrect}")
    print(f"No \\boxed{{}} Found: {no_boxed}")
    print(f"Accuracy: {correct/total*100:.2f}%")
    
    # Token usage statistics
    if results['token_usage']:
        input_tokens = [r['input_tokens'] for r in results['token_usage']]
        output_tokens = [r['output_tokens'] for r in results['token_usage']]
        total_tokens = [r['total_tokens'] for r in results['token_usage']]
        
        print(f"\nToken Usage Statistics:")
        print(f"Average Input Tokens: {np.mean(input_tokens):.1f} ± {np.std(input_tokens):.1f}")
        print(f"Average Output Tokens: {np.mean(output_tokens):.1f} ± {np.std(output_tokens):.1f}")
        print(f"Average Total Tokens: {np.mean(total_tokens):.1f} ± {np.std(total_tokens):.1f}")
        print(f"Total Tokens Used: {sum(total_tokens):,}")
    
    # Response time statistics
    if results['response_times']:
        response_times = results['response_times']
        print(f"\nResponse Time Statistics:")
        print(f"Average Response Time: {np.mean(response_times):.2f}s ± {np.std(response_times):.2f}s")
        print(f"Total Evaluation Time: {sum(response_times):.2f}s")
    
    print("="*60)

def save_detailed_results(results, output_file):
    """Save detailed results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {output_file}")

# Main evaluation code
if __name__ == "__main__":
    # Configuration
    model_path = "/projects/p32013/reasoning/AlphaOne/eval/GARPO/checkpoints/easy_r1/qwen3_1_7b_math_grpo_attention_attention/global_step_15/actor/huggingface"
    output_file = 'aime_evaluation_results.json'
    max_new_tokens = 1000
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Apply custom attention if Qwen3AttentionExtrea is available
    # try:
    #     for layer_idx in range(len(model.model.layers)):
    #         old_attn = model.model.layers[layer_idx].self_attn
    #         new_attn = Qwen3AttentionExtrea(
    #             config=model.config,
    #             layer_idx=layer_idx,
    #             softmax_fn='vanilla'
    #         )
    #         new_attn.load_state_dict(old_attn.state_dict(), strict=False)
    #         model.model.layers[layer_idx].self_attn = new_attn
    #     print("Applied custom attention mechanism")
    # except Exception as e:
    #     print(f"Warning: Could not apply custom attention: {e}")
    
    # Resize embeddings if necessary
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #     model.resize_token_embeddings(len(tokenizer))
    #     print(f"Resized embeddings to {len(tokenizer)}")
    
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = load_aime_dataset()
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, tokenizer, dataset, max_new_tokens)
    
    # Print results
    print_results(results)
    
    # Save detailed results
    save_detailed_results(results, output_file)
    
    print("Evaluation completed!")