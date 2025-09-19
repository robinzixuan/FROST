import os
import argparse
from llm_datasets.math import MATH500
from utils import *
from llm_datasets import GSM8KZero, GSM8K, GPQA, MathBenchDataset, AIME2024, MATH500, MinervaMath
from llm_models import LLMModel
import time
import logging
from evaluator import AccEvaluator
import torch
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from collections import OrderedDict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data(args):
    """
    Prepare dataset based on command line arguments.
    
    Args:
        args: Command line arguments containing:
            data_name: Name of the dataset to load
            reasoning: Whether to include reasoning
            budget: Token budget for responses
            
    Returns:
        Dataset object of the specified type or None if data_name is not supported
    """
    if args.data_name == 'AIME':
        dataset = AIME2024(args, with_reasoning=args.reasoning, budget=args.budget,
                       name=args.data_name, cache=False, split='train')
    elif args.data_name == 'math':
        dataset = MATH500(args, with_reasoning=args.reasoning, budget=args.budget,
                       name=args.data_name, cache=False, split='test')
    elif args.data_name == 'minerva':
        dataset = MinervaMath(args, with_reasoning=args.reasoning, budget=args.budget,
                       name=args.data_name, cache=False, split='test')
    elif args.data_name == 'GSM8K-Zero':
        dataset = GSM8KZero(args, with_reasoning=args.reasoning, budget=args.budget,
                            name=args.data_name, cache=False)
    elif args.data_name == 'GSM8K-Train':
        dataset = GSM8K(args, with_reasoning=args.reasoning, budget=args.budget,
                        name=args.data_name, cache=False, split='train')
    elif args.data_name == 'GSM8K-Test':
        dataset = GSM8K(args, with_reasoning=args.reasoning, budget=args.budget,
                        name=args.data_name, cache=False, split='test')
    else:
        dataset = None
        ValueError(f"Not supported for {args.data_name}")
    return dataset


def data2list(dataset):
    """
    Convert dataset instances to lists of samples and ground truths.
    
    Args:
        dataset: The dataset to convert
        
    Returns:
        tuple: (sample_list, gt_list) where:
            sample_list: List of prompt strings
            gt_list: List of ground truth answers
    """
    sample_list = []
    gt_list = []
    for idx, instance in enumerate(dataset):
        cur_sample = instance['round']
        ground_truth = instance['gold']
        sample_list.append(cur_sample[0]['prompt'])
        gt_list.append(ground_truth)
    return sample_list, gt_list


def safe_softmax(x, dim=-1):
    x = x - torch.max(x, dim=dim, keepdim=True).values  # stabilize
    return torch.softmax(x, dim=dim)

def kurtosis(x, eps=1e-6):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    if len(x.shape) > 2:
        x = x.view(x.shape[0], -1)

    # Normalize to probabilities if logits
    if x.max() > 1 or x.min() < 0:  
        x = safe_softmax(x, dim=-1)

    mu = x.mean(dim=1, keepdims=True)
    s = x.std(dim=1)
    s = torch.where(s < eps, torch.tensor(eps, device=s.device), s)

    z = (x - mu) / s.unsqueeze(1)
    mu4 = (z ** 4.0).mean(dim=1)

    return torch.nan_to_num(mu4, nan=0.0, posinf=1e6, neginf=-1e6)


def calculate_kurtosis(activations):
    if isinstance(activations, torch.Tensor):
        activations = activations.detach()
    else:
        activations = torch.tensor(activations, dtype=torch.float32)

    kurtosis_values = kurtosis(activations)

    # Clean before averaging
    kurtosis_values = torch.nan_to_num(kurtosis_values, nan=0.0, posinf=1e6, neginf=-1e6)

    return float(kurtosis_values.mean().item())



def calculate_max_inf_norm(activations):
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    
    # Replace NaN/inf with finite numbers
    activations = np.nan_to_num(activations, nan=0.0, posinf=1e6, neginf=-1e6)

    # Calculate infinity norm for each sample
    inf_norms = np.max(np.abs(activations), axis=tuple(range(1, len(activations.shape))))
    
    # Return maximum infinity norm
    return float(np.max(inf_norms))


# def calculate_attention_metrics(attention_weights_list):
#     """
#     Calculate average kurtosis and max infinity norm from attention weights.
    
#     Args:
#         attention_weights_list: List of attention weight tensors
        
#     Returns:
#         tuple: (average_kurtosis, max_inf_norm)
#     """
#     if not attention_weights_list:
#         return 0.0, 0.0
    
#     # Normalize all attention weights to have consistent shapes
#     normalized_weights = []
#     for weights in attention_weights_list:
#         if len(weights.shape) == 2:
#             # Already 2D, keep as is
#             normalized_weights.append(weights)
#         elif len(weights.shape) == 3:
#             # 3D tensor: (batch, seq, features) -> flatten to 2D
#             normalized_weights.append(weights.view(weights.shape[0], -1))
#         elif len(weights.shape) == 4:
#             # 4D tensor: (batch, heads, seq, seq) -> flatten to 2D
#             normalized_weights.append(weights.view(weights.shape[0], -1))
#         else:
#             # For other shapes, flatten all dimensions except the first
#             normalized_weights.append(weights.view(weights.shape[0], -1))
    
#     # Concatenate all normalized attention weights
#     max_dim = max(w.shape[1] for w in normalized_weights)
#     padded = [torch.nn.functional.pad(w, (0, max_dim - w.shape[1])) for w in normalized_weights]
#     all_attention_weights = torch.cat(padded, dim=0)
    
#     # Calculate metrics using the new kurtosis function
#     avg_kurtosis = calculate_kurtosis(all_attention_weights)
#     max_inf_norm = calculate_max_inf_norm(all_attention_weights)
    
#     return avg_kurtosis, max_inf_norm

def calculate_attention_metrics(attention_weights_list):
    if not attention_weights_list:
        return 0.0, 0.0
    
    kurtosis_vals = []
    inf_norm_vals = []
    
    for weights in attention_weights_list:
        # Normalize to 2D
        if len(weights.shape) > 2:
            weights = weights.view(weights.shape[0], -1)
        
        kurtosis_vals.append(calculate_kurtosis(weights))
        inf_norm_vals.append(calculate_max_inf_norm(weights))
    
    avg_kurtosis = float(np.mean(kurtosis_vals))
    max_inf_norm = float(np.max(inf_norm_vals))
    
    return avg_kurtosis, max_inf_norm


def attach_act_hooks_for_eval(model):
    """
    Attach hooks to capture activations for evaluation.
    
    Args:
        model: The model to attach hooks to
        
    Returns:
        OrderedDict: Dictionary to store captured activations
    """
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]
            act_dict[name] = (inp, out)

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


def attach_tb_act_hooks(model):
    """
    Attach hooks to capture transformer block activations.
    
    Args:
        model: The model to attach hooks to
        
    Returns:
        OrderedDict: Dictionary to store captured activations
    """
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            act_dict[name] = out[0]

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


def extract_attention_weights_from_hooks(act_dict):
    """
    Extract attention weights from hook-captured activations.
    
    Args:
        act_dict: Dictionary containing captured activations
        
    Returns:
        list: List of attention weight tensors
    """
    attention_weights = []
    
    # Look for attention modules in the captured activations
    for name, (inp, out) in act_dict.items():
        if 'attention' in name.lower() or 'attn' in name.lower():
            if isinstance(out, torch.Tensor):
                # Check if this looks like attention weights (typically 4D: batch, heads, seq, seq)
                if len(out.shape) >= 3:
                    # Flatten to 2D for easier processing
                    if len(out.shape) == 4:  # batch, heads, seq, seq
                        # Reshape to (batch*heads, seq*seq)
                        reshaped = out.view(out.shape[0] * out.shape[1], -1)
                        attention_weights.append(reshaped.detach())
                    elif len(out.shape) == 3:  # batch, seq, features
                        attention_weights.append(out.detach())
        elif 'self_attn' in name.lower():
            if isinstance(out, torch.Tensor) and len(out.shape) >= 3:
                if len(out.shape) == 4:  # batch, heads, seq, seq
                    reshaped = out.view(out.shape[0] * out.shape[1], -1)
                    attention_weights.append(reshaped.detach())
                elif len(out.shape) == 3:  # batch, seq, features
                    attention_weights.append(out.detach())
    
    return attention_weights


def inference_local(args, dataset, model, evaluator):
    """
    Run inference using a local model (e.g. Hugging Face models).
    
    Args:
        args: Command line arguments
        dataset: Dataset to run inference on
        model: The local LLM model instance
        evaluator: AccEvaluator instance for accuracy calculation
        
    Results include accuracy percentage, average token cost per sample, and attention metrics.
    """
    print(model)
    acc_num = 0
    token_num = 0
    results = []
    start_time = time.time()
    logger.info("=" * 30 + 'Requesting' + "=" * 30 + '\n')
    # process data in list
    logger.info(f"data size: {len(dataset)}")
    sample_list, gt_list = data2list(dataset)
    sample_list, gt_list = sample_list[args.start_index:args.end_index], gt_list[args.start_index:args.end_index]
    
    # Clear any previous attention weights
    model.clear_attention_weights()
    
    pred_list = model.query_batch(sample_list)
    
    # Calculate attention metrics
    attention_weights = model.get_attention_weights()
    if attention_weights:
        avg_kurtosis, max_inf_norm = calculate_attention_metrics(attention_weights)
        logger.info(f'Average Kurtosis: {avg_kurtosis:.4f}')
        logger.info(f'Max Infinity Norm: {max_inf_norm:.4f}')
    else:
        logger.info('No attention weights available for analysis')
        avg_kurtosis, max_inf_norm = 0.0, 0.0
    
    # dump to results
    assert len(sample_list) == len(gt_list) == len(pred_list)
    for i in range(len(pred_list)):
        results.append({
            "ground truth": gt_list[i],
            "question": sample_list[i],
            "prediction": pred_list[i],
        })
        acc_num += evaluator.evaluate_sample(results[-1],
                                             cloze=('cloze' in args.data_name) or (
                                                     args.data_name in ['GSM8K', 'GSM8K-Zero', 'AIME']))
        token_num += token_measure(pred_list[i])
    
    # Add metrics to results
    results.append({
        "metrics": {
            "average_kurtosis": avg_kurtosis,
            "max_inf_norm": max_inf_norm,
            "accuracy": 100 * acc_num / len(results),
            "avg_token_cost": token_num / len(results)
        }
    })
    
    logger.info(f'Accuracy: {100 * acc_num / len(results):.2f}%')
    logger.info(f'Token costs: {token_num / len(results):.2f}')
    save_to_jsonl(results, args.output_path)
    logger.info(f'Time cost: {time.time() - start_time}')


def inference_api(args, dataset, model, evaluator, key):
    """
    Run inference using an API-based model (e.g. GPT-4, Claude).
    
    Args:
        args: Command line arguments
        dataset: Dataset to run inference on
        model: The API-based LLM model instance
        evaluator: AccEvaluator instance for accuracy calculation
        key: API key for model access
    """
    acc_num = 0
    results = []
    start_time = time.time()
    logger.info("=" * 30 + 'Requesting' + "=" * 30 + '\n')
    if args.end_index is None:
        args.end_index = len(dataset)
    for idx, instance in enumerate(dataset):
        # logger.info(idx)
        if args.start_index <= idx < args.end_index:
            cur_sample = instance['round']
            ground_truth = instance['gold']
            logger.info('=' * 30 + f"Step: {idx + 1} / {args.end_index}" + '=' * 30)
            # pred = model.query(cur_sample, key=keys[args.key_index])
            logger.info(f"Question: {cur_sample[0]['prompt']}")
            pred = model.query(cur_sample, key=key)
            results.append({
                "ground truth": ground_truth,
                "question": cur_sample[0]['prompt'],
                "prediction": pred[0][0],
            })
            acc_num += evaluator.evaluate_sample(results[-1],
                                                 cloze=('cloze' in args.data_name) or (
                                                         args.data_name in ['GSM8K', 'GSM8K-Zero']))
            logger.info(f'Accuracy: {acc_num / len(results)}')
            # logger.info(f"Ground truth: {ground_truth}")
            # logger.info(f"Prediction: {pred[0][0]}")
            save_to_jsonl(results, args.output_path)
            logger.info(f'Time cost: {time.time() - start_time}')
    
    # Add metrics to results (API models don't have attention weights)
    results.append({
        "metrics": {
            "average_kurtosis": 0.0,  # Not available for API models
            "max_inf_norm": 0.0,      # Not available for API models
            "accuracy": 100 * acc_num / len(results),
            "avg_token_cost": 0.0     # Not calculated for API models
        }
    })
    logger.info('Note: Attention metrics not available for API-based models')


def parse_args():
    """
    Parse command line arguments for the inference script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="=The budget token for our tech.")
    parser.add_argument("--budget", default=None, help="=The budget token for our tech.")
    parser.add_argument("--reasoning", action='store_true', help="If we use LLM reasoning.")
    parser.add_argument("--model", default='DeepSeek-R1-Distill-Qwen-1.5B', help="The model name on huggingface.")
    parser.add_argument("--output_path", default='./tmp',
                        help="The output path to save the model output.")
    parser.add_argument("--n", default=1, type=int, help="Number of samples from LLM.")
    parser.add_argument("--start_index", default=0, type=int, help="The start index for the dataset.")
    parser.add_argument("--end_index", default=None, type=int, help="The end index for the dataset.")
    parser.add_argument("--key_index", default=1, type=int, help="The key index for the dataset.")
    parser.add_argument("--data_name", default='GSM8K-Zero',
                        type=str, help="The dataset name used during our evaluation.")
    return parser.parse_args()


def main():
    # prepare keys and arguments
    args = parse_args()
    args.output_path = os.path.join(args.output_path, 'Phi4', args.data_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    args.output_path = os.path.join(args.output_path,
                                    'output_with_reasoning.jsonl'
                                    if args.reasoning else 'output_without_reasoning_new_prompt.jsonl')
    logger.info(f'Saving to {args.output_path}')
    args.local = (args.model in ['Llama-3.1-8B-Instruct']) or 'Qwen' in args.model  or 'Phi' in args.model or 'oss' in args.model or 'phi' in args.model 
    keys = {
        'yi-lightning': ['your_api_key', 'your_api_key'],
        'gpt-4o-mini': ['your_api_key', 'your_api_key'],
        'gpt-4o-2024-05-13': ['your_api_key', 'your_api_key'],
    }
    key = keys[args.model][args.key_index] if not args.local else None

    # Prepare dataset
    dataset = prepare_data(args)
    # dataset = Subset(dataset, list(range(1500, 2000)))

    # Prepare evaluator
    evaluator = AccEvaluator(dataset)

    # Prepare llm model
    model = LLMModel(args)
    if args.end_index is None:
        args.end_index = len(dataset)
    args.end_index = min(args.end_index, len(dataset))

    # inference
    if args.local:
        inference_local(args, dataset, model, evaluator)
    else:
        inference_api(args, dataset, model, evaluator, key)


if __name__ == "__main__":
    main()
