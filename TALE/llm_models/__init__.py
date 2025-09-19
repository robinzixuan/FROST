from openai import OpenAI
import time
import torch
import requests
from tqdm import tqdm
from utils import *
import random
from vllm import LLM, SamplingParams
import logging
import os


os.environ["XFORMERS_FORCE_DISABLE_SDPA"] = "1"
logger = logging.getLogger(__name__)

API_BASE = "The API BASE"
claude_api_key = ""
max_tokens = 1024


class HuggingFaceModel:
    def __init__(self, args):
        self.args = args
        self.model_path = self.args.model
        self.attention_weights = []  # Store attention weights for analysis
        self.act_dict = None  # Store hook-captured activations
        
        # Import necessary modules
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from phi3_attention import Phi3AttentionExtra
        import torch

        # Monkey patch inside the HF model class
        import transformers.models.phi.modeling_phi as phi_modeling

        class PatchedPhiDecoderLayer(phi_modeling.PhiDecoderLayer):
            def __init__(self, config, layer_idx):
                super().__init__(config, layer_idx)
                # overwrite attention module
                self.self_attn = Phi3AttentionExtra(
                    config=config,
                    layer_idx=layer_idx,
                    softmax_fn="softmax1"
                )

        phi_modeling.PhiDecoderLayer = PatchedPhiDecoderLayer

        # Load model and tokenizer directly (not using vLLM for hook access)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model loaded with hooks for attention weight capture!")

    def inference(self, cur_sample, n_reasoning_paths):
        """
        Run inference with hook-based activation capture.
        """
        import torch
        from inference2 import attach_act_hooks_for_eval, extract_attention_weights_from_hooks
        
        logger.info("Inference processing...")
        
        # Prepare input
        prompt = f"You are a helpful assistant.\nUser: {cur_sample[0]['prompt']}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Attach hooks
        self.act_dict = attach_act_hooks_for_eval(self.model)
        
        # Generate with hooks
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract attention weights from hooks
        attention_weights = extract_attention_weights_from_hooks(self.act_dict)
        self.attention_weights.extend(attention_weights)
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        n_input, n_output = [], []
        return [generated_text.strip()], n_input, n_output

    def inference_batch(self, sample_list, n_reasoning_paths):
        """
        Run batch inference with hook-based activation capture.
        """
        import torch
        from inference2 import attach_act_hooks_for_eval, extract_attention_weights_from_hooks
        
        logger.info("Inference processing...")
        results = []
        logger.info(f"batch size: {self.args.batch_size}")
        
        # Attach hooks once for the entire batch
        self.act_dict = attach_act_hooks_for_eval(self.model)
        
        for i in tqdm(range(0, len(sample_list), self.args.batch_size), desc="Processing", unit="batch"):
            batch_samples = sample_list[i:i + self.args.batch_size]
            
            # Prepare batch inputs
            batch_prompts = [
                f"You are a helpful assistant.\nUser: {sample}\nAssistant:" 
                for sample in batch_samples
            ]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate with hooks
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract attention weights from hooks for this batch
            attention_weights = extract_attention_weights_from_hooks(self.act_dict)
            self.attention_weights.extend(attention_weights)
            
            # Decode outputs
            batch_results = []
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                generated_text = self.tokenizer.decode(
                    output[input_length:], 
                    skip_special_tokens=True
                )
                batch_results.append(generated_text.strip())
            
            results.extend(batch_results)
            
            # Clear cache
            torch.cuda.empty_cache()
        
        assert len(results) == len(sample_list)
        return results

    def get_attention_weights(self):
        """
        Get collected attention weights for analysis.
        
        Returns:
            list: List of attention weight tensors
        """
        return self.attention_weights

    def clear_attention_weights(self):
        """
        Clear stored attention weights.
        """
        self.attention_weights = []

    @staticmethod
    def format_message(message):
        formatted_message = ""
        for msg in message:
            role = msg["role"]
            content = msg["content"]
            formatted_message += f"{role}: {content}\n"
        return formatted_message.strip()


def call_gpt(cur_sample, n_reasoning_paths,
             api_key="your_api_key",
             model='gpt-4o-mini'):
    while True:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant."
                 },
                {"role": "user",
                 "content": cur_sample[0]['prompt']
                 }
            ],
            "max_tokens": max_tokens,
            "seed": 1024,
            "n": n_reasoning_paths
        }

        response = requests.post("https://aigptx.top/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            responses = response.json()
            answers = []
            for i in range(n_reasoning_paths):
                answers.append(responses['choices'][0]['message']['content'])
            n_input = responses['usage']['prompt_tokens']
            n_output = responses['usage']['completion_tokens']
            return answers, n_input, n_output
        else:
            logger.info(f"Error: {response.status_code} - {response.text}")
            time.sleep(random.uniform(1, 3))


def call_Yi_Lightning(cur_sample, n_reasoning_paths,
                      api_key='your_api_key',
                      model='yi-lightning'):
    client = OpenAI(
        api_key=api_key,
        base_url=API_BASE
    )
    while True:
        try:
            responses = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {'role': "system", "content": "You are a helpful assistant."},
                    {'role': "user", "content": cur_sample[0]['prompt']},
                ],
                # stop = stop,
                temperature=0,
                n=n_reasoning_paths,
            )
            break
        except Exception as e:
            logger.info('Error!', e)
            time.sleep(random.uniform(1, 3))
    answers = []
    for i in range(n_reasoning_paths):
        answers.append(responses.choices[i].message.content)
    n_input = responses.usage.prompt_tokens
    n_output = responses.usage.completion_tokens
    return answers, n_input, n_output


class LLMModel:
    def __init__(self, args):
        self.local = False
        self.args = args
        if 'oss' in args.model or 'phi' in args.model:
            self.model = HuggingFaceModel(args=args)
            self.local = True
        elif 'gpt' in args.model or 'o3' in args.model or 'o1' in args.model:  # OpenAI models
            self.model = call_gpt
        elif 'yi' in args.model:
            self.model = call_Yi_Lightning
        elif 'Qwen' in args.model or 'DeepSeek' in args.model or 'Llama' in args.model or 'Phi' in args.model:
            self.model = HuggingFaceModel(args=args)
            self.local = True
        else:
            raise NotImplementedError

    def query_batch(self, sample_list):
        # return self.model.inference_batch_new(sample_list, self.args.n)
        return self.model.inference_batch(sample_list, self.args.n)

    def query(self, sample, key='your_api_key'):
        if self.local:
            return self.model.inference(sample, n_reasoning_paths=self.args.n)
        else:
            return self.model(sample, n_reasoning_paths=self.args.n,
                              api_key=key, model=self.args.model)

    def get_attention_weights(self):
        """
        Get attention weights if using local model.
        
        Returns:
            list: List of attention weight tensors or empty list for API models
        """
        if self.local and hasattr(self.model, 'get_attention_weights'):
            return self.model.get_attention_weights()
        return []

    def clear_attention_weights(self):
        """
        Clear attention weights if using local model.
        """
        if self.local and hasattr(self.model, 'clear_attention_weights'):
            self.model.clear_attention_weights()
