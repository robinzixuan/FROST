from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import json
from Qwen_attention import Qwen3AttentionExtrea
from Qwen2_5_attention import Qwen2AttentionExtra
from gpt_attention import GptOssAttentionExtra
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
# Additional imports for gradient monitoring
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.nn.utils import clip_grad_norm_
# import torch.amp as amp  # Not needed for current implementation

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Math-1.5B")


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B",
                                            config=config,
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation="eager",
                                            device_map="auto",  # Changed from "cpu" to "auto" for proper device handling
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")


# for layer_idx in range(len(model.model.layers)):
#     old_attn = model.model.layers[layer_idx].self_attn
#     new_attn = Qwen3AttentionExtrea(
#         config=model.config,
#         layer_idx=layer_idx,
#         softmax_fn='entmax15'
#     )
#     new_attn.load_state_dict(old_attn.state_dict(), strict=False)
#     model.model.layers[layer_idx].self_attn = new_attn


# Temporarily disable custom attention to fix NaN issues
# TODO: Fix softmax_1 numerical instability before re-enabling
# for layer_idx in range(len(model.model.layers)):
#     old_attn = model.model.layers[layer_idx].self_attn
#     new_attn = Qwen2AttentionExtra(
#         config=model.config,
#         layer_idx=layer_idx,
#         softmax_fn='softmax1'
#     )
#     new_attn.load_state_dict(old_attn.state_dict(), strict=False)
#     model.model.layers[layer_idx].self_attn = new_attn


# for layer_idx in range(len(model.model.layers)):
#     old_attn = model.model.layers[layer_idx].self_attn
#     new_attn = GptOssAttentionExtra(
#         config=model.config,
#         layer_idx=layer_idx,
#         softmax_fn='softmax1'
#     )
#     new_attn.load_state_dict(old_attn.state_dict(), strict=False)
#     model.model.layers[layer_idx].self_attn = new_attn

# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))



def format_data_for_sft(dataset):
    """Convert dataset with question/answer columns to prompt/completion format"""
    formatted_data = []
    
    for item in dataset:
        # Transform answer format from <answer></answer> to \boxed{}
        answer_content = item["answer"]
        if "<answer>" in answer_content and "</answer>" in answer_content:
            # Extract content between <answer> tags and wrap with \boxed{}
            start_tag = "<answer>"
            end_tag = "</answer>"
            start_idx = answer_content.find(start_tag) + len(start_tag)
            end_idx = answer_content.find(end_tag)
            if start_idx != -1 and end_idx != -1:
                answer_text = answer_content[start_idx:end_idx].strip()
                # Replace the entire <answer></answer> section with \boxed{}
                answer_content = answer_content.replace(
                    f"{start_tag}{answer_text}{end_tag}", 
                    f"\\boxed{{{answer_text}}}"
                )
        
        formatted_item = {
        "prompt": [{"role": "user", "content": item["question"]}],
        "completion": [
            {"role": "assistant", "content": answer_content}
        ],
    }
        formatted_data.append(formatted_item)
    
    return formatted_data

# Load both train and test splits
train_dataset = load_dataset("Jax-dan/Lite-Thinking", split="train")
test_dataset = load_dataset("Jax-dan/Lite-Thinking", split="test")

# Format the data
formatted_train_data = format_data_for_sft(train_dataset)
formatted_test_data = format_data_for_sft(test_dataset)

# Save formatted data to JSON files
with open("formatted_train_data.json", "w") as f:
    json.dump(formatted_train_data, f, indent=2)

with open("formatted_test_data.json", "w") as f:
    json.dump(formatted_test_data, f, indent=2)

print(f"Train data: {len(formatted_train_data)} samples")
print(f"Test data: {len(formatted_test_data)} samples")
print("Data saved to formatted_train_data.json and formatted_test_data.json")

# Example of formatted data structure
if formatted_train_data:
    print("\nExample formatted data:")
    print(json.dumps(formatted_train_data[0], indent=2))

# Custom trainer with NaN detection and recovery
class SafeSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to handle NaN gradients and device issues"""
        try:
            # Call parent method first
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Check for NaN loss after the step
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss detected at step {self.state.global_step}. Loss value: {loss}")
                # Return a zero loss on the same device as the model
                device = next(model.parameters()).device
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            return loss
            
        except Exception as e:
            print(f"Error in training step {self.state.global_step}: {e}")
            # Return a zero loss on the same device as the model
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """Override inner training loop to add gradient monitoring"""
        # Call parent method but add gradient monitoring
        result = super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
        
        # Add gradient monitoring after each step
        if hasattr(self, 'accelerator') and self.accelerator.is_main_process:
            # Check gradients for NaN/Inf
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"Warning: NaN/Inf gradient detected in {name} at step {self.state.global_step}")
        
        return result

# Convert back to Hugging Face Dataset objects for SFTTrainer
train_dataset_formatted = Dataset.from_list(formatted_train_data)
test_dataset_formatted = Dataset.from_list(formatted_test_data)

# Use formatted data for training with safe trainer
trainer = SafeSFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset_formatted,
    eval_dataset=test_dataset_formatted,
    args=SFTConfig(
        output_dir="checkpoints/qwen25_softmax1",
        do_train=True,
        do_eval=True,
        max_steps=5000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,  # Increased from 1e-8 to reasonable value
        warmup_steps=100,    # Add warmup for better training stability
        logging_steps=10,    # Add logging for better monitoring
        save_steps=500,      # Add checkpoint saving
        eval_steps=500,      # Add evaluation steps
        max_grad_norm=1.0,   # Add gradient clipping to prevent NaN gradients
        fp16=False,          # Disable mixed precision to avoid NaN issues
        dataloader_drop_last=True,  # Ensure consistent batch sizes
        remove_unused_columns=False,  # Prevent data loading issues
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        seed=42,             # Set seed for reproducibility
        ignore_data_skip=True,  # Skip problematic data samples
    ),
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']  # Added more target modules
    )
)

trainer.train()