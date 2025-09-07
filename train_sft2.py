from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import json
from Qwen_attention import Qwen3AttentionExtrea
from Qwen2_5_attention import Qwen2AttentionExtra
from gpt_attention import GptOssAttentionExtra
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Mxfp4Config
import torch
from peft import prepare_model_for_kbit_training
import torch.utils.checkpoint
torch.utils.checkpoint.use_reentrant = False

config = AutoConfig.from_pretrained("openai/gpt-oss-20b")

quantization_config = Mxfp4Config(dequantize=True)

model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b",
                                            config=config,
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation="eager",
                                            device_map="cpu",
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True,
                                            quantization_config=quantization_config,
                                            )
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")


# for layer_idx in range(len(model.model.layers)):
#     old_attn = model.model.layers[layer_idx].self_attn
#     new_attn = Qwen3AttentionExtrea(
#         config=model.config,
#         layer_idx=layer_idx,
#         softmax_fn='entmax15'
#     )
#     new_attn.load_state_dict(old_attn.state_dict(), strict=False)
#     model.model.layers[layer_idx].self_attn = new_attn


# for layer_idx in range(len(model.model.layers)):
#     old_attn = model.model.layers[layer_idx].self_attn
#     new_attn = Qwen2AttentionExtra(
#         config=model.config,
#         layer_idx=layer_idx,
#         softmax_fn='softmax1'
#     )
#     new_attn.load_state_dict(old_attn.state_dict(), strict=False)
#     model.model.layers[layer_idx].self_attn = new_attn


for layer_idx in range(len(model.model.layers)):
    old_attn = model.model.layers[layer_idx].self_attn
    new_attn = GptOssAttentionExtra(
        config=model.config,
        layer_idx=layer_idx,
        softmax_fn='softmax1'
    )
    new_attn.load_state_dict(old_attn.state_dict(), strict=False)
    model.model.layers[layer_idx].self_attn = new_attn

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

# Convert back to Hugging Face Dataset objects for SFTTrainer
train_dataset_formatted = Dataset.from_list(formatted_train_data)
test_dataset_formatted = Dataset.from_list(formatted_test_data)

model.gradient_checkpointing_enable()  # Disabled to avoid LoRA + gradient checkpointing conflicts
# Note: Gradient checkpointing can cause issues with LoRA training due to parameter sharing conflicts.
# If you need memory optimization, consider reducing batch size or using gradient accumulation instead.
if hasattr(model, '_set_static_graph'):
    model._set_static_graph(True)

# Use formatted data for training
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset_formatted,
    eval_dataset=test_dataset_formatted,
    args=SFTConfig(
        output_dir="checkpoints/gptoss_softmax1",
        do_train=True,
        do_eval=True,
        max_steps=5000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        bf16=True,
    ),
    peft_config=LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj']
    )
)

trainer.train()