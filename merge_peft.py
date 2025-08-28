#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_with_model(model_name, lora_name, output_dir, device="auto"):
    """
    Merge a base model checkpoint with a LoRA adapter and save as a new Hugging Face checkpoint.
    """

    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter: {lora_name}")
    model = PeftModel.from_pretrained(base_model, lora_name)

    print("Merging LoRA weights with base model...")
    # merge_and_unload removes the adapter wrapper and integrates weights
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save merged model
    merged_model.save_pretrained(output_dir)

    # Save tokenizer (always from base model path to get vocab/config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print("✅ Model merging completed successfully!")
    return merged_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge a base model checkpoint with a LoRA adapter")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen3-1.7B",
                        help="Path or name of the base model checkpoint")
    parser.add_argument("--lora_name", type=str,
                        default="/projects/p32013/reasoning/AlphaOne/eval/GARPO1/checkpoints/attention/checkpoint-5000",
                        help="Path or name of the LoRA adapter")
    parser.add_argument("--output_dir", type=str,
                        default="/projects/p32013/reasoning/AlphaOne/eval/GARPO1/checkpoints/attention/final",
                        help="Directory to save the merged model")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to load the model on (auto, cpu, cuda, etc.)")

    args = parser.parse_args()

    try:
        merge_lora_with_model(
            model_name=args.model_name,
            lora_name=args.lora_name,
            output_dir=args.output_dir,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Error during model merging: {e}")
        raise


if __name__ == "__main__":
    main()
