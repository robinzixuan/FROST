#!/bin/bash

set -x

module load cuda/12.4.0-gcc-12.4.0
export CUDA_HOME=/software/cuda/cuda-12.1.0 
module load glibc/2.28-gcc-12.4.0
export HF_HOME="/projects/p32013/.cache/"
# Set up environment variables for wandb
export WANDB_PROJECT="grpo-training"
export WANDB_ENABLED="true"



export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-1.7B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config2.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
