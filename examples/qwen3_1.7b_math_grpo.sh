#!/bin/bash

set -x

module load cuda/12.4.0-gcc-12.4.0
export CUDA_HOME=/software/cuda/cuda-12.1.0 
module load glibc/2.28-gcc-12.4.0
module load gcc/12.4.0-gcc-8.5.0
module load mpi/openmpi-5.0.5-gcc-12.4.0
export HF_HOME="/projects/p32013/.cache/"
# Set up environment variables for wandb
export WANDB_PROJECT="grpo-training"
export WANDB_ENABLED="true"

export TORCHINDUCTOR_DISABLE=1


export PYTHONUNBUFFERED=1

export CC=$(which gcc)
export CXX=$(which g++)

MODEL_PATH=/projects/p32013/reasoning/AlphaOne/eval/GARPO1/checkpoints/gptoss_softmax1/final  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH}
