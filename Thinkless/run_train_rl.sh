export VLLM_ATTENTION_BACKEND=XFORMERS
set -x

module load cuda/12.4.0-gcc-12.4.0
export CUDA_HOME=/software/cuda/cuda-12.1.0 
module load glibc/2.28-gcc-12.4.0
export HF_HOME="/projects/p32013/.cache/"
# Set up environment variables for wandb
export WANDB_PROJECT="grpo-training"
export WANDB_ENABLED="true"

# Run RL training using the warmup model
export MODEL_PATH="microsoft/Phi-4-mini-reasoning"
./scripts/rl/thinkless_phi4.sh --model $MODEL_PATH