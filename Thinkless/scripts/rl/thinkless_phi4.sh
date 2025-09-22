#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="microsoft/Phi-4-reasoning"
fi

# Train over a single node, 4 H100 GPUs
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.std_normalizer=False \
    data.train_files=data/deepscaler/train.parquet \
    data.val_files=data/deepscaler/aime24.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='hybrid-reasoning' \
    trainer.experiment_name='Thinkless-Phi4-Reasoning' \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    actor_rollout_ref.actor.thinkless_alpha=0.001 \
    thinkless_rewards.correct_short_reward=1.0 \
    thinkless_rewards.correct_think_reward=0.5 \
    thinkless_rewards.wrong_think_reward=-1.0 \
    thinkless_rewards.wrong_short_reward=-1.0 \
    trainer.total_epochs=2 "${@:1}"  \
    +actor_rollout_ref.model.use_shm=True \
    +actor_rollout_ref.model.lora_rank=1 \
    +actor_rollout_ref.model.lora_alpha=2 \
    +actor_rollout_ref.model.target_modules=qkv_proj \
    +actor_rollout_ref.model.mixed_precision=bfloat16 \
    +actor_rollout_ref.actor.activation_checkpointing=True \
    +actor_rollout_ref.model.offload_kv_cache=True \
    +actor_rollout_ref.model.override_config.rope_scaling=null
    
    # vLLM without XFORMERS will results in CUDA errors.
    