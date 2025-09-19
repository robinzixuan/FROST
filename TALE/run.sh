set -x

module load cuda/12.4.0-gcc-12.4.0
export CUDA_HOME=/software/cuda/cuda-12.1.0 
module load glibc/2.28-gcc-12.4.0
module load libxmu/1.1.4-gcc-12.3.0
module load vllm/0.10.1-gpt-oss

export HF_HOME="/projects/p32013/.cache/"
export CUDA_VISIBLE_DEVICES="0"
export PYTHONWARNINGS="ignore"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

conda activate vllm

/projects/p32013/conda_envs/vllm/bin/python3.10 -u inference.py --model /projects/p32013/reasoning/AlphaOne/eval/GARPO1/checkpoints/gptoss_softmax1/final --data_name minerva --output_path results/oss-softmax1 --batch_size 4 --reasoning 
# /projects/p32013/conda_envs/vllm/bin/python3.10 -u TALE-EP.py --data_name GSM8K-Test --model openai/gpt-oss-20b
# /projects/p32013/conda_envs/vllm/bin/python3.10 entropy.py