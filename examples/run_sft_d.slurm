#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=logs/%x/%j.out # output file (%j = job ID)
#SBATCH --error=logs/%x/%j.err # error file (%j = job ID)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --gres=gpu:8                 # reserve 8 GPUs per node
#SBATCH --time 2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev             # QoS
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --constraint=a100
#SBATCH --account=cjc@a100           # A100 accounting

set -x -e

echo "START TIME: $(date)"

# set up environment
module purge
module load git-lfs
module load unrar

module load anaconda-py3/2023.03
module load cuda/12.1.0
conda activate llm-train

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# hf env var
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# download
# export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_HUB_OFFLINE="1"
# cache
# export HF_HOME="/projects/bhuang/.cache/huggingface"
# logging
export ACCELERATE_LOG_LEVEL="info"
export TRANSFORMERS_VERBOSITY="info"

# wandb related
# wandb offline
# export WANDB_MODE=offline
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
# export WANDB_PROJECT=hf-whisper-v3

# hf transformers save model in WandbCallback.setup()
# by default in /tmp dir (by tempfile.mkstemp)
# which we don't have access in some clusters
export TMPDIR="/gpfswork/rech/cjc/commun/outputs/tmp"

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# Set your number of GPUs here
num_gpus=8

# LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
#     --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
#     --gradient_accumulation_steps $GRAD_ACC_STEPS \
#     --num_machines $NUM_NODES \
#     --num_processes $WORLD_SIZE \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --machine_rank \$SLURM_PROCID \
#     --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
#     --max_restarts 1 \
#     --role \$(hostname -s): \
#     --tee 3 \
#     "

# train config
# train_config=devtools/dev_sft.yml
# train_config=examples/llama3_70b_sft_fft.yml
train_config=examples/llama3_70b_sft_qlora.yml

# CMD="""
# CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess \
#     $train_config
# """

# python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="bofenghuang/vigogne-3.2", repo_type="dataset", local_dir="vigogne-3.2", local_dir_use_symlinks=False)'

# training params

# deepspeed config
# ds_config=deepspeed_configs/ds_config_zero2_no_offload.json
# ds_config=deepspeed_configs/zero3.json
# ds_config=deepspeed_configs/zero3_bf16_cpuoffload_params.json
#     --deepspeed $ds_config

CMD="""
accelerate launch \
    --num_processes $num_gpus \
    -m axolotl.cli.train \
    $train_config
"""

# run cmd
echo "Starting program..."

{ # try
    echo $CMD
    eval "$CMD"
} || { # catch
    # save log for exception 
    echo "Operation Failed!"
    exit 1
}
exit 0

echo "END TIME: $(date)"
