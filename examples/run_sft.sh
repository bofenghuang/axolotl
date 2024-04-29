#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# Train chat models with full-finetune

# export OMP_NUM_THREADS="1"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
export CUDA_VISIBLE_DEVICES="1,2"

export HF_HOME="/projects/bhuang/.cache/huggingface"

# Set your number of GPUs here
num_gpus=2

# train config
# train_config=devtools/dev_sft.yml
# train_config=examples/llama-3/b_sft_fft.yml
train_config=examples/llama-3/b_sft_qlora.yml
# train_config=examples/llama-3/b_sft_qlora_unpacking.yml

# preprocess datasets - optional but recommended
CMD="""
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess \
    $train_config
"""

# training
    # --deepspeed deepspeed_configs/zero3.json


CMD="""
accelerate launch \
    --main_process_port 29002 \
    --num_processes $num_gpus \
    -m axolotl.cli.train \
    $train_config \
    --deepspeed deepspeed_configs/ds_config_zero2_no_offload.json
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
