#!/bin/bash

# export HF_ENDPOINT=https://hf-mirror.com ## if you have no vpn
export HF_HOME=~/.cache/huggingface  ## default huggingface cache directory
export model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

export TEMP_PARALLEL_FILE="tp_aime_compare"

python -m TNOT.aime_evaluator \
    --model_path $model_path \
    --parallel \
    --max_parallel_gpus 4 \
    --average 5 \
    --split train \
    --version 2024 \
    --times 3 \
    --lr 0.01 \
    --entropy_threshold 3.0 \
    --entropy_weight 0.05 \
    --use_entropy_control \
    --max_retries 20 \
    --adaptive_entropy \
    --adaptive_entropy_N 25 \
    --adaptive_entropy_K 4 \
    --do_sample \
    --temperature 0.6 \
    --max_new_tokens 32768 \
    --minimal_std 0.5 \
    --minimal_threshold 1.8 \

