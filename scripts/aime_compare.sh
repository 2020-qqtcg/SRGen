#!/bin/bash

# 设置错误处理
set -e  # 遇到错误时立即退出
set -o pipefail  # 管道命令中任何一个失败都会导致整个管道失败

# export HF_ENDPOINT=https://hf-mirror.com ## if you have no vpn
export HF_HOME=~/.cache/huggingface  ## default huggingface cache directory
export model_path=/hpc2hdd/home/hlin199/mujian/models/Qwen2.5-7B-Instruct
# export model_path=Qwen/Qwen2.5-7B ## or your local path to Qwen2.5-7B

# 创建错误日志文件
# ERROR_LOG="./error2.log"
# echo "=== TNOT Script Execution Started at $(date) ===" >> "$ERROR_LOG"

# 错误处理函数
handle_error() {    
    local exit_code=$?  
    local line_number=$1
    echo "=== Error occurred at line $line_number with exit code $exit_code at $(date) ===" >> "$ERROR_LOG"
    echo "Error: Script failed at line $line_number with exit code $exit_code"
    exit $exit_code
}

# 设置错误陷阱
trap 'handle_error $LINENO' ERR

echo "Starting TNOT evaluation with error logging..."

python -m TNOT.gpqa_evaluator \
    --model_path $model_path \
    --device cuda:2 \
    --split train \
    --times 3 \
    --lr 0.01 \
    --entropy_threshold 3.0 \
    --entropy_weight 0.25 \
    --use_entropy_control \
    --max_retries 10 \
    --adaptive_entropy \
    --adaptive_entropy_N 25 \
    --adaptive_entropy_K 2.5 \
    --mask_special_tokens \
    --do_sample \
    --temperature 0.6 

python -m TNOT.gpqa_evaluator \
    --model_path $model_path \
    --device cuda:2 \
    --split train \
    --times 3 \
    --lr 0.1 \
    --entropy_threshold 3.0 \
    --entropy_weight 0.25 \
    --use_entropy_control \
    --max_retries 10 \
    --adaptive_entropy \
    --adaptive_entropy_N 25 \
    --adaptive_entropy_K 2.5 \
    --mask_special_tokens \
    --do_sample \
    --temperature 0.6 

python -m TNOT.gpqa_evaluator \
    --model_path $model_path \
    --device cuda:2 \
    --split train \
    --times 3 \
    --lr 0.01 \
    --entropy_threshold 3.0 \
    --entropy_weight 0.75 \
    --use_entropy_control \
    --max_retries 10 \
    --adaptive_entropy \
    --adaptive_entropy_N 25 \
    --adaptive_entropy_K 2.5 \
    --mask_special_tokens \
    --do_sample \
    --temperature 0.6

echo "=== TNOT Script Execution Completed Successfully at $(date) ===" >> "$ERROR_LOG"
echo "All tasks completed successfully!"
