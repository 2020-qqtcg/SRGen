#!/bin/bash

# 设置错误处理
set -e  # 遇到错误时立即退出
set -o pipefail  # 管道命令中任何一个失败都会导致整个管道失败

# export HF_ENDPOINT=https://hf-mirror.com ## if you have no vpn
export HF_HOME=~/.cache/huggingface  ## default huggingface cache directory

export model_path=/root/autodl-tmp/TNOT/models/Qwen2.5-7B
# export model_path=Qwen/Qwen2.5-7B ## or your local path to Qwen2.5-7B

# 创建错误日志文件
ERROR_LOG="./error.log"
echo "=== TNOT Script Execution Started at $(date) ===" >> "$ERROR_LOG"

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

python SLOT/gpqa_evaluator.py \
    --model_path $model_path \
    --times 1 \
    --lr 0.1 \
    --entropy_output_file "my_analysis.jsonl" \
    --entropy_threshold 2.0 \
    --entropy_weight 0.05 \
    --use_entropy_control \
    --max_retries 10 \
    --eval_samples 40 \
    2>&1 | tee -a "$ERROR_LOG"

echo "=== TNOT Script Execution Completed Successfully at $(date) ===" >> "$ERROR_LOG"
echo "All tasks completed successfully!"
