#!/usr/bin/env python3
"""
Extract sample numbers (minus 1) for entries where Answer Correct: False
"""
import re

def extract_wrong_samples(log_file_path):
    """
    读取log文件，找出Answer Correct为False的sample，返回编号-1的列表
    """
    wrong_samples = []
    current_sample = None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 查找Sample编号行
            sample_match = re.match(r'Sample (\d+):', line)
            if sample_match:
                current_sample = int(sample_match.group(1))
                continue
            
            # 查找Answer Correct行
            if 'Answer Correct:' in line and current_sample is not None:
                if 'Answer Correct: False' in line:
                    # 编号减1并添加到列表
                    wrong_samples.append(current_sample - 1)
                
                # 重置current_sample，为下一个sample做准备
                current_sample = None
    
    return wrong_samples

def main():
    log_file = 'log.log'
    
    print(f"正在读取 {log_file}...")
    wrong_samples = extract_wrong_samples(log_file)
    
    print(f"\n找到 {len(wrong_samples)} 个答案错误的sample")
    print(f"编号-1的列表（共{len(wrong_samples)}个）:")
    print(wrong_samples)
    
    # 也可以打印成更整齐的格式
    print(f"\n整齐格式的列表:")
    print("[", end="")
    for i, sample_id in enumerate(wrong_samples):
        if i > 0:
            print(", ", end="")
        if i % 10 == 0 and i > 0:  # 每10个换行
            print("\n ", end="")
        print(sample_id, end="")
    print("]")
    
    # 保存到文件
    output_file = 'wrong_samples.txt'
    with open(output_file, 'w') as f:
        f.write(f"答案错误的sample编号-1列表（共{len(wrong_samples)}个）:\n")
        f.write(str(wrong_samples))
    
    print(f"\n结果已保存到 {output_file}")

if __name__ == "__main__":
    main()