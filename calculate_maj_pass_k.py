#!/usr/bin/env python3
"""
统计模型运行k次结果的maj@k和pass@k准确率脚本

maj@k: 多数投票准确率，对于每个问题，如果k次运行中超过一半的结果正确，则该问题算作正确
pass@k: 通过率，对于每个问题，如果k次运行中至少有一次结果正确，则该问题算作正确

支持任意数量的测试案例（不再限制为30个），可以处理大规模数据集。
支持自定义显示的样本数量和k值范围。
"""

import re
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


def parse_log_file(log_file_path: str) -> Tuple[Dict[int, List[bool]], int, int]:
    """
    解析log文件，提取每个样本在每次运行中的结果
    
    Returns:
        results: {sample_id: [result1, result2, ...]} 每个样本的所有运行结果
        num_samples: 每轮的样本数量
        num_runs: 运行轮数
    """
    results = defaultdict(list)
    current_sample = None
    num_samples = 0
    num_runs = 0
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取评估样本数量
    sample_count_matches = re.findall(r'Number of evaluation samples: (\d+)', content)
    if sample_count_matches:
        num_samples = int(sample_count_matches[0])
        num_runs = len(sample_count_matches)
    
    # 使用正则表达式匹配样本和结果
    sample_pattern = r'Sample (\d+):'
    result_pattern = r'Answer Correct: (True|False)'
    
    # 分割内容为行
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        # 匹配样本号
        sample_match = re.search(sample_pattern, line)
        if sample_match:
            current_sample = int(sample_match.group(1))
        
        # 匹配答案正确性
        result_match = re.search(result_pattern, line)
        if result_match and current_sample is not None:
            is_correct = result_match.group(1) == 'True'
            results[current_sample].append(is_correct)
    
    return dict(results), num_samples, num_runs


def calculate_maj_at_k(results: Dict[int, List[bool]], k: int) -> float:
    """
    计算maj@k准确率
    对于每个问题，如果k次运行中超过一半的结果正确，则该问题算作正确
    如果样本的结果数量少于k，则使用所有可用结果
    """
    correct_samples = 0
    total_samples = len(results)
    
    for sample_id, sample_results in results.items():
        # 取前k个结果，如果不足k个则取所有结果
        k_results = sample_results[:k]
        actual_k = len(k_results)
        
        if actual_k > 0:
            # 多数投票：超过一半正确则算正确
            correct_count = sum(k_results)
            if correct_count > actual_k // 2:
                correct_samples += 1
    
    return correct_samples / total_samples if total_samples > 0 else 0.0


def calculate_pass_at_k(results: Dict[int, List[bool]], k: int) -> float:
    """
    计算pass@k准确率
    对于每个问题，如果k次运行中至少有一次结果正确，则该问题算作正确
    如果样本的结果数量少于k，则使用所有可用结果
    """
    correct_samples = 0
    total_samples = len(results)
    
    for sample_id, sample_results in results.items():
        # 取前k个结果，如果不足k个则取所有结果
        k_results = sample_results[:k]
        
        if len(k_results) > 0:
            # 至少一次正确则算正确
            if any(k_results):
                correct_samples += 1
    
    return correct_samples / total_samples if total_samples > 0 else 0.0


def print_statistics(results: Dict[int, List[bool]], num_samples: int, num_runs: int, max_display_samples: float = 50):
    """
    打印详细的统计信息
    
    Args:
        results: 每个样本的运行结果
        num_samples: 每轮的样本数量
        num_runs: 运行轮数  
        max_display_samples: 最大显示的样本数量，可以是float('inf')表示显示所有
    """
    print(f"=== 日志文件统计信息 ===")
    print(f"样本数量: {num_samples}")
    print(f"运行轮数: {num_runs}")
    print(f"总结果数: {sum(len(r) for r in results.values())}")
    print()
    
    # 计算每个样本的统计
    sample_stats = []
    for sample_id in sorted(results.keys()):
        sample_results = results[sample_id]
        correct_count = sum(sample_results)
        total_count = len(sample_results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        sample_stats.append((sample_id, correct_count, total_count, accuracy))
    
    print(f"=== 每个样本的统计 ===")
    print(f"{'样本ID':<8} {'正确次数':<10} {'总次数':<8} {'准确率':<10}")
    print("-" * 40)
    
    # 显示所有样本的统计，但如果样本太多，可以选择性显示
    if len(sample_stats) <= max_display_samples or max_display_samples == float('inf'):
        # 样本数少于等于阈值时，或者设置为显示所有样本时，显示所有样本
        for sample_id, correct, total, acc in sample_stats:
            print(f"{sample_id:<8} {correct:<10} {total:<8} {acc:.4f}")
    else:
        # 超过阈值时，显示前面一部分和后面一部分，中间用省略号
        show_front = int(max_display_samples // 2)
        show_back = int(max_display_samples - show_front)
        for sample_id, correct, total, acc in sample_stats[:show_front]:
            print(f"{sample_id:<8} {correct:<10} {total:<8} {acc:.4f}")
        print(f"... (省略中间 {len(sample_stats) - int(max_display_samples)} 个样本)")
        for sample_id, correct, total, acc in sample_stats[-show_back:]:
            print(f"{sample_id:<8} {correct:<10} {total:<8} {acc:.4f}")
    print()
    
    # 计算不同k值的maj@k和pass@k
    max_k = max(len(r) for r in results.values()) if results else 1
    print(f"=== maj@k 和 pass@k 统计 ===")
    print(f"{'k':<5} {'maj@k':<10} {'pass@k':<10}")
    print("-" * 25)
    
    k_values = list(range(1, max_k + 1))
    
    for k in k_values:
        maj_k = calculate_maj_at_k(results, k)
        pass_k = calculate_pass_at_k(results, k)
        print(f"{k:<5} {maj_k:.4f}     {pass_k:.4f}")


def main():
    parser = argparse.ArgumentParser(description='计算模型运行结果的maj@k和pass@k准确率')
    parser.add_argument('log_file', help='日志文件路径')
    parser.add_argument('--k', type=int, help='指定k值计算maj@k和pass@k')
    parser.add_argument('--max-display-samples', type=int, default=50, 
                       help='最大显示样本数量 (默认: 50)')
    parser.add_argument('--show-all-samples', action='store_true',
                       help='显示所有样本的详细统计 (忽略max-display-samples限制)')
    
    args = parser.parse_args()
    
    # 解析日志文件
    print(f"正在解析日志文件: {args.log_file}")
    results, num_samples, num_runs = parse_log_file(args.log_file)
    
    if not results:
        print("错误：无法从日志文件中提取结果")
        return
    
    # 打印统计信息
    max_display = float('inf') if args.show_all_samples else args.max_display_samples
    print_statistics(results, num_samples, num_runs, max_display)
    
    # 如果指定了k值，单独计算
    if args.k:
        maj_k = calculate_maj_at_k(results, args.k)
        pass_k = calculate_pass_at_k(results, args.k)
        print(f"\n=== 指定k={args.k}的结果 ===")
        print(f"maj@{args.k}: {maj_k:.4f}")
        print(f"pass@{args.k}: {pass_k:.4f}")


if __name__ == "__main__":
    main()