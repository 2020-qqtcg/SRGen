#!/usr/bin/env python3
"""
Simple script to plot token distribution from JSON file.
Usage: python plot_tokens.py
"""

import json
import matplotlib.pyplot as plt
from collections import Counter
import os

def main():
    # Fixed input file path
    json_file = "analysis/source/token.json"
    
    # Load JSON data
    print(f"Loading data from: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    
    # Extract tokens
    print("Extracting tokens...")
    tokens = []
    for item in data:
        if 'original_token_decoded' in item:
            tokens.append(item['original_token_decoded'])
    
    print(f"Extracted {len(tokens)} tokens")
    
    # Count frequencies
    token_counts = Counter(tokens)
    top_tokens = token_counts.most_common(20)
    
    # Prepare data for plotting
    token_names = [token for token, count in top_tokens]
    frequencies = [count for token, count in top_tokens]
    
    # Create plot similar to the reference image
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(token_names)), frequencies, 
                   color='steelblue', alpha=0.9, edgecolor='darkblue', linewidth=0.5)
    
    # Customize plot to match the reference style
    plt.title('Token Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Tokens', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    
    # Set x-axis labels with rotation
    plt.xticks(range(len(token_names)), token_names, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Style improvements
    plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = 'token_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_file}")
    
    # Print statistics
    total_tokens = len(tokens)
    unique_tokens = len(token_counts)
    
    print(f"\n=== Statistics ===")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    
    print(f"\n=== Top 15 most frequent tokens ===")
    for i, (token, count) in enumerate(top_tokens[:15], 1):
        percentage = (count / total_tokens) * 100
        # Handle special characters in token display
        display_token = repr(token) if any(ord(c) < 32 or ord(c) > 126 for c in token) else token
        print(f"{i:2d}. {display_token:<15}: {count:>6,} ({percentage:>5.2f}%)")

if __name__ == "__main__":
    main()