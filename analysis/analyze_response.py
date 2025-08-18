#!/usr/bin/env python3
"""
Response Entropy Visualization Tool

This script visualizes entropy data recorded by the SLOT model.
It reads JSON files containing entropy information and creates plots
showing entropy changes across token indices.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import seaborn as sns

def load_entropy_data(file_path):
    """
    Load entropy data from JSON file
    
    Args:
        file_path (str): Path to the JSON file containing entropy data
        
    Returns:
        list: List of dictionaries containing entropy records
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entropy records from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def extract_entropy_data(data):
    """
    Extract entropy data from the loaded records
    
    Args:
        data (list): List of entropy records
        
    Returns:
        tuple: (indices, original_entropy, modified_entropy, original_tokens, modified_tokens)
    """
    indices = []
    original_entropy = []
    modified_entropy = []
    original_tokens = []
    modified_tokens = []
    
    for record in data:
        indices.append(record.get('token_index', 0))
        original_entropy.append(record.get('original_entropy', 0.0))
        modified_entropy.append(record.get('modified_entropy', 0.0))
        original_tokens.append(record.get('original_token_decoded', ''))
        modified_tokens.append(record.get('modified_token_decoded', ''))
    
    return indices, original_entropy, modified_entropy, original_tokens, modified_tokens

def create_entropy_plot(indices, original_entropy, modified_entropy, 
                       original_tokens, modified_tokens, output_path=None, 
                       show_tokens=False, max_tokens_display=10):
    """
    Create entropy visualization plot
    
    Args:
        indices (list): Token indices
        original_entropy (list): Original entropy values
        modified_entropy (list): Modified entropy values
        original_tokens (list): Original token texts
        modified_tokens (list): Modified token texts
        output_path (str): Path to save the plot
        show_tokens (bool): Whether to show token annotations
        max_tokens_display (int): Maximum number of tokens to display as annotations
    """
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Response Token Entropy Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Entropy comparison
    ax1.plot(indices, original_entropy, 'o-', label='Original Entropy', 
             color='blue', alpha=0.7, linewidth=2, markersize=6)
    ax1.plot(indices, modified_entropy, 's-', label='Modified Entropy', 
             color='red', alpha=0.7, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Token Index', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.set_title('Entropy Comparison: Original vs Modified', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add token annotations if requested
    if show_tokens and len(indices) <= max_tokens_display:
        for i, (idx, orig_token, mod_token) in enumerate(zip(indices, original_tokens, modified_tokens)):
            if orig_token != mod_token:
                ax1.annotate(f'{orig_token}→{mod_token}', 
                            xy=(idx, original_entropy[i]), 
                            xytext=(idx, original_entropy[i] + 0.1),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                            fontsize=8, ha='center')
    
    # Plot 2: Entropy difference
    entropy_diff = np.array(modified_entropy) - np.array(original_entropy)
    colors = ['red' if diff > 0 else 'green' for diff in entropy_diff]
    
    bars = ax2.bar(indices, entropy_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax2.set_xlabel('Token Index', fontsize=12)
    ax2.set_ylabel('Entropy Difference (Modified - Original)', fontsize=12)
    ax2.set_title('Entropy Change Analysis', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, diff) in enumerate(zip(bars, entropy_diff)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{diff:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # Add statistics
    stats_text = f"""
Statistics:
- Mean Original Entropy: {np.mean(original_entropy):.3f}
- Mean Modified Entropy: {np.mean(modified_entropy):.3f}
- Mean Entropy Change: {np.mean(entropy_diff):.3f}
- Max Entropy Increase: {np.max(entropy_diff):.3f}
- Max Entropy Decrease: {np.min(entropy_diff):.3f}
- Tokens with Increased Entropy: {np.sum(entropy_diff > 0)}/{len(entropy_diff)}
"""
    
    # Add statistics text box
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def create_token_comparison_table(data, output_path=None):
    """
    Create a detailed token comparison table
    
    Args:
        data (list): Entropy data
        output_path (str): Path to save the table
    """
    if not data:
        return
    
    # Create table data
    table_data = []
    for record in data:
        table_data.append([
            record.get('token_index', ''),
            record.get('original_entropy', 0.0),
            record.get('modified_entropy', 0.0),
            record.get('original_token_decoded', ''),
            record.get('modified_token_decoded', ''),
            record.get('modified_entropy', 0.0) - record.get('original_entropy', 0.0)
        ])
    
    # Create table
    fig, ax = plt.subplots(figsize=(14, max(8, len(table_data) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Define column headers
    headers = ['Index', 'Original Entropy', 'Modified Entropy', 'Original Token', 'Modified Token', 'Entropy Diff']
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code entropy differences
    for i, row in enumerate(table_data):
        diff = row[5]
        if diff > 0:
            table[(i+1, 5)].set_facecolor('#ffcdd2')  # Light red for increase
        elif diff < 0:
            table[(i+1, 5)].set_facecolor('#c8e6c9')  # Light green for decrease
    
    plt.title('Detailed Token Comparison Table', fontsize=16, fontweight='bold', pad=20)
    
    if output_path:
        table_path = output_path.replace('.png', '_table.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        print(f"Table saved to {table_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize response entropy data')
    parser.add_argument('--file_path', help='Path to the JSON file containing entropy data')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    parser.add_argument('--show-tokens', action='store_true', 
                       help='Show token annotations on the plot')
    parser.add_argument('--max-tokens', type=int, default=10,
                       help='Maximum number of tokens to display as annotations')
    parser.add_argument('--table', action='store_true',
                       help='Generate a detailed token comparison table')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} does not exist")
        sys.exit(1)
    
    # Load data
    data = load_entropy_data(args.file_path)
    if data is None:
        sys.exit(1)
    
    # Extract data
    indices, original_entropy, modified_entropy, original_tokens, modified_tokens = extract_entropy_data(data)
    
    if not indices:
        print("Error: No valid data found in the file")
        sys.exit(1)
    
    # Generate output path based on JSON file name if not specified
    if not args.output:
        json_file_name = Path(args.file_path).stem
        args.output = f"{json_file_name}.png"
    
    # Create plots
    create_entropy_plot(indices, original_entropy, modified_entropy, 
                       original_tokens, modified_tokens, 
                       output_path=args.output,
                       show_tokens=args.show_tokens,
                       max_tokens_display=args.max_tokens)
    
    # Create table if requested
    if args.table:
        create_token_comparison_table(data, output_path=args.output)

if __name__ == "__main__":
    main()
