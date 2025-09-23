#!/usr/bin/env python3
"""
Script to analyze and visualize the distribution of original_token_decoded values from a JSON file.
Creates a bar chart showing the frequency of each token.
"""

import json
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import os


def load_json_data(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{file_path}': {e}")
        return None


def extract_tokens(data):
    """Extract original_token_decoded values from the data."""
    tokens = []
    for item in data:
        if 'original_token_decoded' in item:
            tokens.append(item['original_token_decoded'])
    return tokens


def plot_token_distribution(tokens, output_file=None, top_n=20):
    """Create a bar chart showing token frequency distribution."""
    # Count token frequencies
    token_counts = Counter(tokens)
    
    # Get top N most frequent tokens
    top_tokens = token_counts.most_common(top_n)
    
    if not top_tokens:
        print("No tokens found to plot.")
        return
    
    # Prepare data for plotting
    token_names = [token for token, count in top_tokens]
    frequencies = [count for token, count in top_tokens]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(token_names)), frequencies, 
                   color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    
    # Customize the plot
    plt.title(f'Token Distribution (Top {top_n})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tokens', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(token_names)), token_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print statistics
    total_tokens = len(tokens)
    unique_tokens = len(token_counts)
    
    print(f"\nStatistics:")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    print(f"\nTop {min(10, len(top_tokens))} most frequent tokens:")
    for i, (token, count) in enumerate(top_tokens[:10], 1):
        percentage = (count / total_tokens) * 100
        print(f"{i:2d}. '{token}': {count:,} ({percentage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Plot token distribution from JSON file')
    parser.add_argument('--input_file', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Output file path for the plot (optional)')
    parser.add_argument('-n', '--top_n', type=int, default=20,
                       help='Number of top tokens to display (default: 20)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_json_data(args.input_file)
    
    if data is None:
        return
    
    # Extract tokens
    print("Extracting tokens...")
    tokens = extract_tokens(data)
    
    if not tokens:
        print("No tokens found in the data.")
        return
    
    # Create output filename if not specified
    output_file = args.output
    if not output_file:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_file = f"{base_name}_token_distribution.png"
    else:
        # Ensure output file has a valid image extension
        valid_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps', '.tiff', '.tif']
        file_ext = os.path.splitext(output_file)[1].lower()
        if not file_ext or file_ext not in valid_extensions:
            output_file = os.path.splitext(output_file)[0] + '.png'
            print(f"Invalid or missing file extension. Output will be saved as: {output_file}")
    
    # Plot distribution
    print("Creating plot...")
    plot_token_distribution(tokens, output_file, args.top_n)


if __name__ == "__main__":
    main()