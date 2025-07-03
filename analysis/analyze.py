import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def load_data(file_path):
    """Loads data from a JSONL file into a pandas DataFrame."""
    try:
        return pd.read_json(file_path, lines=True)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError:
        print(f"Error: Could not decode JSON from {file_path}. Make sure it's a valid JSONL file.")
        return None

def plot_entropy_distributions(df, output_dir):
    """Plots and compares the distribution of original and modified token entropy using line plots based on frequency."""
    if df is None or 'original_entropy' not in df.columns or 'modified_entropy' not in df.columns:
        print("Cannot plot entropy distributions due to missing data.")
        return

    plt.figure(figsize=(12, 7))
    
    # Using histplot with element='poly' to get a frequency polygon (line plot based on counts)
    sns.histplot(data=df, x='original_entropy', bins=50, element="poly", fill=False, color="blue", label='Original Entropy', linewidth=2)
    sns.histplot(data=df, x='modified_entropy', bins=50, element="poly", fill=False, color="red", label='Modified Entropy (after delta)', linestyle="--", linewidth=2)
    
    plt.title('Comparison of Token Entropy Distributions (Frequency Plot)')
    plt.xlabel('Entropy')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis ticks with 0.2 intervals
    import numpy as np
    max_entropy = max(df['original_entropy'].max(), df['modified_entropy'].max())
    min_entropy = min(df['original_entropy'].min(), df['modified_entropy'].min())
    x_ticks = np.arange(0, max_entropy + 0.2, 0.2)
    plt.xticks(x_ticks)
    
    output_path = os.path.join(output_dir, "entropy_distributions_comparison.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory
    print(f"Entropy distributions comparison plot saved to {output_path}")

def analyze_top_tokens(df, output_dir, percentile=30):
    """Analyzes and prints the top N percentile of tokens based on original entropy."""
    if df is None:
        return

    df_sorted = df.sort_values(by='original_entropy', ascending=False)
    
    top_n_count = int(len(df_sorted) * (percentile / 100))
    # Add rank percentile column (smaller percentile means higher rank)
    df_sorted['rank_percentile'] = df_sorted['original_entropy'].rank(pct=True, ascending=False) * 100
    top_tokens_df = df_sorted.head(top_n_count)[['original_predicted_decoded', 'original_entropy', 'rank_percentile']]
    
    print(f"\n--- Top {percentile}% Tokens by Original Entropy ({top_n_count} tokens) ---")
    print(top_tokens_df.to_string())

    # Save to CSV
    output_path = os.path.join(output_dir, f"top_{percentile}_percent_tokens.csv")
    top_tokens_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Top tokens analysis saved to {output_path}")

def plot_entropy_change_net_effect(df, output_dir, bins=50):
    """Plots the net effect of entropy changes (increase count - decrease count) vs original entropy."""
    if df is None or 'original_entropy' not in df.columns or 'modified_entropy' not in df.columns:
        print("Cannot plot entropy change net effect due to missing data.")
        return

    # Calculate entropy differences
    df = df.copy()
    df['entropy_change'] = df['modified_entropy'] - df['original_entropy']
    
    # Create bins for original entropy
    df['entropy_bin'] = pd.cut(df['original_entropy'], bins=bins, include_lowest=True)
    
    # Group by entropy bins and calculate net effect
    net_effects = []
    bin_centers = []
    
    for bin_interval in df['entropy_bin'].cat.categories:
        bin_data = df[df['entropy_bin'] == bin_interval]
        if len(bin_data) == 0:
            continue
            
        # Count increases and decreases
        increases = len(bin_data[bin_data['entropy_change'] > 0])
        decreases = len(bin_data[bin_data['entropy_change'] < 0])
        net_effect = increases - decreases
        
        # Use the midpoint of the bin as x-coordinate
        bin_center = (bin_interval.left + bin_interval.right) / 2
        
        net_effects.append(net_effect)
        bin_centers.append(bin_center)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(bin_centers, net_effects, marker='o', linewidth=2, markersize=4)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.title('Net Effect of Entropy Changes by Original Entropy\n(Entropy Increase Count - Entropy Decrease Count)')
    plt.xlabel('Original Entropy (bin centers)')
    plt.ylabel('Net Effect (Increase Count - Decrease Count)')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, "entropy_change_net_effect.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    print(f"Entropy change net effect plot saved to {output_path}")

def analyze_changed_tokens(df, output_dir):
    """Analyzes tokens that were changed after applying the delta."""
    if df is None:
        return
        
    # Add rank percentile for original_entropy. Higher entropy = smaller percentile.
    df['entropy_rank_percentile'] = df['original_entropy'].rank(pct=True, ascending=False) * 100
    
    changed_tokens = df[df['original_predicted_token'] != df['modified_predicted_token']].copy()
    
    if changed_tokens.empty:
        print("\n--- Analysis of Changed Tokens ---")
        print("No tokens were changed.")
        return
        
    changed_tokens_analysis_df = changed_tokens[[
        'original_predicted_decoded',
        'modified_predicted_decoded',
        'original_entropy',
        'modified_entropy',
        'entropy_diff',
        'entropy_rank_percentile'
    ]]
    
    # Sort the report by rank (ascending) so that top-ranked tokens appear first
    changed_tokens_analysis_df = changed_tokens_analysis_df.sort_values(by='entropy_rank_percentile')
    
    print(f"\n--- Analysis of Changed Tokens ({len(changed_tokens)} tokens) ---")
    print(changed_tokens_analysis_df.to_string())

    # Save to CSV
    output_path = os.path.join(output_dir, "changed_tokens_analysis.csv")
    changed_tokens_analysis_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Changed tokens analysis saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze token entropy from a JSONL file.")
    parser.add_argument(
        "file_path",
        type=str,
        nargs='?',
        default="analysis.jsonl",
        help="Path to the analysis.jsonl file (default: analysis.jsonl in the current directory)."
    )
    args = parser.parse_args()

    # Create analysis directory if it doesn't exist
    output_dir = "analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading data from: {args.file_path}")
    df = load_data(args.file_path)
    
    if df is None:
        print("Exiting due to data loading failure.")
        return

    # 1. Plot entropy distributions
    plot_entropy_distributions(df, output_dir)
    
    # 2. Plot entropy change net effect
    plot_entropy_change_net_effect(df, output_dir)
    
    # 3. Analyze top 20% tokens
    analyze_top_tokens(df, output_dir, percentile=30)
    
    # 4. Analyze changed tokens
    analyze_changed_tokens(df, output_dir)

if __name__ == "__main__":
    main()
