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

def plot_entropy_distributions_comparison(df1, df2, output_dir, label1="Dataset 1", label2="Dataset 2"):
    """Plots and compares the distribution of original and modified token entropy for two datasets."""
    if df1 is None or df2 is None:
        print("Cannot plot entropy distributions due to missing data.")
        return
    
    if 'original_entropy' not in df1.columns or 'modified_entropy' not in df1.columns or \
       'original_entropy' not in df2.columns or 'modified_entropy' not in df2.columns:
        print("Cannot plot entropy distributions due to missing columns.")
        return

    import numpy as np
    
    # Define common bins for consistent comparison
    all_original = list(df1['original_entropy']) + list(df2['original_entropy'])
    all_modified = list(df1['modified_entropy']) + list(df2['modified_entropy'])
    
    original_bins = np.linspace(min(all_original), max(all_original), 51)
    modified_bins = np.linspace(min(all_modified), max(all_modified), 51)

    plt.figure(figsize=(18, 12))
    
    # Original entropy comparison
    plt.subplot(2, 3, 1)
    sns.histplot(data=df1, x='original_entropy', bins=original_bins, element="poly", fill=False, 
                color="blue", label=f'{label1} Original', linewidth=2, stat='percent')
    sns.histplot(data=df2, x='original_entropy', bins=original_bins, element="poly", fill=False, 
                color="red", label=f'{label2} Original', linewidth=2, linestyle="--", stat='percent')
    plt.title('Original Entropy Distributions Comparison')
    plt.xlabel('Original Entropy')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    
    # Modified entropy comparison
    plt.subplot(2, 3, 2)
    sns.histplot(data=df1, x='modified_entropy', bins=modified_bins, element="poly", fill=False, 
                color="blue", label=f'{label1} Modified', linewidth=2, stat='percent')
    sns.histplot(data=df2, x='modified_entropy', bins=modified_bins, element="poly", fill=False, 
                color="red", label=f'{label2} Modified', linewidth=2, linestyle="--", stat='percent')
    plt.title('Modified Entropy Distributions Comparison')
    plt.xlabel('Modified Entropy')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    
    # Original entropy difference (Dataset1 - Dataset2) in percentage
    plt.subplot(2, 3, 3)
    hist1_orig, _ = np.histogram(df1['original_entropy'], bins=original_bins)
    hist2_orig, _ = np.histogram(df2['original_entropy'], bins=original_bins)
    bin_centers_orig = (original_bins[:-1] + original_bins[1:]) / 2
    
    # Convert to percentages
    hist1_orig_pct = (hist1_orig / len(df1)) * 100
    hist2_orig_pct = (hist2_orig / len(df2)) * 100
    diff_orig_pct = hist1_orig_pct - hist2_orig_pct
    
    plt.plot(bin_centers_orig, diff_orig_pct, color='purple', linewidth=2, marker='o', markersize=3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Original Entropy Difference\n({label1} - {label2})')
    plt.xlabel('Original Entropy')
    plt.ylabel('Percentage Difference (%)')
    
    # Set adaptive y-axis ticks based on percentage range
    y_min_orig, y_max_orig = plt.ylim()
    max_abs_val_orig = max(abs(y_min_orig), abs(y_max_orig))
    
    # Choose interval based on the maximum absolute percentage value
    if max_abs_val_orig <= 1:
        interval_orig = 0.2
    elif max_abs_val_orig <= 5:
        interval_orig = 0.5
    elif max_abs_val_orig <= 10:
        interval_orig = 1
    elif max_abs_val_orig <= 20:
        interval_orig = 2
    else:
        interval_orig = 5
    
    y_ticks_orig = np.arange(int(y_min_orig / interval_orig) * interval_orig, 
                            int(y_max_orig / interval_orig + 1) * interval_orig + interval_orig, 
                            interval_orig)
    plt.yticks(y_ticks_orig)
    plt.grid(True)
    
    # Dataset 1: Original vs Modified
    plt.subplot(2, 3, 4)
    sns.histplot(data=df1, x='original_entropy', bins=50, element="poly", fill=False, 
                color="blue", label=f'{label1} Original', linewidth=2, stat='percent')
    sns.histplot(data=df1, x='modified_entropy', bins=50, element="poly", fill=False, 
                color="lightblue", label=f'{label1} Modified', linewidth=2, linestyle=":", stat='percent')
    plt.title(f'{label1}: Original vs Modified Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    
    # Dataset 2: Original vs Modified
    plt.subplot(2, 3, 5)
    sns.histplot(data=df2, x='original_entropy', bins=50, element="poly", fill=False, 
                color="red", label=f'{label2} Original', linewidth=2, stat='percent')
    sns.histplot(data=df2, x='modified_entropy', bins=50, element="poly", fill=False, 
                color="lightcoral", label=f'{label2} Modified', linewidth=2, linestyle=":", stat='percent')
    plt.title(f'{label2}: Original vs Modified Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    
    # Modified entropy difference (Dataset1 - Dataset2) in percentage
    plt.subplot(2, 3, 6)
    hist1_mod, _ = np.histogram(df1['modified_entropy'], bins=modified_bins)
    hist2_mod, _ = np.histogram(df2['modified_entropy'], bins=modified_bins)
    bin_centers_mod = (modified_bins[:-1] + modified_bins[1:]) / 2
    
    # Convert to percentages
    hist1_mod_pct = (hist1_mod / len(df1)) * 100
    hist2_mod_pct = (hist2_mod / len(df2)) * 100
    diff_mod_pct = hist1_mod_pct - hist2_mod_pct
    
    plt.plot(bin_centers_mod, diff_mod_pct, color='orange', linewidth=2, marker='s', markersize=3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Modified Entropy Difference\n({label1} - {label2})')
    plt.xlabel('Modified Entropy')
    plt.ylabel('Percentage Difference (%)')
    
    # Set adaptive y-axis ticks based on percentage range
    y_min_mod, y_max_mod = plt.ylim()
    max_abs_val_mod = max(abs(y_min_mod), abs(y_max_mod))
    
    # Choose interval based on the maximum absolute percentage value
    if max_abs_val_mod <= 1:
        interval_mod = 0.2
    elif max_abs_val_mod <= 5:
        interval_mod = 0.5
    elif max_abs_val_mod <= 10:
        interval_mod = 1
    elif max_abs_val_mod <= 20:
        interval_mod = 2
    else:
        interval_mod = 5
    
    y_ticks_mod = np.arange(int(y_min_mod / interval_mod) * interval_mod, 
                           int(y_max_mod / interval_mod + 1) * interval_mod + interval_mod, 
                           interval_mod)
    plt.yticks(y_ticks_mod)
    plt.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "entropy_distributions_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Entropy distributions comparison plot saved to {output_path}")

def plot_entropy_change_net_effect_comparison(df1, df2, output_dir, label1="Dataset 1", label2="Dataset 2", bins=50):
    """Plots the net effect of entropy changes for two datasets in comparison."""
    if df1 is None or df2 is None:
        print("Cannot plot entropy change net effect due to missing data.")
        return

    def calculate_net_effects(df, bins):
        df = df.copy()
        df['entropy_change'] = df['modified_entropy'] - df['original_entropy']
        df['entropy_bin'] = pd.cut(df['original_entropy'], bins=bins, include_lowest=True)
        
        net_effects = []
        bin_centers = []
        total_tokens = len(df)
        
        for bin_interval in df['entropy_bin'].cat.categories:
            bin_data = df[df['entropy_bin'] == bin_interval]
            if len(bin_data) == 0:
                continue
                
            increases = len(bin_data[bin_data['entropy_change'] > 0])
            decreases = len(bin_data[bin_data['entropy_change'] < 0])
            # Convert to percentages
            increases_pct = (increases / total_tokens) * 100
            decreases_pct = (decreases / total_tokens) * 100
            net_effect_pct = increases_pct - decreases_pct
            
            bin_center = (bin_interval.left + bin_interval.right) / 2
            
            net_effects.append(net_effect_pct)
            bin_centers.append(bin_center)
        
        return bin_centers, net_effects

    # Calculate net effects for both datasets
    bin_centers1, net_effects1 = calculate_net_effects(df1, bins)
    bin_centers2, net_effects2 = calculate_net_effects(df2, bins)

    plt.figure(figsize=(15, 6))
    
    # Combined comparison plot
    plt.subplot(1, 2, 1)
    plt.plot(bin_centers1, net_effects1, marker='o', linewidth=2, markersize=4, 
             color='blue', label=label1)
    plt.plot(bin_centers2, net_effects2, marker='s', linewidth=2, markersize=4, 
             color='red', label=label2, linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.title('Net Effect of Entropy Changes Comparison\n(Increase % - Decrease %)')
    plt.xlabel('Original Entropy (bin centers)')
    plt.ylabel('Net Effect (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Difference plot
    plt.subplot(1, 2, 2)
    # Interpolate to same x-axis for subtraction
    import numpy as np
    all_centers = sorted(set(bin_centers1 + bin_centers2))
    net_effects1_interp = np.interp(all_centers, bin_centers1, net_effects1)
    net_effects2_interp = np.interp(all_centers, bin_centers2, net_effects2)
    difference = net_effects1_interp - net_effects2_interp
    
    plt.plot(all_centers, difference, marker='d', linewidth=2, markersize=4, 
             color='purple', label=f'{label1} - {label2}')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.title(f'Difference in Net Effects\n({label1} - {label2})')
    plt.xlabel('Original Entropy (bin centers)')
    plt.ylabel('Net Effect Difference (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "entropy_change_net_effect_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Entropy change net effect comparison plot saved to {output_path}")

def compare_top_tokens(df1, df2, output_dir, label1="Dataset 1", label2="Dataset 2", percentile=20):
    """Compares the top N percentile of tokens based on original entropy for two datasets."""
    if df1 is None or df2 is None:
        return

    def get_top_tokens(df, percentile):
        df_sorted = df.sort_values(by='original_entropy', ascending=False)
        top_n_count = int(len(df_sorted) * (percentile / 100))
        return df_sorted.head(top_n_count)[['original_predicted_decoded', 'original_entropy']]

    top_tokens1 = get_top_tokens(df1, percentile)
    top_tokens2 = get_top_tokens(df2, percentile)
    
    print(f"\n--- Top {percentile}% Tokens Comparison ---")
    print(f"\n{label1} (Top {len(top_tokens1)} tokens):")
    print(top_tokens1.to_string())
    
    print(f"\n{label2} (Top {len(top_tokens2)} tokens):")
    print(top_tokens2.to_string())
    
    # Find common tokens in top percentile
    common_tokens = set(top_tokens1['original_predicted_decoded']) & set(top_tokens2['original_predicted_decoded'])
    print(f"\n--- Common Tokens in Top {percentile}% ({len(common_tokens)} tokens) ---")
    for token in sorted(common_tokens):
        entropy1 = top_tokens1[top_tokens1['original_predicted_decoded'] == token]['original_entropy'].iloc[0]
        entropy2 = top_tokens2[top_tokens2['original_predicted_decoded'] == token]['original_entropy'].iloc[0]
        print(f"'{token}': {label1}={entropy1:.4f}, {label2}={entropy2:.4f}, diff={entropy1-entropy2:.4f}")

    # Save comparison results
    output_path1 = os.path.join(output_dir, f"top_{percentile}_percent_tokens_{label1.replace(' ', '_')}.csv")
    output_path2 = os.path.join(output_dir, f"top_{percentile}_percent_tokens_{label2.replace(' ', '_')}.csv")
    
    top_tokens1.to_csv(output_path1, index=False, encoding='utf-8')
    top_tokens2.to_csv(output_path2, index=False, encoding='utf-8')
    
    print(f"Top tokens analysis saved to {output_path1} and {output_path2}")

def compare_changed_tokens(df1, df2, output_dir, label1="Dataset 1", label2="Dataset 2"):
    """Compares tokens that were changed after applying the delta for two datasets."""
    if df1 is None or df2 is None:
        return
        
    def analyze_changed_tokens(df):
        df['entropy_rank_percentile'] = df['original_entropy'].rank(pct=True, ascending=False) * 100
        changed_tokens = df[df['original_predicted_token'] != df['modified_predicted_token']].copy()
        
        if changed_tokens.empty:
            return pd.DataFrame()
            
        return changed_tokens[[
            'original_predicted_decoded',
            'modified_predicted_decoded',
            'original_entropy',
            'modified_entropy',
            'entropy_diff',
            'entropy_rank_percentile'
        ]].sort_values(by='entropy_rank_percentile')

    changed1 = analyze_changed_tokens(df1)
    changed2 = analyze_changed_tokens(df2)
    
    print(f"\n--- Changed Tokens Comparison ---")
    print(f"\n{label1} ({len(changed1)} changed tokens):")
    if not changed1.empty:
        print(changed1.to_string())
    else:
        print("No tokens were changed.")
        
    print(f"\n{label2} ({len(changed2)} changed tokens):")
    if not changed2.empty:
        print(changed2.to_string())
    else:
        print("No tokens were changed.")

    # Save comparison results
    if not changed1.empty:
        output_path1 = os.path.join(output_dir, f"changed_tokens_{label1.replace(' ', '_')}.csv")
        changed1.to_csv(output_path1, index=False, encoding='utf-8')
        print(f"Changed tokens analysis for {label1} saved to {output_path1}")
        
    if not changed2.empty:
        output_path2 = os.path.join(output_dir, f"changed_tokens_{label2.replace(' ', '_')}.csv")
        changed2.to_csv(output_path2, index=False, encoding='utf-8')
        print(f"Changed tokens analysis for {label2} saved to {output_path2}")

def main():
    parser = argparse.ArgumentParser(description="Compare token entropy analysis from two JSONL files.")
    parser.add_argument(
        "file1",
        type=str,
        help="Path to the first analysis.jsonl file."
    )
    parser.add_argument(
        "file2", 
        type=str,
        help="Path to the second analysis.jsonl file."
    )
    parser.add_argument(
        "--label1",
        type=str,
        default="Dataset 1",
        help="Label for the first dataset (default: 'Dataset 1')."
    )
    parser.add_argument(
        "--label2",
        type=str, 
        default="Dataset 2",
        help="Label for the second dataset (default: 'Dataset 2')."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for results (default: 'comparison_results')."
    )
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading data from: {args.file1} and {args.file2}")
    df1 = load_data(args.file1)
    df2 = load_data(args.file2)
    
    if df1 is None or df2 is None:
        print("Exiting due to data loading failure.")
        return

    print(f"Loaded {len(df1)} records from {args.file1}")
    print(f"Loaded {len(df2)} records from {args.file2}")

    # 1. Plot entropy distributions comparison
    plot_entropy_distributions_comparison(df1, df2, args.output_dir, args.label1, args.label2)
    
    # 2. Plot entropy change net effect comparison
    plot_entropy_change_net_effect_comparison(df1, df2, args.output_dir, args.label1, args.label2)
    
    # 3. Compare top 20% tokens
    compare_top_tokens(df1, df2, args.output_dir, args.label1, args.label2, percentile=20)
    
    # 4. Compare changed tokens
    compare_changed_tokens(df1, df2, args.output_dir, args.label1, args.label2)

if __name__ == "__main__":
    main()
