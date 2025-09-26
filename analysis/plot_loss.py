import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import matplotlib

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
})

# ---- Helpers for paper-style curves (raw line + smoothed mean + ±1 std band) ----
SMOOTH_WINDOW = 51  # can be tuned

def _rolling_mean_std(y, window=SMOOTH_WINDOW):
    import numpy as np
    s = pd.Series(y, dtype=float)
    mean = s.rolling(window, center=True).mean().to_numpy()
    std = s.rolling(window, center=True).std(ddof=0).to_numpy()
    # Fallback for short series: replace all-NaN with the original values/zeros
    if np.all(np.isnan(mean)):
        mean = np.array(y, dtype=float)
        std = np.zeros_like(mean)
    return mean, std

def _style_axes_with_frame(ax):
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.0)
        ax.spines[side].set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=12, width=2.0)

def _plot_series_with_band(ax, steps, y, color, label=None, window=SMOOTH_WINDOW):
    # raw line (light)
    ax.plot(steps, y, linewidth=1.0, alpha=0.35, color=color)
    # rolling mean + ±1 std band
    mean, std = _rolling_mean_std(y, window=min(window, max(3, (len(y)//20)*2 + 1)))
    ax.plot(steps, mean, linewidth=2.0, color=color, label=label)
    ax.fill_between(steps, mean - std, mean + std, where=~np.isnan(mean),
                    alpha=0.15, color=color, linewidth=0)

def _annotate_last(ax, steps, y, text_prefix="Step"):
    if not steps:
        return
    x_last, y_last = steps[-1], y[-1]
    ax.annotate(f"{text_prefix}: {x_last}, Value: {y_last:.4f}",
                xy=(x_last, y_last),
                xytext=(max(steps)*0.7, y_last),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red"),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                fontsize=9)

def load_loss_data(json_file_path):
    """
    Load loss data from JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing loss data
        
    Returns:
        tuple: (ce_loss_list, entropy_loss_list, loss_list)
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    ce_loss = [item['ce_loss'] for item in data]
    entropy_loss = [item['entropy_loss'] for item in data]
    loss = [item['loss'] for item in data]
    
    return ce_loss, entropy_loss, loss

def plot_loss_curves(json_file_path, save_path=None, title_prefix="Training Loss"):
    """
    Plot 4 subplots: ce_loss curve, entropy_loss curve, total loss curve, and scatter plot.
    
    Args:
        json_file_path (str): Path to the JSON file containing loss data
        save_path (str, optional): Path to save the figure
        title_prefix (str): Prefix for subplot titles
    """
    # Load data
    ce_loss, entropy_loss, loss = load_loss_data(json_file_path)
    
    # Create steps for x-axis
    steps = list(range(1, len(ce_loss) + 1))
    
    # Limit first 3 plots to first 400 data points
    limit = min(500, len(ce_loss))
    steps_limited = steps[:limit]
    ce_loss_limited = ce_loss[:limit]
    entropy_loss_limited = entropy_loss[:limit]
    loss_limited = loss[:limit]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Academic color scheme
    color_ce = '#2E86AB'       # Professional blue
    color_entropy = '#A23B72'  # Professional magenta  
    color_total = '#2D9966'    # Professional green
    
    # Calculate unified y-axis range for the first 3 plots
    all_loss_values = ce_loss_limited + entropy_loss_limited + loss_limited
    y_min, y_max = min(all_loss_values), max(all_loss_values)
    y_margin = (y_max - y_min) * 0.1  # Add 10% margin
    unified_ylim = (y_min - y_margin, y_max + y_margin)

    # Plot 1: CE Loss curve
    _plot_series_with_band(axes[0], steps_limited, ce_loss_limited, color_ce)
    axes[0].set_title(f'{title_prefix}\nCE Loss (First {limit} Steps)', fontweight='bold', pad=15)
    axes[0].set_xlabel('Steps', fontweight='bold')
    axes[0].set_ylabel('CE Loss', fontweight='bold')
    axes[0].set_ylim(unified_ylim)
    _style_axes_with_frame(axes[0])
    _annotate_last(axes[0], steps_limited, ce_loss_limited)

    # Plot 2: Entropy Loss curve
    _plot_series_with_band(axes[1], steps_limited, entropy_loss_limited, color_entropy)
    axes[1].set_title(f'{title_prefix}\nEntropy Loss (First {limit} Steps)', fontweight='bold', pad=15)
    axes[1].set_xlabel('Steps', fontweight='bold')
    axes[1].set_ylabel('Entropy Loss', fontweight='bold')
    axes[1].set_ylim(unified_ylim)
    _style_axes_with_frame(axes[1])
    _annotate_last(axes[1], steps_limited, entropy_loss_limited)

    # Plot 3: Total Loss curve
    _plot_series_with_band(axes[2], steps_limited, loss_limited, color_total)
    axes[2].set_title(f'{title_prefix}\nTotal Loss (First {limit} Steps)', fontweight='bold', pad=15)
    axes[2].set_xlabel('Steps', fontweight='bold')
    axes[2].set_ylabel('Total Loss', fontweight='bold')
    axes[2].set_ylim(unified_ylim)
    _style_axes_with_frame(axes[2])
    _annotate_last(axes[2], steps_limited, loss_limited)
    
    # Plot 4: Scatter plot (CE Loss vs Entropy Loss) - all data
    axes[3].scatter(ce_loss, entropy_loss, c=steps, cmap='viridis',
                   alpha=0.7, s=20, edgecolors='white', linewidth=0.5)
    axes[3].set_title(f'{title_prefix}\nCE vs Entropy Loss', fontweight='bold', pad=15)
    axes[3].set_xlabel('CE Loss', fontweight='bold')
    axes[3].set_ylabel('Entropy Loss', fontweight='bold')
    axes[3].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # Add colorbar for the scatter plot
    cbar = plt.colorbar(axes[3].collections[0], ax=axes[3], shrink=0.8)
    cbar.set_label('Steps', fontweight='bold')

    # Style the scatter subplot
    _style_axes_with_frame(axes[3])

    # Adjust layout with proper spacing
    plt.tight_layout(pad=2.0)
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Loss curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_lr_comparison(json_files, save_path=None, title="Learning Rate Comparison", limit_steps=500, scatter_mode=False):
    """
    Compare the same type of loss across different learning rates.
    
    Args:
        json_files (list): List of paths to JSON files (should be 4 files)
        save_path (str, optional): Path to save the figure
        title (str): Main title for the figure
        limit_steps (int): Number of steps to display for each loss type
        scatter_mode (bool): If True, generate 4 scatter plots (CE vs Entropy Loss)
    """
    if len(json_files) != 4:
        raise ValueError("Exactly 4 JSON files are required for comparison")
    
    # Extract learning rates from filenames
    learning_rates = []
    all_data = []
    
    for json_file in json_files:
        # Extract lr from filename (e.g., "loss_0.1.json" -> "0.1")
        import os
        filename = os.path.basename(json_file)
        lr_value = filename.split('_')[1].split('.json')[0]
        learning_rates.append(lr_value)
        
        # Load data
        ce_loss, entropy_loss, loss = load_loss_data(json_file)
        all_data.append((ce_loss, entropy_loss, loss))
    
    if scatter_mode:
        # Create figure with 4 subplots for scatter plots (2x2 layout)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Convert 2D array to 1D for easier indexing
        
        # Generate scatter plots for each learning rate
        for file_idx, (ce_loss, entropy_loss, loss) in enumerate(all_data):
            ax = axes[file_idx]
            
            # Limit data points
            ce_limited = ce_loss[:min(limit_steps, len(ce_loss))]
            entropy_limited = entropy_loss[:min(limit_steps, len(entropy_loss))]
            steps = list(range(len(ce_limited)))
            
            # Create scatter plot
            scatter = ax.scatter(ce_limited, entropy_limited, c=steps, cmap='viridis', 
                               alpha=0.7, s=15, edgecolors='white', linewidth=0.3)
            
            # Add colorbar for each subplot
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Steps', fontweight='bold', fontsize=10)
            
            # Formatting
            lr_value = learning_rates[file_idx]
            ax.set_title(f'{title}\nlr={lr_value}', fontweight='bold', pad=15)
            ax.set_xlabel('CE Loss', fontweight='bold')
            ax.set_ylabel('Entropy Loss', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
            
            # Apply styling
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
                spine.set_color('black')
            
            ax.tick_params(axis='both', which='major', labelsize=12, width=2.0)
        
        # Adjust layout
        plt.tight_layout(pad=2.0)
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Scatter plot comparison saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return
    
    # Create figure with 3 subplots (one for each loss type) - original mode
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Academic color scheme for different learning rates
    colors = ['#2E86AB', '#A23B72', '#2D9966', '#E67E22']

    # Loss types and their indices
    loss_types = ['CE Loss', 'Entropy Loss', 'Total Loss']
    loss_indices = [0, 1, 2]  # ce_loss, entropy_loss, loss

    for plot_idx, (loss_type, loss_idx) in enumerate(zip(loss_types, loss_indices)):
        ax = axes[plot_idx]

        # Collect all loss values for this type to calculate unified y-range
        all_values_for_type = []

        for file_idx, (ce_loss, entropy_loss, loss) in enumerate(all_data):
            # Get the appropriate loss data
            if loss_idx == 0:
                loss_data = ce_loss
            elif loss_idx == 1:
                loss_data = entropy_loss
            else:
                loss_data = loss

            # Limit data points
            limited_data = loss_data[:min(limit_steps, len(loss_data))]
            # steps start at 1 for clarity
            steps = list(range(1, len(limited_data) + 1))
            # Plot the curve with smooth line and band
            lr_label = f"lr={learning_rates[file_idx]}"
            _plot_series_with_band(ax, steps, limited_data, colors[file_idx], label=lr_label)
            # Collect values for unified y-range
            all_values_for_type.extend(limited_data)

        # Set unified y-axis range for this loss type
        y_min, y_max = min(all_values_for_type), max(all_values_for_type)
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Formatting
        ax.set_title(f'{title}\n{loss_type} (First {limit_steps} Steps)',
                     fontweight='bold', pad=15)
        ax.set_xlabel('Steps', fontweight='bold')
        ax.set_ylabel(loss_type, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        # ax.legend(loc='upper right', fontsize=11)  # Removed per-axes legend

        # Apply styling
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

        ax.tick_params(axis='both', which='major', labelsize=12, width=2.0)

    # Single horizontal legend centered at the bottom (tighter paddings)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=len(labels),
               frameon=True, fancybox=True, framealpha=1.0,
               edgecolor='black', fontsize=16,
               labelspacing=0.4, handletextpad=0.6, borderpad=0.3, columnspacing=1.0)

    # Adjust layout (smaller bottom margin)
    plt.tight_layout(pad=1.6, rect=[0, 0.08, 1, 1])

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Learning rate comparison saved to: {save_path}")
    else:
        plt.show()

    plt.close()

def plot_loss_comparison(json_files, labels, save_path=None, title="Loss Comparison"):
    """
    Compare loss curves from multiple JSON files.
    
    Args:
        json_files (list): List of paths to JSON files
        labels (list): List of labels for each file
        save_path (str, optional): Path to save the figure
        title (str): Main title for the figure
    """
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Color palette for different files
    colors = ['#2E86AB', '#A23B72', '#2D9966', '#E67E22', '#9B59B6', '#F39C12']
    
    for i, (json_file, label) in enumerate(zip(json_files, labels)):
        # Load data
        ce_loss, entropy_loss, loss = load_loss_data(json_file)
        steps = list(range(1, len(ce_loss) + 1))
        color = colors[i % len(colors)]
        
        # Plot curves with banded style
        _plot_series_with_band(axes[0], steps, ce_loss, color, label=label)
        _plot_series_with_band(axes[1], steps, entropy_loss, color, label=label)
        _plot_series_with_band(axes[2], steps, loss, color, label=label)
        axes[3].scatter(ce_loss, entropy_loss, c=color, alpha=0.6, s=15, label=label)
    
    # Set titles and labels
    axes[0].set_title(f'{title}\nCE Loss', fontweight='bold', pad=15)
    axes[0].set_xlabel('Steps', fontweight='bold')
    axes[0].set_ylabel('CE Loss', fontweight='bold')

    axes[1].set_title(f'{title}\nEntropy Loss', fontweight='bold', pad=15)
    axes[1].set_xlabel('Steps', fontweight='bold')
    axes[1].set_ylabel('Entropy Loss', fontweight='bold')

    axes[2].set_title(f'{title}\nTotal Loss', fontweight='bold', pad=15)
    axes[2].set_xlabel('Steps', fontweight='bold')
    axes[2].set_ylabel('Total Loss', fontweight='bold')

    axes[3].set_title(f'{title}\nCE vs Entropy Loss', fontweight='bold', pad=15)
    axes[3].set_xlabel('CE Loss', fontweight='bold')
    axes[3].set_ylabel('Entropy Loss', fontweight='bold')

    # Apply styling
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

        # Make all borders bold and visible
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # Set line width for all spines
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')

        # Format ticks
        ax.tick_params(axis='both', which='major', labelsize=12, width=2.0)

    # Single horizontal legend centered at the bottom for the whole figure (tighter paddings)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center', ncol=len(labels),
               frameon=True, fancybox=True, framealpha=1.0,
               edgecolor='black', fontsize=16,
               labelspacing=0.4, handletextpad=0.6, borderpad=0.3, columnspacing=1.0)

    # Adjust layout (smaller bottom margin)
    plt.tight_layout(pad=1.6, rect=[0, 0.08, 1, 1])

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

# Example usage
if __name__ == "__main__":
    # Learning rate comparison (4 files)
    json_files = [
        "source/loss_0.0001.json",
        "source/loss_0.001.json", 
        "source/loss_0.01.json",
        "source/loss_0.1.json"
    ]
    
    try:
        # Plot learning rate comparison (line charts)
        plot_lr_comparison(json_files, save_path="image/lr_comparison.png", 
                          title="", limit_steps=1000)
        
        # Plot learning rate comparison (scatter plots)
        plot_lr_comparison(json_files, save_path="image/lr_scatter_comparison.png", 
                          title="", limit_steps=1000, 
                          scatter_mode=True)
        
        # Single file plotting (optional)
        # plot_loss_curves("source/loss_0.1.json", save_path="loss_curves.png", 
        #                 title_prefix="Training Progress")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
