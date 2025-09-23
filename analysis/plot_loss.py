import matplotlib.pyplot as plt
import numpy as np
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
    'legend.fontsize': 12,
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
    steps = list(range(len(ce_loss)))
    
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
    
    # Plot 1: CE Loss curve (first 200 points)
    axes[0].plot(steps_limited, ce_loss_limited, color=color_ce, alpha=0.9)
    axes[0].set_title(f'{title_prefix}\nCE Loss (First 200 Steps)', fontweight='bold', pad=15)
    axes[0].set_xlabel('Steps', fontweight='bold')
    axes[0].set_ylabel('CE Loss', fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    axes[0].set_ylim(unified_ylim)
    
    # Plot 2: Entropy Loss curve (first 200 points)
    axes[1].plot(steps_limited, entropy_loss_limited, color=color_entropy, alpha=0.9)
    axes[1].set_title(f'{title_prefix}\nEntropy Loss (First 200 Steps)', fontweight='bold', pad=15)
    axes[1].set_xlabel('Steps', fontweight='bold')
    axes[1].set_ylabel('Entropy Loss', fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    axes[1].set_ylim(unified_ylim)
    
    # Plot 3: Total Loss curve (first 200 points)
    axes[2].plot(steps_limited, loss_limited, color=color_total, alpha=0.9)
    axes[2].set_title(f'{title_prefix}\nTotal Loss (First 200 Steps)', fontweight='bold', pad=15)
    axes[2].set_xlabel('Steps', fontweight='bold')
    axes[2].set_ylabel('Total Loss', fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    axes[2].set_ylim(unified_ylim)
    
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
    
    # Apply consistent styling to all subplots
    for ax in axes:
        # Make all borders bold and visible
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Set line width for all spines
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['right'].set_linewidth(2.0)
        
        # Make all spines bold and black
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        
        # Format ticks
        ax.tick_params(axis='both', which='major', labelsize=12, width=2.0)
    
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
            steps = list(range(len(limited_data)))
            
            # Plot the curve with smooth line
            lr_label = f"lr={learning_rates[file_idx]}"
            ax.plot(steps, limited_data, color=colors[file_idx], 
                   alpha=0.8, label=lr_label, linewidth=1.2)
            
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
        ax.legend(loc='upper right', fontsize=11)
        
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
        steps = list(range(len(ce_loss)))
        color = colors[i % len(colors)]
        
        # Plot curves
        axes[0].plot(steps, ce_loss, color=color, alpha=0.8, label=label)
        axes[1].plot(steps, entropy_loss, color=color, alpha=0.8, label=label)
        axes[2].plot(steps, loss, color=color, alpha=0.8, label=label)
        axes[3].scatter(ce_loss, entropy_loss, c=color, alpha=0.6, s=15, label=label)
    
    # Set titles and labels
    axes[0].set_title(f'{title}\nCE Loss', fontweight='bold', pad=15)
    axes[0].set_xlabel('Steps', fontweight='bold')
    axes[0].set_ylabel('CE Loss', fontweight='bold')
    axes[0].legend()
    
    axes[1].set_title(f'{title}\nEntropy Loss', fontweight='bold', pad=15)
    axes[1].set_xlabel('Steps', fontweight='bold')
    axes[1].set_ylabel('Entropy Loss', fontweight='bold')
    axes[1].legend()
    
    axes[2].set_title(f'{title}\nTotal Loss', fontweight='bold', pad=15)
    axes[2].set_xlabel('Steps', fontweight='bold')
    axes[2].set_ylabel('Total Loss', fontweight='bold')
    axes[2].legend()
    
    axes[3].set_title(f'{title}\nCE vs Entropy Loss', fontweight='bold', pad=15)
    axes[3].set_xlabel('CE Loss', fontweight='bold')
    axes[3].set_ylabel('Entropy Loss', fontweight='bold')
    axes[3].legend()
    
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
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
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
        "source/distill_qwen/loss_0.0001.json",
        "source/distill_qwen/loss_0.001.json", 
        "source/distill_qwen/loss_0.01.json",
        "source/distill_qwen/loss_0.1.json"
    ]
    
    try:
        # Plot learning rate comparison (line charts)
        plot_lr_comparison(json_files, save_path="lr_comparison_qwen.png", 
                          title="", limit_steps=1000)
        
        # Plot learning rate comparison (scatter plots)
        plot_lr_comparison(json_files, save_path="lr_scatter_comparison_qwen.png", 
                          title="", limit_steps=1000, 
                          scatter_mode=True)
        
        # Single file plotting (optional)
        # plot_loss_curves("source/loss_0.1.json", save_path="loss_curves.png", 
        #                 title_prefix="Training Progress")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error: {e}")
