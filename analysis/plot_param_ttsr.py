import matplotlib.pyplot as plt
import numpy as np
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
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
})

def plot_two_line_charts(data1, data2, title1="Chart 1", title2="Chart 2", 
                        xlabel1="X-axis", xlabel2="X-axis",
                        ylabel1="Y-axis", ylabel2="Y-axis",
                        save_path=None):
    """
    Plot two line charts side by side from dictionary data with academic styling.
    
    Args:
        data1 (dict): First dictionary with x-values as keys and y-values as values
        data2 (dict): Second dictionary with x-values as keys and y-values as values
        title1 (str): Title for the first chart
        title2 (str): Title for the second chart
        xlabel1 (str): X-axis label for the first chart
        xlabel2 (str): X-axis label for the second chart
        ylabel1 (str): Y-axis label for the first chart
        ylabel2 (str): Y-axis label for the second chart
        save_path (str, optional): Path to save the figure. If None, display the plot.
    """
    # Create a figure with two subplots side by side with more spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    
    # Academic color scheme
    color1 = '#2E86AB'  # Professional blue
    color2 = '#A23B72'  # Professional magenta
    
    # Convert string keys to float for proper sorting
    def convert_keys(data):
        sorted_items = sorted(data.items(), key=lambda x: float(x[0]))
        x_vals = [float(k) for k, v in sorted_items]
        y_vals = [v for k, v in sorted_items]
        return x_vals, y_vals
    
    # Plot first chart
    x1, y1 = convert_keys(data1)
    ax1.plot(x1, y1, marker='o', color=color1, markerfacecolor=color1, 
             markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax1.set_title(title1, fontweight='bold', pad=15)
    ax1.set_xlabel(xlabel1, fontweight='bold')
    ax1.set_ylabel(ylabel1, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Plot second chart
    x2, y2 = convert_keys(data2)
    ax2.plot(x2, y2, marker='s', color=color2, markerfacecolor=color2, 
             markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax2.set_title(title2, fontweight='bold', pad=15)
    ax2.set_xlabel(xlabel2, fontweight='bold')
    ax2.set_ylabel(ylabel2, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Calculate unified y-axis range for both charts
    all_y_values = y1 + y2
    y_min, y_max = min(all_y_values), max(all_y_values)
    if y_max - y_min < 5:  # For small ranges, use smaller margins
        y_margin = max(0.5, (y_max - y_min) * 0.15)
    else:
        y_margin = (y_max - y_min) * 0.1
    
    # Apply the same y-axis range to both charts
    unified_ylim = (y_min - y_margin, y_max + y_margin)
    ax1.set_ylim(unified_ylim)
    ax2.set_ylim(unified_ylim)
    
    # Set x-axis ticks for better readability
    if len(x1) <= 8:
        ax1.set_xticks(x1)
    if len(x2) <= 8:
        ax2.set_xticks(x2)
    ax1.tick_params(axis='both', which='major', labelsize=13, width=2.0)
    ax2.tick_params(axis='both', which='major', labelsize=13, width=2.0)
    
    # Make all borders bold and visible for complete frame
    for ax in [ax1, ax2]:
        # Make all spines visible and bold
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
    
    # Adjust layout with more spacing between subplots
    plt.subplots_adjust(wspace=0.4)  # Increase horizontal spacing between subplots
    plt.tight_layout(pad=2.0)
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

# Example usage:
if __name__ == "__main__":
    # Example data
    data1 = {
        "0": 71,
        "0.1": 70,
        "0.2": 71,
        "0.3": 70,
        "0.4": 70,
        "0.5": 70,
        "0.6": 70,
        "0.7": 70,
        "0.8": 70,
        "0.9": 70,
        "1": 70,
    }
    
    data2 = {
        "0": 71,
        "0.01": 71,
        "0.03": 71,
        "0.05": 72,
        "0.07": 71,
        "0.09": 70,
        "0.11": 70,
        "0.13": 71,
        "0.15": 71,
        "0.17": 70,
        "0.19": 71,
        "0.21": 71,
        "0.23": 71,
        "0.25": 71,
        "0.27": 71,
        "0.29": 70,
        "0.3": 70
    }
    
    # Plot with example data
    plot_two_line_charts(
        data1, data2,
        title1="MATH500",
        title2="MATH500",
        xlabel1="λ",
        xlabel2="λ",
        ylabel1="Acc(%)",
        ylabel2="Acc(%)",
        save_path="param_ttsr.png"
    )
