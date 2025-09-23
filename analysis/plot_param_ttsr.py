import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
})

def plot_two_line_charts(data1, data2, title1="Chart 1", title2="Chart 2", 
                        xlabel1="X-axis", xlabel2="X-axis",
                        ylabel1="Y-axis", ylabel2="Y-axis",
                        save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # 使用蓝/橙两色
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    
    def convert_keys(data):
        sorted_items = sorted(data.items(), key=lambda x: float(x[0]))
        x_vals = [float(k) for k, v in sorted_items]
        y_vals = [v for k, v in sorted_items]
        return x_vals, y_vals
    
    x1, y1 = convert_keys(data1)
    ax1.plot(x1, y1, marker='o', color=color1, markerfacecolor=color1, 
             markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax1.set_title(title1, pad=15)
    ax1.set_xlabel(xlabel1)
    ax1.set_ylabel(ylabel1)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    x2, y2 = convert_keys(data2)
    ax2.plot(x2, y2, marker='s', color=color2, markerfacecolor=color2, 
             markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
    ax2.set_title(title2, pad=15)
    ax2.set_xlabel(xlabel2)
    ax2.set_ylabel(ylabel2)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    all_y_values = y1 + y2
    y_min, y_max = min(all_y_values), max(all_y_values)
    y_margin = max(0.5, (y_max - y_min) * 0.15) if (y_max - y_min) < 5 else (y_max - y_min) * 0.1
    unified_ylim = (y_min - y_margin, y_max + y_margin)
    ax1.set_ylim(unified_ylim)
    ax2.set_ylim(unified_ylim)
    
    if len(x1) <= 8:
        ax1.set_xticks(x1)
    if len(x2) <= 8:
        ax2.set_xticks(x2)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    
    for ax in [ax1, ax2]:
        for side in ['top', 'right', 'left', 'bottom']:
            ax.spines[side].set_visible(True)
    
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    plt.close()

# 示例
if __name__ == "__main__":
    matplotlib.rcParams.update({
        'axes.edgecolor': 'black',   # 轴脊线颜色
        'xtick.color': 'black',      # （可选）x刻度颜色
        'ytick.color': 'black',      # （可选）y刻度颜色
    })
    data1 = {
        "0": 71, "0.1": 70, "0.2": 71, "0.3": 70, "0.4": 70,
        "0.5": 70, "0.6": 70, "0.7": 70, "0.8": 70, "0.9": 70, "1": 70,
    }
    data2 = {
        "0": 71, "0.01": 71, "0.03": 71, "0.05": 72, "0.07": 71,
        "0.09": 70, "0.11": 70, "0.13": 71, "0.15": 71, "0.17": 70,
        "0.19": 71, "0.21": 71, "0.23": 71, "0.25": 71, "0.27": 71,
        "0.29": 70, "0.3": 70
    }
    plot_two_line_charts(
        data1, data2,
        title1="", title2="",
        xlabel1="λ", xlabel2="λ",
        ylabel1="Accuracy(%)", ylabel2="Accuracy(%)",
        save_path="param_ttsr.png"
    )