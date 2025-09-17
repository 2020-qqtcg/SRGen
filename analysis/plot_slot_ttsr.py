import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_bar_chart(config):
    """
    绘制分组柱状图，支持多个数据集的对比
    
    Args:
        config (dict): 配置字典，包含以下键值：
            - datasets: 数据集配置列表，每个元素包含：
                - name: 数据集名称
                - values: 四种方法的数值列表 [Base, SLOT, TTSR, SLOT+TTSR]
            - method_names: 方法名称列表 (默认为 ['Base', 'SLOT', 'TTSR', 'SLOT+TTSR'])
            - ylabel: y轴标签
            - colors: 颜色列表 (可选)
            - figsize: 图片大小 (可选)
    """
    # 提取配置参数
    datasets = config.get('datasets', [])
    method_names = config.get('method_names', ['Base', 'SLOT', 'TTSR', 'SLOT+TTSR'])
    ylabel = config.get('ylabel', 'Accuracy (%)')
    colors = config.get('colors', ['#CD853F', '#4682B4', '#2ECC71', '#E74C3C'])  # Base, SLOT, TTSR, SLOT+TTSR
    figsize = config.get('figsize', (12, 6))
    flexible_scale = config.get('flexible_scale', True)  # 是否使用灵活刻度
    
    if not datasets:
        raise ValueError("必须提供datasets配置")
    
    # 设置图表参数
    n_datasets = len(datasets)
    n_methods = len(method_names)
    
    # 创建图表 - 不共享y轴，让每个子图有独立的刻度
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=False)
    if n_datasets == 1:
        axes = [axes]
    
    # 为每个数据集创建子图
    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        dataset_name = dataset['name']
        values = dataset['values']
        
        if len(values) != 4:  # 每个数据集应该有4个值：Base, SLOT, TTSR, SLOT+TTSR
            raise ValueError(f"数据集 {dataset_name} 的值数量必须为4 (Base, SLOT, TTSR, SLOT+TTSR)")
        
        # 创建柱状图 - 四种方法
        x_pos = np.arange(len(method_names)) * 0.5  # 减小间距，原来是1.0的间距
        width = 0.3  # 将宽度从0.6减小到0.3（减半）
        
        # 先设置网格线，确保它在柱子下方
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=1)
        
        # 绘制柱状图，设置zorder确保在网格线上方
        bars = ax.bar(x_pos, values, width, color=colors[:len(values)], 
                     alpha=0.8, edgecolor='none', zorder=3)
        
        # 设置子图标题
        ax.set_title(dataset_name, fontsize=14, pad=5)
        
        # 设置x轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, fontsize=11)
        
        # 不在柱子上显示数值（已移除）
        
        # 根据配置决定是否使用灵活刻度
        if flexible_scale:
            # 设置y轴范围以最大化显示差异
            min_val = min(values)
            max_val = max(values)
            
            # 计算合适的y轴范围，给数据留出一些空间
            range_val = max_val - min_val
            if range_val < 1:  # 如果差异很小，扩大显示范围
                padding = max(0.5, range_val * 0.3)
            else:
                padding = range_val * 0.15
            
            y_min = max(0, min_val - padding)  # 确保不低于0
            y_max = max_val + padding
            
            ax.set_ylim(y_min, y_max)
            
            # 设置y轴刻度，显示更多细节
            num_ticks = 6
            y_ticks = np.linspace(y_min, y_max, num_ticks)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
            
            # 每个子图都显示y轴标签（因为刻度不同了）
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            # 使用默认刻度，只在最左边显示y轴标签
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=12)
        
        # 设置刻度参数
        ax.tick_params(axis='both', labelsize=10)
        
        # 加粗整个图表的外围边框
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    # 创建图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[j], label=method_names[j]) 
                      for j in range(len(method_names))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(method_names), 
              bbox_to_anchor=(0.5, -0.08), fontsize=11)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

def plot_bar_chart(config):
    """
    绘制垂直柱状图 (保持向后兼容)
    """
    # 提取配置参数
    title = config.get('title', '柱状图')
    column_names = config.get('column_names', ['列1', '列2', '列3'])
    values = config.get('values', [0, 0, 0])
    xlabel = config.get('xlabel', '')
    ylabel = config.get('ylabel', '数值')
    colors = config.get('colors', ['#2E86C1', '#28B463', '#F39C12'])
    figsize = config.get('figsize', (10, 6))
    
    # 检查数据长度
    if len(column_names) != 3 or len(values) != 3:
        raise ValueError("column_names 和 values 必须都包含3个元素")
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 创建柱状图
    bars = plt.bar(column_names, values, color=colors, alpha=0.8, edgecolor='none')
    
    # 设置标题和标签
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 在每个柱子上显示数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 美化图表
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # 加粗整个图表的外围边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 调整布局
    plt.tight_layout()
    
    return plt.gcf()

def save_plot(config, filename=None, use_grouped=False):
    """
    绘制并保存柱状图
    
    Args:
        config (dict): 配置字典
        filename (str): 保存文件名 (可选)
        use_grouped (bool): 是否使用分组柱状图
    """
    if use_grouped:
        fig = plot_grouped_bar_chart(config)
    else:
        fig = plot_bar_chart(config)
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
    
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 分组柱状图配置（三种方法对比，展示不同刻度效果）
    grouped_config = {
        'datasets': [
            {
                'name': 'AMC',
                'values': [35.0, 35.0, 37.0, 38.0]  # Base, SLOT, TTSR, SLOT+TTSR
            },
            {
                'name': 'MATH500',
                'values': [63.8, 64.2, 69.4, 70.6]  # Base, SLOT, TTSR, SLOT+TTSR
            },
            {
                'name': 'AIME24',
                'values': [13.3, 20.0, 20.0, 30.0]  # Base, SLOT, TTSR, SLOT+TTSR
            }
        ],
        'method_names': ['Base', 'SLOT', 'TTSR', 'S+T'],
        'ylabel': 'Accuracy (%)',
        'colors': ['#AB6F26', '#1B5588', '#1A5B54', '#680D30'],  # Base, SLOT, TTSR, SLOT+TTSR  
        'figsize': (10, 3.5),
        'flexible_scale': True  # 使用灵活刻度来最大化显示差异
    }
    
    # 绘制分组图表
    save_plot(grouped_config, 'grouped_comparison.png', use_grouped=True)
    
    # 原始单个图表配置（保持向后兼容）
    single_config = {
        'title': 'Qwen2.5-Math-7B on MATH500',
        'column_names': ['', 'SLOT', 'SLOT + TTSR'],
        'values': [63.8, 64.2, 70.6],
        'xlabel': '',
        'ylabel': 'accuracy',
        'colors': ['#3498DB', '#E74C3C', '#2ECC71'],
        'figsize': (10, 6)
    }
    
    # 绘制单个图表
    # save_plot(single_config, 'single_comparison.png')

def create_comparison_chart(datasets_data, filename='comparison.png', flexible_scale=True):
    """
    快速创建四种方法对比图表的便捷函数
    
    Args:
        datasets_data (dict): 数据集数据，格式为 {'数据集名': [Base值, SLOT值, TTSR值, SLOT+TTSR值]}
        filename (str): 保存文件名
        flexible_scale (bool): 是否使用灵活刻度以最大化显示差异
    
    Example:
        data = {
            'GPQA': [36.4, 32.3, 35.8, 37.4],
            'MATH500': [63.8, 64.2, 68.4, 70.6],
            'AIME24': [16.7, 20.0, 26.7, 30.0]
        }
        create_comparison_chart(data, 'my_results.png', flexible_scale=True)
    """
    config = {
        'datasets': [{'name': name, 'values': values} for name, values in datasets_data.items()],
        'method_names': ['Base', 'SLOT', 'TTSR', 'SLOT+TTSR'],
        'ylabel': 'Accuracy (%)',
        'colors': ['#AB6F26', '#1B5588', '#2D9966', '#680D30'],
        'figsize': (5 * len(datasets_data), 5),
        'flexible_scale': flexible_scale
    }
    
    save_plot(config, filename, use_grouped=True)
