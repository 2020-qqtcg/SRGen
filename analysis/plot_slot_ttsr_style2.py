# -*- coding: utf-8 -*-
"""
分组柱状图：第二张图的彩色描边 + 彩色 hatch 风格
更新：
- y 轴使用“整数整齐刻度”（1/2/5×10^k），边界对齐到整数
- 默认不显示柱顶数值
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# ---------- 整齐的“整数局部刻度” ----------
def _int_local_ticks(vmin, vmax, n=6):
    """
    整数局部刻度：
    - 在数据两侧加 padding
    - 步长从 {1,2,5}×10^k 中选，但**不小于 1**
    - 始终生成恰好 n 个刻度
    """
    span = max(vmax - vmin, 1e-9)
    pad = max(0.5, 0.15 * span)
    ymin = vmin - pad
    ymax = vmax + pad

    # 目标步长（粗估）
    rough = (ymax - ymin) / (n - 1)

    # 选 1/2/5×10^k，并强制 >= 1
    mag = 10 ** np.floor(np.log10(max(2.0, rough)))
    norm = rough / mag
    if norm <= 1.5:
        step = 1 * mag
    elif norm <= 3:
        step = 2 * mag
    else:
        step = 5 * mag
    step = max(1.0, step)

    # 对齐边界到步长整数倍
    y0 = step * np.floor(ymin / step)
    y1 = step * np.ceil(ymax / step)

    # 确保正好 n 个刻度
    k = int(round((y1 - y0) / step))
    if k < (n - 1):
        y1 = y0 + step * (n - 1)
    elif k > (n - 1):
        y0 = y1 - step * (n - 1)

    ticks = np.linspace(y0, y1, n)
    labels = [f"{int(t)}" for t in ticks]  # 统一整数显示
    return float(y0), float(y1), ticks, labels


# ---------- 分组柱状图（第二图风格） ----------
def plot_grouped_bar_chart(config):
    """
    额外参数：
      - style: 'second_color'（默认：白底+彩色描边+同色 hatch）或 'color'（实心彩色）
      - palette: 颜色列表（默认：橙/紫/绿/酒红）
      - hatches: 每个方法的纹理（默认：['----','////','\\\\\\\\','xx']）
      - show_values: 柱顶是否标注数值（默认 False）
    """
    datasets = config.get('datasets', [])
    method_names = config.get('method_names', ['Base', 'SLOT', 'TTSR', 'S+T'])
    ylabel = config.get('ylabel', 'Accuracy (%)')
    figsize = config.get('figsize', (12, 4))
    flexible_scale = config.get('flexible_scale', True)

    style = config.get('style', 'second_color')
    palette = config.get('palette', ['#F4A261', '#B57EDC', '#6CBF6D', '#8E355A'])
    hatches = config.get('hatches', ['----', '////', '\\\\\\\\', 'xx'])
    show_values = config.get('show_values', False)  # 默认不显示

    if not datasets:
        raise ValueError("必须提供datasets配置")

    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize, sharey=False)
    if n_datasets == 1:
        axes = [axes]

    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        ds_name = dataset['name']
        values = dataset['values']
        if len(values) != len(method_names):
            raise ValueError(f"数据集 {ds_name} 的值数量与方法数不一致")

        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', linewidth=1, alpha=0.55, color='0.6')

        x_pos = np.arange(len(method_names)) * 0.85
        width = 0.55
        bars = []
        for j, v in enumerate(values):
            c = palette[j % len(palette)]
            if style == 'second_color':
                bar = ax.bar(
                    x_pos[j], v, width,
                    facecolor='white', edgecolor=c, linewidth=1.2,
                    hatch=hatches[j % len(hatches)], zorder=3
                )
            else:
                bar = ax.bar(
                    x_pos[j], v, width,
                    color=c, edgecolor='black', linewidth=1.2, zorder=3
                )
            bars.append(bar[0])

        ax.set_title(ds_name, fontsize=12, pad=6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, fontsize=12)

        if flexible_scale:
            y_min, y_max, ticks, labels = _int_local_ticks(min(values), max(values), n=6)
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.set_ylabel(ylabel, fontsize=13)
        else:
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=13)

        if show_values:
            gap = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            for rect, v in zip(bars, values):
                ax.text(rect.get_x() + rect.get_width()/2., v + gap, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=12, color='black')

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')

    legend_handles = [
        Patch(facecolor='white', edgecolor=palette[j], linewidth=2.0,
              hatch=hatches[j % len(hatches)], label=method_names[j])
        for j in range(len(method_names))
    ]
    fig.legend(handles=legend_handles, labels=method_names,
               loc='lower center', ncol=len(method_names),
               bbox_to_anchor=(0.5, -0.02), fontsize=12,
               frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    return fig


# ---------- 单图版本（同风格） ----------
def plot_bar_chart(config):
    title = config.get('title', '柱状图')
    column_names = config.get('column_names', ['列1', '列2', '列3'])
    values = config.get('values', [0, 0, 0])
    xlabel = config.get('xlabel', '')
    ylabel = config.get('ylabel', '数值')
    figsize = config.get('figsize', (10, 6))

    style = config.get('style', 'second_color')
    palette = config.get('palette', ['#F4A261', '#B57EDC', '#6CBF6D'])
    hatches = config.get('hatches', ['----', '////', '\\\\\\\\'])
    show_values = config.get('show_values', False)  # 默认不显示

    if len(column_names) != len(values):
        raise ValueError("column_names 和 values 长度必须一致")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', linewidth=1, alpha=0.55, color='0.6')

    x_pos = np.arange(len(values)); width = 0.6
    bars = []
    for j, v in enumerate(values):
        c = palette[j % len(palette)]
        if style == 'second_color':
            bar = ax.bar(x_pos[j], v, width,
                         facecolor='white', edgecolor=c, linewidth=2.0,
                         hatch=hatches[j % len(hatches)], zorder=3)
        else:
            bar = ax.bar(x_pos[j], v, width,
                         color=c, edgecolor='black', linewidth=1.2, zorder=3)
        bars.append(bar[0])

    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(column_names, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    y_min, y_max, ticks, labels = _int_local_ticks(min(values), max(values), n=6)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    if show_values:
        gap = 0.02 * (y_max - y_min)
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width()/2., v + gap, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=12, color='black')

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')

    legend_handles = [
        Patch(facecolor='white', edgecolor=palette[j], linewidth=2.0,
              hatch=hatches[j % len(hatches)], label=column_names[j])
        for j in range(len(column_names))
    ]
    ax.legend(handles=legend_handles, loc='lower center', ncol=len(column_names),
              bbox_to_anchor=(0.5, -0.18), frameon=True, edgecolor='black', fontsize=12)

    plt.tight_layout()
    return fig


# ---------- 保存与便捷函数 ----------
def save_plot(config, filename=None, use_grouped=False):
    fig = plot_grouped_bar_chart(config) if use_grouped else plot_bar_chart(config)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: {filename}")
    plt.show()


def create_comparison_chart(datasets_data, filename='comparison.png', flexible_scale=True):
    config = {
        'datasets': [{'name': name, 'values': values} for name, values in datasets_data.items()],
        'method_names': ['Base', 'SLOT', 'TTSR', 'S+T'],
        'ylabel': 'Accuracy (%)',
        'palette': ['#F4A261', '#B57EDC', '#6CBF6D', '#8E355A'],
        'hatches': ['----', '////', '\\\\\\\\', 'xx'],
        'figsize': (5 * len(datasets_data), 4),
        'flexible_scale': flexible_scale,
        'style': 'second_color',
        'show_values': True,  # 不显示柱顶数值
    }
    save_plot(config, filename, use_grouped=True)


# ---------- 示例运行 ----------
if __name__ == "__main__":
    grouped_config = {
        'datasets': [
            {'name': 'AMC',     'values': [35.0, 35.0, 37.0, 38.0]},
            {'name': 'MATH500', 'values': [63.8, 64.2, 69.4, 70.6]},
            {'name': 'AIME24',  'values': [13.3, 20.0, 20.0, 30.0]},
        ],
        'method_names': ['Base', 'SLOT', 'SRGen', 'S+S'],
        'ylabel': 'Accuracy (%)',
        'figsize': (10, 3.6),
        'flexible_scale': True,
        'style': 'second_color',
        'palette': ['#F4A261', '#B57EDC', '#6CBF6D', '#C9086F'],
        'hatches': ['----', '////', '\\\\\\\\', 'xxx'],
        'show_values': True,  # 不显示柱顶数值
    }
    save_plot(grouped_config, filename='grouped_comparison_second_style.png', use_grouped=True)