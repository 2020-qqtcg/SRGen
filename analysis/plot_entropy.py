# -*- coding: utf-8 -*-
"""
画 4 个 jsonl 的熵曲线（2 行 × 2 列）
- x 轴：step（第 i 条记录即 step=i）
- y 轴：original_entropy
- 风格：浅色原始曲线 + 滚动均值 + 置信带（±1 std）
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 全局字体与标题样式（仿论文风格）
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'figure.titlesize': 20,
})

# ======= 1) 配置你的四个文件路径 =======
files = [
    "source/entropy/tem_qwen_math.jsonl",
    "source/entropy/tem_distill_llama.jsonl",
    "source/entropy/tem_distill_qwen.jsonl",
    "source/entropy/tem_qwen3_32b.jsonl",
]

title = [
    "Qwen2.5-Math-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Qwen3-32B"
]

SMOOTH_WINDOW = 51  # 平滑窗口（可调）
LINE_COLOR = "tab:orange"

def load_entropy_series(path):
    entropies = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "original_entropy" in obj:
                    v = obj["original_entropy"]
                    if v is not None and np.isfinite(float(v)):
                        entropies.append(float(v))
            except Exception:
                continue
    steps = np.arange(1, len(entropies) + 1)
    return steps, np.array(entropies, dtype=float)

def rolling_mean_std(y, window):
    s = pd.Series(y, dtype=float)
    mean = s.rolling(window, center=True).mean().to_numpy()
    std = s.rolling(window, center=True).std(ddof=0).to_numpy()
    return mean, std

def nice_axes_style(ax):
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)  # 网格在下
    # ——确保四周都有边框——
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_edgecolor("black")
    ax.tick_params(direction="out", length=4, width=0.8, colors="black")

def plot_one(ax, steps, y, title, window=51):
    ax.plot(steps, y, linewidth=1.0, alpha=0.35, color=LINE_COLOR)
    mean, std = rolling_mean_std(y, window=min(window, max(3, len(y)//20*2+1)))
    ax.plot(steps, mean, linewidth=2.5, color=LINE_COLOR)
    ax.fill_between(steps, mean - std, mean + std, where=~np.isnan(mean),
                    alpha=0.15, color=LINE_COLOR, linewidth=0)

    ax.set_xlabel("Inference Step", fontfamily="Times New Roman", fontweight="bold")
    ax.set_ylabel("Entropy", fontfamily="Times New Roman", fontweight="bold", color=LINE_COLOR)
    ax.set_title(title, fontfamily="Times New Roman", fontweight="bold", fontsize=14, loc="center", pad=6)
    nice_axes_style(ax)

    if len(steps) > 0:
        x_last, y_last = steps[-1], y[-1]
        ann = f"Step: {x_last}, Entropy: {y_last:.4f}"
        x_text = steps.max() * 0.68
        y_text = (np.nanmax(mean) if np.isfinite(np.nanmax(mean)) else y.max()) * 0.95
        ax.annotate(
            ann, xy=(x_last, y_last), xytext=(x_text, y_text),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red"),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            fontsize=9
        )

def main(files):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for i, path in enumerate(files[:4]):
        if not os.path.exists(path):
            axes[i].text(0.5, 0.5, f"File not found:\n{path}",
                         ha="center", va="center", fontsize=11)
            axes[i].set_axis_off()
            continue
        steps, y = load_entropy_series(path)
        plot_one(axes[i], steps, y, title[i], window=SMOOTH_WINDOW)

    for j in range(len(files), 4):
        axes[j].set_axis_off()

    # Centered figure title to avoid overlap with the canvas edges
    fig.suptitle("", fontsize=20, fontweight="bold", fontfamily="Times New Roman")
    plt.savefig("image/model_entropy.png", dpi=600, bbox_inches="tight")

if __name__ == "__main__":
    main(files)