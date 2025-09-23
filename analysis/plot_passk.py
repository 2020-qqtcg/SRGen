import numpy as np
import matplotlib.pyplot as plt

# ----- 数据 -----
k = np.arange(1, 49)

# Base
maj_base = np.array([
    0.36, 0.24, 0.34, 0.32, 0.34, 0.32, 0.35, 0.34,
    0.36, 0.34, 0.36, 0.36, 0.37, 0.36, 0.37, 0.36,
    0.37, 0.35, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37,
    0.39, 0.39, 0.39, 0.38, 0.39, 0.37, 0.38, 0.37,
    0.37, 0.36, 0.37, 0.36, 0.36, 0.36, 0.37, 0.36,
    0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36
]) * 100

pass_base = np.array([
    0.36, 0.40, 0.47, 0.48, 0.49, 0.49, 0.51, 0.53,
    0.54, 0.54, 0.55, 0.55, 0.55, 0.56, 0.56, 0.56,
    0.56, 0.56, 0.57, 0.57, 0.59, 0.59, 0.60, 0.60,
    0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.61,
    0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61,
    0.61, 0.61, 0.61, 0.62, 0.62, 0.62, 0.62, 0.62
]) * 100

# SRGen
maj_srgen = np.array([
    0.39, 0.34, 0.41, 0.37, 0.39, 0.36, 0.39, 0.36,
    0.41, 0.38, 0.41, 0.41, 0.44, 0.41, 0.43, 0.41,
    0.42, 0.39, 0.40, 0.39, 0.40, 0.40, 0.40, 0.40,
    0.41, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40, 0.40,
    0.40, 0.40, 0.40, 0.40, 0.41, 0.41, 0.41, 0.41,
    0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41, 0.41
]) * 100

pass_srgen = np.array([
    0.39, 0.51, 0.52, 0.53, 0.55, 0.56, 0.56, 0.56,
    0.57, 0.57, 0.57, 0.57, 0.57, 0.58, 0.58, 0.58,
    0.58, 0.59, 0.60, 0.61, 0.61, 0.61, 0.61, 0.61,
    0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61,
    0.61, 0.61, 0.62, 0.62, 0.62, 0.62, 0.62, 0.63,
    0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63
]) * 100

# 仅保留奇数 k（1,3,...,31）
idx = np.arange(0, 47, 2)
k_odd = k[idx]
maj_b, maj_s = maj_base[idx], maj_srgen[idx]
pas_b, pas_s = pass_base[idx], pass_srgen[idx]

# ----- 画图风格（与示例图一致：网格点线、实线、蓝/橙）-----
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 160,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.8
})

fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8), constrained_layout=True)

for ax in axes:
    ax.set_xlim(1, 47)
    ax.set_xticks(np.arange(1, 48, 4))  # 横坐标刻度间隔为 4
    ax.set_xlabel("k")
    ax.grid(True, which="both")
    ax.tick_params(direction="out")

# 左：Cons@k（maj@k）
axes[0].plot(k_odd, maj_b, marker="o", linestyle="-", linewidth=1.5, color="#1f77b4", markersize=3, label="Base")
axes[0].plot(k_odd, maj_s, marker="s", linestyle="-", linewidth=1.5, color="#ff7f0e", markersize=3, label="SRGen")
axes[0].set_title("Cons@k")
axes[0].set_ylabel("Accuracy(%)")
axes[0].legend(frameon=True)

# 右：Pass@k（pass@k）
axes[1].plot(k_odd, pas_b, marker="o", linestyle="-", linewidth=1.5, color="#1f77b4", markersize=3, label="Base")
axes[1].plot(k_odd, pas_s, marker="s", linestyle="-", linewidth=1.5, color="#ff7f0e", markersize=3, label="SRGen")
axes[1].set_title("Pass@k")
axes[1].set_ylabel("Accuracy(%)")
axes[1].legend(frameon=True, loc='lower right')

# 共享 y 轴范围（留少量边距）
ymin = min(maj_b.min(), maj_s.min(), pas_b.min(), pas_s.min())
ymax = max(maj_b.max(), maj_s.max(), pas_b.max(), pas_s.max())
for ax in axes:
    ax.set_ylim(ymin - 2, ymax + 2)

# 保存 PNG（论文用高分辨率）
plt.savefig("cons_pass_k_iclr.png", dpi=400, bbox_inches="tight")
plt.show()