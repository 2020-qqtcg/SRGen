import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Paper style ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.0,
})

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

# 仅保留奇数 k（1,3,...,47）
idx = np.arange(0, 47, 2)
k_odd = k[idx]
maj_b, maj_s = maj_base[idx], maj_srgen[idx]
pas_b, pas_s = pass_base[idx], pass_srgen[idx]

# ---------- colors ----------
deep_blue = "#1f4e79"   # Base
warm_brown = "#c77034"  # SRGen

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

for ax in axes:
    ax.set_xlim(1, 47)
    ax.set_xticks(np.arange(1, 48, 4))
    ax.set_xlabel("k", fontweight="bold")

    # grid (soft, dotted)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.35)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.20)

    # thick black frame
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.0)
        ax.spines[side].set_color("black")
    ax.tick_params(direction="out", width=2.0, length=5)

# 左：Cons@k（maj@k）
axes[0].plot(k_odd, maj_b, color=deep_blue,  linestyle="-", linewidth=2.4,
             label="Base",  zorder=3)
axes[0].plot(k_odd, maj_s, color=warm_brown, linestyle="-", linewidth=2.4,
             label="SRGen", zorder=3)
axes[0].set_title("Cons@k", pad=6)
axes[0].set_ylabel("Accuracy (%)", fontweight="bold")
axes[0].legend(
    frameon=True, fancybox=True, framealpha=1.0,
    facecolor="white", edgecolor="black", loc="upper right"
)

# 右：Pass@k（pass@k）
axes[1].plot(k_odd, pas_b, color=deep_blue,  linestyle="-", linewidth=2.4,
             label="Base",  zorder=3)
axes[1].plot(k_odd, pas_s, color=warm_brown, linestyle="-", linewidth=2.4,
             label="SRGen", zorder=3)
axes[1].set_title("Pass@k", pad=6)
axes[1].set_ylabel("Accuracy (%)", fontweight="bold")
axes[1].legend(
    frameon=True, fancybox=True, framealpha=1.0,
    facecolor="white", edgecolor="black", loc="lower right"
)

# 共享 y 轴范围（留少量边距）
ymin = min(maj_b.min(), maj_s.min(), pas_b.min(), pas_s.min())
ymax = max(maj_b.max(), maj_s.max(), pas_b.max(), pas_s.max())
for ax in axes:
    margin = 1.0
    ax.set_ylim(ymin - margin, ymax + margin)

# 主标题（可选）
fig.suptitle("", fontsize=20, fontweight="bold")

plt.savefig("image/cons_pass_k_iclr.png", dpi=400, bbox_inches="tight")
plt.show()