# Dual-axis line chart in paper style (like the example figure)
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Paper-like typography ---
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.5,
})

iterations = list(range(0, 11))
times = [40.8, 49.6, 56.5, 60.6, 59.7, 60.5, 60.0, 62.0, 63.9, 63.5, 63.2]
activations = [0.0, 6.0, 6.1, 6.0, 6.0, 6.0, 6.0, 6.0, 6.1, 6.1, 6.1]

baseline = times[0]
pct_increase = [((t - baseline) / baseline) * 100.0 for t in times]

fig = plt.figure(figsize=(8.2, 5.0))
ax = plt.gca()
ax2 = ax.twinx()

# --- Colors (left：暖棕橙；right：深蓝，与示例风格) ---
left_color  = "#c77034"      # warm brown/orange
deep_blue   = "#1f4e79"      # same as your code

# Left y-axis line (white-filled markers, dashed like the example)
line1, = ax.plot(
    iterations, activations,
    linestyle="--", color=left_color, marker="o", markersize=7,
    markerfacecolor="white", markeredgecolor=left_color, markeredgewidth=1.6,
    label="Activations per task", zorder=3
)

# Right y-axis line (deep blue squares, white-filled)
line2, = ax2.plot(
    iterations, pct_increase,
    linestyle="-", color=deep_blue, marker="s", markersize=7,
    markerfacecolor="white", markeredgecolor=deep_blue, markeredgewidth=1.6,
    label="Time increase vs. baseline (%)", zorder=3
)

# --- Axes labels (加粗、与曲线同色的 y 轴标签) ---
ax.set_xlabel("Iterations per activation", fontweight="bold")
ax.set_ylabel("Activations per task",  fontweight="bold", color=left_color)
ax2.set_ylabel("Time increase vs. baseline (%)", fontweight="bold", color=deep_blue)
ax.set_xticks(iterations)

# --- Grid (柔和，仅 y 方向，和示例一致) ---
ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.20)
ax2.grid(False)

# --- Spines: 四周黑色粗边框；右轴/上边显示 ---
for a in (ax, ax2):
    a.spines["top"].set_visible(True)
    a.spines["top"].set_linewidth(2.0)
for side in ["left", "bottom"]:
    ax.spines[side].set_linewidth(2.0)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_linewidth(2.0)

# ticks 粗一些
ax.tick_params(axis="both", width=2.0, length=6)
ax2.tick_params(axis="y",  width=2.0, length=6)

# --- Title (大号、加粗、居中) ---
plt.title("Activations & Time Increase", pad=10)

# --- Legend (白底黑框，像示例里的样式) ---
leg = ax.legend(
    [line1, line2],
    [line1.get_label(), line2.get_label()],
    loc="lower right", frameon=True, fancybox=True, framealpha=1.0,
    facecolor="white", edgecolor="black", fontsize=12
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("image/iters_activations_timepct_v2.png", dpi=300, bbox_inches="tight")
# plt.show()