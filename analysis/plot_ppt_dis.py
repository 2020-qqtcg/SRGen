import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

# ===== 样式配置 =====
rcParams["font.family"] = "DejaVu Sans"
NAVY = "#1F3B70"  # 统一描边/文字色

# 10 根柱：左侧和右侧是“其它…”，中间两根是 even/old（较高且接近）
labels = ["…", "…", "…", "…", "even", "old", "…", "…", "…", "…"]
probs  = [0.030, 0.025, 0.020, 0.015, 0.340, 0.330, 0.120, 0.055, 0.040, 0.025]  # 总和=1

# 每根柱不同颜色（柔和）
COLORS = ["#C9E3FF", "#FAD7E6", "#FFF1B8", "#BFE6D9",
          "#CFEDE2", "#FFC8A2", "#E7E3FF", "#C4F1E0",
          "#FFD6CC", "#D9D7FF"]

# 间距与圆角（可调）
STEP   = 0.65  # 相邻柱中心间距，越小越紧凑
BAR_W  = 0.42  # 柱宽（< STEP）
CORNER = 0.10  # 圆角比例(相对 min(width,height)；0~0.5，小圆角)

# ===== 画布与轴 =====
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
fig.patch.set_alpha(0)      # 透明背景
ax.set_facecolor("none")

# 仅保留底部横轴
for s in ["left", "right", "top"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_linewidth(2)
ax.spines["bottom"].set_color(NAVY)
ax.grid(False)

x = np.arange(len(labels)) * STEP
ax.set_xlim(x[0] - BAR_W, x[-1] + BAR_W)
ax.set_ylim(0, max(probs) * 1.25)
ax.set_xticks(x, labels, color=NAVY, fontsize=12)
ax.tick_params(axis="y", length=0, labelleft=False)

# ===== 圆角长方形柱 =====
def rounded_bar(ax, cx, h, w, fc, ec=NAVY, lw=2.0, corner=CORNER):
    r = corner * min(w, h)  # 小圆角
    patch = FancyBboxPatch(
        (cx - w/2, 0), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, antialiased=True,
        joinstyle="round",
    )
    ax.add_patch(patch)
    return patch

for xi, p, fc, lab in zip(x, probs, COLORS, labels):
    rounded_bar(ax, xi, p, BAR_W, fc)
    # 顶部标注：only even/old 显示百分比，其它显示省略号
    if lab in ("even", "old"):
        ax.text(xi, p + 0.012, f"{int(round(p*100))}%",
                ha="center", va="bottom", fontsize=12, color=NAVY)
    else:
        ax.text(xi, p + 0.010, "…",
                ha="center", va="bottom", fontsize=12, color=NAVY)

plt.tight_layout()
plt.savefig("token_probs_10bars.png", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig("token_probs_10bars.svg", bbox_inches="tight", transparent=True)
plt.show()