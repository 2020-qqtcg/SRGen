import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----- Paper-like style -----
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.linewidth": 2.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.2,
})

def _prep_xy(d):
    """Sort by numeric key; return x(float), y(float), tick_labels(str)."""
    items = sorted(d.items(), key=lambda kv: float(kv[0]))
    x = np.array([float(k) for k, _ in items], dtype=float)
    y = np.array([float(v) for _, v in items], dtype=float)
    labels = [k for k, _ in items]  # keep original text as tick labels
    return x, y, labels

def _gaussian_kernel(size=21, sigma=None):
    size = int(size)
    if size % 2 == 0:
        size += 1
    if sigma is None:
        sigma = size / 6.0
    t = np.arange(size) - size // 2
    k = np.exp(-(t**2) / (2 * sigma**2))
    return k / k.sum()

def _smooth_curve(x, y, dense=300):
    """Interpolate to dense grid then gaussian smooth (no SciPy needed)."""
    if len(x) < 2:
        return x, y
    xd = np.linspace(x.min(), x.max(), dense)
    yd = np.interp(xd, x, y)
    size = max(15, (dense // 20) | 1)
    k = _gaussian_kernel(size=size, sigma=size/6.0)
    # Use edge padding to avoid zero-padding dips at the boundaries
    pad = len(k) // 2
    yd_pad = np.pad(yd, (pad, pad), mode="edge")
    ys_full = np.convolve(yd_pad, k, mode="same")
    ys = ys_full[pad:-pad]
    return xd, ys

def plot_three_dicts_side_by_side(
    d1, d2, d3,
    titles=("Panel 1", "Panel 2", "Panel 3"),
    xlabels=[],
    ylabel="Accuracy (%)",
    save_path=None,
    rotate=30
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)

    datasets = [d1, d2, d3]
    colors = ["#1f4e79", "#2d9966", "#c77034"]  # deep blue, green, warm brown

    for ax, data, title, color, xlabel in zip(axes, datasets, titles, colors, xlabels):
        x, y, labels = _prep_xy(data)

        # connect original points so markers lie on the line
        ax.plot(x, y, color=color, linewidth=2.4, zorder=2)
        ax.plot(x, y, linestyle="none", marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.5, zorder=3)

        # Auto x-range per panel with 5% padding
        xmin, xmax = float(x.min()), float(x.max())
        if xmax > xmin:
            xpad = (xmax - xmin) * 0.05
        else:
            xpad = max(abs(xmin) * 0.05, 0.05)
        ax.set_xlim(xmin - xpad, xmax + xpad)

        # axes labels & title
        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")

        # y-limit with small margin (based on original data)
        y_min, y_max = float(np.min(y)), float(np.max(y))
        margin = max(1.0, (y_max - y_min) * 0.12)
        ax.set_ylim(y_min - margin, y_max + margin)

        # ticks: show all if not too many, else downsample to ≤8
        MAX_XTICKS = 8
        if len(x) <= MAX_XTICKS:
            xticks, xticklabels = x, labels
        else:
            step = int(np.ceil(len(x) / MAX_XTICKS))
            idx = np.arange(0, len(x), step)
            xticks, xticklabels = x[idx], [labels[i] for i in idx]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=rotate, ha="right")

        # grid & frame (reference style)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.9, alpha=0.35)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.20)
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(2.0)
            ax.spines[side].set_color("black")
        ax.tick_params(direction="out", width=2.0, length=5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_single_line_chart(x, y, title=None, xlabel=None, ylabel=None, save_path=None):
    """Plot a single line chart with smoothing."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x_dense = np.linspace(min(x), max(x), 500)
    y_dense = np.interp(x_dense, x, y)

    size = max(15, (len(x_dense) // 20) | 1)
    sigma = size / 6.0
    kernel = _gaussian_kernel(size=size, sigma=sigma)
    # Edge padding to prevent boundary dips (no zeros mixed in)
    pad = len(kernel) // 2
    y_pad = np.pad(y_dense, (pad, pad), mode='edge')
    y_full = np.convolve(y_pad, kernel, mode='same')
    y_smooth = y_full[pad:-pad]

    ax.plot(x_dense, y_smooth, color='blue', linewidth=2)
    ax.plot(x, y, 'o', color='red', markersize=5)

    if title:
        ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to: {save_path}")
    else:
        plt.show()
    plt.close()

# ===== Example =====
if __name__ == "__main__":
    data = {
        "0": 71, "0.1": 70, "0.2": 71, "0.3": 70, "0.4": 70, "0.5": 70,
        "0.6": 70, "0.7": 70, "0.8": 70, "0.9": 70, "1": 70,
        "0.01": 71, "0.03": 71, "0.05": 72, "0.07": 71, "0.09": 70,
        "0.11": 70, "0.13": 71, "0.15": 71, "0.17": 70, "0.19": 71,
        "0.21": 71, "0.23": 71, "0.25": 71, "0.27": 71, "0.29": 70
    }

    data2 = {
        "5": 70,
        "10": 72,
        "15": 72,
        "20": 72,
        "25": 72,
        "30": 74,
        "35": 75,
        "40": 73,
        "45": 72,
        "50": 72,
        "55": 72,
    }

    data3 = {
        "0.0": 73,
        "0.5": 73,
        "1.0": 72,
        "1.5": 74,
        "2.0": 73,
        "2.5": 72,
        "3.0": 72,
        "3.4": 74,
        "3.5": 74,
        "3.6": 74,
        "4.0": 72,
        "4.5": 72,
        "5.0": 72,
        "5.5": 72,
        "6.0": 72
    }

    # data2 = {
    #     "5": 71,
    #     "10": 72,
    #     "15": 73,
    #     "20": 73,
    #     "25": 72,
    #     "30": 74,
    #     "35": 74,
    #     "40": 75,
    #     "45": 71,
    #     "50": 72,
    # }

    # data3 = {
    #     "1.0": 75,
    #     "1.5": 73,
    #     "2.0": 73,
    #     "2.5": 72,
    #     "3.0": 73,
    #     "3.5": 74,
    #     "4.0": 71,
    #     "4.5": 74,
    #     "5.0": 74,
    #     "5.5": 72,
    #     "6.0": 73
    # }

    xlabels = ["λ", "N", "K"]
    # demo: 用同一份数据画 3 个面板（实际用你的三份 dict）
    plot_three_dicts_side_by_side(
        data, data2, data3,
        titles=("MATH500(First100)", "MATH500(First100)", "MATH500(First100)"),
        xlabels=xlabels, ylabel="Accuracy (%)",
        save_path="image/param_ttsr.png"
    )