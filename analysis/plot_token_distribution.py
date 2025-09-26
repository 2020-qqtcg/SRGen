import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np

# —— Paper 风格（与前面一致）——
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 2.0,
    "lines.linewidth": 2.0,
})

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

def _pick_cjk_font():
    # 常见中文字体（按优先级）：macOS / Windows / Linux
    candidates = [
        "PingFang SC", "Hiragino Sans GB",          # macOS
        "Microsoft YaHei", "Microsoft JhengHei",    # Windows
        "SimHei", "SimSun",                         # 通用
        "Noto Sans CJK SC", "Noto Serif CJK SC",    # Linux/通用（思源黑体/宋体）
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_CJK_NAME = _pick_cjk_font()
if _CJK_NAME:
    # Times New Roman 为主，中文回退到可用 CJK 字体，Latin 扩展回退 DejaVu Sans
    mpl.rcParams["font.family"] = ["Times New Roman", _CJK_NAME, "DejaVu Sans"]
else:
    # 没有中文字体也要保证 Latin 扩展字符显示
    mpl.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
    print("⚠️ 未找到可用的中文字体；已回退到 DejaVu Sans。建议安装 'Noto Sans CJK SC' 或 'Microsoft YaHei'。")
# 让无衬线回退与上面一致
mpl.rcParams["font.sans-serif"] = mpl.rcParams["font.family"]

# 让负号正常显示
mpl.rcParams["axes.unicode_minus"] = False

def _load_tokens(json_path, field="original_token_decoded"):
    """
    既支持 JSON 数组文件，也支持 JSONL（一行一个 JSON）。
    返回：list[str] tokens
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    tokens = []
    with p.open("r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":  # JSON array
            data = json.load(f)
            for item in data:
                if isinstance(item, dict) and field in item:
                    tokens.append(str(item[field]))
        else:  # JSON Lines
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and field in obj:
                    tokens.append(str(obj[field]))
    return tokens

def _count_tokens(tokens, lowercase=True, exclude_whitespace=True, exclude_punct=True):
    """
    统计词频，可选：小写化、排除空白/控制符、排除纯标点。
    """
    ctr = Counter()
    for t in tokens:
        s = t.lower() if lowercase else t

        if exclude_whitespace:
            # 常见空白控制符/特殊空格 token
            if s.strip() == "" or s in {"Ġ", "Ċ", "ĊĊ"}:
                continue
        if exclude_punct and re.fullmatch(r"\W+", s or ""):
            # 纯非字母数字（标点/符号）
            continue
        ctr[s] += 1
    return ctr

def plot_token_frequency(json_path,
                         top_n=30,
                         rotate=45,
                         lowercase=True,
                         exclude_whitespace=True,
                         exclude_punct=True,
                         title="",
                         save_path=None):
    """
    绘制 original_token_decoded 的词频柱状图（Top-N）。
    - json_path: 输入 JSON/JSONL 路径
    - top_n: 展示前多少个高频词
    - rotate: x 轴标签旋转角度（斜着排）
    - lowercase/exclude_*: 预处理开关
    - title: 主标题（留空即显示默认）
    - save_path: 保存路径（None 则直接展示）
    """
    tokens = _load_tokens(json_path, field="original_token_decoded")
    freq = _count_tokens(tokens,
                         lowercase=lowercase,
                         exclude_whitespace=exclude_whitespace,
                         exclude_punct=exclude_punct)

    if not freq:
        raise ValueError("没有可用的 token（可能全部被过滤或文件为空）。")

    most = freq.most_common(top_n)
    words, counts = zip(*most)

    fig_w = max(8.0, 0.45 * len(words) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 8.0))

    # Gradient colors from deep blue to light lavender (reference style)
    # Use new Matplotlib colormaps API; fallback for older versions
    try:
        cmap = mpl.colormaps.get_cmap('PuBu')
    except AttributeError:
        cmap = mpl.cm.get_cmap('PuBu')
    # Reverse gradient: dark (left) -> light (right)
    rank = np.linspace(0.95, 0.2, len(words))
    colors = [cmap(r) for r in rank]

    bars = ax.bar(
        range(len(words)), counts,
        color=colors, edgecolor='none', zorder=3, alpha=0.95
    )

    # Fixed ticks then labels (avoid FixedLocator warning)
    ax.set_xticks(range(len(words)))
    # Use composite family for broad glyph coverage
    if '_CJK_NAME' in globals() and _CJK_NAME:
        fp = FontProperties(family=["Times New Roman", _CJK_NAME, "DejaVu Sans"])
    else:
        fp = FontProperties(family=["Times New Roman", "DejaVu Sans"])
    ax.set_xticklabels(words, rotation=30 if rotate is None else rotate, ha='right', fontproperties=fp)

    ax.set_xlabel("", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(title if title else f"Token Frequency (Top {top_n})", pad=8)

    # Grid and spines (reference style)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.9, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.tick_params(direction='out', width=2.0, length=5)

    ax.margins(y=0.06)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to: {save_path}")
    else:
        plt.show()
    plt.close()

# ===== 使用示例 =====
if __name__ == "__main__":
    plot_token_frequency(
        json_path="source/token.json",  # 换成你的实际路径
        top_n=30,               # 展示 Top-40
        rotate=45,              # 斜向排布
        lowercase=True,         # 小写合并
        exclude_whitespace=True,# 过滤空白/特殊 token
        exclude_punct=True,     # 过滤纯标点
        title="",               # 主标题留空即可
        save_path="image/token_freq.png"
    )