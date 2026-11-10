"""
Draw the project showcase figure: accuracy vs. CPU inference latency for every
candidate architecture, with the delivered model marked.

Self-contained by design: this is a presentation figure, not pipeline code.

Usage

    python scripts/plot_showcase_figure.py --lang zh --out reports/figures/
    python scripts/plot_showcase_figure.py --lang en --out reports/figures/

Numbers come from the committed report tables, not from re-running anything:
  - reports/experiments/final_candidates.md   (val exact match, mean +/- SD over 3 seeds)
  - reports/experiments/benchmark.md          (params, ONNX size, latency, throughput)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

# ------------------------------ Data ------------------------------


@dataclass(frozen=True)
class Candidate:
    key: str
    label_en: str
    label_zh: str
    family: str  # "public" | "custom"
    params_m: float
    mflops: int
    onnx_mb: float
    gpu_ms: float  # single image, PyTorch FP32
    gpu_batch: int  # images/s, batch 256
    cpu_ms: float  # single image, ONNX Runtime, 6 threads
    edge_ms: float  # single image, ONNX Runtime, 1 thread (proxy)
    val_exact: float  # validation exact match, mean of 3 seeds
    val_exact_sd: float


CANDIDATES: list[Candidate] = [
    # off-the-shelf architectures
    Candidate("resnet18", "ResNet-18", "ResNet-18", "public", 11.17, 3047, 44.7, 1.43, 3796, 3.77, 10.54, 0.9881, 0.0011),
    Candidate("mobilenet_v3_small", "MobileNetV3-Small", "MobileNetV3-Small", "public", 1.53, 94, 6.3, 1.84, 11079, 0.34, 0.64, 0.9875, 0.0001),
    Candidate("mobilenet_v3_large", "MobileNetV3-Large", "MobileNetV3-Large", "public", 4.21, 355, 17.1, 2.29, 4063, 0.86, 1.86, 0.9881, 0.0003),
    Candidate("efficientnet_b0", "EfficientNet-B0", "EfficientNet-B0", "public", 4.02, 646, 16.5, 3.78, 2357, 1.57, 3.51, 0.9886, 0.0005),
    Candidate("shufflenet_v2_x1_0", "ShuffleNetV2 1.0x", "ShuffleNetV2 1.0x", "public", 1.26, 250, 5.3, 2.31, 7336, 0.82, 1.43, 0.9884, 0.0006),
    # custom compact ResNet family
    Candidate("slim_resnet_w16_d2_std", "SlimResNet w16-d2-std", "SlimResNet w16-d2-std", "custom", 0.70, 18, 2.9, 0.90, 180685, 0.07, 0.10, 0.9806, 0.0027),
    Candidate("slim_resnet_w24_d1_std", "SlimResNet w24-d1-std", "SlimResNet w24-d1-std", "custom", 0.69, 20, 2.8, 0.74, 212927, 0.06, 0.12, 0.9827, 0.0025),
    Candidate("slim_resnet_w32_d1_std", "SlimResNet w32-d1-std", "SlimResNet w32-d1-std", "custom", 1.23, 33, 5.0, 0.64, 162971, 0.08, 0.14, 0.9841, 0.0014),
    Candidate("slim_resnet_w16_d2_small", "SlimResNet w16-d2-small", "SlimResNet w16-d2-small", "custom", 0.70, 192, 2.9, 1.13, 24672, 0.28, 0.72, 0.9868, 0.0003),
    Candidate("slim_resnet_w32_d1_small", "SlimResNet w32-d1-small", "SlimResNet w32-d1-small", "custom", 1.23, 349, 5.0, 0.71, 18166, 0.45, 1.17, 0.9856, 0.0017),
]

CHOSEN = "shufflenet_v2_x1_0"
BASELINE = "resnet18"
OVERLAPPED = "mobilenet_v3_large"  # sits almost on top of the chosen model


# --- Palette (validated: all-pairs CVD dE 24.7, normal-vision dE 33.6, both >= 3:1) ---


SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_PUBLIC = "#2a78d6"  # categorical slot 1
SERIES_CUSTOM = "#eb6834"  # categorical slot 2


TEXT = {
    "zh": {
        "title": "10 個候選架構的準確度—效率取捨",
        "subtitle": "每個點為 3 個隨機種子的平均，誤差為 ±1 標準差",
        "xlabel": "CPU 伺服器單張推論延遲 (ms，對數刻度)",
        "ylabel": "驗證集完全正確率 (8 個標籤全對)",
        "legend_public": "公開架構",
        "legend_custom": "自研發輕量 ResNet",
        "legend_front": "準確度—效率前緣",
        "baseline": "ResNet-18 (基準)",
        "fastest": "自研發架構最快可到 0.06 ms，但完全正確率低於選定模型",
        "callout_head": "ShuffleNetV2 1.0x — 已選定",
        "callout_body": ("驗證集完全正確率 0.9884 ± 0.0006\nONNX 5.3 MB CPU 0.82 ms Edge 1.43 ms\nGPU 每批 256 張時 7,336 張/秒"),
        "footnote": ("Mixed-type Wafer Map Defect Dataset (38,015 張、8 類、多標籤) CPU：ONNX Runtime 6 執行緒中位數 邊緣：CPU 單執行緒代理，僅供排序"),
        "mobilenet_large": "MobileNetV3-Large (重疊)",
    },
    "en": {
        "title": "Accuracy vs. efficiency across 10 candidate architectures",
        "subtitle": "Each point is the mean of 3 seeds; bars are ±1 SD across seeds",
        "xlabel": "CPU server single-image latency (ms, log scale)",
        "ylabel": "Validation exact match (all 8 labels right)",
        "legend_public": "Off-the-shelf",
        "legend_custom": "Custom compact ResNet",
        "legend_front": "Accuracy-efficiency frontier",
        "baseline": "ResNet-18 (baseline)",
        "fastest": "custom nets reach 0.06 ms, but at a lower exact match",
        "callout_head": "ShuffleNetV2 1.0x — selected",
        "callout_body": ("Validation exact match 0.9884 ± 0.0006\nONNX 5.3 MB   CPU 0.82 ms   Edge 1.43 ms\nGPU 7,336 img/s at batch 256"),
        "footnote": ("MixedWM38: 38,015 wafer maps, 8 multi-label classes   CPU: ONNX Runtime, 6 threads   Edge: 1-thread CPU proxy, ranking only"),
        "mobilenet_large": "MobileNetV3-Large (overlap)",
    },
}


CJK_FONTS = ["Noto Sans CJK TC", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", "Source Han Sans TC", "WenQuanYi Zen Hei"]
LATIN_FONTS = ["Noto Sans", "DejaVu Sans", "Arial", "Helvetica"]


def pick_font(lang: str) -> str:
    """
    Return an installed font for `lang`.

    The Chinese figure needs a CJK font: without one every character renders as an
    empty box and matplotlib only warns, which would quietly commit a broken figure.
    """
    wanted = CJK_FONTS if lang == "zh" else LATIN_FONTS
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in wanted:
        if name in installed:
            return name
    if lang == "zh":
        raise SystemExit(
            "No CJK font is installed, so every Chinese character would render as an\n"
            "empty box. Install one and clear matplotlib's font cache:\n"
            "\n"
            "    sudo apt install fonts-noto-cjk      # Debian / Ubuntu / WSL\n"
            "    rm -rf ~/.cache/matplotlib\n"
            "\n"
            f"Looked for: {', '.join(CJK_FONTS)}"
        )
    return "DejaVu Sans"


def pareto_front(points: list[Candidate]) -> list[Candidate]:
    """Lower cpu_ms is better, higher val_exact is better."""
    front = []
    for p in sorted(points, key=lambda c: c.cpu_ms):
        if all(p.val_exact > q.val_exact for q in front):
            front.append(p)
    return front


def check_fits(fig, artists: dict[str, object]) -> None:
    """
    Fail if any text runs off the canvas.

    matplotlib clips an over-long string at the figure edge without a word, so a
    truncated caption only shows up in the rendered PNG. Catch it here instead.
    """
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.dpi
    renderer = fig.canvas.get_renderer()
    for name, artist in artists.items():
        box = artist.get_window_extent(renderer)
        if box.x0 < 0 or box.x1 > width or box.y0 < 0 or box.y1 > height:
            raise SystemExit(f"{name} does not fit the canvas: it spans {box.x0:.0f}..{box.x1:.0f} px of 0..{width:.0f}. Shorten the string in TEXT.")


def draw(lang: str, out_path: Path, scale: int = 1) -> Path:
    t = TEXT[lang]
    family = pick_font(lang)
    plt.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(7.04, 3.96), dpi=100 * scale, facecolor=SURFACE)
    ax = fig.add_axes((0.105, 0.175, 0.872, 0.645))
    ax.set_facecolor(SURFACE)

    # chrome
    ax.set_xscale("log")
    ax.set_xlim(0.045, 7.5)
    ax.set_ylim(0.9770, 0.9900)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=3, width=0.8)
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_yticks([0.980, 0.982, 0.984, 0.986, 0.988])
    ax.set_yticklabels(["0.980", "0.982", "0.984", "0.986", "0.988"])
    ax.set_xlabel(t["xlabel"], fontsize=8.5, color=INK_SECONDARY, labelpad=4)
    ax.set_ylabel(t["ylabel"], fontsize=8.5, color=INK_SECONDARY, labelpad=4)

    # frontier
    front = pareto_front(CANDIDATES)
    ax.plot([c.cpu_ms for c in front], [c.val_exact for c in front], color=INK_SECONDARY, linewidth=1.3, alpha=0.8, zorder=1, solid_capstyle="round")

    # marks
    for c in CANDIDATES:
        colour = SERIES_PUBLIC if c.family == "public" else SERIES_CUSTOM
        marker = "o" if c.family == "public" else "^"
        chosen = c.key == CHOSEN
        if chosen:
            # ring first, behind every marker, so a close neighbour is not swallowed
            ax.plot(c.cpu_ms, c.val_exact, marker=marker, markersize=17, markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.3, alpha=0.9, zorder=2)
        ax.plot(
            c.cpu_ms,
            c.val_exact,
            marker=marker,
            markersize=11 if chosen else 7,
            markerfacecolor=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0 if chosen else 1.2,
            zorder=5 if c.key == OVERLAPPED else (4 if chosen else 3),
        )
        ax.errorbar(c.cpu_ms, c.val_exact, yerr=c.val_exact_sd, fmt="none", ecolor=colour, elinewidth=1.0, capsize=2.5, capthick=1.0, alpha=0.55, zorder=1)

    # direct labels
    by_key = {c.key: c for c in CANDIDATES}

    ax.text(0.185, 0.135, t["fastest"], transform=ax.transAxes, ha="left", va="center", fontsize=7.0, color=INK_MUTED)

    base = by_key[BASELINE]
    ax.annotate(t["baseline"], xy=(base.cpu_ms, base.val_exact), xytext=(53, 17), textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK_SECONDARY)

    # no leader: the label sits in the gap to the right of the pair, which is empty
    mnl = by_key[OVERLAPPED]
    ax.annotate(t["mobilenet_large"], xy=(mnl.cpu_ms, mnl.val_exact), xytext=(118, -10), textcoords="offset points", ha="right", va="center", fontsize=7.5, color=INK_SECONDARY)

    eff = by_key["efficientnet_b0"]
    ax.annotate("EfficientNet-B0", xy=(eff.cpu_ms, eff.val_exact), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)

    sel = by_key[CHOSEN]
    ax.annotate(
        t["callout_head"] + "\n" + t["callout_body"],
        xy=(sel.cpu_ms, sel.val_exact),
        xycoords="data",
        xytext=(0.985, 0.30),
        textcoords="axes fraction",  # was (0.025, 0.97)
        ha="right",
        va="bottom",  # was "left", "top"
        fontsize=7.6,
        color=INK_PRIMARY,
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8),
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8, alpha=0.6, shrinkA=4, shrinkB=12),
    )

    # legend
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6, markerfacecolor=SERIES_PUBLIC, markeredgecolor=SURFACE, markeredgewidth=1.0, label=t["legend_public"]),
        Line2D([], [], marker="^", linestyle="none", markersize=6, markerfacecolor=SERIES_CUSTOM, markeredgecolor=SURFACE, markeredgewidth=1.0, label=t["legend_custom"]),
        Line2D([], [], color=INK_SECONDARY, linewidth=1.3, alpha=0.8, label=t["legend_front"]),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.5, borderpad=0.2, labelspacing=0.35)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    # titles & footnote
    fig.text(0.100, 0.935, t["title"], fontsize=12.5, color=INK_PRIMARY, weight="bold", va="top")
    fig.text(0.100, 0.862, t["subtitle"], fontsize=7.8, color=INK_SECONDARY, va="top")
    fig.text(0.100, 0.035, t["footnote"], fontsize=6.8, color=INK_MUTED, va="bottom")

    # titles & footnote
    title = fig.text(0.100, 0.935, t["title"], fontsize=12.5, color=INK_PRIMARY, weight="bold", va="top")
    subtitle = fig.text(0.100, 0.862, t["subtitle"], fontsize=7.8, color=INK_SECONDARY, va="top")
    footnote = fig.text(0.100, 0.035, t["footnote"], fontsize=6.8, color=INK_MUTED, va="bottom")

    check_fits(fig, {"title": title, "subtitle": subtitle, "footnote": footnote})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, dpi=100 * scale)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("zh", "en"), default="zh")
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()

    for scale, suffix in ((1, ""), (2, "@2x")):
        path = args.out / f"model_selection_{args.lang}{suffix}.png"
        draw(args.lang, path, scale)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
