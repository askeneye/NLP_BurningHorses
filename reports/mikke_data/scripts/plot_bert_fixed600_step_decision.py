from __future__ import annotations

import csv
import html
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "plots" / "bert_fixed600_step_decision_data.csv"
SVG_PATH = ROOT / "plots" / "bert_fixed600_step_decision.svg"
PNG_PATH = ROOT / "plots" / "bert_fixed600_step_decision.png"
PDF_PATH = ROOT / "plots" / "bert_fixed600_step_decision.pdf"

# Recovered from the distillation-step-diagnostics canvas. These are mini-val
# peak steps from the 51 report-grade distilled BERT diagnostic runs.
BY_K = [
    {
        "k_shot": 5,
        "n_runs": 15,
        "mean_best_step": 580,
        "median_best_step": 600,
        "early_stopped": 7,
        "step_counts": {300: 4, 400: 2, 500: 1, 600: 1, 700: 3, 800: 1, 900: 3},
    },
    {
        "k_shot": 10,
        "n_runs": 12,
        "mean_best_step": 608,
        "median_best_step": 600,
        "early_stopped": 6,
        "step_counts": {300: 1, 400: 4, 500: 1, 700: 1, 800: 3, 900: 2},
    },
    {
        "k_shot": 20,
        "n_runs": 12,
        "mean_best_step": 592,
        "median_best_step": 600,
        "early_stopped": 5,
        "step_counts": {200: 1, 400: 3, 500: 1, 600: 2, 800: 5},
    },
    {
        "k_shot": 50,
        "n_runs": 12,
        "mean_best_step": 633,
        "median_best_step": 600,
        "early_stopped": 6,
        "step_counts": {300: 1, 400: 2, 500: 3, 700: 2, 800: 1, 900: 2, 1000: 1},
    },
]

STEP_BINS = [
    ("<=400", lambda step: step <= 400),
    ("500-600", lambda step: 500 <= step <= 600),
    ("700-800", lambda step: 700 <= step <= 800),
    (">=900", lambda step: step >= 900),
]


def write_plot_data() -> list[dict[str, object]]:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for item in BY_K:
        for step, count in sorted(item["step_counts"].items()):
            rows.append(
                {
                    "figure_id": "bert_fixed600_step_decision",
                    "k_shot": item["k_shot"],
                    "n_runs": item["n_runs"],
                    "mean_best_step": item["mean_best_step"],
                    "median_best_step": item["median_best_step"],
                    "early_stopped": item["early_stopped"],
                    "best_step": step,
                    "best_step_count": count,
                    "source": "distillation-step-diagnostics.canvas.tsx",
                }
            )

    with DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def bin_percentages(item: dict[str, object]) -> list[float]:
    step_counts: dict[int, int] = item["step_counts"]  # type: ignore[assignment]
    n_runs = int(item["n_runs"])
    percentages = []
    for _, predicate in STEP_BINS:
        count = sum(count for step, count in step_counts.items() if predicate(step))
        percentages.append(100 * count / n_runs)
    return percentages


def plot() -> None:
    if plt is None:
        render_svg()
        return

    plt.rcParams["font.family"] = ["Inter", "Arial", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"

    k_labels = [str(item["k_shot"]) for item in BY_K]
    x_positions = range(len(BY_K))
    mean_steps = [item["mean_best_step"] for item in BY_K]
    medians = [item["median_best_step"] for item in BY_K]
    early_stop_pct = [100 * item["early_stopped"] / item["n_runs"] for item in BY_K]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), dpi=300)

    ax = axes[0]
    ax.plot(x_positions, mean_steps, color="#4C78A8", marker="o", linewidth=1.8, label="Mean best step")
    ax.plot(x_positions, medians, color="#222222", marker="s", linewidth=1.4, linestyle="--", label="Median best step")
    ax.axhspan(500, 700, color="#4C78A8", alpha=0.08, linewidth=0)
    ax.set_xticks(list(x_positions), k_labels)
    ax.set_xlabel("k-shot setting")
    ax.set_ylabel("Mini-val best step")
    ax.set_ylim(150, 1050)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    bottoms = [0.0] * len(BY_K)
    colors = ["#F58518", "#4C78A8", "#54A24B", "#B279A2"]
    for (label, _), color, values in zip(STEP_BINS, colors, zip(*(bin_percentages(item) for item in BY_K))):
        ax.bar(x_positions, values, bottom=bottoms, label=label, color=color, width=0.62)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.plot(x_positions, early_stop_pct, color="#222222", marker="D", linewidth=1.5, label="Early-stopped")
    ax.set_xticks(list(x_positions), k_labels)
    ax.set_xlabel("k-shot setting")
    ax.set_ylabel("Share of runs (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.28))

    fig.suptitle("Distilled BERT Fixed-600 Step Decision", fontsize=12, fontweight="bold", y=1.05)
    fig.text(
        0.5,
        -0.03,
        "Source: 51 distilled BERT diagnostics. Median best step is 600 for every k-shot group.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()

    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def render_svg() -> None:
    width = 690
    height = 255
    margin_left = 58
    margin_right = 20
    margin_top = 58
    margin_bottom = 45
    gap = 52
    panel_width = (width - margin_left - margin_right - gap) / 2
    panel_height = height - margin_top - margin_bottom

    k_labels = [str(item["k_shot"]) for item in BY_K]
    mean_steps = [float(item["mean_best_step"]) for item in BY_K]
    medians = [float(item["median_best_step"]) for item in BY_K]
    early_stop_pct = [100 * float(item["early_stopped"]) / float(item["n_runs"]) for item in BY_K]
    binned = [bin_percentages(item) for item in BY_K]
    colors = ["#F58518", "#4C78A8", "#54A24B", "#B279A2"]

    def x_pos(index: int, left: float) -> float:
        return left + (index / (len(BY_K) - 1)) * panel_width

    def y_step(value: float, top: float) -> float:
        return top + panel_height - ((value - 150) / (1050 - 150)) * panel_height

    def y_pct(value: float, top: float) -> float:
        return top + panel_height - (value / 105) * panel_height

    def path(points: list[tuple[float, float]]) -> str:
        return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Distilled BERT Fixed-600 Step Decision</title>",
        "<style>text { font-family: Inter, Arial, sans-serif; }</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="22" font-size="14" font-weight="700" text-anchor="middle">Distilled BERT Fixed-600 Step Decision</text>',
    ]

    left = margin_left
    top = margin_top
    parts.append(f'<rect x="{left}" y="{top}" width="{panel_width:.1f}" height="{panel_height:.1f}" fill="none" stroke="#222222" stroke-width="1"/>')
    for tick in [200, 600, 1000]:
        y = y_step(tick, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_width:.1f}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="{left - 7}" y="{y + 4:.1f}" font-size="9" text-anchor="end">{tick}</text>')
    shade_top = y_step(700, top)
    shade_bottom = y_step(500, top)
    parts.append(f'<rect x="{left}" y="{shade_top:.1f}" width="{panel_width:.1f}" height="{shade_bottom - shade_top:.1f}" fill="#4C78A8" opacity="0.08"/>')

    mean_points = [(x_pos(index, left), y_step(value, top)) for index, value in enumerate(mean_steps)]
    median_points = [(x_pos(index, left), y_step(value, top)) for index, value in enumerate(medians)]
    parts.append(f'<path d="{path(mean_points)}" fill="none" stroke="#4C78A8" stroke-width="2"/>')
    parts.append(f'<path d="{path(median_points)}" fill="none" stroke="#222222" stroke-width="1.5" stroke-dasharray="5 4"/>')
    for x, y in mean_points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#4C78A8"/>')
    for x, y in median_points:
        parts.append(f'<rect x="{x - 2.5:.1f}" y="{y - 2.5:.1f}" width="5" height="5" fill="#222222"/>')
    for index, label in enumerate(k_labels):
        x = x_pos(index, left)
        parts.append(f'<text x="{x:.1f}" y="{top + panel_height + 17:.1f}" font-size="10" text-anchor="middle">{html.escape(label)}</text>')
    parts.append(f'<text x="{left + panel_width / 2:.1f}" y="{height - 8}" font-size="10" text-anchor="middle">k-shot setting</text>')
    parts.append(f'<text x="17" y="{top + panel_height / 2:.1f}" font-size="10" transform="rotate(-90 17 {top + panel_height / 2:.1f})">Mini-val best step</text>')
    parts.append(f'<text x="{left}" y="43" font-size="10" fill="#4C78A8">Mean best step</text>')
    parts.append(f'<text x="{left + 95}" y="43" font-size="10" fill="#222222">Median best step</text>')

    left = margin_left + panel_width + gap
    parts.append(f'<rect x="{left}" y="{top}" width="{panel_width:.1f}" height="{panel_height:.1f}" fill="none" stroke="#222222" stroke-width="1"/>')
    for tick in [0, 50, 100]:
        y = y_pct(tick, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_width:.1f}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="{left - 7}" y="{y + 4:.1f}" font-size="9" text-anchor="end">{tick}</text>')
    bar_width = 30
    for index, values in enumerate(binned):
        x = x_pos(index, left)
        bottom = top + panel_height
        for value, color in zip(values, colors):
            bar_height = panel_height * value / 105
            bottom -= bar_height
            parts.append(f'<rect x="{x - bar_width / 2:.1f}" y="{bottom:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>')
    early_points = [(x_pos(index, left), y_pct(value, top)) for index, value in enumerate(early_stop_pct)]
    parts.append(f'<path d="{path(early_points)}" fill="none" stroke="#222222" stroke-width="1.6"/>')
    for x, y in early_points:
        parts.append(f'<rect x="{x - 2.8:.1f}" y="{y - 2.8:.1f}" width="5.6" height="5.6" fill="#222222" transform="rotate(45 {x:.1f} {y:.1f})"/>')
    for index, label in enumerate(k_labels):
        x = x_pos(index, left)
        parts.append(f'<text x="{x:.1f}" y="{top + panel_height + 17:.1f}" font-size="10" text-anchor="middle">{html.escape(label)}</text>')
    parts.append(f'<text x="{left + panel_width / 2:.1f}" y="{height - 8}" font-size="10" text-anchor="middle">k-shot setting</text>')
    parts.append(f'<text x="{left - 44:.1f}" y="{top + panel_height / 2:.1f}" font-size="10" transform="rotate(-90 {left - 44:.1f} {top + panel_height / 2:.1f})">Share of runs (%)</text>')
    legend_x = left
    for index, ((label, _), color) in enumerate(zip(STEP_BINS, colors)):
        x = legend_x + index * 52
        parts.append(f'<rect x="{x}" y="35" width="8" height="8" fill="{color}"/>')
        parts.append(f'<text x="{x + 11}" y="43" font-size="8">{html.escape(label)}</text>')
    parts.append(f'<text x="{width / 2:.1f}" y="{height - 25}" font-size="8" text-anchor="middle">Source: 51 distilled BERT diagnostics. Median best step is 600 for every k-shot group.</text>')
    parts.append("</svg>")

    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    write_plot_data()
    plot()
    print(f"Wrote {DATA_PATH}")
    print(f"Wrote {SVG_PATH}")
    if PNG_PATH.exists():
        print(f"Wrote {PNG_PATH}")
    if PDF_PATH.exists():
        print(f"Wrote {PDF_PATH}")
    if plt is None:
        print("matplotlib is not installed; wrote SVG/CSV only.")


if __name__ == "__main__":
    main()
