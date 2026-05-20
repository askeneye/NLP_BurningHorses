from __future__ import annotations

import csv
import html
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bart_step_budget_fitness_curves.csv"
SVG_PATH = ROOT / "plots" / "bart_step_budget_fitness_curves.svg"
PNG_PATH = ROOT / "plots" / "bart_step_budget_fitness_curves.png"
PDF_PATH = ROOT / "plots" / "bart_step_budget_fitness_curves.pdf"
FONT_FAMILY = "Inter, Arial, sans-serif"

STEPS = list(range(100, 2100, 100))

# Recovered from the original OADA-XE evidence canvas. Values are mean mini-val
# strict span F1 across the three split seeds for fixed-2000 pattern_01 pilots.
CURVES = [
    {
        "k": "k=5",
        "series": "XE fixed-2000",
        "values": [
            0.2250,
            0.3080,
            0.3110,
            0.3357,
            0.2983,
            0.3332,
            0.3338,
            0.3643,
            0.3781,
            0.3952,
            0.3947,
            0.4008,
            0.4194,
            0.3788,
            0.3936,
            0.3800,
            0.3410,
            0.3517,
            0.3500,
            0.3714,
        ],
    },
    {
        "k": "k=5",
        "series": "OADA-XE cap24 fixed-2000",
        "values": [
            0.1953,
            0.2400,
            0.3029,
            0.2734,
            0.3168,
            0.2994,
            0.3045,
            0.3128,
            0.3083,
            0.3365,
            0.3366,
            0.3493,
            0.3733,
            0.3714,
            0.3796,
            0.3865,
            0.3829,
            0.3895,
            0.3880,
            0.3765,
        ],
    },
    {
        "k": "k=10",
        "series": "XE fixed-2000",
        "values": [
            0.2254,
            0.3809,
            0.4814,
            0.5030,
            0.4476,
            0.4719,
            0.4763,
            0.4923,
            0.5065,
            0.4861,
            0.4672,
            0.4651,
            0.4668,
            0.4949,
            0.4877,
            0.4447,
            0.4633,
            0.4260,
            0.4712,
            0.4736,
        ],
    },
    {
        "k": "k=10",
        "series": "OADA-XE cap24 fixed-2000",
        "values": [
            0.1958,
            0.3906,
            0.4276,
            0.4471,
            0.4461,
            0.4820,
            0.4647,
            0.4707,
            0.4469,
            0.4700,
            0.4760,
            0.4781,
            0.4701,
            0.4620,
            0.4850,
            0.4800,
            0.4660,
            0.4732,
            0.4815,
            0.4586,
        ],
    },
    {
        "k": "k=20",
        "series": "XE fixed-2000",
        "values": [
            0.2338,
            0.4267,
            0.4716,
            0.5250,
            0.5650,
            0.5300,
            0.5582,
            0.5617,
            0.5605,
            0.5515,
            0.5728,
            0.5354,
            0.5668,
            0.5755,
            0.5488,
            0.5713,
            0.5742,
            0.5749,
            0.5534,
            0.5426,
        ],
    },
    {
        "k": "k=20",
        "series": "OADA-XE cap24 fixed-2000",
        "values": [
            0.2518,
            0.4519,
            0.5171,
            0.5667,
            0.5483,
            0.5898,
            0.5497,
            0.5644,
            0.5738,
            0.5625,
            0.5821,
            0.5766,
            0.5612,
            0.5917,
            0.5765,
            0.5643,
            0.5674,
            0.5631,
            0.5727,
            0.5418,
        ],
    },
    {
        "k": "k=50",
        "series": "XE fixed-2000",
        "values": [
            0.2042,
            0.4404,
            0.5785,
            0.6138,
            0.6609,
            0.6376,
            0.6692,
            0.6813,
            0.6823,
            0.6902,
            0.6788,
            0.6645,
            0.6862,
            0.6936,
            0.6952,
            0.6791,
            0.6855,
            0.6887,
            0.6963,
            0.6859,
        ],
    },
    {
        "k": "k=50",
        "series": "OADA-XE cap24 fixed-2000",
        "values": [
            0.2209,
            0.4797,
            0.5702,
            0.6425,
            0.6316,
            0.6715,
            0.6596,
            0.6892,
            0.6957,
            0.7162,
            0.6997,
            0.7102,
            0.7075,
            0.6979,
            0.6988,
            0.6904,
            0.7022,
            0.7214,
            0.6926,
            0.7082,
        ],
    },
]


def write_csv() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["k_shot", "series", "step", "mini_val_strict_f1"],
        )
        writer.writeheader()
        for curve in CURVES:
            for step, value in zip(STEPS, curve["values"]):
                writer.writerow(
                    {
                        "k_shot": curve["k"],
                        "series": curve["series"],
                        "step": step,
                        "mini_val_strict_f1": value,
                    }
                )


def plot() -> None:
    if plt is None:
        render_svg()
        return

    plt.rcParams["font.family"] = ["Inter", "Arial", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25), dpi=300, sharex=True, sharey=True)
    colors = {"XE fixed-2000": "#4C78A8", "OADA-XE cap24 fixed-2000": "#F58518"}
    markers = {"XE fixed-2000": "o", "OADA-XE cap24 fixed-2000": "^"}

    for ax, k_shot in zip(axes.flat, ["k=5", "k=10", "k=20", "k=50"]):
        for curve in [curve for curve in CURVES if curve["k"] == k_shot]:
            ax.plot(
                STEPS,
                curve["values"],
                label=curve["series"],
                color=colors[curve["series"]],
                linewidth=1.7,
                marker=markers[curve["series"]],
                markersize=2.5,
                markevery=2,
            )
        ax.axvline(2000, color="#111111", linewidth=1.0, linestyle="--", alpha=0.85)
        ax.set_title(k_shot, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_xlim(100, 2050)
        ax.set_ylim(0.18, 0.75)

    axes[0, 0].set_ylabel("Mini-val strict F1")
    axes[1, 0].set_ylabel("Mini-val strict F1")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 1].set_xlabel("Training step")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("BART Step-Budget Pilot Fitness Curves", y=1.04, fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Dashed vertical line marks the fixed 2000-step budget selected for report-grade BART teacher runs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def render_svg() -> None:
    width = 700
    height = 525
    margin_left = 70
    margin_right = 28
    margin_top = 78
    margin_bottom = 58
    panel_gap_x = 42
    panel_gap_y = 48
    title_y = 24
    plot_width = (width - margin_left - margin_right - panel_gap_x) / 2
    plot_height = (height - margin_top - margin_bottom - panel_gap_y) / 2
    y_min = 0.18
    y_max = 0.75
    colors = {"XE fixed-2000": "#4C78A8", "OADA-XE cap24 fixed-2000": "#F58518"}
    markers = {"XE fixed-2000": "circle", "OADA-XE cap24 fixed-2000": "triangle"}

    def x_pos(step: int, left: float) -> float:
        return left + ((step - 100) / (2000 - 100)) * plot_width

    def y_pos(value: float, top: float) -> float:
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    def marker(series: str, x: float, y: float) -> str:
        color = colors[series]
        if markers[series] == "triangle":
            points = f"{x:.1f},{y - 3:.1f} {x - 3:.1f},{y + 2.6:.1f} {x + 3:.1f},{y + 2.6:.1f}"
            return f'<polygon points="{points}" fill="{color}" stroke="{color}" stroke-width="1"/>'
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{color}" stroke="{color}" stroke-width="1"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>BART Step-Budget Pilot Fitness Curves</title>",
        f"<style>text {{ font-family: {FONT_FAMILY}; }}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="{title_y}" font-size="16" font-weight="700" text-anchor="middle">BART Step-Budget Pilot Fitness Curves</text>',
    ]

    legend_x = margin_left
    legend_y = 50
    for index, series in enumerate(["XE fixed-2000", "OADA-XE cap24 fixed-2000"]):
        y = legend_y
        x1 = legend_x + index * 175
        x2 = x1 + 30
        color = colors[series]
        parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="1.7"/>')
        parts.append(marker(series, (x1 + x2) / 2, y))
        parts.append(f'<text x="{x2 + 8}" y="{y + 4}" font-size="11">{html.escape(series)}</text>')

    panel_positions = {
        "k=5": (margin_left, margin_top),
        "k=10": (margin_left + plot_width + panel_gap_x, margin_top),
        "k=20": (margin_left, margin_top + plot_height + panel_gap_y),
        "k=50": (margin_left + plot_width + panel_gap_x, margin_top + plot_height + panel_gap_y),
    }

    for k_shot, (left, top) in panel_positions.items():
        parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{top - 10:.1f}" font-size="12" font-weight="700" text-anchor="middle">{k_shot}</text>')
        for tick in [0.2, 0.4, 0.6]:
            y = y_pos(tick, top)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
            if left == margin_left:
                parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="10" text-anchor="end">{tick:.1f}</text>')
        parts.append(f'<rect x="{left}" y="{top}" width="{plot_width:.1f}" height="{plot_height:.1f}" fill="none" stroke="#111111" stroke-width="1.2"/>')
        x_2000 = x_pos(2000, left)
        parts.append(f'<line x1="{x_2000:.1f}" y1="{top}" x2="{x_2000:.1f}" y2="{top + plot_height:.1f}" stroke="#111111" stroke-width="1" stroke-dasharray="5 4"/>')
        for step in [100, 1000, 2000]:
            x = x_pos(step, left)
            parts.append(f'<line x1="{x:.1f}" y1="{top + plot_height:.1f}" x2="{x:.1f}" y2="{top + plot_height + 5:.1f}" stroke="#111111" stroke-width="1"/>')
            if top > margin_top:
                parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 18:.1f}" font-size="10" text-anchor="middle">{step}</text>')

        for curve in [curve for curve in CURVES if curve["k"] == k_shot]:
            series = curve["series"]
            points = [(x_pos(step, left), y_pos(value, top)) for step, value in zip(STEPS, curve["values"])]
            parts.append(
                f'<path d="{svg_path(points)}" fill="none" stroke="{colors[series]}" stroke-width="1.7" stroke-opacity="0.95"/>'
            )
            for x, y in points[::2]:
                parts.append(marker(series, x, y))

    parts.append(f'<text x="22" y="{height / 2:.1f}" font-size="13" transform="rotate(-90 22 {height / 2:.1f})">Mini-val strict F1</text>')
    parts.append(f'<text x="{width / 2:.1f}" y="{height - 24}" font-size="13" text-anchor="middle">Training step</text>')
    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 7}" font-size="9" text-anchor="middle">'
        "Dashed vertical line marks the fixed 2000-step BART budget selected for report-grade teacher runs."
        "</text>"
    )
    parts.append("</svg>")

    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    write_csv()
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
