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
            if curve["series"] != "XE fixed-2000":
                continue
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

    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=300)
    colors = {"k=5": "#4C78A8", "k=10": "#F58518", "k=20": "#54A24B", "k=50": "#B279A2"}

    for curve in [curve for curve in CURVES if curve["series"] == "XE fixed-2000"]:
        k_shot = curve["k"]
        ax.plot(
            STEPS,
            curve["values"],
            label=k_shot,
            color=colors[k_shot],
            linewidth=1.7,
            marker="o",
            markersize=2.6,
            markevery=2,
        )
    ax.set_title("BART fitness curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mini-val strict F1")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_xlim(0, 2100)
    ax.set_ylim(0.18, 0.75)
    ax.set_xticks([100, 500, 1000, 1500, 2000])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", frameon=True, framealpha=0.92, fontsize=8)
    fig.tight_layout()

    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def render_svg() -> None:
    width = 500
    height = 300
    margin_left = 58
    margin_right = 22
    margin_top = 45
    margin_bottom = 44
    title_y = 22
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_min = 0.18
    y_max = 0.75
    colors = {"k=5": "#4C78A8", "k=10": "#F58518", "k=20": "#54A24B", "k=50": "#B279A2"}

    def x_pos(step: int) -> float:
        return margin_left + (step / 2100) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    def marker(k_shot: str, x: float, y: float) -> str:
        color = colors[k_shot]
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{color}" stroke="{color}" stroke-width="1"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>BART fitness curve</title>",
        f"<style>text {{ font-family: {FONT_FAMILY}; }}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="{title_y}" font-size="14" font-weight="700" text-anchor="middle">BART fitness curve</text>',
    ]

    for tick in [0.2, 0.4, 0.6]:
        y = y_pos(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 8}" y="{y + 4:.1f}" font-size="10" text-anchor="end">{tick:.1f}</text>')
    parts.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width:.1f}" height="{plot_height:.1f}" fill="none" stroke="#111111" stroke-width="1.2"/>')
    for step in [100, 500, 1000, 1500, 2000]:
        x = x_pos(step)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top + plot_height:.1f}" x2="{x:.1f}" y2="{margin_top + plot_height + 5:.1f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin_top + plot_height + 18:.1f}" font-size="10" text-anchor="middle">{step}</text>')

    for curve in [curve for curve in CURVES if curve["series"] == "XE fixed-2000"]:
        k_shot = curve["k"]
        points = [(x_pos(step), y_pos(value)) for step, value in zip(STEPS, curve["values"])]
        parts.append(
            f'<path d="{svg_path(points)}" fill="none" stroke="{colors[k_shot]}" stroke-width="1.7" stroke-opacity="0.95"/>'
        )
        for x, y in points[::2]:
            parts.append(marker(k_shot, x, y))

    legend_width = 82
    legend_height = 58
    legend_x = margin_left + plot_width - legend_width - 9
    legend_y = margin_top + plot_height - legend_height - 9
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" fill="white" opacity="0.92" stroke="#BBBBBB" stroke-width="0.8"/>')
    for index, k_shot in enumerate(["k=50", "k=20", "k=10", "k=5"]):
        y = legend_y + 13 + index * 11
        x1 = legend_x + 9
        x2 = x1 + 18
        parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{colors[k_shot]}" stroke-width="1.7"/>')
        parts.append(marker(k_shot, (x1 + x2) / 2, y))
        parts.append(f'<text x="{x2 + 7}" y="{y + 3.5:.1f}" font-size="9">{html.escape(k_shot)}</text>')

    parts.append(f'<text x="19" y="{margin_top + plot_height / 2:.1f}" font-size="11" transform="rotate(-90 19 {margin_top + plot_height / 2:.1f})">Mini-val strict F1</text>')
    parts.append(f'<text x="{width / 2:.1f}" y="{height - 7}" font-size="11" text-anchor="middle">Training step</text>')
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
