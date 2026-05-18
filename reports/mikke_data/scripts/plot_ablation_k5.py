from __future__ import annotations

import csv
import html
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ModuleNotFoundError:
    plt = None
    Patch = None


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "plots" / "ablation_k5_test_strict_f1_data.csv"
PNG_PATH = ROOT / "plots" / "ablation_k5_test_strict_f1.png"
PDF_PATH = ROOT / "plots" / "ablation_k5_test_strict_f1.pdf"
SVG_PATH = ROOT / "plots" / "ablation_k5_test_strict_f1.svg"


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: int(row["plot_order"]))


def render_with_matplotlib(rows: list[dict[str, str]]) -> None:
    if plt is None or Patch is None:
        raise RuntimeError("matplotlib is not available")

    labels = [row["method_label"] for row in rows]
    values = [float(row["metric_value"]) for row in rows]
    colors = [row["color"] for row in rows]
    annotations = [row["annotation"] for row in rows]
    is_placeholder = [row["is_placeholder"].lower() == "true" for row in rows]

    fig, ax = plt.subplots(figsize=(6.9, 2.8), dpi=300)
    positions = list(range(len(rows)))
    bars = ax.barh(positions, values, color=colors, edgecolor="#222222", linewidth=0.6)

    for bar, value, annotation, placeholder in zip(bars, values, annotations, is_placeholder):
        if placeholder:
            bar.set_alpha(0.45)
            bar.set_hatch("//")
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}  {annotation}",
            va="center",
            fontsize=7,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Strict span F1", fontsize=9)
    ax.set_xlim(0, 0.75)
    ax.set_title("CoNLL2003 k5 ablation draft", fontsize=10)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor="#222222", alpha=0.8, label="test / ready for draft"),
        Patch(facecolor="#222222", alpha=0.45, hatch="//", label="placeholder or pilot"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def render_svg(rows: list[dict[str, str]]) -> None:
    width = 1050
    height = 420
    margin_left = 330
    margin_right = 160
    margin_top = 54
    row_height = 46
    plot_width = width - margin_left - margin_right
    x_max = 0.75

    def x_pos(value: float) -> float:
        return margin_left + (value / x_max) * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>CoNLL2003 k5 ablation draft</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="Arial, sans-serif" font-size="18" font-weight="700">CoNLL2003 k5 ablation draft</text>',
        '<text x="24" y="396" font-family="Arial, sans-serif" font-size="13">Strict span F1</text>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75]:
        x = x_pos(tick)
        parts.append(f'<line x1="{x:.1f}" y1="44" x2="{x:.1f}" y2="360" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="{x - 10:.1f}" y="378" font-family="Arial, sans-serif" font-size="12">{tick:.2f}</text>')

    for index, row in enumerate(rows):
        y = margin_top + index * row_height
        value = float(row["metric_value"])
        bar_width = x_pos(value) - margin_left
        label = html.escape(row["method_label"])
        color = row["color"]
        annotation = html.escape(row["annotation"])
        opacity = "0.45" if row["is_placeholder"].lower() == "true" else "0.92"
        hatch = " (placeholder)" if row["is_placeholder"].lower() == "true" else ""

        parts.append(f'<text x="24" y="{y + 19}" font-family="Arial, sans-serif" font-size="13">{label}</text>')
        parts.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="25" '
            f'fill="{color}" fill-opacity="{opacity}" stroke="#222222" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pos(value) + 8:.1f}" y="{y + 17}" font-family="Arial, sans-serif" font-size="12">'
            f'{value:.3f} {annotation}{hatch}</text>'
        )

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = load_rows()
    if plt is None:
        render_svg(rows)
    else:
        render_with_matplotlib(rows)


if __name__ == "__main__":
    main()
