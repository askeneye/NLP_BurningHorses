from __future__ import annotations

import csv
import html
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "agreement_filter_subset_size_scores.csv"
PLOT_DATA_PATH = ROOT / "plots" / "agreement_filter_subset_size_strict_f1_data.csv"
PNG_PATH = ROOT / "plots" / "agreement_filter_subset_size_strict_f1.png"
PDF_PATH = ROOT / "plots" / "agreement_filter_subset_size_strict_f1.pdf"
SVG_PATH = ROOT / "plots" / "agreement_filter_subset_size_strict_f1.svg"
FONT_FAMILY = "Inter, Arial, sans-serif"
LINE_COLOR = "#E45756"
SUBSET_ORDER = [500, 1000, 2000, 5000]
SUBSET_LABELS = {
    500: "top 500",
    1000: "top 1000",
    2000: "top 2000",
    5000: "top 5000",
}


def read_validation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["eval_split"] != "validation" and row["score_source"] != "validation":
                continue
            if not row["metric_value_percent"]:
                continue
            subset_rows = int(row["subset_rows"])
            if subset_rows not in SUBSET_ORDER:
                continue
            rows.append(
                {
                    **row,
                    "method_label": row["method_label"].replace("->", "\u2192"),
                    "subset_rows": subset_rows,
                    "subset_tick_label": SUBSET_LABELS[subset_rows],
                    "metric_value_percent": float(row["metric_value_percent"]),
                    "color": LINE_COLOR,
                }
            )
    return sorted(rows, key=lambda row: SUBSET_ORDER.index(int(row["subset_rows"])))


def write_plot_data(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "figure_id",
        "method_id",
        "method_label",
        "subset_name",
        "subset_rows",
        "subset_tick_label",
        "unlabeled_pool_rows",
        "pool_fraction",
        "eval_split",
        "metric_name",
        "metric_label",
        "metric_value",
        "metric_value_percent",
        "status",
        "score_source",
        "is_placeholder",
        "color",
        "source_path",
    ]
    with PLOT_DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({"figure_id": "agreement_filter_subset_size_strict_f1", **row})


def render_with_matplotlib(rows: list[dict[str, object]]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    plt.rcParams["font.family"] = ["Inter", "Arial", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"

    x_values = list(range(len(rows)))
    y_values = [float(row["metric_value_percent"]) for row in rows]
    tick_labels = [str(row["subset_tick_label"]) for row in rows]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=300)
    ax.plot(
        x_values,
        y_values,
        color=LINE_COLOR,
        linewidth=1.6,
        marker="o",
        markersize=3.8,
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(tick_labels, fontweight="bold")
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(-0.2, len(rows) - 0.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([str(tick) for tick in [0, 25, 50, 75, 100]])
    ax.set_xlabel("Teacher labels agreement based set size", fontsize=13)
    ax.set_ylabel("Distilled BERT student strict span F1", fontsize=13)
    ax.set_title("Distilled BERT performance from teacher set size", loc="center")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    plt.close(fig)


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def render_svg(rows: list[dict[str, object]]) -> None:
    width = 560
    height = 560
    left = 96
    right = 36
    top = 78
    bottom = 108
    plot_width = width - left - right
    plot_height = height - top - bottom
    edge_inset_fraction = 0.2
    tick_width = plot_width / (len(rows) - 1 + 2 * edge_inset_fraction)
    x_lookup = {
        int(row["subset_rows"]): left + (index + edge_inset_fraction) * tick_width
        for index, row in enumerate(rows)
    }

    def y_pos(value: float) -> float:
        return top + plot_height - (value / 100) * plot_height

    points = [
        (x_lookup[int(row["subset_rows"])], y_pos(float(row["metric_value_percent"])))
        for row in rows
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Distilled BERT performance from teacher set size</title>",
        "<style>text { font-family: Inter, Arial, sans-serif; }</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2 + 28:.1f}" y="34" font-family="{FONT_FAMILY}" font-size="16" font-weight="700" text-anchor="middle">Distilled BERT performance from teacher set size</text>',
        f'<text x="34" y="398" font-family="{FONT_FAMILY}" font-size="15" transform="rotate(-90 34 398)">Distilled BERT student strict span F1</text>',
    ]

    for tick in [0, 25, 50, 75, 100]:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="84" y="{y + 5:.1f}" font-family="{FONT_FAMILY}" font-size="14" text-anchor="end">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#DDDDDD" stroke-width="1"/>')
    parts.append(f'<line x1="{left + plot_width}" y1="{top}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#DDDDDD" stroke-width="1"/>')

    path = svg_path(points)
    parts.append(f'<path d="{path}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6" stroke-opacity="0.95"/>')
    for row, (x, y) in zip(rows, points):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{LINE_COLOR}" stroke="{LINE_COLOR}" stroke-width="1.5"/>')
        value_label = html.escape(f'{float(row["metric_value_percent"]):.2f}')
        parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" font-family="{FONT_FAMILY}" font-size="11" fill="#444444" text-anchor="middle">{value_label}</text>')

    for row in rows:
        x = x_lookup[int(row["subset_rows"])]
        label = html.escape(str(row["subset_tick_label"]))
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height - 8}" stroke="#111111" stroke-width="1.8"/>')
        parts.append(f'<text x="{x:.1f}" y="482" font-family="{FONT_FAMILY}" font-size="14" font-weight="700" text-anchor="middle">{label}</text>')

    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="522" font-family="{FONT_FAMILY}" font-size="15" text-anchor="middle">Teacher labels agreement based set size</text>')
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = read_validation_rows()
    write_plot_data(rows)
    if plt is not None:
        render_with_matplotlib(rows)
    else:
        render_svg(rows)


if __name__ == "__main__":
    main()
