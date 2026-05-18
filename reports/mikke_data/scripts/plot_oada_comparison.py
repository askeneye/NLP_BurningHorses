from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "oada_comparison_scores.csv"
PLOT_DATA_PATH = ROOT / "plots" / "oada_comparison_strict_f1_data.csv"
PNG_PATH = ROOT / "plots" / "oada_comparison_strict_f1.png"
PDF_PATH = ROOT / "plots" / "oada_comparison_strict_f1.pdf"
SVG_PATH = ROOT / "plots" / "oada_comparison_strict_f1.svg"
FONT_FAMILY = "Inter, Arial, sans-serif"

METHOD_ORDER = [
    "ours_bert",
    "oada_bert",
    "ours_distilled",
    "oada_best",
]

COLORS = {
    "ours": "#7B61FF",
    "oada_paper": "#8F8F8F",
}

LEGEND_ITEMS = [
    {"kind": "heading", "label": "Ours", "color": COLORS["ours"]},
    {"kind": "method", "method_id": "ours_distilled", "label": "Pat+Perm BART ensemble \u2192 BERT"},
    {"kind": "method", "method_id": "ours_bert", "label": "BERT baseline"},
    {"kind": "gap"},
    {"kind": "heading", "label": "OADA best performer", "color": COLORS["oada_paper"]},
    {"kind": "method", "method_id": "oada_best", "label": "PromptNER+OADA"},
    {"kind": "method", "method_id": "oada_bert", "label": "BERT-tagger baseline"},
]


def read_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "k_shot": int(row["k_shot"]),
                    "metric_value_percent": float(row["metric_value_percent"]),
                    "color": COLORS[row["source"]],
                }
            )
    method_rank = {method_id: index for index, method_id in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: (method_rank[str(row["method_id"])], int(row["k_shot"])))


def write_plot_data(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source",
        "method_id",
        "method_label",
        "model_family",
        "k_shot",
        "metric_name",
        "metric_label",
        "metric_value",
        "metric_value_percent",
        "score_source",
        "status",
        "color",
        "notes",
    ]
    with PLOT_DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def line_style(row: dict[str, object]) -> str:
    return "--" if row["model_family"] == "baseline" else "-"


def label_for_legend(method_id: str, default_label: str) -> str:
    for item in LEGEND_ITEMS:
        if item.get("method_id") == method_id:
            return str(item["label"])
    return default_label


def render_with_matplotlib(rows: list[dict[str, object]]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    plt.rcParams["font.family"] = ["Inter", "Arial", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)

    fig, ax = plt.subplots(figsize=(6.9, 2.8), dpi=300)
    for method_id in METHOD_ORDER:
        method_rows = sorted(by_method[method_id], key=lambda row: int(row["k_shot"]))
        first = method_rows[0]
        ax.plot(
            [int(row["k_shot"]) for row in method_rows],
            [float(row["metric_value_percent"]) for row in method_rows],
            color=str(first["color"]),
            linestyle=line_style(first),
            linewidth=1.6,
            marker="o",
            markersize=3.6,
            label=label_for_legend(method_id, str(first["method_label"])),
        )

    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(["k=5", "k=10", "k=20", "k=50"], fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Strict span F1", fontsize=14)
    ax.set_title("Our results vs OADA")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    ordered_handles = []
    ordered_labels = []
    for item in LEGEND_ITEMS:
        if item["kind"] == "heading":
            ordered_handles.append(plt.Line2D([], [], color="none"))
            ordered_labels.append(str(item["label"]))
        elif item["kind"] == "gap":
            ordered_handles.append(plt.Line2D([], [], color="none"))
            ordered_labels.append("")
        else:
            label = str(item["label"])
            ordered_handles.append(handle_by_label[label])
            ordered_labels.append(label)
    legend = ax.legend(ordered_handles, ordered_labels, frameon=False, fontsize=8)
    for text, item in zip(legend.get_texts(), LEGEND_ITEMS):
        if item["kind"] == "heading":
            text.set_weight("bold")
            text.set_color(str(item["color"]))
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def render_svg(rows: list[dict[str, object]]) -> None:
    width = 1050
    height = 460
    left = 90
    right = 360
    top = 58
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    k_values = [5, 10, 20, 50]
    x_lookup = {k: left + index * (plot_width / (len(k_values) - 1)) for index, k in enumerate(k_values)}

    def y_pos(value: float) -> float:
        return top + plot_height - (value / 100) * plot_height

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Our results vs OADA</title>",
        "<style>text { font-family: Inter, Arial, sans-serif; }</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-family="{FONT_FAMILY}" font-size="18" font-weight="700">Our results vs OADA</text>',
        f'<text x="24" y="225" font-family="{FONT_FAMILY}" font-size="16" transform="rotate(-90 24 225)">Strict span F1</text>',
    ]

    for tick in [0, 25, 50, 75, 100]:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="52" y="{y + 4:.1f}" font-family="{FONT_FAMILY}" font-size="12">{tick}</text>')

    for k in k_values:
        x = x_lookup[k]
        parts.append(f'<text x="{x - 12:.1f}" y="410" font-family="{FONT_FAMILY}" font-size="12" font-weight="700">k={k}</text>')

    legend_x1 = left + plot_width + 35
    legend_x2 = left + plot_width + 65
    legend_text_x = left + plot_width + 74
    legend_y = top
    legend_methods: dict[str, dict[str, object]] = {}
    for method_id in METHOD_ORDER:
        method_rows = sorted(by_method[method_id], key=lambda row: int(row["k_shot"]))
        first = method_rows[0]
        legend_methods[method_id] = first
        color = str(first["color"])
        dash = ' stroke-dasharray="7 5"' if line_style(first) == "--" else ""
        points = [
            (x_lookup[int(row["k_shot"])], y_pos(float(row["metric_value_percent"])))
            for row in method_rows
        ]
        path = svg_path(points)
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6" stroke-opacity="0.95"{dash}/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{color}" stroke="{color}" stroke-width="1.5"/>')


    legend_y = top
    for item in LEGEND_ITEMS:
        if item["kind"] == "heading":
            label = html.escape(str(item["label"]))
            color = str(item["color"])
            parts.append(f'<text x="{legend_x1}" y="{legend_y + 4}" font-family="{FONT_FAMILY}" font-size="12" font-weight="700" fill="{color}">{label}</text>')
            legend_y += 24
            continue
        if item["kind"] == "gap":
            legend_y += 12
            continue
        method_id = str(item["method_id"])
        row = legend_methods[method_id]
        color = str(row["color"])
        dash = ' stroke-dasharray="7 5"' if line_style(row) == "--" else ""
        parts.append(f'<line x1="{legend_x1}" y1="{legend_y}" x2="{legend_x2}" y2="{legend_y}" stroke="{color}" stroke-width="1.6"{dash}/>')
        label = html.escape(str(item["label"]))
        parts.append(f'<text x="{legend_text_x}" y="{legend_y + 4}" font-family="{FONT_FAMILY}" font-size="12">{label}</text>')
        legend_y += 24

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = read_rows()
    write_plot_data(rows)
    if plt is None:
        render_svg(rows)
    else:
        render_with_matplotlib(rows)


if __name__ == "__main__":
    main()
