from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
FONT_FAMILY = "Inter, Arial, sans-serif"


@dataclass(frozen=True)
class PlotBundle:
    title: str
    source_path: Path
    plot_data_path: Path
    png_path: Path
    pdf_path: Path
    svg_path: Path
    k50_label_y: dict[str, float]


DEFAULT_BUNDLE = PlotBundle(
    title="Our results vs OADA",
    source_path=ROOT / "data" / "oada_comparison_scores.csv",
    plot_data_path=ROOT / "plots" / "oada_comparison_strict_f1_data.csv",
    png_path=ROOT / "plots" / "oada_comparison_strict_f1.png",
    pdf_path=ROOT / "plots" / "oada_comparison_strict_f1.pdf",
    svg_path=ROOT / "plots" / "oada_comparison_strict_f1.svg",
    k50_label_y={
        "ours_distilled": 80.0,
        "oada_best": 74.5,
        "ours_bert": 69.0,
        "oada_bert": 63.5,
    },
)

FIXED600_BUNDLE = PlotBundle(
    title="Our results vs OADA",
    source_path=ROOT / "data" / "oada_comparison_scores_fixed600.csv",
    plot_data_path=ROOT / "plots" / "oada_comparison_strict_f1_fixed600_data.csv",
    png_path=ROOT / "plots" / "oada_comparison_strict_f1_fixed600.png",
    pdf_path=ROOT / "plots" / "oada_comparison_strict_f1_fixed600.pdf",
    svg_path=ROOT / "plots" / "oada_comparison_strict_f1_fixed600.svg",
    k50_label_y={
        "ours_distilled": 80.0,
        "oada_best": 74.5,
        "ours_bert": 69.0,
        "oada_bert": 63.5,
    },
)

METHOD_ORDER = [
    "ours_bert",
    "oada_bert",
    "ours_distilled",
    "oada_best",
]

HERO_METHODS = ("ours_distilled", "oada_best")
BASELINE_METHODS = ("ours_bert", "oada_bert")
LEGEND_ITEM_HEIGHT = 15
LEGEND_GAP_HEIGHT = 5
LEGEND_PADDING = 10
LEGEND_RISE = 32
FIGURE_HEIGHT = 460
FIGURE_WIDTH = FIGURE_HEIGHT * 4 // 3
FIGURE_LEFT = 90
FIGURE_RIGHT = 48
FIGURE_TOP = 58
FIGURE_BOTTOM = 70
MARKER_RADIUS_OURS = 1.6
MARKER_RADIUS_OADA = 1.2
LEGEND_BOX_WIDTH = 252

COLORS = {
    "ours": "#7B61FF",
    "oada_paper": "#8F8F8F",
}

LEGEND_ITEMS = [
    {"kind": "heading", "label": "Ours", "color": COLORS["ours"]},
    {"kind": "method", "method_id": "ours_distilled", "label": "Pat+OADA BART \u2192 BERT"},
    {"kind": "method", "method_id": "ours_bert", "label": "BERT baseline"},
    {"kind": "gap"},
    {"kind": "heading", "label": "OADA best performer", "color": COLORS["oada_paper"]},
    {"kind": "method", "method_id": "oada_best", "label": "PromptNER+OADA"},
    {"kind": "method", "method_id": "oada_bert", "label": "BERT baseline"},
]


def read_rows(bundle: PlotBundle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with bundle.source_path.open(newline="", encoding="utf-8") as handle:
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


def write_plot_data(bundle: PlotBundle, rows: list[dict[str, object]]) -> None:
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
    with bundle.plot_data_path.open("w", newline="", encoding="utf-8") as handle:
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


def render_with_matplotlib(bundle: PlotBundle, rows: list[dict[str, object]]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    plt.rcParams["font.family"] = ["Inter", "Arial", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH / 100, FIGURE_HEIGHT / 100), dpi=300)
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
            markersize=MARKER_RADIUS_OADA * 2 if first["source"] == "oada_paper" else MARKER_RADIUS_OURS * 2,
            label=label_for_legend(method_id, str(first["method_label"])),
        )

    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(["k=5", "k=10", "k=20", "k=50"], fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(-4, 59)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Strict span F1", fontsize=14)
    ax.set_xlabel("Few-shot gold labels", fontsize=14)
    ax.set_title(bundle.title, loc="center")
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
    legend = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="lower right",
        frameon=True,
        framealpha=0.93,
        fontsize=8,
        labelspacing=0.35,
        handletextpad=0.6,
        borderpad=0.5,
    )
    for text, item in zip(legend.get_texts(), LEGEND_ITEMS):
        if item["kind"] == "heading":
            text.set_weight("bold")
            text.set_color(str(item["color"]))
    fig.tight_layout()
    fig.savefig(bundle.png_path, bbox_inches="tight")
    fig.savefig(bundle.pdf_path, bbox_inches="tight")
    plt.close(fig)


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def combined_hero_label(x: float, y_top: float, ours_val: float, oada_val: float) -> str:
    ours_text = html.escape(f"{ours_val:.2f}")
    oada_text = html.escape(f"{oada_val:.2f}")
    return (
        f'<text x="{x:.1f}" y="{y_top - 10:.1f}" font-family="{FONT_FAMILY}" font-size="10.5" text-anchor="middle">'
        f'<tspan fill="{COLORS["ours"]}">{ours_text}</tspan>'
        f'<tspan fill="#666666"> · </tspan>'
        f'<tspan fill="{COLORS["oada_paper"]}">{oada_text}</tspan>'
        f"</text>"
    )


def legend_block_height() -> float:
    height = 0.0
    for item in LEGEND_ITEMS:
        if item["kind"] == "gap":
            height += LEGEND_GAP_HEIGHT
        else:
            height += LEGEND_ITEM_HEIGHT
    return height


def render_svg(bundle: PlotBundle, rows: list[dict[str, object]]) -> None:
    width = FIGURE_WIDTH
    height = FIGURE_HEIGHT
    left = FIGURE_LEFT
    right = FIGURE_RIGHT
    top = FIGURE_TOP
    bottom = FIGURE_BOTTOM
    plot_width = width - left - right
    plot_height = height - top - bottom
    plot_bottom = top + plot_height
    x_tick_label_y = plot_bottom + 18
    x_axis_label_y = plot_bottom + 40
    y_axis_label_y = top + plot_height / 2
    k_values = [5, 10, 20, 50]
    left_inset_fraction = 0.2
    right_inset_fraction = 0.5
    tick_width = plot_width / (len(k_values) - 1 + left_inset_fraction + right_inset_fraction)
    x_lookup = {k: left + (index + left_inset_fraction) * tick_width for index, k in enumerate(k_values)}

    def y_pos(value: float) -> float:
        return top + plot_height - (value / 100) * plot_height

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)
    k50_label_y = {method_id: y_pos(value) for method_id, value in bundle.k50_label_y.items()}

    method_points: dict[str, list[tuple[dict[str, object], float, float]]] = {}
    legend_methods: dict[str, dict[str, object]] = {}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(bundle.title)}</title>",
        "<style>text { font-family: Inter, Arial, sans-serif; }</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_width / 2:.1f}" y="30" font-family="{FONT_FAMILY}" font-size="18" font-weight="700" text-anchor="middle">{html.escape(bundle.title)}</text>',
        f'<text x="24" y="{y_axis_label_y:.1f}" font-family="{FONT_FAMILY}" font-size="16" '
        f'transform="rotate(-90 24 {y_axis_label_y:.1f})">Strict span F1</text>',
    ]

    for tick in [0, 25, 50, 75, 100]:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="72" y="{y + 5:.1f}" font-family="{FONT_FAMILY}" font-size="14" text-anchor="end">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#DDDDDD" stroke-width="1"/>')
    parts.append(f'<line x1="{left + plot_width}" y1="{top}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#DDDDDD" stroke-width="1"/>')

    for k in k_values:
        x = x_lookup[k]
        parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom - 8}" stroke="#111111" stroke-width="1.8"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{x_tick_label_y:.1f}" font-family="{FONT_FAMILY}" font-size="14" '
            f'font-weight="700" text-anchor="middle">k={k}</text>'
        )
    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{x_axis_label_y:.1f}" font-family="{FONT_FAMILY}" '
        f'font-size="16" text-anchor="middle">Few-shot gold labels</text>'
    )

    for method_id in METHOD_ORDER:
        method_rows = sorted(by_method[method_id], key=lambda row: int(row["k_shot"]))
        first = method_rows[0]
        legend_methods[method_id] = first
        color = str(first["color"])
        dash = ' stroke-dasharray="7 5"' if line_style(first) == "--" else ""
        points = [
            (row, x_lookup[int(row["k_shot"])], y_pos(float(row["metric_value_percent"])))
            for row in method_rows
        ]
        method_points[method_id] = points
        path = svg_path([(x, y) for _, x, y in points])
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6" stroke-opacity="0.95"{dash}/>')
        dot_radius = MARKER_RADIUS_OADA if first["source"] == "oada_paper" else MARKER_RADIUS_OURS
        for _, x, y in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot_radius}" fill="{color}" stroke="{color}" stroke-width="1"/>'
            )

    ours_hero = {int(row["k_shot"]): (x, y, float(row["metric_value_percent"])) for row, x, y in method_points["ours_distilled"]}
    oada_hero = {int(row["k_shot"]): (x, y, float(row["metric_value_percent"])) for row, x, y in method_points["oada_best"]}
    for k in k_values:
        ox, oy, ours_val = ours_hero[k]
        _, oy_oada, oada_val = oada_hero[k]
        y_top = min(oy, oy_oada)
        parts.append(combined_hero_label(ox, y_top, ours_val, oada_val))

    k50_baseline_labels: dict[str, dict[str, object]] = {}
    for method_id in BASELINE_METHODS:
        method_rows = sorted(by_method[method_id], key=lambda row: int(row["k_shot"]))
        first = method_rows[0]
        value_offset = -10 if first["source"] == "ours" else 18
        for row, x, y in method_points[method_id]:
            value_label = html.escape(f'{float(row["metric_value_percent"]):.2f}')
            if int(row["k_shot"]) == 50:
                k50_baseline_labels[method_id] = {
                    "label": value_label,
                    "x": x + 18,
                    "color": str(first["color"]),
                }
                continue
            parts.append(
                f'<text x="{x:.1f}" y="{y + value_offset:.1f}" font-family="{FONT_FAMILY}" font-size="10.5" '
                f'fill="{first["color"]}" text-anchor="middle">{value_label}</text>'
            )

    for method_id in BASELINE_METHODS:
        label = k50_baseline_labels[method_id]
        parts.append(
            f'<text x="{float(label["x"]):.1f}" y="{k50_label_y[method_id]:.1f}" font-family="{FONT_FAMILY}" '
            f'font-size="10.5" fill="{label["color"]}" text-anchor="start">{label["label"]}</text>'
        )

    legend_height = legend_block_height()
    legend_box_width = LEGEND_BOX_WIDTH
    legend_box_x = left + plot_width - legend_box_width - LEGEND_PADDING
    legend_box_y = top + plot_height - legend_height - LEGEND_PADDING - LEGEND_RISE
    legend_x1 = legend_box_x + 12
    legend_x2 = legend_x1 + 30
    legend_text_x = legend_x2 + 10
    parts.append(
        f'<rect x="{legend_box_x:.1f}" y="{legend_box_y:.1f}" width="{legend_box_width:.1f}" '
        f'height="{legend_height + 8:.1f}" fill="white" fill-opacity="0.93" stroke="#E6E6E6" stroke-width="1"/>'
    )

    legend_y = legend_box_y + 12
    for item in LEGEND_ITEMS:
        if item["kind"] == "heading":
            label = html.escape(str(item["label"]))
            color = str(item["color"])
            parts.append(
                f'<text x="{legend_x1}" y="{legend_y + 4}" font-family="{FONT_FAMILY}" font-size="11" '
                f'font-weight="700" fill="{color}">{label}</text>'
            )
            legend_y += LEGEND_ITEM_HEIGHT
            continue
        if item["kind"] == "gap":
            legend_y += LEGEND_GAP_HEIGHT
            continue
        method_id = str(item["method_id"])
        row = legend_methods[method_id]
        color = str(row["color"])
        dash = ' stroke-dasharray="7 5"' if line_style(row) == "--" else ""
        parts.append(
            f'<line x1="{legend_x1}" y1="{legend_y}" x2="{legend_x2}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="1.6"{dash}/>'
        )
        label = html.escape(str(item["label"]))
        parts.append(
            f'<text x="{legend_text_x}" y="{legend_y + 4}" font-family="{FONT_FAMILY}" font-size="11">{label}</text>'
        )
        legend_y += LEGEND_ITEM_HEIGHT

    parts.append("</svg>")
    bundle.svg_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot our results vs OADA.")
    parser.add_argument(
        "--fixed600",
        action="store_true",
        help="Use fixed-600-step BERT scores and write *_fixed600 plot outputs.",
    )
    return parser.parse_args()


def main(bundle: PlotBundle = DEFAULT_BUNDLE) -> None:
    rows = read_rows(bundle)
    write_plot_data(bundle, rows)
    render_svg(bundle, rows)
    if plt is not None:
        render_with_matplotlib(bundle, rows)


if __name__ == "__main__":
    args = parse_args()
    main(FIXED600_BUNDLE if args.fixed600 else DEFAULT_BUNDLE)
