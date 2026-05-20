from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FONT_FAMILY = "Inter, Arial, sans-serif"

FIGURE_HEIGHT = 460
FIGURE_WIDTH = FIGURE_HEIGHT * 4 // 3
FIGURE_LEFT = 90
FIGURE_RIGHT = 48
FIGURE_TOP = 58
FIGURE_BOTTOM = 70
MARKER_RADIUS = 3.0
LEGEND_BOX_WIDTH = 322
LEGEND_ITEM_HEIGHT = 15
LEGEND_PADDING = 10
LEGEND_RISE = 32
LEGEND_FONT_SIZE = 13
Y_AXIS_MIN = 30
Y_AXIS_MAX = 80
Y_AXIS_TICKS = [30, 40, 50, 60, 70, 80]


@dataclass(frozen=True)
class PlotBundle:
    title: str
    mean_source: Path
    ensemble_tagging_dir: Path
    source_path: Path
    plot_data_path: Path
    png_path: Path
    pdf_path: Path
    svg_path: Path


DEFAULT_BUNDLE = PlotBundle(
    title="Main ablation ladder",
    mean_source=ROOT / "data" / "main_ablation_mean_scores_fixed600_final.csv",
    ensemble_tagging_dir=REPO_ROOT / "reports" / "ensemble_tagging",
    source_path=ROOT / "data" / "ablation_ladder_scores.csv",
    plot_data_path=ROOT / "plots" / "ablation_ladder_strict_f1_data.csv",
    png_path=ROOT / "plots" / "ablation_ladder_strict_f1.png",
    pdf_path=ROOT / "plots" / "ablation_ladder_strict_f1.pdf",
    svg_path=ROOT / "plots" / "ablation_ladder_strict_f1.svg",
)

METHOD_ORDER = [
    "bert",
    "pat_perm_bart",
    "pat_perm_bart_to_bert",
    "pat_perm_bart_ensemble_to_bert",
]

METHOD_COLORS = {
    "bert": "#8F8F8F",
    "pat_perm_bart": "#F58518",
    "pat_perm_bart_to_bert": "#54A24B",
    "pat_perm_bart_ensemble_to_bert": "#7B61FF",
}

METHOD_STYLES = {
    "bert": "baseline",
    "pat_perm_bart": "teacher",
    "pat_perm_bart_to_bert": "confidence",
    "pat_perm_bart_ensemble_to_bert": "agreement",
}

LEGEND_LABELS = {
    "bert": "BERT",
    "pat_perm_bart": "OADA BART single teacher",
    "pat_perm_bart_to_bert": "OADA BART single teacher confidence \u2192 BERT",
    "pat_perm_bart_ensemble_to_bert": "OADA BART ensemble agreement \u2192 BERT",
}

MEAN_METHOD_IDS = {
    "bert",
    "pat_perm_bart",
    "pat_perm_bart_to_bert",
    "pat_perm_bart_ensemble_to_bert",
}


def read_mean_rows(bundle: PlotBundle) -> dict[tuple[str, int], dict[str, object]]:
    rows: dict[tuple[str, int], dict[str, object]] = {}
    with bundle.mean_source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] != "mean":
                continue
            if row["method_id"] not in MEAN_METHOD_IDS:
                continue
            key = (row["method_id"], int(row["k_shot"]))
            rows[key] = {
                "method_id": row["method_id"],
                "method_label": LEGEND_LABELS[row["method_id"]],
                "k_shot": int(row["k_shot"]),
                "metric_value": float(row["metric_value"]),
                "metric_value_percent": float(row["metric_value_percent"]),
                "score_source": row["score_source"],
                "source_path": row["source_path"],
                "color": METHOD_COLORS[row["method_id"]],
            }
    return rows


def read_pat_bart_ensemble_vote_rows(bundle: PlotBundle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k_shot in (5, 10, 20, 50):
        values: list[float] = []
        source_paths: list[str] = []
        for seed in (42, 142, 242):
            path = bundle.ensemble_tagging_dir / (
                f"k{k_shot}_seed{seed}_pat_no_perm_verbalized_final2000_test_metrics.json"
            )
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            values.append(float(payload["metrics"]["Span_Strict_F1"]))
            source_paths.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
        mean_value = statistics.mean(values)
        rows.append(
            {
                "method_id": "pat_bart_ensemble_vote",
                "method_label": LEGEND_LABELS["pat_bart_ensemble_vote"],
                "k_shot": k_shot,
                "metric_value": mean_value,
                "metric_value_percent": mean_value * 100,
                "score_source": "test",
                "source_path": ";".join(source_paths),
                "color": METHOD_COLORS["pat_bart_ensemble_vote"],
            }
        )
    return rows


def build_rows(bundle: PlotBundle) -> list[dict[str, object]]:
    mean_rows = read_mean_rows(bundle)
    output: list[dict[str, object]] = []
    for method_id in METHOD_ORDER:
        if method_id in MEAN_METHOD_IDS:
            for k_shot in (5, 10, 20, 50):
                output.append(mean_rows[(method_id, k_shot)])
        elif method_id == "pat_bart_ensemble_vote":
            output.extend(read_pat_bart_ensemble_vote_rows(bundle))
    method_rank = {method_id: index for index, method_id in enumerate(METHOD_ORDER)}
    return sorted(output, key=lambda row: (method_rank[str(row["method_id"])], int(row["k_shot"])))


def write_source_csv(bundle: PlotBundle, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method_id",
        "method_label",
        "k_shot",
        "metric_value",
        "metric_value_percent",
        "score_source",
        "source_path",
        "color",
    ]
    with bundle.source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot_data(bundle: PlotBundle, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method_id",
        "method_label",
        "k_shot",
        "metric_value_percent",
        "score_source",
        "source_path",
        "color",
    ]
    with bundle.plot_data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def line_style(method_id: str) -> str:
    if METHOD_STYLES.get(method_id) == "baseline":
        return "--"
    return "-"


def svg_dash(method_id: str) -> str:
    style = line_style(method_id)
    if style == "--":
        return ' stroke-dasharray="7 5"'
    return ""


def stroke_width(method_id: str) -> str:
    if METHOD_STYLES.get(method_id) == "agreement":
        return "2.2"
    if METHOD_STYLES.get(method_id) == "confidence":
        return "1.9"
    return "1.7"


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


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
    right_inset_fraction = 0.2
    tick_width = plot_width / (len(k_values) - 1 + left_inset_fraction + right_inset_fraction)
    x_lookup = {k: left + (index + left_inset_fraction) * tick_width for index, k in enumerate(k_values)}

    def y_pos(value: float) -> float:
        span = Y_AXIS_MAX - Y_AXIS_MIN
        normalized = (value - Y_AXIS_MIN) / span
        return top + plot_height - normalized * plot_height

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)

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

    for tick in Y_AXIS_TICKS:
        y = y_pos(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="72" y="{y + 5:.1f}" font-family="{FONT_FAMILY}" font-size="14" text-anchor="end">{tick}</text>'
        )
    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        f'fill="none" stroke="#111111" stroke-width="1.5"/>'
    )

    for k in k_values:
        x = x_lookup[k]
        parts.append(
            f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom - 8}" stroke="#111111" stroke-width="1.8"/>'
        )
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
        dash = svg_dash(method_id)
        points = [
            (x_lookup[int(row["k_shot"])], y_pos(float(row["metric_value_percent"])))
            for row in method_rows
        ]
        path = svg_path(points)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width(method_id)}" '
            f'stroke-opacity="0.95"{dash}/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{MARKER_RADIUS}" fill="{color}" stroke="{color}" stroke-width="1"/>'
            )
        if METHOD_STYLES.get(method_id) == "agreement":
            for row, (x, y) in zip(method_rows, points):
                score = f'{float(row["metric_value_percent"]):.1f}'
                parts.append(
                    f'<text x="{x:.1f}" y="{y - 12:.1f}" font-family="{FONT_FAMILY}" font-size="11" '
                    f'font-weight="700" text-anchor="middle" fill="{color}">{score}</text>'
                )

    legend_height = len(METHOD_ORDER) * LEGEND_ITEM_HEIGHT
    legend_box_x = left + plot_width - LEGEND_BOX_WIDTH - LEGEND_PADDING
    legend_box_y = top + plot_height - legend_height - LEGEND_PADDING - LEGEND_RISE
    legend_x1 = legend_box_x + 12
    legend_x2 = legend_x1 + 30
    legend_text_x = legend_x2 + 10
    parts.append(
        f'<rect x="{legend_box_x:.1f}" y="{legend_box_y:.1f}" width="{LEGEND_BOX_WIDTH:.1f}" '
        f'height="{legend_height + 8:.1f}" fill="white" fill-opacity="0.93" stroke="#E6E6E6" stroke-width="1"/>'
    )

    legend_y = legend_box_y + 12
    for method_id in reversed(METHOD_ORDER):
        row = legend_methods[method_id]
        color = str(row["color"])
        dash = svg_dash(method_id)
        parts.append(
            f'<line x1="{legend_x1}" y1="{legend_y}" x2="{legend_x2}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="{stroke_width(method_id)}"{dash}/>'
        )
        label = html.escape(LEGEND_LABELS[method_id])
        parts.append(
            f'<text x="{legend_text_x}" y="{legend_y + 4}" font-family="{FONT_FAMILY}" '
            f'font-size="{LEGEND_FONT_SIZE}">{label}</text>'
        )
        legend_y += LEGEND_ITEM_HEIGHT

    parts.append("</svg>")
    bundle.svg_path.write_text("\n".join(parts), encoding="utf-8")


def render_with_matplotlib(bundle: PlotBundle, rows: list[dict[str, object]]) -> None:
    if plt is None:
        return

    plt.rcParams["font.family"] = ["Inter", "Arial", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH / 100, FIGURE_HEIGHT / 100), dpi=300)
    for method_id in METHOD_ORDER:
        method_rows = sorted(by_method[method_id], key=lambda row: int(row["k_shot"]))
        first = method_rows[0]
        x_values = [int(row["k_shot"]) for row in method_rows]
        y_values = [float(row["metric_value_percent"]) for row in method_rows]
        ax.plot(
            x_values,
            y_values,
            color=str(first["color"]),
            linestyle=line_style(method_id),
            linewidth=float(stroke_width(method_id)),
            marker="o",
            markersize=MARKER_RADIUS * 2,
            label=LEGEND_LABELS[method_id],
        )
        if METHOD_STYLES.get(method_id) == "agreement":
            for x_value, y_value in zip(x_values, y_values):
                ax.text(
                    x_value,
                    y_value + 1.8,
                    f"{y_value:.1f}",
                    color=str(first["color"]),
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(["k=5", "k=10", "k=20", "k=50"], fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(-4, 59)
    ax.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    ax.set_yticks(Y_AXIS_TICKS)
    ax.set_ylabel("Strict span F1", fontsize=14)
    ax.set_xlabel("Few-shot gold labels", fontsize=14)
    ax.set_title(bundle.title, loc="center")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="lower right",
        frameon=True,
        framealpha=0.93,
        fontsize=10,
        labelspacing=0.35,
        handletextpad=0.6,
        borderpad=0.5,
    )
    fig.tight_layout()
    fig.savefig(bundle.png_path, bbox_inches="tight")
    fig.savefig(bundle.pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the ablation ladder subset.")
    return parser.parse_args()


def main(bundle: PlotBundle = DEFAULT_BUNDLE) -> None:
    rows = build_rows(bundle)
    write_source_csv(bundle, rows)
    write_plot_data(bundle, rows)
    render_svg(bundle, rows)
    render_with_matplotlib(bundle, rows)


if __name__ == "__main__":
    main()
