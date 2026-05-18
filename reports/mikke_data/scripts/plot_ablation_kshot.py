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
MEAN_SOURCE = ROOT / "data" / "main_ablation_mean_scores.csv"
CURATED_PATH = ROOT / "curated" / "ablation_kshot_results.csv"
PLOT_DATA_PATH = ROOT / "plots" / "ablation_kshot_strict_f1_data.csv"
PNG_PATH = ROOT / "plots" / "ablation_kshot_strict_f1.png"
PDF_PATH = ROOT / "plots" / "ablation_kshot_strict_f1.pdf"
SVG_PATH = ROOT / "plots" / "ablation_kshot_strict_f1.svg"
FONT_FAMILY = "Inter, Arial, sans-serif"

METHOD_ORDER = [
    "bert",
    "pat_perm_bart",
    "pat_perm_bart_ensemble_vote",
    "pat_perm_bart_ensemble_to_bert",
    "seed_perm_bart_ensemble_vote",
    "seed_perm_bart_ensemble_to_bert",
]

METHOD_COLORS = {
    "bert": "#4C78A8",
    "pat_perm_bart": "#F58518",
    "pat_perm_bart_ensemble_vote": "#B279A2",
    "pat_perm_bart_ensemble_to_bert": "#E45756",
    "seed_perm_bart_ensemble_vote": "#54A24B",
    "seed_perm_bart_ensemble_to_bert": "#72B7B2",
}

METHOD_STYLES = {
    "bert": "single",
    "pat_perm_bart": "single",
    "pat_perm_bart_ensemble_vote": "ensemble",
    "pat_perm_bart_ensemble_to_bert": "distilled",
    "seed_perm_bart_ensemble_vote": "ensemble",
    "seed_perm_bart_ensemble_to_bert": "distilled",
}


def display_label(label: str) -> str:
    return label.replace("->", "\u2192")


def read_mean_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with MEAN_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] != "mean":
                continue
            if not row["metric_value_percent"]:
                continue
            rows.append(
                {
                    **row,
                    "method_label": display_label(row["method_label"]),
                    "k_shot": int(row["k_shot"]),
                    "metric_value_percent": float(row["metric_value_percent"]),
                    "color": METHOD_COLORS.get(row["method_id"], "#666666"),
                }
            )
    return rows


def write_curated(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "row_type",
        "plot_group",
        "dataset",
        "task",
        "label_level",
        "tagging_scheme",
        "method_id",
        "method_label",
        "model",
        "k_shot",
        "split_seed",
        "eval_split",
        "metric_name",
        "metric_label",
        "metric_value",
        "metric_value_percent",
        "n_scores",
        "expected_scores",
        "completed_scores",
        "status",
        "score_source",
        "is_placeholder",
        "needs_more_runs",
        "source_path",
        "notes",
        "last_updated",
    ]
    with CURATED_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plot_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    method_rank = {method_id: index for index, method_id in enumerate(METHOD_ORDER)}
    output = []
    for row in sorted(rows, key=lambda item: (method_rank[str(item["method_id"])], int(item["k_shot"]))):
        output.append(
            {
                "figure_id": "ablation_kshot_strict_f1",
                "method_id": row["method_id"],
                "method_label": row["method_label"],
                "k_shot": row["k_shot"],
                "mean_strict_f1_percent": row["metric_value_percent"],
                "n_splits": row["n_scores"],
                "expected_scores": row["expected_scores"],
                "completed_scores": row["completed_scores"],
                "score_source": row["score_source"],
                "status": row["status"],
                "is_placeholder": row["is_placeholder"],
                "needs_more_runs": row["needs_more_runs"],
                "color": row["color"],
                "source_path": row["source_path"],
            }
        )
    return output


def write_plot_data(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "figure_id",
        "method_id",
        "method_label",
        "k_shot",
        "mean_strict_f1_percent",
        "n_splits",
        "expected_scores",
        "completed_scores",
        "score_source",
        "status",
        "is_placeholder",
        "needs_more_runs",
        "color",
        "source_path",
    ]
    with PLOT_DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def line_style(method_id: str, is_placeholder: bool) -> str:
    if is_placeholder or METHOD_STYLES.get(method_id) == "single":
        return "--"
    return "-"


def marker_size(method_id: str) -> float:
    if METHOD_STYLES.get(method_id) == "ensemble":
        return 4.0
    return 3.6


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def render_with_matplotlib(rows: list[dict[str, object]]) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    plt.rcParams["font.family"] = ["Inter", "Arial", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"

    fig, ax = plt.subplots(figsize=(6.9, 2.8), dpi=300)
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_id"])].append(row)

    for method_id in METHOD_ORDER:
        if method_id not in by_method:
            continue
        method_rows = by_method[method_id]
        method_rows = sorted(method_rows, key=lambda row: int(row["k_shot"]))
        placeholder = method_rows[0]["is_placeholder"] == "true"
        linestyle = line_style(method_id, placeholder)
        color = str(method_rows[0]["color"])
        markerface = "white" if placeholder else color
        linewidth = 1.15
        markersize = marker_size(method_id)
        x_values = [int(row["k_shot"]) for row in method_rows]
        y_values = [float(row["mean_strict_f1_percent"]) for row in method_rows]
        if len(method_rows) == 1:
            ax.scatter(
                x_values,
                y_values,
                marker="D",
                s=markersize * 10,
                color=color,
                facecolors=markerface,
                label=str(method_rows[0]["method_label"]),
            )
        else:
            plot_kwargs = {
                "marker": "o",
                "color": color,
                "markerfacecolor": markerface,
                "linestyle": linestyle,
                "linewidth": linewidth,
                "markersize": markersize,
                "label": str(method_rows[0]["method_label"]),
            }
            if METHOD_STYLES.get(method_id) == "ensemble":
                ax.plot(
                    x_values,
                    y_values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.8,
                    alpha=0.95,
                )
                ax.plot(
                    x_values,
                    y_values,
                    color="white",
                    linestyle=linestyle,
                    linewidth=1.3,
                    alpha=0.95,
                )
            ax.plot(
                x_values,
                y_values,
                **plot_kwargs,
            )

    ax.set_xticks([5, 10, 20, 50])
    ax.set_xticklabels(["k=5", "k=10", "k=20", "k=50"], fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Strict span F1", fontsize=14)
    ax.set_title("CoNLL2003 ablation by k-shot setting")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def render_svg(rows: list[dict[str, object]]) -> None:
    width = 1050
    height = 460
    left = 90
    right = 420
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
        "<title>CoNLL2003 ablation by k-shot setting</title>",
        "<style>text { font-family: Inter, Arial, sans-serif; }</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-family="{FONT_FAMILY}" font-size="18" font-weight="700">CoNLL2003 ablation by k-shot setting</text>',
        f'<text x="24" y="225" font-family="{FONT_FAMILY}" font-size="16" transform="rotate(-90 24 225)">Strict span F1</text>',
    ]

    for tick in [0, 25, 50, 75, 100]:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="52" y="{y + 4:.1f}" font-family="{FONT_FAMILY}" font-size="12">{tick}</text>')

    for k in k_values:
        x = x_lookup[k]
        parts.append(f'<text x="{x - 12:.1f}" y="410" font-family="{FONT_FAMILY}" font-size="12" font-weight="700">k={k}</text>')

    legend_y = top
    for method_id in METHOD_ORDER:
        if method_id not in by_method:
            continue
        method_rows = by_method[method_id]
        method_rows = sorted(method_rows, key=lambda row: int(row["k_shot"]))
        color = str(method_rows[0]["color"])
        placeholder = method_rows[0]["is_placeholder"] == "true"
        dash = ' stroke-dasharray="7 5"' if line_style(method_id, placeholder) == "--" else ""
        opacity = "0.62" if placeholder else "0.95"
        stroke_width = "1.6"
        dot_radius = marker_size(method_id)
        points = [
            (x_lookup[int(row["k_shot"])], y_pos(float(row["mean_strict_f1_percent"])))
            for row in method_rows
        ]
        if len(points) > 1:
            path = svg_path(points)
            if METHOD_STYLES.get(method_id) == "ensemble":
                parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4.2" stroke-opacity="{opacity}"{dash}/>')
                parts.append(f'<path d="{path}" fill="none" stroke="white" stroke-width="1.8" stroke-opacity="0.95"{dash}/>')
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-opacity="{opacity}"{dash}/>')
        for x, y in points:
            fill = "white" if placeholder else color
            if len(points) == 1:
                diamond = f"{x:.1f},{y - 5:.1f} {x + 5:.1f},{y:.1f} {x:.1f},{y + 5:.1f} {x - 5:.1f},{y:.1f}"
                parts.append(f'<polygon points="{diamond}" fill="{fill}" stroke="{color}" stroke-width="2"/>')
                if METHOD_STYLES.get(method_id) == "ensemble":
                    inner_diamond = f"{x:.1f},{y - 2.6:.1f} {x + 2.6:.1f},{y:.1f} {x:.1f},{y + 2.6:.1f} {x - 2.6:.1f},{y:.1f}"
                    parts.append(f'<polygon points="{inner_diamond}" fill="white" stroke="white" stroke-width="1"/>')
            else:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot_radius}" fill="{fill}" stroke="{color}" stroke-width="1.5"/>')
        label = html.escape(str(method_rows[0]["method_label"]))
        source_label = "validation" if "validation" in str(method_rows[0]["score_source"]) else "test"
        if placeholder:
            suffix = f" (placeholder {source_label})"
        elif len(points) == 1 and int(method_rows[0]["k_shot"]) == 5:
            suffix = " (k5 only)"
        elif method_rows[0]["needs_more_runs"] == "true" or len(points) < len(k_values):
            suffix = " (partial k)"
        else:
            suffix = ""
        if len(points) == 1:
            legend_x = left + plot_width + 50
            diamond = f"{legend_x},{legend_y - 5} {legend_x + 5},{legend_y} {legend_x},{legend_y + 5} {legend_x - 5},{legend_y}"
            parts.append(f'<polygon points="{diamond}" fill="white" stroke="{color}" stroke-width="2"/>')
        else:
            if METHOD_STYLES.get(method_id) == "ensemble":
                parts.append(f'<line x1="{left + plot_width + 35}" y1="{legend_y}" x2="{left + plot_width + 65}" y2="{legend_y}" stroke="{color}" stroke-width="4.2"{dash}/>')
                parts.append(f'<line x1="{left + plot_width + 35}" y1="{legend_y}" x2="{left + plot_width + 65}" y2="{legend_y}" stroke="white" stroke-width="1.8"{dash}/>')
            parts.append(f'<line x1="{left + plot_width + 35}" y1="{legend_y}" x2="{left + plot_width + 65}" y2="{legend_y}" stroke="{color}" stroke-width="{stroke_width}"{dash}/>')
        parts.append(f'<text x="{left + plot_width + 74}" y="{legend_y + 4}" font-family="{FONT_FAMILY}" font-size="12">{label}{suffix}</text>')
        legend_y += 24

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    curated_rows = read_mean_rows()
    method_rank = {method_id: index for index, method_id in enumerate(METHOD_ORDER)}
    curated_rows = sorted(curated_rows, key=lambda row: (method_rank[str(row["method_id"])], int(row["k_shot"])))
    write_curated(curated_rows)
    plot_rows = make_plot_rows(curated_rows)
    write_plot_data(plot_rows)
    if plt is None:
        render_svg(plot_rows)
    else:
        render_with_matplotlib(plot_rows)


if __name__ == "__main__":
    main()
