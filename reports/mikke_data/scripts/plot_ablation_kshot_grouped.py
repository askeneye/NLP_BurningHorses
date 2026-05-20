from __future__ import annotations

import csv
import html
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FONT_FAMILY = "Inter, Arial, sans-serif"
PLOT_BORDER_COLOR = "#111111"
Y_AXIS_MIN = 30
Y_AXIS_MAX = 80
Y_AXIS_TICKS = [30, 40, 50, 60, 70, 80]
MARKER_RADIUS = 1.6
HERO_MARKER_RADIUS = 2.0
LINE_WIDTH = 1.9
VOTE_LINE_WIDTH = 1.1
HERO_LINE_WIDTH = 2.2
OADA_ENSEMBLE_COLOR = "#54A24B"

K_VALUES = [5, 10, 20, 50]
SEEDS = [42, 142, 242]


@dataclass(frozen=True)
class PlotBundle:
    title: str
    mean_source: Path
    ensemble_tagging_dir: Path
    source_path: Path
    plot_data_path: Path
    svg_path: Path


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    family: str
    legend_label: str
    color: str
    style: str
    source: str = "mean_csv"


BUNDLE = PlotBundle(
    title="CoNLL2003 ablation by k-shot setting",
    mean_source=ROOT / "data" / "main_ablation_mean_scores_fixed600_final.csv",
    ensemble_tagging_dir=REPO_ROOT / "reports" / "ensemble_tagging",
    source_path=ROOT / "data" / "ablation_kshot_strict_f1_fixed600_grouped.csv",
    plot_data_path=ROOT / "plots" / "ablation_kshot_strict_f1_fixed600_grouped_data.csv",
    svg_path=ROOT / "plots" / "ablation_kshot_strict_f1_fixed600_grouped.svg",
)


METHOD_SPECS = [
    MethodSpec(
        "pat_perm_bart_ensemble_to_bert",
        "Pat+OADA BART ensemble",
        "Distilled -> BERT",
        "#7B61FF",
        "solid",
    ),
    MethodSpec(
        "pat_perm_bart_ensemble_vote",
        "Pat+OADA BART ensemble",
        "Majority vote",
        "#7B61FF",
        "thin",
    ),
    MethodSpec(
        "seed_perm_bart_ensemble_to_bert",
        "OADA BART ensemble",
        "Distilled -> BERT",
        OADA_ENSEMBLE_COLOR,
        "solid",
    ),
    MethodSpec(
        "seed_perm_bart_ensemble_vote",
        "OADA BART ensemble",
        "Majority vote",
        OADA_ENSEMBLE_COLOR,
        "thin",
    ),
    MethodSpec(
        "pat_bart_ensemble_to_bert",
        "Pat BART ensemble",
        "Distilled -> BERT",
        "#F58518",
        "solid",
    ),
    MethodSpec(
        "pat_bart_ensemble_vote",
        "Pat BART ensemble",
        "Majority vote",
        "#F58518",
        "thin",
        source="pat_bart_vote_json",
    ),
    MethodSpec(
        "pat_perm_bart_to_bert",
        "Single models",
        "OADA BART (single teacher) -> BERT",
        "#8F8F8F",
        "solid",
    ),
    MethodSpec(
        "pat_perm_bart",
        "Single models",
        "OADA BART (single teacher)",
        "#8F8F8F",
        "dashed",
    ),
    MethodSpec(
        "bert",
        "Single models",
        "BERT baseline",
        "#8F8F8F",
        "dotted",
    ),
]

SPEC_BY_METHOD = {spec.method_id: spec for spec in METHOD_SPECS}


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def read_mean_csv_rows(bundle: PlotBundle) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with bundle.mean_source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method_id = row["method_id"]
            if row["row_type"] != "mean" or method_id not in SPEC_BY_METHOD:
                continue
            if not row["metric_value_percent"]:
                continue
            spec = SPEC_BY_METHOD[method_id]
            k_shot = int(row["k_shot"])
            rows[(method_id, k_shot)] = {
                "method_id": method_id,
                "family": spec.family,
                "legend_label": spec.legend_label,
                "k_shot": k_shot,
                "metric_value": float(row["metric_value"]),
                "metric_value_percent": float(row["metric_value_percent"]),
                "score_source": row["score_source"],
                "source_path": row["source_path"],
                "color": spec.color,
                "style": spec.style,
            }
    return rows


def read_pat_bart_vote_rows(bundle: PlotBundle) -> list[dict[str, Any]]:
    spec = SPEC_BY_METHOD["pat_bart_ensemble_vote"]
    output: list[dict[str, Any]] = []
    for k_shot in K_VALUES:
        values: list[float] = []
        source_paths: list[str] = []
        for seed in SEEDS:
            path = bundle.ensemble_tagging_dir / (
                f"k{k_shot}_seed{seed}_pat_no_perm_verbalized_final2000_test_metrics.json"
            )
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            values.append(float(payload["metrics"]["Span_Strict_F1"]))
            source_paths.append(repo_relative(path))
        mean_value = statistics.mean(values)
        output.append(
            {
                "method_id": spec.method_id,
                "family": spec.family,
                "legend_label": spec.legend_label,
                "k_shot": k_shot,
                "metric_value": mean_value,
                "metric_value_percent": mean_value * 100,
                "score_source": "test",
                "source_path": ";".join(source_paths),
                "color": spec.color,
                "style": spec.style,
            }
        )
    return output


def build_rows(bundle: PlotBundle) -> list[dict[str, Any]]:
    mean_rows = read_mean_csv_rows(bundle)
    output: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        if spec.source == "pat_bart_vote_json":
            output.extend(read_pat_bart_vote_rows(bundle))
            continue
        for k_shot in K_VALUES:
            output.append(mean_rows[(spec.method_id, k_shot)])
    method_rank = {spec.method_id: index for index, spec in enumerate(METHOD_SPECS)}
    return sorted(output, key=lambda row: (method_rank[row["method_id"]], int(row["k_shot"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method_id",
        "family",
        "legend_label",
        "k_shot",
        "metric_value",
        "metric_value_percent",
        "score_source",
        "source_path",
        "color",
        "style",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dash_for_style(style: str) -> str:
    if style == "dotted":
        return ' stroke-dasharray="1 5" stroke-linecap="round"'
    if style == "dashed":
        return ' stroke-dasharray="7 5"'
    return ""


def stroke_width_for_spec(spec: MethodSpec) -> str:
    if spec.method_id == "pat_perm_bart_ensemble_to_bert":
        return f"{HERO_LINE_WIDTH:.1f}"
    if spec.style == "thin":
        return f"{VOTE_LINE_WIDTH:.1f}"
    return f"{LINE_WIDTH:.1f}"


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))


def render_svg(bundle: PlotBundle, rows: list[dict[str, Any]]) -> None:
    width = 1160
    height = 560
    left = 90
    right = 440
    top = 58
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    plot_bottom = top + plot_height
    edge_inset_fraction = 0.2
    tick_width = plot_width / (len(K_VALUES) - 1 + 2 * edge_inset_fraction)
    x_lookup = {k: left + (index + edge_inset_fraction) * tick_width for index, k in enumerate(K_VALUES)}

    def y_pos(value: float) -> float:
        span = Y_AXIS_MAX - Y_AXIS_MIN
        normalized = (value - Y_AXIS_MIN) / span
        return top + plot_height - normalized * plot_height

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method_id"]].append(row)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(bundle.title)}</title>",
        "<style>text { font-family: Inter, Arial, sans-serif; }</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_width / 2:.1f}" y="30" font-family="{FONT_FAMILY}" font-size="18" font-weight="700" text-anchor="middle">{html.escape(bundle.title)}</text>',
        f'<text x="24" y="{top + plot_height / 2:.1f}" font-family="{FONT_FAMILY}" font-size="16" transform="rotate(-90 24 {top + plot_height / 2:.1f})">Strict span F1</text>',
    ]

    for tick in Y_AXIS_TICKS:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#DDDDDD" stroke-width="1"/>')
        parts.append(f'<text x="72" y="{y + 5:.1f}" font-family="{FONT_FAMILY}" font-size="14" text-anchor="end">{tick}</text>')

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        f'fill="none" stroke="{PLOT_BORDER_COLOR}" stroke-width="1.5"/>'
    )

    for k_shot in K_VALUES:
        x = x_lookup[k_shot]
        parts.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom - 8}" stroke="#111111" stroke-width="1.8"/>')
        parts.append(f'<text x="{x:.1f}" y="{plot_bottom + 20:.1f}" font-family="{FONT_FAMILY}" font-size="14" font-weight="700" text-anchor="middle">k={k_shot}</text>')

    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{plot_bottom + 48:.1f}" font-family="{FONT_FAMILY}" font-size="16" text-anchor="middle">Few-shot gold labels</text>'
    )

    for spec in METHOD_SPECS:
        method_rows = sorted(by_method[spec.method_id], key=lambda row: int(row["k_shot"]))
        points = [
            (x_lookup[int(row["k_shot"])], y_pos(float(row["metric_value_percent"])))
            for row in method_rows
        ]
        dash = dash_for_style(spec.style)
        path = svg_path(points)
        stroke_width = stroke_width_for_spec(spec)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{spec.color}" stroke-width="{stroke_width}" stroke-opacity="0.95"{dash}/>'
        )
        for x, y in points:
            radius = (
                HERO_MARKER_RADIUS
                if spec.method_id == "pat_perm_bart_ensemble_to_bert"
                else MARKER_RADIUS
            )
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{spec.color}" stroke="{spec.color}" stroke-width="0.8"/>'
            )

    legend_x = left + plot_width + 38
    legend_y = top + 8
    legend_line_x1 = legend_x
    legend_line_x2 = legend_x + 34
    legend_text_x = legend_line_x2 + 12
    current_family = ""
    for spec in METHOD_SPECS:
        if spec.family != current_family:
            if current_family:
                legend_y += 12
            current_family = spec.family
            parts.append(
                f'<text x="{legend_x}" y="{legend_y:.1f}" font-family="{FONT_FAMILY}" font-size="12" font-weight="700">{html.escape(spec.family)}</text>'
            )
            legend_y += 18
        dash = dash_for_style(spec.style)
        stroke_width = stroke_width_for_spec(spec)
        parts.append(
            f'<line x1="{legend_line_x1}" y1="{legend_y:.1f}" x2="{legend_line_x2}" y2="{legend_y:.1f}" stroke="{spec.color}" stroke-width="{stroke_width}"{dash}/>'
        )
        parts.append(
            f'<text x="{legend_text_x}" y="{legend_y + 4:.1f}" font-family="{FONT_FAMILY}" font-size="12">{html.escape(spec.legend_label)}</text>'
        )
        legend_y += 18

    parts.append("</svg>")
    bundle.svg_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.svg_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = build_rows(BUNDLE)
    write_csv(BUNDLE.source_path, rows)
    write_csv(BUNDLE.plot_data_path, rows)
    render_svg(BUNDLE, rows)
    print(f"Wrote {BUNDLE.svg_path}")
    print(f"Wrote {BUNDLE.plot_data_path}")


if __name__ == "__main__":
    main()
