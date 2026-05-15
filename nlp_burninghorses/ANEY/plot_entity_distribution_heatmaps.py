from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_ROOT = Path("results/Training_data_entity_distributions")

CONLL_LONG = RESULTS_ROOT / "conll2003_train_entity_distribution_long.csv"
FEWNERD_LONG = RESULTS_ROOT / "fewnerd_train_entity_distribution_long.csv"


def plot_heatmap(
    csv_path: Path,
    output_path: Path,
    title: str,
    figsize: tuple[int, int],
) -> None:
    df = pd.read_csv(csv_path)

    # Average over seeds/splits for each K and entity type
    avg_df = (
        df.groupby(["entity_type", "k"], as_index=False)["mention_count"]
        .mean()
    )

    heatmap_df = avg_df.pivot(
        index="entity_type",
        columns="k",
        values="mention_count",
    )

    heatmap_df = heatmap_df.sort_index()
    heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        heatmap_df.values,
        aspect="auto",
    )

    ax.set_title(title)
    ax.set_xlabel("K-shot")
    ax.set_ylabel("Entity type")

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns)

    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average entity mentions")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    plot_heatmap(
        CONLL_LONG,
        RESULTS_ROOT / "conll2003_entity_distribution_heatmap.png",
        "CoNLL-2003 entity distribution averaged over seeds",
        figsize=(7, 4),
    )

    plot_heatmap(
        FEWNERD_LONG,
        RESULTS_ROOT / "fewnerd_entity_distribution_heatmap.png",
        "Few-NERD fine-grained entity distribution averaged over seeds",
        figsize=(8, 14),
    )


if __name__ == "__main__":
    main()