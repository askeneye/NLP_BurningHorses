# MIKKE Report Data

This folder is the report-facing handover area for data, plot inputs, and figure outputs used in the MIKKE report.

The raw experiment outputs can stay where the training and evaluation scripts write them. This folder should contain curated, clearly labeled data that is ready to be inspected, plotted, or handed over to another agent.

## Workflow

1. When an experiment completes, compile its relevant scores into a curated CSV under `curated/`.
2. Mark each row with a clear status, source split, and whether it is final or placeholder data.
3. Use scripts under `scripts/` as the reproducible source of truth for report plots.
4. Export plot-ready CSVs and figures under `plots/`.
5. Update `plot_todo.md` when a plot moves from placeholder to final.

Plain Python scripts should be treated as the final plotting workflow. Notebooks are useful for exploration, but final report figures should be reproducible from scripts.

## Folder Layout

- `csv_templates/`: Empty CSV templates for new curated result tables and plot registry rows.
- `curated/`: Normalized CSV files compiled from completed or in-progress experiments.
- `plots/`: Report-ready plot data and rendered figures.
- `scripts/`: Reproducible Python scripts for compiling curated data and generating plots.
- `data_dictionary.md`: Field definitions and status semantics for curated CSVs.
- `styleguide.md`: Shared report language, labels, and visual conventions.
- `plot_todo.md`: Inventory of candidate plots and their readiness status.

## Data Status Rule

Never leave a score ambiguous. If a value comes from validation, pilot runs, incomplete seeds, or manual placeholder data, mark it explicitly in the curated CSV and in `plot_todo.md`.
