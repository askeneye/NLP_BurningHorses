# Slave Machine Todo

Purpose: offload independent work from the main machine. Assume the slave has faster CPU but less VRAM. Prefer CPU-heavy JSONL/statistics work and smaller BERT jobs. Avoid long BART training unless VRAM has been tested.

## Setup Checks

Run these first after syncing the repo and artifacts:

```powershell
git status --short
python --version
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi
```

Report these back before taking large jobs:

- OS and Python version.
- CPU model and core/thread count.
- RAM amount.
- GPU model, VRAM amount, driver/CUDA version.
- Whether `torch.cuda.is_available()` is true.
- Free disk space on the data/model drive.
- Whether the repo venv imports `torch`, `transformers`, `datasets`, and `seqeval`.

Quick hardware command options:

```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-PSDrive C
nvidia-smi
python -c "import torch, transformers, datasets; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('transformers', transformers.__version__); print('datasets', datasets.__version__)"
```

Required artifact roots to copy/sync:

- `data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000/`
- `data/interim/conll2003_soft_labels/bart_base/pet_oada_verbalized/`
- `data/interim/conll2003_kshot_bert/`
- `models/conll2003_kshot_seq2seq/bart_base/pet_oada_verbalized/` only if running BART inference or diagnostics.

## Performance Smoke Report

Run a tiny timing job before accepting a large batch. Report wall-clock time, peak VRAM if visible in `nvidia-smi`, and whether the run completed without CPU/GPU memory pressure.

### BERT Smoke Test

Use a short run on one small teacher file:

```powershell
$env:DISTILL_SPLIT="k5_seed42"
$env:DISTILL_TEACHER_SPLIT="smoke"
$env:DISTILL_TEACHER_VARIANT="hard_argmax"
$env:DISTILL_TRAIN_FILE="data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000/k5_seed42/unlabeled_pool/ensemble_hard_argmax.jsonl"
$env:DISTILL_OUTPUT_DIR="models/_slave_smoke/bert_k5_seed42_hard_argmax_100steps"
$env:DISTILL_MAX_STEPS="100"
$env:DISTILL_EVAL_EVERY="100"
$env:DISTILL_PATIENCE="2"
Measure-Command { python nlp_burninghorses/bert_conll_distill.py }
```

Report:

- elapsed time for 100 steps,
- approximate seconds per step,
- peak VRAM,
- whether batch size 16 worked or needed batch size 8.

### BART Inference Smoke Test

Only run if BART artifacts were copied:

```powershell
$env:SOFT_LABEL_MODEL_DIR="models/conll2003_kshot_seq2seq/bart_base/pet_oada_verbalized/k5_seed42/pattern_01_xe_verbalized_final2000"
$env:SOFT_LABEL_CHECKPOINT_DIR="$env:SOFT_LABEL_MODEL_DIR/final_step"
$env:SOFT_LABEL_INPUT_FILE="data/interim/conll2003_kshot_bert/k5_seed42/unlabeled_pool.jsonl"
$env:SOFT_LABEL_OUTPUT_DIR="data/interim/_slave_smoke/hard_labels/k5_seed42/pattern_01"
$env:SOFT_LABEL_HARD_ONLY="1"
$env:SOFT_LABEL_BATCH_SIZE="8"
$env:SOFT_LABEL_MAX_GENERATION_LENGTH="128"
Measure-Command { python nlp_burninghorses/MIKKE/scripts/generate_bart_member_soft_labels.py }
```

Stop this after confirming it starts and reports stable progress, unless a full smoke file is desired. Report rows/second or time for the first few hundred rows.

## Best Slave Tasks

### 1. Agreement-Filtered Teacher Subsets

Goal: create smaller distilled-BERT training files from the full verbalized ensemble teachers.

Inputs:

- `ensemble_hard_argmax.jsonl`
- `ensemble_vote_normalized.jsonl`
- `ensemble_vote_temp2.jsonl`

Suggested subset sizes:

- `top_500`
- `top_1000`
- `top_2000`
- `top_5000`

Selection policy:

- rank examples by mean vote confidence over non-padding tokens,
- prefer examples with at least one predicted entity,
- keep a small controlled fraction of all-`O` examples only if needed,
- preserve original row order inside the final subset after selection.

Output convention:

```text
data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000_filtered/{split}/unlabeled_pool/{subset_name}/ensemble_hard_argmax.jsonl
```

### 2. BERT Distillation Subset Ablations

Goal: test whether filtered teacher sets beat the full 14k unlabeled pool.

Start with:

- `k5_seed42`
- `hard_argmax`
- `top_500`, `top_1000`, `top_2000`, full pool

Command template:

```powershell
$env:DISTILL_SPLIT="k5_seed42"
$env:DISTILL_TEACHER_SPLIT="unlabeled_pool_filtered_top1000"
$env:DISTILL_TEACHER_VARIANT="hard_argmax"
$env:DISTILL_TRAIN_FILE="data/interim/conll2003_ensemble_teachers/bart_base/pet_oada_verbalized_final2000_filtered/k5_seed42/unlabeled_pool/top_1000/ensemble_hard_argmax.jsonl"
$env:DISTILL_OUTPUT_DIR="models/bert_conll_distilled/k5_seed42/verbalized_final2000_filtered/top_1000_hard_argmax"
$env:DISTILL_MAX_STEPS="1000"
python nlp_burninghorses/bert_conll_distill.py
```

If VRAM is tight, reduce batch size:

```powershell
$env:DISTILL_TRAIN_BATCH_SIZE="8"
$env:DISTILL_EVAL_BATCH_SIZE="8"
```

### 3. Ensemble Diversity Diagnostics

Goal: quantify whether PET-pattern ensembles are meaningfully diverse.

Compute per split:

- token-level vote entropy,
- mean vote confidence,
- pairwise member disagreement,
- percentage of examples with unanimous predictions,
- span-level pairwise F1/Jaccard,
- entity-type confusion/disagreement counts.

Useful comparisons:

- verbalized PET/OADA ensemble vs old-label PET/OADA ensemble,
- `k5_seed42` vs `k10_seed42`,
- low-k vs high-k,
- PET-pattern ensemble vs future seed-only ensemble.

Output convention:

```text
reports/ensemble_diversity/{method}/{split}_unlabeled_pool_summary.json
reports/ensemble_diversity/{method}/{split}_unlabeled_pool_examples.jsonl
```

### 4. Result Extraction Tables

Goal: collect completed run summaries into CSV/JSON tables for report writing.

Inputs:

- `models/**/training_summary.json`
- `data/interim/**/manifest.json`

Outputs:

```text
reports/tables/bart_verbalized_vs_old_final2000.csv
reports/tables/distilled_bert_teacher_comparison.csv
reports/tables/teacher_artifact_inventory.csv
```

## Lower Priority Slave Tasks

### BART Hard-Label Inference

Only run this if the slave GPU can load `facebook/bart-base` comfortably. Use smaller batches if VRAM is limited:

```powershell
$env:SOFT_LABEL_BATCH_SIZE="8"
$env:SOFT_LABEL_HARD_ONLY="1"
```

Prefer inference over training on the slave. BART training should stay on the main GPU unless a smoke test confirms stable VRAM.

### Verbalizer Mini-Study

Possible future mini-study: train small pilots with alternative verbalizer maps.

Examples:

- `PER=person,LOC=location,ORG=organization,MISC=other`
- `PER=human,LOC=place,ORG=company,MISC=other`
- `PER=name,LOC=place,ORG=group,MISC=miscellaneous`

Use only a small split first, for example `k5_seed42 pattern_01`, before scaling.

## Do Not Prioritize On Slave

- Full BART ensemble training unless VRAM is verified.
- Verbalized true-soft scaling. This is currently shelved.
- New Few-NERD experiments before CoNLL distillation/filtering conclusions are clearer.

