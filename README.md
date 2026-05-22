# Fire Horses

All generated artifacts are written to `reproduction/`, so you can delete that folder and rerun from a clean state.

## 1) Environment setup

From repository root:

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your shell does not resolve `python` after activation, use `python3` for the same commands.

Quick check:

```bash
python --version
```

Expected: `Python 3.11.x`.

## 2) Run reproduction

### A) Single-teacher -> BERT (faster)

```bash
python scripts/run_single_teacher_to_bert.py
```

Typical runtime: usually shorter than full ensemble (hardware dependent).

### B) Full ensemble -> BERT (promoted best performer)

```bash
python scripts/run_single_ensemble_to_bert.py
```

Typical runtime: several hours (often ~3-10+ hours depending on GPU/CPU).

### Expected scores

Default runs use `k5_seed242`
**single-teacher -> BERT** expected test F1 is `0.5632`
**full ensemble -> BERT** expected test F1 is `0.6232`
Report tables may show different values because they report the mean over split seeds `42`, `142`, and `242`

### Final score file

After step `07`, check:

`reproduction/logs/step07_score_bert_manifest.json`

This manifest includes `report_file`, which points to the final metrics JSON for the run.
