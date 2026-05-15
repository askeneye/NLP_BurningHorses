from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / "pyproject.toml").exists():
            return path

    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "nlp_burninghorses" / "MIKKE" / "config.yaml"
CONFIG_SECTION = "bart_oada_xe"

CONFIG_TO_ENV = {
    "loss_type": "BART_LOSS_TYPE",
    "kshot_split": "BART_KSHOT_SPLIT",
    "pattern": "BART_PATTERN",
    "source_pattern_id": "BART_SOURCE_PATTERN_ID",
    "train_file": "BART_TRAIN_FILE",
    "output_dir": "BART_OUTPUT_DIR",
    "train_batch_size": "BART_TRAIN_BATCH_SIZE",
    "eval_batch_size": "BART_EVAL_BATCH_SIZE",
    "learning_rate": "BART_LEARNING_RATE",
    "max_steps": "BART_MAX_STEPS",
    "eval_steps": "BART_EVAL_STEPS",
    "use_early_stopping": "BART_USE_EARLY_STOPPING",
    "model_selection_strategy": "BART_MODEL_SELECTION_STRATEGY",
    "oada_tau_start": "BART_OADA_TAU_START",
    "oada_tau_end": "BART_OADA_TAU_END",
    "oada_tau_warmup_steps": "BART_OADA_TAU_WARMUP_STEPS",
    "oada_candidate_cap": "BART_OADA_CANDIDATE_CAP",
    "oada_candidate_seed": "BART_OADA_CANDIDATE_SEED",
}


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file) or {}


def apply_oada_xe_config() -> Path:
    config_path = Path(os.environ.get("BART_RUN_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    config = read_config(config_path)
    section = config.get(CONFIG_SECTION, {})

    if not isinstance(section, dict):
        raise ValueError(f"{config_path} section '{CONFIG_SECTION}' must be a mapping.")

    for config_key, env_name in CONFIG_TO_ENV.items():
        if env_name in os.environ:
            continue

        value = section.get(config_key)
        if value is not None:
            os.environ[env_name] = str(value)

    return config_path


def main() -> None:
    config_path = apply_oada_xe_config()

    from nlp_burninghorses.MIKKE.scripts.bart import main as run_bart

    print(f"Loaded BART OADA-XE preset from: {config_path}")
    run_bart()


if __name__ == "__main__":
    main()
