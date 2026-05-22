from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from scripts._lib.kshot_split import greedy_k_shot_support_split
from scripts._lib.repro_io import load_yaml, read_json_lines, write_json, write_jsonl


def _strip_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"tokens": row["tokens"], "ner_tags": row["ner_tags"]} for row in rows]


def _split_mini_val(
    validation_rows: list[dict[str, Any]], mini_val_size: int, mini_val_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(mini_val_seed)
    chosen = rng.sample(validation_rows, k=min(mini_val_size, len(validation_rows)))
    chosen_keys = {(tuple(row["tokens"]), tuple(row["ner_tags"])) for row in chosen}
    filtered = [
        row
        for row in validation_rows
        if (tuple(row["tokens"]), tuple(row["ner_tags"])) not in chosen_keys
    ]
    return chosen, filtered


def resolve_step01_paths(root: Path, config: dict[str, Any]) -> dict[str, Path | str]:
    default_profile = config.get("default_profile", {})
    k_shot = int(default_profile.get("k_shot", 5))
    split_seed = int(default_profile.get("split_seed", 242))
    split_name = str(default_profile.get("split_name", f"k{k_shot}_seed{split_seed}"))
    paths = config.get("paths", {})
    data_root = root / str(paths.get("data_root", "reproduction/data"))
    interim_root = root / str(paths.get("interim_root", "reproduction/data/interim"))
    step00_root = data_root / "conll2003"
    output_root = interim_root / split_name / "base"
    return {
        "k_shot": k_shot,
        "split_seed": split_seed,
        "split_name": split_name,
        "step00_root": step00_root,
        "output_root": output_root,
        "output_split_dir": output_root,
    }


def load_step01_inputs(step00_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    train_rows = read_json_lines(step00_root / "train.jsonl")
    validation_raw_rows = read_json_lines(step00_root / "validation_raw.jsonl")
    test_rows = read_json_lines(step00_root / "test.jsonl")
    step00_meta = json.loads((step00_root / "metadata.json").read_text(encoding="utf-8"))
    label_list = list(step00_meta["label_list"])
    return train_rows, validation_raw_rows, test_rows, label_list


def build_shared_eval_splits(
    validation_raw_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mini_val_rows, validation_rows = _split_mini_val(validation_raw_rows, mini_val_size=200, mini_val_seed=42)
    return _strip_id(validation_rows), _strip_id(mini_val_rows), _strip_id(test_rows)


def create_support_and_unlabeled_pool(
    train_rows: list[dict[str, Any]],
    k_shot: int,
    split_seed: int,
    label_list: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_core = _strip_id(train_rows)
    id_to_label = {index: label for index, label in enumerate(label_list)}
    return greedy_k_shot_support_split(
        train_core,
        k_shot,
        id_to_label=id_to_label,
        rng=random.Random(split_seed),
    )


def write_step01_split_files(
    output_root: Path,
    output_split_dir: Path,
    validation_core: list[dict[str, Any]],
    mini_val_core: list[dict[str, Any]],
    test_core: list[dict[str, Any]],
    support: list[dict[str, Any]],
    rest: list[dict[str, Any]],
) -> None:
    write_jsonl(output_root / "validation.jsonl", validation_core)
    write_jsonl(output_root / "test.jsonl", test_core)
    write_jsonl(output_root / "mini_val.jsonl", mini_val_core)
    write_jsonl(output_split_dir / "train.jsonl", support)
    write_jsonl(output_split_dir / "unlabeled_pool.jsonl", rest)


def write_step01_metadata(
    output_root: Path,
    output_split_dir: Path,
    *,
    k_shot: int,
    split_seed: int,
    label_list: list[str],
    validation_core: list[dict[str, Any]],
    mini_val_core: list[dict[str, Any]],
    test_core: list[dict[str, Any]],
    validation_raw_rows: list[dict[str, Any]],
    support: list[dict[str, Any]],
    rest: list[dict[str, Any]],
) -> None:
    id_to_label = {index: label for index, label in enumerate(label_list)}
    write_json(
        output_root / "metadata.json",
        {
            "dataset": "conll2003",
            "k": k_shot,
            "seed": split_seed,
            "run_index": 0,
            "num_train_support_sentences": len(support),
            "num_unlabeled_pool_sentences": len(rest),
            "num_validation_sentences": len(validation_core),
            "num_test_sentences": len(test_core),
            "num_mini_eval_sentences": len(mini_val_core),
            "mini_eval_source": "validation",
            "mini_eval_seed": 42,
            "mini_eval_requested_size": 200,
            "validation_excludes_mini_eval": True,
            "unlabeled_pool_contains_gold_labels": True,
            "label_list": label_list,
            "id_to_label": {str(index): label for index, label in id_to_label.items()},
            "text_column": "tokens",
            "label_column": "ner_tags",
            "num_raw_validation_sentences": len(validation_raw_rows),
        },
    )


def verify_step01_legacy_train(root: Path, split_name: str, support: list[dict[str, Any]]) -> bool | None:
    legacy_split_path = root / "data/interim/conll2003_kshot_bert" / split_name / "train.jsonl"
    if not legacy_split_path.exists():
        return None
    return support == read_json_lines(legacy_split_path)


def print_step01_summary(
    output_root: Path,
    split_name: str,
    support: list[dict[str, Any]],
    rest: list[dict[str, Any]],
    validation_core: list[dict[str, Any]],
    mini_val_core: list[dict[str, Any]],
    test_core: list[dict[str, Any]],
    exact_match: bool | None,
) -> None:
    print("Created reproduction k-shot split")
    print(f"output_root={output_root}")
    print(
        f"split={split_name} train={len(support)} unlabeled_pool={len(rest)} "
        f"validation={len(validation_core)} mini_val={len(mini_val_core)} test={len(test_core)}"
    )
    if exact_match is not None:
        print(f"legacy_train_exact_match={exact_match}")


def run_step01_kshot(root: Path, config_path: Path) -> None:
    config = load_yaml(config_path)
    resolved = resolve_step01_paths(root, config)
    train_rows, validation_raw_rows, test_rows, label_list = load_step01_inputs(resolved["step00_root"])
    validation_core, mini_val_core, test_core = build_shared_eval_splits(validation_raw_rows, test_rows)
    support, rest = create_support_and_unlabeled_pool(
        train_rows,
        int(resolved["k_shot"]),
        int(resolved["split_seed"]),
        label_list,
    )
    write_step01_split_files(
        resolved["output_root"],
        resolved["output_split_dir"],
        validation_core,
        mini_val_core,
        test_core,
        support,
        rest,
    )
    write_step01_metadata(
        resolved["output_root"],
        resolved["output_split_dir"],
        k_shot=int(resolved["k_shot"]),
        split_seed=int(resolved["split_seed"]),
        label_list=label_list,
        validation_core=validation_core,
        mini_val_core=mini_val_core,
        test_core=test_core,
        validation_raw_rows=validation_raw_rows,
        support=support,
        rest=rest,
    )
    exact_match = verify_step01_legacy_train(root, str(resolved["split_name"]), support)
    print_step01_summary(
        resolved["output_root"],
        str(resolved["split_name"]),
        support,
        rest,
        validation_core,
        mini_val_core,
        test_core,
        exact_match,
    )
